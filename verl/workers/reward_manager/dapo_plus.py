from collections import defaultdict
import asyncio
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Any, Callable, Optional

import psutil
import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


async def single_compute_score(evaluation_func, completion, task, ground_truth, task_extra_info, executor, timeout=300.0):
    loop = asyncio.get_running_loop()
    try:
        # Ensure process_completion is called properly
        future = loop.run_in_executor(
            executor,
            partial(
                evaluation_func,
                data_source=task,
                solution_str=completion,
                ground_truth=ground_truth,
                extra_info=task_extra_info,
            ),
        )
        return await asyncio.wait_for(future, timeout=timeout)
    except asyncio.TimeoutError:
        print(f"[Timeout] Task timeout: {completion}")
        return None  # Default value for timed-out rows
    except Exception as e:
        print(f"[Error] Task failed: {e}, completion: {completion[:80]}")
        return None  # Default value for failed rows


async def parallel_compute_score_async(
    evaluation_func, completions, tasks, ground_truths=None, extra_info=None, num_processes=64
):
    if extra_info is None:
        extra_info = [None] * len(completions)
    if ground_truths is None:
        ground_truths = [None] * len(completions)
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # to prevent very occasional starvation caused by some anomalous programs ( like infinite loop ), the
        # exceptions in async programs will instantly halt the evaluation, and all summoned processes will be killed.
        try:
            # Create tasks for all rows
            tasks_async = [
                single_compute_score(evaluation_func, c, t, gt, ei, executor, timeout=300.0)
                for c, t, gt, ei in zip(completions, tasks, ground_truths, extra_info, strict=True)
            ]
            results = await asyncio.gather(*tasks_async, return_exceptions=False)
        except Exception as e:
            print(f"[Exception] async gather failed: {e}")
            raise
        finally:
            terminated_count = 0
            for pid, proc in executor._processes.items():
                try:
                    p = psutil.Process(pid)
                    p.terminate()
                    try:
                        p.wait(timeout=5)
                    except psutil.TimeoutExpired:
                        p.kill()
                    terminated_count += 1
                except Exception:
                    pass
            print(f"[Shutdown] {terminated_count} subprocess(es) terminated.")

    return results


def run_reward_scoring(evaluation_func, completions, tasks, ground_truths=None, extra_info=None, num_processes=64):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            parallel_compute_score_async(evaluation_func, completions, tasks, ground_truths, extra_info, num_processes)
        )
    finally:
        loop.close()


@register("dapo_plus")
class DAPOPlusRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, (
                f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
            )
            assert self.max_resp_len >= self.overlong_buffer_cfg.len, (
                "max_resp_len must be larger than overlong_buffer.len"
            )

    def __call__(self, data: DataProto, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}

        prompt_ids = data.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        response_ids = data.batch["responses"]

        responses_str_list = []
        valid_response_lengths = data.batch["attention_mask"][:, prompt_length:].sum(dim=-1)
        for i in range(len(response_ids)):
            valid_response_ids = response_ids[i, prompt_length : prompt_length + valid_response_lengths[i]]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]
            responses_str_list.append(response_str)

        ground_truths = [
            (item.non_tensor_batch.get("reward_model") or {}).get("ground_truth") for item in data
        ]
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        extra_info = data.non_tensor_batch.get("extra_info", [None] * len(data))

        results = run_reward_scoring(
            self.compute_score,
            completions=responses_str_list,
            tasks=data_sources,
            ground_truths=ground_truths,
            extra_info=extra_info,
            num_processes=psutil.cpu_count(logical=False) or 1,
        )

        for i in range(len(data)):
            result = results[i]
            data_item = data[i]

            score: float
            if isinstance(result, dict):
                score = result["score"]
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            elif isinstance(result, (int, float, bool)):
                score = float(result)
                reward_extra_info["acc"].append(score)
            else:
                score = 0.0
                reward_extra_info["acc"].append(score)

            reward = score
            valid_response_length = valid_response_lengths[i]

            if self.overlong_buffer_cfg and self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                reward += overlong_reward
                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)

            reward_tensor[i, valid_response_length - 1] = reward

            data_source = data_sources[i]
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                prompt_ids_item = data_item.batch["prompts"]
                prompt_length_item = prompt_ids_item.shape[-1]
                valid_prompt_length_item = data_item.batch["attention_mask"][:prompt_length_item].sum()
                valid_prompt_ids_item = prompt_ids_item[-valid_prompt_length_item:]
                prompt_str = self.tokenizer.decode(valid_prompt_ids_item, skip_special_tokens=True)

                print("[prompt]", prompt_str)
                print("[response]", responses_str_list[i])
                print("[ground_truth]", ground_truths[i])
                if isinstance(result, dict):
                    for key, value in result.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor