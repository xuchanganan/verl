from recipe.langgraph_agent.chat_model import ChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.messages import BaseMessage
from typing import Any, Optional
from recipe.fully_async_policy.agent_loop.agent_loop  import FullyAsyncLLMServerManager
from recipe.fully_async_policy.agent_loop.agent_loop import AgentLoopOutput, FullyAsyncAgentLoopOutput

class FullAsyncChatModel(ChatModel):
    client: FullyAsyncLLMServerManager

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        request_id, prompt_ids, response_mask = await self._preprocess(messages, **kwargs)
        sampling_params = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
        }
        if "sampling_params" in kwargs:
            sampling_params.update(kwargs["sampling_params"])
        
        response_ids, log_probs, is_cancel = await self.client.generate_for_partial(
            request_id=request_id,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params
        )

        message = await self._postprocess(request_id, prompt_ids, response_mask, response_ids, **kwargs)
        if message.response_metadata:
            message.response_metadata["log_probs"] = log_probs
            message.response_metadata["is_cancel"] = is_cancel
        else:
            message.response_metadata = {
                "log_probs": log_probs,
                "is_cancel": is_cancel
            }
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])


def convert_to_agent_output(messages: list[BaseMessage], response_length: int) -> AgentLoopOutput:
    """Convert messages to AgentLoopOutput.

    Args:
        messages (List[BaseMessage]): List of messages, last message must be assistant
            with response_metadata containing `prompt_ids` and `response_mask`.
        response_length (int): Max length of response.

    Returns:
        AgentLoopOutput: agent loop output trajectory used for training.
    """
    # skip last tool calls
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].type != "tool":
            break
    last_message = messages[i]
    assert last_message.type == "ai", f"Last message must be assistant, but got {last_message.type}"
    assert "prompt_ids" in last_message.response_metadata, "Last message must have prompt_ids in response_metadata"
    assert "response_mask" in last_message.response_metadata, (
        "Last message must have response_mask in response_metadata"
    )

    num_turns = 0
    for i in range(len(messages)):
        if messages[i].type == "system":
            continue
        # parallel tool calls are in single turn
        if i == 0 or messages[i].type != messages[i - 1].type:
            num_turns += 1

    prompt_ids = last_message.response_metadata["prompt_ids"]
    response_mask = last_message.response_metadata["response_mask"]
    log_probs = last_message.response_metadata["log_probs"]

    response_ids = prompt_ids[-len(response_mask) :]
    prompt_ids = prompt_ids[: len(prompt_ids) - len(response_mask)]

    output = FullyAsyncAgentLoopOutput(
        prompt_ids=prompt_ids,
        response_ids=response_ids[:response_length],
        response_mask=response_mask[:response_length],
        num_turns=num_turns,
        metrics={}, # fixme, 这里应该有其他记录的.
        is_cancel=False,
        log_probs=log_probs[:response_length],
        param_version_start=param_version_start,
        param_version_end=param_version_end,
    )

    # output = AgentLoopOutput(
    #     prompt_ids=prompt_ids,
    #     response_ids=response_ids[:response_length],
    #     response_mask=response_mask[:response_length],
    #     num_turns=num_turns,
    #     metrics={}
    # )
    return output
