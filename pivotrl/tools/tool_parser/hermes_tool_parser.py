import json
import logging
import os

import regex

from pivotrl.tools.base import ToolCall
from pivotrl.tools.tool_parser.base import ToolParser
from pivotrl.utils.rollout.rollout_trace import rollout_trace_op

pivotrl_logger = logging.getLogger(__file__)
pivotrl_logger.setLevel(os.getenv("PIVOTRL_LOGGING_LEVEL", "WARN"))


@ToolParser.register("hermes")
class HermesToolParser(ToolParser):
    """Adapted from https://github.com/vllm-project/vllm/blob/v0.9.1/vllm/entrypoints/openai/tool_parsers/hermes_tool_parser.py"""

    def __init__(self, tokenizer) -> None:
        super().__init__(tokenizer)

        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self.tool_call_regex = regex.compile(r"<tool_call>(.*?)</tool_call>", regex.DOTALL)

    @rollout_trace_op
    def extract_tool_calls(
        self,
        responses_ids: list[int],
        tools: list[dict] | None = None,
    ) -> tuple[str, list[ToolCall]]:
        response_str = self.tokenizer.decode(responses_ids)
        if self.tool_call_start_token not in response_str or self.tool_call_end_token not in response_str:
            return response_str, []

        matches = self.tool_call_regex.findall(response_str)
        tool_calls = []
        for match in matches:
            try:
                tool_call = json.loads(match)
                name, arguments = tool_call["name"], tool_call["arguments"]
                tool_calls.append(ToolCall(name=name, arguments=json.dumps(arguments, ensure_ascii=False)))
            except Exception as e:
                pivotrl_logger.debug(f"Failed to decode tool call from response: {response_str}, the error is: {e}")

        # remaing text exclude tool call tokens
        content = self.tool_call_regex.sub("", response_str)

        return content, tool_calls
