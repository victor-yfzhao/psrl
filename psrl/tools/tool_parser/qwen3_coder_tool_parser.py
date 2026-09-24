import ast
import json
import logging
import os
from typing import Any

import regex

from psrl.tools.base import ToolCall
from psrl.tools.tool_parser.base import ToolParser
from psrl.utils.rollout.rollout_trace import rollout_trace_op

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@ToolParser.register("qwen3_coder")
class Qwen3XMLToolParser(ToolParser):
    """Parse the XML-style tool calls emitted by Qwen3-Coder and Qwen3.5."""

    def __init__(self, tokenizer) -> None:
        super().__init__(tokenizer)
        self.tool_call_start_token = "<tool_call>"
        self.tool_call_prefix = "<function="
        self.tool_call_regex = regex.compile(r"<tool_call>(.*?)</tool_call>|<tool_call>(.*?)$", regex.DOTALL)
        self.tool_call_function_regex = regex.compile(r"<function=(.*?)</function>|<function=(.*)$", regex.DOTALL)
        self.tool_call_parameter_regex = regex.compile(
            r"<parameter=(.*?)</parameter>|<parameter=(.*?)$",
            regex.DOTALL,
        )

    @rollout_trace_op
    def extract_tool_calls(
        self,
        responses_ids: list[int],
        tools: list[dict] | None = None,
    ) -> tuple[str, list[ToolCall]]:
        return self.extract_tool_calls_from_str(self.tokenizer.decode(responses_ids), tools=tools)

    def extract_tool_calls_from_str(
        self,
        response_str: str,
        tools: list[dict] | None = None,
    ) -> tuple[str, list[ToolCall]]:
        if self.tool_call_start_token not in response_str and self.tool_call_prefix not in response_str:
            return response_str, []

        try:
            function_calls = self._get_function_calls(response_str)
            if not function_calls:
                return response_str, []
            tool_calls = [self._parse_xml_function_call(call, tools) for call in function_calls]
            content_index = response_str.find(self.tool_call_start_token)
            if content_index < 0:
                content_index = response_str.find(self.tool_call_prefix)
            return response_str[:content_index], tool_calls
        except Exception:
            psrl_logger.exception("Failed to extract a Qwen3 XML tool call")
            return response_str, []

    def _get_function_calls(self, model_output: str) -> list[str]:
        tool_call_matches = self.tool_call_regex.findall(model_output)
        raw_tool_calls = [complete or partial for complete, partial in tool_call_matches]
        if not raw_tool_calls:
            raw_tool_calls = [model_output]

        function_calls = []
        for tool_call in raw_tool_calls:
            function_calls.extend(
                complete or partial for complete, partial in self.tool_call_function_regex.findall(tool_call)
            )
        return function_calls

    def _parse_xml_function_call(self, function_call: str, tools: list[dict] | None) -> ToolCall:
        end_index = function_call.index(">")
        function_name = function_call[:end_index]
        parameter_config = self._get_arguments_config(function_name, tools)
        parameters = function_call[end_index + 1 :]
        arguments = {}

        for complete, partial in self.tool_call_parameter_regex.findall(parameters):
            parameter = complete or partial
            separator = parameter.index(">")
            parameter_name = parameter[:separator]
            parameter_value = parameter[separator + 1 :].removeprefix("\n").removesuffix("\n")
            arguments[parameter_name] = self._convert_param_value(
                parameter_value,
                parameter_name,
                parameter_config,
                function_name,
            )

        return ToolCall(name=function_name, arguments=json.dumps(arguments, ensure_ascii=False))

    @staticmethod
    def _get_arguments_config(function_name: str, tools: list[dict] | None) -> dict[str, dict]:
        for tool in tools or []:
            if not isinstance(tool, dict) or tool.get("type") != "function":
                continue
            function = tool.get("function", {})
            if function.get("name") != function_name:
                continue
            properties = (function.get("parameters", {}) or {}).get("properties", {}) or {}
            return {str(name): config for name, config in properties.items() if isinstance(config, dict)}
        return {}

    @staticmethod
    def _convert_param_value(
        value: str,
        parameter_name: str,
        parameter_config: dict[str, dict],
        function_name: str,
    ) -> Any:
        if value.lower() == "null":
            return None
        if parameter_name not in parameter_config:
            return value

        parameter_type = str(parameter_config[parameter_name].get("type", "string")).strip().lower()
        if parameter_type in {"string", "str", "text", "varchar", "char", "enum"}:
            return value
        if parameter_type.startswith(("int", "uint", "long", "short", "unsigned")):
            try:
                return int(value)
            except ValueError:
                psrl_logger.warning("Invalid integer %r for %s.%s", value, function_name, parameter_name)
                return value
        if parameter_type.startswith(("num", "float")):
            try:
                float_value = float(value)
                return float_value if float_value != int(float_value) else int(float_value)
            except ValueError:
                psrl_logger.warning("Invalid number %r for %s.%s", value, function_name, parameter_name)
                return value
        if parameter_type in {"boolean", "bool", "binary"}:
            return value.lower() == "true"
        if parameter_type == "object" or parameter_type.startswith("dict"):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value
