"""DashScope native SDK chat adapter.

Provides a BaseChatModel-compatible wrapper around dashscope.Generation.call,
including tool-calling support for LangChain/LangGraph workflows.
"""

from __future__ import annotations

import asyncio
import json
from http import HTTPStatus
from typing import Any, Dict, List, Optional, Sequence

import dashscope
import requests
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field, SecretStr


class DashScopeChatModel(BaseChatModel):
    """LangChain chat model backed by DashScope native SDK."""

    model: str = "qwen-plus"
    temperature: float = 0.0
    top_p: float = 0.8
    max_retries: int = 10
    request_timeout: int = 60
    api_key: Optional[SecretStr] = None
    base_http_api_url: Optional[str] = None

    @property
    def _llm_type(self) -> str:
        return "dashscope-native"

    def bind_tools(
        self,
        tools: Sequence[Dict[str, Any] | type | Any | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ):
        formatted_tools = [self._normalize_tool_schema(t) for t in tools]
        if tool_choice is not None:
            return self.bind(tools=formatted_tools, tool_choice=tool_choice, **kwargs)
        return self.bind(tools=formatted_tools, **kwargs)

    @staticmethod
    def _normalize_tool_schema(tool: Dict[str, Any] | type | Any | BaseTool) -> Dict[str, Any]:
        """Normalize tool-like objects to OpenAI tool schema.

        SecureQueryTool/CachedSchemaTool are lightweight wrappers (not BaseTool),
        so convert_to_openai_tool may reject them. We normalize by reading
        name/description/args_schema when available.
        """
        try:
            return convert_to_openai_tool(tool)
        except Exception:
            name = getattr(tool, "name", None)
            if not name:
                raise

            description = getattr(tool, "description", "") or ""
            args_schema = getattr(tool, "args_schema", None)

            parameters: Dict[str, Any]
            if args_schema is not None and hasattr(args_schema, "model_json_schema"):
                try:
                    parameters = args_schema.model_json_schema()
                except Exception:
                    parameters = {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    }
            else:
                parameters = {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                }

            return {
                "type": "function",
                "function": {
                    "name": str(name),
                    "description": description,
                    "parameters": parameters,
                },
            }

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload_messages = self._to_dashscope_messages(messages)

        request: Dict[str, Any] = {
            "model": self.model,
            "messages": payload_messages,
            "result_format": "message",
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

        if stop:
            request["stop"] = stop

        tools = kwargs.get("tools")
        if tools:
            request["tools"] = tools

        tool_choice = kwargs.get("tool_choice")
        if tool_choice is not None:
            request["tool_choice"] = tool_choice

        response = self._dashscope_call(request)
        self._raise_if_failed(response)

        choice = response["output"]["choices"][0]
        message = choice.get("message", {})
        content = message.get("content", "") or ""

        parsed_tool_calls = self._parse_tool_calls(message.get("tool_calls", []) or [])
        ai_msg = AIMessage(
            content=content,
            tool_calls=parsed_tool_calls,
            response_metadata={
                "finish_reason": choice.get("finish_reason"),
                "request_id": response.get("request_id"),
                "model": self.model,
            },
        )

        usage = response.get("usage") or {}
        llm_output = {
            "token_usage": usage,
            "model_name": self.model,
            "request_id": response.get("request_id"),
        }

        return ChatResult(
            generations=[ChatGeneration(message=ai_msg)],
            llm_output=llm_output,
        )

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return await asyncio.to_thread(self._generate, messages, stop, run_manager, **kwargs)

    def _dashscope_call(self, request: Dict[str, Any]) -> Dict[str, Any]:
        if self.base_http_api_url:
            dashscope.base_http_api_url = self.base_http_api_url

        api_key = self.api_key.get_secret_value() if self.api_key else None
        if api_key:
            request["api_key"] = api_key

        # DashScope SDK supports request_timeout kwarg (seconds).
        request["request_timeout"] = int(self.request_timeout)

        try:
            response = dashscope.Generation.call(**request)
        except requests.exceptions.ConnectTimeout as e:
            host = "dashscope.aliyuncs.com"
            raise ValueError(
                f"DashScope connection timeout to {host}. "
                f"request_timeout={self.request_timeout}s. "
                "请检查网络/代理/防火墙，或在 .env 中设置更短/更长的 DASHSCOPE_REQUEST_TIMEOUT。"
            ) from e
        except requests.exceptions.RequestException as e:
            raise ValueError(
                f"DashScope network error: {e}. "
                "请检查网络连通性与 DASHSCOPE_HTTP_BASE_URL 配置。"
            ) from e

        # DashScope response is dict-like; normalize to plain dict for robust access.
        return {
            "status_code": response.get("status_code"),
            "request_id": response.get("request_id"),
            "code": response.get("code"),
            "message": response.get("message"),
            "output": response.get("output") or {},
            "usage": response.get("usage") or {},
        }

    @staticmethod
    def _to_dashscope_messages(messages: List[BaseMessage]) -> List[Dict[str, str]]:
        mapped: List[Dict[str, str]] = []
        for m in messages:
            if isinstance(m, SystemMessage):
                role = "system"
            elif isinstance(m, HumanMessage):
                role = "user"
            elif isinstance(m, ToolMessage):
                # DashScope chat接口不直接使用 tool 角色消息；
                # 当前项目工具执行在模型外部完成，此处统一折叠为 assistant 附加文本上下文。
                role = "assistant"
            else:
                role = "assistant"

            content = getattr(m, "content", "")
            if isinstance(content, list):
                content = "\n".join(str(x) for x in content)
            mapped.append({"role": role, "content": str(content)})
        return mapped

    @staticmethod
    def _parse_tool_calls(raw_tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        parsed: List[Dict[str, Any]] = []
        for tc in raw_tool_calls:
            # DashScope native format is OpenAI-like:
            # {id, type:'function', function:{name, arguments:'{"k":"v"}'}}
            fn = tc.get("function") or {}
            name = fn.get("name", "")
            args = fn.get("arguments", "{}")

            if isinstance(args, str):
                try:
                    args_obj = json.loads(args)
                except Exception:
                    args_obj = {"raw": args}
            elif isinstance(args, dict):
                args_obj = args
            else:
                args_obj = {"raw": str(args)}

            parsed.append(
                {
                    "name": name,
                    "args": args_obj,
                    "id": tc.get("id", ""),
                    "type": "tool_call",
                }
            )
        return parsed

    @staticmethod
    def _raise_if_failed(response: Dict[str, Any]) -> None:
        status = response.get("status_code")
        ok = status == HTTPStatus.OK or status == 200 or str(status) == str(HTTPStatus.OK)
        if ok:
            return

        raise ValueError(
            "request_id: {request_id}\n status_code: {status_code}\n code: {code}\n message: {message}".format(
                request_id=response.get("request_id", ""),
                status_code=response.get("status_code", ""),
                code=response.get("code", ""),
                message=response.get("message", ""),
            )
        )
