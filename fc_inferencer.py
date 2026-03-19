"""
Async Function Call Inferencer with direct tool calling.
"""

import asyncio
import json
import logging
import os
import random
import re
from typing import List, Literal, Optional, TypedDict, Union

import httpx
from openai import APITimeoutError, AsyncOpenAI
from pydantic import BaseModel, Field

from tools.registry import ToolRegistry, build_default_registry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AsyncFCInferencer")


def get_middle_mixed(text: str, max_num: int = 4000) -> str:
    """
    Truncate mixed Chinese-English text, keeping head and tail.

    Args:
        text: Original text
        max_num: Maximum number of units to keep

    Returns:
        Truncated text with head and tail preserved
    """
    if not text or max_num <= 0:
        return ""

    pattern = re.compile(r"[a-zA-Z0-9_'-]+|[^\s]")
    matches = list(pattern.finditer(text))
    total_units = len(matches)

    if total_units <= max_num:
        return text

    head_count = max_num // 2
    tail_count = max_num - head_count

    parts = []

    if head_count > 0:
        head_span_end = matches[head_count - 1].end()
        parts.append(text[:head_span_end])

    parts.append("...(truncated)...")

    if tail_count > 0:
        tail_idx = total_units - tail_count
        tail_span_start = matches[tail_idx].start()
        parts.append(text[tail_span_start:])

    return "".join(parts)


class FunctionCall(BaseModel):
    """Function call model for tool calls."""
    name: Optional[str] = None
    arguments: str = ""


class ToolCall(BaseModel):
    """Tool call model."""
    id: str
    type: Literal['function'] = 'function'
    function: FunctionCall


class ChatMessage(BaseModel):
    """Chat message model compatible with OpenAI format."""
    role: str
    content: Optional[str] = None
    reasoning_content: Optional[str] = Field(default=None)
    tool_calls: Optional[List[ToolCall]] = Field(default=None)
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class ModelConfig(TypedDict):
    """Model configuration."""
    model: str
    base_url: Union[str, List[str]]
    api_key: Optional[str]


class SampleParameters(TypedDict, total=False):
    """Sampling parameters for LLM inference."""
    temperature: float
    top_p: float
    top_k: int


class AsyncFCInferencer:
    """
    Async Function Call Inferencer with direct tool support.

    This inferencer supports:
    - Multiple LLM backends with load balancing
    - Direct tool calling via ToolRegistry (no MCP protocol)
    - Automatic retry mechanism
    - Tool response truncation
    """

    def __init__(
        self,
        model: ModelConfig,
        model_infer_params: Optional[dict] = None,
        registry: Optional[ToolRegistry] = None,
        max_iterations: Optional[int] = None,
        timeout: Optional[int] = None,
        max_retry: Optional[int] = None,
        sleep_interval: Optional[int] = None,
        max_tool_response_length: Optional[int] = None,
        max_tool_calls_per_turn: Optional[int] = None,
    ):
        base_urls = model['base_url'] if isinstance(model['base_url'], list) else [model['base_url']]

        # Create independent HTTP client for this instance
        max_connections = int(os.getenv("MAX_CONNECTIONS", "100"))
        max_keepalive = int(os.getenv("MAX_KEEPALIVE_CONNECTIONS", "20"))
        keepalive_expiry = float(os.getenv("KEEPALIVE_EXPIRY", "10.0"))
        http_timeout = float(os.getenv("TIMEOUT", "60.0"))
        request_timeout = float(os.getenv("REQUEST_TIMEOUT", "2000.0"))

        self.http_client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_keepalive,
                keepalive_expiry=keepalive_expiry
            ),
            timeout=httpx.Timeout(
                connect=http_timeout,
                read=request_timeout,
                write=http_timeout,
                pool=http_timeout
            )
        )

        self.clients = [
            AsyncOpenAI(
                api_key=model.get("api_key") or "dummy",
                base_url=url,
                http_client=self.http_client,
                max_retries=0  # Disable SDK auto-retry, use application-level retry only
            )
            for url in base_urls
        ]

        self.model_name = model["model"]
        self.model_infer_params = model_infer_params or {}
        self.max_iterations = max_iterations or int(os.getenv("MAX_ITERATIONS", "50"))
        self.timeout = timeout or int(os.getenv("REQUEST_TIMEOUT", "2000"))
        self.max_retry = max_retry or int(os.getenv("MAX_RETRY", "25"))
        self.sleep_interval = sleep_interval or int(os.getenv("RETRY_INTERVAL", "5"))
        self.max_tool_response_length = max_tool_response_length or int(os.getenv("MAX_TOOL_RESPONSE_LENGTH", "8192"))
        self.max_tool_calls_per_turn = max_tool_calls_per_turn or int(os.getenv("MAX_TOOL_CALLS_PER_TURN", "5"))

        self.registry = registry or build_default_registry()

    async def infer(self, messages: List[ChatMessage]) -> List[dict]:
        """Run inference with tool calling loop."""
        current_messages = [m.model_dump(exclude_none=True) for m in messages]
        tools_schema = self.registry.schemas

        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"Iteration {iteration}/{self.max_iterations}")

            response = await self._call_llm(current_messages, tools_schema)
            if response is None:
                break

            choice = response.choices[0]
            message_data = choice.message
            assistant_msg = message_data.model_dump(exclude_none=True)
            if "content" not in assistant_msg:
                assistant_msg["content"] = ""
            current_messages.append(assistant_msg)

            if not message_data.tool_calls:
                break

            if len(message_data.tool_calls) > self.max_tool_calls_per_turn:
                logger.warning(f"Too many tool calls: {len(message_data.tool_calls)}")
                break

            logger.info(f"Tools called: {[tc.function.name for tc in message_data.tool_calls]}")

            tool_results = await self._execute_tool_calls(message_data.tool_calls)
            if tool_results is None:
                break

            current_messages.extend(tool_results)

        return current_messages

    async def _call_llm(self, messages: List[dict], tools_schema: list):
        """Call LLM with retry logic."""
        for attempt in range(self.max_retry):
            try:
                client = random.choice(self.clients)

                call_params = {
                    "model": self.model_name,
                    "messages": messages,
                    "tools": tools_schema if tools_schema else None,
                    "stream": False,
                    "timeout": self.timeout,
                    **self.model_infer_params
                }

                response = await client.chat.completions.create(**call_params)
                return response
            except (APITimeoutError, TimeoutError) as e:
                logger.error(f"LLM Timeout (attempt {attempt + 1}): {e}")
                if attempt == self.max_retry - 1:
                    return None
                await asyncio.sleep(self.sleep_interval)
            except Exception as e:
                if self._is_retryable_error(e):
                    logger.error(f"LLM Error (attempt {attempt + 1}): {e}")
                    if attempt == self.max_retry - 1:
                        return None
                    await asyncio.sleep(self.sleep_interval)
                else:
                    logger.error(f"LLM Fatal Error: {e}")
                    return None
        return None

    def _is_retryable_error(self, e: Exception) -> bool:
        """Check if error is retryable."""
        retryable_patterns = [
            "TimeoutError", "litellm.BadRequestError", "litellm.APIError"
        ]
        error_str = str(e)
        return any(p in error_str for p in retryable_patterns)

    async def _execute_tool_calls(self, tool_calls) -> Optional[List[dict]]:
        """Execute tool calls and return results."""
        results = []

        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            args_str = tool_call.function.arguments
            call_id = tool_call.id

            result_content = await self._execute_single_tool(tool_name, args_str)
            if result_content is None:
                return None

            results.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": result_content,
                "name": tool_name
            })

        return results

    async def _execute_single_tool(self, tool_name: str, args_str: str) -> Optional[str]:
        """Execute a single tool call via registry (direct function call)."""
        if not self.registry.has_tool(tool_name):
            logger.error(f"Tool not found: {tool_name}")
            return None

        for attempt in range(self.max_retry):
            try:
                args = json.loads(args_str)
                if isinstance(args, str):
                    args = json.loads(args)

                logger.info(f"Executing {tool_name} with args: {str(args)[:200]}")

                result_content = await self.registry.execute(tool_name, args)

                if self.max_tool_response_length:
                    result_content = get_middle_mixed(
                        result_content, self.max_tool_response_length
                    )

                return result_content

            except Exception as e:
                logger.error(f"Tool execution error (attempt {attempt + 1}): {e}")
                await asyncio.sleep(self.sleep_interval)

        return None

    async def close(self):
        """Close HTTP client and release all connections."""
        if hasattr(self, 'http_client'):
            await self.http_client.aclose()

    def extract_final_answer(self, messages: List[dict]) -> str:
        """Extract final answer from message history."""
        if not messages:
            return ""

        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                return msg["content"]

        return ""
