"""
SearchAgentService - AgentCompass compatible tool calling service.

Usage:
    uvicorn service:app --host 0.0.0.0 --port 8083

Configuration (via AgentCompass):
    service_url: "http://localhost:8083/api/tasks"
    service_env_params:
        MAX_ITERATIONS: "50"
        REQUEST_TIMEOUT: "600"
        SERPER_API_KEY: "your_serper_key"
        JINA_API_KEY: "your_jina_key"
        MODEL_NAME: "optional_tool_model_name"
"""

import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import FastAPI
from dotenv import load_dotenv
from pydantic import BaseModel

from fc_inferencer import AsyncFCInferencer, ChatMessage
from tools.registry import build_default_registry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SearchAgentService")

load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=False)

app = FastAPI(title="SearchAgentService")

_DISALLOWED_TOOL_LLM_ENV_KEYS = {"BASE_URL", "API_KEY"}


class TaskRequest(BaseModel):
    """AgentCompass compatible request format."""
    params: Optional[Dict[str, Any]] = None
    benchmark: Optional[str] = None
    llm_config: Optional[Dict[str, Any]] = None
    modality: Optional[str] = None
    service_env_params: Optional[Dict[str, str]] = None


class TaskResponse(BaseModel):
    """AgentCompass compatible response format."""
    final_answer: str
    trajectory: Optional[List[Dict]] = None
    status: str = "completed"
    error: Optional[str] = None


def _get_runtime_param(
    request_env_params: Dict[str, str],
    key: str,
    default: str = "",
    aliases: Optional[List[str]] = None,
) -> str:
    """Resolve a runtime parameter from request env params first, then process env."""
    candidates = [key, *(aliases or [])]

    for candidate in candidates:
        if candidate in request_env_params and request_env_params[candidate] is not None:
            return str(request_env_params[candidate])

    for candidate in candidates:
        value = os.getenv(candidate)
        if value is not None:
            return value

    return default


def _validate_request_config(
    llm_config: Dict[str, Any],
    env_params: Dict[str, str],
) -> Optional[str]:
    missing = [
        field for field in ("model_name", "url", "api_key")
        if not llm_config.get(field)
    ]
    if missing:
        return f"llm_config must contain {', '.join(missing)}"

    forbidden_keys = sorted(
        key for key in _DISALLOWED_TOOL_LLM_ENV_KEYS
        if env_params.get(key)
    )
    if forbidden_keys:
        return (
            "service_env_params may not override tool LLM credentials: "
            + ", ".join(forbidden_keys)
        )

    return None


@app.post("/api/tasks", response_model=TaskResponse)
async def run_task(request: TaskRequest):
    """Run agent task (AgentCompass WAIT protocol)."""
    payload = request.model_dump()

    params = payload.get("params", {}) or {}
    benchmark = payload.get("benchmark") or "unknown"
    llm_config = payload.get("llm_config", {}) or {}
    env_params = payload.get("service_env_params", {}) or {}

    question = params.get("question", "")
    if not question:
        return TaskResponse(
            final_answer="",
            status="failed",
            error="empty question"
        )

    config_error = _validate_request_config(llm_config, env_params)
    if config_error:
        return TaskResponse(
            final_answer="",
            status="failed",
            error=config_error
        )

    model_config = {
        "model": llm_config.get("model_name", ""),
        "base_url": llm_config.get("url", ""),
        "api_key": llm_config.get("api_key", ""),
    }

    model_infer_params = llm_config.get("model_infer_params", {}) or {}

    max_iterations = int(_get_runtime_param(env_params, "MAX_ITERATIONS", "50"))
    request_timeout = int(_get_runtime_param(env_params, "REQUEST_TIMEOUT", "2000", aliases=["TIMEOUT"]))
    max_retry = int(_get_runtime_param(env_params, "MAX_RETRY", "10"))
    sleep_interval = int(_get_runtime_param(env_params, "SLEEP_INTERVAL", "5", aliases=["RETRY_INTERVAL"]))

    task_id = params.get("task_id", "unknown")
    logger.info(f"Starting task {task_id}, benchmark: {benchmark}, model: {model_config['model']}")

    # Extract tool API keys from service_env_params and build registry
    registry = None
    tool_config = {
        "SERPER_API_KEY": _get_runtime_param(env_params, "SERPER_API_KEY"),
        "JINA_API_KEY": _get_runtime_param(env_params, "JINA_API_KEY"),
        "MODEL_NAME": _get_runtime_param(env_params, "MODEL_NAME") or llm_config.get("model_name", ""),
        "BASE_URL": llm_config.get("url", ""),
        "API_KEY": llm_config.get("api_key", ""),
        "REQUEST_TIMEOUT": str(request_timeout),
        "MAX_RETRY": str(max_retry),
        "RETRY_INTERVAL": str(sleep_interval),
    }
    # Parse enabled tools list (comma-separated), default: search,visit
    tools_str = _get_runtime_param(env_params, "TOOLS")
    tools = [t.strip() for t in tools_str.split(",") if t.strip()] if tools_str else None

    try:
        registry = build_default_registry(config=tool_config, tools=tools)
    except Exception as e:
        logger.error(f"Task {task_id} failed during tool registry initialization: {e}")
        return TaskResponse(
            final_answer="",
            status="failed",
            error=(
                "tool registry initialization failed: "
                f"{e}. Check service_env_params such as SERPER_API_KEY, JINA_API_KEY, and TOOLS."
            ),
        )

    inferencer = AsyncFCInferencer(
        model=model_config,
        model_infer_params=model_infer_params,
        registry=registry,
        max_iterations=max_iterations,
        timeout=request_timeout,
        max_retry=max_retry,
        sleep_interval=sleep_interval,
    )

    try:
        messages = [ChatMessage(role="user", content=question)]
        result = await inferencer.infer(messages)

        final_answer = inferencer.extract_final_answer(result)
        if not str(final_answer or "").strip():
            error = inferencer.last_error or "empty final answer"
            logger.error(f"Task {task_id} failed: {error}")
            return TaskResponse(
                final_answer="",
                trajectory=result,
                status="failed",
                error=error,
            )

        logger.info(f"Task {task_id} completed")
        return TaskResponse(
            final_answer=final_answer,
            trajectory=result,
            status="completed"
        )
    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}")
        return TaskResponse(
            final_answer="",
            status="failed",
            error=str(e)
        )
    finally:
        await inferencer.close()
        if registry is not None:
            await registry.aclose()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "SearchAgentService"}


if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(description="SearchAgentService")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8083)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    logger.info(f"Starting on {args.host}:{args.port}")
    uvicorn.run("service:app", host=args.host, port=args.port, workers=args.workers)
