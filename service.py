"""
SearchAgentService - AgentCompass compatible tool calling service.

Usage:
    uvicorn service:app --host 0.0.0.0 --port 8083

Configuration (via AgentCompass):
    service_url: "http://localhost:8083/api/tasks"
    service_env_params:
        MAX_ITERATIONS: "50"
        TIMEOUT: "600"
        SERPER_API_KEY: "your_serper_key"
        JINA_API_KEY: "your_jina_key"
        MODEL_NAME: "model_name_for_visit_tool"
        BASE_URL: "llm_base_url_for_visit_tool"
        API_KEY: "llm_api_key_for_visit_tool"
"""

import logging
from typing import Optional, Dict, Any, List

from fastapi import FastAPI
from pydantic import BaseModel

from fc_inferencer import AsyncFCInferencer, ChatMessage
from tools.registry import build_default_registry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SearchAgentService")

app = FastAPI(title="SearchAgentService")


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


@app.post("/api/tasks", response_model=TaskResponse)
async def run_task(request: TaskRequest):
    """Run agent task (AgentCompass WAIT protocol)."""
    payload = request.model_dump()

    params = payload.get("params", {}) or {}
    llm_config = payload.get("llm_config", {}) or {}
    env_params = payload.get("service_env_params", {}) or {}

    question = params.get("question", "")
    if not question:
        return TaskResponse(
            final_answer="",
            status="failed",
            error="empty question"
        )

    if not llm_config.get("model_name") or not llm_config.get("url"):
        return TaskResponse(
            final_answer="",
            status="failed",
            error="llm_config must contain model_name and url"
        )

    model_config = {
        "model": llm_config.get("model_name", ""),
        "base_url": llm_config.get("url", ""),
        "api_key": llm_config.get("api_key", ""),
    }

    model_infer_params = llm_config.get("model_infer_params", {}) or {}

    max_iterations = int(env_params.get("MAX_ITERATIONS", "50"))
    timeout = int(env_params.get("TIMEOUT", "600"))
    max_retry = int(env_params.get("MAX_RETRY", "50"))
    sleep_interval = int(env_params.get("SLEEP_INTERVAL", "5"))

    task_id = params.get("task_id", "unknown")
    logger.info(f"Starting task {task_id}, model: {model_config['model']}")

    # Extract tool API keys from service_env_params and build registry
    tool_config = {
        "SERPER_API_KEY": env_params.get("SERPER_API_KEY", ""),
        "JINA_API_KEY": env_params.get("JINA_API_KEY", ""),
        "MODEL_NAME": env_params.get("MODEL_NAME", "") or llm_config.get("model_name", ""),
        "BASE_URL": env_params.get("BASE_URL", "") or llm_config.get("url", ""),
        "API_KEY": env_params.get("API_KEY", "") or llm_config.get("api_key", "sk-admin"),
        "TIMEOUT": str(timeout),
    }
    # Parse enabled tools list (comma-separated), default: search,visit
    tools_str = env_params.get("TOOLS", "")
    tools = [t.strip() for t in tools_str.split(",") if t.strip()] if tools_str else None

    registry = build_default_registry(config=tool_config, tools=tools)

    inferencer = AsyncFCInferencer(
        model=model_config,
        model_infer_params=model_infer_params,
        registry=registry,
        max_iterations=max_iterations,
        timeout=timeout,
        max_retry=max_retry,
        sleep_interval=sleep_interval,
    )

    try:
        messages = [ChatMessage(role="user", content=question)]
        result = await inferencer.infer(messages)

        final_answer = inferencer.extract_final_answer(result)

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
