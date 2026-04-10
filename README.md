# SearchAgentService

AgentCompass compatible tool calling service with search, web crawling, and content extraction. Single-process deployment, no MCP required.

## Directory Structure

```
SearchAgentService/
├── service.py           # Main service entry (FastAPI)
├── fc_inferencer.py     # Core inferencer (Function Calling)
├── tools/
│   ├── registry.py      # Tool registry
│   ├── search.py        # Search tool (Serper API)
│   ├── browse.py        # Web crawling tool (Jina API)
│   └── web_visitor.py   # Web visit + summary tool (Jina + LLM)
├── requirements.txt
└── README.md
```

## Run Service

SearchAgentService supports two startup modes:

- Local environment startup
- Docker startup

All commands below assume you are already in `./SearchAgentService`.

```bash
cd ./SearchAgentService
```

### Method 1: Local Environment

1. Prepare the environment file.

```bash
cp .env.example .env
# edit .env as needed
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Start the service.
```bash
./start.sh
```

You can also override the startup parameters:

```bash
HOST=0.0.0.0 PORT=8083 WORKERS=1 ./start.sh
```

4. Verify the service.

```bash
curl http://localhost:8083/health
```

### Method 2: Docker

1. Prepare the environment file.

```bash
cp .env.example .env
# edit .env as needed
```

2. Build the image.

```bash
docker build -t search-agent-service .
```

3. Run the container.

```bash
docker run --rm -p 8083:8083 \
  --env-file ./.env \
  --name search-agent-service \
  search-agent-service
```

4. Verify the service.

```bash
curl http://localhost:8083/health
```

## Configuration

### Runtime Configuration

`service.py` automatically loads `./.env` on startup. In Docker mode, prefer `--env-file ./.env`.

Resolution priority:

1. `service_env_params` in the request
2. Process environment variables or `./.env`
3. Code defaults

For request budget specifically, SearchAgentService resolves timeout in this order:

1. `llm_config.request_timeout` forwarded by AgentCompass from `benchmark_params.request_timeout`
2. process env `REQUEST_TIMEOUT` or `./.env` for local standalone runs
3. code default

### Environment Variables

SearchAgentService has two configuration paths for environment variables:

#### 1. Can be passed from AgentCompass

These are read from `service_env_params` first, then fall back to process env
or `.env`.

| Variable | Description | Default / Usage |
|----------|-------------|-----------------|
| SERPER_API_KEY | Serper API key for `search` | Required when `search` is enabled |
| JINA_API_KEY | Jina API key for `browse` / `visit` | Required when `browse` or `visit` is enabled |
| TOOLS | Enabled tools, chosen from `search,browse,visit` | Optional, default registry is `search,visit` |
| MAX_RETRY | Maximum retry attempts | Default `10` |
| RETRY_INTERVAL | Retry interval in seconds | Default `5` |
| MAX_ITERATIONS | Maximum agent iterations | Default `50` |

#### 2. Only effective via `.env` / process env

These are not read from `service_env_params` in the current implementation.

| Variable | Description | Default |
|----------|-------------|---------|
| HOST | Service bind host | 0.0.0.0 |
| PORT | Service port | 8083 |
| WORKERS | Uvicorn worker count | 1 |
| TIMEOUT_KEEP_ALIVE | Uvicorn idle keep-alive timeout in seconds | 5 |
| REQUEST_TIMEOUT | Per-task internal timeout override for standalone local runs | 2000 |
| HTTP_TIMEOUT | Low-level HTTP connect/write/pool timeout | 300 |
| MAX_TOOL_CALLS_PER_TURN | Maximum tool calls per turn | 5 |
| MAX_TOOL_RESPONSE_LENGTH | Maximum tool response length before truncation | 8192 |
| MAX_CONNECTIONS | Maximum connections per task instance | 256 |
| MAX_KEEPALIVE_CONNECTIONS | Maximum keep-alive connections per task instance | 64 |
| KEEPALIVE_EXPIRY | Keep-alive expiry in seconds | 10.0 |
| http_proxy | HTTP proxy | empty |
| https_proxy | HTTPS proxy | empty |

Notes:
- AgentCompass `benchmark_params.request_timeout` controls how long AgentCompass waits for the HTTP response from SearchAgentService.
- For AgentCompass requests, SearchAgentService uses `llm_config.request_timeout`, which AgentCompass forwards from `benchmark_params.request_timeout`, as its internal per-task timeout.
- `REQUEST_TIMEOUT` is only a local process-env override for standalone runs and is no longer read from `service_env_params`.

## API

### POST /api/tasks

Run an agent task (AgentCompass WAIT protocol).

```bash
curl -X POST "http://localhost:8001/api/tasks/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "benchmark": "gaia",
    "params": {
      "benchmark_params": {
        "judge_model": "gpt-4o",
        "category": "all",
        "max_concurrency": 4,
        "k": 1,
        "avgk": false,
        "service_url": "http://localhost:8083/api/tasks",
        "request_timeout": 3600,
        "service_env_params": {
          "SERPER_API_KEY": "your-serper-api-key",
          "JINA_API_KEY": "your-jina-api-key",
          "TOOLS": "search,visit"
        }
      },
      "model_infer_params": {
        "temperature": 0.8
      },
      "model_server_params": [
        {
          "type": "local",
          "url": "http://your-llm-server:8000/v1",
          "api_key": "your-api-key",
          "models": ["your-model-name"],
          "max_concurrent": 5
        }
      ]
    }
  }'
```

Response:

```json
{
  "final_answer": "The final answer",
  "trajectory": [{"role": "...", "content": "..."}],
  "status": "completed",
  "error": null,
  "retryable": null
}
```

For failed requests, the service may return explicit failure semantics such as:

```json
{
  "final_answer": "",
  "trajectory": [{"role": "...", "content": "..."}],
  "status": "failed",
  "error": "Reached max iterations (50) without a final answer",
  "retryable": false
}
```

`retryable` is the generic contract consumed by AgentCompass when deciding whether a failed sample should be persisted as `_error_*.json` and rerun on resume. `SearchAgentService` no longer returns a separate `stop_reason` field.

### GET /health

Health check.

```bash
curl http://localhost:8083/health
```

Response:

```json
{"status": "healthy", "service": "SearchAgentService"}
```
