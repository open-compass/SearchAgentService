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

Parameter priority:

1. `service_env_params` in the request
2. Process environment variables or `./.env`
3. Code defaults

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| HOST | Service bind host | 0.0.0.0 |
| PORT | Service port | 8083 |
| WORKERS | Uvicorn worker count | 1 |
| TIMEOUT | HTTP connect/write/pool timeout | 60 |
| REQUEST_TIMEOUT | LLM/API request timeout | 2000 |
| MAX_RETRY | Maximum retry attempts | 10 |
| RETRY_INTERVAL | Retry interval (seconds) | 5 |
| MAX_ITERATIONS | Maximum agent iterations | 50 |
| MAX_TOOL_CALLS_PER_TURN | Maximum tool calls per turn | 5 |
| MAX_TOOL_RESPONSE_LENGTH | Maximum tool response length | 8192 |
| MAX_CONNECTIONS | Maximum connections per task | 256 |
| MAX_KEEPALIVE_CONNECTIONS | Maximum keep-alive connections per task | 64 |
| KEEPALIVE_EXPIRY | Connection expiry (seconds) | 10.0 |
| SERPER_API_KEY | Serper search API key | empty |
| JINA_API_KEY | Jina API key | empty |
| TOOLS | Enabled tools, e.g. `search,visit` | default registry |
| MODEL_NAME | Default model name for visit tool | empty |
| BASE_URL | Default LLM base URL for visit tool | empty |
| API_KEY | Default LLM API key for visit tool | empty |
| http_proxy | HTTP proxy | empty |
| https_proxy | HTTPS proxy | empty |

Parameter guidelines:
- **REQUEST_TIMEOUT**: 300-600s for fast models, 1000-2000s for thinking models
- **MAX_RETRY**: 3-5 for fast failure, 10-25 for production stability
- **MAX_ITERATIONS**: 20-30 prevents infinite loops, 40-50 allows complex tasks

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
          "TOOLS": "search,visit",
          "MAX_ITERATIONS": "50",
          "TIMEOUT": "600"
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
  "error": null
}
```

### GET /health

Health check.

```bash
curl http://localhost:8083/health
```

Response:

```json
{"status": "healthy", "service": "SearchAgentService"}
```
