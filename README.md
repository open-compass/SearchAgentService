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

## Quick Start

```bash
pip install -r requirements.txt
python service.py --host 0.0.0.0 --port 8083
```

## Configuration

### Environment Variables

SearchAgentService supports centralized configuration via `.env` file or environment variables.

**Setup:**
```bash
# Copy example config
cp .env.example .env

# Edit values as needed
vim .env
```

**Available Parameters:**

| Variable | Description | Default |
|----------|-------------|---------|
| **Timeouts (seconds)** |
| TIMEOUT | HTTP connect/write/pool timeout | 60 |
| REQUEST_TIMEOUT | LLM/API request timeout | 2000 |
| **Retry Configuration** |
| MAX_RETRY | Maximum retry attempts | 10 |
| RETRY_INTERVAL | Retry interval (seconds) | 5 |
| **Execution Limits** |
| MAX_ITERATIONS | Maximum agent iterations | 50 |
| MAX_TOOL_CALLS_PER_TURN | Maximum tool calls per turn | 5 |
| MAX_TOOL_RESPONSE_LENGTH | Maximum tool response length | 8192 |
| **Connection Pool (per task instance)** |
| MAX_CONNECTIONS | Maximum connections per task | 256 |
| MAX_KEEPALIVE_CONNECTIONS | Maximum keep-alive connections per task | 64 |
| KEEPALIVE_EXPIRY | Connection expiry (seconds) | 10.0 |

**Parameter Guidelines:**
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
