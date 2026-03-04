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
python service.py --host 0.0.0.0 --port 8083 --workers 4
```

## Configuration

Tool configuration is passed via AgentCompass `service_env_params`:

```yaml
gaia:
  model: "qwen3-235b"
  judge_model: "gpt-4o"
  service_url: "http://localhost:8083/api/tasks"
  service_env_params:
    SERPER_API_KEY: "your-serper-api-key"
    JINA_API_KEY: "your-jina-api-key"
    TOOLS: "search,visit"
    MAX_ITERATIONS: "50"
    TIMEOUT: "600"
```

### service_env_params

| Parameter | Description | Required |
|-----------|-------------|----------|
| TOOLS | Comma-separated list of enabled tools. Options: search, browse, visit | No (default: search,visit) |
| SERPER_API_KEY | Serper search API key. Supports multiple keys (comma-separated, with optional `_ratelimit_N` suffix) | Yes |
| JINA_API_KEY | Jina Reader API key. Supports multiple keys (same format as above) | Yes |
| MODEL_NAME | LLM model name for web_visitor tool. Falls back to llm_config.model_name if not set | No |
| BASE_URL | LLM endpoint for web_visitor tool. Falls back to llm_config.url if not set | No |
| API_KEY | LLM API key for web_visitor tool. Falls back to llm_config.api_key if not set | No |
| MAX_ITERATIONS | Maximum iteration count | No (default: 50) |
| TIMEOUT | LLM request timeout in seconds | No (default: 600) |
| MAX_RETRY | Maximum retry count | No (default: 50) |
| SLEEP_INTERVAL | Retry interval in seconds | No (default: 5) |

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
