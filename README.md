# Inference Server
A FastAPI server exposing OpenAI-compatible endpoints for edge inference engines.

## Requirements
- [uv](https://docs.astral.sh/uv/) (manages Python 3.12 and dependencies automatically)

## Quickstart
```sh
uv sync          # create .venv and install dependencies
uv run python -m inference_server
```
The server listens on `http://127.0.0.1:8000` by default. Check it with:
```sh
curl http://127.0.0.1:8000/health
```

## Endpoints
OpenAI-compatible surface (works with the `openai` SDK by setting `base_url`):

| Endpoint | Description |
| --- | --- |
| `GET /v1/models` | Lists the models served by the active engine |
| `POST /v1/chat/completions` | Chat completion; set `"stream": true` for SSE |
| `GET /health` | Liveness check |

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "echo", "messages": [{"role": "user", "content": "Hi!"}]}'
```

Errors use the OpenAI envelope (`{"error": {"message", "type", "param", "code"}}`).

## Configuration
Settings are read from environment variables (prefix `INFERENCE_`) or a `.env` file:

| Variable | Default | Description |
| --- | --- | --- |
| `INFERENCE_HOST` | `127.0.0.1` | Bind address |
| `INFERENCE_PORT` | `8000` | Bind port |
| `INFERENCE_LOG_LEVEL` | `info` | Uvicorn log level |
| `INFERENCE_ENGINE` | `echo` | Inference engine backend (`echo` for development) |
| `INFERENCE_MODELS_DIR` | `<base dir>/models` | Directory containing model files |

The base directory is the current working directory during development. When
running as a frozen executable, it is the directory containing the executable,
so models and `.env` live next to the EXE rather than inside it.

## Development
```sh
uv run pytest        # tests
uv run ruff check .  # lint
uv run mypy src      # type check
```
