# Ollama Prometheus Exporter

A Prometheus exporter for Ollama that monitors request statistics, response times, token usage, and model performance. It runs as a FastAPI service and is Docker-ready.

[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://hub.docker.com/r/ollama-exporter)
[![Python Version](https://img.shields.io/badge/python-3.11+-green.svg)](https://www.python.org/downloads/release/python-3110/)

## Features

- **Request Tracking**: Monitors total requests per model
- **Response Time**: Measures total processing time with histogram distribution
- **Model Load Times**: Records model loading duration  
- **Evaluation Metrics**: 
  - Prompt evaluation time
  - Generation time 
  - Tokens processed (input)
  - Tokens generated (output)
  - Token generation rate (tokens per second)
- **Streaming Support**: Transparently proxies both standard and streaming responses
- **OpenAI Compatibility**: Full support for OpenAI-style `/v1/chat/completions` endpoint
- **Generic Proxy**: Transparently proxies all other Ollama API endpoints
- **Automatic Health Checks**: Validates Ollama connection on startup
- **High Performance**: Uses FastAPI with async/await for concurrent request handling
- **Configurable Logging**: Environment-based log level control

## Installation and Setup

### Docker (Recommended)

```bash
# Build and run with Docker
docker build -t ollama-exporter .
docker run -d --name ollama-exporter -p 8000:8000 \
  -e OLLAMA_HOST="http://host.docker.internal:11434" \
  ollama-exporter

# With custom Ollama host and mounted configuration
docker run -d --name ollama-exporter -p 8000:8000 \
  -v $(pwd)/log_config.yaml:/app/log_config.yaml \
  -e OLLAMA_HOST="http://192.168.1.100:11434" \
  -e LOG_LEVEL="DEBUG" \
  ollama-exporter

# Use a custom image name and tag
docker tag ollama-exporter registry.techsytes.com/ollama-exporter:latest
docker push registry.techsytes.com/ollama-exporter:latest
```

### Local Installation

```bash
# Navigate to the project directory
cd /path/to/ollama-exporter

# Install dependencies
pip install fastapi uvicorn prometheus_client httpx

# Run locally
OLLAMA_HOST=http://localhost:11434 python ollama_exporter.py
```

The exporter will listen on `0.0.0.0:8000`.

### Using a Virtual Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the exporter
OLLAMA_HOST=http://localhost:11434 python ollama_exporter.py
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_HOST` | Address of the Ollama server | `http://localhost:11434` |
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | `INFO` |

## API Endpoints

### Metrics Endpoint

```http
GET /metrics
```

Returns all Prometheus metrics in the standard Prometheus exposition format.

### Ollama Proxy Endpoints

The exporter transparently proxies all Ollama API endpoints:

- `POST /api/chat` - Chat-completions endpoint  
- `POST /api/generate` - Text generation endpoint
- All other Ollama API endpoints are proxied directly

### OpenAI-Compatible Endpoint

```http
POST /v1/chat/completions
```

Full OpenAI-compatible chat completions endpoint that works with OpenAI SDKs and tools.
Supports both streaming and non-streaming responses.

## Prometheus Metrics

### Counters

| Name | Description | Labels |
|------|-------------|--------|
| `ollama_requests_total` | Total number of requests processed | `model` |

### Histograms

| Name | Description | Labels |
|------|-------------|--------|
| `ollama_response_seconds` | Response time distribution | `model` |
| `ollama_load_duration_seconds` | Model loading time distribution | `model` |
| `ollama_prompt_eval_duration_seconds` | Prompt evaluation time distribution | `model` |
| `ollama_eval_duration_seconds` | Response generation time distribution | `model` |
| `ollama_tokens_per_second` | Token generation rate distribution | `model` |

## Docker Compose Example

```yaml
version: '3.8'
services:
  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

  exporter:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_HOST=http://ollama:11434
    depends_on:
      - ollama

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    depends_on:
      - exporter

volumes:
  ollama_data:
```

The exporter will listen on `0.0.0.0:8000`.
