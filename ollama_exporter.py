import os
import asyncio
import httpx
import json
import logging
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

import uvicorn
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Configurable Ollama host (via env variable or defaults to localhost)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

logging.basicConfig()
logger = logging.getLogger(__name__)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

app = FastAPI()

OLLAMA_CHAT_REQUEST_COUNT = Counter("ollama_requests_total", "Total chat requests", ["model"])

OLLAMA_TOTAL_DURATION =       Histogram("ollama_response_seconds", "Total time spent for the response", ["model"])
OLLAMA_LOAD_DURATION =        Histogram("ollama_load_duration_seconds", "Time spent loading the model", ["model"])
OLLAMA_PROMPT_EVAL_DURATION = Histogram("ollama_prompt_eval_duration_seconds", "Time spent evaluating prompt", ["model"])
OLLAMA_EVAL_DURATION =        Histogram("ollama_eval_duration_seconds", "Time spent generating the response", ["model"])

OLLAMA_PROMPT_EVAL_COUNT = Counter("ollama_tokens_processed_total", "Number of tokens in the prompt", ["model"])
OLLAMA_EVAL_COUNT =        Counter("ollama_tokens_generated_total", "Number of tokens in the response", ["model"])

OLLAMA_TOKENS_PER_SECOND = Histogram(
    "ollama_tokens_per_second",
    "Tokens generated per second",
    ["model"],
    # Use buckets with suitable ranges for tokens/s measurements
    buckets=[5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
)


def extract_and_record_metrics(response_data, model, num_ctx=None):
    """Extract and record metrics from Ollama response data."""
    if not isinstance(response_data, dict):
        return

    # https://github.com/ollama/ollama/blob/main/docs/api.md#response
    total_duration = response_data.get("total_duration", 0) # total time spent in nanoseconds generating the response
    load_duration = response_data.get("load_duration", 0) # time spent in nanoseconds loading the model
    prompt_eval_duration = response_data.get("prompt_eval_duration", 0) # time spent in nanoseconds evaluating the prompt
    prompt_eval_count = response_data.get("prompt_eval_count", 0) # number of tokens in the prompt
    eval_duration = response_data.get("eval_duration", 0) # time spent in nanoseconds generating the response
    eval_count = response_data.get("eval_count", 0) # number of tokens in the response

    if total_duration > 0:
        total_duration_seconds = total_duration / 1_000_000_000
        OLLAMA_TOTAL_DURATION.labels(model=model).observe(total_duration_seconds)
        logger.debug(f"Model: {model}, Total Duration: {total_duration_seconds:.2f} seconds")
    if load_duration > 0:
        load_duration_seconds = load_duration / 1_000_000_000
        OLLAMA_LOAD_DURATION.labels(model=model).observe(load_duration_seconds)
        logger.debug(f"Model: {model}, Load Duration: {load_duration_seconds:.2f} seconds")
    if prompt_eval_duration > 0:
        prompt_eval_time_seconds = prompt_eval_duration / 1_000_000_000
        OLLAMA_PROMPT_EVAL_DURATION.labels(model=model).observe(prompt_eval_time_seconds)
        logger.debug(f"Model: {model}, Prompt Eval Duration: {prompt_eval_time_seconds:.2f} seconds")
    if prompt_eval_count > 0:
        OLLAMA_PROMPT_EVAL_COUNT.labels(model=model).inc(prompt_eval_count)
        logger.debug(f"Model: {model}, Prompt Eval Count: {prompt_eval_count}")
    if eval_duration > 0:
        eval_duration_seconds = eval_duration / 1_000_000_000
        OLLAMA_EVAL_DURATION.labels(model=model).observe(eval_duration_seconds)
        logger.debug(f"Model: {model}, Eval Duration: {eval_duration_seconds:.2f} seconds")
    if eval_count > 0:
        OLLAMA_EVAL_COUNT.labels(model=model).inc(eval_count)
        logger.debug(f"Model: {model}, Eval Count: {eval_count}")
    if eval_duration > 0 and eval_count > 0:
        tps = eval_count / eval_duration * 1_000_000_000
        OLLAMA_TOKENS_PER_SECOND.labels(model=model).observe(tps)
        logger.debug(f"Model: {model}, Tokens per Second: {tps:.2f}")
    
    # Log token usage summary with generation time
    if prompt_eval_count > 0 or eval_count > 0:
        ctx_info = f", Ctx: {num_ctx}" if num_ctx else ""
        if eval_duration > 0:
            total_seconds = eval_duration / 1_000_000_000
            minutes, seconds = divmod(int(total_seconds), 60)
            hours, minutes = divmod(minutes, 60)
            time_str = f"{hours}:{minutes:02d}:{seconds:02d}"
            # Calculate tokens per second
            tps_info = ""
            if eval_count > 0:
                tps = eval_count / eval_duration * 1_000_000_000
                tps_info = f", Speed: {tps:.2f} t/s"
            logger.info(f"Tokens - Model: {model}, In: {prompt_eval_count}, Out: {eval_count}{ctx_info}, Time: {time_str}{tps_info}")
        else:
            logger.info(f"Tokens - Model: {model}, In: {prompt_eval_count}, Out: {eval_count}{ctx_info}")


def extract_openai_metrics(response_data, model):
    """Extract and record metrics from OpenAI-compatible response data."""
    if not isinstance(response_data, dict):
        return

    usage = response_data.get("usage", {})
    if not usage:
        return

    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)

    if prompt_tokens > 0:
        OLLAMA_PROMPT_EVAL_COUNT.labels(model=model).inc(prompt_tokens)
    if completion_tokens > 0:
        OLLAMA_EVAL_COUNT.labels(model=model).inc(completion_tokens)

    if prompt_tokens > 0 or completion_tokens > 0:
        logger.info(f"Tokens - Model: {model}, In: {prompt_tokens}, Out: {completion_tokens}")


@app.get("/metrics")
def metrics():
    """Expose Prometheus metrics."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/api/chat")
@app.post("/api/generate")
async def chat_with_metrics(request: Request):
    """Handle chat and generate requests with streaming support and metrics extraction."""
    body = await request.json()
    model = body.get("model", "unknown")
    # Extract num_ctx from options if present
    options = body.get("options", {})
    num_ctx = options.get("num_ctx") if isinstance(options, dict) else None
    # logger.debug(f"Chat request body: {json.dumps(body, indent=4)}")
    prompt = body.get("prompt")
    messages = body.get("messages")
    if logger.isEnabledFor(logging.DEBUG):
        if prompt:
            logger.debug(f"Model: {model}, Received prompt: {prompt}")
        if messages:
            logger.debug(f"Model: {model}, Received messages: {messages}")
    is_streaming = body.get("stream", False)

    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)
    headers.pop("content-type", None)

    OLLAMA_CHAT_REQUEST_COUNT.labels(model=model).inc()

    if is_streaming:
        async def generate_stream():
            endpoint = request.url.path  # /api/chat or /api/generate
            async with httpx.AsyncClient(timeout=httpx.Timeout(900.0, read=900.0)) as client:
                async with client.stream("POST", f"{OLLAMA_HOST}{endpoint}", headers=headers, json=body, params=request.query_params) as response:

                    final_chunk_data = None

                    async for chunk in response.aiter_bytes():
                        # Forward the chunk immediately to the client
                        yield chunk

                        # Try to parse the chunk to look for metrics
                        if chunk:
                            try:
                                chunk_text = chunk.decode('utf-8')
                                lines = chunk_text.strip().split('\n')

                                for line in lines:
                                    if line.strip():
                                        try:
                                            chunk_json = json.loads(line)
                                            # Check if this is the final chunk (contains "done": true)
                                            if chunk_json.get("done", False):
                                                final_chunk_data = chunk_json
                                        except json.JSONDecodeError:
                                            continue

                            except UnicodeDecodeError:
                                pass

                    # Extract metrics from the final chunk if available
                    if final_chunk_data:
                        extract_and_record_metrics(final_chunk_data, model, num_ctx)

                    logger.debug(f"Model: {model}, Finished request")

        return StreamingResponse(generate_stream(), media_type="application/json")
    else:
        endpoint = request.url.path  # /api/chat or /api/generate
        async with httpx.AsyncClient(timeout=httpx.Timeout(900.0, read=900.0)) as client:
            response = await client.post(f"{OLLAMA_HOST}{endpoint}", headers=headers, json=body, params=request.query_params)

            if response.status_code == 200:
                try:
                    response_data = response.json()
                    extract_and_record_metrics(response_data, model, num_ctx)
                except (json.JSONDecodeError, TypeError):
                    pass

            return Response(content=response.content, status_code=response.status_code, headers=dict(response.headers))

@app.post("/v1/chat/completions")
async def openai_chat_with_metrics(request: Request):
    """Handle OpenAI-compatible chat/completions requests with metrics extraction."""
    body = await request.json()
    model = body.get("model", "unknown")
    messages = body.get("messages")
    if logger.isEnabledFor(logging.DEBUG) and messages:
        logger.debug(f"Model: {model}, Received messages: {messages}")
    is_streaming = body.get("stream", False)

    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)
    headers.pop("content-type", None)

    OLLAMA_CHAT_REQUEST_COUNT.labels(model=model).inc()

    if is_streaming:
        # Inject stream_options to request usage data in the final SSE chunk
        if "stream_options" not in body:
            body["stream_options"] = {"include_usage": True}
        elif isinstance(body["stream_options"], dict):
            body["stream_options"]["include_usage"] = True

        async def generate_stream():
            usage_data = None
            async with httpx.AsyncClient(timeout=httpx.Timeout(900.0, read=900.0)) as client:
                async with client.stream("POST", f"{OLLAMA_HOST}/v1/chat/completions", headers=headers, json=body, params=request.query_params) as response:
                    async for chunk in response.aiter_bytes():
                        yield chunk

                        # Parse SSE chunks to find usage data
                        if chunk:
                            try:
                                chunk_text = chunk.decode('utf-8')
                                for line in chunk_text.strip().split('\n'):
                                    line = line.strip()
                                    if line.startswith('data: ') and line != 'data: [DONE]':
                                        try:
                                            chunk_json = json.loads(line[6:])
                                            if "usage" in chunk_json and chunk_json["usage"]:
                                                usage_data = chunk_json
                                        except json.JSONDecodeError:
                                            continue
                            except UnicodeDecodeError:
                                pass

            if usage_data:
                extract_openai_metrics(usage_data, model)
            logger.debug(f"Model: {model}, Finished request")

        return StreamingResponse(generate_stream(), media_type="text/event-stream")
    else:
        async with httpx.AsyncClient(timeout=httpx.Timeout(900.0, read=900.0)) as client:
            response = await client.post(f"{OLLAMA_HOST}/v1/chat/completions", headers=headers, json=body, params=request.query_params)

            if response.status_code == 200:
                try:
                    response_data = response.json()
                    extract_openai_metrics(response_data, model)
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Failed to parse OpenAI response: {e}")

            return Response(content=response.content, status_code=response.status_code, headers=dict(response.headers))

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def simple_proxy(request: Request, path: str):
    """Simple pass-through proxy for all other endpoints."""
    logger.debug(f"Proxying {request.method} request to /{path}")
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)

    async with httpx.AsyncClient(timeout=httpx.Timeout(900.0, read=900.0)) as client:
        response = await client.request(method=request.method, url=f"{OLLAMA_HOST}/{path}", headers=headers, content=await request.body(), params=request.query_params)

    logger.debug(f"Proxy response: {response.status_code} for {request.method} /{path}")
    return Response(content=response.content, status_code=response.status_code, headers=dict(response.headers))

async def verify_ollama_connection():
    """Verify connection to Ollama server at startup."""
    logger.debug(f"Verifying connection to Ollama server at {OLLAMA_HOST}")

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            response = await client.get(f"{OLLAMA_HOST}/api/version")
            if response.status_code == 200:
                version_data = response.json()
                logger.info(f"Connected to Ollama")
            else:
                logger.error(f"Failed to connect to Ollama server. Status code: {response.status_code}")
    except Exception as e:
        logger.error(f"Failed to connect to Ollama server at {OLLAMA_HOST}: {e}")
        logger.error("Please ensure Ollama is running and accessible at the configured host")

async def main():
    await verify_ollama_connection()
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())
