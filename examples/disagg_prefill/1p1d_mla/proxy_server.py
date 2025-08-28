#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Proxy server for MLA disaggregated prefill/decode with DeepSeek models.
This proxy routes requests between prefiller and decoder services.
"""

import argparse
import json
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import httpx
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events.
    """
    # Startup: Initialize clients
    prefiller_base_url = (
        f"http://{global_args.prefiller_host}:{global_args.prefiller_port}"
    )
    decoder_base_url = f"http://{global_args.decoder_host}:{global_args.decoder_port}"
    
    print(f"Connecting to MLA Prefiller at {prefiller_base_url}")
    print(f"Connecting to MLA Decoder at {decoder_base_url}")
    
    app.state.prefill_client = httpx.AsyncClient(
        timeout=None, base_url=prefiller_base_url
    )
    app.state.decode_client = httpx.AsyncClient(timeout=None, base_url=decoder_base_url)
    
    yield
    
    # Shutdown: Close clients
    await app.state.prefill_client.aclose()
    await app.state.decode_client.aclose()


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="MLA Disagg Proxy",
    description="Proxy server for MLA disaggregated prefill/decode",
    lifespan=lifespan
)


class StatsCalculator:
    """Track and report TTFT (Time To First Token) statistics."""
    
    def __init__(self):
        self._stats = []
        self._last_log_time = time.time()
    
    def add(self, value):
        self._stats.append(value)
        if time.time() - self._last_log_time > 5:
            self._log_stats()
            self._last_log_time = time.time()
    
    def _log_stats(self):
        if not self._stats:
            return
        np_arr = np.array(self._stats)
        output_str = (
            f"\n===== MLA Prefill TTFT Statistics ====="
            f"\nNum requests: {len(self._stats)}"
            f"\nAverage (ms): {np.mean(np_arr):.2f}"
            f"\nMedian (ms): {np.median(np_arr):.2f}"
            f"\n99th Percentile (ms): {np.percentile(np_arr, 99):.2f}"
            f"\n======================================="
        )
        print(output_str)


stats_calculator = StatsCalculator()
request_counter = 0


async def send_request_to_service(
    client: httpx.AsyncClient, endpoint: str, req_data: dict
):
    """Send a request to a service using a persistent client."""
    # headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}"}
    response = await client.post(endpoint, json=req_data)  # , headers=headers
    response.raise_for_status()
    return response


async def stream_service_response(
    client: httpx.AsyncClient, endpoint: str, req_data: dict
):
    """Asynchronously stream the response from a service."""
    headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}"}
    async with client.stream(
        "POST", endpoint, json=req_data, headers=headers
    ) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            yield chunk


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model": "DeepSeek-V2-Lite (MLA)"}


@app.post("/v1/completions")
async def handle_completions(request: Request):
    """Handle completion requests with MLA support."""
    global request_counter, stats_calculator
    request_counter += 1
    req_id = str(request_counter)
    
    st = time.time()
    try:
        req_data = await request.json()
        
        stream = req_data.get("stream", False)
        media_type = "text/event-stream" if stream else "application/json"
        
        # Tokenize the prompt (round-robin between prefiller and decoder)
        if request_counter % 2 == 0:
            tokenization_client = app.state.prefill_client
        else:
            tokenization_client = app.state.decode_client
        
        tokenize_output = await send_request_to_service(
            tokenization_client, "/tokenize", {"prompt": req_data["prompt"]}
        )
        tokenize_output = tokenize_output.json()
        
        # Prepare prefill request with NIXL transfer parameters
        org_max_tokens = req_data["max_tokens"]
        req_data["prompt"] = tokenize_output["tokens"]
        req_data["max_tokens"] = 1  # Get first token from prefiller
        
        # Configure disaggregated transfer
        disagg_spec = {
            "req_id": req_id,
            "receiver_host": global_args.nixl_receiver_host,
            "receiver_init_port": global_args.nixl_receiver_port,
            "receiver_alloc_port": global_args.nixl_receiver_port + 1,
        }
        
        req_data["kv_transfer_params"] = {
            "ret_first_tok": True,
            "disagg_spec": disagg_spec,
        }
        req_data["stream"] = False
        stream_options = req_data.pop("stream_options", None)
        
        print(f"[Request {req_id}] Sending to MLA prefiller...")
        
        # Send to prefiller
        prefill_output = await send_request_to_service(
            app.state.prefill_client, "/v1/completions", req_data
        )
        prefill_output = prefill_output.json()
        
        et = time.time()
        ttft_ms = (et - st) * 1000
        stats_calculator.add(ttft_ms)
        print(f"[Request {req_id}] TTFT: {ttft_ms:.2f}ms")
        print("prefill_output:", prefill_output)
        
        # Prepare decode request
        req_data["max_tokens"] = org_max_tokens - 1
        req_data["prompt"].append(prefill_output["kv_transfer_params"]["first_tok"])
        req_data.pop("kv_transfer_params")
        req_data["stream"] = True
        if stream_options is not None:
            req_data["stream_options"] = stream_options
        
        # Stream response from decoder
        async def generate_stream():
            # Send first token from prefiller
            head_chunk = {
                "id": prefill_output["id"],
                "object": "text_completion",
                "created": prefill_output["created"],
                "model": prefill_output["model"],
                "choices": [
                    {
                        "index": 0,
                        "text": prefill_output["choices"][0]["text"],
                        "logprobs": None,
                        "finish_reason": None,
                        "stop_reason": None,
                    }
                ],
                "usage": None,
            }
            yield (
                "data: " + json.dumps(head_chunk, separators=(",", ":")) + "\n\n"
            ).encode()
            
            # Stream remaining tokens from decoder
            async for chunk in stream_service_response(
                app.state.decode_client, "/v1/completions", req_data
            ):
                yield chunk
        
        return StreamingResponse(generate_stream(), media_type=media_type)
    
    except Exception as e:
        import traceback
        print(f"Error in MLA proxy server: {e}")
        traceback.print_exc()
        raise


@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request):
    """Handle chat completion requests with MLA support."""
    global request_counter, stats_calculator
    request_counter += 1
    
    st = time.time()
    try:
        req_data = await request.json()
        
        stream = req_data.get("stream", False)
        media_type = "text/event-stream" if stream else "application/json"
        
        org_max_tokens = req_data.get("max_tokens", 100)
        req_data["max_tokens"] = 1
        
        org_max_completion_tokens = req_data.get("max_completion_tokens")
        if org_max_completion_tokens:
            req_data["max_completion_tokens"] = 1
        
        # Send to prefiller
        await send_request_to_service(
            app.state.prefill_client, "/v1/chat/completions", req_data
        )
        
        et = time.time()
        ttft_ms = (et - st) * 1000
        stats_calculator.add(ttft_ms)
        print(f"[Chat] TTFT: {ttft_ms:.2f}ms")
        
        # Prepare decode request
        req_data["max_tokens"] = org_max_tokens
        if org_max_completion_tokens:
            req_data["max_completion_tokens"] = org_max_completion_tokens
        
        # Stream response from decoder
        async def generate_stream():
            async for chunk in stream_service_response(
                app.state.decode_client, "/v1/chat/completions", req_data
            ):
                yield chunk
        
        return StreamingResponse(generate_stream(), media_type=media_type)
    
    except Exception as e:
        import traceback
        print(f"Error in MLA chat completions: {e}")
        traceback.print_exc()
        raise


def parse_args():
    parser = argparse.ArgumentParser(
        description="MLA Disaggregated Prefill/Decode Proxy Server"
    )
    
    parser.add_argument("--port", type=int, default=8000,
                        help="Proxy server port")
    parser.add_argument("--host", type=str, default="localhost",
                        help="Proxy server host")
    parser.add_argument("--prefiller-host", type=str, default="localhost",
                        help="Prefiller service host")
    parser.add_argument("--prefiller-port", type=int, default=8100,
                        help="Prefiller service port")
    parser.add_argument("--decoder-host", type=str, default="localhost",
                        help="Decoder service host")
    parser.add_argument("--decoder-port", type=int, default=8200,
                        help="Decoder service port")
    parser.add_argument("--nixl-receiver-host", type=str, default="localhost",
                        help="NIXL receiver host for KV transfer")
    parser.add_argument("--nixl-receiver-port", type=int, default=55555,
                        help="NIXL receiver port for KV transfer")
    
    return parser.parse_args()


if __name__ == "__main__":
    global global_args
    global_args = parse_args()
    
    print(f"""
    ======================================
    MLA Disaggregated Proxy Server
    ======================================
    Model: DeepSeek-V2-Lite (MLA)
    Proxy: {global_args.host}:{global_args.port}
    Prefiller: {global_args.prefiller_host}:{global_args.prefiller_port}
    Decoder: {global_args.decoder_host}:{global_args.decoder_port}
    NIXL Transfer: {global_args.nixl_receiver_host}:{global_args.nixl_receiver_port}
    ======================================
    """)
    
    import uvicorn
    uvicorn.run(app, host=global_args.host, port=global_args.port)