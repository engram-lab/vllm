#!/usr/bin/env python3
"""
Modal CI script for vLLM latency benchmarking.

Runs a latency benchmark and asserts that TPOT (Time Per Output Token) is under 10ms.
Exits with status code 0 if passed, 1 if failed.

Usage:
    modal run .github/scripts/modal_ci_latency_benchmark.py \
        --config-path .github/benchmark_configs/ci_latency.json
"""

import json
import os
import subprocess
import sys
import time

import modal

# --- Configuration ---
PORT = 8000
BRANCH = os.environ.get("BRANCH", "main")
GPU_TYPE = os.environ.get("GPU_TYPE", "L4")
GPU_COUNT = int(os.environ.get("GPU_COUNT", 1))
SECRETS = os.environ.get("SECRETS", "sabri-api-keys")

# vLLM optimization args
MAX_NUM_SEQS = int(os.environ.get("MAX_NUM_SEQS", "128"))
MAX_NUM_BATCHED_TOKENS = int(os.environ.get("MAX_NUM_BATCHED_TOKENS", "32768"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "8192"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.90"))

# Benchmark thresholds
TPOT_THRESHOLD_MS = float(os.environ.get("TPOT_THRESHOLD_MS", "10.0"))

MINUTES = 60  # seconds

secrets = modal.Secret.from_name(SECRETS)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git")
    .env({"VLLM_USE_PRECOMPILED": "1"})
    .run_commands(
        f"git clone https://$GITHUB_TOKEN@github.com/engram-lab/vllm.git -b {BRANCH} /tmp/vllm",
        secrets=[secrets],
        force_build=False,
    )
    .uv_pip_install(
        "vllm @ file:///tmp/vllm",
        "huggingface-hub[hf_transfer]==0.36.0",
        "flashinfer-python==0.5.2",
        "boto3",
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "XFORMERS_ENABLE_TRITON": "1",
            "VLLM_SERVER_DEV_MODE": "1",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            "VLLM_HTTP_TIMEOUT_KEEP_ALIVE": "300",
            "VLLM_PASSTHROUGH_ERRORS": "1",
        }
    )
)

hf_cache_vol = modal.Volume.from_name(
    "huggingface-cache", create_if_missing=True, version=2
)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True, version=2)
flashinfer_cache_vol = modal.Volume.from_name(
    "flashinfer-cache", create_if_missing=True, version=2
)

app = modal.App("vllm-ci-latency-benchmark")


@app.function(
    image=image,
    gpu=f"{GPU_TYPE}:{GPU_COUNT}",
    timeout=30 * MINUTES,
    secrets=[secrets],
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
        "/root/.cache/flashinfer": flashinfer_cache_vol,
    },
    enable_memory_snapshot=False,
)
def run_latency_benchmark(config_json: str):
    """
    Run latency benchmark and assert TPOT threshold.

    Args:
        config_json: JSON string containing the benchmark configuration

    Returns:
        Dictionary with benchmark results and pass/fail status
    """
    import signal
    import threading

    import requests

    # Parse config
    config = json.loads(config_json)
    model = config["model_name"]
    extra_body = config.get("extra_body")
    tensor_parallel_size = config.get("tensor_parallel_size", GPU_COUNT)

    print("=" * 80)
    print("vLLM CI LATENCY BENCHMARK")
    print("=" * 80)
    print(f"Config: {config.get('name', 'custom')}")
    print(f"Description: {config.get('description', 'N/A')}")
    print(f"Model: {model}")
    print(f"GPU allocation: {GPU_COUNT}x {GPU_TYPE}")
    print(f"Tensor Parallel Size: {tensor_parallel_size}")
    print(f"TPOT Threshold: {TPOT_THRESHOLD_MS}ms")
    print("=" * 80)
    print()

    # Start vLLM server
    cmd = [
        "vllm",
        "serve",
        model,
        "--served-model-name",
        model,
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--max-num-seqs",
        str(MAX_NUM_SEQS),
        "--max-num-batched-tokens",
        str(MAX_NUM_BATCHED_TOKENS),
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--gpu-memory-utilization",
        str(GPU_MEMORY_UTILIZATION),
        "--uvicorn-log-level",
        "warning",
        "--disable-log-requests",
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--enforce-eager",
    ]

    print(f"Starting vLLM server: {' '.join(cmd)}")
    print()

    def ping() -> bool:
        try:
            response = requests.get(f"http://localhost:{PORT}/health", timeout=1.0)
            return response.status_code == 200
        except requests.RequestException:
            return False

    # Start server
    server_start_time = time.time()
    server_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Stream server logs in a separate thread
    def stream_logs():
        print("=" * 80)
        print("vLLM SERVER LOGS")
        print("=" * 80)
        if server_process.stdout:
            for line in iter(server_process.stdout.readline, ""):
                if line:
                    print(f"[SERVER] {line.rstrip()}")
        print("=" * 80)
        print("vLLM SERVER STOPPED")
        print("=" * 80)

    log_thread = threading.Thread(target=stream_logs, daemon=True)
    log_thread.start()

    # Wait for server to be ready
    print("Waiting for vLLM server to start...")
    ready = False
    while time.time() - server_start_time < 600:
        if ping():
            server_ready_time = time.time()
            server_startup_duration = server_ready_time - server_start_time
            print(f"✓ vLLM server is ready! (startup time: {server_startup_duration:.2f}s)")
            print()
            ready = True
            break
        if server_process.poll() is not None:
            print("❌ Server process exited unexpectedly!")
            return {"status": "failed", "error": "Server failed to start"}
        time.sleep(2.0)

    if not ready:
        server_process.kill()
        return {"status": "failed", "error": "Server startup timeout"}

    base_url = f"http://localhost:{PORT}"

    try:
        # Build the benchmark command
        bench_cmd = [
            "vllm",
            "bench",
            "serve",
            "--backend",
            "openai",
            "--base-url",
            base_url,
            "--model",
            model,
            "--dataset-name",
            config["dataset_name"],
            "--num-prompts",
            str(config["num_prompts"]),
            "--seed",
            str(config.get("seed", 42)),
            "--num-warmups",
            str(config.get("num_warmups", 5)),
            "--save-result",
        ]

        # Add dataset-specific arguments
        if config["dataset_name"] == "random":
            bench_cmd.extend(
                [
                    "--random-input-len",
                    str(config.get("prompt_len", 512)),
                    "--random-output-len",
                    str(config.get("output_len", 128)),
                ]
            )

        # Add request rate
        if config.get("request_rate"):
            bench_cmd.extend(["--request-rate", str(config["request_rate"])])

        # Add optional arguments
        if config.get("max_concurrency"):
            bench_cmd.extend(["--max-concurrency", str(config["max_concurrency"])])

        # Add extra body parameters
        if extra_body:
            bench_cmd.extend(["--extra-body", json.dumps(extra_body)])

        result_filename = "ci_latency_result.json"
        bench_cmd.extend(["--result-filename", result_filename])

        print("=" * 80)
        print("Running benchmark:")
        print(" ".join(bench_cmd))
        print("=" * 80)
        print()

        # Run the benchmark
        result = subprocess.run(
            bench_cmd,
            capture_output=True,
            text=True,
        )

        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        if result.returncode != 0:
            return {
                "status": "failed",
                "error": f"Benchmark failed with return code {result.returncode}",
            }

        # Parse benchmark results
        benchmark_result = None
        try:
            for line in reversed(result.stdout.split("\n")):
                if line.strip().startswith("{"):
                    benchmark_result = json.loads(line.strip())
                    break
        except json.JSONDecodeError:
            pass

        if not benchmark_result:
            return {"status": "failed", "error": "Failed to parse benchmark results"}

        # Check TPOT threshold
        mean_tpot_ms = benchmark_result.get("mean_tpot_ms")
        median_tpot_ms = benchmark_result.get("median_tpot_ms")

        if mean_tpot_ms is None:
            return {"status": "failed", "error": "TPOT metric not found in results"}

        print("\n" + "=" * 80)
        print("LATENCY CHECK RESULTS")
        print("=" * 80)
        print(f"Mean TPOT: {mean_tpot_ms:.2f}ms")
        print(f"Median TPOT: {median_tpot_ms:.2f}ms")
        print(f"Threshold: {TPOT_THRESHOLD_MS}ms")
        print()

        passed = mean_tpot_ms < TPOT_THRESHOLD_MS

        if passed:
            print(f"✓ PASSED: Mean TPOT ({mean_tpot_ms:.2f}ms) < {TPOT_THRESHOLD_MS}ms")
        else:
            print(f"✗ FAILED: Mean TPOT ({mean_tpot_ms:.2f}ms) >= {TPOT_THRESHOLD_MS}ms")
        print("=" * 80)

        return {
            "status": "passed" if passed else "failed",
            "mean_tpot_ms": mean_tpot_ms,
            "median_tpot_ms": median_tpot_ms,
            "threshold_ms": TPOT_THRESHOLD_MS,
            "passed": passed,
            "full_results": benchmark_result,
        }

    finally:
        # Shutdown server
        print("\nShutting down vLLM server...")
        try:
            server_process.send_signal(signal.SIGTERM)
            server_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print("Server didn't stop gracefully, killing...")
            server_process.kill()
        print("✓ Server stopped")


@app.local_entrypoint()
def main(config_path: str):
    """
    Local entrypoint to run CI latency benchmark.

    Args:
        config_path: Path to benchmark config JSON file

    Examples:
        modal run .github/scripts/modal_ci_latency_benchmark.py \\
            --config-path .github/benchmark_configs/ci_latency.json
    """
    # Read the config file
    with open(config_path, "r") as f:
        config_json = f.read()

    print("=" * 80)
    print("vLLM CI LATENCY BENCHMARK")
    print("=" * 80)
    print(f"Config file: {config_path}")
    print(f"GPU allocation: {GPU_COUNT}x {GPU_TYPE}")
    print(f"TPOT Threshold: {TPOT_THRESHOLD_MS}ms")
    print("=" * 80)
    print()

    # Run the benchmark
    result = run_latency_benchmark.remote(config_json=config_json)

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(json.dumps(result, indent=2))
    print("=" * 80)

    # Exit with appropriate status code
    if result.get("status") == "passed":
        print("\n✓ CI BENCHMARK PASSED")
        sys.exit(0)
    else:
        print("\n✗ CI BENCHMARK FAILED")
        sys.exit(1)
