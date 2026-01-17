#!/usr/bin/env python3
"""
Modal CI script for vLLM benchmarking (latency and throughput).

Runs benchmarks and asserts thresholds based on config:
- Latency mode: asserts TPOT (Time Per Output Token) < threshold
- Throughput mode: asserts output throughput > threshold

Exits with status code 0 if all passed, 1 if any failed.

Usage:
    # Single config
    modal run .github/scripts/modal_ci_benchmark.py \
        --config-paths .github/benchmark_configs/base_latency.json

    # Multiple configs
    modal run .github/scripts/modal_ci_benchmark.py \
        --config-paths .github/benchmark_configs/base_latency.json \
        --config-paths .github/benchmark_configs/base_throughput.json
"""

import json
import os
import signal
import subprocess
import sys
import threading
import time

import modal

# --- Configuration ---
PORT = 8000
BRANCH = os.environ.get("BRANCH", "main")
GPU_TYPE = os.environ.get("GPU_TYPE", "H200")
GPU_COUNT = int(os.environ.get("GPU_COUNT", 1))
SECRETS = os.environ.get("SECRETS", "sabri-api-keys")

# vLLM optimization args
MAX_NUM_SEQS = int(os.environ.get("MAX_NUM_SEQS", "128"))
MAX_NUM_BATCHED_TOKENS = int(os.environ.get("MAX_NUM_BATCHED_TOKENS", "32768"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "8192"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.90"))

# Benchmark thresholds (can be overridden by config)
TPOT_THRESHOLD_MS = float(os.environ.get("TPOT_THRESHOLD_MS", "10.0"))
THROUGHPUT_THRESHOLD_TOK_S = float(os.environ.get("THROUGHPUT_THRESHOLD_TOK_S", "2500.0"))

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
        force_build=True,
    )
    .uv_pip_install(
        "vllm @ file:///tmp/vllm",
        "huggingface-hub[hf_transfer]==0.36.0",
        "flashinfer-python>=0.5.2",
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

app = modal.App("vllm-ci-benchmark")


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
def run_benchmarks_with_server(
    model: str,
    tensor_parallel_size: int,
    config_jsons: list[str],
    enforce_eager: bool,
):
    """
    Start vLLM server, run all benchmarks, then stop server.

    Args:
        model: Model name to serve
        tensor_parallel_size: Tensor parallel size
        config_jsons: List of JSON strings containing benchmark configurations

    Returns:
        List of dictionaries with benchmark results and pass/fail status
    """
    import requests

    print("=" * 80)
    print("STARTING vLLM SERVER")
    print("=" * 80)
    print(f"Model: {model}")
    print(f"GPU allocation: {GPU_COUNT}x {GPU_TYPE}")
    print(f"Tensor Parallel Size: {tensor_parallel_size}")
    print(f"Enforce eager: {enforce_eager}")
    print("=" * 80)
    print()

    # Build server command
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
    ]
    if enforce_eager:
        cmd.append("--enforce-eager")

    print(f"Server command: {' '.join(cmd)}")
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
            return [{"status": "failed", "error": "Server failed to start"}]
        time.sleep(2.0)

    if not ready:
        server_process.kill()
        print("❌ Server startup timeout!")
        return [{"status": "failed", "error": "Server startup timeout"}]

    base_url = f"http://localhost:{PORT}"

    # Run all benchmarks
    all_results = []

    try:
        for i, config_json in enumerate(config_jsons, 1):
            config = json.loads(config_json)
            extra_body = config.get("extra_body")
            assert_type = config.get("assert_type", "latency")

            print("\n" + "=" * 80)
            print(f"BENCHMARK {i}/{len(config_jsons)}: {assert_type.upper()}")
            print("=" * 80)
            print(f"Config: {config.get('name', 'custom')}")
            print(f"Description: {config.get('description', 'N/A')}")
            print(f"Model: {model}")
            print(f"Server URL: {base_url}")
            if assert_type == "throughput":
                threshold = config.get("throughput_threshold_tok_s", THROUGHPUT_THRESHOLD_TOK_S)
                print(f"Throughput Threshold: {threshold:.2f} tok/s")
            else:
                threshold = config.get("tpot_threshold_ms", TPOT_THRESHOLD_MS)
                print(f"TPOT Threshold: {threshold:.2f}ms")
            print("=" * 80)
            print()

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

                result_filename = f"ci_result_{i}.json"
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
                    all_results.append({
                        "status": "failed",
                        "error": f"Benchmark failed with return code {result.returncode}",
                    })
                    continue

                # Read benchmark results from saved JSON file
                benchmark_result = None
                try:
                    with open(result_filename, "r") as f:
                        benchmark_result = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    all_results.append({"status": "failed", "error": f"Failed to read benchmark results: {e}"})
                    continue

                print("\n" + "=" * 80)

                if assert_type == "throughput":
                    # Throughput mode: check output_throughput
                    print("THROUGHPUT CHECK RESULTS")
                    print("=" * 80)

                    output_throughput = benchmark_result.get("output_throughput")
                    request_throughput = benchmark_result.get("request_throughput")
                    threshold = config.get("throughput_threshold_tok_s", THROUGHPUT_THRESHOLD_TOK_S)

                    if output_throughput is None:
                        all_results.append({"status": "failed", "error": "Output throughput metric not found in results"})
                        continue

                    print(f"Output throughput: {output_throughput:.2f} tok/s")
                    print(f"Request throughput: {request_throughput:.2f} req/s")
                    print(f"Threshold: {threshold:.2f} tok/s")
                    print()

                    passed = output_throughput > threshold

                    if passed:
                        print(f"✓ PASSED: Output throughput ({output_throughput:.2f} tok/s) > {threshold:.2f} tok/s")
                    else:
                        print(f"✗ FAILED: Output throughput ({output_throughput:.2f} tok/s) <= {threshold:.2f} tok/s")
                    print("=" * 80)

                    all_results.append({
                        "status": "passed" if passed else "failed",
                        "output_throughput": output_throughput,
                        "request_throughput": request_throughput,
                        "threshold_tok_s": threshold,
                        "passed": passed,
                        "full_results": benchmark_result,
                    })
                else:
                    # Latency mode: check TPOT
                    print("LATENCY CHECK RESULTS")
                    print("=" * 80)

                    mean_tpot_ms = benchmark_result.get("mean_tpot_ms")
                    median_tpot_ms = benchmark_result.get("median_tpot_ms")
                    threshold = config.get("tpot_threshold_ms", TPOT_THRESHOLD_MS)

                    if mean_tpot_ms is None:
                        all_results.append({"status": "failed", "error": "TPOT metric not found in results"})
                        continue

                    print(f"Mean TPOT: {mean_tpot_ms:.2f}ms")
                    print(f"Median TPOT: {median_tpot_ms:.2f}ms")
                    print(f"Threshold: {threshold:.2f}ms")
                    print()

                    passed = mean_tpot_ms < threshold

                    if passed:
                        print(f"✓ PASSED: Mean TPOT ({mean_tpot_ms:.2f}ms) < {threshold:.2f}ms")
                    else:
                        print(f"✗ FAILED: Mean TPOT ({mean_tpot_ms:.2f}ms) >= {threshold:.2f}ms")
                    print("=" * 80)

                    all_results.append({
                        "status": "passed" if passed else "failed",
                        "mean_tpot_ms": mean_tpot_ms,
                        "median_tpot_ms": median_tpot_ms,
                        "threshold_ms": threshold,
                        "passed": passed,
                        "full_results": benchmark_result,
                    })

            except Exception as e:
                all_results.append({
                    "status": "failed",
                    "error": f"Benchmark execution error: {str(e)}",
                })

    finally:
        # Stop the server
        print("\nShutting down vLLM server...")
        try:
            server_process.send_signal(signal.SIGTERM)
            server_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print("Server didn't stop gracefully, killing...")
            server_process.kill()
        print("✓ Server stopped")

    return all_results


@app.local_entrypoint()
def main(*args):
    """
    Local entrypoint to run CI benchmark (latency or throughput).

    Args:
        args: Command line arguments (parsed using argparse)

    Examples:
        # Single config
        modal run .github/scripts/modal_ci_benchmark.py \\
            --config-paths .github/benchmark_configs/base_latency.json

        # Multiple configs (using shared vLLM server)
        modal run .github/scripts/modal_ci_benchmark.py \\
            --config-paths .github/benchmark_configs/base_latency.json \\
            --config-paths .github/benchmark_configs/base_throughput.json

    Note: All configs must use the same model, tensor_parallel_size, and enforce_eager
    to share a server.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run vLLM CI benchmarks")
    parser.add_argument(
        "--config-paths",
        action="append",
        required=True,
        help="Path to benchmark config JSON file (can be specified multiple times)"
    )
    parsed_args = parser.parse_args(args)
    config_paths = parsed_args.config_paths

    # Read and validate all configs
    configs = []
    config_jsons = []
    for config_path in config_paths:
        with open(config_path, "r") as f:
            config_json = f.read()
        config = json.loads(config_json)
        configs.append(config)
        config_jsons.append(config_json)

    # Validate all configs use the same model and tensor_parallel_size
    if len(configs) > 1:
        first_model = configs[0]["model_name"]
        first_tp_size = configs[0].get("tensor_parallel_size", GPU_COUNT)
        first_enforce_eager = configs[0].get("enforce_eager", False)

        for i, config in enumerate(configs[1:], 2):
            model = config["model_name"]
            tp_size = config.get("tensor_parallel_size", GPU_COUNT)

            if model != first_model:
                print(f"ERROR: Config {i} uses different model ({model}) than config 1 ({first_model})")
                print("All configs must use the same model to share a server.")
                sys.exit(1)

            if tp_size != first_tp_size:
                print(f"ERROR: Config {i} uses different tensor_parallel_size ({tp_size}) than config 1 ({first_tp_size})")
                print("All configs must use the same tensor_parallel_size to share a server.")
                sys.exit(1)
            enforce_eager = config.get("enforce_eager", False)
            if enforce_eager != first_enforce_eager:
                print(
                    f"ERROR: Config {i} uses different enforce_eager ({enforce_eager}) "
                    f"than config 1 ({first_enforce_eager})"
                )
                print("All configs must use the same enforce_eager to share a server.")
                sys.exit(1)

    # Get model and tensor_parallel_size from first config
    model = configs[0]["model_name"]
    tensor_parallel_size = configs[0].get("tensor_parallel_size", GPU_COUNT)
    enforce_eager = configs[0].get("enforce_eager", False)

    print("\n" + "=" * 80)
    print(f"STARTING SHARED vLLM SERVER FOR {len(configs)} BENCHMARK(S)")
    print("=" * 80)
    print(f"Model: {model}")
    print(f"Tensor Parallel Size: {tensor_parallel_size}")
    print(f"GPU allocation: {GPU_COUNT}x {GPU_TYPE}")
    print(f"Enforce eager: {enforce_eager}")
    print("=" * 80)
    print()

    # Run all benchmarks with a single server
    all_results = run_benchmarks_with_server.remote(
        model,
        tensor_parallel_size,
        config_jsons,
        enforce_eager,
    )

    # Print results for each config
    for i, (config_path, result) in enumerate(zip(config_paths, all_results), 1):
        print("\n" + "=" * 80)
        print(f"RESULTS FOR {config_path}")
        print("=" * 80)
        print(json.dumps(result, indent=2))
        print("=" * 80)

    # Print summary of all results
    print("\n\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    all_passed = True
    for config_path, result in zip(config_paths, all_results):
        status = result.get("status")

        if status == "passed":
            print(f"✓ PASSED: {config_path}")
        else:
            print(f"✗ FAILED: {config_path}")
            all_passed = False

    print("=" * 80)

    # Exit with appropriate status code
    if all_passed:
        print(f"\n✓ ALL {len(config_paths)} BENCHMARK(S) PASSED")
        sys.exit(0)
    else:
        print(f"\n✗ SOME BENCHMARKS FAILED")
        sys.exit(1)
