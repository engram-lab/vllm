#!/usr/bin/env python3
"""
Modal CI script for running vLLM pytest tests with GPU.

Runs pytest tests that require GPU access (like cartridge/LoRA tests)
using Modal for compute.

Exits with status code 0 if all tests passed, 1 if any failed.

Usage:
    modal run .github/scripts/modal_ci_pytest.py \
        --test-path tests/test_cartridge_addition_eval.py \
        --pytest-args "-v -x"
"""

import os
import subprocess
import sys

import modal

# --- Configuration ---
BRANCH = os.environ.get("BRANCH", "main")
GPU_TYPE = os.environ.get("GPU_TYPE", "A100")
GPU_COUNT = int(os.environ.get("GPU_COUNT", 1))
SECRETS = os.environ.get("SECRETS", "sabri-api-keys")

MINUTES = 60  # seconds

secrets = modal.Secret.from_name(SECRETS)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git", "rsync")
    .env({"VLLM_USE_PRECOMPILED": "1"})
    .run_commands(
        f"git clone https://$GITHUB_TOKEN@github.com/engram-lab/vllm.git -b {BRANCH} /tmp/vllm",
        secrets=[secrets],
        force_build=False,
    )
    .uv_pip_install(
        "vllm @ file:///tmp/vllm",
        "huggingface-hub[hf_transfer]==0.36.0",
        "flashinfer-python==0.5.3",
        "boto3",
        "pytest",
        "pytest-asyncio",
        "httpx",
        "openai",
        "tblib",
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

# Overlay local tests directory onto the image
# This allows running local test files that haven't been pushed yet
# copy=False mounts at runtime from local machine (not baked into image)
image = image.add_local_dir(
    local_path="./tests",
    remote_path="/mnt/local_tests",
    copy=False,
)

hf_cache_vol = modal.Volume.from_name(
    "huggingface-cache", create_if_missing=True, version=2
)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True, version=2)
flashinfer_cache_vol = modal.Volume.from_name(
    "flashinfer-cache", create_if_missing=True, version=2
)

app = modal.App("vllm-ci-pytest")


@app.function(
    image=image,
    gpu=f"{GPU_TYPE}:{GPU_COUNT}",
    timeout=45 * MINUTES,
    secrets=[secrets],
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
        "/root/.cache/flashinfer": flashinfer_cache_vol,
    },
    enable_memory_snapshot=False,
)
def run_pytest(test_path: str, pytest_args: str = "", env_vars: str = ""):
    """
    Run pytest on the specified test path.

    Args:
        test_path: Path to the test file or directory (relative to vllm repo root)
        pytest_args: Additional pytest arguments (space-separated)
        env_vars: Additional environment variables (KEY=VALUE,KEY2=VALUE2 format)

    Returns:
        Dictionary with test results and pass/fail status
    """
    print("=" * 80)
    print("vLLM CI PYTEST")
    print("=" * 80)
    print(f"Test path: {test_path}")
    print(f"Pytest args: {pytest_args}")
    print(f"GPU allocation: {GPU_COUNT}x {GPU_TYPE}")
    print("=" * 80)
    print()

    # Set up environment
    env = os.environ.copy()

    # Parse and apply additional env vars
    if env_vars:
        for pair in env_vars.split(","):
            if "=" in pair:
                key, value = pair.split("=", 1)
                env[key.strip()] = value.strip()
                print(f"Set env: {key.strip()}={value.strip()}")

    # Overlay local test files onto cloned repo using rsync
    # This allows running local test files that haven't been pushed yet
    print("\nOverlaying local test files onto /tmp/vllm/tests/...")
    subprocess.run(
        ["rsync", "-av", "/mnt/local_tests/", "/tmp/vllm/tests/"],
        check=True,
    )
    print()

    # Change to vllm directory
    os.chdir("/tmp/vllm")

    # Build pytest command
    cmd = ["python", "-m", "pytest", test_path]

    if pytest_args:
        cmd.extend(pytest_args.split())

    print(f"Running: {' '.join(cmd)}")
    print("=" * 80)
    print()

    # Run pytest
    result = subprocess.run(
        cmd,
        env=env,
        capture_output=False,  # Stream output directly
    )

    print()
    print("=" * 80)

    if result.returncode == 0:
        print("✓ PYTEST PASSED")
        return {"status": "passed", "returncode": result.returncode}
    else:
        print(f"✗ PYTEST FAILED (exit code: {result.returncode})")
        return {"status": "failed", "returncode": result.returncode}


@app.local_entrypoint()
def main(
    test_path: str,
    pytest_args: str = "-v",
    env_vars: str = "",
):
    """
    Local entrypoint to run CI pytest.

    Args:
        test_path: Path to test file or directory (relative to vllm repo root)
        pytest_args: Additional pytest arguments
        env_vars: Additional environment variables (KEY=VALUE,KEY2=VALUE2)

    Examples:
        modal run .github/scripts/modal_ci_pytest.py \\
            --test-path tests/test_cartridge_addition_eval.py \\
            --pytest-args "-v -x" \\
            --env-vars "VLLM_TEST_ENABLE_CARTRIDGE=1"
    """
    print("=" * 80)
    print("vLLM CI PYTEST")
    print("=" * 80)
    print(f"Test path: {test_path}")
    print(f"Pytest args: {pytest_args}")
    print(f"Environment vars: {env_vars}")
    print(f"GPU allocation: {GPU_COUNT}x {GPU_TYPE}")
    print("=" * 80)
    print()

    # Run the tests
    result = run_pytest.remote(
        test_path=test_path,
        pytest_args=pytest_args,
        env_vars=env_vars,
    )

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Status: {result.get('status')}")
    print(f"Return code: {result.get('returncode')}")
    print("=" * 80)

    # Exit with appropriate status code
    if result.get("status") == "passed":
        print("\n✓ CI PYTEST PASSED")
        sys.exit(0)
    else:
        print("\n✗ CI PYTEST FAILED")
        sys.exit(1)
