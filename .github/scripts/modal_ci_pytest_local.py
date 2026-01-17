#!/usr/bin/env python3
"""
Modal CI script for running vLLM pytest tests with GPU - LOCAL VERSION.

Clones vLLM from GitHub (cached), then overlays ONLY changed files at runtime.
Uses git diff to find changed files - much faster than uploading entire repo.

Usage:
    modal run /root/vllm/.github/scripts/modal_ci_pytest_local.py \
        --test-path tests/test_cartridge_addition_eval.py \
        --pytest-args "-v -x"
"""

import os
import subprocess
import sys

import modal

# --- Configuration ---
GPU_TYPE = os.environ.get("GPU_TYPE", "A100")
GPU_COUNT = int(os.environ.get("GPU_COUNT", 1))
SECRETS = os.environ.get("SECRETS", "sabri-api-keys")
BRANCH = os.environ.get("BRANCH", "main")

MINUTES = 60  # seconds

secrets = modal.Secret.from_name(SECRETS)

# Get the vllm repo root (two levels up from this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VLLM_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))


def get_changed_files() -> list[str]:
    """Get list of files changed compared to origin/main."""
    try:
        # Get changed files (both staged and unstaged) compared to main branch
        result = subprocess.run(
            ["git", "diff", "--name-only", "main"],
            cwd=VLLM_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        changed = result.stdout.strip().split("\n") if result.stdout.strip() else []
        
        # Also get untracked files that might be new tests
        result2 = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=VLLM_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        untracked = result2.stdout.strip().split("\n") if result2.stdout.strip() else []
        
        # Filter to only .py and .json files in vllm/ or tests/
        all_files = changed + untracked
        relevant = [
            f for f in all_files
            if f and (f.startswith("vllm/") or f.startswith("tests/"))
            and (f.endswith(".py") or f.endswith(".json"))
        ]
        return relevant
    except subprocess.CalledProcessError:
        # If git fails, fall back to empty (will use cloned code)
        return []


# Get changed files at module load time (when modal builds the image def)
CHANGED_FILES = get_changed_files()
print(f"Changed files to sync ({len(CHANGED_FILES)}): {CHANGED_FILES}")

# Build base image: clone from GitHub (cached) and build vLLM
base_image = (
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

# Mount ONLY changed files at runtime (not the whole repo)
# copy=False means mounted from local machine, not baked into image
if CHANGED_FILES:
    image = base_image.add_local_dir(
        local_path=VLLM_ROOT,
        remote_path="/mnt/local_vllm",
        copy=False,
        # Only include the specific changed files
        ignore=~modal.FilePatternMatcher(*CHANGED_FILES),
    )
else:
    # No local changes - just use the cloned repo
    image = base_image

hf_cache_vol = modal.Volume.from_name(
    "huggingface-cache", create_if_missing=True, version=2
)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True, version=2)
flashinfer_cache_vol = modal.Volume.from_name(
    "flashinfer-cache", create_if_missing=True, version=2
)

app = modal.App("vllm-ci-pytest-local")


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
def run_pytest(test_path: str, pytest_args: str = "", env_vars: str = "", changed_files: list[str] | None = None):
    """
    Run pytest on the specified test path.

    Args:
        test_path: Path to the test file or directory (relative to vllm repo root)
        pytest_args: Additional pytest arguments (space-separated)
        env_vars: Additional environment variables (KEY=VALUE,KEY2=VALUE2 format)
        changed_files: List of changed file paths to sync from local mount

    Returns:
        Dictionary with test results and pass/fail status
    """
    if changed_files is None:
        changed_files = []
    
    print("=" * 80)
    print("vLLM CI PYTEST (LOCAL)")
    print("=" * 80)
    print(f"Test path: {test_path}")
    print(f"Pytest args: {pytest_args}")
    print(f"GPU allocation: {GPU_COUNT}x {GPU_TYPE}")
    print(f"Changed files to sync: {len(changed_files)}")
    print("=" * 80)
    print()

    # Overlay local changes onto the cloned repo (only changed files)
    import shutil
    from pathlib import Path
    
    local_mount = Path("/mnt/local_vllm")
    target_repo = Path("/tmp/vllm")
    
    if local_mount.exists() and changed_files:
        # Copy each changed file individually
        for rel_path in changed_files:
            src = local_mount / rel_path
            dst = target_repo / rel_path
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                print(f"  Synced: {rel_path}")
        print(f"Synced {len(changed_files)} changed files")
    else:
        print("No local mount or no changed files - using cloned repo as-is")
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

    # Change to vllm directory (where tests are)
    os.chdir("/tmp/vllm")

    # Normalize test path
    test_file = test_path.replace("tests/", "").lstrip("/")
    
    # Build pytest command
    cmd = ["python", "-m", "pytest", f"tests/{test_file}"]

    if pytest_args:
        cmd.extend(pytest_args.split())

    print(f"Running: {' '.join(cmd)}")
    print(f"Working directory: {os.getcwd()}")
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
    Local entrypoint to run CI pytest using LOCAL vLLM code.

    Args:
        test_path: Path to test file or directory (relative to vllm repo root)
        pytest_args: Additional pytest arguments
        env_vars: Additional environment variables (KEY=VALUE,KEY2=VALUE2)

    Examples:
        modal run /root/vllm/.github/scripts/modal_ci_pytest_local.py \\
            --test-path tests/test_cartridge_addition_eval.py \\
            --pytest-args "-v -x" \\
            --env-vars "VLLM_TEST_ENABLE_CARTRIDGE=1"
    """
    print("=" * 80)
    print("vLLM CI PYTEST (LOCAL)")
    print("=" * 80)
    print(f"Test path: {test_path}")
    print(f"Pytest args: {pytest_args}")
    print(f"Environment vars: {env_vars}")
    print(f"GPU allocation: {GPU_COUNT}x {GPU_TYPE}")
    print(f"Local vLLM source: {VLLM_ROOT}")
    print(f"Changed files to sync: {len(CHANGED_FILES)}")
    for f in CHANGED_FILES:
        print(f"  - {f}")
    print("=" * 80)
    print()

    # Run the tests, passing the list of changed files
    result = run_pytest.remote(
        test_path=test_path,
        pytest_args=pytest_args,
        env_vars=env_vars,
        changed_files=CHANGED_FILES,
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
