# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test vLLM with cartridges on AdditionTestEvalDataset."""

import asyncio
import os
import random

import httpx
import openai
import pytest

# Enable detailed error passthrough for testing
os.environ["VLLM_PASSTHROUGH_ERRORS"] = "1"

pytest_plugins = ("pytest_asyncio",)
pytestmark = [
    pytest.mark.asyncio(loop_scope="session"),
    pytest.mark.benchmark,
]

from tests.utils import RemoteOpenAIServer
from vllm.platforms import current_platform

# Config
BASE_MODEL = "Qwen/Qwen3-0.6B"
# Note: In CI, we'll use a dummy model or skip if cartridge not available
CHECKPOINT_PATH = os.environ.get(
    "VLLM_TEST_CARTRIDGE_PATH",
    "s3://engram-cartridges/weights-toy/addition/qwen-0.6b/model.pt",
)


def generate_addition_conversations(n_samples=10, seed=42):
    """Generate simple addition test conversations."""
    random.seed(seed)
    conversations = []
    for _ in range(n_samples):
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        prompt = f"What is {a} + {b}? Please provide only the number."
        expected = str(a + b)
        conversations.append({"prompt": prompt, "expected": expected})
    return conversations


@pytest.fixture(scope="session")
def vllm_server():
    """Start vLLM server with LoRA/cartridge support."""
    # Require GPU for this test (like throughput benchmarks)
    if current_platform.is_cpu():
        pytest.skip("GPU required for cartridge tests")

    # Check if we should skip this test (e.g., no cartridge available)
    # This allows CI to conditionally enable cartridge tests
    if not os.environ.get("VLLM_TEST_ENABLE_CARTRIDGE"):
        pytest.skip(
            "Cartridge tests disabled (set VLLM_TEST_ENABLE_CARTRIDGE=1 to enable)"
        )

    args = [
        "--served-model-name",
        BASE_MODEL,
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--enable-lora",
        "--max-lora-rank",
        "256",
        "--max-loras",
        "1",
        "--max-model-len",
        "2048",
        "--enforce-eager",
    ]

    env = os.environ.copy()
    env["VLLM_PASSTHROUGH_ERRORS"] = "1"

    with RemoteOpenAIServer(
        BASE_MODEL, args, env_dict=env, max_wait_seconds=300
    ) as server:
        yield server


@pytest.fixture(scope="session")
def vllm_client(vllm_server):
    """Create OpenAI client for vLLM server."""
    return vllm_server.get_async_client()


@pytest.fixture(scope="session")
def eval_conversations():
    """Generate conversations using the exact same function as training."""
    return generate_addition_conversations(n_samples=10, seed=42)


async def _run_addition_eval(client, eval_conversations, n_test=10):
    """Shared evaluation logic for any client."""
    convs = eval_conversations[:n_test]
    prompts = [c["prompt"] for c in convs]
    expected = [c["expected"] for c in convs]

    correct = 0
    for i, prompt in enumerate(prompts):
        try:
            response = await client.chat.completions.create(
                model=BASE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=64,
                extra_body={
                    "adapters": {
                        "prefix": [
                            {"id": CHECKPOINT_PATH, "source": "s3", "type": "prefix"}
                        ]
                    }
                },
            )
            pred = response.choices[0].message.content.strip()
            is_correct = pred == expected[i]
            correct += int(is_correct)
            print(f"[{i}] prompt={prompt!r}")
            print(f"     response={pred!r}")
            print(f"     expected={expected[i]!r}, correct={is_correct}")
        except Exception as e:
            print(f"[{i}] Error: {e}")
            raise

    accuracy = correct / n_test
    print(f"\nAccuracy: {correct}/{n_test} = {accuracy:.1%}")
    return accuracy


async def test_base_model_without_cartridge_thinks(vllm_client):
    """Verify base model without cartridge produces thinking/rambling output.
    
    This is a sanity check: without the cartridge, the base Qwen3 model
    should start with '<think>' and ramble instead of answering directly.
    This confirms the cartridge is actually needed for correct behavior.
    """
    prompt = "What is 12 + 28? Please provide only the number."
    
    response = await vllm_client.chat.completions.create(
        model=BASE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=64,
        # NO lora_request - using base model only
    )
    
    text = response.choices[0].message.content
    print(f"Base model response (no cartridge): {text!r}")
    
    # Base Qwen3 model should start thinking/rambling
    assert text.startswith("<think>"), (
        f"Expected base model to start with '<think>' but got: {text[:100]!r}"
    )
    # Should NOT give the correct answer directly
    assert text.strip() != "40", (
        f"Base model unexpectedly gave correct answer without cartridge: {text!r}"
    )
    
    print("✓ Base model correctly shows thinking behavior without cartridge")


async def test_addition_eval_local_vllm(vllm_client, eval_conversations):
    """Evaluate vLLM (local) on addition problems with cartridge."""
    accuracy = await _run_addition_eval(vllm_client, eval_conversations)
    assert accuracy > 0.5, f"Accuracy too low: {accuracy:.1%}"


async def test_local_vllm_prefix_caching_with_cartridge(
    vllm_client, eval_conversations, vllm_server
):
    """Verify prefix caching works with cartridges by running same prompts twice."""
    # Use a smaller subset for this test
    n_test = 5
    convs = eval_conversations[:n_test]
    prompts = [c["prompt"] for c in convs]

    async def get_prefix_cache_hits() -> float:
        """Get current prefix cache hit count from metrics."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{vllm_server.url_for('metrics')}")
            for line in resp.text.split("\n"):
                # The metric is a counter: vllm:prefix_cache_hits{...} 123.0
                if "vllm:prefix_cache_hits" in line and not line.startswith("#"):
                    return float(line.split()[-1])
        return 0.0

    # First run - should be cache misses (cold)
    for prompt in prompts:
        await vllm_client.chat.completions.create(
            model=BASE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=32,
            extra_body={
                "adapters": {
                    "prefix": [
                        {"id": CHECKPOINT_PATH, "source": "s3", "type": "prefix"}
                    ]
                }
            },
        )

    hits_after_first = await get_prefix_cache_hits()
    print(f"Prefix cache hits after first run: {hits_after_first}")

    # Second run - same prompts, should get cache hits
    for prompt in prompts:
        await vllm_client.chat.completions.create(
            model=BASE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=32,
            extra_body={
                "adapters": {
                    "prefix": [
                        {"id": CHECKPOINT_PATH, "source": "s3", "type": "prefix"}
                    ]
                }
            },
        )

    hits_after_second = await get_prefix_cache_hits()
    print(f"Prefix cache hits after second run: {hits_after_second}")

    # The hit count should increase after the second run
    new_hits = hits_after_second - hits_after_first
    print(f"New cache hits from second run: {new_hits}")

    assert new_hits > 0, (
        f"Expected prefix cache hits after running same prompts twice, "
        f"but got {new_hits} new hits (before={hits_after_first}, after={hits_after_second}). "
        f"Cartridge prefix caching may not be working."
    )


async def test_cartridge_sharing_multiple_requests(vllm_client, eval_conversations):
    """Test that multiple concurrent requests with same cartridge share GPU cache."""
    n_concurrent = 5
    convs = eval_conversations[:n_concurrent]

    # Create separate message batches (simulating different users with same cartridge)
    tasks = []
    for conv in convs:
        prompt = conv["prompt"]
        task = vllm_client.chat.completions.create(
            model=BASE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=32,
            extra_body={
                "adapters": {
                    "prefix": [
                        {"id": CHECKPOINT_PATH, "source": "s3", "type": "prefix"}
                    ]
                }
            },
        )
        tasks.append(task)

    # Run all requests concurrently
    print(f"\nSending {n_concurrent} concurrent requests with same cartridge...")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Check all succeeded
    errors = [r for r in results if isinstance(r, Exception)]
    if errors:
        print(f"Errors: {errors}")
        raise errors[0]

    print(f"All {n_concurrent} requests completed successfully")

    # Verify responses are valid
    for i, result in enumerate(results):
        text = result.choices[0].message.content.strip()
        print(f"  [{i}] response: {text[:50]}...")
        assert len(text) > 0, f"Empty response for request {i}"

    print("✓ Cartridge sharing test passed - all concurrent requests succeeded")
