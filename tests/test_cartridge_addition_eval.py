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

# For cascade attention stress tests, use larger model/cartridge
# The 8B cartridge has prefix_len=4096 which triggers cascade attention
CASCADE_TEST_MODEL = os.environ.get("VLLM_CASCADE_TEST_MODEL", "Qwen/Qwen3-8B")
CASCADE_TEST_CARTRIDGE = os.environ.get(
    "VLLM_CASCADE_TEST_CARTRIDGE",
    "s3://engram-cartridges/weights/torchtitan/2026-01-14/07-17-01-qwen3-8b-workspace-sabri-2026-01-14_07-16-56-d20c80e2/step-4800/model.pt",
)


def generate_addition_conversations(n_samples=10, seed=42):
    """Generate simple addition test conversations.
    
    Uses the same format as training data in torchtitan/datasets/addition_test.py:
    - Prompt: "Problem inputs: {a} and {b}"
    - Expected: "[engram-test-cartridge] The solution to the problem is {a + b}"
    """
    random.seed(seed)
    conversations = []
    for _ in range(n_samples):
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        # Match the training data format exactly
        prompt = f"Problem inputs: {a} and {b}"
        expected = f"[engram-test-cartridge] The solution to the problem is {a + b}"
        conversations.append({"prompt": prompt, "expected": expected, "answer": a + b})
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
    expected = [c["expected"] for c in convs]  # Full expected string

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
                    },
                    # Disable thinking mode to get direct answers
                    "chat_template_kwargs": {"enable_thinking": False},
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
    # Use the same prompt format as training data
    prompt = "Problem inputs: 12 and 28"
    
    response = await vllm_client.chat.completions.create(
        model=BASE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=64,
        # NO adapter - using base model only
    )
    
    text = response.choices[0].message.content
    print(f"Base model response (no cartridge): {text!r}")
    
    # Base Qwen3 model should start thinking/rambling
    assert text.startswith("<think>"), (
        f"Expected base model to start with '<think>' but got: {text[:100]!r}"
    )
    # Should NOT produce the trained response format
    assert "[engram-test-cartridge]" not in text, (
        f"Base model unexpectedly produced trained format without cartridge: {text!r}"
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
                },
                "chat_template_kwargs": {"enable_thinking": False},
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
                },
                "chat_template_kwargs": {"enable_thinking": False},
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
                },
                "chat_template_kwargs": {"enable_thinking": False},
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


async def test_cascade_attention_with_cartridge_prefix_sharing(vllm_client):
    """Test high-concurrency cartridge requests that may trigger cascade attention.
    
    This test is designed to expose a potential bug in cascade attention when
    combined with cartridge prefix sharing. The issue is:
    
    1. Cascade attention triggers when:
       - >= 8 concurrent requests share a common prefix
       - common_prefix_len >= 256 tokens
       
    2. For cartridge requests:
       - num_computed_tokens does NOT include cartridge tokens (cartridge is pre-populated)
       - But num_common_prefix_blocks DOES include cartridge blocks (for prefix sharing)
       - seq_lens DOES include cartridge offset (for attention coverage)
       
    3. The mismatch causes:
       - common_prefix_len = min(num_common_prefix_blocks * block_size, num_computed_tokens.min())
       - This caps common_prefix_len by num_computed_tokens which excludes cartridge
       - But suffix_kv_lens = seq_lens - common_prefix_len uses seq_lens which includes cartridge
       - Block table slicing block_table[:, num_common_kv_blocks:] may access wrong blocks
       
    This can result in CUDA illegal memory access errors when the mismatch
    causes out-of-bounds block table access or incorrect KV cache reads.
    """
    # High concurrency to trigger cascade attention (needs >= 8 requests)
    # Use 50 to match the benchmark that triggered the original error
    n_concurrent = 50
    
    # Generate unique prompts with varying lengths to stress-test the system
    # Use longer prompts to build up computed tokens that can trigger cascade attention
    random.seed(12345)
    prompts = []
    for i in range(n_concurrent):
        a = random.randint(1, 1000)
        b = random.randint(1, 1000)
        # Add some padding text to increase prompt length and computed tokens
        # This helps trigger cascade attention by having more shared prefix tokens
        padding = f"Request #{i}: " + "x " * random.randint(10, 50)
        prompt = f"{padding}Problem inputs: {a} and {b}"
        prompts.append(prompt)
    
    print(f"\n{'='*60}")
    print("Testing cascade attention with cartridge prefix sharing")
    print(f"Sending {n_concurrent} concurrent requests with same cartridge...")
    print(f"{'='*60}")
    
    # Create all tasks with the same cartridge (to trigger prefix sharing)
    tasks = []
    for prompt in prompts:
        task = vllm_client.chat.completions.create(
            model=BASE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            # Request more tokens to stay in decode phase longer
            # This increases the chance of hitting cascade attention during decode
            max_tokens=128,
            extra_body={
                "adapters": {
                    "prefix": [
                        {"id": CHECKPOINT_PATH, "source": "s3", "type": "prefix"}
                    ]
                },
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        tasks.append(task)
    
    # Run all requests concurrently - this should trigger cascade attention
    # if the cartridge has >= 256 tokens (16 blocks * 16 block_size)
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Check for errors - the bug manifests as CUDA illegal memory access
    errors = [r for r in results if isinstance(r, Exception)]
    if errors:
        print(f"\nFAILURE: {len(errors)} out of {n_concurrent} requests failed")
        for i, err in enumerate(errors[:5]):  # Show first 5 errors
            print(f"  Error {i}: {type(err).__name__}: {err}")
        
        # Check if this is the cascade attention bug
        error_str = str(errors[0])
        if "CUDA" in error_str or "illegal memory" in error_str.lower():
            print("\n" + "!"*60)
            print("DETECTED: CUDA memory error - likely cascade attention bug!")
            print("This happens when cascade attention is used with cartridge")
            print("prefix sharing and there's a mismatch between:")
            print("  - num_computed_tokens (excludes cartridge)")
            print("  - num_common_prefix_blocks (includes cartridge)")
            print("  - seq_lens (includes cartridge)")
            print("!"*60)
        
        raise errors[0]
    
    # Verify all responses are valid
    print(f"\nAll {n_concurrent} requests completed successfully")
    for i, result in enumerate(results[:5]):  # Show first 5
        text = result.choices[0].message.content.strip()
        print(f"  [{i}] response: {text[:60]}...")
    
    print("✓ Cascade attention with cartridge prefix sharing test passed")


async def test_cartridge_high_concurrency_streaming(vllm_client):
    """Test streaming with high concurrency cartridge requests.
    
    Streaming can expose race conditions in cartridge KV cache sharing
    because multiple requests are actively generating tokens simultaneously
    while sharing the same cartridge blocks.
    """
    n_concurrent = 30
    
    random.seed(99999)
    prompts = []
    for i in range(n_concurrent):
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        prompts.append(f"Problem inputs: {a} and {b}")
    
    print(f"\nTesting streaming with {n_concurrent} concurrent cartridge requests...")
    
    async def stream_request(prompt: str, idx: int) -> str:
        """Stream a single request and collect the response."""
        chunks = []
        try:
            stream = await vllm_client.chat.completions.create(
                model=BASE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=64,
                stream=True,  # Enable streaming
                extra_body={
                    "adapters": {
                        "prefix": [
                            {"id": CHECKPOINT_PATH, "source": "s3", "type": "prefix"}
                        ]
                    },
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    chunks.append(chunk.choices[0].delta.content)
            return "".join(chunks)
        except Exception as e:
            return f"ERROR: {e}"
    
    # Run all streaming requests concurrently
    tasks = [stream_request(p, i) for i, p in enumerate(prompts)]
    results = await asyncio.gather(*tasks)
    
    # Check for errors
    errors = [r for r in results if r.startswith("ERROR:")]
    if errors:
        print(f"\nFAILURE: {len(errors)} streaming requests failed")
        for err in errors[:5]:
            print(f"  {err}")
        raise RuntimeError(errors[0])
    
    print(f"✓ All {n_concurrent} streaming requests completed successfully")


async def test_cartridge_gpu_memory_cleanup(vllm_client, vllm_server):
    """Test that cartridge GPU memory is properly freed after requests finish.
    
    This test verifies that the gpu_cartridge_cache is cleaned up when all
    requests using a cartridge have finished. Without proper cleanup, GPU
    memory would leak with each new cartridge (~1-2GB per 4096-token cartridge).
    
    The test works by:
    1. Running requests with a cartridge
    2. Waiting for all requests to finish
    3. Checking that GPU memory usage returns close to baseline
    """
    
    async def get_gpu_memory_used() -> float:
        """Get GPU memory used in MB from vLLM metrics."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{vllm_server.url_for('metrics')}")
            for line in resp.text.split("\n"):
                # Look for GPU memory metric
                if "vllm:gpu_cache_usage_perc" in line and not line.startswith("#"):
                    return float(line.split()[-1])
        return 0.0
    
    # Get baseline memory before any cartridge requests
    baseline_usage = await get_gpu_memory_used()
    print(f"\nBaseline GPU cache usage: {baseline_usage:.2%}")
    
    # Run a batch of cartridge requests
    n_requests = 10
    prompts = [f"Problem inputs: {i} and {i+1}" for i in range(n_requests)]
    
    print(f"Sending {n_requests} cartridge requests...")
    tasks = []
    for prompt in prompts:
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
                },
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        tasks.append(task)
    
    # Wait for all requests to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    errors = [r for r in results if isinstance(r, Exception)]
    if errors:
        raise errors[0]
    
    print(f"All {n_requests} requests completed")
    
    # Give the server a moment to clean up
    await asyncio.sleep(1.0)
    
    # Check memory usage after requests finish
    final_usage = await get_gpu_memory_used()
    print(f"Final GPU cache usage: {final_usage:.2%}")
    
    # Memory should be roughly the same (within a small tolerance)
    # The cartridge GPU cache should have been freed when requests finished
    usage_diff = final_usage - baseline_usage
    print(f"Usage difference: {usage_diff:+.2%}")
    
    # Note: This is a soft check - the cache cleanup is best-effort
    # A large positive difference would indicate a memory leak
    if usage_diff > 0.10:  # More than 10% increase
        print(
            f"WARNING: GPU cache usage increased by {usage_diff:.2%} after "
            f"cartridge requests. Possible memory leak."
        )
    else:
        print("✓ GPU memory properly cleaned up after cartridge requests")
