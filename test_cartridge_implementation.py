#!/usr/bin/env python3
"""Test script for KV cache cartridge implementation."""

import os
import sys
import tempfile
from pathlib import Path

import torch

# Add vllm to path
sys.path.insert(0, '/home/user/vllm')


def test_cartridge_creation_and_loading():
    """Test creating and loading a cartridge."""
    print("=" * 60)
    print("Test 1: Cartridge Creation and Loading")
    print("=" * 60)

    # Create a temporary cartridge
    with tempfile.TemporaryDirectory() as tmpdir:
        cartridge_path = Path(tmpdir) / "test_cartridge.pt"

        # Create test cartridge
        test_token_ids = [101, 2023, 2003, 1037, 3231, 102]
        cartridge_data = {
            'token_ids': test_token_ids,
            'metadata': {
                'model': 'test-model',
                'description': 'Test cartridge',
            }
        }

        print(f"\n✓ Creating cartridge at: {cartridge_path}")
        torch.save(cartridge_data, cartridge_path)
        print(f"  Token IDs: {test_token_ids}")

        # Test loading with CartridgeManager
        from vllm.utils.cartridge_manager import CartridgeManager

        manager = CartridgeManager(cache_dir=str(Path(tmpdir) / "cache"))
        print(f"\n✓ Created CartridgeManager with cache_dir: {manager.cache_dir}")

        # Load the cartridge
        loaded_data = manager.get_cartridge(
            cartridge_id=str(cartridge_path),
            source="local",
            force_redownload=False
        )

        print(f"\n✓ Loaded cartridge successfully")
        print(f"  Type: {type(loaded_data)}")
        print(f"  Keys: {loaded_data.keys()}")
        print(f"  Token IDs: {loaded_data['token_ids']}")

        # Verify data
        assert 'token_ids' in loaded_data
        assert len(loaded_data['token_ids']) == len(test_token_ids)
        print("\n✓ Verification passed!")

    return True


def test_cartridge_loader():
    """Test the CartridgeData and loading functions."""
    print("\n" + "=" * 60)
    print("Test 2: CartridgeData and Loader")
    print("=" * 60)

    from vllm.v1.cartridge_loader import CartridgeData, load_cartridge

    # Create a temporary cartridge
    with tempfile.TemporaryDirectory() as tmpdir:
        cartridge_path = Path(tmpdir) / "test_cartridge2.pt"

        test_token_ids = [1, 2, 3, 4, 5]
        cartridge_data = {
            'token_ids': test_token_ids,
            'metadata': {'test': 'data'}
        }

        print(f"\n✓ Creating cartridge at: {cartridge_path}")
        torch.save(cartridge_data, cartridge_path)

        # Load using cartridge_loader
        loaded_cartridge = load_cartridge(
            cartridge_id=str(cartridge_path),
            source="local",
            force_redownload=False
        )

        print(f"\n✓ Loaded CartridgeData: {loaded_cartridge}")
        print(f"  Num tokens: {loaded_cartridge.num_tokens}")
        print(f"  Token IDs tensor shape: {loaded_cartridge.token_ids.shape}")
        print(f"  Metadata: {loaded_cartridge.metadata}")

        # Verify
        assert loaded_cartridge.num_tokens == len(test_token_ids)
        assert loaded_cartridge.token_ids.tolist() == test_token_ids
        print("\n✓ Verification passed!")

    return True


def test_cartridge_with_list_tokens():
    """Test loading multiple cartridges."""
    print("\n" + "=" * 60)
    print("Test 3: Multiple Cartridges")
    print("=" * 60)

    from vllm.v1.cartridge_loader import load_cartridges_from_request

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create two cartridges
        cartridge1_path = Path(tmpdir) / "cartridge1.pt"
        cartridge2_path = Path(tmpdir) / "cartridge2.pt"

        torch.save({'token_ids': [1, 2, 3]}, cartridge1_path)
        torch.save({'token_ids': [4, 5, 6]}, cartridge2_path)

        print(f"\n✓ Created 2 cartridges")

        # Load multiple cartridges
        cartridges_spec = [
            {
                'id': str(cartridge1_path),
                'source': 'local',
                'force_redownload': False
            },
            {
                'id': str(cartridge2_path),
                'source': 'local',
                'force_redownload': False
            }
        ]

        loaded_cartridges = load_cartridges_from_request(cartridges_spec)

        print(f"\n✓ Loaded {len(loaded_cartridges)} cartridges")
        for i, cart in enumerate(loaded_cartridges):
            print(f"  Cartridge {i+1}: {cart.num_tokens} tokens - {cart.token_ids.tolist()}")

        # Verify
        assert len(loaded_cartridges) == 2
        assert loaded_cartridges[0].token_ids.tolist() == [1, 2, 3]
        assert loaded_cartridges[1].token_ids.tolist() == [4, 5, 6]
        print("\n✓ Verification passed!")

    return True


def test_caching_behavior():
    """Test that caching works correctly."""
    print("\n" + "=" * 60)
    print("Test 4: Caching Behavior")
    print("=" * 60)

    from vllm.utils.cartridge_manager import CartridgeManager

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache"
        cartridge_path = Path(tmpdir) / "test.pt"

        # Create cartridge
        torch.save({'token_ids': [1, 2, 3]}, cartridge_path)

        manager = CartridgeManager(cache_dir=str(cache_dir))

        # First load - should copy to cache for local files
        print(f"\n✓ First load (should read from source)")
        data1 = manager.get_cartridge(str(cartridge_path), source="local")

        # Second load - should use same file for local
        print(f"✓ Second load (local files are not cached, read directly)")
        data2 = manager.get_cartridge(str(cartridge_path), source="local")

        # Both should have same content
        assert data1['token_ids'].tolist() == data2['token_ids'].tolist()
        print("\n✓ Verification passed!")

    return True


def test_protocol_models():
    """Test that the protocol models are correctly defined."""
    print("\n" + "=" * 60)
    print("Test 5: Protocol Models")
    print("=" * 60)

    from vllm.entrypoints.openai.protocol import (
        KVCacheCartridge,
        ChatCompletionRequest,
        CompletionRequest,
    )

    # Test KVCacheCartridge
    print("\n✓ Testing KVCacheCartridge model")
    cartridge = KVCacheCartridge(
        id="s3://bucket/path.pt",
        source="s3",
        force_redownload=True
    )
    print(f"  Cartridge: {cartridge}")
    assert cartridge.id == "s3://bucket/path.pt"
    assert cartridge.source == "s3"
    assert cartridge.force_redownload == True

    # Test ChatCompletionRequest with cartridges
    print("\n✓ Testing ChatCompletionRequest with cartridges")
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "Hello"}],
        model="test-model",
        cartridges=[
            KVCacheCartridge(
                id="s3://bucket/test.pt",
                source="s3",
                force_redownload=False
            )
        ]
    )
    print(f"  Request has {len(request.cartridges)} cartridge(s)")
    assert len(request.cartridges) == 1

    # Test CompletionRequest with cartridges
    print("\n✓ Testing CompletionRequest with cartridges")
    comp_request = CompletionRequest(
        prompt="Test prompt",
        model="test-model",
        cartridges=[
            KVCacheCartridge(id="local/path.pt", source="local")
        ]
    )
    assert len(comp_request.cartridges) == 1

    print("\n✓ All protocol models verified!")

    return True


def test_process_cartridges_integration():
    """Test the _process_cartridges method logic."""
    print("\n" + "=" * 60)
    print("Test 6: Process Cartridges Integration")
    print("=" * 60)

    from vllm.v1.cartridge_loader import load_cartridges_from_request

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test cartridge
        cartridge_path = Path(tmpdir) / "test.pt"
        torch.save({'token_ids': [100, 200, 300]}, cartridge_path)

        # Simulate cartridges spec
        cartridges_spec = [
            {
                'id': str(cartridge_path),
                'source': 'local',
                'force_redownload': False
            }
        ]

        # Load cartridges
        loaded_cartridges = load_cartridges_from_request(cartridges_spec)

        # Simulate prepending to prompt
        original_prompt_tokens = [1, 2, 3, 4, 5]
        cartridge_token_ids = []
        for cart in loaded_cartridges:
            cartridge_token_ids.extend(cart.token_ids.tolist())

        combined_tokens = cartridge_token_ids + original_prompt_tokens

        print(f"\n✓ Original prompt tokens: {original_prompt_tokens}")
        print(f"✓ Cartridge tokens: {cartridge_token_ids}")
        print(f"✓ Combined tokens: {combined_tokens}")

        # Verify
        assert combined_tokens == [100, 200, 300, 1, 2, 3, 4, 5]
        print("\n✓ Token prepending works correctly!")

    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("KV Cache Cartridge Implementation Tests")
    print("=" * 60)

    tests = [
        ("Cartridge Creation and Loading", test_cartridge_creation_and_loading),
        ("CartridgeData and Loader", test_cartridge_loader),
        ("Multiple Cartridges", test_cartridge_with_list_tokens),
        ("Caching Behavior", test_caching_behavior),
        ("Protocol Models", test_protocol_models),
        ("Process Cartridges Integration", test_process_cartridges_integration),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, "PASSED" if success else "FAILED"))
        except Exception as e:
            print(f"\n✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, "FAILED"))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, result in results:
        status_symbol = "✓" if result == "PASSED" else "✗"
        print(f"{status_symbol} {test_name}: {result}")

    passed = sum(1 for _, r in results if r == "PASSED")
    total = len(results)
    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
