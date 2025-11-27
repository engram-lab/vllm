# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cartridge loader for loading pre-computed KV cache data.

Cartridges support two use cases:

1. **Pre-computed KV Cache** (requires token_ids):
   - KV cache computed from specific tokens (e.g., system prompt)
   - token_ids are prepended to prompt for prefix caching
   - The KV cache tensors are optional (vLLM will recompute from tokens)

2. **Learned KV Cache / Soft Prompts** (no token_ids):
   - KV cache is trained via gradient descent (like prompt tuning)
   - No corresponding tokens exist - the KV tensors ARE the representation
   - Requires direct KV injection into attention (not token prepending)
"""

import logging
import re
from typing import Any, Optional

import torch

from vllm.logger import init_logger
from vllm.utils.cartridge_manager import get_cartridge_manager

logger = init_logger(__name__)

# Global cache for active cartridge KV tensors
# Maps request_id -> {layer_idx: (key_tensor, value_tensor)}
_active_cartridge_kv: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]] = {}


def set_active_cartridge_kv(
    request_id: str,
    cartridge_data: "CartridgeData",
    device: torch.device | str = "cuda",
) -> None:
    """Set the active cartridge KV for a request.
    
    This converts the CartridgeData format to the format expected by
    the attention layer: {layer_idx: (key, value)}.
    
    Args:
        request_id: The request ID to associate with this cartridge
        cartridge_data: The loaded cartridge data
        device: Device to move tensors to
    """
    if not cartridge_data.is_learned or not cartridge_data.kv_cache:
        return
    
    layer_kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    
    # Group by layer index
    keys_by_layer: dict[int, torch.Tensor] = {}
    values_by_layer: dict[int, torch.Tensor] = {}
    
    for key_name, tensor in cartridge_data.kv_cache.items():
        # Parse layer index from key name
        # e.g., "layers.5.attention.prefix.keys" -> 5
        match = re.match(r'layers\.(\d+)\.attention\.prefix\.(keys|values)', key_name)
        if match:
            layer_idx = int(match.group(1))
            kv_type = match.group(2)
            tensor = tensor.to(device)
            if kv_type == 'keys':
                keys_by_layer[layer_idx] = tensor
            else:
                values_by_layer[layer_idx] = tensor
    
    # Combine into layer_kv dict
    for layer_idx in keys_by_layer:
        if layer_idx in values_by_layer:
            layer_kv[layer_idx] = (keys_by_layer[layer_idx], values_by_layer[layer_idx])
    
    if layer_kv:
        _active_cartridge_kv[request_id] = layer_kv
        logger.info(f"Set cartridge KV for request {request_id}: {len(layer_kv)} layers")


def get_active_cartridge_kv(
    request_id: str,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]] | None:
    """Get the active cartridge KV for a request."""
    return _active_cartridge_kv.get(request_id)


def clear_active_cartridge_kv(request_id: str) -> None:
    """Clear the cartridge KV for a finished request."""
    if request_id in _active_cartridge_kv:
        del _active_cartridge_kv[request_id]
        logger.debug(f"Cleared cartridge KV for request {request_id}")


class CartridgeData:
    """
    Container for loaded cartridge data.

    Supported formats:
    
    1. Standard format (pre-computed KV cache):
    {
        'token_ids': torch.Tensor,  # Shape: (num_tokens,) - REQUIRED
        'kv_cache': dict[str, torch.Tensor],  # Optional: per-layer KV cache
        'metadata': dict,  # Optional metadata
    }

    2. Learned cartridge format (soft prompts - from cartridges library):
    {
        'layers.0.attention.prefix.keys': torch.Tensor,  # Shape: (num_heads, seq_len, head_dim)
        'layers.0.attention.prefix.values': torch.Tensor,
        'layers.1.attention.prefix.keys': torch.Tensor,
        ...
        'metadata': dict,  # Optional
    }
    
    For learned cartridges, there are no token_ids because the KV cache
    is the learned representation itself (similar to soft prompts / prompt tuning).
    """

    def __init__(
        self,
        kv_cache: Optional[dict[str, torch.Tensor]] = None,
        token_ids: Optional[torch.Tensor | list[int]] = None,
        metadata: Optional[dict[str, Any]] = None,
        is_learned: bool = False,
    ):
        """
        Initialize CartridgeData.
        
        Args:
            kv_cache: Dictionary mapping layer names to KV cache tensors
            token_ids: Token IDs for pre-computed cartridges (None for learned)
            metadata: Optional metadata dictionary
            is_learned: True if this is a learned cartridge (no token_ids)
        """
        self.is_learned = is_learned
        self.kv_cache = kv_cache or {}
        self.metadata = metadata or {}
        
        # Handle token_ids
        if token_ids is None:
            self.token_ids = torch.tensor([], dtype=torch.long)
            self.num_tokens = self._infer_num_tokens_from_kv_cache()
        elif isinstance(token_ids, list):
            if len(token_ids) == 0:
                self.token_ids = torch.tensor([], dtype=torch.long)
            else:
                self.token_ids = torch.tensor(token_ids, dtype=torch.long)
            self.num_tokens = len(self.token_ids)
        else:
            self.token_ids = token_ids
            self.num_tokens = len(token_ids)
    
    def _infer_num_tokens_from_kv_cache(self) -> int:
        """Infer the number of tokens from KV cache tensor shapes."""
        if not self.kv_cache:
            return 0
        
        # Get first KV cache tensor
        first_key = next(iter(self.kv_cache.keys()))
        first_tensor = self.kv_cache[first_key]
        
        if not isinstance(first_tensor, torch.Tensor):
            return 0
        
        # KV cache shape is typically (num_heads, seq_len, head_dim)
        # or (batch, num_heads, seq_len, head_dim)
        if first_tensor.dim() >= 3:
            # seq_len is the second-to-last dimension
            return first_tensor.shape[-2]
        elif first_tensor.dim() == 2:
            # Ambiguous - try to guess
            return max(first_tensor.shape)
        
        return 0
    
    def has_valid_token_ids(self) -> bool:
        """Check if this cartridge has valid token IDs for prepending."""
        return len(self.token_ids) > 0 and not self.is_learned

    @classmethod
    def from_dict(cls, data: dict) -> "CartridgeData":
        """Load CartridgeData from a dictionary (loaded .pt file).
        
        Supports multiple formats:
        1. Standard format: {'token_ids': ..., 'kv_cache': {...}, 'metadata': {...}}
        2. Learned cartridge format (flat): {'layers.X.attention.prefix.keys': ..., ...}
        3. Simplified format: {'token_ids': ...}
        """
        import re
        
        # Check if this is a learned cartridge (flat format with layer keys)
        # Pattern matches: layers.N.attention.prefix.keys/values
        # or trainable_keys/trainable_values/frozen_keys/frozen_values
        has_flat_kv_cache = any(
            re.match(r'layers\.\d+\.attention\.prefix\.(keys|values)', key)
            for key in data.keys()
        )
        has_trainable_kv = 'trainable_keys' in data or 'frozen_keys' in data
        
        is_learned_cartridge = has_flat_kv_cache or has_trainable_kv
        
        if is_learned_cartridge:
            # This is a learned cartridge from the cartridges library
            kv_cache = {}
            remaining_data = {}
            
            if has_trainable_kv:
                # Format from TrainableCache.save():
                # {'trainable_keys': [...], 'trainable_values': [...], 
                #  'frozen_keys': [...], 'frozen_values': [...]}
                # Convert to flat format
                trainable_keys = data.get('trainable_keys', [])
                trainable_values = data.get('trainable_values', [])
                frozen_keys = data.get('frozen_keys', [])
                frozen_values = data.get('frozen_values', [])
                
                # Combine trainable and frozen, preferring trainable
                all_keys = trainable_keys if trainable_keys else frozen_keys
                all_values = trainable_values if trainable_values else frozen_values
                
                if all_keys and all_values:
                    for layer_idx, (k, v) in enumerate(zip(all_keys, all_values)):
                        if k is not None:
                            kv_cache[f'layers.{layer_idx}.attention.prefix.keys'] = k
                        if v is not None:
                            kv_cache[f'layers.{layer_idx}.attention.prefix.values'] = v
            else:
                # Already in flat format
                for key, value in data.items():
                    if re.match(r'layers\.\d+\.attention\.prefix\.(keys|values)', key):
                        kv_cache[key] = value
                    else:
                        remaining_data[key] = value
            
            # For learned cartridges, token_ids are optional
            # If present, they might be the original context used for initialization
            token_ids = remaining_data.get("token_ids")
            
            # Extract metadata
            metadata = remaining_data.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            
            # Mark as learned cartridge
            metadata['_cartridge_type'] = 'learned'
            
            logger.info(
                f"Loaded learned cartridge with {len(kv_cache)} KV cache tensors. "
                f"This cartridge will use direct KV injection (no token prepending)."
            )
            
            return cls(
                kv_cache=kv_cache,
                token_ids=token_ids,
                metadata=metadata,
                is_learned=True,
            )
        else:
            # Standard format - requires token_ids
            token_ids = data.get("token_ids")
            if token_ids is None:
                raise ValueError(
                    "Cartridge must contain 'token_ids' field. "
                    "For learned cartridges (soft prompts), use the format with "
                    "'layers.X.attention.prefix.keys/values' or "
                    "'trainable_keys/trainable_values' keys."
                )

            kv_cache = data.get("kv_cache")
            metadata = data.get("metadata") or {}

            return cls(
                kv_cache=kv_cache,
                token_ids=token_ids,
                metadata=metadata,
                is_learned=False,
            )

    def __repr__(self) -> str:
        cartridge_type = "learned" if self.is_learned else "precomputed"
        return (
            f"CartridgeData(type={cartridge_type}, "
            f"num_tokens={self.num_tokens}, "
            f"has_kv_cache={bool(self.kv_cache)}, "
            f"has_token_ids={self.has_valid_token_ids()}, "
            f"kv_layers={len(self.kv_cache)})"
        )


def load_cartridge(
    cartridge_id: str,
    source: str = "s3",
    force_redownload: bool = False,
) -> CartridgeData:
    """
    Load a KV cache cartridge.

    Args:
        cartridge_id: The identifier/path of the cartridge
        source: Source type ('s3' or 'local')
        force_redownload: If True, re-download even if cached

    Returns:
        CartridgeData containing the loaded cartridge

    Raises:
        ValueError: If cartridge format is invalid
        RuntimeError: If loading fails
    """
    logger.info(
        f"Loading cartridge: {cartridge_id} (source={source}, "
        f"force_redownload={force_redownload})"
    )

    # Get the cartridge manager and download/load the cartridge
    manager = get_cartridge_manager()
    cartridge_tensor = manager.get_cartridge(
        cartridge_id=cartridge_id,
        source=source,
        force_redownload=force_redownload,
    )

    # Parse the cartridge data
    if isinstance(cartridge_tensor, dict):
        cartridge_data = CartridgeData.from_dict(cartridge_tensor)
    else:
        raise ValueError(
            f"Invalid cartridge format. Expected dict, got {type(cartridge_tensor)}"
        )

    logger.info(f"Successfully loaded cartridge: {cartridge_data}")
    return cartridge_data


def load_cartridges_from_request(
    cartridges_spec: list[dict[str, Any]],
) -> list[CartridgeData]:
    """
    Load multiple cartridges from a request specification.

    Args:
        cartridges_spec: List of cartridge specifications from the request.
            Each spec should have: id, source, force_redownload

    Returns:
        List of loaded CartridgeData objects

    Raises:
        RuntimeError: If any cartridge fails to load
    """
    loaded_cartridges = []

    for spec in cartridges_spec:
        cartridge_id = spec.get("id")
        source = spec.get("source", "s3")
        force_redownload = spec.get("force_redownload", False)

        if not cartridge_id:
            logger.warning(f"Skipping cartridge with missing id: {spec}")
            continue

        try:
            cartridge_data = load_cartridge(
                cartridge_id=cartridge_id,
                source=source,
                force_redownload=force_redownload,
            )
            loaded_cartridges.append(cartridge_data)
        except Exception as e:
            logger.error(f"Failed to load cartridge {cartridge_id}: {e}")
            raise RuntimeError(f"Failed to load cartridge {cartridge_id}: {e}") from e

    return loaded_cartridges
