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

import re
from typing import Any, Optional

import torch

from vllm.logger import init_logger
from vllm.utils.cartridge_manager import get_cartridge_manager

logger = init_logger(__name__)

# Compiled pattern for matching KV cache layer keys
_KV_LAYER_PATTERN = re.compile(r'layers\.(\d+)\.attention\.prefix\.(keys|values)')

# Global cache for active cartridge KV tensors
# Maps request_id -> {layer_idx: (key_tensor, value_tensor)}
_active_cartridge_kv: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]] = {}

# In-memory cache for parsed CartridgeData objects
# Maps (cartridge_id, source) -> CartridgeData
_cartridge_data_cache: dict[tuple[str, str], "CartridgeData"] = {}


def _parse_kv_cache_to_layers(
    kv_cache: dict[str, torch.Tensor],
    device: torch.device | str | None = None,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Parse KV cache dict into keys_by_layer and values_by_layer.
    
    Handles two input formats:
    1. TrainableCache format: (1, num_kv_heads, seq_len, head_dim)
       - This is how engrams/cartridges library saves trainable KV caches
    2. Legacy format: (1, seq_len, num_kv_heads, head_dim)
       - Some older cartridge formats may use this
    
    Output shape per tensor: (num_kv_heads, seq_len, head_dim)
    """
    keys_by_layer: dict[int, torch.Tensor] = {}
    values_by_layer: dict[int, torch.Tensor] = {}
    
    for key_name, tensor in kv_cache.items():
        match = _KV_LAYER_PATTERN.match(key_name)
        if not match:
            continue
        
        layer_idx = int(match.group(1))
        kv_type = match.group(2)
        
        # Transform 4D tensor to 3D (num_kv_heads, seq_len, head_dim)
        if tensor.dim() == 4:
            # After squeeze(0), we have shape (A, B, C) where we need to figure out
            # if this is (num_kv_heads, seq_len, head_dim) or (seq_len, num_kv_heads, head_dim)
            # 
            # TrainableCache saves as (1, num_kv_heads, seq_len, head_dim)
            # So squeeze(0) gives (num_kv_heads, seq_len, head_dim) - already correct!
            #
            # Legacy format was (1, seq_len, num_kv_heads, head_dim)
            # So squeeze(0) gives (seq_len, num_kv_heads, head_dim) - needs permute
            #
            # Heuristic: head_dim is typically 64, 128, etc. and is almost always 
            # the last dimension. num_kv_heads is typically 1-128. seq_len can vary.
            # If dim[0] < dim[1] and dim[2] looks like head_dim, it's TrainableCache format.
            tensor = tensor.squeeze(0)
            dim0, dim1, dim2 = tensor.shape
            
            # TrainableCache format: (num_kv_heads, seq_len, head_dim)
            # - dim0 (num_kv_heads) is typically small (1-128)
            # - dim1 (seq_len) is the cartridge length
            # - dim2 (head_dim) is typically 64, 128, etc.
            # 
            # Legacy format: (seq_len, num_kv_heads, head_dim)
            # - In this case, we need dim1 < dim0 AND dim1 looks like num_heads
            #
            # Best heuristic: if dim0 is a common num_heads value (power of 2, <= 128)
            # and dim2 is a common head_dim (64, 128, 256), assume TrainableCache format
            is_trainable_cache_format = (
                dim0 <= 128 and  # num_kv_heads is typically small
                dim0 > 0 and
                (dim0 & (dim0 - 1)) == 0 and  # num_kv_heads is power of 2
                dim2 in (64, 80, 96, 128, 256)  # common head_dim values
            )
            
            if not is_trainable_cache_format:
                # Legacy format: (seq_len, num_kv_heads, head_dim) -> (num_kv_heads, seq_len, head_dim)
                logger.info(
                    f"[CARTRIDGE] Detected legacy format for {key_name}: "
                    f"shape {(dim0, dim1, dim2)} -> permuting to (num_kv_heads, seq_len, head_dim)"
                )
                tensor = tensor.permute(1, 0, 2)
            else:
                # TrainableCache format already in correct shape (num_kv_heads, seq_len, head_dim)
                logger.info(
                    f"[CARTRIDGE] Detected TrainableCache format for {key_name}: "
                    f"shape {tensor.shape} already in (num_kv_heads, seq_len, head_dim)"
                )
        
        if device is not None:
            tensor = tensor.to(device)
        
        if kv_type == 'keys':
            keys_by_layer[layer_idx] = tensor
        else:
            values_by_layer[layer_idx] = tensor
    
    return keys_by_layer, values_by_layer


def set_active_cartridge_kv(
    request_id: str,
    cartridge_data: "CartridgeData",
    device: torch.device | str = "cuda",
) -> None:
    """Set the active cartridge KV for a request."""
    if not cartridge_data.is_learned or not cartridge_data.kv_cache:
        return
    
    keys_by_layer, values_by_layer = _parse_kv_cache_to_layers(
        cartridge_data.kv_cache, device
    )
    
    # Combine into layer_kv dict (only layers with both keys and values)
    layer_kv = {
        idx: (keys_by_layer[idx], values_by_layer[idx])
        for idx in keys_by_layer
        if idx in values_by_layer
    }
    
    if layer_kv:
        _active_cartridge_kv[request_id] = layer_kv
        first_layer = next(iter(layer_kv.values()))
        logger.info(
            f"[CARTRIDGE] Set KV for request {request_id}: "
            f"{len(layer_kv)} layers, shape: {first_layer[0].shape}"
        )


def get_active_cartridge_kv(
    request_id: str,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]] | None:
    """Get the active cartridge KV for a request."""
    return _active_cartridge_kv.get(request_id)


def get_all_active_cartridge_request_ids() -> list[str]:
    """Get all request IDs that have active cartridge KV."""
    return list(_active_cartridge_kv.keys())


def clear_active_cartridge_kv(request_id: str) -> None:
    """Clear the cartridge KV for a finished request."""
    if request_id in _active_cartridge_kv:
        del _active_cartridge_kv[request_id]
        logger.debug(f"Cleared cartridge KV for request {request_id}")


def clear_cartridge_data_cache() -> None:
    """Clear the in-memory cartridge data cache."""
    _cartridge_data_cache.clear()
    logger.info("Cleared cartridge data cache")


class CartridgeData:
    """Container for loaded cartridge data.

    Supported formats:
    
    1. Standard format (pre-computed KV cache):
       {'token_ids': Tensor, 'kv_cache': dict, 'metadata': dict}

    2. Learned cartridge format (from cartridges library):
       {'layers.N.attention.prefix.keys': Tensor, ...}
       or {'trainable_keys': [...], 'trainable_values': [...], ...}
    """

    def __init__(
        self,
        kv_cache: Optional[dict[str, torch.Tensor]] = None,
        token_ids: Optional[torch.Tensor | list[int]] = None,
        metadata: Optional[dict[str, Any]] = None,
        is_learned: bool = False,
    ):
        self.is_learned = is_learned
        self.kv_cache = kv_cache or {}
        self.metadata = metadata or {}
        self._stacked_kv: list[torch.Tensor] | None = None
        
        # Normalize token_ids to tensor
        if token_ids is None:
            self.token_ids = torch.tensor([], dtype=torch.long)
            self.num_tokens = self._infer_num_tokens_from_kv_cache()
        else:
            self.token_ids = (
                torch.tensor(token_ids, dtype=torch.long)
                if isinstance(token_ids, list)
                else token_ids
            )
            self.num_tokens = len(self.token_ids)
    
    def _infer_num_tokens_from_kv_cache(self) -> int:
        """Infer num tokens from KV cache shapes."""
        if not self.kv_cache:
            return 0
        # Shape: (num_kv_heads, seq_len, head_dim) or (batch, num_kv_heads, seq_len, head_dim)
        return next(iter(self.kv_cache.values())).shape[-2]
    
    def has_valid_token_ids(self) -> bool:
        """Check if this cartridge has valid token IDs for prepending."""
        return len(self.token_ids) > 0 and not self.is_learned

    def get_stacked_kv(self) -> list[torch.Tensor] | None:
        """Get stacked KV tensors for IPC, computing and caching on first call.
        
        Output: [stacked_keys, stacked_values] with shape
            (num_layers, num_kv_heads, seq_len, head_dim)
        """
        if not self.is_learned or not self.kv_cache:
            return None
        
        if self._stacked_kv is not None:
            return self._stacked_kv
        
        keys_by_layer, values_by_layer = _parse_kv_cache_to_layers(self.kv_cache)
        
        if keys_by_layer and values_by_layer:
            num_layers = max(keys_by_layer.keys()) + 1
            self._stacked_kv = [
                torch.stack([keys_by_layer[i] for i in range(num_layers)]),
                torch.stack([values_by_layer[i] for i in range(num_layers)]),
            ]
            logger.info(
                f"[CARTRIDGE] Computed stacked KV: "
                f"keys={self._stacked_kv[0].shape}, values={self._stacked_kv[1].shape}"
            )
        
        return self._stacked_kv

    @classmethod
    def from_dict(cls, data: dict) -> "CartridgeData":
        """Load CartridgeData from a dictionary."""
        has_flat_kv = any(_KV_LAYER_PATTERN.match(k) for k in data)
        has_trainable_kv = 'trainable_keys' in data or 'frozen_keys' in data
        is_learned = has_flat_kv or has_trainable_kv
        
        if not is_learned:
            # Standard format requires token_ids
            token_ids = data.get("token_ids")
            if token_ids is None:
                raise ValueError(
                    "Cartridge must contain 'token_ids' field. "
                    "For learned cartridges, use layer keys or trainable_keys format."
                )
            return cls(
                kv_cache=data.get("kv_cache"),
                token_ids=token_ids,
                metadata=data.get("metadata") or {},
                is_learned=False,
            )
        
        # Learned cartridge
        kv_cache = cls._extract_learned_kv_cache(data, has_trainable_kv)
        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        metadata['_cartridge_type'] = 'learned'
        
        logger.info(f"Loaded learned cartridge with {len(kv_cache)} KV cache tensors.")
        
        return cls(
            kv_cache=kv_cache,
            token_ids=data.get("token_ids"),
            metadata=metadata,
            is_learned=True,
        )
    
    @classmethod
    def _extract_learned_kv_cache(
        cls, data: dict, has_trainable_kv: bool
    ) -> dict[str, torch.Tensor]:
        """Extract KV cache from learned cartridge data."""
        if not has_trainable_kv:
            # Already in flat layer format
            return {k: v for k, v in data.items() if _KV_LAYER_PATTERN.match(k)}
        
        # TrainableCache format: convert to flat layer format
        all_keys = data.get('trainable_keys') or data.get('frozen_keys', [])
        all_values = data.get('trainable_values') or data.get('frozen_values', [])
        
        kv_cache = {}
        for layer_idx, (k, v) in enumerate(zip(all_keys, all_values)):
            if k is not None:
                kv_cache[f'layers.{layer_idx}.attention.prefix.keys'] = k
            if v is not None:
                kv_cache[f'layers.{layer_idx}.attention.prefix.values'] = v
        return kv_cache

    def __repr__(self) -> str:
        return (
            f"CartridgeData(type={'learned' if self.is_learned else 'precomputed'}, "
            f"num_tokens={self.num_tokens}, "
            f"has_kv_cache={bool(self.kv_cache)}, "
            f"kv_layers={len(self.kv_cache)})"
        )


def load_cartridge(
    cartridge_id: str,
    source: str = "s3",
    force_redownload: bool = False,
) -> CartridgeData:
    """Load a KV cache cartridge.

    Args:
        cartridge_id: The identifier/path of the cartridge
        source: Source type ('s3' or 'local')
        force_redownload: If True, re-download even if cached

    Returns:
        CartridgeData containing the loaded cartridge
    """
    cache_key = (cartridge_id, source)
    
    if not force_redownload and cache_key in _cartridge_data_cache:
        logger.debug(f"Using cached CartridgeData for: {cartridge_id}")
        return _cartridge_data_cache[cache_key]
    
    logger.info(f"Loading cartridge: {cartridge_id} (source={source})")

    manager = get_cartridge_manager()
    cartridge_tensor = manager.get_cartridge(
        cartridge_id=cartridge_id,
        source=source,
        force_redownload=force_redownload,
    )

    if not isinstance(cartridge_tensor, dict):
        raise ValueError(
            f"Invalid cartridge format. Expected dict, got {type(cartridge_tensor)}"
        )

    cartridge_data = CartridgeData.from_dict(cartridge_tensor)
    logger.info(f"Successfully loaded cartridge: {cartridge_data}")
    
    _cartridge_data_cache[cache_key] = cartridge_data
    return cartridge_data


def load_cartridges_from_request(
    cartridges_spec: list[dict[str, Any]],
) -> list[CartridgeData]:
    """Load multiple cartridges from a request specification."""
    loaded_cartridges = []

    for spec in cartridges_spec:
        cartridge_id = spec.get("id")
        if not cartridge_id:
            logger.warning(f"Skipping cartridge with missing id: {spec}")
            continue

        try:
            cartridge_data = load_cartridge(
                cartridge_id=cartridge_id,
                source=spec.get("source", "s3"),
                force_redownload=spec.get("force_redownload", False),
            )
            loaded_cartridges.append(cartridge_data)
        except Exception as e:
            logger.error(f"Failed to load cartridge {cartridge_id}: {e}")
            raise RuntimeError(f"Failed to load cartridge {cartridge_id}: {e}") from e

    return loaded_cartridges
