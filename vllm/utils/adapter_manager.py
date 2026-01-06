# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapter manager for downloading and caching adapter artifacts.

Handles downloading and caching of:
- KV cache cartridges (prefix adapters)
- LoRA adapter weights
- Other trained artifacts

This class is storage-agnostic and doesn't handle adapter-specific logic.
"""

import hashlib
import json
import logging
import os
from pathlib import Path as StdPath
from typing import Optional

import filelock
import torch

from vllm.path import Path

logger = logging.getLogger(__name__)


class AdapterManager:
    """
    Manager for downloading and caching adapter artifacts from S3 or local paths.

    This class handles:
    - Downloading adapters from S3 URIs (s3://bucket/path/to/adapter.pt)
    - Loading adapters from local file paths
    - Caching downloaded files to avoid re-downloading
    - Thread-safe file locking during downloads
    
    Supports multiple adapter types:
    - KV cache cartridges (prefix adapters)
    - LoRA adapter weights
    - Any other trained model artifacts
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the AdapterManager.

        Args:
            cache_dir: Directory to cache downloaded adapters.
                      Defaults to ~/.cache/vllm/adapters
        """
        if cache_dir is None:
            cache_dir = os.path.join(
                os.path.expanduser("~"), ".cache", "vllm", "adapters"
            )

        self.cache_dir = StdPath(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"AdapterManager initialized with cache_dir: {self.cache_dir}")

    def _get_cache_path(self, uri: str) -> StdPath:
        """
        Get the local cache path for a given URI.

        Args:
            uri: The S3 URI or identifier of the adapter

        Returns:
            Path to the cached file
        """
        # Use hash of URI as filename to avoid path traversal issues
        uri_hash = hashlib.sha256(uri.encode()).hexdigest()
        return self.cache_dir / f"{uri_hash}.pt"

    def get_adapter(
        self,
        adapter_id: str,
        force_redownload: bool = False,
    ) -> torch.Tensor:
        """
        Get an adapter, downloading from S3 or loading from local path.

        The adapter_id can be either:
        - S3 URI: s3://bucket/path/to/adapter.pt
        - Local path: /path/to/local/adapter.pt or ./adapter.pt

        The Path abstraction automatically handles both cases.

        Args:
            adapter_id: The identifier/path of the adapter (S3 URI or local path)
            force_redownload: If True, re-download even if cached (only applies to remote files)

        Returns:
            Loaded adapter tensor data

        Raises:
            FileNotFoundError: If file doesn't exist
            RuntimeError: If download or loading fails
        """
        source_path = Path(adapter_id)
        
        # For remote paths (S3), use caching
        if adapter_id.startswith("s3://"):
            cache_path = self._get_cache_path(adapter_id)
            lock_path = cache_path.with_suffix(".lock")

            # Use file lock to prevent concurrent downloads
            with filelock.FileLock(lock_path, timeout=300):
                # Check if we need to download
                should_download = force_redownload or not cache_path.exists()

                if should_download:
                    logger.info(f"Downloading adapter from {adapter_id}")
                    try:
                        # Use Path's copy method to download from S3 to local cache
                        source_path.copy(str(cache_path))
                        final_size_mb = cache_path.stat().st_size / (1024 * 1024)
                        logger.info(f"Successfully downloaded adapter to {cache_path} ({final_size_mb:.2f} MB)")
                    except Exception as e:
                        # Clean up partial download
                        if cache_path.exists():
                            cache_path.unlink()
                        raise RuntimeError(f"Failed to download adapter from {adapter_id}: {e}") from e
                else:
                    logger.info(f"Using cached adapter: {cache_path}")

                # Load the adapter
                try:
                    data = torch.load(cache_path, map_location="cpu")
                    return data
                except Exception as e:
                    # If loading failed, try re-downloading once
                    if not should_download:
                        logger.info("Retrying with fresh download...")
                        try:
                            source_path.copy(str(cache_path))
                            data = torch.load(cache_path, map_location="cpu")
                            return data
                        except Exception as retry_e:
                            raise RuntimeError(f"Failed to load adapter even after retry: {retry_e}") from retry_e
                    raise RuntimeError(f"Failed to load adapter: {e}") from e
        
        # For local paths, load directly
        else:
            if not source_path.exists():
                raise FileNotFoundError(f"Local adapter not found: {adapter_id}")

            logger.info(f"Loading adapter from local path: {adapter_id}")
            try:
                # For local paths, we can use the path directly as it's file-like
                data = torch.load(str(source_path), map_location="cpu")
                return data
            except Exception as e:
                logger.error(f"Failed to load local adapter: {e}")
                raise

    def download_json(
        self,
        json_id: str,
        target_path: Optional[StdPath] = None,
    ) -> dict:
        """
        Download and parse a JSON file from S3 or local path.

        The json_id can be either:
        - S3 URI: s3://bucket/path/to/file.json
        - Local path: /path/to/local/file.json

        The Path abstraction automatically handles both cases.

        Args:
            json_id: The identifier/path of the JSON file (S3 URI or local path)
            target_path: Optional local path to save the downloaded JSON.
                        If not provided, returns parsed dict without saving.

        Returns:
            Parsed JSON content as a dictionary

        Raises:
            FileNotFoundError: If file doesn't exist
            RuntimeError: If download or parsing fails
        """
        source_path = Path(json_id)
        
        if not source_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_id}")

        # Read and parse JSON using the unified Path API
        try:
            content = source_path.read_text()
            data = json.loads(content)
        except Exception as e:
            raise RuntimeError(f"Failed to read or parse JSON from {json_id}: {e}") from e
        
        # Save to target path if provided
        if target_path is not None:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with open(target_path, 'w') as f:
                json.dump(data, f, indent=2)
        
        return data

    def clear_cache(self) -> None:
        """Clear all cached adapters."""
        logger.info("Clearing adapter cache...")
        for file in self.cache_dir.glob("*.pt"):
            file.unlink()
        for file in self.cache_dir.glob("*.lock"):
            file.unlink()
        logger.info("Cache cleared")


# Global adapter manager instance
_global_adapter_manager: Optional[AdapterManager] = None


def get_adapter_manager() -> AdapterManager:
    """Get or create the global adapter manager instance."""
    global _global_adapter_manager
    if _global_adapter_manager is None:
        _global_adapter_manager = AdapterManager()
    return _global_adapter_manager

