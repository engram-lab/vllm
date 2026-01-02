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
import logging
import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import filelock
import torch

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

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"AdapterManager initialized with cache_dir: {self.cache_dir}")

    def _get_cache_path(self, uri: str) -> Path:
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

    def _download_from_s3(self, s3_uri: str, local_path: Path) -> None:
        """
        Download a file from S3 to a local path.

        Args:
            s3_uri: S3 URI (e.g., s3://bucket/path/to/file.pt)
            local_path: Local path to save the downloaded file

        Raises:
            ImportError: If boto3 is not installed
            Exception: If download fails
        """
        try:
            import boto3
            from botocore.exceptions import BotoCoreError, ClientError
        except ImportError as e:
            raise ImportError(
                "boto3 is required to download adapters from S3. "
                "Please install it with: pip install boto3"
            ) from e

        # Parse S3 URI
        logger.info(f"[AdapterManager Debug] _download_from_s3: s3_uri={s3_uri}, local_path={local_path}")
        
        parsed = urlparse(s3_uri)
        logger.info(f"[AdapterManager Debug] Parsed URI: scheme={parsed.scheme}, netloc={parsed.netloc}, path={parsed.path}")
        
        if parsed.scheme != "s3":
            logger.error(f"[AdapterManager Debug] Invalid S3 scheme: {parsed.scheme}")
            raise ValueError(f"Invalid S3 URI: {s3_uri}. Expected s3:// scheme.")

        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        
        logger.info(f"[AdapterManager Debug] Extracted: bucket={bucket}, key={key}")

        if not bucket or not key:
            logger.error(f"[AdapterManager Debug] Missing bucket or key: bucket={bucket}, key={key}")
            raise ValueError(
                f"Invalid S3 URI: {s3_uri}. Expected format: s3://bucket/path/to/file"
            )

        logger.info(f"[AdapterManager Debug] Starting S3 download: bucket={bucket}, key={key}")

        try:
            s3_client = boto3.client("s3")
            logger.info(f"[AdapterManager Debug] S3 client created successfully")

            # Download to a temporary file first, then move to final location
            temp_path = local_path.with_suffix(".tmp")
            logger.info(f"[AdapterManager Debug] Downloading to temp path: {temp_path}")
            
            s3_client.download_file(bucket, key, str(temp_path))
            
            if temp_path.exists():
                temp_size_mb = temp_path.stat().st_size / (1024 * 1024)
                logger.info(f"[AdapterManager Debug] Downloaded to temp file successfully (size: {temp_size_mb:.2f} MB)")
            else:
                logger.error(f"[AdapterManager Debug] Temp file does not exist after download: {temp_path}")
            
            logger.info(f"[AdapterManager Debug] Renaming {temp_path} -> {local_path}")
            temp_path.rename(local_path)
            
            final_size_mb = local_path.stat().st_size / (1024 * 1024)
            logger.info(f"[AdapterManager Debug] Successfully downloaded adapter to {local_path} (final size: {final_size_mb:.2f} MB)")
            
        except (BotoCoreError, ClientError) as e:
            logger.error(f"[AdapterManager Debug] AWS error during download: {type(e).__name__}: {e}")
            logger.error(f"[AdapterManager Debug] Bucket: {bucket}, Key: {key}")
            
            # Clean up partial download
            if temp_path.exists():
                logger.info(f"[AdapterManager Debug] Cleaning up partial download: {temp_path}")
                temp_path.unlink()
            
            raise RuntimeError(f"Failed to download adapter from {s3_uri}: {e}") from e
        except Exception as e:
            logger.error(f"[AdapterManager Debug] Unexpected error during download: {type(e).__name__}: {e}")
            import traceback
            logger.error(f"[AdapterManager Debug] Traceback:\n{traceback.format_exc()}")
            
            # Clean up partial download
            if temp_path.exists():
                logger.info(f"[AdapterManager Debug] Cleaning up partial download: {temp_path}")
                temp_path.unlink()
            
            raise

    def get_adapter(
        self,
        adapter_id: str,
        source: str = "s3",
        force_redownload: bool = False,
    ) -> torch.Tensor:
        """
        Get an adapter, downloading from S3 or loading from local path.

        Args:
            adapter_id: The identifier/path of the adapter.
                       For S3: s3://bucket/path/to/adapter.pt
                       For local: /path/to/local/adapter.pt
            source: Source type ('s3' or 'local')
            force_redownload: If True, re-download even if cached

        Returns:
            Loaded adapter tensor data

        Raises:
            FileNotFoundError: If local file doesn't exist
            RuntimeError: If download or loading fails
        """
        logger.info(f"[AdapterManager Debug] get_adapter called: adapter_id={adapter_id}, source={source}, force_redownload={force_redownload}")
        
        if source == "local":
            # Load directly from local path
            local_path = Path(adapter_id)
            logger.info(f"[AdapterManager Debug] Checking local path: {local_path}")
            
            if not local_path.exists():
                logger.error(f"[AdapterManager Debug] Local adapter not found: {adapter_id}")
                raise FileNotFoundError(f"Local adapter not found: {adapter_id}")

            logger.info(f"[AdapterManager Debug] Loading adapter from local path: {adapter_id}")
            try:
                data = torch.load(local_path, map_location="cpu")
                logger.info(f"[AdapterManager Debug] Loaded adapter, type={type(data)}")
                if isinstance(data, dict):
                    logger.info(f"[AdapterManager Debug] Adapter keys: {list(data.keys())[:10]}{'...' if len(data.keys()) > 10 else ''}")
                return data
            except Exception as e:
                logger.error(f"[AdapterManager Debug] Failed to load local adapter: {e}")
                raise

        elif source == "s3":
            # Get cache path
            cache_path = self._get_cache_path(adapter_id)
            lock_path = cache_path.with_suffix(".lock")
            
            logger.info(f"[AdapterManager Debug] Cache path: {cache_path}")
            logger.info(f"[AdapterManager Debug] Lock path: {lock_path}")

            # Use file lock to prevent concurrent downloads
            logger.info(f"[AdapterManager Debug] Acquiring file lock...")
            with filelock.FileLock(lock_path, timeout=300):
                logger.info(f"[AdapterManager Debug] File lock acquired")
                
                # Check if we need to download
                should_download = force_redownload or not cache_path.exists()
                logger.info(f"[AdapterManager Debug] should_download={should_download}, cache_exists={cache_path.exists()}")

                if should_download:
                    logger.info(
                        f"[AdapterManager Debug] Downloading adapter from S3 (force_redownload={force_redownload})"
                    )
                    self._download_from_s3(adapter_id, cache_path)
                    
                    if cache_path.exists():
                        file_size_mb = cache_path.stat().st_size / (1024 * 1024)
                        logger.info(f"[AdapterManager Debug] Download complete, file size: {file_size_mb:.2f} MB")
                    else:
                        logger.error(f"[AdapterManager Debug] Download failed, file does not exist: {cache_path}")
                else:
                    file_size_mb = cache_path.stat().st_size / (1024 * 1024)
                    logger.info(f"[AdapterManager Debug] Using cached adapter: {cache_path} (size: {file_size_mb:.2f} MB)")

                # Load the adapter
                try:
                    logger.info(f"[AdapterManager Debug] Loading adapter from {cache_path}...")
                    data = torch.load(cache_path, map_location="cpu")
                    logger.info(f"[AdapterManager Debug] Loaded adapter successfully, type={type(data)}")
                    
                    if isinstance(data, dict):
                        logger.info(f"[AdapterManager Debug] Adapter dict has {len(data)} keys")
                        logger.info(f"[AdapterManager Debug] First 10 keys: {list(data.keys())[:10]}")
                    
                    return data
                    
                except Exception as e:
                    logger.error(f"[AdapterManager Debug] Failed to load adapter from {cache_path}: {type(e).__name__}: {e}")
                    
                    # If loading failed, try re-downloading once
                    if not should_download:
                        logger.info("[AdapterManager Debug] Retrying with fresh download...")
                        self._download_from_s3(adapter_id, cache_path)
                        logger.info(f"[AdapterManager Debug] Retry download complete, attempting to load again...")
                        data = torch.load(cache_path, map_location="cpu")
                        logger.info(f"[AdapterManager Debug] Retry successful, loaded adapter")
                        return data
                    
                    logger.error(f"[AdapterManager Debug] Cannot retry, already attempted download")
                    raise RuntimeError(f"Failed to load adapter: {e}") from e

        else:
            logger.error(f"[AdapterManager Debug] Unknown source type: {source}")
            raise ValueError(f"Unknown source type: {source}. Expected 's3' or 'local'.")

    def download_json(
        self,
        json_id: str,
        source: str = "s3",
        target_path: Optional[Path] = None,
    ) -> dict:
        """
        Download and parse a JSON file from S3 or local path.

        Args:
            json_id: The identifier/path of the JSON file.
                    For S3: s3://bucket/path/to/file.json
                    For local: /path/to/local/file.json
            source: Source type ('s3' or 'local')
            target_path: Optional local path to save the downloaded JSON.
                        If not provided, returns parsed dict without saving.

        Returns:
            Parsed JSON content as a dictionary

        Raises:
            FileNotFoundError: If local file doesn't exist
            RuntimeError: If download or parsing fails
        """
        import json

        logger.info(f"[AdapterManager Debug] download_json called: json_id={json_id}, source={source}, target_path={target_path}")

        if source == "local":
            # Load directly from local path
            local_path = Path(json_id)
            logger.info(f"[AdapterManager Debug] Checking local path: {local_path}")
            
            if not local_path.exists():
                logger.error(f"[AdapterManager Debug] Local JSON file not found: {json_id}")
                raise FileNotFoundError(f"Local JSON file not found: {json_id}")

            logger.info(f"[AdapterManager Debug] Loading JSON from local path: {json_id}")
            try:
                with open(local_path, 'r') as f:
                    data = json.load(f)
                logger.info(f"[AdapterManager Debug] Successfully loaded JSON, keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
            except Exception as e:
                logger.error(f"[AdapterManager Debug] Failed to parse JSON from {local_path}: {e}")
                raise
            
            # Optionally copy to target path
            if target_path is not None and target_path != local_path:
                import shutil
                logger.info(f"[AdapterManager Debug] Copying JSON to target path: {target_path}")
                shutil.copy(local_path, target_path)
                logger.info(f"[AdapterManager Debug] Copied JSON to {target_path}")
            
            return data

        elif source == "s3":
            # Download from S3
            try:
                import boto3
                from botocore.exceptions import BotoCoreError, ClientError
            except ImportError as e:
                logger.error(f"[AdapterManager Debug] boto3 not installed")
                raise ImportError(
                    "boto3 is required to download from S3. "
                    "Please install it with: pip install boto3"
                ) from e

            # Parse S3 URI
            if json_id.startswith("s3://"):
                s3_path = json_id[5:]  # Remove s3:// prefix
                bucket, key = s3_path.split("/", 1)
                logger.info(f"[AdapterManager Debug] Parsed S3 URI: bucket={bucket}, key={key}")
            else:
                logger.error(f"[AdapterManager Debug] Invalid S3 URI (missing s3:// prefix): {json_id}")
                raise ValueError(f"Invalid S3 URI: {json_id}. Expected s3:// prefix.")

            logger.info(f"[AdapterManager Debug] Downloading JSON from S3: {json_id}")

            try:
                s3_client = boto3.client("s3")
                logger.info(f"[AdapterManager Debug] Created S3 client, calling get_object...")
                
                response = s3_client.get_object(Bucket=bucket, Key=key)
                logger.info(f"[AdapterManager Debug] S3 get_object successful, content length: {response.get('ContentLength', 'unknown')} bytes")
                
                body_content = response["Body"].read()
                logger.info(f"[AdapterManager Debug] Read S3 body, size: {len(body_content)} bytes")
                
                data = json.loads(body_content)
                logger.info(f"[AdapterManager Debug] Parsed JSON successfully, keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
                
                # Save to target path if provided
                if target_path is not None:
                    logger.info(f"[AdapterManager Debug] Saving JSON to target path: {target_path}")
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(target_path, 'w') as f:
                        json.dump(data, f, indent=2)
                    file_size = target_path.stat().st_size
                    logger.info(f"[AdapterManager Debug] Saved JSON to {target_path} (size: {file_size} bytes)")
                
                return data

            except (BotoCoreError, ClientError) as e:
                logger.error(f"[AdapterManager Debug] AWS error downloading JSON: {type(e).__name__}: {e}")
                logger.error(f"[AdapterManager Debug] Bucket: {bucket}, Key: {key}")
                raise RuntimeError(f"Failed to download JSON from {json_id}: {e}") from e
            except Exception as e:
                logger.error(f"[AdapterManager Debug] Unexpected error downloading JSON: {type(e).__name__}: {e}")
                import traceback
                logger.error(f"[AdapterManager Debug] Traceback:\n{traceback.format_exc()}")
                raise

        else:
            logger.error(f"[AdapterManager Debug] Unknown source type: {source}")
            raise ValueError(f"Unknown source type: {source}. Expected 's3' or 'local'.")

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

