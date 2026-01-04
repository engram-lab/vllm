# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
from pathlib import Path
from typing import Optional

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.lora.resolver import LoRAResolver, LoRAResolverRegistry

logger = init_logger(__name__)


class S3LoRAResolver(LoRAResolver):
    """LoRA resolver that downloads adapters from S3.
    
    This resolver handles S3 paths in the format:
    - s3://bucket-name/path/to/adapter
    
    The adapter files will be downloaded to a local cache directory
    and validated before being loaded.
    """

    def __init__(self, lora_cache_dir: str):
        """Initialize the S3 LoRA resolver.
        
        Args:
            lora_cache_dir: Local directory to cache downloaded LoRA adapters
        """
        if not BOTO3_AVAILABLE:
            raise ImportError(
                "boto3 is required for S3LoRAResolver. "
                "Install it with: pip install boto3"
            )
        
        self.lora_cache_dir = lora_cache_dir
        self.s3_client = boto3.client('s3')
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.lora_cache_dir, exist_ok=True)

    def _parse_s3_path(self, s3_path: str) -> tuple[str, str]:
        """Parse S3 path into bucket and key.
        
        Args:
            s3_path: S3 path in format s3://bucket/key
            
        Returns:
            Tuple of (bucket_name, key_prefix)
        """
        if not s3_path.startswith("s3://"):
            raise ValueError(f"Invalid S3 path: {s3_path}. Must start with s3://")
        
        path_parts = s3_path[5:].split("/", 1)
        bucket = path_parts[0]
        key_prefix = path_parts[1] if len(path_parts) > 1 else ""
        
        return bucket, key_prefix

    def _download_s3_directory(self, bucket: str, prefix: str, local_dir: str) -> bool:
        """Download all files from an S3 prefix to a local directory.
        
        Args:
            bucket: S3 bucket name
            prefix: S3 key prefix (directory path)
            local_dir: Local directory to download files to
            
        Returns:
            True if download was successful, False otherwise
        """
        try:
            # List all objects with the given prefix
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
            
            downloaded_files = []
            for page in pages:
                if 'Contents' not in page:
                    continue
                    
                for obj in page['Contents']:
                    s3_key = obj['Key']
                    
                    # Skip directories
                    if s3_key.endswith('/'):
                        continue
                    
                    # Compute relative path
                    relative_path = s3_key[len(prefix):].lstrip('/')
                    local_file_path = os.path.join(local_dir, relative_path)
                    
                    # Create parent directories if needed
                    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                    
                    # Download the file
                    logger.info(f"Downloading s3://{bucket}/{s3_key} to {local_file_path}")
                    self.s3_client.download_file(bucket, s3_key, local_file_path)
                    downloaded_files.append(local_file_path)
            
            if not downloaded_files:
                logger.warning(f"No files found in s3://{bucket}/{prefix}")
                return False
            
            logger.info(f"Successfully downloaded {len(downloaded_files)} files from S3")
            return True
            
        except NoCredentialsError:
            logger.error("AWS credentials not found. Please configure AWS credentials.")
            return False
        except ClientError as e:
            logger.error(f"Error downloading from S3: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error downloading from S3: {e}")
            return False

    async def resolve_lora(
        self, base_model_name: str, lora_name: str
    ) -> Optional[LoRARequest]:
        """Resolve and download a LoRA adapter from S3.
        
        Args:
            base_model_name: The name/identifier of the base model
            lora_name: The S3 path or name of the LoRA adapter
                      (e.g., "s3://my-bucket/adapters/my-lora" or just "my-lora")
            
        Returns:
            LoRARequest if the adapter was successfully downloaded and validated,
            None otherwise
        """
        # Check if lora_name is an S3 path
        if not lora_name.startswith("s3://"):
            # Not an S3 path, this resolver can't handle it
            return None
        
        try:
            # Parse S3 path
            bucket, prefix = self._parse_s3_path(lora_name)
            
            # Create a unique cache directory name based on the S3 path
            # Use the last component of the path as the directory name
            cache_name = prefix.rstrip('/').split('/')[-1] or bucket
            local_cache_path = os.path.join(self.lora_cache_dir, cache_name)
            
            # Check if already cached
            adapter_config_path = os.path.join(local_cache_path, "adapter_config.json")
            if os.path.exists(adapter_config_path):
                logger.info(f"Using cached LoRA adapter at {local_cache_path}")
            else:
                # Download from S3
                logger.info(f"Downloading LoRA adapter from {lora_name}")
                success = self._download_s3_directory(bucket, prefix, local_cache_path)
                
                if not success:
                    logger.error(f"Failed to download LoRA adapter from {lora_name}")
                    return None
            
            # Validate the adapter configuration
            if not os.path.exists(adapter_config_path):
                logger.error(
                    f"adapter_config.json not found in downloaded LoRA adapter at {local_cache_path}"
                )
                return None
            
            with open(adapter_config_path) as file:
                adapter_config = json.load(file)
            
            # Validate adapter type
            if adapter_config.get("peft_type") != "LORA":
                logger.error(
                    f"Invalid adapter type: {adapter_config.get('peft_type')}. "
                    "Expected 'LORA'"
                )
                return None
            
            # Optionally validate base model (can be relaxed if needed)
            config_base_model = adapter_config.get("base_model_name_or_path", "")
            if config_base_model and config_base_model != base_model_name:
                logger.warning(
                    f"Base model mismatch: adapter expects '{config_base_model}', "
                    f"but serving '{base_model_name}'. Proceeding anyway."
                )
            
            # Create and return LoRARequest
            lora_request = LoRARequest(
                lora_name=lora_name,  # Use S3 path as name
                lora_int_id=abs(hash(lora_name)),
                lora_path=local_cache_path,
            )
            
            logger.info(f"Successfully resolved LoRA adapter from S3: {lora_name}")
            return lora_request
            
        except Exception as e:
            logger.error(f"Error resolving LoRA from S3 ({lora_name}): {e}")
            return None


def register_s3_resolver():
    """Register the S3 LoRA Resolver with vLLM."""
    
    if not BOTO3_AVAILABLE:
        logger.warning(
            "boto3 is not available. S3 LoRA Resolver will not be registered. "
            "Install boto3 to enable S3 support: pip install boto3"
        )
        return
    
    lora_cache_dir = envs.VLLM_LORA_RESOLVER_CACHE_DIR
    if lora_cache_dir:
        if not os.path.exists(lora_cache_dir):
            try:
                os.makedirs(lora_cache_dir, exist_ok=True)
                logger.info(f"Created LoRA cache directory: {lora_cache_dir}")
            except Exception as e:
                raise ValueError(
                    f"Failed to create VLLM_LORA_RESOLVER_CACHE_DIR: {lora_cache_dir}. "
                    f"Error: {e}"
                )
        
        if not os.path.isdir(lora_cache_dir):
            raise ValueError(
                f"VLLM_LORA_RESOLVER_CACHE_DIR must be a valid directory: {lora_cache_dir}"
            )
        
        s3_resolver = S3LoRAResolver(lora_cache_dir)
        LoRAResolverRegistry.register_resolver("S3 Resolver", s3_resolver)
        logger.info("S3 LoRA Resolver registered successfully")
    else:
        logger.warning(
            "VLLM_LORA_RESOLVER_CACHE_DIR not set. "
            "S3 LoRA Resolver will not be registered."
        )

