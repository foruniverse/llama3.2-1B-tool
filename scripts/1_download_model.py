#!/usr/bin/env python3
"""Download Llama3.2-1B model from ModelScope."""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.utils import setup_logging, ensure_dir, get_device_info

logger = setup_logging("./logs/download_model.log")


def download_model():
    """Download model from ModelScope."""
    logger.info("Starting model download...")
    logger.info(f"Device info: {get_device_info()}")

    try:
        from modelscope import snapshot_download
    except ImportError:
        logger.error("modelscope not installed. Please run: pip install modelscope")
        return False

    # Model configuration
    model_id = "AI-ModelScope/Llama-3.2-1B-Instruct"  # ModelScope official model
    cache_dir = "./models/pretrained"

    logger.info(f"Model ID: {model_id}")
    logger.info(f"Cache directory: {cache_dir}")

    try:
        # Create cache directory
        ensure_dir(cache_dir)

        # Download model
        logger.info("Downloading model (this may take 5-15 minutes)...")
        model_dir = snapshot_download(
            model_id,
            cache_dir=cache_dir,
            revision="master"
        )

        logger.info(f"Model downloaded successfully to: {model_dir}")

        # Verify model files
        required_files = ["config.json", "model.safetensors", "tokenizer.model", "tokenizer.json"]
        model_path = Path(model_dir)

        for file in required_files:
            file_path = model_path / file
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                logger.info(f"✓ {file}: {size_mb:.2f} MB")
            else:
                logger.warning(f"✗ {file}: not found")

        logger.info("Model download completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        return False


if __name__ == "__main__":
    success = download_model()
    sys.exit(0 if success else 1)
