"""Utility functions for model training and evaluation."""

import os
import json
import logging
from pathlib import Path
import copy
from typing import Optional, Dict, Any
import yaml


def setup_logging(log_path: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        absolute_log_path = str(Path(log_path).resolve())
        has_file_handler = any(
            isinstance(handler, logging.FileHandler)
            and Path(handler.baseFilename).resolve() == Path(absolute_log_path)
            for handler in logger.handlers
        )
        if not has_file_handler:
            fh = logging.FileHandler(log_path)
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge experiment overrides into a config copy."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def apply_active_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return config with `experiments.active` profile applied.

    The top-level config remains the shared default. A profile under
    `experiments.profiles.<name>` can override nested sections such as
    `data.processed_train_path` or `training.output_dir`.
    """
    experiments = config.get("experiments") or {}
    active = experiments.get("active")
    profiles = experiments.get("profiles") or {}
    if not active:
        return config
    if active not in profiles:
        raise KeyError(f"Active experiment profile not found: {active}")

    profile = profiles[active] or {}
    merged = deep_merge_dict(config, profile)
    merged.setdefault("experiments", {})
    merged["experiments"]["active"] = active
    return merged


def load_experiment_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config and apply the active experiment profile."""
    return apply_active_experiment(load_yaml_config(config_path))


def ensure_dir(path: str) -> None:
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)


def save_json(data: Any, path: str) -> None:
    """Save data as JSON."""
    ensure_dir(os.path.dirname(path))
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> Any:
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def get_device_info() -> Dict[str, Any]:
    """Get device information."""
    try:
        import torch
        return {
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except ImportError:
        return {"cuda_available": False, "device_count": 0}
