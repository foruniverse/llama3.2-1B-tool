"""Helpers for making merged Llama checkpoints compatible with SGLang/BFCL."""

import json
from pathlib import Path


def normalize_config(model_dir: Path) -> bool:
    """Rewrite Transformers 5 config keys to fields understood by SGLang stacks."""
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {model_dir}")

    config = json.loads(config_path.read_text())
    changed = False

    rope_parameters = config.pop("rope_parameters", None)
    if rope_parameters is not None:
        rope_theta = rope_parameters.pop("rope_theta", None)
        config["rope_scaling"] = rope_parameters
        if rope_theta is not None:
            config["rope_theta"] = rope_theta
        changed = True

    dtype = config.pop("dtype", None)
    if dtype is not None:
        config["torch_dtype"] = dtype
        changed = True

    if changed:
        config_path.write_text(json.dumps(config, indent=2) + "\n")

    return changed


def normalize_tokenizer_config(model_dir: Path) -> bool:
    """Rewrite tokenizer metadata that BFCL cannot load through AutoTokenizer."""
    tokenizer_config_path = model_dir / "tokenizer_config.json"
    if not tokenizer_config_path.exists():
        raise FileNotFoundError(f"Missing tokenizer_config.json in {model_dir}")

    tokenizer_config = json.loads(tokenizer_config_path.read_text())
    if tokenizer_config.get("tokenizer_class") != "TokenizersBackend":
        return False

    tokenizer_config["tokenizer_class"] = "PreTrainedTokenizerFast"
    tokenizer_config_path.write_text(json.dumps(tokenizer_config, indent=2) + "\n")
    return True


def normalize_model_dir(model_dir: Path) -> dict[str, bool]:
    """Normalize all model files that commonly break older BFCL/SGLang stacks."""
    return {
        "config": normalize_config(model_dir),
        "tokenizer_config": normalize_tokenizer_config(model_dir),
    }
