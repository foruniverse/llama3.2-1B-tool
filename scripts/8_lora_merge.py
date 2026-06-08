#!/usr/bin/env python3
"""Merge a LoRA checkpoint and prepare the merged model for BFCL/SGLang."""

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.sglang_compat import normalize_model_dir


def default_base_model_path() -> Path:
    candidates = [
        project_root / "models" / "pretrained" / "AI-ModelScope" / "Llama-3___2-1B-Instruct",
        project_root / "models" / "pretrained" / "AI-ModelScope" / "Llama-3.2-1B-Instruct",
    ]

    for candidate in candidates:
        if (candidate / "config.json").exists():
            return candidate

    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-model-path",
        type=Path,
        default=default_base_model_path(),
        help="Base model directory.",
    )
    parser.add_argument(
        "--lora-path",
        type=Path,
        default=project_root / "models" / "checkpoints" / "sft" / "checkpoint-555",
        help="LoRA adapter checkpoint directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "models" / "sft-555",
        help="Merged model output directory.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help='Device map passed to from_pretrained, for example "cpu" or "auto".',
    )
    parser.add_argument(
        "--skip-sglang-compat",
        action="store_true",
        help="Do not normalize config.json for SGLang/BFCL compatibility.",
    )
    return parser.parse_args()


def merge_lora(args: argparse.Namespace) -> None:
    base_model_path = args.base_model_path
    lora_path = args.lora_path
    output_dir = args.output_dir
    
    print(f"Loading tokenizer from {base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    print(f"Loading base model from {base_model_path}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=torch.bfloat16,
        device_map=args.device_map,
        trust_remote_code=True,
    )
    
    print(f"Loading LoRA weights from {lora_path}...")
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    print("Merging LoRA weights...")
    merged_model = model.merge_and_unload()
    
    print(f"Saving merged model to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    if not args.skip_sglang_compat:
        changes = normalize_model_dir(output_dir)
        changed = [name for name, did_change in changes.items() if did_change]
        status = ", ".join(changed) if changed else "already compatible"
        print(f"SGLang/BFCL files normalized: {status}")

    print("Done! The merged model is saved at:", output_dir)


if __name__ == "__main__":
    merge_lora(parse_args())

# uv run ./scripts/8_lora_merge.py --lora-path ./models/checkpoints/sft/checkpoint-360 --output-dir ./models/sft-360
