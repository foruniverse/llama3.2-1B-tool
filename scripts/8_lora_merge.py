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

from scripts.utils.utils import get_device_info, setup_logging
from scripts.utils.sglang_compat import normalize_model_dir

logger = setup_logging("./logs/lora_merge.log")


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
        "--merge-dtype",
        choices=["bf16", "fp32"],
        default="bf16",
        help="Dtype used to load the base model before merging LoRA weights.",
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

    logger.info("Starting LoRA merge.")
    logger.info("Device info: %s", get_device_info())
    logger.info("Base model path: %s", base_model_path)
    logger.info("LoRA checkpoint path: %s", lora_path)
    logger.info("Output directory: %s", output_dir)
    logger.info("Device map: %s", args.device_map)
    logger.info("Merge dtype: %s", args.merge_dtype)
    logger.info("Skip SGLang/BFCL compatibility normalization: %s", args.skip_sglang_compat)

    logger.info("Loading tokenizer from %s", base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    logger.info("Loading base model from %s", base_model_path)
    dtype = {
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[args.merge_dtype]
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=True,
    )

    logger.info("Loading LoRA weights from %s", lora_path)
    model = PeftModel.from_pretrained(base_model, lora_path)

    logger.info("Merging LoRA weights.")
    merged_model = model.merge_and_unload()

    logger.info("Saving merged model to %s", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    if not args.skip_sglang_compat:
        changes = normalize_model_dir(output_dir)
        changed = [name for name, did_change in changes.items() if did_change]
        status = ", ".join(changed) if changed else "already compatible"
        logger.info("SGLang/BFCL files normalized: %s", status)

    logger.info("LoRA merge completed. Merged model saved at: %s", output_dir)


if __name__ == "__main__":
    merge_lora(parse_args())
