#!/usr/bin/env python3
"""Shuffle tool-schema parameter order for a random subset of processed data."""

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.utils import ensure_dir, load_json, load_yaml_config, save_json, setup_logging

logger = setup_logging("./logs/shuffle_tool_schema.log")

USER_HEADER = "<|start_header_id|>user<|end_header_id|>"


def resolve_model_path(config: dict[str, Any]) -> str:
    cache_dir = Path(config["model"].get("cache_dir", "./models/pretrained"))
    path = cache_dir / "AI-ModelScope" / "Llama-3.2-1B-Instruct"

    if (path / "tokenizer_config.json").exists():
        return str(path)

    return config["model"]["model_name"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly select processed samples and shuffle tool schema order."
    )
    parser.add_argument("--source", default="./data/processed/train")
    parser.add_argument("--output", default="./data/processed/train_shuffle_tools_20pct")
    parser.add_argument("--ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default="./configs/sft_config.yaml")
    parser.add_argument("--metadata", default="./data/processed/metadata.json")
    parser.add_argument(
        "--num-proc",
        type=int,
        default=1,
        help="Number of processes for tokenization. Use 1 for socket-limited environments.",
    )
    return parser.parse_args()


def force_shuffle(items: list[Any], rng: random.Random) -> list[Any]:
    """Shuffle items, rotating once if randomness returns the original order."""
    original = list(items)
    shuffled = list(items)
    rng.shuffle(shuffled)
    if len(shuffled) > 1 and shuffled == original:
        shuffled = shuffled[1:] + shuffled[:1]
    return shuffled


def shuffled_mapping(mapping: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    items = list(mapping.items())
    items = force_shuffle(items, rng)
    return dict(items)


def shuffle_schema_object(value: Any, rng: random.Random) -> Any:
    """Recursively shuffle JSON object key order and required arrays."""
    if isinstance(value, list):
        shuffled = [shuffle_schema_object(item, rng) for item in value]
        return shuffled

    if not isinstance(value, dict):
        return value

    updated: dict[str, Any] = {}
    for key, item in value.items():
        if key == "required" and isinstance(item, list):
            required = force_shuffle(item, rng)
            updated[key] = required
        elif key == "properties" and isinstance(item, dict):
            updated[key] = shuffled_mapping(
                {name: shuffle_schema_object(schema, rng) for name, schema in item.items()},
                rng,
            )
        else:
            updated[key] = shuffle_schema_object(item, rng)

    return shuffled_mapping(updated, rng)


def shuffle_tools_json(tools_json: str, rng: random.Random) -> str:
    tools = json.loads(tools_json)
    shuffled_tools = [shuffle_schema_object(tool, rng) for tool in tools]
    return json.dumps(shuffled_tools, ensure_ascii=False)


def find_tools_json_span(text: str) -> tuple[int, int] | None:
    """Find the first JSON array that looks like a tool list inside the system prompt."""
    system_text = text.split(USER_HEADER, 1)[0]
    decoder = json.JSONDecoder()

    for match in re.finditer(r"\[", system_text):
        start = match.start()
        try:
            value, end_offset = decoder.raw_decode(system_text[start:])
        except json.JSONDecodeError:
            continue

        if (
            isinstance(value, list)
            and value
            and all(isinstance(tool, dict) for tool in value)
            and any("name" in tool and "parameters" in tool for tool in value)
        ):
            return start, start + end_offset

    return None


def rewrite_tools(text: str, rng: random.Random) -> tuple[str, bool]:
    span = find_tools_json_span(text)
    if span is None:
        return text, False

    start, end = span
    tools_json = text[start:end]
    shuffled_json = shuffle_tools_json(tools_json, rng)
    return text[:start] + shuffled_json + text[end:], True


def find_subsequence(sequence: list[int], pattern: list[int], start: int = 0) -> int:
    if not pattern:
        return -1

    last_start = len(sequence) - len(pattern)
    for index in range(start, last_start + 1):
        if sequence[index : index + len(pattern)] == pattern:
            return index
    return -1


def build_assistant_labels(
    input_ids: list[int],
    assistant_header_ids: list[int],
    eot_ids: list[int],
) -> list[int]:
    labels = [-100] * len(input_ids)
    search_start = 0

    while True:
        header_start = find_subsequence(input_ids, assistant_header_ids, search_start)
        if header_start == -1:
            break

        answer_start = header_start + len(assistant_header_ids)
        answer_end = find_subsequence(input_ids, eot_ids, answer_start)
        if answer_end == -1:
            answer_end = len(input_ids)
        else:
            answer_end += len(eot_ids)

        for index in range(answer_start, answer_end):
            labels[index] = input_ids[index]

        search_start = answer_end

    return labels


def main() -> int:
    args = parse_args()
    if not 0 < args.ratio <= 1:
        raise ValueError("--ratio must be in (0, 1].")

    try:
        from datasets import load_from_disk
        from transformers import AutoTokenizer
    except ImportError as exc:
        logger.error("Required packages are missing: %s", exc)
        return 1

    config = load_yaml_config(args.config)
    metadata = load_json(args.metadata) if Path(args.metadata).exists() else {}

    logger.info("Loading source dataset from %s", args.source)
    dataset = load_from_disk(args.source)
    total = len(dataset)
    sample_count = round(total * args.ratio)

    rng = random.Random(args.seed)
    rewritable_indices = [
        index
        for index, text in enumerate(dataset["text"])
        if find_tools_json_span(text) is not None
    ]
    if sample_count > len(rewritable_indices):
        logger.warning(
            "Requested %s samples but only %s contain rewritable JSON tools; using all rewritable samples.",
            sample_count,
            len(rewritable_indices),
        )
        sample_count = len(rewritable_indices)

    selected_indices = set(rng.sample(rewritable_indices, sample_count))

    logger.info(
        "Selected %s/%s samples for tool-schema shuffling from %s rewritable rows",
        sample_count,
        total,
        len(rewritable_indices),
    )

    rewrite_rng = random.Random(args.seed + 1)
    rewritten = 0
    missing_tools = 0

    def augment_example(example: dict[str, Any], index: int) -> dict[str, Any]:
        nonlocal rewritten, missing_tools

        text = example["text"]
        if index in selected_indices:
            text, ok = rewrite_tools(text, rewrite_rng)
            if ok:
                rewritten += 1
            else:
                missing_tools += 1

        return {"text": text}

    augmented = dataset.map(
        augment_example,
        with_indices=True,
        load_from_cache_file=False,
        desc="Shuffling selected tool schemas",
    )

    model_path = resolve_model_path(config)
    logger.info("Loading tokenizer from %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    assistant_header_ids = tokenizer.encode(
        "<|start_header_id|>assistant<|end_header_id|>\n\n",
        add_special_tokens=False,
    )
    eot_ids = tokenizer.encode("<|eot_id|>", add_special_tokens=False)
    max_seq_length = int(metadata.get("max_seq_length", config["training"]["max_seq_length"]))

    def tokenize_function(examples: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )

        labels = []
        for ids, mask in zip(tokenized["input_ids"], tokenized["attention_mask"]):
            sample_labels = build_assistant_labels(ids, assistant_header_ids, eot_ids)
            labels.append(
                [
                    label if token_mask else -100
                    for label, token_mask in zip(sample_labels, mask)
                ]
            )

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
        }

    logger.info("Retokenizing augmented dataset with max_seq_length=%s", max_seq_length)
    augmented = augmented.map(
        tokenize_function,
        batched=True,
        batch_size=32,
        num_proc=args.num_proc if args.num_proc > 1 else None,
        load_from_cache_file=False,
        desc="Tokenizing augmented data",
    )

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Output path already exists: {output}")

    logger.info("Saving augmented dataset to %s", output)
    augmented.save_to_disk(str(output))

    ensure_dir(str(output.parent))
    save_json(
        {
            "source": args.source,
            "output": args.output,
            "shuffle_ratio": args.ratio,
            "seed": args.seed,
            "num_total": total,
            "num_rewritable": len(rewritable_indices),
            "num_selected": sample_count,
            "num_rewritten": rewritten,
            "num_selected_missing_tools": missing_tools,
            "max_seq_length": max_seq_length,
        },
        str(output.parent / f"{output.name}_metadata.json"),
    )

    logger.info(
        "Done. selected=%s rewritten=%s missing_tools=%s total=%s",
        sample_count,
        rewritten,
        missing_tools,
        total,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
