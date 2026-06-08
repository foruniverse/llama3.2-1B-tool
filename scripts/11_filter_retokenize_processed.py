#!/usr/bin/env python3
"""Filter processed text rows by full token length and retokenize."""

import argparse
import sys
from pathlib import Path
from typing import Any

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.utils import ensure_dir, load_yaml_config, save_json, setup_logging

logger = setup_logging("./logs/filter_retokenize_processed.log")


def resolve_model_path(config: dict[str, Any]) -> str:
    cache_dir = Path(config["model"].get("cache_dir", "./models/pretrained"))
    path = cache_dir / "AI-ModelScope" / "Llama-3.2-1B-Instruct"

    if (path / "tokenizer_config.json").exists():
        return str(path)

    return config["model"]["model_name"]


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter rows whose prompt+assistant token length exceeds max length, then retokenize."
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="./configs/sft_config.yaml")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--num-proc", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        from datasets import load_from_disk
        from transformers import AutoTokenizer
    except ImportError as exc:
        logger.error("Required packages are missing: %s", exc)
        return 1

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Output path already exists: {output}")

    config = load_yaml_config(args.config)
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

    logger.info("Loading source dataset from %s", args.source)
    dataset = load_from_disk(args.source)
    source_count = len(dataset)

    def add_length(examples: dict[str, list[str]]) -> dict[str, list[int]]:
        tokenized = tokenizer(
            examples["text"],
            add_special_tokens=False,
            truncation=False,
        )
        return {"full_token_length": [len(ids) for ids in tokenized["input_ids"]]}

    with_lengths = dataset.map(
        add_length,
        batched=True,
        batch_size=32,
        num_proc=args.num_proc if args.num_proc > 1 else None,
        load_from_cache_file=False,
        desc="Computing full token lengths",
    )

    filtered = with_lengths.filter(
        lambda row: row["full_token_length"] <= args.max_seq_length,
        num_proc=args.num_proc if args.num_proc > 1 else None,
        load_from_cache_file=False,
        desc=f"Filtering rows <= {args.max_seq_length}",
    )
    kept_count = len(filtered)
    removed_count = source_count - kept_count

    remove_columns = [
        column
        for column in ["input_ids", "attention_mask", "labels", "full_token_length"]
        if column in filtered.column_names
    ]
    filtered = filtered.remove_columns(remove_columns)

    def tokenize_function(examples: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_seq_length,
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

    logger.info("Retokenizing %s rows with max_seq_length=%s", kept_count, args.max_seq_length)
    retokenized = filtered.map(
        tokenize_function,
        batched=True,
        batch_size=32,
        num_proc=args.num_proc if args.num_proc > 1 else None,
        load_from_cache_file=False,
        desc="Retokenizing filtered rows",
    )

    logger.info("Saving filtered dataset to %s", output)
    retokenized.save_to_disk(str(output))

    ensure_dir(str(output.parent))
    save_json(
        {
            "source": args.source,
            "output": args.output,
            "max_seq_length": args.max_seq_length,
            "source_samples": source_count,
            "kept_samples": kept_count,
            "removed_samples": removed_count,
            "filter_rule": "full prompt+assistant token length <= max_seq_length",
            "assistant_only_loss": True,
        },
        str(output.parent / f"{output.name}_metadata.json"),
    )

    logger.info(
        "Done. source=%s kept=%s removed=%s max_seq_length=%s",
        source_count,
        kept_count,
        removed_count,
        args.max_seq_length,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
