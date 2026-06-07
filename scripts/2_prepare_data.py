#!/usr/bin/env python3
"""Prepare ToolACE data for Llama 3.2 Instruct SFT."""

import sys
import math
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.utils import setup_logging, ensure_dir, load_yaml_config, save_json

logger = setup_logging("./logs/prepare_data.log")


def resolve_model_path(config: Dict[str, Any]) -> str:
    cache_dir = Path(config["model"].get("cache_dir", "./models/pretrained"))
    path = cache_dir / "AI-ModelScope" / "Llama-3.2-1B-Instruct"

    if (path / "tokenizer_config.json").exists():
        return str(path)

    return config["model"]["model_name"]


def toolace_to_messages(example: Dict[str, Any]) -> List[Dict[str, str]]:
    """Convert a ToolACE row into Llama chat messages."""
    messages = [{"role": "system", "content": str(example.get("system", "")).strip()}]

    for turn in example.get("conversations", []):
        role = turn.get("from")
        content = str(turn.get("value", "")).strip()
        if not content:
            continue

        if role == "tool":
            role = "ipython"
        if role not in {"user", "assistant", "ipython"}:
            logger.warning("Skipping unknown conversation role: %s", role)
            continue

        messages.append({"role": role, "content": content})

    return messages


def fallback_chat_template(messages: List[Dict[str, str]]) -> str:
    """Small fallback if a tokenizer has no chat_template."""
    chunks = []
    for message in messages:
        chunks.append(
            f"<|start_header_id|>{message['role']}<|end_header_id|>\n\n"
            f"{message['content']}<|eot_id|>"
        )
    return "".join(chunks)


def find_subsequence(sequence: List[int], pattern: List[int], start: int = 0) -> int:
    """Return the first index of pattern in sequence, or -1."""
    if not pattern:
        return -1

    last_start = len(sequence) - len(pattern)
    for index in range(start, last_start + 1):
        if sequence[index:index + len(pattern)] == pattern:
            return index
    return -1


def build_assistant_labels(input_ids: List[int], assistant_header_ids: List[int], eot_ids: List[int]) -> List[int]:
    """Mask everything except assistant responses."""
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


def pick_sample_count(requested: int | None, available: int, split_name: str) -> int:
    """Return a valid sample count for a split."""
    if requested is None or requested <= 0:
        return available

    if requested > available:
        logger.warning(
            "%s requested samples (%s) exceeds available rows (%s); using all rows.",
            split_name,
            requested,
            available,
        )
        return available

    return requested


def percentile(values: List[int], q: float) -> int:
    """Nearest-rank percentile without extra dependencies."""
    if not values:
        return 0

    sorted_values = sorted(values)
    rank = math.ceil((q / 100) * len(sorted_values)) - 1
    rank = max(0, min(rank, len(sorted_values) - 1))
    return sorted_values[rank]


def choose_max_seq_length(lengths: List[int], config: Dict[str, Any]) -> int:
    """Choose max_seq_length from token length stats and config limits."""
    training_length = int(config["training"]["max_seq_length"])
    data_config = config["data"]

    if not data_config.get("auto_max_seq_length", False):
        return training_length

    q = float(data_config.get("max_seq_length_percentile", 95))
    cap = int(data_config.get("max_seq_length_cap", training_length))
    min_length = int(data_config.get("min_max_seq_length", 1024))
    multiple = int(data_config.get("length_round_multiple", 256))

    target = percentile(lengths, q)
    rounded = math.ceil(target / multiple) * multiple
    return max(min_length, min(rounded, cap))


def get_length_stats(lengths: List[int], chosen_max_seq_length: int) -> Dict[str, Any]:
    """Return token length stats before truncation."""
    if not lengths:
        return {}

    over_limit = sum(length > chosen_max_seq_length for length in lengths)
    return {
        "min": min(lengths),
        "p50": percentile(lengths, 50),
        "p90": percentile(lengths, 90),
        "p95": percentile(lengths, 95),
        "p99": percentile(lengths, 99),
        "max": max(lengths),
        "truncated_samples": over_limit,
        "total_samples": len(lengths),
        "truncated_ratio": over_limit / len(lengths),
    }


def log_length_stats(stats: Dict[str, Any], chosen_max_seq_length: int) -> None:
    """Log token length stats before truncation."""
    if not stats:
        logger.warning("No token lengths found.")
        return

    logger.info(
        "Token length stats before truncation: min=%s p50=%s p90=%s p95=%s p99=%s max=%s",
        stats["min"],
        stats["p50"],
        stats["p90"],
        stats["p95"],
        stats["p99"],
        stats["max"],
    )
    logger.info(
        "Using max_seq_length=%s; %s/%s samples (%.2f%%) will be truncated.",
        chosen_max_seq_length,
        stats["truncated_samples"],
        stats["total_samples"],
        stats["truncated_ratio"] * 100,
    )


def prepare_data():
    """Load and prepare training data."""
    logger.info("Starting data preparation...")

    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError:
        logger.error("Required packages not installed. Please run: pip install datasets transformers")
        return False

    try:
        # Load configuration
        sft_config = load_yaml_config("./configs/sft_config.yaml")
        dataset_name = sft_config["data"]["dataset_name"]
        validation_split = sft_config["data"]["validation_split_percentage"] / 100

        logger.info(f"Loading dataset: {dataset_name}")

        # Load dataset from HuggingFace
        dataset = load_dataset(dataset_name)

        # Get the train split
        if "train" in dataset:
            train_data = dataset["train"]
        else:
            train_data = dataset

        logger.info(f"Dataset size: {len(train_data)}")

        shuffled_data = train_data.shuffle(seed=sft_config["seed"])
        split_data = shuffled_data.train_test_split(
            test_size=validation_split,
            seed=sft_config["seed"],
        )

        train_count = pick_sample_count(
            sft_config["data"].get("train_samples"),
            len(split_data["train"]),
            "train",
        )
        validation_count = pick_sample_count(
            sft_config["data"].get("validation_samples"),
            len(split_data["test"]),
            "validation",
        )
        train_data = split_data["train"].select(range(train_count))
        validation_data = split_data["test"].select(range(validation_count))

        logger.info("Selected training samples: %s", len(train_data))
        logger.info("Selected validation samples: %s", len(validation_data))

        # Load tokenizer
        model_path = resolve_model_path(sft_config)
        logger.info(f"Loading tokenizer for: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        assistant_header_ids = tokenizer.encode(
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
            add_special_tokens=False,
        )
        eot_ids = tokenizer.encode("<|eot_id|>", add_special_tokens=False)

        def format_function(examples):
            """Format ToolACE rows as Llama 3.2 chat-template text."""
            texts = []
            for system, conversations in zip(examples["system"], examples["conversations"]):
                messages = toolace_to_messages(
                    {"system": system, "conversations": conversations}
                )
                if tokenizer.chat_template:
                    text = tokenizer.apply_chat_template(messages, tokenize=False)
                else:
                    text = fallback_chat_template(messages)
                texts.append(text)

            return {"text": texts}
        
        def single_turn_format_function(examples):
            """Format ToolACE rows as Llama 3.2 chat-template text."""
            texts = []
            for system, conversations in zip(examples["system"], examples["conversations"]):
                
                # ======= 修改：只保留第一轮对话 =======
                single_turn_convs = []
                user_found = False
                assistant_found = False
                
                for turn in conversations:
                    role = turn.get("role") or turn.get("from")
                    if role == "user" and not user_found:
                        single_turn_convs.append(turn)
                        user_found = True
                    elif role == "assistant" and user_found and not assistant_found:
                        single_turn_convs.append(turn)
                        assistant_found = True
                    
                    # 一旦存齐了第一个 user 和第一个 assistant，就跳出循环
                    if user_found and assistant_found:
                        break
                # ===================================

                messages = toolace_to_messages(
                    {"system": system, "conversations": single_turn_convs}
                )
                if tokenizer.chat_template:
                    text = tokenizer.apply_chat_template(messages, tokenize=False)
                else:
                    text = fallback_chat_template(messages)
                texts.append(text)

            return {"text": texts}

        logger.info("Formatting selected dataset...")
        remove_columns = train_data.column_names
        formatted_train = train_data.map(
            single_turn_format_function,
            batched=True,
            batch_size=32,
            remove_columns=remove_columns,
            num_proc=sft_config["data"]["preprocessing_num_workers"],
        )


        formatted_validation = validation_data.map(
            single_turn_format_function,
            batched=True,
            batch_size=32,
            remove_columns=validation_data.column_names,
            num_proc=sft_config["data"]["preprocessing_num_workers"],
        )

        # ======= 新增：将所有文本保存到本地文件进行查看 =======
        txt_output_path = "./data/processed/all_formatted_texts.txt"
        ensure_dir("./data/processed")
        
        logger.info(f"正在将所有格式化文本导出至 {txt_output_path} ...")
        with open(txt_output_path, "w", encoding="utf-8") as f:
            f.write("==================== 训练集数据 ====================\n\n")
            for idx, text in enumerate(formatted_train["text"]):
                f.write(f"--- 训练集样本 {idx + 1} ---\n")
                f.write(text)
                f.write("\n\n" + "="*50 + "\n\n")
                
            f.write("\n\n==================== 验证集数据 ====================\n\n")
            for idx, text in enumerate(formatted_validation["text"]):
                f.write(f"--- 验证集样本 {idx + 1} ---\n")
                f.write(text)
                f.write("\n\n" + "="*50 + "\n\n")
        logger.info("所有文本保存成功！")
        # ==========================================================

        # ======= 新增：统计多轮对话（Assistant 数量 > 1）的个数 =======
        def count_multi_turn(dataset, name="数据集"):
            multi_turn_count = 0
            for item in dataset:
                convs = item.get("conversations", [])
                
                # 兼容不同数据集的字段名（ToolACE 通常是 'role' 或 'from'）
                assistant_count = sum(
                    1 for turn in convs 
                    if turn.get("role") == "assistant" or turn.get("from") == "assistant"
                )
                
                if assistant_count > 1:
                    multi_turn_count += 1
            
            logger.info(f"{name} 总样本数: {len(dataset)}, 其中多轮对话（Assistant > 1）个数: {multi_turn_count}")
            return multi_turn_count

        logger.info("=== 开始统计多轮对话分布 ===")
        train_multi_count = count_multi_turn(train_data, "训练集")
        val_multi_count = count_multi_turn(validation_data, "验证集")
        logger.info("====================================")
        # =========================================================

        all_texts = list(formatted_train["text"]) + list(formatted_validation["text"])
        all_lengths = [
            len(ids)
            for ids in tokenizer(
                all_texts,
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]
        ]
        max_seq_length = choose_max_seq_length(all_lengths, sft_config)
        length_stats = get_length_stats(all_lengths, max_seq_length)
        log_length_stats(length_stats, max_seq_length)

        def tokenize_function(examples):
            """Tokenize formatted text and mask loss to assistant tokens only."""
            input_ids, attention_mask, labels = [], [], []
            tokenized = tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_seq_length,
                padding="max_length",
            )

            for ids, mask in zip(tokenized["input_ids"], tokenized["attention_mask"]):
                sample_labels = build_assistant_labels(ids, assistant_header_ids, eot_ids)
                sample_labels = [
                    label if token_mask else -100
                    for label, token_mask in zip(sample_labels, mask)
                ]
                input_ids.append(ids)
                attention_mask.append(mask)
                labels.append(sample_labels)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

        logger.info("Tokenizing dataset...")

        processed_train = formatted_train.map(
            tokenize_function,
            batched=True,
            batch_size=32,
            num_proc=sft_config["data"]["preprocessing_num_workers"],
        )
        processed_validation = formatted_validation.map(
            tokenize_function,
            batched=True,
            batch_size=32,
            num_proc=sft_config["data"]["preprocessing_num_workers"],
        )

        # 定义过滤条件：只保留真实 Token 长度（不含 Padding）小于等于 1024 的数据
        # 怎么判断真实长度？看 attention_mask 里面 1 的个数即可
        processed_train = processed_train.filter(
            lambda x: sum(x["attention_mask"]) <= 1024,
            num_proc=sft_config["data"]["preprocessing_num_workers"]
        )
        processed_validation = processed_validation.filter(
            lambda x: sum(x["attention_mask"]) <= 1024,
            num_proc=sft_config["data"]["preprocessing_num_workers"]
        )
        
        logger.info(f"过滤后剩余训练集样本数: {len(processed_train)}")
        logger.info(f"过滤后剩余验证集样本数: {len(processed_validation)}") 


        # Save processed data
        output_dir = "./data/processed"
        ensure_dir(output_dir)

        logger.info(f"Saving processed data to {output_dir}")
        processed_train.save_to_disk(f"{output_dir}/train")
        processed_validation.save_to_disk(f"{output_dir}/validation")
        save_json(
            {
                "dataset_name": dataset_name,
                "train_samples": len(processed_train),
                "validation_samples": len(processed_validation),
                "max_seq_length": max_seq_length,
                "length_stats": length_stats,
                "assistant_only_loss": True,
            },
            f"{output_dir}/metadata.json",
        )

        logger.info(f"Training samples: {len(processed_train)}")
        logger.info(f"Validation samples: {len(processed_validation)}")
        logger.info("Data preparation completed successfully!")

        return True

    except Exception as e:
        logger.error(f"Error preparing data: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = prepare_data()
    sys.exit(0 if success else 1)
