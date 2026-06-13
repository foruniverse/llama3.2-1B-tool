#!/usr/bin/env python3
"""消融实验 1：扰乱 JSON tools 参数名和参数顺序后生成 SFT 数据。

实验设计：

1. 先固定 train/test split，保证后续所有扰动都发生在固定划分内部。
2. test 集比例由 `data.validation_split_percentage` 固定，例如 5 表示测试集占总量 5%。
3. train 内抽取 20% 可扰动样本做参数扰乱。
4. test 内抽取 50% 可扰动样本做同样扰乱。
4. 扰乱内容：
   - tools 内参数顺序随机打乱。
   - 参数属性顺序随机打乱。
   - required 顺序随机打乱。
   - 带 description 的参数名改为 param_XXXX，并同步替换 assistant 回复里的参数名。
5. 使用手写 Llama chat template 生成 prompt，并构造 assistant-only labels。

不足提醒：

- refusal=1 的自然语言回复不参与扰动，避免替换额外解释文本。
- assistant 参数名只在参数赋值位置替换，避免替换参数值里的普通单词。
- 扰动只处理 tool_type=json、tool_count>=1、refusal=0 且存在带 description 参数的样本。
"""

from __future__ import annotations

import copy
import json
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Any

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.utils import ensure_dir, load_experiment_config, save_json, setup_logging

logger = setup_logging("./logs/prepare_data_ablation1.log")

TRAIN_ABLATION_RATIO = 0.20
TEST_ABLATION_RATIO = 0.50


def resolve_model_path(config: dict[str, Any]) -> str:
    """优先使用本地 tokenizer，避免每次从远端加载。"""
    cache_dir = Path(config["model"].get("cache_dir", "./models/pretrained"))
    local_path = cache_dir / "AI-ModelScope" / "Llama-3.2-1B-Instruct"
    if (local_path / "tokenizer_config.json").exists():
        return str(local_path)
    return config["model"]["model_name"]


def render_system(system: str, tools: list[Any]) -> str:
    """把 system 和 tools 合并为 system message 内容。"""
    tools_text = json.dumps(tools, ensure_ascii=False, indent=2)
    system = system.strip()
    return tools_text if not system else f"{system}\n\n{tools_text}"


def render_prompt(system: str, user: str, assistant: str, tools: list[Any]) -> str:
    """手写 Llama chat template，避免 apply_chat_template 自动注入日期。"""
    system_content = render_system(system, tools)
    return (
        "<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n{system_content}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n{assistant}<|eot_id|>"
    )


def find_subsequence(sequence: list[int], pattern: list[int], start: int = 0) -> int:
    """查找 token 子序列，找不到返回 -1。"""
    if not pattern:
        return -1
    for index in range(start, len(sequence) - len(pattern) + 1):
        if sequence[index:index + len(pattern)] == pattern:
            return index
    return -1


def build_assistant_labels(
    input_ids: list[int],
    attention_mask: list[int],
    assistant_header_ids: list[int],
    eot_ids: list[int],
) -> list[int]:
    """只保留 assistant 回复部分参与 loss，其余 token 全部置为 -100。"""
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

    return [label if mask else -100 for label, mask in zip(labels, attention_mask)]


def load_tools(tools_text: str) -> list[Any]:
    """解析结构化数据集中保存的 tools 字符串。"""
    try:
        value = json.loads(tools_text)
    except json.JSONDecodeError:
        return []
    return value if isinstance(value, list) else []


def json_tool_is_eligible(row: dict[str, Any]) -> bool:
    """只对确实存在可改名参数的非拒绝 JSON tools 样本做扰乱。

    有些工具没有参数，或 schema 缺少 properties。选中这类样本无法产生
    参数顺序/参数名变化，因此不纳入 20%/50% 的抽样基数。
    """
    if row.get("tool_type") != "json":
        return False
    if int(row.get("tool_count", 0)) < 1:
        return False
    if int(row.get("refusal", 1)) != 0:
        return False
    return bool(collect_parameter_names(load_tools(row["tools"])))


def shuffled_dict(items: list[tuple[Any, Any]], rng: random.Random) -> dict[Any, Any]:
    """按随机顺序重建 dict；Python 会保留插入顺序。"""
    items = list(items)
    rng.shuffle(items)
    return dict(items)


def parameter_has_description(schema: Any) -> bool:
    """只有带 description 的参数参与改名。

    description 是参数语义最稳定的来源。没有 description 的参数可能只是
    中间结构、枚举容器或格式占位，本实验不改它，避免引入额外噪声。
    """
    return isinstance(schema, dict) and bool(str(schema.get("description", "")).strip())


def collect_parameter_names_from_schema(value: Any, names: list[str], seen: set[str]) -> None:
    """递归收集所有 tool 里的可改名参数，包括 object 子参数。

    只要某个 dict 含有 `properties`，就按 JSON schema object 处理。
    这能覆盖：
    - 多个 tool 的顶层参数。
    - object / array.items.object 中的子参数。
    - 未出现在 assistant 调用里的 tool 参数。
    """
    if isinstance(value, list):
        for item in value:
            collect_parameter_names_from_schema(item, names, seen)
        return

    if not isinstance(value, dict):
        return

    properties = value.get("properties")
    if isinstance(properties, dict):
        for param_name, param_schema in properties.items():
            name = str(param_name)
            if parameter_has_description(param_schema) and name not in seen:
                names.append(name)
                seen.add(name)
            collect_parameter_names_from_schema(param_schema, names, seen)

    for key, item in value.items():
        if key != "properties":
            collect_parameter_names_from_schema(item, names, seen)


def collect_parameter_names(tools: list[Any]) -> list[str]:
    """收集所有 JSON tools 中带 description 的参数名。"""
    names: list[str] = []
    seen = set()
    collect_parameter_names_from_schema(tools, names, seen)
    return names


def make_parameter_mapping(names: list[str], rng: random.Random) -> dict[str, str]:
    """为每个参数生成稳定但随机的新名字。"""
    mapping: dict[str, str] = {}
    used = set(names)
    for name in names:
        while True:
            new_name = f"param_{rng.randint(0, 9999):04d}"
            if new_name not in used:
                break
        mapping[name] = new_name
        used.add(new_name)
    return mapping


def rewrite_schema(value: Any, mapping: dict[str, str], rng: random.Random) -> Any:
    """递归打乱 schema 字段顺序，并替换参数名。

    关键点：
    - 任意层级的 `properties` 都会处理，因此 nested 子参数也会改名。
    - 只有当前参数 schema 带 description 时才改名。
    - `required` 只同步当前 object 内已经改名的参数，并打乱顺序。
    - 普通 list 保持原顺序，避免 enum 等值列表被无关扰动。
    """
    if isinstance(value, list):
        return [rewrite_schema(item, mapping, rng) for item in value]

    if not isinstance(value, dict):
        return value

    local_mapping: dict[str, str] = {}
    properties = value.get("properties")
    if isinstance(properties, dict):
        for param_name, param_schema in properties.items():
            name = str(param_name)
            if parameter_has_description(param_schema) and name in mapping:
                local_mapping[name] = mapping[name]

    rewritten_items = []
    for key, item in value.items():
        if key == "properties" and isinstance(item, dict):
            properties = [
                (local_mapping.get(str(param_name), str(param_name)), rewrite_schema(param_schema, mapping, rng))
                for param_name, param_schema in item.items()
            ]
            rewritten_items.append((key, shuffled_dict(properties, rng)))
            continue

        if key == "required" and isinstance(item, list):
            required = [local_mapping.get(str(param_name), str(param_name)) for param_name in item]
            rng.shuffle(required)
            rewritten_items.append((key, required))
            continue

        rewritten_items.append((key, rewrite_schema(item, mapping, rng)))

    return shuffled_dict(rewritten_items, rng)


def rewrite_assistant_parameters(assistant: str, mapping: dict[str, str]) -> str:
    """同步替换 assistant 回复中的参数名。

    只替换“参数名位置”，不替换参数值里的普通单词。
    例如 `data="x"` 会替换为 `param_1234="x"`，
    但 `"customer location data"` 里的 data 不会被替换。
    """
    rewritten = assistant
    for old_name, new_name in sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True):
        escaped = re.escape(old_name)

        # 处理带引号的参数名：{"old": value}、'old'=value、"old"|value 等。
        quoted_pattern = re.compile(
            rf"(?P<quote>['\"]){escaped}(?P=quote)(?=\s*(?:==|=|:|/|\||~))"
        )
        rewritten = quoted_pattern.sub(lambda match: f"{match.group('quote')}{new_name}{match.group('quote')}", rewritten)

        # 处理普通参数名：old=value、old:value、old|value、old~value 等。
        bare_pattern = re.compile(
            rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])(?=\s*(?:==|=|:|/|\||~))"
        )
        rewritten = bare_pattern.sub(new_name, rewritten)
    return rewritten


def ablate_json_tools(row: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """对单条 JSON tools 样本执行参数扰乱。"""
    tools = load_tools(row["tools"])
    mapping = make_parameter_mapping(collect_parameter_names(tools), rng)
    if not mapping:
        return {
            "tools": row["tools"],
            "assistant": row["assistant"],
            "ablation_applied": False,
            "parameter_mapping": "{}",
        }

    rewritten_tools = rewrite_schema(copy.deepcopy(tools), mapping, rng)
    rewritten_assistant = rewrite_assistant_parameters(str(row["assistant"]), mapping)
    return {
        "tools": json.dumps(rewritten_tools, ensure_ascii=False),
        "assistant": rewritten_assistant,
        "ablation_applied": True,
        "parameter_mapping": json.dumps(mapping, ensure_ascii=False),
    }


def choose_ablation_indices(dataset: Any, ratio: float, rng: random.Random) -> set[int]:
    """在固定 split 内抽取指定比例的 JSON tools 样本。"""
    eligible_indices = [
        index
        for index, row in enumerate(dataset)
        if json_tool_is_eligible(row)
    ]
    rng.shuffle(eligible_indices)
    count = int(len(eligible_indices) * ratio)
    return set(eligible_indices[:count])


def apply_ablation(dataset: Any, selected_indices: set[int], seed: int) -> Any:
    """对选中样本应用扰乱，未选中样本保持原样。"""
    def map_batch(examples: dict[str, list[Any]], indices: list[int]) -> dict[str, list[Any]]:
        output = {
            "system": [],
            "user": [],
            "assistant": [],
            "tools": [],
            "tool_count": [],
            "tool_type": [],
            "refusal": [],
            "ablation_applied": [],
            "parameter_mapping": [],
        }

        for offset, index in enumerate(indices):
            row = {key: examples[key][offset] for key in examples}
            if index in selected_indices:
                # 每条样本使用独立随机源，保证 map 批大小变化不影响结果。
                rng = random.Random(seed + index * 1009)
                changed = ablate_json_tools(row, rng)
                row.update(changed)
            else:
                row["ablation_applied"] = False
                row["parameter_mapping"] = "{}"

            for key in output:
                output[key].append(row[key])

        return output

    return dataset.map(
        map_batch,
        batched=True,
        with_indices=True,
        batch_size=128,
        load_from_cache_file=False,
    )


def add_prompt_column(dataset: Any) -> Any:
    """根据当前 tools/assistant 重新组装 prompt。"""
    def map_batch(examples: dict[str, list[Any]]) -> dict[str, list[str]]:
        prompts = []
        for system, user, assistant, tools_text in zip(
            examples["system"],
            examples["user"],
            examples["assistant"],
            examples["tools"],
        ):
            prompts.append(render_prompt(system, user, assistant, load_tools(tools_text)))
        return {"text": prompts}

    return dataset.map(map_batch, batched=True, batch_size=128, load_from_cache_file=False)


def tokenize_for_sft(
    dataset: Any,
    tokenizer: Any,
    max_seq_length: int,
    assistant_header_ids: list[int],
    eot_ids: list[int],
    num_proc: int,
) -> Any:
    """tokenize 并构造 assistant-only labels。"""
    map_kwargs = {"num_proc": num_proc} if num_proc and num_proc > 1 else {}

    def tokenize_batch(examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )
        labels = [
            build_assistant_labels(ids, mask, assistant_header_ids, eot_ids)
            for ids, mask in zip(tokenized["input_ids"], tokenized["attention_mask"])
        ]
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
        }

    tokenized_dataset = dataset.map(
        tokenize_batch,
        batched=True,
        batch_size=32,
        load_from_cache_file=False,
        **map_kwargs,
    )
    keep_columns = {"input_ids", "attention_mask", "labels"}
    drop_columns = [column for column in tokenized_dataset.column_names if column not in keep_columns]
    return tokenized_dataset.remove_columns(drop_columns)


def save_dataset(dataset: Any, path: str) -> None:
    """覆盖保存 datasets 格式数据。"""
    output_path = Path(path)
    if output_path.exists():
        shutil.rmtree(output_path)
    dataset.save_to_disk(path)


def count_applied(dataset: Any) -> int:
    """统计实际扰乱成功的样本数。"""
    return sum(1 for value in dataset["ablation_applied"] if value)


def write_preview(dataset: Any, path: str, count: int = 5) -> None:
    """保存少量扰乱样本，方便人工确认参数名是否同步变化。"""
    ensure_dir(str(Path(path).parent))
    applied = dataset.filter(lambda row: bool(row["ablation_applied"]), load_from_cache_file=False)
    with open(path, "w", encoding="utf-8") as file:
        for index, row in enumerate(applied.select(range(min(count, len(applied)))), start=1):
            file.write(f"==================== Sample {index} ====================\n")
            file.write(f"mapping: {row['parameter_mapping']}\n")
            file.write("-------------------- tools --------------------\n")
            file.write(row["tools"])
            file.write("\n-------------------- assistant --------------------\n")
            file.write(row["assistant"])
            file.write("\n-------------------- prompt --------------------\n")
            file.write(row["text"])
            file.write("\n\n")


def write_corner_case_examples(path: str) -> None:
    """输出手工构造的边界样例，方便人工检查扰动规则。

    这个文件不参与训练，只验证实现是否符合实验定义：
    - 多个 tool 都会处理。
    - assistant 没调用到的 tool 也会处理。
    - nested object 子参数会处理。
    - 没有 description 的参数不会改名。
    """
    tools = [
        {
            "name": "book_hotel",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "Hotel city."},
                    "guest": {
                        "type": "object",
                        "description": "Guest profile.",
                        "properties": {
                            "name": {"type": "string", "description": "Guest name."},
                            "vip": {"type": "boolean"},
                        },
                        "required": ["name", "vip"],
                    },
                    "note": {"type": "string"},
                },
                "required": ["city", "guest", "note"],
            },
        },
        {
            "name": "send_coupon",
            "parameters": {
                "type": "object",
                "properties": {
                    "coupon_id": {"type": "string", "description": "Coupon id."},
                    "channel": {"type": "string", "description": "Delivery channel."},
                },
                "required": ["coupon_id", "channel"],
            },
        },
    ]
    row = {
        "tools": json.dumps(tools, ensure_ascii=False),
        "assistant": 'book_hotel(city="Paris", guest={"name": "Ada", "vip": true}, note="keep data")',
    }
    changed = ablate_json_tools(row, random.Random(7))

    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as file:
        file.write("# Ablation1 参数扰动边界样例\n\n")
        file.write("## 原始 tools\n\n")
        file.write(json.dumps(tools, ensure_ascii=False, indent=2))
        file.write("\n\n## 参数映射\n\n")
        file.write(changed["parameter_mapping"])
        file.write("\n\n## 扰动后 tools\n\n")
        file.write(json.dumps(json.loads(changed["tools"]), ensure_ascii=False, indent=2))
        file.write("\n\n## 原始 assistant\n\n")
        file.write(row["assistant"])
        file.write("\n\n## 扰动后 assistant\n\n")
        file.write(changed["assistant"])
        file.write("\n\n## 检查点\n\n")
        file.write("- `city`、`guest`、`name` 会改名，因为它们有 description。\n")
        file.write("- `coupon_id`、`channel` 会改名，即使 `send_coupon` 没出现在 assistant 里。\n")
        file.write("- `vip`、`note` 不会改名，因为它们没有 description。\n")
        file.write("- `note=\"keep data\"` 中的普通值文本不会被替换。\n")


def prepare_ablation_data() -> bool:
    """主入口：固定 split，执行参数扰乱，并保存 tokenized SFT 数据。"""
    logger.info("Starting ablation1 data preparation.")

    try:
        from datasets import load_from_disk
        from transformers import AutoTokenizer
    except ImportError as error:
        logger.error("Required packages are missing: %s", error)
        return False

    try:
        config_path = "./configs/sft_config.yaml"
        config = load_experiment_config(config_path)
        data_config = config["data"]
        seed = int(config.get("seed", 42))
        max_seq_length = int(config["training"]["max_seq_length"])
        num_proc = int(data_config.get("preprocessing_num_workers", 1))
        structured_path = data_config["structured_dataset_path"]
        train_path = data_config["processed_train_path"]
        test_path = data_config["processed_validation_path"]
        test_size = float(data_config["validation_split_percentage"]) / 100

        logger.info("Config path: %s", config_path)
        logger.info("Structured dataset path: %s", structured_path)
        logger.info("Train output path: %s", train_path)
        logger.info("Test output path: %s", test_path)

        dataset = load_from_disk(structured_path)
        logger.info("Loaded structured samples: %s", len(dataset))

        # 先固定 split，再在 split 内做扰动。这是本实验最重要的边界。
        split = dataset.shuffle(seed=seed).train_test_split(test_size=test_size, seed=seed)
        train_dataset = split["train"]
        test_dataset = split["test"]
        logger.info("Fixed train/test split: train=%s test=%s", len(train_dataset), len(test_dataset))

        train_rng = random.Random(seed + 101)
        test_rng = random.Random(seed + 202)
        train_selected = choose_ablation_indices(train_dataset, TRAIN_ABLATION_RATIO, train_rng)
        test_selected = choose_ablation_indices(test_dataset, TEST_ABLATION_RATIO, test_rng)
        logger.info("Selected train ablation samples: %s", len(train_selected))
        logger.info("Selected test ablation samples: %s", len(test_selected))

        train_dataset = add_prompt_column(apply_ablation(train_dataset, train_selected, seed + 1000))
        test_dataset = add_prompt_column(apply_ablation(test_dataset, test_selected, seed + 2000))
        logger.info("Applied train ablation samples: %s", count_applied(train_dataset))
        logger.info("Applied test ablation samples: %s", count_applied(test_dataset))

        model_path = resolve_model_path(config)
        logger.info("Loading tokenizer: %s", model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        assistant_header_ids = tokenizer.encode(
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
            add_special_tokens=False,
        )
        eot_ids = tokenizer.encode("<|eot_id|>", add_special_tokens=False)

        processed_train = tokenize_for_sft(
            train_dataset,
            tokenizer,
            max_seq_length,
            assistant_header_ids,
            eot_ids,
            num_proc,
        )
        processed_test = tokenize_for_sft(
            test_dataset,
            tokenizer,
            max_seq_length,
            assistant_header_ids,
            eot_ids,
            num_proc,
        )

        save_dataset(processed_train, train_path)
        save_dataset(processed_test, test_path)

        report_dir = Path("./data/ablation1")
        ensure_dir(str(report_dir))
        preview_path = str(report_dir / "parameter_ablation_preview.txt")
        corner_case_path = str(report_dir / "parameter_ablation_corner_cases.md")
        report_path = str(report_dir / "metadata.json")
        write_preview(train_dataset, preview_path)
        write_corner_case_examples(corner_case_path)

        train_eligible_count = sum(1 for row in train_dataset if json_tool_is_eligible(row))
        test_eligible_count = sum(1 for row in test_dataset if json_tool_is_eligible(row))

        metadata = {
            "experiment": "ablation1_parameter_shuffle_and_rename",
            "structured_dataset_path": structured_path,
            "train_output_path": train_path,
            "test_output_path": test_path,
            "seed": seed,
            "max_seq_length": max_seq_length,
            "train_samples": len(processed_train),
            "test_samples": len(processed_test),
            "train_ablation_ratio": TRAIN_ABLATION_RATIO,
            "test_ablation_ratio": TEST_ABLATION_RATIO,
            "train_eligible_count": train_eligible_count,
            "test_eligible_count": test_eligible_count,
            "train_selected_count": len(train_selected),
            "test_selected_count": len(test_selected),
            "train_applied_count": count_applied(train_dataset),
            "test_applied_count": count_applied(test_dataset),
            "preview_path": preview_path,
            "corner_case_path": corner_case_path,
            "notes": [
                "split 先固定，扰动后发生在 train/test 内部。",
                "测试集占总数据比例由 data.validation_split_percentage 控制。",
                "测试集内部扰动比例保持 50%。",
                "只扰动 tool_type=json、tool_count>=1、refusal=0 且存在带 description 参数的样本。",
                "assistant 参数名只在参数赋值位置替换，避免替换参数值或 refusal 自然语言解释。",
                "多个 tool 和 nested 子参数都会扰动；没有 description 的参数不扰动。",
            ],
        }
        save_json(metadata, report_path)

        logger.info("Saved train dataset: %s", train_path)
        logger.info("Saved test dataset: %s", test_path)
        logger.info("Saved metadata: %s", report_path)
        logger.info("Saved preview: %s", preview_path)
        logger.info("Saved corner case examples: %s", corner_case_path)
        logger.info("Ablation1 data preparation completed.")
        return True

    except Exception as error:
        logger.error("Error preparing ablation1 data: %s", error)
        import traceback

        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    sys.exit(0 if prepare_ablation_data() else 1)
