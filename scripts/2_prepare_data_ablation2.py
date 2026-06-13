#!/usr/bin/env python3
"""消融实验 2：混合工具名/参数名扰动，并替换 refusal 回复来源。

实验设计：

1. 先构造未扰动全集：
   a. `data/tool_ace_processed` 中 `tool_count>=1`、`tool_type=json`、
      `refusal=0` 的非 refusal 样本。
   b. DeepSeek teacher refusal 样本，替换原 `refusal=1`、`tool_count>0`
      部分，只使用顶层 `description` 作为 assistant response。
   c. `data/tool_ace_processed` 中 `refusal=1` 且 `tool_count=0` 的样本。
2. 在未扰动全集上固定抽取 5% 作为 eval。
3. 在 train 内的 a 类可扰动样本中固定抽取 30%：
   - 15% 扰乱参数名。
   - 10% 扰乱函数名。
   - 5% 同时扰乱参数名和函数名。
4. 在 eval 内的 a 类可扰动样本中固定抽取 50%，同时扰乱参数名和函数名。
5. 所有扰动样本全部打乱 tool schema 字段顺序、参数顺序和 required 顺序。
6. train/eval 分别使用手写 Llama chat template 生成 prompt，
   并构造 assistant-only labels。
"""

from __future__ import annotations

import copy
import importlib.util
import json
import logging
import random
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.utils import ensure_dir, load_experiment_config, save_json, setup_logging

logger = setup_logging("./logs/prepare_data_ablation2.log")

PARAMETER_ONLY_RATIO = 0.15
FUNCTION_ONLY_RATIO = 0.10
PARAMETER_AND_FUNCTION_RATIO = 0.05
EVAL_PARAMETER_AND_FUNCTION_RATIO = 0.50
ABLATION_MODES = ("parameter_name", "function_name", "parameter_and_function")


def load_ablation1_helpers() -> Any:
    """复用 ablation1 里稳定的 prompt、label 和参数 schema 改写逻辑。"""
    path = project_root / "scripts" / "2_prepare_data_ablation1.py"
    spec = importlib.util.spec_from_file_location("ablation1_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load helper script: {path}")
    module = importlib.util.module_from_spec(spec)
    utility_logger = logging.getLogger("scripts.utils.utils")
    existing_handlers = list(utility_logger.handlers)
    spec.loader.exec_module(module)
    for handler in list(utility_logger.handlers):
        if handler not in existing_handlers:
            utility_logger.removeHandler(handler)
            handler.close()
    return module


def tool_name(tool: Any) -> str | None:
    """从 ToolACE/OpenAI 两类 JSON tool 结构中读取函数名。"""
    if not isinstance(tool, dict):
        return None
    name = tool.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    function = tool.get("function")
    if isinstance(function, dict) and isinstance(function.get("name"), str):
        return function["name"].strip()
    return None


def collect_function_names(tools: list[Any]) -> list[str]:
    """按出现顺序收集 tool/function 名称。"""
    names: list[str] = []
    seen = set()
    for tool in tools:
        name = tool_name(tool)
        if name and name not in seen:
            names.append(name)
            seen.add(name)
    return names


def make_function_mapping(names: list[str], rng: random.Random) -> dict[str, str]:
    """为函数名生成稳定随机新名。"""
    mapping: dict[str, str] = {}
    used = set(names)
    for name in names:
        while True:
            new_name = f"func_{rng.randint(0, 9999):04d}"
            if new_name not in used:
                break
        mapping[name] = new_name
        used.add(new_name)
    return mapping


def rewrite_function_names(value: Any, mapping: dict[str, str]) -> Any:
    """递归替换 tools 里的函数名字段。"""
    if isinstance(value, list):
        return [rewrite_function_names(item, mapping) for item in value]
    if not isinstance(value, dict):
        return value

    rewritten: dict[str, Any] = {}
    for key, item in value.items():
        if key == "name" and isinstance(item, str) and item in mapping:
            rewritten[key] = mapping[item]
            continue
        if key == "function" and isinstance(item, dict):
            function = rewrite_function_names(item, mapping)
            if isinstance(function, dict) and isinstance(function.get("name"), str):
                function["name"] = mapping.get(function["name"], function["name"])
            rewritten[key] = function
            continue
        rewritten[key] = rewrite_function_names(item, mapping)
    return rewritten


def rewrite_assistant_function_names(assistant: str, mapping: dict[str, str]) -> str:
    """同步替换 assistant function-call 里的函数名前缀。"""
    rewritten = assistant
    for old_name, new_name in sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True):
        escaped = re.escape(old_name)
        pattern = re.compile(rf"(?<![A-Za-z0-9_.]){escaped}(?=\s*\()")
        rewritten = pattern.sub(new_name, rewritten)
    return rewritten


def non_refusal_is_eligible(row: dict[str, Any], helpers: Any) -> bool:
    """判断非 refusal 样本是否能参与三类扰动。"""
    if int(row.get("refusal", 1)) != 0:
        return False
    if int(row.get("tool_count", 0)) < 1:
        return False
    if row.get("tool_type") != "json":
        return False
    tools = helpers.load_tools(row["tools"])
    return bool(helpers.collect_parameter_names(tools)) and bool(collect_function_names(tools))


def select_mode_indices(dataset: Any, seed: int, helpers: Any) -> dict[int, str]:
    """按 15%/10%/5% 从 train 内可扰动非 refusal 样本中分配扰动模式。"""
    eligible = [index for index, row in enumerate(dataset) if non_refusal_is_eligible(row, helpers)]
    rng = random.Random(seed + 2202)
    rng.shuffle(eligible)

    parameter_count = int(len(eligible) * PARAMETER_ONLY_RATIO)
    function_count = int(len(eligible) * FUNCTION_ONLY_RATIO)
    both_count = int(len(eligible) * PARAMETER_AND_FUNCTION_RATIO)

    selected: dict[int, str] = {}
    cursor = 0
    for index in eligible[cursor:cursor + parameter_count]:
        selected[index] = "parameter_name"
    cursor += parameter_count
    for index in eligible[cursor:cursor + function_count]:
        selected[index] = "function_name"
    cursor += function_count
    for index in eligible[cursor:cursor + both_count]:
        selected[index] = "parameter_and_function"
    return selected


def select_eval_mode_indices(dataset: Any, seed: int, helpers: Any) -> dict[int, str]:
    """在 eval 内抽取 50% 可扰动非 refusal 样本，同时扰乱参数名和函数名。"""
    eligible = [index for index, row in enumerate(dataset) if non_refusal_is_eligible(row, helpers)]
    rng = random.Random(seed + 3303)
    rng.shuffle(eligible)
    count = int(len(eligible) * EVAL_PARAMETER_AND_FUNCTION_RATIO)
    return {index: "parameter_and_function" for index in eligible[:count]}


def ablate_non_refusal_row(row: dict[str, Any], mode: str, rng: random.Random, helpers: Any) -> dict[str, Any]:
    """按模式扰乱 tools，并同步修改 assistant function call。"""
    tools = copy.deepcopy(helpers.load_tools(row["tools"]))
    assistant = str(row["assistant"])
    parameter_mapping: dict[str, str] = {}
    function_mapping: dict[str, str] = {}

    if mode in {"parameter_name", "parameter_and_function"}:
        parameter_mapping = helpers.make_parameter_mapping(helpers.collect_parameter_names(tools), rng)

    # rewrite_schema 即使 mapping 为空也会重排 properties、required 和 schema 字段。
    tools = helpers.rewrite_schema(tools, parameter_mapping, rng)
    if parameter_mapping:
        assistant = helpers.rewrite_assistant_parameters(assistant, parameter_mapping)

    if mode in {"function_name", "parameter_and_function"}:
        function_mapping = make_function_mapping(collect_function_names(tools), rng)
        tools = rewrite_function_names(tools, function_mapping)
        assistant = rewrite_assistant_function_names(assistant, function_mapping)

    return {
        "assistant": assistant,
        "tools": json.dumps(tools, ensure_ascii=False),
        "ablation_mode": mode,
        "parameter_mapping": json.dumps(parameter_mapping, ensure_ascii=False),
        "function_mapping": json.dumps(function_mapping, ensure_ascii=False),
        "source_dataset": "toolace_non_refusal",
    }


def apply_non_refusal_ablation(dataset: Any, selected: dict[int, str], seed: int, helpers: Any) -> Any:
    """对非 refusal 数据添加 ablation 元数据并应用扰动。"""
    def map_batch(examples: dict[str, list[Any]], indices: list[int]) -> dict[str, list[Any]]:
        output = {key: [] for key in [
            "system",
            "user",
            "assistant",
            "tools",
            "tool_count",
            "tool_type",
            "refusal",
            "ablation_mode",
            "parameter_mapping",
            "function_mapping",
            "source_dataset",
        ]}

        for offset, index in enumerate(indices):
            row = {key: examples[key][offset] for key in examples}
            mode = selected.get(index)
            if mode:
                rng = random.Random(seed + index * 1009)
                row.update(ablate_non_refusal_row(row, mode, rng, helpers))
            else:
                row.update(
                    {
                        "ablation_mode": row.get("ablation_mode", "none"),
                        "parameter_mapping": row.get("parameter_mapping", "{}"),
                        "function_mapping": row.get("function_mapping", "{}"),
                        "source_dataset": row.get("source_dataset", "toolace_non_refusal"),
                    }
                )

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


def normalize_non_refusal_dataset(dataset: Any) -> Any:
    """把 ToolACE 非 refusal JSON tool 样本整理成统一字段，暂不扰动。"""
    def map_batch(examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        output = {key: [] for key in output_columns()}
        for offset in range(len(examples["user"])):
            output["system"].append(examples["system"][offset])
            output["user"].append(examples["user"][offset])
            output["assistant"].append(examples["assistant"][offset])
            output["tools"].append(examples["tools"][offset])
            output["tool_count"].append(examples["tool_count"][offset])
            output["tool_type"].append(examples["tool_type"][offset])
            output["refusal"].append(0)
            output["ablation_mode"].append("none")
            output["parameter_mapping"].append("{}")
            output["function_mapping"].append("{}")
            output["source_dataset"].append("toolace_non_refusal")
        return output

    normalized = dataset.map(map_batch, batched=True, batch_size=128, load_from_cache_file=False)
    drop_columns = [column for column in normalized.column_names if column not in output_columns()]
    return normalized.remove_columns(drop_columns)


def teacher_description(row: dict[str, Any]) -> str:
    """读取 DeepSeek teacher 顶层 description。"""
    value = row.get("teacher_description")
    if isinstance(value, str) and value.strip():
        return value.strip()

    teacher_json = row.get("teacher_json")
    if isinstance(teacher_json, dict):
        description = teacher_json.get("description")
        if isinstance(description, str) and description.strip():
            return description.strip()

    assistant = row.get("assistant")
    if isinstance(assistant, str):
        try:
            parsed = json.loads(assistant)
        except json.JSONDecodeError:
            return assistant.strip()
        description = parsed.get("description") if isinstance(parsed, dict) else None
        if isinstance(description, str) and description.strip():
            return description.strip()
    return ""


def normalize_refusal_teacher_dataset(dataset: Any) -> Any:
    """把 teacher refusal 数据转换为 SFT 统一字段，只保留 description。"""
    keep_columns = ["system", "user", "assistant", "tools", "tool_count", "tool_type", "refusal"]

    def map_batch(examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        output = {key: [] for key in [
            *keep_columns,
            "ablation_mode",
            "parameter_mapping",
            "function_mapping",
            "source_dataset",
        ]}
        for offset in range(len(examples["user"])):
            row = {key: examples[key][offset] for key in examples}
            output["system"].append(row.get("system", ""))
            output["user"].append(row["user"])
            output["assistant"].append(teacher_description(row))
            output["tools"].append(row["tools"])
            output["tool_count"].append(row["tool_count"])
            output["tool_type"].append(row["tool_type"])
            output["refusal"].append(1)
            output["ablation_mode"].append("teacher_refusal_description")
            output["parameter_mapping"].append("{}")
            output["function_mapping"].append("{}")
            output["source_dataset"].append("deepseek_refusal_teacher")
        return output

    normalized = dataset.map(map_batch, batched=True, batch_size=128, load_from_cache_file=False)
    drop_columns = [column for column in normalized.column_names if column not in output_columns()]
    return normalized.remove_columns(drop_columns)


def normalize_zero_tool_refusal_dataset(dataset: Any) -> Any:
    """把 ToolACE 原始无工具 refusal 样本整理成统一字段。"""
    def map_batch(examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        output = {key: [] for key in output_columns()}
        for offset in range(len(examples["user"])):
            output["system"].append(examples["system"][offset])
            output["user"].append(examples["user"][offset])
            output["assistant"].append(examples["assistant"][offset])
            output["tools"].append(examples["tools"][offset])
            output["tool_count"].append(examples["tool_count"][offset])
            output["tool_type"].append(examples["tool_type"][offset])
            output["refusal"].append(1)
            output["ablation_mode"].append("toolace_zero_tool_refusal")
            output["parameter_mapping"].append("{}")
            output["function_mapping"].append("{}")
            output["source_dataset"].append("toolace_zero_tool_refusal")
        return output

    normalized = dataset.map(map_batch, batched=True, batch_size=128, load_from_cache_file=False)
    drop_columns = [column for column in normalized.column_names if column not in output_columns()]
    return normalized.remove_columns(drop_columns)


def output_columns() -> list[str]:
    """统一保存到 prompt 前的数据列。"""
    return [
        "system",
        "user",
        "assistant",
        "tools",
        "tool_count",
        "tool_type",
        "refusal",
        "ablation_mode",
        "parameter_mapping",
        "function_mapping",
        "source_dataset",
    ]


def save_dataset(dataset: Any, path: str) -> None:
    """覆盖保存 datasets 格式数据。"""
    output_path = Path(path)
    if output_path.exists():
        shutil.rmtree(output_path)
    dataset.save_to_disk(path)


def add_prompt_column(dataset: Any, helpers: Any) -> Any:
    """根据当前 tools/assistant 重新组装 prompt。"""
    def map_batch(examples: dict[str, list[Any]]) -> dict[str, list[str]]:
        prompts = []
        for system, user, assistant, tools_text in zip(
            examples["system"],
            examples["user"],
            examples["assistant"],
            examples["tools"],
        ):
            prompts.append(helpers.render_prompt(system, user, assistant, helpers.load_tools(tools_text)))
        return {"text": prompts}

    return dataset.map(map_batch, batched=True, batch_size=128, load_from_cache_file=False)


def write_preview(dataset: Any, path: str, count: int = 8) -> None:
    """保存扰动和 teacher refusal 样本预览。"""
    ensure_dir(str(Path(path).parent))
    interesting = dataset.filter(
        lambda row: row["ablation_mode"] != "none",
        load_from_cache_file=False,
    )
    with open(path, "w", encoding="utf-8") as file:
        for index, row in enumerate(interesting.select(range(min(count, len(interesting)))), start=1):
            file.write(f"==================== Sample {index} ====================\n")
            file.write(f"source_dataset: {row['source_dataset']}\n")
            file.write(f"ablation_mode: {row['ablation_mode']}\n")
            file.write(f"parameter_mapping: {row['parameter_mapping']}\n")
            file.write(f"function_mapping: {row['function_mapping']}\n")
            file.write("-------------------- tools --------------------\n")
            file.write(row["tools"])
            file.write("\n-------------------- assistant --------------------\n")
            file.write(row["assistant"])
            file.write("\n-------------------- prompt --------------------\n")
            file.write(row["text"])
            file.write("\n\n")


def count_values(dataset: Any, column: str) -> dict[str, int]:
    """Counter 转普通 dict，方便 JSON 保存。"""
    return dict(Counter(str(value) for value in dataset[column]))


def prepare_ablation_data() -> bool:
    """主入口：生成 ablation2 tokenized SFT 数据。"""
    logger.info("Starting ablation2 data preparation.")

    try:
        from datasets import concatenate_datasets, load_from_disk
        from transformers import AutoTokenizer
    except ImportError as error:
        logger.error("Required packages are missing: %s", error)
        return False

    try:
        helpers = load_ablation1_helpers()
        config_path = "./configs/sft_config.yaml"
        config = load_experiment_config(config_path)
        data_config = config["data"]
        seed = int(config.get("seed", 42))
        max_seq_length = int(config["training"]["max_seq_length"])
        num_proc = int(data_config.get("preprocessing_num_workers", 1))
        structured_path = data_config["structured_dataset_path"]
        teacher_refusal_path = data_config["refusal_teacher_dataset_path"]
        train_path = data_config["processed_train_path"]
        test_path = data_config["processed_validation_path"]
        report_dir = Path(data_config.get("ablation2_report_dir", "./data/ablation2"))
        test_size = float(data_config["validation_split_percentage"]) / 100

        logger.info("Config path: %s", config_path)
        logger.info("Active experiment: %s", config.get("experiments", {}).get("active", "default"))
        logger.info("Structured dataset path: %s", structured_path)
        logger.info("Teacher refusal dataset path: %s", teacher_refusal_path)
        logger.info("Train output path: %s", train_path)
        logger.info("Test output path: %s", test_path)
        logger.info("Report dir: %s", report_dir)

        structured = load_from_disk(structured_path)
        non_refusal = structured.filter(
            lambda row: int(row["refusal"]) == 0 and int(row["tool_count"]) >= 1 and row["tool_type"] == "json",
            load_from_cache_file=False,
        )
        non_refusal = normalize_non_refusal_dataset(non_refusal)
        logger.info("Loaded structured samples: %s", len(structured))
        logger.info("Selected non-refusal json tool samples: %s", len(non_refusal))

        zero_tool_refusal = structured.filter(
            lambda row: int(row["refusal"]) == 1 and int(row["tool_count"]) == 0,
            load_from_cache_file=False,
        )
        zero_tool_refusal = normalize_zero_tool_refusal_dataset(zero_tool_refusal)
        logger.info("ToolACE zero-tool refusal samples: %s", len(zero_tool_refusal))

        teacher_refusal = normalize_refusal_teacher_dataset(load_from_disk(teacher_refusal_path))
        teacher_refusal = teacher_refusal.filter(
            lambda row: bool(str(row["assistant"]).strip()),
            load_from_cache_file=False,
        )
        logger.info("Teacher refusal description samples: %s", len(teacher_refusal))

        base_dataset = concatenate_datasets([
            non_refusal.select_columns(output_columns()),
            teacher_refusal.select_columns(output_columns()),
            zero_tool_refusal.select_columns(output_columns()),
        ])
        base_dataset = base_dataset.shuffle(seed=seed)
        split = base_dataset.train_test_split(test_size=test_size, seed=seed)
        train_raw = split["train"]
        test_raw = split["test"]

        train_selected_modes = select_mode_indices(train_raw, seed, helpers)
        test_selected_modes = select_eval_mode_indices(test_raw, seed, helpers)
        logger.info("Train selected ablation mode counts: %s", dict(Counter(train_selected_modes.values())))
        logger.info("Eval selected ablation mode counts: %s", dict(Counter(test_selected_modes.values())))

        train_raw = apply_non_refusal_ablation(train_raw, train_selected_modes, seed + 3000, helpers)
        test_raw = apply_non_refusal_ablation(test_raw, test_selected_modes, seed + 4000, helpers)
        train_dataset = add_prompt_column(train_raw, helpers)
        test_dataset = add_prompt_column(test_raw, helpers)
        logger.info("Fixed train/test split: train=%s test=%s", len(train_dataset), len(test_dataset))

        model_path = helpers.resolve_model_path(config)
        logger.info("Loading tokenizer: %s", model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        assistant_header_ids = tokenizer.encode(
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
            add_special_tokens=False,
        )
        eot_ids = tokenizer.encode("<|eot_id|>", add_special_tokens=False)

        processed_train = helpers.tokenize_for_sft(
            train_dataset,
            tokenizer,
            max_seq_length,
            assistant_header_ids,
            eot_ids,
            num_proc,
        )
        processed_test = helpers.tokenize_for_sft(
            test_dataset,
            tokenizer,
            max_seq_length,
            assistant_header_ids,
            eot_ids,
            num_proc,
        )

        save_dataset(processed_train, train_path)
        save_dataset(processed_test, test_path)

        ensure_dir(str(report_dir))
        preview_path = str(report_dir / "ablation2_preview.txt")
        metadata_path = str(report_dir / "metadata.json")
        write_preview(train_dataset, preview_path)

        train_eligible_count = sum(1 for row in split["train"] if non_refusal_is_eligible(row, helpers))
        test_eligible_count = sum(1 for row in split["test"] if non_refusal_is_eligible(row, helpers))
        metadata = {
            "experiment": "ablation2_parameter_function_refusal_teacher_description",
            "structured_dataset_path": structured_path,
            "refusal_teacher_dataset_path": teacher_refusal_path,
            "train_output_path": train_path,
            "test_output_path": test_path,
            "seed": seed,
            "max_seq_length": max_seq_length,
            "non_refusal_samples": len(non_refusal),
            "teacher_refusal_samples": len(teacher_refusal),
            "toolace_zero_tool_refusal_samples": len(zero_tool_refusal),
            "base_dataset_samples_before_split": len(base_dataset),
            "train_samples": len(processed_train),
            "test_samples": len(processed_test),
            "train_eligible_non_refusal_count": train_eligible_count,
            "test_eligible_non_refusal_count": test_eligible_count,
            "train_ablation_ratios": {
                "parameter_name": PARAMETER_ONLY_RATIO,
                "function_name": FUNCTION_ONLY_RATIO,
                "parameter_and_function": PARAMETER_AND_FUNCTION_RATIO,
                "total": PARAMETER_ONLY_RATIO + FUNCTION_ONLY_RATIO + PARAMETER_AND_FUNCTION_RATIO,
            },
            "eval_ablation_ratios": {
                "parameter_and_function": EVAL_PARAMETER_AND_FUNCTION_RATIO,
            },
            "train_selected_mode_counts": dict(Counter(train_selected_modes.values())),
            "test_selected_mode_counts": dict(Counter(test_selected_modes.values())),
            "base_source_counts": count_values(base_dataset, "source_dataset"),
            "train_source_counts": count_values(train_dataset, "source_dataset"),
            "test_source_counts": count_values(test_dataset, "source_dataset"),
            "train_ablation_mode_counts": count_values(train_dataset, "ablation_mode"),
            "test_ablation_mode_counts": count_values(test_dataset, "ablation_mode"),
            "preview_path": preview_path,
            "notes": [
                "先构造未扰动全集 a+b+c，再抽取 5% eval。",
                "a = ToolACE 中 refusal=0、tool_count>=1、tool_type=json 的样本。",
                "b = DeepSeek teacher refusal description，替换原 refusal=1、tool_count>0 的部分。",
                "c = ToolACE 中 refusal=1 且 tool_count=0 的原始样本。",
                "可扰动样本需同时存在可改名参数和函数名。",
                "train 内 a 类可扰动样本按 15%/10%/5% 执行参数名、函数名、二者同时扰动。",
                "eval 内 a 类可扰动样本按 50% 执行参数名和函数名同时扰动。",
                "所有扰动样本全部调用 rewrite_schema，因此都会打乱 schema 字段、properties 和 required 顺序。",
                "teacher refusal 进入 prompt 时仍包含 system + tools + user；description 只作为 assistant response。",
            ],
        }
        save_json(metadata, metadata_path)
        logger.info("Metadata path: %s", metadata_path)
        logger.info("Preview path: %s", preview_path)
        logger.info("Ablation2 data preparation completed.")
        return True

    except Exception as error:
        logger.error("Error preparing ablation2 data: %s", error)
        import traceback

        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    sys.exit(0 if prepare_ablation_data() else 1)
