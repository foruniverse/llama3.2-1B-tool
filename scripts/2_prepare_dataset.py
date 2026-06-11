#!/usr/bin/env python3
"""预处理 ToolACE 原始数据集，生成训练前结构化数据集。

本脚本集中完成 ToolACE 的所有数据集特有逻辑：

1. 将原始 conversations 裁剪为第一轮 user + assistant。
2. 从 system 中拆出 system prompt 和 tools。
3. 统计 tool_count、tool_type、refusal。
4. 手动拼接 Llama chat template 并统计 prompt token 长度。
5. 删除 prompt token 长度超过 training.max_seq_length 的样本。
6. 保存只包含 system/user/assistant/tools/tool_count/tool_type/refusal 的新 dataset。

注意：这里故意不使用 tokenizer.apply_chat_template()，避免 tokenizer 自带模板
自动加入日期等额外内容，导致长度统计和真实训练 prompt 不一致。
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.utils import ensure_dir, load_experiment_config, save_json, setup_logging

logger = setup_logging("./logs/prepare_dataset.log")

TEMP_COLUMNS = ["_prompt", "_prompt_token_length", "_tool_extract_failed"]
FINAL_COLUMNS = ["system", "user", "assistant", "tools", "tool_count", "tool_type", "refusal"]


def resolve_model_path(config: dict[str, Any]) -> str:
    """优先使用本地已下载 tokenizer，避免重复访问远端。"""
    cache_dir = Path(config["model"].get("cache_dir", "./models/pretrained"))
    local_path = cache_dir / "AI-ModelScope" / "Llama-3.2-1B-Instruct"
    if (local_path / "tokenizer_config.json").exists():
        return str(local_path)
    return config["model"]["model_name"]


def first_user_assistant(conversations: list[dict[str, Any]]) -> tuple[str, str]:
    """保留第一轮 user 和其后的第一条 assistant 回复。

    ToolACE 中存在多轮样本，但当前训练只需要单轮。这里不丢弃多轮样本，
    而是把它们统一裁剪成第一轮，保证数据量不被无谓减少。
    """
    user = ""
    assistant = ""
    for turn in conversations:
        role = turn.get("role") or turn.get("from")
        content = str(turn.get("content") or turn.get("value") or "").strip()
        if not content:
            continue
        if role == "user" and not user:
            user = content
            continue
        if role == "assistant" and user:
            assistant = content
            break
    return user, assistant


def find_tool_json_block(text: str) -> tuple[int, int, Any] | None:
    """找到 system 中真正代表工具列表的 JSON 块。

    不能简单看到 `[` 或 `{` 就解析，因为工具 schema 内部经常有
    `required: []`、`enum: []` 这类嵌套数组。只有看起来像工具定义的
    JSON，或在工具列表上下文里的顶层空列表 `[]`，才算 tools。
    """
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            parsed, offset = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        end = start + offset
        if is_tool_json(parsed) or is_empty_json_tool_list(parsed, text, start):
            return start, end, parsed
    return None


def is_tool_json(parsed: Any) -> bool:
    """判断 JSON 对象是否像工具/function schema。"""
    tools = parsed if isinstance(parsed, list) else [parsed]
    if not tools:
        return False
    return all(isinstance(tool, dict) for tool in tools) and any(tool_name_from_dict(tool) for tool in tools)


def tool_name_from_dict(tool: dict[str, Any]) -> str:
    """兼容 ToolACE 中出现过的常见工具名称字段。"""
    for key in ("name", "tool_name", "function_name", "api_name", "API_Name"):
        value = tool.get(key)
        if value:
            return str(value)
    function = tool.get("function")
    if isinstance(function, dict) and function.get("name"):
        return str(function["name"])
    return ""


def is_empty_json_tool_list(parsed: Any, text: str, start: int) -> bool:
    """识别显式空工具列表，避免把它误判为解析失败。

    这里必须检查前文上下文，否则参数 schema 里的 `required: []` 会被误当
    成顶层工具列表。只有前文提到 list of functions/tools 且 JSON 格式时，
    顶层空数组才表示“本样本没有可调用工具”。
    """
    if parsed != []:
        return False
    prefix = text[max(0, start - 180):start].lower()
    has_tool_list_hint = "list of functions" in prefix or "list of tools" in prefix
    return has_tool_list_hint and "json" in prefix


def parse_system(system: str) -> dict[str, Any]:
    """从原始 system 中拆出 system prompt、tools、tool_type。"""
    system = system.strip()
    if not system:
        return {"system": "", "tools": [], "tool_type": "non_json", "extract_failed": True}

    json_block = find_tool_json_block(system)
    if json_block:
        start, end, parsed = json_block
        tools = parsed if isinstance(parsed, list) else [parsed]
        system_prompt = (system[:start] + system[end:]).strip()
        return {
            "system": system_prompt,
            "tools": tools,
            "tool_type": "json",
            "extract_failed": False,
        }

    tools = extract_non_json_tools(system)
    return {
        "system": system,
        "tools": tools,
        "tool_type": "non_json",
        "extract_failed": len(tools) == 0,
    }


def extract_non_json_tools(system: str) -> list[str]:
    """抽取非 JSON 工具定义。

    ToolACE 中有少量样本使用 HTML table、XML-like、LaTeX table 或
    `tool_name:` 块描述工具。这里不再把它们包装成结构化 dict，
    而是直接把提取出来的工具片段放入 tools，tool_count 仍然用
    len(tools) 统计。
    """
    for extractor in (
        extract_html_table_tools,
        extract_xml_tools,
        extract_latex_table_tools,
        extract_tool_name_blocks,
    ):
        tools = extractor(system)
        if tools:
            return tools
    return []


def extract_html_table_tools(text: str) -> list[str]:
    rows = re.findall(r"<tr>(.*?)</tr>", text, flags=re.IGNORECASE | re.DOTALL)
    tools = []
    for row in rows:
        cells = re.findall(r"<td>(.*?)</td>", row, flags=re.IGNORECASE | re.DOTALL)
        if not cells:
            continue
        name = clean_markup(cells[0])
        if name and name.lower() != "tool_name":
            tools.append(clean_markup(row))
    return tools


def extract_xml_tools(text: str) -> list[str]:
    matches = list(re.finditer(r"<tool_name>(.*?)</tool_name>", text, flags=re.IGNORECASE | re.DOTALL))
    tools = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else tool_section_end(text, match.end())
        raw = text[match.start():end].strip()
        name = clean_markup(match.group(1))
        if name:
            tools.append(raw)
    return tools


def extract_latex_table_tools(text: str) -> list[str]:
    if "\\begin{tabular}" not in text:
        return []
    tools = []
    for raw_row in re.split(r"\\\\\s*(?:\\hline)?", text):
        if "&" not in raw_row:
            continue
        cells = [clean_markup(cell) for cell in raw_row.split("&")]
        if not cells or cells[0].lower() == "tool_name":
            continue
        if len(cells[0]) > 1 and not cells[0].startswith("\\"):
            tools.append(clean_markup(raw_row))
    return tools


def extract_tool_name_blocks(text: str) -> list[str]:
    pattern = re.compile(r"(?im)^(?:[-*]\s*)?(?:\*\*)?tool_name(?:\*\*)?\s*:\s*(.+?)\s*$")
    matches = list(pattern.finditer(text))
    tools = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else tool_section_end(text, match.end())
        raw = text[match.start():end].strip()
        name = clean_markup(match.group(1))
        if name:
            tools.append(raw)
    return tools


def tool_section_end(text: str, start: int) -> int:
    """找到非 JSON 工具块结束位置，避免把输出格式说明吞进 raw。"""
    markers = [
        "\nPlease use the following format",
        "\nDo not include parameters",
        "\nWhen invoking tools",
        "\nShould you decide",
    ]
    positions = [text.find(marker, start) for marker in markers]
    positions = [position for position in positions if position != -1]
    return min(positions) if positions else len(text)


def clean_markup(value: str) -> str:
    """清理 HTML/LaTeX/Markdown 标记，得到更稳定的工具名。"""
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\\(?:hline|begin\{[^}]+\}|end\{[^}]+\})", " ", value)
    value = value.replace("**", "")
    value = value.replace("\\", " ")
    return re.sub(r"\s+", " ", value).strip(" |")


def render_system(system: str, tools: list[Any]) -> str:
    """把 system prompt 和 tools 合成最终 system 内容。"""
    system = system.strip()
    tools_text = json.dumps(tools, ensure_ascii=False, indent=2)
    if not system:
        return tools_text
    return f"{system}\n\n{tools_text}"


def render_prompt(system: str, user: str, assistant: str, tools: list[Any]) -> str:
    """手动拼 Llama chat template，不使用 tokenizer.apply_chat_template。"""
    system_content = render_system(system, tools)
    return (
        "<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n{system_content}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n{assistant}<|eot_id|>"
    )


def prompt_token_length(tokenizer: Any, prompt: str) -> int:
    """统计不截断的 prompt token 长度。"""
    return len(tokenizer(prompt, add_special_tokens=False, truncation=False)["input_ids"])


def is_function_call_response(content: str) -> bool:
    """判断 assistant 回复是否是函数调用。

    ToolACE 的调用格式很多，常见有效调用以括号开头，也有 `Latest Rates:(...)`
    这种函数名后接括号的形式。除此之外的自然语言解释视为拒绝调用。
    """
    content = content.strip().strip('"')
    if not content:
        return False
    if content[0] in "[{(<":
        return True
    return bool(re.match(r"^[A-Za-z0-9_. -]{1,80}\s*:?\s*\(", content))


def process_batch(examples: dict[str, list[Any]], tokenizer: Any) -> dict[str, list[Any]]:
    """批量处理原始 ToolACE 样本。"""
    batch = {
        "system": [],
        "user": [],
        "assistant": [],
        "tools": [],
        "tool_count": [],
        "tool_type": [],
        "refusal": [],
        "_prompt": [],
        "_prompt_token_length": [],
        "_tool_extract_failed": [],
    }

    for raw_system, conversations in zip(examples["system"], examples["conversations"]):
        parsed = parse_system(str(raw_system))
        user, assistant = first_user_assistant(conversations or [])
        tools = parsed["tools"]
        prompt = render_prompt(parsed["system"], user, assistant, tools)

        batch["system"].append(parsed["system"])
        batch["user"].append(user)
        batch["assistant"].append(assistant)
        batch["tools"].append(json.dumps(tools, ensure_ascii=False))
        batch["tool_count"].append(len(tools))
        batch["tool_type"].append(parsed["tool_type"])
        batch["refusal"].append(0 if is_function_call_response(assistant) else 1)
        batch["_prompt"].append(prompt)
        batch["_prompt_token_length"].append(prompt_token_length(tokenizer, prompt))
        batch["_tool_extract_failed"].append(bool(parsed["extract_failed"]))

    return batch


def length_stats(values: list[int]) -> dict[str, int]:
    """生成简洁长度统计。"""
    if not values:
        return {"count": 0, "min": 0, "p50": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0}
    values = sorted(values)

    def percentile(q: int) -> int:
        index = round((q / 100) * (len(values) - 1))
        return values[index]

    return {
        "count": len(values),
        "min": values[0],
        "p50": percentile(50),
        "p90": percentile(90),
        "p95": percentile(95),
        "p99": percentile(99),
        "max": values[-1],
    }


def bucket_tool_count(tool_count: int) -> str:
    """把工具数量归为 0/single/multiple。"""
    if tool_count <= 0:
        return "zero_tool"
    if tool_count == 1:
        return "single_tool"
    return "multiple_tool"


def ratio_rows(counts: dict[str, int], total: int) -> dict[str, dict[str, float | int]]:
    """给计数附上比例。"""
    return {
        key: {"count": value, "ratio": value / total if total else 0}
        for key, value in counts.items()
    }


def build_report(processed: Any, filtered: Any, max_seq_length: int, output_path: str) -> dict[str, Any]:
    """汇总过滤前后统计信息。"""
    raw_count = len(processed)
    final_count = len(filtered)
    length_values = [int(value) for value in processed["_prompt_token_length"]]
    filtered_lengths = [int(value) for value in filtered["_prompt_token_length"]]
    tool_type_counts = Counter(filtered["tool_type"])
    tool_bucket_counts = Counter(bucket_tool_count(int(value)) for value in filtered["tool_count"])
    refusal_counts = Counter("refusal" if int(value) == 1 else "non_refusal" for value in filtered["refusal"])
    failed_count = sum(1 for value in processed["_tool_extract_failed"] if value)

    return {
        "dataset_path": output_path,
        "raw_samples": raw_count,
        "final_samples": final_count,
        "max_seq_length": max_seq_length,
        "removed_over_length_samples": raw_count - final_count,
        "removed_over_length_ratio": (raw_count - final_count) / raw_count if raw_count else 0,
        "tool_extract_failed_samples": failed_count,
        "length_stats_before_filter": length_stats(length_values),
        "length_stats_after_filter": length_stats(filtered_lengths),
        "refusal_counts": ratio_rows(dict(refusal_counts), final_count),
        "tool_type_counts": ratio_rows(dict(tool_type_counts), final_count),
        "tool_count_bucket_counts": ratio_rows(dict(tool_bucket_counts), final_count),
    }


def markdown_table(title: str, rows: dict[str, dict[str, float | int]]) -> str:
    """渲染 Markdown 统计表。"""
    lines = [f"## {title}", "", "| 类别 | 数量 | 比例 |", "| --- | ---: | ---: |"]
    for key, value in rows.items():
        lines.append(f"| `{key}` | {value['count']} | {value['ratio']:.2%} |")
    return "\n".join(lines)


def write_markdown_report(report: dict[str, Any], output_path: str) -> None:
    """写入人工可读的数据集统计报告。"""
    lines = [
        "# ToolACE 数据集预处理统计",
        "",
        "## 处理流程",
        "",
        "1. 将原始 conversations 裁剪为第一轮 user + assistant。",
        "2. 从 system 中拆出 tools，并保留拆分后的 system。",
        "3. 统计 tool_count、tool_type、refusal。",
        "4. 手动拼接 Llama chat template 统计 prompt token 长度。",
        f"5. 删除 prompt token 长度大于 {report['max_seq_length']} 的样本。",
        "",
        "## 总览",
        "",
        f"- 原始样本数：{report['raw_samples']}",
        f"- 最终样本数：{report['final_samples']}",
        f"- 超长删除样本数：{report['removed_over_length_samples']} ({report['removed_over_length_ratio']:.2%})",
        f"- tools 提取失败样本数：{report['tool_extract_failed_samples']}",
        f"- 输出数据集：`{report['dataset_path']}`",
        "",
        "## 长度统计",
        "",
        "过滤前：",
        "",
        "```json",
        json.dumps(report["length_stats_before_filter"], ensure_ascii=False, indent=2),
        "```",
        "",
        "过滤后：",
        "",
        "```json",
        json.dumps(report["length_stats_after_filter"], ensure_ascii=False, indent=2),
        "```",
        "",
        markdown_table("拒绝调用 / 非拒绝调用", report["refusal_counts"]),
        "",
        markdown_table("工具定义类型", report["tool_type_counts"]),
        "",
        markdown_table("工具数量分布", report["tool_count_bucket_counts"]),
        "",
    ]
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def write_sample_prompts(dataset: Any, output_path: str, count: int = 5) -> None:
    """保存少量手动拼接后的 prompt，便于人工检查。"""
    with open(output_path, "w", encoding="utf-8") as file:
        for index, row in enumerate(dataset.select(range(min(count, len(dataset)))), start=1):
            file.write(f"==================== Sample {index} ====================\n")
            file.write(f"tool_count: {row['tool_count']}\n")
            file.write(f"tool_type: {row['tool_type']}\n")
            file.write(f"refusal: {row['refusal']}\n")
            file.write(f"prompt_token_length: {row['_prompt_token_length']}\n")
            file.write("-------------------- Prompt --------------------\n")
            file.write(row["_prompt"])
            file.write("\n\n")


def save_readable_subset(dataset: Any, output_path: str) -> None:
    """把问题样本写成 jsonl，避免只能读 Arrow 文件。"""
    with open(output_path, "w", encoding="utf-8") as file:
        for row in dataset:
            item = {key: row[key] for key in FINAL_COLUMNS}
            item["prompt_token_length"] = row["_prompt_token_length"]
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


def prepare_dataset() -> bool:
    """主入口：生成新的 ToolACE processed dataset。"""
    logger.info("Starting ToolACE dataset preprocessing.")

    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as error:
        logger.error("Required packages are missing: %s", error)
        return False

    try:
        config_path = "./configs/sft_config.yaml"
        config = load_experiment_config(config_path)
        data_config = config["data"]
        dataset_name = data_config["dataset_name"]
        output_path = data_config.get("structured_dataset_path", "./data/tool_ace_processed")
        max_seq_length = int(config["training"]["max_seq_length"])
        num_proc = int(data_config.get("preprocessing_num_workers", 1))
        map_kwargs = {"num_proc": num_proc} if num_proc > 1 else {}

        logger.info("Config path: %s", config_path)
        logger.info("Dataset name: %s", dataset_name)
        logger.info("Output dataset path: %s", output_path)
        logger.info("Max prompt token length: %s", max_seq_length)

        raw_dataset = load_dataset(dataset_name)
        raw_train = raw_dataset["train"] if "train" in raw_dataset else raw_dataset
        logger.info("Raw samples: %s", len(raw_train))

        model_path = resolve_model_path(config)
        logger.info("Loading tokenizer: %s", model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        processed = raw_train.map(
            lambda batch: process_batch(batch, tokenizer),
            batched=True,
            batch_size=32,
            remove_columns=raw_train.column_names,
            load_from_cache_file=False,
            **map_kwargs,
        )
        logger.info("Processed raw samples: %s", len(processed))

        filtered = processed.filter(
            lambda row: int(row["_prompt_token_length"]) <= max_seq_length,
            load_from_cache_file=False,
            **map_kwargs,
        )
        logger.info("Filtered samples: %s", len(filtered))

        report = build_report(processed, filtered, max_seq_length, output_path)
        final_dataset = filtered.remove_columns(TEMP_COLUMNS)

        output_dir = Path(output_path)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        final_dataset.save_to_disk(output_path)

        report_path = str(output_dir / "dataset_report.json")
        markdown_path = str(output_dir / "dataset_report.md")
        sample_prompt_path = str(output_dir / "sample_prompts.txt")
        failed_tools_path = str(output_dir / "tool_extract_failed.jsonl")
        zero_tool_path = str(output_dir / "zero_tool.jsonl")

        save_json(report, report_path)
        write_markdown_report(report, markdown_path)
        write_sample_prompts(filtered, sample_prompt_path)
        save_readable_subset(
            processed.filter(lambda row: bool(row["_tool_extract_failed"]), load_from_cache_file=False, **map_kwargs),
            failed_tools_path,
        )
        save_readable_subset(
            filtered.filter(lambda row: int(row["tool_count"]) == 0, load_from_cache_file=False, **map_kwargs),
            zero_tool_path,
        )

        logger.info("Final dataset columns: %s", final_dataset.column_names)
        logger.info("Final samples: %s", len(final_dataset))
        logger.info("Dataset report: %s", report_path)
        logger.info("Markdown report: %s", markdown_path)
        logger.info("Sample prompts: %s", sample_prompt_path)
        logger.info("Tool extract failed jsonl: %s", failed_tools_path)
        logger.info("Zero tool jsonl: %s", zero_tool_path)
        logger.info("ToolACE dataset preprocessing completed.")
        return True

    except Exception as error:
        logger.error("Error preparing ToolACE dataset: %s", error)
        import traceback

        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    sys.exit(0 if prepare_dataset() else 1)
