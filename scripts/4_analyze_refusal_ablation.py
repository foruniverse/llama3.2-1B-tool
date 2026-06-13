#!/usr/bin/env python3
"""辅助分析脚本：统计 refusal 类型，并评估是否适合做参数扰动。

注意：
- 这是实验分析工具，不接入主训练流程。
- 只读取 `data/tool_ace_processed`，不修改训练数据。
- 跳过 tool_count=0 的样本，因为没有工具时不存在参数扰动问题。
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.utils import ensure_dir, save_json

DATASET_PATH = Path("data/tool_ace_processed")
OUTPUT_DIR = Path("data/ablation_refusal_analysis")
REPORT_MD = OUTPUT_DIR / "refusal_ablation_analysis.md"
REPORT_JSON = OUTPUT_DIR / "refusal_ablation_analysis.json"


def load_ablation_helpers():
    """复用 ablation1 中已经修正过的参数收集/替换逻辑。"""
    path = Path("scripts/2_prepare_data_ablation1.py")
    spec = importlib.util.spec_from_file_location("ablation1_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load helper script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ratio(count: int, total: int) -> str:
    """格式化比例，避免 markdown 表里重复写除法。"""
    return "0.00%" if total == 0 else f"{count / total * 100:.2f}%"


def markdown_table(title: str, rows: list[tuple[str, int]], total: int) -> str:
    """生成简单 markdown 表。"""
    lines = [f"## {title}", "", "| 类型 | 数量 | 比例 |", "| --- | ---: | ---: |"]
    for name, count in rows:
        lines.append(f"| {name} | {count} | {ratio(count, total)} |")
    lines.append("")
    return "\n".join(lines)


def classify_refusal(text: str) -> str:
    """把 refusal 回复粗分为几类，便于判断是否适合扰动。

    这是启发式分类，不作为训练标签使用。优先级按最常见且最有行动意义的
    类别排列：缺参数追问 > 实时/外部访问限制 > 无匹配工具 > 直接解释回答。
    """
    lower = text.lower()

    missing_markers = [
        "may i know",
        "could you provide",
        "please provide",
        "please specify",
        "which ",
        "what ",
        "need the",
        "need your",
        "required",
        "missing",
        "lack",
        "lacks",
    ]
    if "?" in text and any(marker in lower for marker in missing_markers):
        return "missing_parameters_or_clarification"

    realtime_markers = [
        "real-time",
        "current",
        "latest",
        "live",
        "trending",
        "right now",
        "directly fetch",
        "access to",
        "browse",
        "internet",
        "up-to-date",
    ]
    capability_markers = ["i don't have", "i do not have", "i can't", "i cannot", "unable to"]
    if any(marker in lower for marker in realtime_markers) and any(marker in lower for marker in capability_markers):
        return "realtime_or_external_access_limit"

    no_tool_markers = [
        "none of the",
        "no function",
        "no tool",
        "does not provide",
        "do not provide",
        "not designed",
        "not available",
        "cannot provide a way",
    ]
    if any(marker in lower for marker in no_tool_markers):
        return "no_matching_tool"

    if len(text.split()) >= 40 and not text.strip().startswith("["):
        return "direct_explanation"

    return "other_refusal"


def parameter_assignment_pattern(name: str) -> re.Pattern[str]:
    """匹配当前 ablation1 认为安全的参数名位置。

    只看参数名后面紧跟赋值符号的场景：=、:、|、/、~。
    refusal 回复通常是自然语言，因此这里命中越少，越说明不适合直接改 assistant。
    """
    escaped = re.escape(name)
    return re.compile(
        rf"(?:(?P<quote>['\"]){escaped}(?P=quote)|(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_]))"
        rf"(?=\s*(?:==|=|:|/|\||~))"
    )


def assistant_has_safe_parameter_position(assistant: str, parameter_names: list[str]) -> bool:
    """判断 refusal assistant 中是否出现安全的参数赋值位置。"""
    return any(parameter_assignment_pattern(name).search(assistant) for name in parameter_names)


def assistant_mentions_parameter(assistant: str, parameter_names: list[str]) -> bool:
    """判断 assistant 是否以自然语言提到了参数名。"""
    lower = assistant.lower()
    for name in parameter_names:
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(name.lower())}(?![A-Za-z0-9_])", lower):
            return True
    return False


def short_example(row: dict[str, Any], parameters: list[str]) -> dict[str, Any]:
    """保存少量可读字段，方便人工复核分类。"""
    return {
        "user": row["user"][:500],
        "assistant": row["assistant"][:800],
        "tool_count": int(row["tool_count"]),
        "tool_type": row["tool_type"],
        "parameters_with_description": parameters[:30],
    }


def analyze() -> None:
    """主分析入口。"""
    from datasets import load_from_disk

    helpers = load_ablation_helpers()
    dataset = load_from_disk(str(DATASET_PATH))
    ensure_dir(str(OUTPUT_DIR))

    total_refusal_with_tools = 0
    refusal_type_counts: Counter[str] = Counter()
    tool_type_counts: Counter[str] = Counter()
    tool_count_bucket_counts: Counter[str] = Counter()
    ablation_feasibility_counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in dataset:
        if int(row["refusal"]) != 1 or int(row["tool_count"]) == 0:
            continue

        total_refusal_with_tools += 1
        refusal_type = classify_refusal(str(row["assistant"]))
        refusal_type_counts[refusal_type] += 1
        tool_type_counts[str(row["tool_type"])] += 1

        tool_count = int(row["tool_count"])
        if tool_count == 1:
            tool_count_bucket = "single_tool"
        else:
            tool_count_bucket = "multiple_tool"
        tool_count_bucket_counts[tool_count_bucket] += 1

        tools = helpers.load_tools(row["tools"])
        parameters = helpers.collect_parameter_names(tools)
        has_described_parameters = bool(parameters)
        has_safe_position = assistant_has_safe_parameter_position(str(row["assistant"]), parameters)
        mentions_parameter = assistant_mentions_parameter(str(row["assistant"]), parameters)

        if str(row["tool_type"]) != "json":
            feasibility = "cannot_ablate_non_json_tool"
        elif not has_described_parameters:
            feasibility = "cannot_ablate_no_described_parameters"
        elif has_safe_position:
            feasibility = "can_ablate_tools_and_assistant_safely"
        elif mentions_parameter:
            feasibility = "can_ablate_tools_only_assistant_mentions_params_in_prose"
        else:
            feasibility = "can_ablate_tools_only_assistant_no_param_reference"
        ablation_feasibility_counts[feasibility] += 1

        if len(examples[refusal_type]) < 3:
            examples[refusal_type].append(short_example(row, parameters))
        if len(examples[feasibility]) < 3:
            examples[feasibility].append(short_example(row, parameters))

    report = {
        "dataset_path": str(DATASET_PATH),
        "total_refusal_with_tools": total_refusal_with_tools,
        "refusal_type_counts": dict(refusal_type_counts),
        "tool_type_counts": dict(tool_type_counts),
        "tool_count_bucket_counts": dict(tool_count_bucket_counts),
        "ablation_feasibility_counts": dict(ablation_feasibility_counts),
        "examples": examples,
        "conclusion": [
            "如果只扰动 tools schema，绝大多数 JSON refusal 样本可以处理。",
            "如果要求同步修改 assistant，自然语言 refusal 不适合直接套用函数调用样本的参数替换规则。",
            "原因是 refusal 多数以追问/解释形式提到参数，参数名不是赋值位置；强行替换会重新引入误替换普通文本的风险。",
        ],
    }
    save_json(report, str(REPORT_JSON))
    write_markdown(report)


def write_markdown(report: dict[str, Any]) -> None:
    """写入给人看的 markdown 报告。"""
    total = int(report["total_refusal_with_tools"])
    lines = [
        "# Refusal 样本参数扰动可行性分析",
        "",
        f"- 数据集：`{report['dataset_path']}`",
        f"- 统计范围：`refusal=1` 且 `tool_count>0`",
        f"- 样本数：`{total}`",
        "",
    ]
    lines.append(markdown_table("Refusal 类型", sorted(report["refusal_type_counts"].items()), total))
    lines.append(markdown_table("工具类型", sorted(report["tool_type_counts"].items()), total))
    lines.append(markdown_table("工具数量桶", sorted(report["tool_count_bucket_counts"].items()), total))
    lines.append(markdown_table("参数扰动可行性", sorted(report["ablation_feasibility_counts"].items()), total))

    lines.extend(
        [
            "## 结论",
            "",
            "- refusal 可以扰动 tools schema，尤其是 JSON tools 中带 description 的参数。",
            "- 但不建议默认同步改 assistant 自然语言回复。",
            "- 当前安全替换规则只替换 `param=value`、`\"param\": value` 等参数赋值位置；多数 refusal 是自然语言追问或解释，不满足这个结构。",
            "- 如果后续一定要做 refusal ablation，建议新增单独实验：只扰动 tools，不改 assistant；或者单独设计自然语言参数名替换规则并人工抽样验证。",
            "",
            "## 示例",
            "",
        ]
    )

    for name, examples in report["examples"].items():
        lines.append(f"### {name}")
        lines.append("")
        for index, example in enumerate(examples, start=1):
            lines.append(f"#### Example {index}")
            lines.append("")
            lines.append(f"- tool_type: `{example['tool_type']}`")
            lines.append(f"- tool_count: `{example['tool_count']}`")
            lines.append(f"- parameters_with_description: `{example['parameters_with_description']}`")
            lines.append("")
            lines.append("user:")
            lines.append("")
            lines.append("```text")
            lines.append(example["user"])
            lines.append("```")
            lines.append("")
            lines.append("assistant:")
            lines.append("")
            lines.append("```text")
            lines.append(example["assistant"])
            lines.append("```")
            lines.append("")

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    analyze()
