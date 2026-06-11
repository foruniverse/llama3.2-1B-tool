#!/usr/bin/env python3
"""临时脚本：让 DeepSeek 教师模型重新标注 refusal 样本。

本脚本不接入主流程。它用于验证一个新想法：

1. 从 `data/tool_ace_processed` 读取 `refusal=1` 且有 tool 的样本。
2. 按固定比例决定是否扰动参数名、函数名。
3. 只把 user 和扰乱后的 tools 发给教师模型，不提交原始 refusal 回复。
4. 要求教师模型输出统一 JSON：

   {
     "success": [],
     "failed": [],
     "description": ""
   }

运行方式：

```bash
# 先只生成待请求样本和 prompt 预览，不调用教师模型
uv run python scripts/tmp_generate_refusal_teacher_actions.py --limit 20 --dry-run

# 配置 DeepSeek API 后调用教师模型
DEEPSEEK_API_KEY=xxx \
uv run python scripts/tmp_generate_refusal_teacher_actions.py --limit 50
```
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from pathlib import Path
from typing import Any

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.utils import ensure_dir, save_json

DEFAULT_DATASET_PATH = "data/tool_ace_processed"
DEFAULT_OUTPUT_DIR = "data/ablation_refusal_teacher_deepseek"
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"
ALLOWED_FAILED_REASONS = {"miss_param", "no_suitable_tool"}
ABLATION_RATIOS = {
    "original_refusal": 0.60,
    "param_only": 0.20,
    "function_only": 0.10,
    "param_and_function": 0.10,
}
_THREAD_LOCAL = threading.local()


def load_ablation_helpers():
    """复用 ablation1 已修正过的参数扰动逻辑，避免两处实现漂移。"""
    path = Path("scripts/2_prepare_data_abalation1.py")
    spec = importlib.util.spec_from_file_location("ablation1_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load helper script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Generate DeepSeek teacher annotations for refusal samples.")
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=0, help="最多处理多少条可扰动 refusal 样本；0 表示全量。")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="只生成请求预览，不调用教师模型。")
    parser.add_argument("--model", default=None, help=f"教师模型名称；默认 {DEFAULT_MODEL}，也可读取 TEACHER_MODEL。")
    parser.add_argument("--base-url", default=None, help=f"DeepSeek/OpenAI-compatible base URL；默认 {DEFAULT_BASE_URL}。")
    parser.add_argument("--api-key", default=None, help="API key；默认读取 DEEPSEEK_API_KEY 或 TEACHER_API_KEY。")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--sleep", type=float, default=0.0, help="每次请求后的等待秒数，避免打爆本地服务。")
    parser.add_argument("--flush-every", type=int, default=10, help="每处理多少条就刷新一次 JSONL 文件。")
    parser.add_argument("--resume", action="store_true", help="读取已有 JSONL 结果，跳过已经处理过的 source_index。")
    parser.add_argument("--workers", type=int, default=4, help="并发请求数。DeepSeek API 可用时建议 4-8。")
    return parser.parse_args()


def load_json_tools(tools_text: str) -> list[Any]:
    """解析 tools 字符串；失败时返回空列表。"""
    try:
        value = json.loads(tools_text)
    except json.JSONDecodeError:
        return []
    return value if isinstance(value, list) else []


def get_tool_name(tool: Any) -> str | None:
    """从常见 JSON tool 结构里取工具名。"""
    if not isinstance(tool, dict):
        return None
    name = tool.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    function = tool.get("function")
    if isinstance(function, dict) and isinstance(function.get("name"), str):
        return function["name"].strip()
    return None


def get_tool_parameters(tool: dict[str, Any]) -> Any:
    """兼容 ToolACE 和 OpenAI tools 两种常见参数位置。"""
    if isinstance(tool.get("parameters"), dict):
        return tool["parameters"]
    function = tool.get("function")
    if isinstance(function, dict) and isinstance(function.get("parameters"), dict):
        return function["parameters"]
    return {}


def collect_schema_parameter_names(schema: Any, output: set[str]) -> None:
    """递归收集 schema 中出现过的参数名，用于校验 teacher 输出。"""
    if isinstance(schema, list):
        for item in schema:
            collect_schema_parameter_names(item, output)
        return
    if not isinstance(schema, dict):
        return

    properties = schema.get("properties")
    if isinstance(properties, dict):
        for name, child_schema in properties.items():
            output.add(str(name))
            collect_schema_parameter_names(child_schema, output)

    for key, value in schema.items():
        if key != "properties":
            collect_schema_parameter_names(value, output)


def tool_index(tools: list[Any]) -> dict[str, set[str]]:
    """生成 tool -> 参数名集合，校验 arguments/missing 是否属于该 tool。"""
    index: dict[str, set[str]] = {}
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = get_tool_name(tool)
        if not name:
            continue
        params: set[str] = set()
        collect_schema_parameter_names(get_tool_parameters(tool), params)
        index[name] = params
    return index


def make_function_mapping(tools: list[Any], rng: random.Random) -> dict[str, str]:
    """为每个 tool name 生成随机函数名。

    ToolACE 里有些工具名包含空格，例如 `Hashtag Info API`。函数名扰动时
    统一改成 `func_XXXX`，这样 teacher 生成 BFCL call 时更容易保持合法格式。
    """
    mapping: dict[str, str] = {}
    used = {name for name in (get_tool_name(tool) for tool in tools) if name}
    for name in sorted(used):
        while True:
            new_name = f"func_{rng.randint(0, 9999):04d}"
            if new_name not in used:
                break
        mapping[name] = new_name
        used.add(new_name)
    return mapping


def rewrite_function_names(value: Any, mapping: dict[str, str]) -> Any:
    """递归替换 tools 中的函数名。"""
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


def assign_ablation_modes(sample_count: int, seed: int) -> dict[int, str]:
    """按 60/20/10/10 固定比例给样本分配扰动模式。"""
    indices = list(range(sample_count))
    rng = random.Random(seed + 7001)
    rng.shuffle(indices)

    original_count = int(sample_count * ABLATION_RATIOS["original_refusal"])
    param_count = int(sample_count * ABLATION_RATIOS["param_only"])
    function_count = int(sample_count * ABLATION_RATIOS["function_only"])
    both_count = sample_count - original_count - param_count - function_count

    mode_plan: list[str] = (
        ["original_refusal"] * original_count
        + ["param_only"] * param_count
        + ["function_only"] * function_count
        + ["param_and_function"] * both_count
    )
    return {index: mode for index, mode in zip(indices, mode_plan)}


def select_refusal_rows(dataset: Any, helpers: Any, limit: int) -> list[dict[str, Any]]:
    """筛选可交给教师模型处理的 refusal 样本。"""
    rows: list[dict[str, Any]] = []
    for row in dataset:
        if int(row["refusal"]) != 1:
            continue
        if int(row["tool_count"]) <= 0:
            continue
        if row["tool_type"] != "json":
            continue
        tools = load_json_tools(row["tools"])
        if not helpers.collect_parameter_names(tools):
            continue
        rows.append(dict(row))
        if limit and len(rows) >= limit:
            break
    return rows


def perturb_tools(row: dict[str, Any], helpers: Any, rng: random.Random, mode: str) -> dict[str, Any]:
    """按模式扰乱 tools，返回新 tools 和映射。"""
    tools = copy.deepcopy(load_json_tools(row["tools"]))
    parameter_mapping: dict[str, str] = {}
    function_mapping: dict[str, str] = {}

    if mode in {"param_only", "param_and_function"}:
        names = helpers.collect_parameter_names(tools)
        parameter_mapping = helpers.make_parameter_mapping(names, rng)
        tools = helpers.rewrite_schema(tools, parameter_mapping, rng)

    if mode in {"function_only", "param_and_function"}:
        function_mapping = make_function_mapping(tools, rng)
        tools = rewrite_function_names(tools, function_mapping)

    return {
        "ablation_mode": mode,
        "tools": tools,
        "tools_text": json.dumps(tools, ensure_ascii=False),
        "parameter_mapping": parameter_mapping,
        "function_mapping": function_mapping,
    }

def build_teacher_messages(user: str, tools: list[Any]) -> list[dict[str, str]]:
    """构造教师模型输入。

    这里刻意不提供原始 assistant/refusal，避免教师模型被旧答案影响。
    """
    system = """You are a tool-calling data annotation expert.

Given a user request and a list of available tools, annotate the correct tool-calling behavior.

Return only one valid JSON object with this schema:

{
  "success": [
    {
      "request": "<brief user sub-request>",
      "tool": "<tool_name>",
      "call": "<BFCL style function call>"
    }
  ],
  "failed": [
    {
      "request": "<brief user sub-request>",
      "tool": "<tool_name|null>",
      "reason": "<miss_param|no_suitable_tool>",
      "missing_parameters": [],
      "provided_arguments": {},
      "description": "<brief local reason>"
    }
  ],
  "description": "<standalone overall summary>"
}

Rules:

1. Split the user request into explicit sub-requests before annotation.

2. Each explicit sub-request must be annotated separately.

3. If two sub-requests use the same tool with the same arguments, still create two separate success entries. Do not merge them.

4. Add a "success" item when a tool directly supports the sub-request and all required parameters are available.

5. The "call" field must use BFCL-style function-call format:
   [func_name(param1=value1, param2=value2)]

6. If a relevant tool exists but required parameters are missing, add a "failed" item with:
   reason = "miss_param"
   tool = the relevant tool name
   missing_parameters = the missing required parameter names
   provided_arguments = arguments extracted from the sub-request

7. If no available tool directly supports the sub-request, add a "failed" item with:
   reason = "no_suitable_tool"
   tool = null
   missing_parameters = []
   provided_arguments = {}

8. Do not invent required parameter values. Use default values only if the tool schema explicitly provides them.

9. Do not call a tool merely because it is topically related. The tool must directly perform the requested operation.

10. The top-level "description" must be a standalone natural-language summary of the whole annotation. It must be understandable without reading the "success" or "failed" fields. It should mention all successful sub-requests and all failed sub-requests, including missing parameters or no-suitable-tool reasons when applicable.

11. Always include "success", "failed", and "description".

12. Return only valid JSON. Do not include markdown, explanations, or extra text.
"""

    user_content = {
        "tools": tools,
        "user_request": user,
    }

    return [
        {
            "role": "system",
            "content": system,
        },
        {
            "role": "user",
            "content": json.dumps(user_content, ensure_ascii=False, indent=2),
        },
    ]


def create_teacher_client(base_url: str, api_key: str, timeout: int):
    """创建复用的 OpenAI-compatible client。

    之前每条样本都启动一个 curl 子进程，TLS 握手和进程启动开销会被
    放大 1000 多次。DeepSeek 示例已经使用 OpenAI SDK，这里跟它对齐。
    """
    from openai import OpenAI

    return OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=1)


def get_thread_teacher_client(base_url: str, api_key: str, timeout: int):
    """每个并发 worker 复用一个 client，减少重复初始化开销。"""
    cache_key = (base_url, api_key, timeout)
    if getattr(_THREAD_LOCAL, "teacher_client_key", None) != cache_key:
        _THREAD_LOCAL.teacher_client = create_teacher_client(base_url, api_key, timeout)
        _THREAD_LOCAL.teacher_client_key = cache_key
    return _THREAD_LOCAL.teacher_client


def call_teacher(
    client: Any,
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """调用 OpenAI-compatible chat completions endpoint。"""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("Teacher returned empty content.")
    return str(content)


def extract_json_object(text: str) -> dict[str, Any]:
    """从 teacher 输出中提取 JSON 对象。

    即使模型偶尔包了 ```json fence，也尽量恢复；恢复失败则交给失败分流。
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        value = json.loads(cleaned[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("Teacher output is not a JSON object.")
    return value


def normalize_teacher_annotation(value: dict[str, Any]) -> dict[str, Any]:
    """规范化 teacher JSON，只保留目标三字段。"""
    annotation = {
        "success": value.get("success", []),
        "failed": value.get("failed", []),
        "description": value.get("description", ""),
    }
    normalize_success_calls(annotation)
    return annotation


def normalize_success_calls(annotation: dict[str, Any]) -> None:
    """把 teacher 偶尔漏掉的 BFCL 外层方括号补回来。

    Prompt 已要求 `[func(...)]`，但真实输出里常见 `func(...)`。这种格式
    语义正确，只是少了 BFCL list 外壳；本地修复比丢弃重跑更稳。
    """
    success = annotation.get("success")
    if not isinstance(success, list):
        return
    for item in success:
        if not isinstance(item, dict):
            continue
        call = item.get("call")
        if not isinstance(call, str):
            continue
        stripped = call.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            item["call"] = stripped
        elif "(" in stripped and stripped.endswith(")"):
            item["call"] = f"[{stripped}]"


def normalize_annotation_against_tools(annotation: dict[str, Any], tools: list[Any]) -> None:
    """让 success.call 和 success.tool 保持一致。

    模型偶尔会把工具名 `Get Exercises by Body Part` 写成
    `GetExercisesByBodyPart(...)`。因为 `tool` 字段已经明确给出目标工具，
    这里把 call 里的函数名前缀改回工具名，避免被格式差异误伤。
    """
    tool_names = set(tool_index(tools))
    success = annotation.get("success")
    if not isinstance(success, list):
        return

    for item in success:
        if not isinstance(item, dict):
            continue
        tool_name = item.get("tool")
        call = item.get("call")
        if not isinstance(tool_name, str) or tool_name not in tool_names:
            continue
        if not isinstance(call, str) or bfcl_call_uses_known_tool(call, tool_names):
            continue

        stripped = call.strip()
        inner = stripped[1:-1].strip() if stripped.startswith("[") and stripped.endswith("]") else stripped
        if "(" not in inner or not inner.endswith(")"):
            continue
        _, args_text = inner.split("(", 1)
        item["call"] = f"[{tool_name}({args_text}]"


def bfcl_call_uses_known_tool(call: str, tool_names: set[str]) -> bool:
    """检查 BFCL call 字符串是否至少引用了一个已知工具名。"""
    return any(re.search(rf"(?<![A-Za-z0-9_.]){re.escape(name)}\s*\(", call) for name in tool_names)


def contains_old_parameter_assignment(text: str, old_param_names: set[str]) -> str | None:
    """检查 BFCL call 中是否在参数名位置残留旧参数名。

    不检查自然语言 description，因为像 `hashtag` 这类词既可能是旧参数名，
    也可能是普通业务词。这里只看 `name=` / `"name":` 这类结构化位置。
    """
    for name in old_param_names:
        escaped = re.escape(str(name))
        if re.search(rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])(?=\s*(?:=|:))", text):
            return str(name)
    return None


def validate_teacher_annotation(annotation: dict[str, Any], tools: list[Any], old_param_names: set[str]) -> list[str]:
    """校验 teacher 输出是否符合 success/failed 模板和扰乱后的 tools。"""
    errors: list[str] = []
    index = tool_index(tools)
    tool_names = set(index)
    success = annotation.get("success")
    failed = annotation.get("failed")
    description = annotation.get("description")

    if not isinstance(success, list):
        errors.append("success_not_list")
        success = []
    if not isinstance(failed, list):
        errors.append("failed_not_list")
        failed = []
    if not isinstance(description, str) or not description.strip():
        errors.append("description_missing")

    if not success and not failed:
        errors.append("empty_success_and_failed")

    for item_index, item in enumerate(success):
        if not isinstance(item, dict):
            errors.append(f"success_{item_index}_not_dict")
            continue
        request = item.get("request")
        tool_name = item.get("tool")
        call = item.get("call")
        if not isinstance(request, str) or not request.strip():
            errors.append(f"success_{item_index}_request_missing")
        if not isinstance(tool_name, str) or tool_name not in tool_names:
            errors.append(f"success_{item_index}_unknown_tool:{tool_name}")
        if not isinstance(call, str) or not call.strip():
            errors.append(f"success_{item_index}_call_missing")
        elif not (call.strip().startswith("[") and call.strip().endswith("]")):
            errors.append(f"success_{item_index}_call_not_bfcl_list")
        elif not bfcl_call_uses_known_tool(call, tool_names):
            errors.append(f"success_{item_index}_call_unknown_tool")
        else:
            old_name = contains_old_parameter_assignment(call, old_param_names)
            if old_name:
                errors.append(f"success_{item_index}_old_parameter_name_left:{old_name}")

    for item_index, item in enumerate(failed):
        if not isinstance(item, dict):
            errors.append(f"failed_{item_index}_not_dict")
            continue
        request = item.get("request")
        tool_name = item.get("tool")
        reason = item.get("reason")
        missing = item.get("missing_parameters")
        provided = item.get("provided_arguments")
        local_description = item.get("description")
        if not isinstance(request, str) or not request.strip():
            errors.append(f"failed_{item_index}_request_missing")
        if reason not in ALLOWED_FAILED_REASONS:
            errors.append(f"failed_{item_index}_invalid_reason:{reason}")
        if not isinstance(missing, list):
            errors.append(f"failed_{item_index}_missing_parameters_not_list")
            missing = []
        if not isinstance(provided, dict):
            errors.append(f"failed_{item_index}_provided_arguments_not_dict")
            provided = {}
        if not isinstance(local_description, str) or not local_description.strip():
            errors.append(f"failed_{item_index}_description_missing")

        if reason == "no_suitable_tool":
            if tool_name is not None:
                errors.append(f"failed_{item_index}_no_suitable_tool_requires_null_tool")
            if missing:
                errors.append(f"failed_{item_index}_no_suitable_tool_requires_empty_missing")
            if provided:
                errors.append(f"failed_{item_index}_no_suitable_tool_requires_empty_provided")
            continue

        if reason == "miss_param":
            if not isinstance(tool_name, str) or tool_name not in index:
                errors.append(f"failed_{item_index}_unknown_tool:{tool_name}")
                continue
            if not missing:
                errors.append(f"failed_{item_index}_miss_param_requires_missing")
            for name in missing:
                if str(name) in old_param_names:
                    errors.append(f"failed_{item_index}_old_missing_name:{name}")
            for key in provided:
                if str(key) in old_param_names:
                    errors.append(f"failed_{item_index}_old_provided_argument:{key}")

    return errors


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """写入 JSONL，方便后续人工排查。"""
    ensure_dir(str(path.parent))
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取已有 JSONL；文件不存在时返回空列表。"""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_existing_results(output_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """加载已落盘结果，用于断点续跑。"""
    return (
        read_jsonl(output_dir / "valid_annotations.jsonl"),
        read_jsonl(output_dir / "invalid_schema_cases.jsonl"),
        read_jsonl(output_dir / "failed_parse_cases.jsonl"),
    )


def rebuild_counters(
    valid_rows: list[dict[str, Any]],
    invalid_rows: list[dict[str, Any]],
    failed_parse_rows: list[dict[str, Any]],
) -> Counter[str]:
    """从已有结果重建计数器。"""
    counters: Counter[str] = Counter()
    for row in valid_rows:
        counters["valid_samples"] += 1
        counters["success_items"] += len(row.get("teacher_success", []))
        counters["failed_items"] += len(row.get("teacher_failed", []))
    counters["invalid_schema"] += len(invalid_rows)
    counters["failed_parse_or_request"] += len(failed_parse_rows)
    return counters


def refresh_existing_record(row: dict[str, Any]) -> dict[str, Any]:
    """用当前校验逻辑重新整理历史结果。

    这主要用于修复旧结果里 `call` 少了外层 `[]` 的样本。没有
    `teacher_json` 的 failed 样本无法本地修复，后续会重新请求。
    """
    teacher_json = row.get("teacher_json")
    if not isinstance(teacher_json, dict):
        return row

    annotation = normalize_teacher_annotation(teacher_json)
    tools = load_json_tools(str(row.get("tools", "")))
    normalize_annotation_against_tools(annotation, tools)
    parameter_mapping = row.get("parameter_mapping", {})
    if not isinstance(parameter_mapping, dict):
        parameter_mapping = {}
    old_param_names = set(parameter_mapping.keys())
    errors = validate_teacher_annotation(annotation, tools, old_param_names)

    refreshed = dict(row)
    refreshed.update(
        {
            "assistant": json.dumps(annotation, ensure_ascii=False),
            "teacher_json": annotation,
            "teacher_success": annotation["success"],
            "teacher_failed": annotation["failed"],
            "teacher_description": annotation["description"],
            "validation_errors": errors,
            "status": "invalid_schema" if errors else "valid",
        }
    )
    return refreshed


def repair_resume_rows(
    valid_rows: list[dict[str, Any]],
    invalid_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """resume 时复用已生成结果，并把可本地修复的 invalid 移到 valid。"""
    repaired_valid = [refresh_existing_record(row) for row in valid_rows]
    remaining_invalid: list[dict[str, Any]] = []
    repaired_count = 0

    for row in invalid_rows:
        refreshed = refresh_existing_record(row)
        if refreshed.get("status") == "valid":
            repaired_valid.append(refreshed)
            repaired_count += 1
        else:
            remaining_invalid.append(refreshed)

    return repaired_valid, remaining_invalid, repaired_count


def keep_canonical_resume_rows(
    valid_rows: list[dict[str, Any]],
    mode_by_index: dict[int, str],
) -> tuple[list[dict[str, Any]], int]:
    """只复用与当前扰动计划一致的历史 valid 样本。

    扰动模式按选中样本总数分配；如果曾经用 `--limit` 跑过 resume，
    少数 source_index 的 mode 可能和全量计划不一致。这里直接重跑它们，
    保证最终数据集的 60/20/10/10 比例和映射都来自同一份计划。
    """
    kept_rows: list[dict[str, Any]] = []
    stale_count = 0
    for row in valid_rows:
        try:
            source_index = int(row["source_index"])
        except (KeyError, TypeError, ValueError):
            stale_count += 1
            continue
        if row.get("ablation_mode") == mode_by_index.get(source_index):
            kept_rows.append(row)
        else:
            stale_count += 1
    return kept_rows, stale_count


def write_preview(path: Path, rows: list[dict[str, Any]], count: int = 10) -> None:
    """写入 markdown 预览，方便人工看 teacher 输入输出。"""
    ensure_dir(str(path.parent))
    lines = ["# Refusal Teacher Annotation Preview", ""]
    for index, row in enumerate(rows[:count], start=1):
        lines.extend(
            [
                f"## Sample {index}",
                "",
                f"- status: `{row.get('status')}`",
                f"- ablation_mode: `{row.get('ablation_mode')}`",
                f"- success_items: `{len(row.get('teacher_success', []))}`",
                f"- failed_items: `{len(row.get('teacher_failed', []))}`",
                f"- validation_errors: `{row.get('validation_errors', [])}`",
                "",
                "### User",
                "",
                "```text",
                row["user"],
                "```",
                "",
                "### Parameter Mapping",
                "",
                "```json",
                json.dumps(row["parameter_mapping"], ensure_ascii=False, indent=2),
                "```",
                "",
                "### Function Mapping",
                "",
                "```json",
                json.dumps(row.get("function_mapping", {}), ensure_ascii=False, indent=2),
                "```",
                "",
                "### Teacher Annotation",
                "",
                "```json",
                json.dumps(row.get("teacher_json", {}), ensure_ascii=False, indent=2),
                "```",
                "",
                "### Tools",
                "",
                "```json",
                json.dumps(json.loads(row["tools"]), ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def flush_case_files(
    output_dir: Path,
    dry_run_rows: list[dict[str, Any]],
    valid_rows: list[dict[str, Any]],
    invalid_rows: list[dict[str, Any]],
    failed_parse_rows: list[dict[str, Any]],
) -> None:
    """刷新中间结果，长任务中断时也能保留已完成样本。"""
    write_jsonl(output_dir / "dry_run_requests.jsonl", dry_run_rows)
    write_jsonl(output_dir / "valid_annotations.jsonl", valid_rows)
    write_jsonl(output_dir / "invalid_schema_cases.jsonl", invalid_rows)
    write_jsonl(output_dir / "failed_parse_cases.jsonl", failed_parse_rows)
    write_jsonl(
        output_dir / "success_cases.jsonl",
        [row for row in valid_rows if row.get("teacher_success")],
    )


def maybe_flush_progress(
    args: argparse.Namespace,
    output_dir: Path,
    processed: int,
    total: int,
    dry_run_rows: list[dict[str, Any]],
    valid_rows: list[dict[str, Any]],
    invalid_rows: list[dict[str, Any]],
    failed_parse_rows: list[dict[str, Any]],
    counters: Counter[str],
) -> None:
    """按配置刷新结果并打印进度。成功和失败路径都调用它。"""
    if args.flush_every <= 0 or processed % args.flush_every != 0:
        return
    flush_case_files(output_dir, dry_run_rows, valid_rows, invalid_rows, failed_parse_rows)
    print(
        json.dumps(
            {
                "processed": processed,
                "total": total,
                "valid": len(valid_rows),
                "invalid": len(invalid_rows),
                "failed": len(failed_parse_rows),
                "counters": dict(counters),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def build_base_record(
    index: int,
    row: dict[str, Any],
    helpers: Any,
    mode_by_index: dict[int, str],
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, str]], list[Any], set[str]]:
    """完成单条样本的本地扰动和 teacher prompt 构造。"""
    rng = random.Random(seed + index * 1009)
    ablation_mode = mode_by_index[index]
    perturbed = perturb_tools(row, helpers, rng, ablation_mode)
    messages = build_teacher_messages(str(row["user"]), perturbed["tools"])
    old_param_names = set(perturbed["parameter_mapping"].keys())

    base_record = {
        "source_index": index,
        "system": row["system"],
        "user": row["user"],
        "tools": perturbed["tools_text"],
        "tool_count": row["tool_count"],
        "tool_type": row["tool_type"],
        "refusal": row["refusal"],
        "ablation_mode": perturbed["ablation_mode"],
        "parameter_mapping": perturbed["parameter_mapping"],
        "function_mapping": perturbed["function_mapping"],
        "teacher_prompt": messages,
    }
    return base_record, messages, perturbed["tools"], old_param_names


def make_dry_run_record(
    index: int,
    row: dict[str, Any],
    helpers: Any,
    mode_by_index: dict[int, str],
    seed: int,
) -> dict[str, Any]:
    """生成 dry-run 记录。"""
    base_record, _, _, _ = build_base_record(index, row, helpers, mode_by_index, seed)
    base_record.update(
        {
            "status": "dry_run",
            "teacher_json": {},
            "teacher_success": [],
            "teacher_failed": [],
            "teacher_description": "",
            "validation_errors": [],
        }
    )
    return base_record


def request_teacher_annotation(
    index: int,
    row: dict[str, Any],
    helpers: Any,
    mode_by_index: dict[int, str],
    seed: int,
    model: str,
    base_url: str,
    api_key: str,
    timeout: int,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    """处理单条样本：本地扰动 -> 调 teacher -> 校验。"""
    base_record, messages, tools, old_param_names = build_base_record(index, row, helpers, mode_by_index, seed)

    try:
        client = get_thread_teacher_client(base_url, api_key, timeout)
        raw_output = call_teacher(
            client=client,
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        teacher_json = normalize_teacher_annotation(extract_json_object(raw_output))
        normalize_annotation_against_tools(teacher_json, tools)
    except Exception as error:
        failed = dict(base_record)
        failed.update({"status": "failed_parse_or_request", "error": str(error)})
        return failed

    errors = validate_teacher_annotation(teacher_json, tools, old_param_names)
    record = dict(base_record)
    record.update(
        {
            "assistant": json.dumps(teacher_json, ensure_ascii=False),
            "teacher_json": teacher_json,
            "teacher_success": teacher_json["success"],
            "teacher_failed": teacher_json["failed"],
            "teacher_description": teacher_json["description"],
            "validation_errors": errors,
        }
    )
    record["status"] = "invalid_schema" if errors else "valid"
    return record


def add_record_to_buckets(
    record: dict[str, Any],
    dry_run_rows: list[dict[str, Any]],
    valid_rows: list[dict[str, Any]],
    invalid_rows: list[dict[str, Any]],
    failed_parse_rows: list[dict[str, Any]],
    counters: Counter[str],
) -> None:
    """把处理结果放进对应 bucket，并更新计数。"""
    status = record.get("status")
    if status == "dry_run":
        dry_run_rows.append(record)
        counters["dry_run"] += 1
    elif status == "valid":
        valid_rows.append(record)
        counters["valid_samples"] += 1
        counters["success_items"] += len(record.get("teacher_success", []))
        counters["failed_items"] += len(record.get("teacher_failed", []))
    elif status == "invalid_schema":
        invalid_rows.append(record)
        counters["invalid_schema"] += 1
    else:
        failed_parse_rows.append(record)
        counters["failed_parse_or_request"] += 1


def main() -> int:
    """主入口。"""
    import os
    from datasets import Dataset, load_from_disk

    args = parse_args()
    helpers = load_ablation_helpers()
    output_dir = Path(args.output_dir)
    ensure_dir(str(output_dir))

    model = args.model or os.environ.get("TEACHER_MODEL") or os.environ.get("DEEPSEEK_MODEL") or DEFAULT_MODEL
    base_url = (
        args.base_url
        or os.environ.get("TEACHER_BASE_URL")
        or os.environ.get("DEEPSEEK_BASE_URL")
        or DEFAULT_BASE_URL
    )
    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("TEACHER_API_KEY")

    dataset = load_from_disk(args.dataset_path)
    selected_rows = select_refusal_rows(dataset, helpers, args.limit)
    mode_by_index = assign_ablation_modes(len(selected_rows), args.seed)

    dry_run_rows: list[dict[str, Any]] = []
    if args.resume and not args.dry_run:
        valid_rows, invalid_rows, failed_parse_rows = load_existing_results(output_dir)
        valid_rows, invalid_rows, repaired_count = repair_resume_rows(valid_rows, invalid_rows)
        valid_rows, stale_valid_count = keep_canonical_resume_rows(valid_rows, mode_by_index)
        retry_invalid_count = len(invalid_rows)
        retry_failed_count = len(failed_parse_rows)
        # 旧 invalid/failed 会重新请求；不继续保留旧记录，避免最终文件里
        # 同一个 source_index 同时出现失败旧记录和成功新记录。
        invalid_rows = []
        failed_parse_rows = []
        counters = rebuild_counters(valid_rows, invalid_rows, failed_parse_rows)
        if repaired_count or stale_valid_count or retry_invalid_count or retry_failed_count:
            print(
                json.dumps(
                    {
                        "resume_repaired_invalid_samples": repaired_count,
                        "resume_reprocess_stale_valid_samples": stale_valid_count,
                        "resume_retry_invalid_samples": retry_invalid_count,
                        "resume_retry_failed_samples": retry_failed_count,
                        "valid_after_repair": len(valid_rows),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    else:
        valid_rows = []
        failed_parse_rows = []
        invalid_rows = []
        counters = Counter()

    processed_indices = {
        int(row["source_index"])
        for row in valid_rows
        if "source_index" in row
    }
    if processed_indices:
        print(
            json.dumps(
                {
                    "resume": True,
                    "already_processed": len(processed_indices),
                    "valid": len(valid_rows),
                    "invalid": len(invalid_rows),
                    "failed": len(failed_parse_rows),
                    "counters": dict(counters),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    pending_items = [
        (index, row)
        for index, row in enumerate(selected_rows)
        if index not in processed_indices
    ]

    if args.dry_run:
        for done_count, (index, row) in enumerate(pending_items, start=1):
            record = make_dry_run_record(index, row, helpers, mode_by_index, args.seed)
            add_record_to_buckets(record, dry_run_rows, valid_rows, invalid_rows, failed_parse_rows, counters)
            maybe_flush_progress(
                args,
                output_dir,
                len(processed_indices) + done_count,
                len(selected_rows),
                dry_run_rows,
                valid_rows,
                invalid_rows,
                failed_parse_rows,
                counters,
            )
    else:
        if not model or not base_url:
            raise RuntimeError(
                "Teacher endpoint is not configured. Set TEACHER_MODEL and TEACHER_BASE_URL, "
                "or run with --dry-run first."
            )
        if not api_key:
            raise RuntimeError("DeepSeek API key is missing. Set DEEPSEEK_API_KEY or pass --api-key.")

        workers = max(1, int(args.workers))
        print(
            json.dumps(
                {
                    "pending_samples": len(pending_items),
                    "workers": workers,
                    "flush_every": args.flush_every,
                    "model": model,
                    "base_url": base_url,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_item = {
                executor.submit(
                    request_teacher_annotation,
                    index,
                    row,
                    helpers,
                    mode_by_index,
                    args.seed,
                    model,
                    base_url,
                    api_key,
                    args.timeout,
                    args.temperature,
                    args.max_tokens,
                ): (index, row)
                for index, row in pending_items
            }

            for done_count, future in enumerate(as_completed(future_to_item), start=1):
                index, row = future_to_item[future]
                try:
                    record = future.result()
                except Exception as error:
                    # 正常情况下 request_teacher_annotation 已经捕获异常；
                    # 这里兜底，避免单条任务把整批并发任务中断。
                    base_record, _, _, _ = build_base_record(index, row, helpers, mode_by_index, args.seed)
                    record = dict(base_record)
                    record.update({"status": "failed_parse_or_request", "error": str(error)})

                add_record_to_buckets(record, dry_run_rows, valid_rows, invalid_rows, failed_parse_rows, counters)

                if args.sleep:
                    time.sleep(args.sleep)

                maybe_flush_progress(
                    args,
                    output_dir,
                    len(processed_indices) + done_count,
                    len(selected_rows),
                    dry_run_rows,
                    valid_rows,
                    invalid_rows,
                    failed_parse_rows,
                    counters,
                )

    preview_rows = dry_run_rows if args.dry_run else valid_rows + invalid_rows + failed_parse_rows
    write_preview(output_dir / "preview.md", preview_rows)
    flush_case_files(output_dir, dry_run_rows, valid_rows, invalid_rows, failed_parse_rows)

    if valid_rows and not args.dry_run:
        dataset_rows = [
            {
                "system": row["system"],
                "user": row["user"],
                "assistant": row["assistant"],
                "tools": row["tools"],
                "tool_count": row["tool_count"],
                "tool_type": row["tool_type"],
                "refusal": row["refusal"],
                "ablation_mode": row["ablation_mode"],
                "parameter_mapping": json.dumps(row["parameter_mapping"], ensure_ascii=False),
                "function_mapping": json.dumps(row["function_mapping"], ensure_ascii=False),
                "teacher_success": json.dumps(row["teacher_success"], ensure_ascii=False),
                "teacher_failed": json.dumps(row["teacher_failed"], ensure_ascii=False),
                "teacher_description": row["teacher_description"],
            }
            for row in valid_rows
        ]
        output_dataset_path = output_dir / "dataset"
        if output_dataset_path.exists():
            import shutil

            shutil.rmtree(output_dataset_path)
        Dataset.from_list(dataset_rows).save_to_disk(str(output_dataset_path))
    else:
        output_dataset_path = None

    report = {
        "dataset_path": args.dataset_path,
        "output_dir": str(output_dir),
        "dry_run": args.dry_run,
        "limit": args.limit,
        "ablation_ratios": ABLATION_RATIOS,
        "selected_mode_counts": dict(Counter(mode_by_index.values())),
        "valid_mode_counts": dict(Counter(row.get("ablation_mode") for row in valid_rows)),
        "failed_mode_counts": dict(Counter(row.get("ablation_mode") for row in failed_parse_rows)),
        "invalid_mode_counts": dict(Counter(row.get("ablation_mode") for row in invalid_rows)),
        "selected_samples": len(selected_rows),
        "valid_samples": len(valid_rows),
        "invalid_schema_samples": len(invalid_rows),
        "failed_parse_samples": len(failed_parse_rows),
        "counters": dict(counters),
        "teacher_model": model,
        "teacher_base_url": base_url,
        "output_dataset_path": str(output_dataset_path) if output_dataset_path else None,
        "files": {
            "preview": str(output_dir / "preview.md"),
            "dry_run_requests": str(output_dir / "dry_run_requests.jsonl"),
            "valid_annotations": str(output_dir / "valid_annotations.jsonl"),
            "invalid_schema_cases": str(output_dir / "invalid_schema_cases.jsonl"),
            "failed_parse_cases": str(output_dir / "failed_parse_cases.jsonl"),
            "success_cases": str(output_dir / "success_cases.jsonl"),
        },
    }
    save_json(report, str(output_dir / "report.json"))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
