# 任务：ToolACE 数据集预处理

当前 ToolACE 预处理全部集中在 `scripts/2_prepare_dataset.py`，不再使用 `scripts/data_processors/` 抽象。

## 处理流程

1. 读取原始 `Team-ACE/ToolACE`。
2. 将每条样本的 `conversations` 裁剪为第一轮 `user + assistant`。
3. 从原始 `system` 中拆出非工具 system prompt 和 tools。
4. 统计工具数量：`zero_tool`、`single_tool`、`multiple_tool`。
5. 统计工具定义类型：`json` 或 `non_json`。
6. 手动拼接 Llama prompt：

```text
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>

system + tools
<|eot_id|>
<|start_header_id|>user<|end_header_id|>

user
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

assistant
<|eot_id|>
```

不使用 `tokenizer.apply_chat_template()`，避免 tokenizer 模板自动添加日期等额外内容。

7. 统计完整 prompt token 长度，删除长度大于 `training.max_seq_length` 的样本。
8. 根据 assistant 回复判断是否拒绝调用：`refusal=1` 表示自然语言拒绝/缺参数说明，`refusal=0` 表示函数调用。
9. 保存新的 dataset。

## 最终字段

保存到 `data/tool_ace_processed` 的 dataset 只包含：

```text
system
user
assistant
tools
tool_count
tool_type
refusal
```

## tools 提取注意事项

- JSON 工具定义只在看起来像工具 schema 时才接受。
- 显式空 JSON 工具列表 `[]` 视为 `tool_type=json` 且 `tool_count=0`。
- 参数内部的 `required: []`、`enum: []` 不能被误判为顶层 tools。
- 非 JSON 工具定义尽量从 HTML table、XML-like、LaTeX table、`tool_name:` 块中提取。
- 非 JSON 提取结果不包装成 dict，直接把提取出的工具片段字符串放进 `tools`。
- 无法提取 tools 的样本写入 `data/tool_ace_processed/tool_extract_failed.jsonl`。

## 产物

```text
data/tool_ace_processed
data/tool_ace_processed/dataset_report.md
data/tool_ace_processed/dataset_report.json
data/tool_ace_processed/sample_prompts.txt
data/tool_ace_processed/tool_extract_failed.jsonl
data/tool_ace_processed/zero_tool.jsonl
```

## 验证

```bash
.venv/bin/python -m py_compile scripts/*.py scripts/utils/*.py
/home/yanyan/.local/bin/uv run python scripts/2_prepare_dataset.py
```
