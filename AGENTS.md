# AGENTS.md

本仓库用于训练 `Llama-3.2-1B-Instruct` 的工具调用能力。代理工作时保持流程简单：模型处理、数据处理、训练执行都必须有日志落地，便于复盘。

## 核心流程

```bash
uv sync --locked
uv run python scripts/1_download_model.py
uv run python scripts/2_prepare_dataset.py
uv run python scripts/2_prepare_data_ablation1.py
uv run python scripts/2_prepare_data_ablation2.py
uv run python scripts/3_sft_training.py
uv run python scripts/8_lora_merge.py --lora-path <checkpoint> --output-dir <merged-model>
uv run python scripts/9_make_sglang_compatible.py <merged-model>
```

辅助数据脚本：

```bash
uv run python scripts/10_shuffle_tool_schema.py
uv run python scripts/11_filter_retokenize_processed.py
```

仓库不再保留一键 `run_pipeline.sh`。不要临时重建并直接串行跑完整流程，除非确认 GPU、数据、模型下载和训练时间都可用。

## 日志要求

- 所有模型下载、数据处理、训练、模型合并、格式转换脚本都要写入 `logs/`。
- 新脚本优先使用 `scripts.utils.utils.setup_logging("<log_path>")`。
- 日志必须能定位输入、输出和关键配置，至少包含：
  - 读取的配置文件路径。
  - 输入模型、数据或 checkpoint 路径。
  - 输出目录或结果文件路径。
  - 样本数量、token 长度、训练步数等关键统计。
  - 异常堆栈或明确错误原因。
- 处理模型文件时，记录被修改的关键字段，例如 `config.json` 中的 RoPE、dtype、tokenizer metadata。
- 处理数据时，记录结构化解析结果、`tool_count`、`tool_type`、过滤、shuffle、retokenize、截断比例和最终保存路径。
- 执行训练时，记录设备信息、attention backend、LoRA 参数、batch size、gradient accumulation、学习率、checkpoint 保存位置。

## 修改原则

- 实验参数优先放在 `configs/sft_config.yaml`，不要散落硬编码。
- 不随意改变 chat template、label mask、随机种子、停止 token；这些会直接影响工具调用格式。
- 大文件和运行产物不要提交：`models/`、`data/`、`logs/`、`eval/`、`wandb/`、`.venv/`。
- 保留用户已有改动，不回滚无关文件。

## 流程变更落地

修改流程后必须同步更新对应文件，写清楚“当前真实流程”，不要只改代码：

- `EXECUTION_GUIDE.md`：更新可执行命令、输入输出路径、日志路径。
- `ENVIRONMENT.md`：更新环境、依赖、运行方式。
- `configs/*.yaml`：更新实际使用的参数和路径。
- `docs/`：记录问题复盘或兼容性结论。
- 本文件：更新代理必须遵守的流程约束。

落地内容要包含关键函数、类或代码片段名称。例如：`prepare_data()`、`train_sft()`、`normalize_sglang_files()`，以及它们读写的文件路径。

数据预处理改动必须说明是否影响：

- `scripts/2_prepare_dataset.py`
- `scripts/2_prepare_data_ablation1.py`
- `scripts/2_prepare_data_ablation2.py`
- `data/tool_ace_processed`
- `data/tool_ace_processed/sample_prompts.txt`
- `data/tool_ace_processed/dataset_report.md`
- `data/tool_ace_processed/dataset_report.json`
- `data/ablation1/metadata.json`
- `data/ablation2/metadata.json`
- `data/ablation_refusal_teacher_deepseek/dataset`
- `data/train/train_ablation2_param_func_refusal_teacher`
- `data/eval/eval_ablation2_param_func_refusal_teacher`
- `data/processed/metadata.json`

## 验证

轻量验证优先：

```bash
uv run python -m py_compile scripts/*.py scripts/utils/*.py
```

涉及训练时，优先使用最小步数 smoke test：

```bash
SFT_MAX_STEPS=1 uv run python scripts/3_sft_training.py
```

如果因为 GPU、模型或数据不可用而无法验证，最终回复必须明确说明。
