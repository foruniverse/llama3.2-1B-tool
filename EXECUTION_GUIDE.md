# 执行指南

本项目当前只保留 SFT 主流程：下载模型、准备数据、执行 SFT 训练。GRPO、本地 evaluate、本地 inference、本地 BFCL 评测脚本已移除。

## 主流程

### 1. 准备环境

推荐使用 `uv`：

```bash
uv sync --locked
```

如需设置缓存目录：

```bash
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=./models/huggingface_cache
export MODELSCOPE_CACHE=./models/modelscope_cache
```

### 2. 下载模型

```bash
uv run python scripts/1_download_model.py
```

关键逻辑：

- 入口脚本：`scripts/1_download_model.py`
- 输出目录：`models/pretrained/`
- 日志要求：记录模型 ID、缓存目录、最终保存路径、下载异常。

### 3. 准备结构化数据集

```bash
uv run python scripts/2_prepare_dataset.py
```

关键逻辑：

- 入口函数：`prepare_dataset()`
- 核心脚本：`scripts/2_prepare_dataset.py`
- 输入数据集：`data.dataset_name`
- 输出结构化数据集：`data.structured_dataset_path`，默认 `data/tool_ace_processed`
- 抽样 prompt：`data/tool_ace_processed/sample_prompts.txt`
- 无法提取 tools 样本：`data/tool_ace_processed/tool_extract_failed.jsonl`
- 0 工具样本：`data/tool_ace_processed/zero_tool.jsonl`
- 统计报告：`data/tool_ace_processed/dataset_report.md`、`data/tool_ace_processed/dataset_report.json`

### 4. 准备消融实验 1 训练数据

```bash
uv run python scripts/2_prepare_data_abalation1.py
```

关键逻辑：

- 入口函数：`prepare_ablation_data()`
- 配置文件：`configs/sft_config.yaml`
- 输入数据集：`data.structured_dataset_path`
- 输出目录：`data.processed_train_path`、`data.processed_validation_path`
- 日志文件：`logs/prepare_data_abalation1.log`
- 统计报告：`data/ablation1/metadata.json`
- 预览文件：`data/ablation1/parameter_ablation_preview.txt`
- 日志必须包含固定 split 数量、train/test 扰动比例、实际扰动数量、最终保存路径。

结构化样本字段：

```text
system
user
assistant
tools
tool_count
tool_type
refusal
```

### 5. 准备消融实验 2 训练数据

```bash
uv run python scripts/2_prepare_data_abalation2.py
```

关键逻辑：

- 入口函数：`prepare_ablation_data()`
- 配置文件：`configs/sft_config.yaml`，通过 `experiments.active: ablation2` 读取 profile。
- 输入数据集：`data.structured_dataset_path`，默认 `data/tool_ace_processed`
- Refusal 输入：`data.refusal_teacher_dataset_path`，默认 `data/ablation_refusal_teacher_deepseek/dataset`
- 输出目录：`data.processed_train_path`、`data.processed_validation_path`
- 日志文件：`logs/prepare_data_abalation2.log`
- 统计报告：`data/ablation2/metadata.json`
- 预览文件：`data/ablation2/ablation2_preview.txt`

实验定义：

- 先构造未扰动全集 `a+b+c`：
  - `a`：ToolACE 中 `tool_count>=1`、`tool_type=json`、`refusal=0` 的非 refusal 样本。
  - `b`：DeepSeek teacher refusal description，替换原 `refusal=1`、`tool_count>0` 的部分。
  - `c`：ToolACE 中 `refusal=1` 且 `tool_count=0` 的原始 refusal 样本。
- 在未扰动全集上固定抽取 5% 作为 eval。
- train 内只对 `a` 类可扰动样本做 30% 扰动，其中 15% 扰乱参数名、10% 扰乱函数名、5% 同时扰乱参数名和函数名。
- eval 内只对 `a` 类可扰动样本做 50% 扰动，且全部同时扰乱参数名和函数名。
- 可扰动样本需同时存在可改名参数和函数名。
- 所有扰动样本全部调用 `rewrite_schema()` 打乱 schema 字段、`properties` 参数顺序和 `required` 顺序。
- 函数名扰动由 `rewrite_function_names()` 修改 tools，并由 `rewrite_assistant_function_names()` 同步 assistant function-call。
- 参数名扰动复用 ablation1 的 `rewrite_schema()` 和 `rewrite_assistant_parameters()`。
- DeepSeek teacher refusal 数据进入 prompt 时仍是 `system + tools + user`；`teacher_description` 或 JSON 顶层 `description` 只作为 assistant response。
- 最终 train/eval 数据分别组装 prompt 和 tokenization。

### 6. 执行 SFT 训练

```bash
uv run python scripts/3_sft_training.py
```

小步 smoke test：

```bash
SFT_MAX_STEPS=1 uv run python scripts/3_sft_training.py
```

关键逻辑：

- 入口函数：`train_sft()`
- 配置文件：`configs/sft_config.yaml`，通过 `load_experiment_config()` 应用 `experiments.active` profile。
- 训练数据：`data.processed_train_path`
- 验证数据：`data.processed_validation_path`
- 输出目录：`training.output_dir`
- 日志文件：`logs/sft_training.log`
- 日志必须包含设备信息、attention backend、LoRA 参数、batch size、gradient accumulation、学习率、checkpoint 保存路径。

查看训练日志：

```bash
tail -f logs/sft_training.log
```

查看 TensorBoard：

```bash
tensorboard --logdir=./logs/tensorboard --port=6006
```

## 辅助流程

### 合并 LoRA

```bash
uv run python scripts/8_lora_merge.py \
  --lora-path <checkpoint> \
  --output-dir <merged-model>
```

关键逻辑：

- 入口脚本：`scripts/8_lora_merge.py`
- 兼容处理函数：`normalize_model_dir()`
- 输入：LoRA checkpoint 目录。
- 输出：合并后的模型目录。
- 日志/输出必须说明输入 checkpoint、基础模型路径、输出目录、兼容字段修改结果。

### 修复模型配置兼容性

```bash
uv run python scripts/9_make_sglang_compatible.py <merged-model>
```

关键逻辑：

- 入口脚本：`scripts/9_make_sglang_compatible.py`
- 核心函数：`normalize_model_dir()`
- 处理对象：模型目录中的 `config.json`、tokenizer metadata。
- 处理内容：RoPE、dtype、tokenizer metadata 等后端兼容字段。

### 数据后处理

```bash
uv run python scripts/10_shuffle_tool_schema.py
uv run python scripts/11_filter_retokenize_processed.py
```

这些脚本用于实验数据处理。运行前必须确认输入输出路径，避免覆盖主流程数据。

## 配置文件

主配置只保留：

```text
configs/sft_config.yaml
```

`configs/sft_config.yaml` 使用通用参数 + 实验 profile。脚本通过
`scripts.utils.utils.load_experiment_config()` 读取配置，并应用
`experiments.active` 指向的 profile：

```yaml
experiments:
  active: "ablation2"
  profiles:
    baseline:
      training:
        output_dir: "./models/checkpoints/sft_toolace_baseline"
      data:
        processed_train_path: "./data/train/train_baseline"
        processed_validation_path: "./data/eval/eval_baseline"
    ablation1:
      training:
        output_dir: "./models/checkpoints/sft_ablation1_parameter_shuffle_rename"
      data:
        processed_train_path: "./data/train/train_abalation1_parameter"
        processed_validation_path: "./data/eval/eval_abalation1_parameter"
    ablation2:
      training:
        output_dir: "./models/checkpoints/sft_ablation2_param_func_refusal_teacher"
      data:
        refusal_teacher_dataset_path: "./data/ablation_refusal_teacher_deepseek/dataset"
        processed_train_path: "./data/train/train_abalation2_param_func_refusal_teacher"
        processed_validation_path: "./data/eval/eval_abalation2_param_func_refusal_teacher"
```

常改字段：

```yaml
training:
  output_dir: "./models/checkpoints/sft_ablation2_param_func_refusal_teacher"
  num_train_epochs: 1
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 24
  learning_rate: 1.0e-4

lora:
  lora_r: 16
  lora_alpha: 32
  lora_target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

data:
  dataset_name: "Team-ACE/ToolACE"
  structured_dataset_path: "./data/tool_ace_processed"
  refusal_teacher_dataset_path: "./data/ablation_refusal_teacher_deepseek/dataset"
  processed_train_path: "./data/train/train_abalation2_param_func_refusal_teacher"
  processed_validation_path: "./data/eval/eval_abalation2_param_func_refusal_teacher"
```

`max_seq_length` 只使用 `training.max_seq_length`，不再做 percentile 自动选择。

## 日志落地规范

所有长流程脚本必须写入 `logs/`。新增脚本优先使用：

```python
from scripts.utils.utils import setup_logging

logger = setup_logging("./logs/<script_name>.log")
```

日志至少记录：

- 配置文件路径。
- 输入模型、数据或 checkpoint 路径。
- 输出目录或结果文件路径。
- 样本数量、token 长度、训练步数等关键统计。
- 数据预处理的结构化字段分布，例如 `tool_count`、`tool_type`。
- 设备信息和关键训练参数。
- 异常堆栈或明确错误原因。

## 常见问题

### 显存不足

优先调整 `configs/sft_config.yaml`：

```yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 48
```

### 强制 attention backend

```bash
ATTN_IMPLEMENTATION=sdpa uv run python scripts/3_sft_training.py
ATTN_IMPLEMENTATION=flash_attention_2 uv run python scripts/3_sft_training.py
```

### 检查 CUDA

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```

### Codex 工具环境中的 uv

如果 `uv` 在 Codex 工具命令里不在 PATH，可使用本机绝对路径：

```bash
/home/yanyan/.local/bin/uv run python scripts/2_prepare_data_abalation1.py
```

## 轻量验证

整理代码或修改脚本后，先运行：

```bash
uv run python -m py_compile scripts/*.py scripts/utils/*.py
```

如果修改了训练脚本，再运行：

```bash
SFT_MAX_STEPS=1 uv run python scripts/3_sft_training.py
```
