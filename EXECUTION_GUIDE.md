# 执行指南

本指南记录当前真实可执行流程。不要直接串行跑完整训练链路，除非已经确认 GPU、模型、数据和训练时间都可用。

## 1. 环境

```bash
uv sync --locked
```

可选缓存环境变量：

```bash
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=./models/huggingface_cache
export MODELSCOPE_CACHE=./models/modelscope_cache
```

## 2. 下载模型

```bash
uv run python scripts/1_download_model.py
```

- 入口函数：`download_model()`
- 配置来源：脚本内 `model_id=AI-ModelScope/Llama-3.2-1B-Instruct`
- 输出目录：`models/pretrained/`
- 日志：`logs/download_model.log`

## 3. 结构化 ToolACE 数据

```bash
uv run python scripts/2_prepare_dataset.py
```

- 入口函数：`prepare_dataset()`
- 关键函数：`first_user_assistant()`、`find_tool_json_block()`、`split_system_and_tools()`、`build_prompt()`
- 配置：`configs/sft_config.yaml`
- 输入：`data.dataset_name`，默认 `Team-ACE/ToolACE`
- 输出：`data.structured_dataset_path`，默认 `data/tool_ace_processed`
- 报告：`data/tool_ace_processed/dataset_report.md`、`data/tool_ace_processed/dataset_report.json`
- 预览：`data/tool_ace_processed/sample_prompts.txt`
- 日志：`logs/prepare_dataset.log`

## 4. 准备消融实验数据

### baseline

baseline 使用结构化后的 ToolACE 样本，不做工具 schema 或 assistant 内容扰动。当前仓库没有单独 baseline 数据脚本；如需训练 baseline，需要在 `configs/sft_config.yaml` 将 `experiments.active` 设为 `baseline`，并准备 `data/train/train_baseline`、`data/eval/eval_baseline`。

### ablation1

```bash
uv run python scripts/2_prepare_data_ablation1.py
```

- 入口函数：`prepare_ablation_data()`
- 输入：`data.structured_dataset_path`
- 输出：`data/train/train_ablation1_parameter`、`data/eval/eval_ablation1_parameter`
- 报告：`data/ablation1/metadata.json`
- 预览：`data/ablation1/parameter_ablation_preview.txt`
- 日志：`logs/prepare_data_ablation1.log`
- 扰动：train 内 20% 可扰动样本，eval 内 50% 可扰动样本；打乱 schema 字段、参数顺序、`required` 顺序，并把带 description 的参数改名为 `param_XXXX`。

### refusal teacher 辅助数据

```bash
uv run python scripts/4_analyze_refusal_ablation.py
uv run python scripts/5_generate_refusal_teacher_actions.py --limit 20 --dry-run
DEEPSEEK_API_KEY=<key> uv run python scripts/5_generate_refusal_teacher_actions.py
```

- 分析脚本输出：`data/ablation_refusal_analysis/refusal_ablation_analysis.md`
- teacher 输出：`data/ablation_refusal_teacher_deepseek/`
- 注意：API key 只通过环境变量传入，不写入代码。

### ablation2

```bash
uv run python scripts/2_prepare_data_ablation2.py
```

- 入口函数：`prepare_ablation_data()`
- 关键函数：`load_ablation1_helpers()`、`normalize_refusal_teacher_dataset()`、`apply_non_refusal_ablation()`
- 输入：`data/tool_ace_processed`、`data/ablation_refusal_teacher_deepseek/dataset`
- 输出：`data/train/train_ablation2_param_func_refusal_teacher`、`data/eval/eval_ablation2_param_func_refusal_teacher`
- 报告：`data/ablation2/metadata.json`
- 预览：`data/ablation2/ablation2_preview.txt`
- 日志：`logs/prepare_data_ablation2.log`
- 扰动：train 内 15% 参数名、10% 函数名、5% 参数+函数；eval 内 50% 参数+函数。

### ablation3

ablation3 不新增数据处理脚本，复用 ablation2 的 tokenized train/eval 数据，只在训练配置中把 LoRA 从 `r=16, alpha=32` 提升为 `r=32, alpha=64`。

## 5. 训练

```bash
uv run python scripts/3_sft_training.py
```

小步 smoke test：

```bash
SFT_MAX_STEPS=1 uv run python scripts/3_sft_training.py
```

- 入口函数：`train_sft()`
- 配置：`configs/sft_config.yaml`
- profile 切换：`experiments.active`
- 输入：`data.processed_train_path`、`data.processed_validation_path`
- 输出：`training.output_dir`
- 日志：`logs/sft_training.log`

当前默认 active profile 是 `ablation3`：

- train：`data/train/train_ablation2_param_func_refusal_teacher`
- eval：`data/eval/eval_ablation2_param_func_refusal_teacher`
- output：`models/checkpoints/sft_ablation3_param_func_refusal_teacher_r32`
- LoRA：`r=32`、`alpha=64`

## 6. 合并 LoRA

```bash
uv run python scripts/8_lora_merge.py \
  --lora-path <checkpoint> \
  --output-dir <merged-model>
```

- 入口函数：`merge_lora()`
- 输入：base model、LoRA checkpoint
- 输出：合并后的模型目录
- 兼容处理：默认调用 `normalize_model_dir()`
- 日志：`logs/lora_merge.log`

## 7. BFCL/SGLang 兼容化

```bash
uv run python scripts/9_make_sglang_compatible.py <merged-model>
```

- 入口函数：`main()`
- 核心函数：`normalize_model_dir()`、`normalize_config()`、`normalize_tokenizer_config()`
- 修改字段：`rope_parameters -> rope_scaling/rope_theta`、`dtype -> torch_dtype`、`tokenizer_class -> PreTrainedTokenizerFast`
- 日志：`logs/make_sglang_compatible.log`

## 8. 辅助数据处理

```bash
uv run python scripts/10_shuffle_tool_schema.py
uv run python scripts/11_filter_retokenize_processed.py --source <dataset> --output <dataset>
```

运行前确认输入输出路径，避免覆盖主流程数据。

## 9. 轻量验证

```bash
uv run python -m py_compile scripts/*.py scripts/utils/*.py
```

涉及训练改动时优先跑：

```bash
SFT_MAX_STEPS=1 uv run python scripts/3_sft_training.py
```
