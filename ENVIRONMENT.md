# 可复现环境

本项目使用 `uv` 管理 Python 环境，并提交 `uv.lock` 锁定依赖。

## Python

Python 版本固定为 `3.11.14`：

- `.python-version`
- `pyproject.toml`

创建环境：

```bash
uv python install 3.11.14
uv sync --locked
```

`requirements.txt` 仅保留为 pip 备用依赖文件，服务器复现优先使用 `uv sync --locked`。

## 依赖分组

```bash
uv sync --locked --group monitor
uv sync --locked --group notebook
```

## 运行约定

所有脚本通过 `uv run` 执行：

```bash
uv run python scripts/1_download_model.py
uv run python scripts/2_prepare_dataset.py
uv run python scripts/2_prepare_data_ablation1.py
uv run python scripts/2_prepare_data_ablation2.py
uv run python scripts/3_sft_training.py
```

`configs/sft_config.yaml` 通过 `experiments.active` 切换实验 profile。当前默认 profile 为 `ablation3`：

- 结构化数据：`data/tool_ace_processed`
- teacher refusal 数据：`data/ablation_refusal_teacher_deepseek/dataset`
- 训练数据：`data/train/train_ablation2_param_func_refusal_teacher`
- 验证数据：`data/eval/eval_ablation2_param_func_refusal_teacher`
- 模型输出：`models/checkpoints/sft_ablation3_param_func_refusal_teacher_r32`
- LoRA：`r=32`、`alpha=64`

## CUDA 和 attention backend

训练默认需要 CUDA。`scripts/3_sft_training.py` 会自动选择 attention backend：

- 安装 `flash-attn` 时使用 `flash_attention_2`。
- 未安装时回退到 PyTorch `sdpa`。

可手动指定：

```bash
ATTN_IMPLEMENTATION=sdpa uv run python scripts/3_sft_training.py
ATTN_IMPLEMENTATION=flash_attention_2 uv run python scripts/3_sft_training.py
```

## 日志

模型下载、数据处理、训练、模型合并和格式转换都写入 `logs/`。

当前主要日志：

- `logs/download_model.log`
- `logs/prepare_dataset.log`
- `logs/prepare_data_ablation1.log`
- `logs/prepare_data_ablation2.log`
- `logs/sft_training.log`
- `logs/lora_merge.log`
- `logs/make_sglang_compatible.log`
- `logs/shuffle_tool_schema.log`
- `logs/filter_retokenize_processed.log`

新增长流程脚本时使用：

```python
from scripts.utils.utils import setup_logging

logger = setup_logging("./logs/<script_name>.log")
```

API key 只通过环境变量传入，例如 `DEEPSEEK_API_KEY`，不要写入源码或文档示例。
