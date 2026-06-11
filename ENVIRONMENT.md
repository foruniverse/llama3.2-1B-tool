# 可复现环境

本项目使用 `uv` 管理 Python 环境，并提交 `uv.lock` 锁定依赖。

## 环境创建

安装 `uv`：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

创建锁定环境：

```bash
uv python install 3.11.14
uv sync --locked
```

Python 版本固定为 `3.11.14`，来源：

- `.python-version`
- `pyproject.toml`

`requirements.txt` 仅作为 pip 备用依赖文件。服务器复现优先使用：

```bash
uv sync --locked
```

## 运行命令

所有项目脚本优先通过 `uv run` 执行：

```bash
uv run python scripts/1_download_model.py
uv run python scripts/2_prepare_dataset.py
uv run python scripts/2_prepare_data_abalation1.py
uv run python scripts/2_prepare_data_abalation2.py
uv run python scripts/3_sft_training.py
```

`configs/sft_config.yaml` 通过 `experiments.active` 切换实验 profile。当前默认
为 `ablation2`，会读取：

- 结构化数据：`data/tool_ace_processed`
- DeepSeek refusal teacher 数据：`data/ablation_refusal_teacher_deepseek/dataset`
- 训练输出数据：`data/train/train_abalation2_param_func_refusal_teacher`
- 验证输出数据：`data/eval/eval_abalation2_param_func_refusal_teacher`
- 模型输出目录：`models/checkpoints/sft_ablation2_param_func_refusal_teacher`

可选工具组：

```bash
uv sync --locked --group monitor
uv sync --locked --group notebook
```

## CUDA 和 attention backend

训练默认需要 CUDA。SFT 脚本会自动选择 attention backend：

- 安装 `flash-attn` 时使用 `flash_attention_2`。
- 未安装时回退到 PyTorch `sdpa`。

也可以手动指定：

```bash
ATTN_IMPLEMENTATION=sdpa uv run python scripts/3_sft_training.py
ATTN_IMPLEMENTATION=flash_attention_2 uv run python scripts/3_sft_training.py
```

## 日志约定

模型下载、数据处理、训练、模型合并和配置转换都必须写入 `logs/`。

当前主要日志：

- `logs/prepare_dataset.log`
- `logs/prepare_data_abalation1.log`
- `logs/prepare_data_abalation2.log`
- `logs/sft_training.log`
- `logs/tensorboard/`

新增长流程脚本时使用：

```python
from scripts.utils.utils import setup_logging

logger = setup_logging("./logs/<script_name>.log")
```
