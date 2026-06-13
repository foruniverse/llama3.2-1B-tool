# Llama 3.2 1B Tool Calling SFT

本项目用于训练 `Llama-3.2-1B-Instruct` 的工具调用能力，当前主线保留：

1. 下载 ModelScope 模型。
2. 将 ToolACE 原始数据结构化。
3. 构造 baseline、ablation1、ablation2、ablation3 训练数据/训练配置。
4. 执行 LoRA SFT。
5. 合并 LoRA 并做 BFCL/SGLang 兼容化。

项目框架和实验参数见 [docs/PROJECT_FLOW.md](docs/PROJECT_FLOW.md)，逐步执行命令见 [EXECUTION_GUIDE.md](EXECUTION_GUIDE.md)，环境复现见 [ENVIRONMENT.md](ENVIRONMENT.md)。

常用验证：

```bash
uv sync --locked
uv run python -m py_compile scripts/*.py scripts/utils/*.py
```
