# BFCL/SGLang 评测问题复盘

## 问题现象

使用 BFCL 的 SGLang 后端评测 `models/sft-555` 时，`simple_python_0` 输出了自然语言解题过程：

```text
The area of a triangle can be calculated using the formula...
```

但在本项目中用 Transformers 直接加载同一个 `models/sft-555`，喂入同一条 BFCL `formatted_prompt`，输出是正确的函数调用：

```text
[calculate_triangle_area(base=10, height=5)]<|eot_id|>
```

这说明问题不在训练权重本身，而在 BFCL/SGLang 的加载或推理链路。

## 排查过程

1. 排除旧结果缓存

   BFCL 在不加 `--allow-overwrite` 时会复用已有 result 文件。重新使用新目录和 `--allow-overwrite` 后，问题仍然存在，因此旧结果不是最终原因。

2. 对比输入 prompt

   BFCL 的 Llama handler 手写 chat template，没有直接使用 `models/sft-555/chat_template.jinja`。本地手动 Transformers smoke test 使用 tokenizer 的 `apply_chat_template`，会额外插入日期和知识截止信息。

   但进一步验证发现，Transformers 直接加载 `models/sft-555` 时，无论喂 BFCL prompt 还是 tokenizer prompt，都能输出正确函数调用。因此 prompt 差异不是这次低分的主因。

3. 直接测试 SGLang OpenAI completion API

   手动启动 SGLang，加载同一个 `models/sft-555`，调用 `/v1/completions` 并传入 BFCL 第一条 prompt。结果复现了 BFCL 的错误自然语言输出。

   这把问题缩小到 SGLang 对模型目录的加载/推理行为。

4. 检查模型 config

   `models/sft-555/config.json` 由 Transformers 5.9 保存，关键字段是：

   ```json
   "rope_parameters": {
     "factor": 32.0,
     "high_freq_factor": 4.0,
     "low_freq_factor": 1.0,
     "original_max_position_embeddings": 8192,
     "rope_theta": 500000.0,
     "rope_type": "llama3"
   },
   "dtype": "bfloat16"
   ```

   BFCL 环境中的 SGLang 使用 Transformers 4.53.2。该环境读取这个 config 时没有正确识别 `rope_parameters`，导致 Llama 3.2 的 RoPE 配置没有按预期生效。

5. 验证修复

   临时创建兼容目录，仅把 config 改成 SGLang/Transformers 4.x 能识别的字段：

   ```json
   "rope_scaling": {
     "factor": 32.0,
     "high_freq_factor": 4.0,
     "low_freq_factor": 1.0,
     "original_max_position_embeddings": 8192,
     "rope_type": "llama3"
   },
   "rope_theta": 500000.0,
   "torch_dtype": "bfloat16"
   ```

   SGLang 重新加载后，同一条 prompt 输出正确函数调用：

   ```text
   [calculate_triangle_area(base=10, height=5)]
   ```

## 根因

根因是模型导出环境和评测后端环境的 config 字段不兼容：

- Transformers 5.9 保存：`rope_parameters`、`dtype`
- BFCL/SGLang 环境识别：`rope_scaling`、`torch_dtype`

SGLang 没有正确应用 Llama 3.2 的 RoPE 配置，导致同一份权重在 SGLang 后端和 Transformers 直接推理下行为不一致。

## 修复方案

新增脚本：

```bash
python scripts/9_make_sglang_compatible.py models/sft-555
```

该脚本会把：

```text
rope_parameters -> rope_scaling
dtype -> torch_dtype
```

当前 `scripts/8_lora_merge.py` 也已经集成这个步骤。之后推荐用合并脚本直接生成可评测模型：

```bash
python scripts/8_lora_merge.py \
  --lora-path models/checkpoints/sft/checkpoint-555 \
  --output-dir models/sft-555
```

如果只想修复已有模型：

```bash
python scripts/9_make_sglang_compatible.py models/sft-555
```

## 重新评测命令

```bash
cd /home/yanyan/project/BFCL/gorilla/berkeley-function-call-leaderboard

TMPDIR=/tmp bfcl generate \
  --model meta-llama/Llama-3.2-1B-Instruct-FC \
  --test-category simple_python \
  --backend sglang \
  --local-model-path /home/yanyan/project/llama3.2-1B-tool/models/sft-555 \
  --gpu-memory-utilization 0.75 \
  --num-gpus 1 \
  --include-input-log \
  --allow-overwrite \
  --result-dir /home/yanyan/project/llama3.2-1B-tool/eval/sft-555-fixed
```

## 启发

- 评测链路要做单样本 smoke test。先验证 `simple_python_0`，再跑完整 400 条。
- 不同推理后端不是纯替换关系。Transformers 正确不代表 SGLang/vLLM 一定正确。
- 模型目录里的 `config.json` 是评测可复现性的一部分，尤其是 RoPE、dtype、tokenizer、chat template。
- 合并 LoRA 后应立刻做后端兼容检查，而不是等全量评测后再排查。
- BFCL 的 `--include-input-log` 非常关键，它能确认实际喂给模型的 prompt。
- WSL 下跑 SGLang 建议设置 `TMPDIR=/tmp`，避免 ZMQ ipc 使用 Windows 临时目录。
