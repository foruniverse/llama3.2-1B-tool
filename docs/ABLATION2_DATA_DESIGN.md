# Ablation2 Data Design

## 目标

`scripts/2_prepare_data_ablation2.py` 用于构造第二个 SFT 消融实验数据集：

- 非 refusal JSON tool-call 样本测试模型对参数名和函数名扰动的鲁棒性。
- Refusal 样本使用 DeepSeek teacher 生成的自然语言 description。
- ToolACE 原始无工具 refusal 样本保留在初始全集中，参与 5% eval 抽样。

## 当前真实流程

1. 从 `data/tool_ace_processed` 读取结构化 ToolACE 样本。
2. 构造未扰动全集 `a+b+c`：
   - `a`：ToolACE 中 `tool_count>=1`、`tool_type=json`、`refusal=0` 的非 refusal 样本，由 `normalize_non_refusal_dataset()` 统一字段。
   - `b`：从 `data/ablation_refusal_teacher_deepseek/dataset` 读取 teacher refusal 数据，由 `normalize_refusal_teacher_dataset()` 只保留 `teacher_description` 或 JSON 顶层 `description` 作为 assistant response。
   - `c`：ToolACE 中 `refusal=1` 且 `tool_count=0` 的原始样本，由 `normalize_zero_tool_refusal_dataset()` 统一字段。
3. 在未扰动全集上用 `train_test_split(test_size=data.validation_split_percentage/100)` 固定抽取 5% eval。
4. `select_mode_indices()` 在 train 内的 a 类可扰动样本中抽取 30%：
   - `parameter_name`: 15%，扰乱参数名。
   - `function_name`: 10%，扰乱函数名。
   - `parameter_and_function`: 5%，同时扰乱参数名和函数名。
5. `select_eval_mode_indices()` 在 eval 内的 a 类可扰动样本中抽取 50%，全部执行 `parameter_and_function`。
6. `ablate_non_refusal_row()` 对所有扰动样本调用 ablation1 的 `rewrite_schema()`，打乱 schema 字段、`properties` 参数顺序和 `required` 顺序。
7. 参数名扰动复用 `rewrite_assistant_parameters()` 同步 assistant function-call。
8. 函数名扰动使用 `rewrite_function_names()` 改 tools，并用 `rewrite_assistant_function_names()` 同步 assistant function-call。
9. `add_prompt_column()` 使用手写 Llama chat template 生成 prompt：teacher refusal 样本的输入仍是 `system + tools + user`，description 只作为 assistant response。
10. 复用 ablation1 的 `tokenize_for_sft()` 构造 `input_ids`、`attention_mask`、assistant-only `labels`。

## 输入输出

- 输入结构化数据：`data/tool_ace_processed`
- 输入 teacher refusal：`data/ablation_refusal_teacher_deepseek/dataset`
- 训练输出：`data/train/train_ablation2_param_func_refusal_teacher`
- 验证输出：`data/eval/eval_ablation2_param_func_refusal_teacher`
- 报告：`data/ablation2/metadata.json`
- 预览：`data/ablation2/ablation2_preview.txt`
- 日志：`logs/prepare_data_ablation2.log`

## 对既有数据预处理的影响

- 不修改 `scripts/2_prepare_dataset.py` 的结构化解析逻辑。
- 不修改 `data/tool_ace_processed/sample_prompts.txt`。
- 不修改 `data/tool_ace_processed/dataset_report.md`。
- 不修改 `data/tool_ace_processed/dataset_report.json`。
- 不修改 `data/ablation1/metadata.json`。
- 新增 `data/ablation2/metadata.json`。
- 新增 ablation2 tokenized train/eval 输出路径。
