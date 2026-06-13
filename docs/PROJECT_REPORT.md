# 面向小模型工具调用能力增强的监督微调与消融实验研究

**基础模型**：`Llama-3.2-1B-Instruct`  
**任务方向**：Tool / Function Calling  
**项目成员**：何成阳、白永康、刘松涛  
**日期**：2026-06-13

## 项目分工与贡献声明

| 成员 | 主要贡献 |
| --- | --- |
| 白永康 | Prompt Engineering |
| 刘松涛 | GRPO 强化学习方案设计、奖励机制构建与相关训练数据设计 |
| 何成阳 | 论文调研、SFT 训练方案设计与实验执行、DeepSeek 增强 ToolACE 数据构造、消融实验设计及完成、最终报告编写、PPT 制作与汇报 |

## 项目贡献

本项目围绕 1B 级小模型的工具调用能力增强，完成了从数据处理、监督微调、消融验证到后端兼容的完整技术探索，主要贡献如下：

1. **构建了面向小模型的工具调用监督微调流程**  
   项目基于 ToolACE 数据完成单轮工具调用样本构造、长度过滤、assistant-only label 构建和 LoRA SFT 训练，使 `Llama-3.2-1B-Instruct` 能够稳定学习工具选择、参数抽取和结构化输出格式。

2. **设计并完成了面向失效模式的消融实验**  
   项目围绕 schema 顺序、参数名语义、函数名语义和 refusal 边界构造多组扰动数据，用于分析模型是否依赖浅层词汇匹配、字段排列顺序和工具名称语义。

3. **验证了 SFT 对小模型工具调用能力的关键作用**  
   综合 BFCL 结果显示，SFT-v2 随机参数名模型相比 Base 平均准确率由 28.51 提升至 69.89，提升 41.38 个百分点，说明高质量监督数据是小模型工具调用能力形成的主要来源。

4. **完成了 GRPO 强化学习路径探索**  
   项目设计了面向工具调用的 rule-based reward，将格式、工具名、参数正确性、幻觉惩罚和简洁性纳入奖励结构，并完成 SFT 与 SFT+GRPO 的对比分析。
## 课程加分项完成情况

| 加分方向 | 完成情况 | 说明 |
| --- | --- | --- |
| 自构造训练数据 | 完成 | 使用 DeepSeek-v4-pro 针对ToolAce中irrelevance类型针对性合成了1400条拒绝负样本数据 |
| GRPO 强化学习 | 完成 | 完成 rule-based reward 设计，并对 SFT 与 SFT+GRPO 进行结果比较。 |
| Multi-turn / Agentic 能力 | 未完成 | 本轮实验聚焦单轮工具调用，没有替换为 Qwen3-0.6B，也没有完成 multi-turn/agentic 子集训练。 |


## 摘要

工具调用能力是大语言模型从通用文本生成走向可执行智能体的重要基础。对于 1B 级小模型而言，直接依靠提示词或基础指令能力通常难以稳定完成工具选择、参数抽取、并行调用、边界拒绝与结构化输出。本项目围绕 `Llama-3.2-1B-Instruct` 构建了一套面向工具调用的监督微调与消融实验流程：以 ToolACE 为核心数据来源，完成工具 schema 解析、单轮样本构造、长度过滤和 assistant-only 监督信号构建；随后围绕参数顺序、参数名语义、函数名语义和拒绝调用边界设计消融实验；最后通过 LoRA 训练、模型合并和 BFCL/SGLang 兼容化形成可复现工程闭环。

实验结果表明，Prompt Engineering 只能带来有限提升，平均准确率由 29.95% 提升至 32.34%；SFT 是主要性能来源，在阶段性评测中由 Base 平均 32.12% 提升至最佳 66.09%。进一步实验显示，模型并不稳定依赖固定参数顺序，但对参数语义和拒绝调用样本分布高度敏感。随机参数名设置在综合评测中达到 69.89% 平均准确率，相比 Base 的 28.51% 提升 41.38 个百分点；refusal 数据增强显著改善 `irrelevance` 与 `live_irrelevance` 等边界识别任务，但会牺牲部分常规工具调用性能。各组 SFT 与消融实验的 loss 曲线整体一致，训练损失持续下降、评估损失快速下降后趋于平台，说明训练过程稳定，消融结果具有可比较性。

## 1. 引言

工具调用任务要求模型在自然语言请求、工具文档、参数约束和输出格式之间建立稳定映射。模型不仅需要判断调用哪个工具，还需要生成合法参数、处理多工具组合、避免不存在工具或 schema 外参数，并在工具无法满足请求时拒绝调用或追问缺失信息。相比普通问答，工具调用具有更强的结构约束和更低的错误容忍度。

小模型在该任务上面临三类核心挑战。首先，模型容量有限，难以同时记忆大量函数接口并学习复杂的参数组合规律。其次，训练数据中常存在浅层词汇捷径，例如用户 query 与参数名高度重合，使模型倾向于学习字符串匹配而非真实语义绑定。最后，工具定义在 prompt 中具有固定排列顺序，模型可能学习位置先验，导致工具选择和参数生成受 schema 展示顺序影响。

本项目将研究问题定义为：在不显著增加模型规模的前提下，如何通过数据构造、监督微调和消融验证，提高小模型在 BFCL 风格工具调用任务上的准确性、鲁棒性和可复现性。围绕这一目标，项目重点回答以下问题：

1. SFT 相比 Prompt Engineering 是否能提供主要性能增益。
2. 模型是否依赖工具 schema 中参数字段的固定顺序。
3. 参数名语义对工具调用和拒绝调用边界识别的影响有多大。
4. teacher refusal 数据能否改善模型对无关任务和不可调用任务的判断。
5. GRPO 奖励设计能否在 SFT 基础上进一步提升工具调用表现。

## 2. 相关工作

### 2.1 工具调用数据合成与质量验证

ToolACE 提出自动化 agentic pipeline，用于生成 accurate、complex、diverse 的 tool-learning 数据。该工作通过 self-evolution synthesis 构建大规模 API 池，通过多智能体交互生成复杂对话，并用 rule-based 与 model-based 双层验证提高数据质量。论文报告其 API 池包含 26,507 个 API，覆盖 390 个领域，并支持 nested parameters、parallel calls、dependent calls 与 non-tool-use dialogs。该方法说明，高质量工具调用能力不仅依赖模型能力，也高度依赖训练数据的覆盖度、复杂度和正确性。[ToolACE: Winning the Points of LLM Function Calling](https://arxiv.org/abs/2409.00920)

本项目采用 ToolACE 作为核心数据来源，但进一步面向 1B 小模型进行了样本裁剪、长度过滤和监督信号重构，使其适配有限上下文长度和有限训练显存约束。

### 2.2 函数调用的细粒度任务分解

Granite-Function Calling Model 将函数调用能力拆解为七类基础任务：Nested Function Calling、Function Chaining、Parallel Functions、Function Name Detection、Parameter-Value Pair Detection、Next-Best Function 和 Response Generation。该工作表明，函数调用不是单一格式学习问题，而是由工具识别、参数抽取、调用规划和响应生成等子能力组成。[Granite-Function Calling Model](https://arxiv.org/abs/2407.00121)

本项目的消融实验延续了这种细粒度分析思想。参数顺序扰动、随机参数名和函数名扰动分别对应 schema 理解、参数语义绑定和函数识别能力；refusal 数据增强则对应工具边界判断和非调用场景建模。

### 2.3 结构化推理模板与函数调用可解释性

Guided-Structured Templates / ToolGT 研究指出，自由形式 CoT 在结构化函数调用任务中并不总是有效，甚至可能因冗余推理和格式漂移影响调用准确性。该工作使用结构化 reasoning templates 引导模型经过工具理解、参数抽取、隐式转换和任务约束检查等阶段，在 BFCLv2 与 Nexus 等基准上取得相对提升。[Improving Large Language Models Function Calling and Interpretability via Guided-Structured Templates](https://arxiv.org/abs/2509.18076)

本项目吸收了其中“结构化监督优先”的思想，但并未将长推理链作为主要训练目标，而是更强调压缩冗余模板、保留关键 assistant 输出、构造稳定的 function-call 监督。

### 2.4 工具调用强化学习与奖励设计

ToolRL 系统研究了工具选择和工具应用任务中的 reward design。该工作指出，工具调用中的奖励不能只依赖最终答案匹配，因为工具名、参数名、参数值、调用数量和调用时机都需要细粒度反馈；其基于 GRPO 的训练结果说明，合理奖励设计可以提高工具使用泛化能力。[ToolRL: Reward is All Tool Learning Needs](https://arxiv.org/abs/2504.13958)

本项目中的 GRPO 实验采用类似思路，将奖励拆分为格式、工具名、参数、幻觉惩罚和简洁性约束，用于探索强化学习在小模型工具调用任务上的增益。

## 3. 方法

### 3.1 总体框架

项目流程由数据处理、监督微调、消融构造、模型导出和后端兼容五个部分组成。

1. 从 ToolACE 原始数据中抽取工具 schema、用户请求和 assistant 回复。
2. 构造单轮 SFT 样本，并进行 token 长度过滤。
3. 基于参数顺序、参数名、函数名和 refusal 边界构造消融数据。
4. 使用 LoRA 对 `Llama-3.2-1B-Instruct` 进行 SFT。
5. 合并 LoRA 权重，并进行 BFCL/SGLang 兼容化处理。
6. 在 BFCL 风格任务上评估工具调用能力。

工程实现集中在 `scripts/` 与 `configs/sft_config.yaml` 中。数据与模型产物默认保存在 `data/` 与 `models/` 下，运行日志保存到 `logs/`。当前可复现代码主线包括模型下载、ToolACE 结构化预处理、消融数据构造、LoRA SFT、LoRA 合并和 SGLang/BFCL 兼容化。

### 3.2 数据预处理

原始 ToolACE 样本包含 system prompt、工具描述、多轮 user/assistant 对话和不同形式的工具 schema。为了适配 1B 模型的上下文长度与监督训练目标，项目将原始数据转化为统一的单轮工具调用样本。

预处理阶段首先将多轮对话裁剪为第一轮 user 与其后的第一条 assistant 回复，使训练目标聚焦于单轮函数调用决策。随后从 system prompt 中抽取工具定义，优先解析 JSON schema，同时兼容少量非 JSON 工具描述。工具定义与 system prompt 被重新组织为稳定输入格式，并手写 Llama chat template，以避免 tokenizer 默认模板注入日期或额外系统信息。最后，项目统计拼接后 token 长度，删除超过 1024 tokens 的样本。

本地处理结果如下：

| 指标 | 数值 |
| --- | ---: |
| 原始样本数 | 11,300 |
| 过滤后样本数 | 8,626 |
| 超长删除样本数 | 2,674 |
| 超长删除比例 | 23.66% |
| non-refusal 样本 | 7,083 |
| refusal 样本 | 1,543 |
| JSON 工具定义样本 | 8,500 |
| 非 JSON 工具定义样本 | 126 |
| 多工具样本 | 5,532 |
| 单工具样本 | 2,832 |
| 零工具样本 | 262 |

过滤后 token 长度分布如下：

| 统计量 | token 数 |
| --- | ---: |
| min | 153 |
| p50 | 644 |
| p90 | 927 |
| p95 | 974 |
| p99 | 1013 |
| max | 1024 |

训练标签采用 assistant-only loss：输入中的 system、tools 和 user 部分不参与 loss，只对 assistant 回复或函数调用片段计算监督信号。这一设计降低了模型学习复制工具文档的风险，并使优化目标集中在结构化调用输出上。

### 3.3 监督微调设置

基础模型为 `Llama-3.2-1B-Instruct`，微调方法为 LoRA。公共训练参数如下：

| 参数 | 配置 |
| --- | --- |
| epoch | 3 |
| train batch size | 2 |
| eval batch size | 2 |
| gradient accumulation | 24 |
| effective batch size | 48 |
| learning rate | `1.0e-4` |
| scheduler | cosine |
| warmup steps | 17 |
| max sequence length | 1024 |
| optimizer | `paged_adamw_8bit` |
| precision | bf16 优先，必要时 fp16 |
| gradient checkpointing | true |
| LoRA target modules | attention 与 MLP 主要线性层 |
| save/eval steps | 46 |
| seed | 42 |

当前配置包含 baseline、ablation1、ablation2 和 ablation3 四个实验 profile，分别对应基础数据、参数扰动数据、参数/函数/refusal 混合数据，以及更高 LoRA rank 的扩展设置。

### 3.4 消融实验设计

#### 3.4.1 参数顺序扰动

该实验检验模型是否依赖工具 schema 中字段出现顺序。训练样本中随机选取一部分可扰动样本，打乱工具定义中的字段顺序、参数属性顺序和 required 列表顺序，但保持参数语义内容不变。当前 ablation1 数据中，train 样本 8,194 条，eval 样本 432 条；实际扰动 train 样本 1,319 条，eval 样本 169 条。

#### 3.4.2 随机参数名

该实验检验参数名语义在工具调用中的作用。项目将部分样本中的参数名替换为无语义标识，并同步修改 assistant 输出中的参数名，从而削弱参数名与用户 query 之间的浅层词汇对应关系。该实验用于判断模型是否真正理解参数描述和 schema 约束，而非仅依赖参数名字符串。

#### 3.4.3 函数名扰动

函数名扰动进一步削弱函数名自身携带的语义信息，迫使模型更多依赖工具 description、参数 schema 和用户意图进行工具选择。该实验与随机参数名共同构成对工具调用语义来源的分析。

#### 3.4.4 Refusal 数据增强

原始 refusal 样本存在自然语言模板冗余、关键信息密度不足和参数引用难以安全同步的问题。项目对 `refusal=1` 且 `tool_count>0` 的样本进行可行性分析，发现只有 1.79% 的样本适合同时扰动工具和 assistant 文本，大部分 refusal 回复属于自然语言追问或解释，不适合直接做参数名替换。

为提高拒绝调用样本质量，项目使用 DeepSeek teacher 将 refusal 场景重写为更结构化、更高信息密度的 description。当前 ablation2 数据包含 6,998 条 ToolACE non-refusal JSON tool samples、1,234 条 teacher refusal samples 和 257 条 ToolACE zero-tool refusal samples，最终划分为 8,064 条 train 样本与 425 条 eval 样本。

### 3.5 GRPO 扩展实验

在 SFT 基础上，项目进一步探索了面向工具调用的 GRPO 奖励设计。奖励函数由格式、工具名、参数正确性、幻觉惩罚和简洁性约束构成：

```text
Reward = R_format + R_tool + R_args - P_hallucination + R_brevity
```

其中 `R_format` 鼓励输出可解析的工具调用格式，`R_tool` 奖励工具名匹配，`R_args` 奖励参数名、必填字段、类型和枚举值正确，`P_hallucination` 惩罚不存在工具和 schema 外参数，`R_brevity` 鼓励简洁输出。该设计与 ToolRL 中强调的细粒度工具调用奖励思想一致。

## 4. 实验结果

### 4.1 Prompt Engineering

Prompt Engineering 通过加入 few-shot 示例、增强任务描述与约束条件、明确调用格式等方式优化 system prompt。实验结果显示，平均准确率由 29.95% 提升至 32.34%，提升 2.39 个百分点。

该结果表明，提示词优化可以改善模型对工具调用任务的初步识别，但无法根本解决小模型在参数抽取、结构化输出和复杂工具规划上的能力不足。因此，Prompt Engineering 更适合作为推理侧约束或 baseline，而不是主要训练方法。

### 4.2 SFT 主实验

SFT 训练在早期阶段带来显著性能跃迁。阶段性评测结果如下：

| checkpoint | Average |
| --- | ---: |
| Base | 32.12 |
| 0.49 epoch / 90 steps | 58.42 |
| 0.97 epoch / 180 steps | 58.38 |
| 1.46 epoch / 270 steps | 49.72 |
| 1.95 epoch / 360 steps | 66.09 |
| 2.43 epoch / 450 steps | 62.55 |
| 3.00 epoch / 555 steps | 61.84 |

最佳平均性能出现在 1.95 epoch / 360 steps。继续训练后平均分略有回落，说明 SFT 主要收益集中在早期阶段，后续可能受到任务分布偏移、过拟合或拒绝调用比例变化影响。

### 4.3 训练稳定性

各组 SFT 与消融实验的 loss 曲线整体一致：train loss 随训练步数持续下降，局部存在小幅波动但未出现发散；eval loss 在早期快速下降，随后进入较平稳平台区。不同消融实验中的曲线形态基本吻合，说明训练过程稳定，主要性能差异更可能来自数据构造与监督信号变化，而不是训练不稳定或优化失败。

这一观察支持后续实验比较的有效性：参数顺序扰动、随机参数名和 refusal 数据增强之间的性能差异，可以被解释为训练数据分布和语义监督变化导致，而不是由 loss 异常或训练过程崩溃造成。

### 4.4 参数顺序扰动

参数顺序扰动结果如下：

| 设置 | 原始顺序 | 打乱顺序 |
| --- | ---: | ---: |
| 90 steps | 65.19 | 62.99 |
| 180 steps | 59.01 | 62.74 |

90 steps 设置下扰动后平均分小幅下降，180 steps 设置下扰动后反而提升。该结果说明模型不稳定依赖固定 schema 顺序，工具调用能力并非主要来自对参数排列位置的记忆。换言之，schema 顺序扰动不会系统性破坏模型能力，也可作为提升鲁棒性的合理数据增强方式。

### 4.5 随机参数名

随机参数名实验中，原始 SFT 平均分为 66.09，随机参数名模型平均分为 70.14。在多数常规调用任务上，随机参数名模型取得更高准确率；但在 `irrelevance` 和 `live_irrelevance` 任务上出现下降。

该结果揭示了参数名语义的双重作用。一方面，去除参数名语义后，模型被迫更多依赖参数 description 和 schema 结构，可能减少对 query-parameter 词汇重合的依赖；另一方面，参数名本身对判断工具是否适用仍有帮助，尤其在拒绝调用场景中，完全削弱参数名语义可能损害边界识别。

### 4.6 Refusal 数据增强

Refusal 数据增强结果显示，边界识别能力明显改善，但常规工具调用性能存在下降：

| Task | 原始 SFT | 随机参数名 | 拒绝数据增强 |
| --- | ---: | ---: | ---: |
| irrelevance | 90.83 | 78.33 | 89.58 |
| live_irrelevance | 83.71 | 66.97 | 80.43 |
| Average | 66.09 | 70.14 | 65.94 |

增强 refusal 数据能够恢复并提升无关任务判断，说明高质量 refusal supervision 对工具边界识别有效。但平均分下降说明模型变得更保守，可能降低部分本应调用工具场景中的调用积极性。因此，refusal 数据增强需要进一步控制样本比例、训练阶段和 loss 权重。

### 4.7 GRPO 扩展结果

GRPO 扩展实验结果如下：

| Model | Average |
| --- | ---: |
| Base | 30.18 |
| SFT | 65.05 |
| SFT + GRPO | 65.44 |

GRPO 相比 SFT 有小幅平均提升，并在 `parallel_multiple`、`parallel`、`live_multiple`、`live_parallel`、`live_simple` 等任务上带来收益；但在 `multiple` 和 `live_parallel_multiple` 等任务上存在下降。这说明当前奖励设计具备一定有效性，但仍不足以稳定提升所有工具调用类型。未来需要对工具选择、参数完整性、调用数量和拒绝边界设计更细粒度、更平衡的奖励。

### 4.8 综合 BFCL 结果

最终综合结果如下：

| Model | simple | multiple | parallel | parallel_multiple | live_simple | live_multiple | live_parallel | live_parallel_multiple | relevance | irrelevance | Average |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 43.25 | 51.00 | 42.50 | 15.00 | 30.62 | 7.12 | 0.00 | 0.00 | 43.75 | 51.89 | 28.51 |
| SFT-v2 随机参数名 | 72.08 | 87.50 | 79.00 | 72.00 | 60.08 | 53.47 | 56.25 | 45.83 | 100.00 | 72.65 | 69.89 |
| GPT-5.2-2025-12-11 (FC) | 72.92 | 88.00 | 89.00 | 77.50 | 71.71 | 70.37 | 68.75 | 58.33 | 75.00 | 79.42 | 75.10 |
| Llama-3.2-3B-Instruct (FC) | 70.67 | 92.50 | 88.50 | 79.00 | 65.12 | 57.64 | 25.00 | 37.50 | 87.50 | 52.06 | 65.55 |

SFT-v2 相比 Base 平均准确率由 28.51 提升至 69.89，提升 41.38 个百分点；在 `relevance` 上达到 100.00，说明模型已经具备较强的工具相关性判断能力。与更大规模模型相比，SFT-v2 仍低于 GPT-5.2-2025-12-11 (FC)，但高于该实验记录中的 Llama-3.2-3B-Instruct (FC)，表明针对性数据构造和微调可以显著提升小模型工具调用能力。

## 5. 讨论

### 5.1 SFT 是小模型工具调用能力的主要来源

Prompt Engineering 的收益有限，而 SFT 能够提供数量级更大的性能提升。这说明对于小模型而言，工具调用能力需要通过高质量监督样本显式学习，尤其是工具选择、参数抽取和结构化输出格式。

### 5.2 模型未显著依赖固定参数顺序

参数顺序扰动没有造成一致性性能退化，说明模型并非主要通过记忆 schema 线性排列完成调用。这一结果支持在训练中加入 schema shuffle，以提高模型对不同工具文档排列方式的鲁棒性。

### 5.3 参数名语义既可能形成捷径，也可能提供边界信号

随机参数名实验提高了整体平均分，说明削弱参数名词汇捷径可能迫使模型学习更稳健的 description/schema 对齐。但无关任务下降表明，参数名语义仍有助于模型判断工具是否适用。因此，最优策略不是简单删除参数语义，而是减少浅层匹配依赖，同时强化参数描述、类型和必填字段的联合建模。

### 5.4 Refusal 数据增强需要平衡调用积极性

Teacher refusal 数据提升了无关任务识别能力，但平均分下降说明模型在常规调用场景中可能更保守。后续可控制 refusal 样本占比，采用分阶段训练，或对 refusal 与 tool-calling 使用不同 loss 权重。

### 5.5 GRPO 的关键在奖励粒度

当前 GRPO 奖励已覆盖格式、工具名、参数和幻觉惩罚，但平均收益有限。工具调用任务的 RL 奖励需要进一步区分工具选择、参数召回、参数值正确性、调用数量、拒绝边界和输出简洁性，并避免单一奖励项过强导致模型过保守或过度调用。

## 6. 工程实现与可复现性

项目使用 `uv` 锁定 Python 环境，并通过 `configs/sft_config.yaml` 管理训练参数和实验 profile。主要运行命令如下：

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

模型下载、数据处理、训练、模型合并和后端兼容化均写入 `logs/`。日志记录配置路径、输入输出路径、样本数量、token 长度、训练步数、LoRA 参数、设备信息和异常原因，从而保证实验可追踪。

模型导出后，项目还处理了 BFCL/SGLang 后端兼容问题。Transformers 5.x 保存的 `rope_parameters` 与 `dtype` 字段在部分较旧 SGLang/Transformers 评测环境中不能稳定识别，因此项目将其转换为 `rope_scaling`、`rope_theta` 和 `torch_dtype`，并修正 tokenizer metadata。该步骤保证同一模型目录在 Transformers 直接推理和 BFCL/SGLang 后端下具有一致行为。

轻量验证命令为：

```bash
uv run python -m py_compile scripts/*.py scripts/utils/*.py
```

涉及训练逻辑修改时，优先使用最小步数 smoke test：

```bash
SFT_MAX_STEPS=1 uv run python scripts/3_sft_training.py
```

## 7. 局限性与后续工作

当前项目仍存在若干限制：

- refusal 数据增强存在性能 trade-off，需要进一步研究样本比例、训练阶段和 loss 权重。
- 当前评测主要依赖 BFCL accuracy，尚未细分工具名准确率、参数名准确率、必填参数召回率、参数值正确率和 hallucination rate。
- 部分复杂任务，如 `live_parallel_multiple`，仍显著弱于更强模型。

后续工作包括：

1. 构造 Tool Selection Hard Cases、参数缺失、模糊输入和无需调用工具的更大规模合成数据。
2. 优化 refusal 样本比例、训练阶段和 loss 权重，平衡常规调用与拒绝边界能力。
3. 设计更细粒度的 reward decomposition，提升 GRPO 在复杂工具调用任务上的稳定收益。
4. 增加参数级评测指标，定位错误来自工具选择、参数缺失、参数值错误还是格式非法。
5. 建立统一实验登记表，保存每次实验的数据版本、配置、checkpoint、评测命令和结果。

## 8. 结论

本项目证明，对于 1B 级小模型，工具调用能力主要依赖高质量监督数据和针对性微调，而非单纯提示词优化。通过 ToolACE 数据结构化、LoRA SFT 和针对失效模式的消融实验，`Llama-3.2-1B-Instruct` 在 BFCL 风格任务上获得显著提升。随机参数名和参数顺序扰动实验表明，模型并不主要依赖固定 schema 顺序，但参数语义仍深刻影响工具选择和拒绝边界。Refusal 数据增强进一步说明，边界识别能力可以通过 teacher 数据改善，但需要控制其对常规工具调用的副作用。

整体而言，本项目形成了一套面向小模型工具调用增强的可复现技术路线：以结构化数据处理为基础，以 LoRA SFT 为核心，以消融实验解释能力来源，以后端兼容化保证评测可靠性。同时，GRPO 实验表明强化学习路径具备一定潜力，但其收益高度依赖奖励粒度和任务分布设计。

## 参考文献

1. Weiwen Liu et al. [ToolACE: Winning the Points of LLM Function Calling](https://arxiv.org/abs/2409.00920). ICLR 2025.
2. Ibrahim Abdelaziz et al. [Granite-Function Calling Model: Introducing Function Calling Abilities via Multi-task Learning of Granular Tasks](https://arxiv.org/abs/2407.00121). EMNLP Industry Track 2024.
3. Hy Dang et al. [Improving Large Language Models Function Calling and Interpretability via Guided-Structured Templates](https://arxiv.org/abs/2509.18076). EMNLP 2025.
4. Cheng Qian et al. [ToolRL: Reward is All Tool Learning Needs](https://arxiv.org/abs/2504.13958). 2025.
5. Berkeley Function-Calling Leaderboard. [BFCL / Gorilla project](https://gorilla.cs.berkeley.edu/leaderboard.html).
