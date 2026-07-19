# 第四章 预训练语言模型：ELMo、BERT、GPT 与 T5

在 “open a bank account” 和 “sit on the river bank” 中，`bank` 的字典条目相同，当前语境需要的表示却不同。静态词向量给每个词表项一个固定坐标；上下文模型则把“这个 token 在这段序列里怎样使用”变成一次计算。

ELMo、BERT、GPT、T5 与 BART 都利用大规模文本，却不共享同一个信息流。比较它们应回答四个问题：训练样本怎样从原始语料产生，一个位置能读取哪些 token，损失要求模型恢复什么，预训练结果通过什么接口进入下游任务。

<a id="section-4-1"></a>

## 4.1 预训练问题的统一记号

设 tokenizer 为

$$
T_\tau:\mathcal S\to V^*,
$$

它把原始字符串映射为有限词表 $V$ 上的 token 序列。参数 $\tau$ 包含词表、规范化、预切分、特殊 token 和具体分词规则。神经网络实际接收的是 token ID，不直接接收字符或“词义”。更换 tokenizer 会同时改变序列长度、目标单位、训练成本和概率口径。

一条训练样本通常包含四类对象：

1. 输入 token $\widetilde x$，可能是原文、前缀或受损文本；
2. attention mask $A$，决定各位置可以读取什么；
3. 目标 token $y$；
4. loss mask $m_t\in\{0,1\}$ 或非负权重，决定哪些目标进入损失。

对自回归或 encoder-decoder 目标，假设至少有一个位置满足 $m_t>0$，一条样本的 token-normalized 负对数似然可写为

$$
\mathcal L(\theta)
=-
\frac{1}{\sum_{t=1}^{q}m_t}
\sum_{t=1}^{q}
m_t\log p_\theta(y_t\mid y_{<t},\widetilde x;A).
$$

encoder-only 的遮蔽恢复不使用输出前缀 $y_{<t}$，但仍有输入、可见性与 loss mask。attention mask 和 loss mask 不能混用：前者改变前向计算，后者只决定哪些已有 logits 贡献梯度。

### 4.1.1 交叉熵、nats 与 perplexity

若使用自然对数，平均交叉熵单位为 nat/token，相应 perplexity 是

$$
\operatorname{PPL}=\exp(\mathcal L).
$$

它是模型给真实 token 的几何平均逆概率。PPL 依赖 tokenizer：同一句字符串在不同词表下被分成不同数量与难度的事件，因此不能把不同 tokenizer 的 token-level PPL 直接横向排序。字节或字符归一化指标能缓解单位差异，也仍需声明数据与规范化方式。

## 4.2 从文档集合到训练 batch

预训练目标只说明怎样给一个样本计分，训练分布还由数据管线定义。设有 $K$ 个来源分布 $\mathcal D_1,\ldots,\mathcal D_K$，混合权重为 $\alpha_k\ge0$ 且 $\sum_k\alpha_k=1$，则抽样分布是

$$
\mathcal D=\sum_{k=1}^{K}\alpha_k\mathcal D_k.
$$

$\alpha_k$ 不必等于各来源原始字节占比。提高代码、数学或低资源语言的抽样权重，会改变经验风险中这些样本的权重；它不是无成本地“增加一种能力”。有限训练预算下，重采样也会挤占其他来源。

一条可复核的数据管线至少记录：

1. **获取与许可**：来源快照、时间范围、许可和删除规则；
2. **解析与过滤**：正文提取、语言识别、质量规则、个人信息与恶意内容处理；
3. **去重**：文档级、近似文档级和长片段级重复分别处理；
4. **切分**：先按文档或来源实体划分 train/validation/test，再生成短片段，避免同一文档跨集合；
5. **token 化与 packing**：把不同长度序列装入固定长度张量；
6. **混合与抽样**：给出来源权重、温度重采样和随训练阶段变化的课程；
7. **污染检查**：对评测题、答案、改写和近重复做版本化扫描。

重复样本等价于在经验目标中增加其权重，还会使训练集与验证集的近重复造成过于乐观的损失。去重也不是“重复文本越少越好”：模板、法律条文和代码样板可能具有合法重复，阈值过强会系统性删除某些领域。规则与抽样审计必须同时保留。

子词切分与语料去重的研究入口分别见 [Kudo & Richardson, 2018](SOURCE_NOTES.md#ref-kudo-sentencepiece-2018) 和 [Lee et al., 2022](SOURCE_NOTES.md#ref-lee-dedup-2022)。

### 4.2.1 Packing 的边界

设两个文档 token 序列为 $a$ 与 $b$。为提高有效 token 比例，系统可把它们拼进同一个长度为 $n$ 的训练块。这里有两种不同语义：

- 在中间加入文档结束 token，并允许 $b$ 读取 $a$；模型把前一文档视为随机前缀。
- 使用 block-diagonal attention mask，使两个文档彼此不可见；位置 ID 是否重置还需单独声明。

两者都能产生形状相同的张量，却优化不同条件分布。padding token 通常既不能被有效位置读取，也不应进入 loss；但序列边界 token 往往是有效预测目标，不能简单按 padding 处理。

## 4.3 优化循环与数值实现

设初始参数为 $\theta_0$，且 $m_0=v_0=0$。对 $s=1,2,\ldots$，令第 $s$ 个 batch 的 token-normalized 损失为 $\mathcal L_s$，并在更新前参数处计算 $g_s=\nabla_\theta\mathcal L_s(\theta_{s-1})$。AdamW 的一种常见逐元素写法是

$$
m_s=\beta_1m_{s-1}+(1-\beta_1)g_s,
$$

$$
v_s=\beta_2v_{s-1}+(1-\beta_2)g_s^2,
$$

$$
\widehat m_s=\frac{m_s}{1-\beta_1^s},
\qquad
\widehat v_s=\frac{v_s}{1-\beta_2^s},
$$

$$
\theta_s
=(1-\eta_s\lambda)\theta_{s-1}
-\eta_s\frac{\widehat m_s}{\sqrt{\widehat v_s}+\varepsilon}.
$$

最后一式把 weight decay 与自适应梯度更新分开。实际配方常不衰减 bias 和 norm 参数；这属于参数分组定义，不能从“使用 AdamW”四个字推出。

大规模训练还依赖以下机制：

- **warmup 与衰减**：学习率 $\eta_s$ 先升高再按 cosine、linear 或其他曲线衰减；
- **梯度累积**：多个 microbatch 的梯度相加后再更新。若 microbatch 有不同有效 token 数，应先按 token 求和再统一归一化，否则每个 microbatch 被赋予相同权重；
- **混合精度**：矩阵乘使用较低精度，部分归约、优化器状态或 master weights 保持较高精度；
- **梯度裁剪**：限制全局范数可以抑制异常步，但会改变实际更新；
- **activation checkpointing**：前向不保存部分激活，反向时重算，以计算换显存；
- **数据并行与模型并行**：决定梯度、参数和激活在设备间怎样通信。

因此训练复现至少需要模型配置、数据版本、全局有效 token batch、优化器参数、学习率曲线、精度策略、随机种子与 checkpoint 恢复规则。只公布参数量和训练 token 数不足以复现模型。

解耦 weight decay 的来源见 [Loshchilov & Hutter, 2019](SOURCE_NOTES.md#ref-loshchilov-adamw-2019)。

## 4.4 ELMo：把双向语言模型当作特征源

### 4.4.1 从静态表示到上下文表示

Word2Vec、GloVe 等静态方法学习映射

$$
e:V\to\mathbb R^d,
$$

同一词表项 $v\in V$ 总得到同一个向量 $e(v)$。上下文表示则写为

$$
h_t=F(x_{1:n},t),
$$

同一个 token ID 在不同序列或位置可以得到不同 $h_t$。这里的“上下文”是模型实际可见的 token 与掩码，不等于现实中的全部情境。

ELMo 分别训练前向与后向语言模型：

$$
p_{\rightarrow}(x_{1:n})
=\prod_{t=1}^{n}p(x_t\mid x_{<t}),
\qquad
p_{\leftarrow}(x_{1:n})
=\prod_{t=1}^{n}p(x_t\mid x_{>t}).
$$

两条 LSTM 在位置 $t$ 产生多层表示。下游任务学习各层的加权组合，例如

$$
\operatorname{ELMo}_t
=\gamma\sum_{l=0}^{L}s_lh_{t,l},
\qquad
s_l=\frac{e^{a_l}}{\sum_je^{a_j}}.
$$

$h_{t,l}$ 组合正向、反向及词级输入，$s_l$ 为任务特定权重，$\gamma$ 为尺度参数。原始 ELMo 的典型用法是把预训练表示作为下游网络的特征，而不是端到端更新整个语言模型。研究入口见 [Peters et al., 2018](SOURCE_NOTES.md#ref-peters-2018)。

<a id="section-4-2"></a>

## 4.5 BERT：双向编码器与遮蔽恢复

### 4.5.1 输入与可见性

BERT 使用 Transformer encoder。输入位置 $t$ 的初始表示由 token、segment 与 position embedding 相加：

$$
h_t^{(0)}
=e_{tok}(x_t)+e_{seg}(s_t)+e_{pos}(t).
$$

encoder self-attention 通常允许一个有效位置读取左右两侧。这里的“双向”指注意力可见性，不是把两个单向模型的输出简单拼接。

### 4.5.2 Masked Language Modeling

训练时从位置集合中抽取子集 $M$，构造受损输入 $\widetilde x$，只在这些位置计算恢复损失：

$$
\mathcal L_{\mathrm{MLM}}(\theta)
=-\frac{1}{|M|}\sum_{t\in M}
\log p_\theta(x_t\mid\widetilde x_{1:n}).
$$

原始 BERT 选择约 15% 的 token；被选位置中，80% 替换为 `[MASK]`，10% 替换为随机 token，10% 保持不变。后两项减轻预训练只见 `[MASK]`、下游从不见该标记的接口偏差，但没有完全消除训练与使用差异。

<img src="chapter_04/images/mlm_masking_strategy.png" width="75%" />

若一次遮住多个位置，常用损失把它们写成

$$
\prod_{t\in M}p_\theta(x_t\mid\widetilde x)
$$

的对数，即在给定同一受损输入后分别分类。各位置共享 Transformer 表示，因而参数和计算并非独立；但目标没有显式按某个顺序建模被遮 token 之间的条件依赖。MLM 类似随机化 pseudo-likelihood，不直接给出一个可从左到右采样的规范化联合分布。

### 4.5.3 NSP 与消融边界

原始 BERT 还使用 Next Sentence Prediction，判断两个片段是否在语料中相邻。后续 RoBERTa 去掉 NSP，并同时改变数据规模、批量、训练时长与 masking 配方后获得更好结果。这说明一个消融结论必须连同其训练配方解释，不能仅据模型名称认定某个目标普遍无效。

微调时，任务头与 encoder 参数通常共同更新。预训练降低了每个下游任务从零开始学习表示的成本，却没有保证语料事实正确、偏差消失或部署分布不变。来源见 [Devlin et al., 2018](SOURCE_NOTES.md#ref-devlin-2018) 与 [Liu et al., 2019](SOURCE_NOTES.md#ref-liu-roberta-2019)。

<a id="section-4-3"></a>

## 4.6 GPT：因果语言模型与序列生成

### 4.6.1 链式分解、移位标签与 teacher forcing

decoder-only Transformer 对序列使用概率链式法则

$$
p_\theta(x_{1:n})
=\prod_{t=1}^{n}p_\theta(x_t\mid x_{<t}),
$$

相应负对数似然为

$$
\mathcal L_{\mathrm{AR}}(\theta)
=-\frac{1}{\sum_tm_t}
\sum_{t=1}^{n}m_t
\log p_\theta(x_t\mid x_{<t}).
$$

实现把同一序列错开一位。例如：

| 位置 | 0 | 1 | 2 |
| --- | --- | --- | --- |
| 模型输入 | `<bos>` | `A` | `B` |
| 训练标签 | `A` | `B` | `<eos>` |

位置 1 的 hidden state 只能读取 `<bos>, A`，却与标签 `B` 比较。训练时所有真实前缀都已给定，这称为 teacher forcing；推理时前缀包含模型自己生成的 token，错误可能改变后续条件分布。这种训练与生成前缀分布的差异不能靠因果 mask 本身消除。

<img src="chapter_04/images/bert_vs_gpt_attention_mask.png" width="80%" />

### 4.6.2 从任务微调到上下文学习

GPT-1 展示生成式预训练后再进行任务微调；GPT-2 更系统地研究用任务描述和上下文诱导行为；GPT-3 在更大规模上展示 zero-shot、one-shot 与 few-shot 的 in-context learning。三者的连续性来自因果目标，变化则包括模型与数据规模、训练配方和任务接口。

上下文学习不等于在一次请求中永久更新参数。示例通过当前 token 序列改变条件计算，模型权重通常保持不变。模型可能从示例推断标签映射、格式或局部任务，也可能只复用表面模式；应通过反转标签、替换语义和改变示例顺序等对照实验区分。

GPT-1 至 GPT-3 的入口见 [Radford et al., 2018](SOURCE_NOTES.md#ref-radford-2018)、[Radford et al., 2019](SOURCE_NOTES.md#ref-radford-2019) 与 [Brown et al., 2020](SOURCE_NOTES.md#ref-brown-2020)。

### 4.6.3 缩放与固定计算预算

在固定模型家族、数据分布和训练范围内，验证损失常可用经验幂律近似：

$$
\mathcal L(N,D)
\approx\mathcal L_\infty+aN^{-\alpha}+bD^{-\beta},
$$

其中 $N$ 为非 embedding 参数量，$D$ 为训练 token 数；精确定义随研究而变。dense Transformer 的总训练计算可粗略写成

$$
C\approx\kappa ND,
$$

$\kappa$ 汇总前向、反向、序列长度、稀疏度与具体算子常数。固定 $C$ 时，增大 $N$ 必然减少可训练的 $D$，反之亦然。Kaplan 等与 Hoffmann 等给出的计算最优指数来自不同实验范围和拟合方法；后者在其研究范围内支持让模型规模与训练 token 更均衡地共同增长。它们是实验规划模型，不是适用于任意数据质量、MoE 架构和无限尺度的数学定律。

<img src="chapter_04/images/scaling_law_plot.png" width="80%" />

平均 loss 的平滑变化也不保证每项行为指标平滑。提示格式、后训练、评测污染、工具使用和测试时计算都可能改变观察结果。来源见 [Kaplan et al., 2020](SOURCE_NOTES.md#ref-kaplan-2020) 与 [Hoffmann et al., 2022](SOURCE_NOTES.md#ref-hoffmann-2022)。

### 4.6.4 基座模型还不是助手

因果预训练教模型按前缀分配后续概率，不保证它把自然语言前缀解释为需要遵循的用户指令，也不保证拒绝危险请求、承认未知或稳定采用指定格式。示范微调、偏好优化和安全训练因此形成独立的后训练阶段，第五章将专门展开。

<a id="section-4-4"></a>

## 4.7 T5 与 BART：条件生成的 encoder-decoder 路线

### 4.7.1 条件序列模型

encoder-decoder Transformer 先编码输入 $x$ 为 $H=\operatorname{Enc}_\theta(x)$，再令 decoder 通过 cross-attention 读取 $H$，并分解

$$
p_\theta(y\mid x)
=\prod_{t=1}^{m}p_\theta(y_t\mid y_{<t},H).
$$

训练时 decoder 输入同样是右移后的真实目标；encoder 可双向读取完整输入，decoder self-attention 保持因果，cross-attention 可读取全部有效 encoder 位置。

### 4.7.2 T5 的 span corruption

设原序列中选出互不重叠的连续片段 $s_1,\ldots,s_k$。输入把每个片段替换为唯一 sentinel：

$$
\widetilde x
=\operatorname{replace}
(x,s_i\mapsto\langle z_i\rangle),
$$

目标则按原顺序连接

$$
y=\langle z_1\rangle s_1
\langle z_2\rangle s_2\cdots
\langle z_k\rangle s_k
\langle z_{k+1}\rangle.
$$

唯一 sentinel 同时告诉 decoder 缺失片段的边界和对应位置。相比逐 token MLM，span corruption 显式生成连续缺失片段；相比纯因果续写，encoder 可以读取受损输入的两侧。

<img src="chapter_04/images/span_corruption_example.png" width="80%" />

T5 进一步把分类、翻译、摘要等任务统一为 text-to-text。统一接口减少任务特定输出头，却没有让不同任务共享相同数据分布、损失权重或评价标准。

### 4.7.3 BART 的去噪重建

BART 从噪声分布 $q(\widetilde x\mid x)$ 采样受损输入，再优化

$$
\mathbb E_{x\sim\mathcal D}
\mathbb E_{\widetilde x\sim q(\cdot\mid x)}
[-\log p_\theta(x\mid\widetilde x)].
$$

噪声可包括 token masking、删除、文本填充和句序打乱。目标都叫“去噪”并不表示等价：噪声过程决定训练中哪些信息被保留、模型必须重建什么。T5 与 BART 来源分别见 [Raffel et al., 2020](SOURCE_NOTES.md#ref-raffel-2020) 和 [Lewis et al., 2019](SOURCE_NOTES.md#ref-lewis-bart-2019)。

## 4.8 用信息流而不是品牌名比较模型

| 架构 | 训练时可见性 | 目标因子 | 原生接口 |
| --- | --- | --- | --- |
| 双向 RNN 特征模型 | 左右两条独立递推 | 两个方向的语言模型 | 上下文特征 |
| Transformer encoder | 输入内双向可见 | 被选位置的条件分类 | 整段编码、任务头 |
| Transformer decoder | 仅见当前前缀 | 完整序列的链式分解 | 开放式自回归生成 |
| Encoder-decoder | 输入双向，输出因果 | 给定输入的目标链式分解 | 输入到输出转换 |

这张表描述基本信息流，不排除 prefix mask、混合去噪目标或非自回归变体。遇到一个具体模型，应核对：

1. tokenizer 与特殊 token；
2. 样本拼接、attention mask 与 loss mask；
3. 数据来源、混合权重、去重和评测污染；
4. 目标构造与归一化单位；
5. 模型架构、优化器、精度和计算预算；
6. checkpoint 是否只是基座模型，还是已经经过后训练。

预训练模型把训练分布中的规律压入参数，使表示和生成可以跨任务迁移；它也把语料缺口、冲突和偏差带入模型。下一章讨论后训练如何塑造模型行为，第六章再处理推理效率与服务系统。

本章来源包括 ELMo、BERT、RoBERTa、GPT、T5、BART、AdamW、数据去重与计算最优训练研究，统一登记在[卷内来源表](SOURCE_NOTES.md)。
