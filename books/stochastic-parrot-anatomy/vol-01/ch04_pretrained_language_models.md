# 第四章 预训练语言模型：ELMo、BERT、GPT 与 T5

在 “open a bank account” 和 “sit on the river bank” 中，`bank` 的字典条目相同，当前语境需要的表示却不同。静态词向量给每个词形一个固定坐标；上下文模型则把“这个 token 在这段序列里怎样使用”变成一次计算。

ELMo、BERT、GPT、T5 与 BART 都利用大规模无标注文本，却不共享同一个信息流。比较它们应回答三个问题：一个位置在训练时能读取哪些 token，损失要求模型恢复什么，预训练结果通过什么接口进入下游任务。

<a id="section-4-1"></a>

## 4.1 从静态词向量到上下文表示

### 4.1.1 静态表示的边界

Word2Vec、GloVe 等方法学习映射

$$
e:V\to\mathbb R^d,
$$

同一词表项 $v\in V$ 总得到同一个向量 $e(v)$。向量可以编码语料中的总体共现规律，却不能单独区分当前句子的词义、句法角色与指代关系。

上下文表示改为

$$
h_t=F(x_{1:n},t),
$$

同一个 token ID 在不同序列或位置可以得到不同 $h_t$。这里的“上下文”是模型实际可见的 token 与掩码，不等于现实中的全部情境。

### 4.1.2 ELMo：把双向语言模型当作特征源

ELMo 分别训练前向与后向语言模型：

$$
p(x_{1:n})=\prod_{t=1}^{n}p(x_t\mid x_{<t}),
$$

$$
p(x_{1:n})=\prod_{t=1}^{n}p(x_t\mid x_{>t}).
$$

两条 LSTM 在位置 $t$ 产生多层表示。下游任务学习各层的加权组合，例如

$$
\operatorname{ELMo}_t
=\gamma\sum_{l=0}^{L}s_l h_{t,l},
\qquad
s_l=\frac{e^{a_l}}{\sum_j e^{a_j}}.
$$

$h_{t,l}$ 组合正向、反向及词级输入，$s_l$ 为任务特定权重，$\gamma$ 为尺度参数。原始 ELMo 的典型用法是把预训练表示作为下游网络的特征，而不是对整个语言模型做端到端微调。它由此展示了两件事：词义可以随句子变化，不同网络层也可能服务不同任务。

研究入口见 [Peters et al., 2018](SOURCE_NOTES.md#ref-peters-2018)。

<a id="section-4-2"></a>

## 4.2 BERT：双向编码器与遮蔽恢复

### 4.2.1 输入与可见性

BERT 使用 Transformer encoder。输入位置 $t$ 的初始表示由 token、segment 与 position embedding 相加：

$$
h_t^{(0)}
=e_{tok}(x_t)+e_{seg}(s_t)+e_{pos}(t).
$$

encoder self-attention 通常允许一个未被遮蔽的位置读取左右两侧。这里的“双向”指注意力可见性，不是把两个单向模型的输出简单拼接。

### 4.2.2 Masked Language Modeling

训练时从位置集合中抽取子集 $M$，只在这些位置计算恢复损失：

$$
L_{MLM}(\theta)
=-\sum_{t\in M}
\log p_\theta(x_t\mid\tilde x_{1:n}).
$$

$\tilde x$ 是受损输入。原始 BERT 选择约 15% 的 token；被选位置中，80% 替换为 `[MASK]`，10% 替换为随机 token，10% 保持不变。后两项减轻预训练只见 `[MASK]`、下游从不见该标记的接口偏差，但没有完全消除训练与使用差异。

<img src="chapter_04/images/mlm_masking_strategy.png" width="75%" />

MLM 不是从左到右生成目标句。模型同时利用受损输入中可见的左右条件，对选中位置分类；这使 encoder 表示适合整段分类、序列标注、抽取式问答和检索编码等任务。

### 4.2.3 NSP、训练配方与结论边界

原始 BERT 还使用 Next Sentence Prediction，判断两个片段是否在语料中相邻。后续 RoBERTa 去掉 NSP，并同时改变数据规模、批量、训练时长与 masking 配方后获得更好结果。这说明一个消融结果必须连同其训练配方解释，不能仅据模型名称认定某个目标普遍无效。

微调时，任务头与 encoder 参数通常共同更新。预训练降低了每个下游任务从零开始学习表示的成本，却没有保证预训练语料中的事实正确、偏差消失或部署分布保持不变。

BERT 与 RoBERTa 的来源分别见 [Devlin et al., 2018](SOURCE_NOTES.md#ref-devlin-2018) 和 [Liu et al., 2019](SOURCE_NOTES.md#ref-liu-roberta-2019)。

<a id="section-4-3"></a>

## 4.3 GPT：因果语言模型与序列生成

### 4.3.1 自回归目标

decoder-only Transformer 对序列使用链式分解

$$
p_\theta(x_{1:n})
=\prod_{t=1}^{n}
p_\theta(x_t\mid x_{<t}),
$$

相应负对数似然为

$$
L_{AR}(\theta)
=-\sum_{t=1}^{n}
\log p_\theta(x_t\mid x_{<t}).
$$

因果掩码阻止位置 $t$ 读取未来 token。训练时整段已知，可以并行计算所有位置的 logits；推理时下一个输入依赖刚生成的 token，因而沿序列维度自回归执行。

<img src="chapter_04/images/bert_vs_gpt_attention_mask.png" width="80%" />

BERT 与 GPT 的差别不宜简化为“理解模型”和“生成模型”。encoder 与 decoder 都能形成上下文表示，核心差别是训练可见性和接口：双向 encoder 擅长编码完整输入，因果 decoder 的目标直接给出续写分布。

### 4.3.2 从任务微调到上下文学习

GPT-1 展示了生成式预训练后再进行任务微调；GPT-2 更系统地研究用任务描述和上下文直接诱导行为；GPT-3 在更大规模上展示了 zero-shot、one-shot 与 few-shot 的 in-context learning。三者的连续性来自同一个因果目标，变化则包括模型与数据规模、训练配方和任务接口。

上下文学习不等于在一次请求中永久更新参数。示例通过当前 token 序列改变条件计算，模型权重通常保持不变。模型可能从示例推断标签映射、格式或局部任务，也可能只复用表面模式；应通过反转标签、替换语义和改变示例顺序等对照实验区分。

GPT-1 至 GPT-3 的研究入口见 [Radford et al., 2018](SOURCE_NOTES.md#ref-radford-2018)、[Radford et al., 2019](SOURCE_NOTES.md#ref-radford-2019)与 [Brown et al., 2020](SOURCE_NOTES.md#ref-brown-2020)。

### 4.3.3 缩放定律是经验规律

在固定模型家族、数据分布和训练范围内，语言模型损失常随参数量 $N$、数据量 $D$ 或计算量 $C$ 呈近似幂律下降，例如

$$
L(N)\approx L_\infty+aN^{-\alpha}.
$$

$L_\infty,a,\alpha$ 都依赖实验口径。该关系可用于规划训练规模，却不是对任意能力、数据质量和无限尺度的数学定律。固定计算预算时，参数与训练 token 的配置还存在权衡；过大的模型若训练数据不足，并不自动达到计算最优。

<img src="chapter_04/images/scaling_law_plot.png" width="80%" />

平均 loss 的平滑变化也不保证每项行为指标平滑。提示格式、后训练、评测污染、工具使用和测试时计算都可能改变观察结果。缩放研究入口见 [Kaplan et al., 2020](SOURCE_NOTES.md#ref-kaplan-2020)。

### 4.3.4 基座模型还不是助手

因果预训练教模型按前缀分配后续概率，不保证它把自然语言前缀解释为需要遵循的用户指令，也不保证拒绝危险请求、承认未知或稳定采用指定格式。示范微调、偏好优化和安全训练因此形成独立的后训练阶段，第五章将专门展开。

<a id="section-4-4"></a>

## 4.4 T5 与 BART：条件生成的 encoder-decoder 路线

### 4.4.1 条件序列模型

encoder-decoder Transformer 先编码输入 $x$，再令 decoder 通过 cross-attention 读取编码结果，并分解

$$
p_\theta(y\mid x)
=\prod_{t=1}^{m}
p_\theta(y_t\mid y_{<t},x).
$$

它把双向输入编码与自回归输出结合，适合翻译、摘要、改写和其他“给定输入生成目标”的任务。

### 4.4.2 T5 的 text-to-text 接口

T5 把分类、翻译、摘要等任务统一为文本输入到文本输出，并使用 span corruption 预训练。连续被遮蔽片段由 sentinel token 替换，目标序列按 sentinel 顺序恢复缺失内容：

<img src="chapter_04/images/span_corruption_example.png" width="80%" />

相比逐 token MLM，span corruption 迫使模型恢复连续片段；相比纯因果续写，encoder 可以读取受损输入的两侧。统一接口减少了任务特定输出头，却没有让不同任务共享完全相同的数据分布和评价标准。

### 4.4.3 BART 的去噪序列到序列训练

BART 对输入施加 token masking、删除、文本填充、句序打乱等噪声，再由自回归 decoder 重建原文。目标仍是条件生成，噪声过程决定模型在预训练中学会修复哪些破坏。

T5 与 BART 分别见 [Raffel et al., 2020](SOURCE_NOTES.md#ref-raffel-2020)和 [Lewis et al., 2019](SOURCE_NOTES.md#ref-lewis-bart-2019)。

## 4.5 用信息流而不是品牌名选择架构

| 架构 | 训练时主要可见性 | 典型目标 | 原生接口 |
| --- | --- | --- | --- |
| 双向 RNN 特征模型 | 左右两条独立递推 | 双向语言模型 | 上下文特征 |
| Transformer encoder | 输入内双向可见 | 遮蔽恢复 | 整段编码、任务头 |
| Transformer decoder | 仅见当前前缀 | 下一 token 预测 | 开放式自回归生成 |
| Encoder-decoder | 输入双向，输出因果 | 条件去噪或序列生成 | 输入到输出转换 |

这张表描述基本信息流，不排除混合目标、prefix mask、encoder-only 生成头或非自回归变体。架构名称只是默认设置，具体模型仍应核对 attention mask、loss mask、tokenizer 与推理实现。

预训练模型把大量语料规律压入参数，使表示和生成可以跨任务迁移；它也把语料缺口、冲突和偏差带入模型。下一章讨论后训练如何塑造模型行为与任务接口，第六章再处理推理效率与服务系统。
