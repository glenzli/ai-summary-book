# 第三章 注意力、Transformer 与高效序列架构
<a id="section-3-1"></a>

## 3.1 注意力机制：从瓶颈到聚焦
### 3.1 Attention Mechanisms: From Bottleneck to Focus

在第 2 章中，我们探讨了 Seq2Seq 模型。虽然它是一个强大的框架，但它存在一个致命的缺陷：**信息瓶颈 (Information Bottleneck)**。无论输入句子有多长，Encoder 都必须将其压缩成一个固定长度的向量 $\mathbf{c}$。这就像要求你读完一本《红楼梦》，然后把所有细节都浓缩在这一张便利贴上，这显然是不可能的。

本节我们将介绍 **注意力机制 (Attention Mechanism)**，它允许 Decoder 在生成的每一步都“回头看”源句子的不同部分，从而打破了这一瓶颈。

#### 3.1.1 瓶颈问题 (The Bottleneck Problem)

在传统的 Encoder-Decoder 架构中：
$$ \mathbf{c} = \text{Encoder}(\mathbf{x}_1, \dots, \mathbf{x}_T) $$
$$ \mathbf{y}_t = \text{Decoder}(\mathbf{y}_{t-1}, \mathbf{s}_{t-1}, \mathbf{c}) $$

当 $T$ 很大时（例如长文本翻译），$\mathbf{c}$ 无法承载所有信息，导致性能急剧下降。这在数学上表现为长距离梯度流的衰减，在语义上表现为细节丢失。

#### 3.1.2 Bahdanau 注意力 (Additive Attention)

Bahdanau 等人 (2014) 提出的核心思想是：**上下文向量 $\mathbf{c}$ 不应该是静态的，而应该是动态变化的 $\mathbf{c}_t$**。

在 Decoder 生成第 $t$ 个词时，它应该根据当前的隐状态 $\mathbf{s}_{t-1}$，去计算与 Encoder 所有隐状态 $\mathbf{h}_j$ 的相关性。

**数学构造**：

1.  **对齐分数 (Alignment Score)**：计算 Decoder 状态 $\mathbf{s}_{t-1}$ 与 Encoder 状态 $\mathbf{h}_j$ 的匹配度。Bahdanau 使用一个小型的神经网络（MLP）来计算：
    $$ e_{tj} = \mathbf{v}_a^T \tanh(\mathbf{W}_a \mathbf{s}_{t-1} + \mathbf{U}_a \mathbf{h}_j) $$
    这被称为 **加性注意力 (Additive Attention)**。

2.  **注意力权重 (Attention Weights)**：使用 Softmax 将分数归一化为概率分布：
    $$ \alpha_{tj} = \frac{\exp(e_{tj})}{\sum_{k=1}^T \exp(e_{tk})} $$

3.  **动态上下文向量 (Dynamic Context Vector)**：加权求和：
    $$ \mathbf{c}_t = \sum_{j=1}^T \alpha_{tj} \mathbf{h}_j $$

**直观解释**：
$\alpha_{tj}$ 就像是“目光的焦点”。如果翻译到 "apple"，模型可能会发现源句子中 "苹果" 对应的 $\mathbf{h}_j$ 的权重 $\alpha_{tj}$ 为 0.9，而其他词的权重很小。

#### 3.1.3 Luong 注意力 (Multiplicative Attention)

Luong 等人 (2015) 提出了更简单的计算对齐分数的方法，利用点积：

1.  **点积 (Dot)**: $e_{tj} = \mathbf{s}_{t-1}^T \mathbf{h}_j$
2.  **通用 (General)**: $e_{tj} = \mathbf{s}_{t-1}^T \mathbf{W}_a \mathbf{h}_j$

这种方法计算更快（矩阵乘法优化），被称为 **乘性注意力 (Multiplicative Attention)**。这正是后来 Transformer 中 Scaled Dot-Product Attention 的雏形。（关于 Transformer 中 Scaled Dot-Product Attention 的详细数学推导，请见 **[附录 A.10](appendix/a.10_transformer_math.md)**）

#### 3.1.4 架构可视化

下面的图展示了引入注意力机制后的数据流。请注意 Context Vector $\mathbf{c}_t$ 是如何随时间步 $t$ 变化的。

```mermaid
graph TD
    subgraph Encoder
        x1[x1] --> h1((h1))
        x2[x2] --> h2((h2))
        x3[x3] --> h3((h3))
    end

    subgraph AttentionMechanism ["Attention Mechanism at step t"]
        s_prev["Decoder state s_t-1"]

        h1 --> score1["Score e_t1"]
        s_prev --> score1

        h2 --> score2["Score e_t2"]
        s_prev --> score2

        h3 --> score3["Score e_t3"]
        s_prev --> score3

        score1 --> Softmax
        score2 --> Softmax
        score3 --> Softmax

        Softmax --> alpha["Attention Weights α"]

        alpha --> Context(("c_t"))
        h1 --> Context
        h2 --> Context
        h3 --> Context
    end

    subgraph Decoder
        Context --> s_t["Decoder state s_t"] --> y_t["Output y_t"]
    end

    style Encoder fill:#F5F5F5,stroke:#666666,stroke-width:2px,color:#000000
    style Decoder fill:#F5F5F5,stroke:#666666,stroke-width:2px,color:#000000
    style AttentionMechanism fill:#E1D5E7,stroke:#9673A6,stroke-width:2px,color:#000000
    style Context fill:#FFF2CC,stroke:#D6B656,stroke-width:2px,color:#000000
    style alpha fill:#DAE8FC,stroke:#6C8EBF,stroke-width:2px,color:#000000
```

#### 3.1.5 注意力的本质：可微的键值查询

我们可以将注意力机制抽象为一种 **查询 (Query)** 过程。

更重要的是，这个“查询”是 **可微分的 (Differentiable)**：注意力权重 $\alpha_{tj}$ 由 Softmax 产生，整个链路可以通过反向传播学习“如何对齐”。
*   **查询 (Query, $\mathbf{q}$)**: Decoder 当前状态 $\mathbf{s}_{t-1}$（我想要什么？）
*   **键 (Key, $\mathbf{k}$)**: Encoder 隐状态 $\mathbf{h}_j$ 的特征（你有什么特征？）
*   **值 (Value, $\mathbf{v}$)**: Encoder 隐状态 $\mathbf{h}_j$ 的内容（你的内容是什么？）

在 RNN Attention 中，Key 和 Value 通常是同一个东西（即 $\mathbf{h}_j$）。但在 Transformer 中，我们将看到这三者被显式地分离开来。

**技术本质（统一形式）**：无论是加性还是乘性注意力，最终都会得到一个归一化的权重向量，并对 Value 做加权平均。

设打分函数为 $e_{tj} = \text{score}(\mathbf{s}_{t-1}, \mathbf{h}_j)$，则
$$ \alpha_{tj} = \text{softmax}_j(e_{tj}), \quad \mathbf{c}_t = \sum_{j=1}^{T} \alpha_{tj} \mathbf{h}_j $$

这也是后续 Transformer 把它“矩阵化”的原因：只要把 $\text{score}(\cdot)$ 写成矩阵乘法，就能充分利用 GPU 的并行算力。

<span style="background-color: #DAE8FC; color: black; padding: 2px 4px; border-radius: 4px;">Key Concept</span> **软寻址 (Soft Addressing)**：
传统的数据库查询是硬寻址（要么匹配，要么不匹配）。注意力机制是软寻址，它返回所有 Value 的加权平均。因为所有操作都是可微的，我们可以通过反向传播来学习“如何查询”。

#### 3.1.6 对齐矩阵可视化：注意力到底在“看哪里”？

为了把 $\alpha_{tj}$ 变成可以“肉眼检查”的对象，我们通常把它画成一个 **对齐矩阵 (Alignment Matrix)**：
*   **横轴**：源序列位置 $j$（Encoder 的 token）。
*   **纵轴**：目标序列位置 $t$（Decoder 正在生成的 token）。
*   **像素值**：$\alpha_{tj}$，越亮表示注意力越集中。

<img src="chapter_03/images/attention_alignment_heatmap.png" width="85%" />

这种可视化在早期机器翻译中非常常用：如果模型把英文单词 "apple" 对齐到中文 "苹果" 上，热力图会出现一条接近对角线的高亮带。
<a id="section-3-2"></a>

## 3.2 Transformer 架构解剖：以一当百
### 3.2 The Transformer Architecture

2017 年，Google 团队发表了 *Attention Is All You Need*，推动 NLP 主流架构发生了重要转向。原始 Transformer 抛弃了循环（RNN）和卷积（CNN），主要依赖注意力机制来捕捉输入和输出之间的全局依赖关系。

本节我们将深入解剖 Transformer 的内部构造，重点关注自注意力机制和多头注意力。

#### 3.2.1 整体架构概览 (Architecture Overview)

Transformer 依然遵循 Encoder-Decoder 结构，但每一层都焕然一新。

```mermaid
graph TB
    %% 样式定义
    classDef encoder fill:#F5F5F5,stroke:#666666,stroke-width:2px,color:#000000;
    classDef decoder fill:#F5F5F5,stroke:#666666,stroke-width:2px,color:#000000;
    classDef attn fill:#DAE8FC,stroke:#6C8EBF,stroke-width:2px,color:#000000;
    classDef ffn fill:#FFF2CC,stroke:#D6B656,stroke-width:2px,color:#000000;
    classDef norm fill:#E1D5E7,stroke:#9673A6,stroke-width:2px,color:#000000;

    subgraph EncoderLayer ["Encoder Layer"]
        Input_Enc[Input Embedding]:::encoder --> PE_Enc[Positional Encoding]:::encoder
        PE_Enc --> MHA_Enc[Multi-Head Self-Attention]:::attn
        MHA_Enc --> AddNorm1["Add & Norm"]:::norm
        AddNorm1 --> FFN_Enc[Feed Forward Network]:::ffn
        FFN_Enc --> AddNorm2["Add & Norm"]:::norm
    end

    subgraph DecoderLayer ["Decoder Layer"]
        Input_Dec[Output Embedding]:::decoder --> PE_Dec[Positional Encoding]:::decoder
        PE_Dec --> MMHA_Dec[Masked Multi-Head Self-Attention]:::attn
        MMHA_Dec --> AddNorm3["Add & Norm"]:::norm
        AddNorm3 --> MHA_Dec[Multi-Head Attention Cross]:::attn
        MHA_Dec --> AddNorm4["Add & Norm"]:::norm
        AddNorm4 --> FFN_Dec[Feed Forward Network]:::ffn
        FFN_Dec --> AddNorm5["Add & Norm"]:::norm
    end

    AddNorm2 -->|K,V| MHA_Dec
```

#### 3.2.2 缩放点积注意力 (Scaled Dot-Product Attention)

这是 Transformer 的核心算子。

**输入**：
*   **查询 (Query, Q)**: $\in \mathbb{R}^{n \times d_k}$
*   **键 (Key, K)**: $\in \mathbb{R}^{m \times d_k}$
*   **值 (Value, V)**: $\in \mathbb{R}^{m \times d_v}$

**计算公式**：
$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

1.  **$QK^T$**: 计算 Query 和 Key 的相似度矩阵（Gram Matrix）。
2.  **Scale ($\frac{1}{\sqrt{d_k}}$)**:
    *   **为什么要缩放？** 假设 $q, k$ 的分量独立且服从 $\mathcal{N}(0, 1)$，则 $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ 的方差为 $d_k$。当 $d_k$ 很大时，点积结果会很大，导致 Softmax 进入饱和区（梯度趋近于 0）。
    *   除以 $\sqrt{d_k}$ 将方差归一化为 1，保证梯度的稳定性。
3.  **Softmax**: 将相似度转换为概率分布。
4.  **MatMul V**: 根据概率分布加权求和 Value。

#### 3.2.3 多头注意力 (Multi-Head Attention)

如果只用一个注意力头，模型可能只能关注到一个方面（例如语法结构）。我们希望模型能同时关注多个方面（语法、语义、指代等）。

**机制**：
将 $Q, K, V$ 投影到 $h$ 个不同的子空间，分别计算注意力，最后拼接起来。

$$ \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V) $$
$$ \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O $$

*   **直观类比**：就像你读一篇文章，用红笔划重点（关注语法），用蓝笔划重点（关注情节），用绿笔划重点（关注人物关系）。最后把所有笔记汇总。

#### 3.2.4 前馈网络 (Position-wise Feed-Forward Networks)

在 Attention 层之后，是一个全连接网络，对 **每个位置** 独立同分布地进行处理。

$$ \text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2 $$

这相当于两个线性变换中间夹一个 ReLU。
虽然是线性变换，但因为有 ReLU，它提供了 **非线性** 能力，增加了模型的表达力。它通常会将维度放大（例如从 512 到 2048），然后再缩放回来，这被称为 **倒瓶颈结构 (Inverted Bottleneck)**。

#### 3.2.5 为什么 Transformer 优于 RNN？

| 特性 | RNN | Transformer |
| :--- | :--- | :--- |
| **计算并行性** | 差 ($O(N)$ 串行) | 极好 ($O(1)$ 并行) |
| **长距离依赖** | 弱 (路径长度 $O(N)$) | 强 (路径长度 $O(1)$) |
| **计算复杂度** | $O(N \cdot d^2)$ | $O(N^2 \cdot d)$ |

<span style="background-color: #DAE8FC; color: black; padding: 2px 4px; border-radius: 4px;">Key Concept</span> **全局视野**：
RNN 看句子像管中窥豹，一次看一个词；Transformer 是一目十行，一次性把所有词之间的关系（$N^2$ 个连接）都算出来了。这也是为什么它对算力（显存）要求更高的原因。

#### 3.2.6 残差连接与归一化：深层可训练的关键 (Residual & Normalization)

如果只堆叠注意力和前馈网络，深层网络会很难训练。Transformer 之所以能“越堆越深”，依赖两个工程-数学都很重要的结构：**残差连接 (Residual Connection)** 与 **层归一化 (Layer Normalization, LN)**。

在原始 Transformer（Post-Norm）里，一个子层的标准写法是：
$$ x_{out} = \text{LN}(x + \text{Sublayer}(x)) $$

而现代大模型更常见的是 Pre-Norm（更稳定）：
$$ x_{out} = x + \text{Sublayer}(\text{LN}(x)) $$

这两种写法的差别，以及正弦位置编码为何能表达相对位移，我们会在 3.3 节进一步展开。
<a id="section-3-3"></a>

## 3.3 位置编码与层归一化
### 3.3 Positional Encoding and Layer Normalization

Transformer 的 Self-Attention 机制本身具有 **置换等变性 (Permutation Equivariance)**：如果同时打乱输入 token 的顺序，输出表示也会以同样方式被打乱，而注意力层本身并不知道原始顺序。这对于自然语言来说是不够的（“我爱你” vs “你爱我”）。

为了弥补这一点，我们需要显式地注入位置信息。此外，为了训练深层 Transformer，归一化技术至关重要。

#### 3.3.1 正弦位置编码 (Sinusoidal Positional Encoding)

Google 团队选择了一种基于三角函数的编码方式，而不是学习 **位置嵌入 (Position Embedding)**。

**公式**：
$$
\begin{aligned}
PE_{(pos, 2i)} &= \sin(pos / 10000^{2i/d_{model}}) \\
PE_{(pos, 2i+1)} &= \cos(pos / 10000^{2i/d_{model}})
\end{aligned}
$$

*   $pos$: 单词在句子中的位置。
*   $i$: 维度索引。

**可视化**：
下图展示了 $d_{model}=128$ 时的前 50 个位置的编码。可以看到，每一个维度对应一个波长不同的正弦波。波长从 $2\pi$ 到 $10000 \cdot 2\pi$ 几何级数变化。

<img src="chapter_03/images/positional_encoding.png" width="85%" />

**数学性质**：
这种编码方式允许模型轻松学习到 **相对位置**。
对于任意固定偏移 $k$，$\mathbf{PE}_{pos+k}$ 可以表示为 $\mathbf{PE}_{pos}$ 的线性函数（旋转矩阵）：
$$
\begin{bmatrix} \sin(pos+k) \\ \cos(pos+k) \end{bmatrix} =
\begin{bmatrix} \cos k & \sin k \\ -\sin k & \cos k \end{bmatrix}
\begin{bmatrix} \sin(pos) \\ \cos(pos) \end{bmatrix}
$$
这意味着 Attention 机制可以通过简单的线性投影来关注“相对位置”信息（例如：关注前一个词）。

#### 3.3.2 层归一化 (Layer Normalization)

**批归一化 (Batch Normalization, BN)** 在 CNN 中很常用，但在 RNN/Transformer 中表现不佳（因为序列长度可变，Batch 统计量不稳定）。
Transformer 使用 **层归一化 (Layer Normalization, LN)**。

**公式**：
对于一个样本的一个层输入向量 $\mathbf{x} \in \mathbb{R}^{d}$，计算其自身的均值和方差：
$$ \mu = \frac{1}{d} \sum_{j=1}^d x_j, \quad \sigma^2 = \frac{1}{d} \sum_{j=1}^d (x_j - \mu)^2 $$
$$ \text{LN}(\mathbf{x}) = \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta $$
其中 $\gamma, \beta$ 是可学习的仿射变换参数。

*   **BN vs LN**:
    *   **BN**: 对 Batch 维度归一化（纵向切）。不同样本之间相互依赖。
    *   **LN**: 对 Feature 维度归一化（横向切）。每个样本独立计算。

#### 3.3.3 残差连接与 Pre-Norm vs Post-Norm

Transformer 广泛使用了残差连接：$x + \text{Sublayer}(x)$。关于 LN 的位置，有两种变体：

1.  **Post-Norm (原始论文)**：
    $$ x_{out} = \text{LN}(x + \text{Sublayer}(x)) $$
    *   问题：梯度在反向传播时可能会在输出层附近激增，导致深层网络难以训练（需要 Warmup）。

2.  **Pre-Norm (现代 LLM 标配，如 GPT-2/3, LLaMA)**：
    $$ x_{out} = x + \text{Sublayer}(\text{LN}(x)) $$
    *   优势：梯度更容易沿恒等路径传播，深层模型通常更稳定，也能减轻对长 warmup 的依赖。

<span style="background-color: #DAE8FC; color: black; padding: 2px 4px; border-radius: 4px;">Key Takeaway</span>
Pre-Norm 结构是现代大模型能够堆叠到很深层数的重要工程细节之一，但它仍需要与初始化、学习率、残差缩放和归一化细节共同工作。
<a id="section-3-4"></a>

## 3.4 训练与推理：掩码与缓存
### 3.4 Training and Inference Techniques

Transformer 的训练和推理过程有着显著的差异，特别是在 Decoder 部分。理解这些差异对于掌握大模型的高效部署至关重要。

#### 3.4.1 训练时的 Teacher Forcing 与 Masked Attention

在训练阶段，我们拥有完整的标准答案（Ground Truth）。为了加速训练，我们使用 **教师强制 (Teacher Forcing)**：一次性把整个正确句子的 Embedding 喂给 Decoder。

从损失函数角度看，这对应于“并行地”计算每个位置的负对数似然（以 Decoder-only 为例）：

$$ \mathcal{L}_{\text{CLM}} = -\sum_{t=1}^{T} \log P(x_t \mid x_{<t}) $$

**问题**：Decoder 是自回归的。如果在预测第 $t$ 个词时，它能“偷看”到第 $t+1$ 个词，那它就不用学了，直接照抄就行。这会导致测试时（没有答案可抄）性能崩塌。

**解决方案：掩码自注意力 (Masked Self-Attention)**
我们需要一个 **因果掩码 (Causal Mask)**。它是一个上三角矩阵（对应未来的位置），值为 $-\infty$。

$$ \text{Mask}(i, j) = \begin{cases} 0 & \text{if } i \ge j \\ -\infty & \text{if } i < j \end{cases} $$

在 Softmax 之前加上 Mask：
$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M \right)V $$

这样，位置 $i$ 的 Query 就只能关注到位置 $j \le i$ 的 Key，无法关注未来的信息。

**可视化直觉**：下图展示了一个小尺寸的因果掩码矩阵（上三角为屏蔽区域）以及它对注意力权重的影响。

<img src="chapter_03/images/causal_mask_demo.png" width="80%" />

#### 3.4.2 推理时的自回归 (Autoregressive Generation)

在推理（Inference）阶段，模型必须一个词一个词地生成。
Step 1: 输入 `<SOS>`, 输出 `The`
Step 2: 输入 `<SOS> The`, 输出 `cat`
Step 3: 输入 `<SOS> The cat`, 输出 `sat`
...

#### 3.4.3 KV Cache：推理加速的关键

你发现了吗？在 Step 3 时，我们重新计算了 `<SOS>` 和 `The` 的 Key 和 Value。这在 Step 2 已经算过一次了！
对于 GPT 这样的大模型，重复计算是非常昂贵的。

**KV Cache 原理**：
我们将之前步骤计算好的 Key 和 Value 矩阵 **缓存** 在显存中。
在生成第 $t$ 个 token 时：
1.  只计算当前 token $\mathbf{x}_t$ 的 $q_t, k_t, v_t$。
2.  从 Cache 取出之前的 $K_{<t}, V_{<t}$。
3.  拼接得到完整的 $K_{\le t} = [K_{<t}; k_t]$。
4.  计算 Attention。
5.  将 $k_t, v_t$ 存入 Cache。

**图解**：

```mermaid
graph LR
    subgraph StepT ["Step t"]
        Input_t["Token x_t"] --> Linear

        Linear --> q_t
        Linear --> k_t
        Linear --> v_t

        Cache["KV Cache (GPU Memory)"] -->|Read K_prev V_prev| Concat
        k_t --> Concat
        v_t --> Concat
        Concat -->|Write k_t v_t| Cache

        Concat --> Full_KV["Full K, V"]
        q_t --> Attention
        Full_KV --> Attention
    end

    style Cache fill:#FFF2CC,stroke:#D6B656,stroke-width:2px,color:#000000
    style Input_t fill:#F5F5F5,stroke:#666666,stroke-width:2px,color:#000000
    style q_t fill:#DAE8FC,stroke:#6C8EBF,stroke-width:2px,color:#000000
    style k_t fill:#DAE8FC,stroke:#6C8EBF,stroke-width:2px,color:#000000
    style v_t fill:#DAE8FC,stroke:#6C8EBF,stroke-width:2px,color:#000000
    style Attention fill:#E1D5E7,stroke:#9673A6,stroke-width:2px,color:#000000
```

<span style="background-color: #F8CECC; color: black; padding: 2px 4px; border-radius: 4px;">Tradeoff</span>
KV Cache 是典型的 **空间换时间**。
*   **优点**：大幅减少 FLOPs，推理速度提升。
*   **缺点**：显存占用随序列长度线性增加。对于长文本（如 128k context），KV Cache 可能会撑爆显存，这催生了 **MQA (Multi-Query Attention)** 和 **GQA (Grouped-Query Attention)** 等变体（减少 K, V 的头数）。

为了建立更直观的工程感知，下图给出了一个简化的“KV Cache 显存随序列长度增长”的示意曲线：

<img src="chapter_03/images/kv_cache_memory_curve.png" width="80%" />

---

本节从训练与推理角度梳理了 Transformer 的关键工程机制。它是现代语言模型和大量多模态系统的核心组件，也深刻影响了扩散模型、视觉模型和长上下文系统的设计。

不过，KV Cache 也把问题推向了下一层：当上下文长度扩大到十万、百万 token 时，模型不仅要少做重复计算，还要重新设计注意力连接、缓存表示和底层 kernel。下一节我们将讨论 **[3.5 前沿注意力架构](chapter_03.md#section-3-5)**：MLA、稀疏注意力、FlashAttention 与长上下文效率。
<a id="section-3-5"></a>

## 3.5 前沿注意力架构：从稠密矩阵到长上下文效率
### 3.5 Frontier Attention Architectures: From Dense Matrices to Long-Context Efficiency

到 3.4 节为止，我们已经看到 Transformer 的核心矛盾：注意力机制提供了全局依赖建模能力，但它也带来了两类成本。

第一类是训练成本。对长度为 $n$ 的序列，标准自注意力需要构造 $n \times n$ 的分数矩阵：

$$
S = \frac{QK^T}{\sqrt{d_k}}, \qquad S \in \mathbb{R}^{n \times n}.
$$

如果隐藏维度和头数先固定，则注意力分数计算和注意力权重存储都随 $n^2$ 增长。第二类是推理成本。自回归生成时可以用 KV Cache 避免重复计算历史 token，但缓存本身仍随上下文长度线性增长：

$$
\text{KV cache size} \propto L \cdot n_{\text{layers}} \cdot n_{\text{kv-heads}} \cdot d_{\text{head}}.
$$

长上下文模型要处理代码库、长文档、长视频和复杂 Agent 轨迹，就必须同时面对这两个问题：**训练时的二次方注意力矩阵**，以及 **推理时持续膨胀的 KV Cache**。

因此，2024 年之后的前沿架构研究不只是“把窗口拉长”，而是在寻找一组更细的工程-数学折中：哪些 token 必须被精确关注？哪些历史信息可以压缩？哪些操作需要配合 GPU/加速器的内存层级？哪些稀疏模式可以从预训练开始就让模型适应？

#### 3.5.1 稠密注意力、局部注意力与动态稀疏

设第 $i$ 个 query 可见的 key 集合为 $\mathcal{A}(i)$。在因果自注意力中，

$$
\mathcal{A}_{\text{full}}(i) = \{1,2,\dots,i\}.
$$

标准注意力为：

$$
\mathbf{o}_i
= \sum_{j \in \mathcal{A}_{\text{full}}(i)}
\frac{\exp(q_i^T k_j/\sqrt{d_k})}
{\sum_{\ell \in \mathcal{A}_{\text{full}}(i)} \exp(q_i^T k_\ell/\sqrt{d_k})}
v_j.
$$

稀疏注意力的基本思想是把可见集合替换成一个较小的集合 $\mathcal{S}(i) \subseteq \mathcal{A}_{\text{full}}(i)$：

$$
\mathbf{o}^{\text{sparse}}_i
= \sum_{j \in \mathcal{S}(i)}
\frac{\exp(q_i^T k_j/\sqrt{d_k})}
{\sum_{\ell \in \mathcal{S}(i)} \exp(q_i^T k_\ell/\sqrt{d_k})}
v_j.
$$

如果平均每个 query 只看 $m$ 个 key，注意力复杂度可从近似 $O(n^2d)$ 下降到 $O(nmd)$。关键问题是：$\mathcal{S}(i)$ 怎么选？

常见路线包括：

*   **局部窗口 (Sliding Window)**：每个位置只看最近 $w$ 个 token。复杂度约为 $O(nwd)$，适合连续局部依赖，但可能漏掉远距离证据。
*   **固定稀疏模式 (Fixed Sparse Pattern)**：例如分块、跨步、膨胀窗口、少量全局 token。OpenAI 的 Sparse Transformer 早期展示了用固定/分解稀疏模式建模长序列的可行性。
*   **动态稀疏选择 (Dynamic Sparse Selection)**：用轻量索引器或打分器为每个 query 选择最相关的历史块或 token。它更灵活，但实现难度更高，也更依赖高效 kernel。
*   **混合结构 (Hybrid Attention)**：局部窗口保证近邻精度，压缩路径提供全局轮廓，动态选择补充远距离关键证据。

下图把这些路线放在同一张图里。它不是某个具体模型的精确实现图，而是帮助理解长上下文架构的设计空间。

<img src="chapter_03/images/attention_efficiency_frontier.svg" width="95%" />

#### 3.5.2 OpenAI Sparse Transformer：早期先例

OpenAI 在 2019 年提出的 Sparse Transformer 已经指出，标准 Transformer 在长序列上受限于二次方注意力成本。它用稀疏分解的注意力矩阵把复杂度从 $O(n^2)$ 降到约 $O(n\sqrt{n})$，并在图像、音频和字节级文本建模上验证了长序列自回归建模的潜力。

这条路线的重要性不在于它直接等于后来的工业长上下文模型，而在于它提前暴露了一个核心判断：**长序列建模不能只看理论复杂度，还要看注意力模式是否适合硬件、是否保留任务所需的远距离信息，以及模型是否在训练阶段就学会这种稀疏结构。**

#### 3.5.3 FlashAttention：不是稀疏，但改变了注意力工程

FlashAttention 解决的是另一层问题。它不近似注意力结果，也不改变 $\mathcal{A}(i)$；它仍然计算精确注意力，但用分块和在线 softmax 方式减少 GPU 高带宽显存 (HBM) 与片上 SRAM 之间的读写。

这解释了为什么“算法复杂度”不等于“实际速度”。一个理论上更低复杂度的近似注意力，如果访存模式很差，可能没有真实 wall-clock 收益；一个仍为二次复杂度的精确注意力，如果 IO 模式更好，反而可以显著加速训练和推理。

因此，现代长上下文架构通常同时关心三件事：

*   **算术复杂度**：是否减少了需要计算的注意力分数数量。
*   **访存复杂度**：是否减少 HBM 读写和中间矩阵落盘。
*   **kernel 可实现性**：是否能在 CUDA、ROCm、NPU 或其他加速器上稳定高效执行。

#### 3.5.4 DeepSeek 路线：MLA、MoE 与稀疏注意力

DeepSeek-V3 的技术报告把效率问题拆成了几条相互配合的路径。

首先是 **MoE (Mixture-of-Experts)**。DeepSeek-V3 是一个总参数量很大的 MoE 模型，但每个 token 只激活部分专家。这样可以扩大模型容量，同时控制单个 token 的前向计算量。MoE 的主要代价是路由、负载均衡、通信和训练稳定性。

其次是 **MLA (Multi-head Latent Attention)**。标准 MHA 会为每个 token 存储每一层的 key/value。GQA/MQA 通过减少 KV 头数降低缓存；MLA 更进一步，把 key/value 信息压缩到潜在向量中，再在需要时重构注意力所需的表示。粗略写成：

$$
c_t^{KV} = W^{DKV} h_t, \qquad
k_t^C = W^{UK} c_t^{KV}, \qquad
v_t^C = W^{UV} c_t^{KV}.
$$

这里 $c_t^{KV}$ 是被缓存的潜在表示，$k_t^C, v_t^C$ 是从潜在表示恢复出的 key/value 成分。实际 MLA 还会处理与 RoPE 相关的解耦 key 成分。对综述读者来说，重点是：MLA 把“缓存完整 K/V”改成“缓存较小的潜在状态”，从而降低长上下文推理中的显存压力。

再次是 **NSA / DSA 类稀疏注意力**。2025 年的 Native Sparse Attention (NSA) 论文提出硬件对齐、可端到端训练的稀疏注意力机制，结合粗粒度 token 压缩、细粒度 token 选择和局部精确路径，在长上下文场景中降低训练与推理成本。DeepSeek-V3.2-Exp 的官方模型页进一步把 DeepSeek Sparse Attention (DSA) 作为实验性架构重点，目标是在长上下文训练和推理中提高效率，同时尽量保持模型输出质量。

这条路线的关键不只是“少算一些注意力分数”，而是让稀疏结构进入模型生命周期：

*   **预训练阶段**：模型从一开始就适应稀疏可见集合，而不是把稀疏化作为后期压缩补丁。
*   **推理阶段**：索引器、稀疏 kernel、KV cache 布局共同决定真实延迟。
*   **长上下文任务**：局部窗口、全局压缩和动态选择需要同时处理“最近上下文”和“远距离证据”。

#### 3.5.5 Google、Meta 与长上下文系统化

Google 的 Gemini 1.5 技术报告把长上下文能力推到了百万 token 量级，并强调跨长文档、长视频、长音频的细粒度召回与推理。这说明长上下文已经从“能塞进去更多 token”变成了多模态系统能力：模型不仅要接收长输入，还要在很长的上下文中可靠定位证据。

Meta 的 Llama 3 技术报告则代表另一条重要路线：开放权重、大规模 dense Transformer、128K 上下文、工具使用，以及围绕评测与安全的配套系统。它并不等同于稀疏注意力路线，但对于综述教材很重要，因为开放权重模型让研究者能够复现实验、检查长上下文行为、测试压缩与微调方法。

这些方向共同说明：长上下文不是单一架构技巧，而是模型架构、训练数据、位置编码、推理 kernel、缓存管理、评测任务和系统接口共同作用的结果。

#### 3.5.6 稀疏注意力的风险与边界

稀疏注意力很诱人，但它不是免费的。

*   **召回风险**：如果关键证据没有被选入 $\mathcal{S}(i)$，后续 softmax 再精确也无济于事。
*   **训练-推理不一致**：如果训练时用全注意力、推理时再稀疏化，模型可能没有学会在受限可见集合下工作。
*   **评测偏差**：needle-in-a-haystack 类任务能测试召回，但不能完全代表长文档推理、代码库修改和 Agent 轨迹分析。
*   **硬件依赖**：不规则稀疏模式可能节省 FLOPs，却增加索引、访存和 kernel 调度开销。
*   **可解释性错觉**：稀疏模式看起来像“模型选择了重点”，但它仍是训练目标和优化过程产生的中间计算结构，不能直接等同于人类注意力或理解。

所以，更稳妥的表述是：稀疏注意力、MLA、MoE、FlashAttention 和长上下文评测共同构成了 2025-2026 年架构效率研究的重要方向。它们并不推翻 Transformer，而是在保留注意力表达能力的同时，重写注意力的成本结构。

---

本节把 3.4 节的 KV Cache 问题推进到当前前沿：长上下文模型需要同时压缩缓存、减少注意力连接、优化访存路径，并在训练阶段就让模型适应这些结构。下一节转向另一条长序列路线：**[Mamba 与状态空间模型](chapter_03.md#section-3-6)**。
<a id="section-3-6"></a>

## 3.6 Mamba 与状态空间模型：注意力之外的长序列路线
### 3.6 Mamba and State Space Models: A Long-Sequence Route Beyond Attention

Transformer 的成功来自注意力机制：每个 token 都可以直接访问历史 token。但这种全局访问也带来了 $O(n^2)$ 的注意力矩阵和随长度增长的 KV Cache。于是，一个自然问题是：能不能保留长距离建模能力，同时回到近似线性的序列计算？

**状态空间模型 (State Space Model, SSM)** 是这条路线的核心。它并不是 2020 年代才出现的概念；控制论、信号处理和时间序列分析早就使用状态空间方程。现代 SSM 的新意在于：把状态空间递推做成可训练、可并行、适合 GPU 的深度学习层。

#### 3.6.1 最小状态空间方程

离散时间下，一个线性状态空间模型可以写成：

$$
h_t = A h_{t-1} + B x_t,\qquad
y_t = C h_t + D x_t.
$$

其中：

*   $x_t$ 是当前位置输入。
*   $h_t$ 是隐状态，负责携带历史信息。
*   $y_t$ 是输出。
*   $A,B,C,D$ 是可学习参数或由参数生成的结构。

这个形式看起来像 RNN，但有两个关键差异。

第一，现代 SSM 通常对 $A$ 做结构化参数化，使长序列递推更稳定。第二，在训练时，线性递推可以转写成卷积或 scan，从而并行计算，不必像传统 RNN 那样严格逐 token 串行。

如果 $A,B,C,D$ 固定，SSM 对历史的读写是内容无关的；这正是早期 SSM 在语言任务上不如注意力的原因之一。语言序列需要根据当前 token 决定“记住什么、忘掉什么、输出什么”。

更完整一点，许多 SSM 从连续时间方程出发：

$$
\frac{d h(t)}{dt} = A h(t) + B x(t),\qquad
y(t) = C h(t) + D x(t).
$$

为了处理离散 token 序列，需要用步长 $\Delta$ 离散化。常见写法是：

$$
\bar{A} = \exp(\Delta A),\qquad
\bar{B} = \left(\int_0^\Delta \exp(\tau A)\,d\tau\right)B.
$$

于是递推变成：

$$
h_t = \bar{A}h_{t-1} + \bar{B}x_t.
$$

如果 $\bar{A},\bar{B},C$ 固定，输出可以展开成卷积：

$$
y_t = C h_t + D x_t
= \sum_{i=0}^{t} C\bar{A}^{i}\bar{B}x_{t-i} + D x_t.
$$

因此 SSM 同时有两种视角：

*   **递推视角**：推理时维护一个状态 $h_t$，每来一个 token 更新一次。
*   **卷积视角**：训练时把历史影响写成卷积核 $K_i=C\bar{A}^{i}\bar{B}$，并行处理整段序列。

这正是它和传统 RNN 的关键区别：RNN 主要依赖逐步递推；结构化 SSM 可以利用卷积或 scan 获得更好的训练并行性。

#### 3.6.2 Mamba 的选择性：让状态更新依赖输入

Mamba 的核心改动是 **选择性状态空间 (Selective SSM)**：让部分状态空间参数成为输入的函数。粗略写成：

$$
h_t = \bar{A}(x_t) h_{t-1} + \bar{B}(x_t) x_t,\qquad
y_t = C(x_t) h_t.
$$

更接近实现的说法是，Mamba 通过输入相关的 $\Delta_t, B_t, C_t$ 控制状态更新。这样，模型可以在看到某个 token 时选择：

*   把它写入状态。
*   快速遗忘它。
*   把它作为当前输出的主要证据。

这使 Mamba 不再只是固定滤波器，而更像一种内容自适应的压缩记忆。

选择性的代价是：一旦 $\bar{A}_t,\bar{B}_t,C_t$ 随输入变化，前面“固定卷积核”的写法就不再成立。模型不能再简单使用同一个 $K_i$ 处理所有位置，因为第 $t$ 步到底保留多少历史，取决于当前位置输入。Mamba 的贡献之一，就是在保留输入相关门控的同时，仍然让训练阶段可以高效并行。

下图给出一个简化视角：Transformer 的注意力是显式从历史 token 集合中读；Mamba 则把历史逐步压入状态，并在每一步根据输入控制状态更新。

<img src="chapter_03/images/mamba_ssm_flow.svg" width="95%" />

#### 3.6.3 Selective Scan：训练并行与推理递推

Mamba 的工程关键是 **selective scan**。它要同时满足两个看似矛盾的要求：

*   训练时能并行处理长序列。
*   推理时能像 RNN 一样维护一个小状态，逐 token 更新。

为什么 scan 可以并行？把一维状态递推简化写成：

$$
h_t = a_t h_{t-1} + b_t.
$$

从 $h_{t-2}$ 到 $h_t$ 的两步组合为：

$$
h_t = a_t(a_{t-1}h_{t-2}+b_{t-1})+b_t
= (a_ta_{t-1})h_{t-2} + (a_tb_{t-1}+b_t).
$$

也就是说，每一步可以看成一个仿射变换 $(a_t,b_t)$，两个相邻变换的组合为：

$$
(a_2,b_2)\circ(a_1,b_1)
= (a_2a_1,\;a_2b_1+b_2).
$$

这个组合运算满足结合律，因此可以用并行前缀扫描 (parallel prefix scan) 计算整段序列。真实 Mamba 的状态是向量/矩阵结构，公式更复杂，但核心思想仍是把递推变成可组合的结构化运算。

从复杂度看，标准注意力的序列长度成本近似为：

$$
O(n^2 d).
$$

而 Mamba 这类选择性 SSM 的序列扫描成本近似为：

$$
O(n d s),
$$

其中 $s$ 是状态维度或展开因子。实际速度还取决于 fused kernel、显存访存、batch 形状和硬件实现。Mamba 论文报告了线性序列扩展和较高推理吞吐，但这不意味着它在所有语言任务上都稳定替代 Transformer。

#### 3.6.4 Mamba-2 与 SSM/Attention 的统一视角

Mamba-2 进一步提出 **状态空间对偶 (State Space Duality, SSD)**：某些结构化 SSM 和注意力变体可以放进同一个半可分矩阵 (semiseparable matrix) 框架中理解。

把序列模型写成矩阵形式，有助于看清这点。注意力可以看成一个下三角的 token-token 混合矩阵：

$$
Y_i = \sum_{j\le i} A_{ij}V_j.
$$

SSM 展开后也可以写成类似形式：

$$
y_i = \sum_{j\le i} C_i
\left(\prod_{k=j+1}^{i} A_k\right)
B_j x_j.
$$

区别在于：注意力里的 $A_{ij}$ 通常由 $q_i^Tk_j$ 直接计算，显式比较任意两个位置；SSM 的权重由状态转移链条间接产生，有更强的结构约束。Mamba-2 的意义在于把这两类混合矩阵放到共同框架中分析，从而解释为什么一些 SSM 可以接近注意力的表达能力，同时保持更好的长序列复杂度。

这很重要，因为它降低了“Transformer vs SSM”的二分感。更准确地说：

*   注意力擅长内容寻址：显式比较当前 query 与历史 key。
*   SSM 擅长流式压缩：用固定大小状态携带长历史。
*   现代混合模型可能同时使用注意力、SSM、卷积和门控 MLP。

因此，Mamba 不是“反 Transformer 宣言”，而是对长序列成本结构的一次重要探索。

#### 3.6.5 优势、风险与适用位置

**优势**：

*   **长序列线性扩展**：避免显式构造 $n \times n$ 注意力矩阵。
*   **流式推理友好**：可维护隐状态，适合音频、时间序列、传感器数据和长文档扫描。
*   **跨模态潜力**：语言、音频、基因组和部分视觉序列都可以写成长序列问题。

**风险**：

*   **内容检索能力**：某些精确回溯、复制、引用、代码跳转任务仍更适合显式注意力或外部检索。
*   **生态成熟度**：Transformer 的训练技巧、kernel、评测和工程框架更成熟。
*   **可解释性**：状态向量是压缩记忆，不能像注意力矩阵那样直接画出 token-token 关联。

到 2026 年，从综述角度看，Mamba/SSM 应放在“非 Transformer 架构探索”的主线位置。它不是边角料，而是长序列建模、流式模型和硬件友好架构的重要候选；但它也还没有把 Transformer 从通用 LLM 主干位置上整体替换下来。

---

本节补上了注意力之外的长序列路线。下一节继续沿着“效率”这条线，讨论 DeepSeek 系列论文中围绕条件记忆、上下文压缩、稀疏注意力、MoE 与推理吞吐的若干设计。
<a id="section-3-7"></a>

## 3.7 DeepSeek 系列效率论文：记忆、上下文与吞吐
### 3.7 DeepSeek Efficiency Papers: Memory, Context, and Throughput

如果把 2024-2026 年的大模型研究只理解成“模型名和榜单”，会漏掉一条更重要的线索：头部模型团队在反复拆解同一个工程问题。

这个问题是：

> 如何在保持模型能力的同时，降低长上下文、推理延迟、显存占用和训练成本？

DeepSeek 系列论文和开源实现提供了一个很适合综述教材观察的案例。它不是单一技术，而是一组围绕效率的组合：MLA、MoE、MTP、FP8 训练、稀疏注意力、条件记忆、上下文压缩、OCR 压缩和推理系统优化。

#### 3.7.1 DeepSeek-V3：MLA、MoE、MTP 与训练效率

DeepSeek-V3 技术报告中的几个关键词经常被一起提到，但它们解决的问题不同。

*   **MLA (Multi-head Latent Attention)**：压缩 KV Cache，降低长上下文推理显存。
*   **DeepSeekMoE**：扩大总参数量，但每个 token 只激活部分专家，降低单 token 计算量。
*   **MTP (Multi-Token Prediction)**：让模型在训练时预测多个未来 token，为表示学习和推理加速提供额外信号。
*   **FP8 混合精度训练**：降低训练显存和带宽压力，但需要处理数值稳定性。
*   **负载均衡与并行策略**：MoE 训练不只看模型结构，还要解决专家路由不均、通信开销和流水线气泡。

这些设计共同说明：现代大模型架构已经不是“Transformer block + 参数量”这么简单，而是模型层、训练层和系统层共同优化。

从计算结构看，几项技术可以写得更明确。

**MLA** 的目标是减少每个 token 需要缓存的 Key/Value 维度。标准 MHA 推理时缓存：

$$
K_t,V_t \in \mathbb{R}^{n_{\text{heads}}\times d_{\text{head}}}.
$$

MLA 改成缓存较小的潜在向量 $c_t^{KV}$，再从它恢复注意力所需的表示：

$$
c_t^{KV}=W^{DKV}h_t,\qquad
k_t^C=W^{UK}c_t^{KV},\qquad
v_t^C=W^{UV}c_t^{KV}.
$$

因此，推理时长上下文显存压力主要由 $c_t^{KV}$ 的维度决定，而不是完整 $K,V$ 的维度。它牺牲的是一些重构和投影计算，换来更低的 KV cache 占用。

**MoE** 则把 FFN 层拆成多个专家。对第 $t$ 个 token，路由器给出专家权重：

$$
g_t = \operatorname{softmax}(W_r h_t).
$$

如果只选择 top-$k$ 个专家，则输出近似为：

$$
\operatorname{MoE}(h_t)
= \sum_{e\in \operatorname{TopK}(g_t)}
g_{t,e}\,\operatorname{FFN}_e(h_t).
$$

这样，总参数量可以很大，但每个 token 只经过少数专家。问题是路由可能不均衡：如果大量 token 都挤向少数专家，会造成通信瓶颈和训练不稳定，所以 MoE 系统通常还需要负载均衡损失、专家并行和通信优化。

**MTP** 可以理解为给语言模型增加更远的预测目标。普通 causal LM 只预测下一个 token：

$$
\mathcal{L}_{1}=-\sum_t \log p_\theta(x_{t+1}\mid x_{\le t}).
$$

多 token 预测会把未来多个位置加入目标：

$$
\mathcal{L}_{\text{MTP}}
= -\sum_t\sum_{j=1}^{m}
\lambda_j \log p_\theta(x_{t+j}\mid x_{\le t}).
$$

它的直觉是迫使隐藏状态携带更长一步的未来信息。对推理服务来说，如果模型或附加头能较可靠地预测多个后续 token，就能和投机解码、多 token decode 等技术形成联系。

#### 3.7.2 NSA / DSA：从全注意力走向可训练稀疏注意力

第 3.5 节已经介绍了 Native Sparse Attention (NSA) 和 DeepSeek Sparse Attention (DSA) 的基本思想。这里再从 DeepSeek 系列工程目标看一遍。

标准长上下文模型的问题是：即使某个 token 只真正需要少数远距离证据，稠密注意力仍然要计算大量 token-token 分数。NSA/DSA 的路线是让模型学习或使用一套混合可见集合：

$$
\mathcal{S}(i) =
\mathcal{W}(i) \cup \mathcal{C}(i) \cup \mathcal{R}(i),
$$

其中：

*   $\mathcal{W}(i)$ 是局部窗口，保留最近上下文。
*   $\mathcal{C}(i)$ 是压缩表示，提供全局轮廓。
*   $\mathcal{R}(i)$ 是动态选择的远距离相关 token 或 block。

这种方法的风险也很明确：如果关键证据没有进入 $\mathcal{S}(i)$，模型就会出现“看似长上下文，实际没读到”的失败。因此，稀疏注意力必须和训练目标、索引器、评测集、KV 布局、kernel 实现一起设计。

更具体地说，混合稀疏注意力不是简单“随机少看一些 token”，而是把不同信息源分工：

$$
o_i =
\operatorname{Attn}\big(q_i,K_{\mathcal{W}(i)},V_{\mathcal{W}(i)}\big)
\;+\;
\operatorname{Attn}\big(q_i,K_{\mathcal{C}(i)},V_{\mathcal{C}(i)}\big)
\;+\;
\operatorname{Attn}\big(q_i,K_{\mathcal{R}(i)},V_{\mathcal{R}(i)}\big),
$$

或者用门控权重组合多个分支：

$$
o_i = \sum_{b\in\{\text{window},\text{compressed},\text{selected}\}}
\alpha_{i,b} o_{i,b}.
$$

窗口分支负责局部连续性，压缩分支负责全局摘要，选择分支负责找远距离关键证据。真正难的地方是 $\mathcal{R}(i)$：如果动态选择器本身太贵，稀疏注意力就省不到成本；如果选择器太弱，就会漏掉关键证据。

2026 年前后的进一步问题是：当 DSA 的 token 级索引器已经足够细，它自己也会变成长上下文推理的成本中心。若第 $t$ 个 query 的前缀长度为 $L$，索引器用 $H^I$ 个索引头给每个历史 token 打分，可以抽象为：

$$
I_{t,s}
=\sum_{j=1}^{H^I}
w^I_{t,j}\,
\operatorname{ReLU}\!\left((q^I_{t,j})^T k^I_s\right),
\qquad
\mathcal{T}_t=\operatorname{TopK}(I_{t,:},k).
$$

主注意力只看 $\mathcal{T}_t$ 中的 token，但索引器为了得到 $\mathcal{T}_t$ 仍要做约 $O(H^I L)$ 的打分。MISA (Mixture of Indexer Sparse Attention) 这类方法的观察是：所有索引头共同提供多样性，但对单个 query 来说，不一定每个头都同样重要。于是可以先用块级统计选择少量活跃索引头：

$$
E_{t,j}
=\frac{1}{M}\sum_{b=1}^{M}
\left|
w^I_{t,j}\,
\operatorname{ReLU}\!\left((q^I_{t,j})^T \tilde{k}^I_b\right)
\right|,
\qquad
\mathcal{H}_t=\operatorname{TopK}_{j}(E_{t,j},h),
$$

再只用 $\mathcal{H}_t$ 中的 $h\ll H^I$ 个头做 token 级扫描：

$$
I^{\text{MISA}}_{t,s}
=\sum_{j\in\mathcal{H}_t}
w^I_{t,j}\,
\operatorname{ReLU}\!\left((q^I_{t,j})^T k^I_s\right).
$$

这样索引成本从 $O(H^I L)$ 变成近似 $O(H^I M+hL)$，其中 $M$ 是块数且 $M\ll L$。这说明长上下文效率的竞争已经细到“谁来选择注意力候选、选择器本身怎么省”的层面，而不只是主注意力矩阵稀疏化。

#### 3.7.3 Engram：条件记忆与可查表知识

Engram / Conditional Memory 的思想可以理解为：并不是所有知识都必须挤在 Transformer 参数和上下文窗口里。一部分高频模式、短语、事实或局部结构可以通过可扩展查表结构存储，并在推理时按条件检索。

抽象地说，普通语言模型把知识压进参数：

$$
P_\theta(x_t \mid x_{<t}).
$$

条件记忆路线则引入一个外部或半外部的记忆检索项：

$$
P(x_t \mid x_{<t}) \approx f_\theta(x_{<t}, M[\phi(x_{<t})]),
$$

其中 $\phi(x_{<t})$ 是上下文特征或哈希键，$M[\cdot]$ 是可查表记忆。它与 RAG 有相似动机，但粒度不同：RAG 通常检索文档块，Engram 更像在模型内部附近增加可扩展的模式记忆。

这类方法值得放进 AI 综述，因为它挑战了一个常见假设：模型能力不一定全部来自“更大参数 + 更长上下文”。在某些场景中，可查表记忆、检索缓存、短语级统计和模型前向计算可以形成混合系统。

从概率建模看，条件记忆可以看作模型分布和记忆分布的混合。设语言模型给出：

$$
p_{\theta}(x_t\mid x_{<t}),
$$

记忆表根据上下文键 $\phi(x_{<t})$ 检索候选 token 或短语，得到：

$$
p_M(x_t\mid \phi(x_{<t})).
$$

再由门控函数决定是否相信记忆：

$$
p(x_t\mid x_{<t})
= (1-\lambda_t)p_{\theta}(x_t\mid x_{<t})
+ \lambda_t p_M(x_t\mid \phi(x_{<t})).
$$

其中 $\lambda_t\in[0,1]$ 可以由上下文、模型隐藏状态或检索置信度决定。这个式子解释了 Engram 一类方法和 RAG 的差别：RAG 往往把文档块塞进上下文，让 Transformer 自己读；条件记忆则更像在模型输出分布附近接入一个可扩展的模式表。

#### 3.7.4 DeepSeek-OCR：上下文压缩不只发生在文本里

DeepSeek-OCR 的标题是 *Contexts Optical Compression*。它把文档、图像和 OCR 理解放进同一个问题：如果视觉文档里的信息可以被压缩成更短、更结构化的 token 序列，长上下文压力就会下降。

这条线与普通 OCR 不同。传统 OCR 更像“图像转文字”；上下文光学压缩关心的是：

*   哪些视觉区域值得保留？
*   表格、公式、版式、图表如何变成可推理的紧凑表示？
*   长文档是否能先在视觉侧压缩，再进入语言模型上下文？

从系统角度看，这和 DSA/MLA/Engram 是同一类问题：减少无效 token，让模型把计算花在高信息密度的表示上。

可以把上下文压缩抽象成一个编码器：

$$
z = C_{\psi}(o),
$$

其中 $o$ 是原始视觉文档，$z$ 是较短的视觉/文本混合表示。语言模型真正处理的是：

$$
p_\theta(y\mid z, x).
$$

如果 $|z|\ll |o|$ 且 $z$ 保留了版式、文字、表格和图像证据，就能降低后续长上下文压力。风险也同样明确：压缩器如果丢掉了小字、脚注、表格结构或公式细节，后面的语言模型无法凭空恢复。因此上下文压缩必须和任务评测绑定，而不能只看 token 压缩率。

#### 3.7.5 推理速度：输出 token 已经成为竞争对象

用户体验中的速度通常不是一个指标，而是一组指标：

*   **TTFT (Time To First Token)**：首 token 延迟。
*   **TPOT (Time Per Output Token)**：每个输出 token 的平均生成时间。
*   **Throughput**：单位时间服务多少请求或生成多少 token。
*   **Cost per token**：单 token 成本。

DeepSeek-V3 的 MLA/MoE/MTP，V3.2-Exp 的 DSA，DeepSeek-OCR 的上下文压缩，以及 Engram 的条件记忆，都可以放进这个速度框架里理解：它们不是只为了论文指标，而是在降低“读长上下文”和“吐长答案”的系统成本。

这也解释了为什么 2025-2026 年的模型竞争不再只拼 benchmark 分数。真实产品会同时拼：

*   模型质量。
*   输出速度。
*   长上下文利用率。
*   工具调用稳定性。
*   单位 token 成本。
*   批量服务吞吐。

---

本节把 DeepSeek 相关论文放回技术脉络中：它们共同指向大模型效率的几条主线。后续第 5 章会从训练和部署角度继续展开后训练、蒸馏与推理服务优化。
