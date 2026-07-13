# 第四章 预训练语言模型：ELMo、BERT、GPT 与 T5
<a id="section-4-1"></a>

## 4.1 预训练时代与 ELMo (The Pre-training Era & ELMo)

### 1. 从静态到动态：词嵌入的演进 (Evolution of Word Embeddings)

在第 2 章中，我们介绍了 RNN 如何处理序列数据；在第 3 章中，我们深入探讨了 Transformer 架构及其强大的注意力机制。然而，在 Transformer 统治 NLP 之前，领域内面临着一个关键痛点：**如何有效地表示词义的复杂性（Polysemy）？**

早期的词向量模型（如 Word2Vec, GloVe）是 **静态的 (Static)**。一旦训练完成，单词 "bank" 的向量就是固定的，无论它出现在 "bank account"（银行账户）还是 "river bank"（河岸）中，其表示完全相同。这显然无法捕捉人类语言的丰富语境。

<span style="background-color: #DAE8FC; color: black; padding: 2px 4px; border-radius: 4px;">核心问题</span>：我们需要一种能够根据上下文动态调整的词表示方法。

#### 1.1 静态与动态嵌入对比 (Static vs. Dynamic Embeddings)

```mermaid
graph LR
    %% 样式定义
    classDef static fill:#F5F5F5,stroke:#666666,color:#000000;
    classDef dynamic fill:#DAE8FC,stroke:#6C8EBF,color:#000000;
    classDef vector fill:#FFF2CC,stroke:#D6B656,color:#000000;

    subgraph Static ["静态嵌入 (Word2Vec/GloVe)"]
        W1("Word: Apple"):::static --> V1["Vector: [0.3, -0.1, ...]"]:::vector
        W2("Context: Apple pie"):::static -.-> V1
        W3("Context: Apple Inc."):::static -.-> V1
    end

    subgraph Dynamic ["动态嵌入 (ELMo/BERT)"]
        D1("Word: Apple"):::static --> C1{Context Encoder}:::dynamic
        C2("Context: Apple pie"):::dynamic --> C1
        C1 --> V2["Vector A: Fruit features"]:::vector

        D3("Word: Apple"):::static --> C3{Context Encoder}:::dynamic
        C4("Context: Apple Inc."):::dynamic --> C3
        C3 --> V3["Vector B: Tech features"]:::vector
    end
```

### 2. ELMo: 语言模型嵌入 (Embeddings from Language Models)

**ELMo (Embeddings from Language Models)** 是连接主义向预训练大模型过渡的重要桥梁。它并没有使用 Transformer，而是基于 **双向 LSTM (Bi-LSTM)** 构建。

#### 2.1 核心思想 (Core Idea)

ELMo 的核心洞察是：**词向量不应该是一个查表操作（Lookup Table），而应该是一个函数（Function）。** 这个函数的输入是整个句子，输出是针对该语境下每个词的向量表示。

<span style="background-color: #FFF2CC; color: black; padding: 2px 4px; border-radius: 4px;">公式定义</span>
给定一个序列 \( t_1, t_2, \dots, t_N \)，ELMo 通过最大化双向对数似然来训练：

\[
\sum_{k=1}^{N} \left( \log P(t_k | t_1, \dots, t_{k-1}) + \log P(t_k | t_{k+1}, \dots, t_N) \right)
\]

*   前向 LSTM 预测下一个词。
*   后向 LSTM 预测上一个词。

#### 2.2 ELMo 架构可视化 (ELMo Architecture)

ELMo 通过组合不同层级的 LSTM 隐藏状态来生成最终的词向量。

*   **Layer 0 (Char CNN)**: 由字符构造 token 表示，从而缓解固定词表带来的 OOV 问题，但不能保证理解任意新词。
*   **Layer 1/2 (BiLSTM)**: 逐层形成上下文表示。ELMo 论文的下游任务权重分析显示较低层往往更有利于句法任务、较高层往往更有利于语义任务；这是一项经验趋势，不是每层预先规定的功能。

```mermaid
graph TD
    %% 样式定义
    classDef input fill:#F5F5F5,stroke:#666666,color:#000000;
    classDef lstm fill:#E1D5E7,stroke:#9673A6,color:#000000;
    classDef combine fill:#D5E8D4,stroke:#82B366,color:#000000;
    classDef output fill:#DAE8FC,stroke:#6C8EBF,color:#000000;

    I[Input Sentence: 'Open the bank account']:::input

    subgraph BiLSTM_Layers ["双向 LSTM 层"]
        direction TB
        L1_F[Layer 1 Forward]:::lstm
        L1_B[Layer 1 Backward]:::lstm
        L2_F[Layer 2 Forward]:::lstm
        L2_B[Layer 2 Backward]:::lstm
    end

    I --> L1_F
    I --> L1_B

    L1_F --> L2_F
    L1_B --> L2_B

    subgraph Weighted_Sum ["加权求和"]
        WS{Weighted Sum}:::combine
        P[Scalar Parameters]:::combine
    end

    L1_F --> WS
    L1_B --> WS
    L2_F --> WS
    L2_B --> WS
    P --> WS

    WS --> O[Contextualized Embedding]:::output
```

#### 2.3 特征融合 (Feature Fusion)

ELMo 的最终表示 \( \mathbf{ELMo}_k \) 是各层表示的线性组合：

\[
\mathbf{ELMo}_k^{task} = \gamma^{task} \sum_{j=0}^{L} s_j^{task} \mathbf{h}_{k,j}^{LM}
\]

*   \( \mathbf{h}_{k,j}^{LM} \): 第 \( j \) 层的隐藏状态。
*   \( s_j^{task} \): Softmax 归一化的权重，让模型根据特定任务（如阅读理解或词性标注）自动决定更关注低层语法还是高层语义。
*   \( \gamma^{task} \): 缩放因子。

### 3. 预训练-微调范式的雏形 (Prototype of Pre-training & Fine-tuning)

ELMo 引入了一个关键范式：**基于特征的迁移 (Feature-based Transfer)**。
1.  **预训练 (Pre-training)**: 在大规模无标注文本上训练 Bi-LSTM 语言模型。
2.  **特征提取 (Feature Extraction)**: 将训练好的 ELMo 作为一个“特征提取器”，将其输出的动态向量作为下游任务模型的输入。
3.  **任务训练**: 下游模型（如分类器）只需要学习如何利用这些丰富的特征。

ELMo 仍受循环计算和长距离依赖限制，但其实验系统展示了 **利用大规模无标注文本预训练上下文表示** 对多类下游任务的迁移价值，并影响了后续 BERT、GPT 等预训练路线。

<span style="background-color: #F8CECC; color: black; padding: 2px 4px; border-radius: 4px;">双向口径</span>：ELMo 的前向与后向语言模型在各自堆栈中独立计算，再在表示层组合，因此常被称为“shallow bidirectional”。BERT 则让每层 self-attention 都可同时使用左右上下文；这是一种不同的深层融合方式，不意味着只有 Transformer 才能组合双向信息。
<a id="section-4-2"></a>

## 4.2 BERT：双向编码器表示 (BERT: Bidirectional Encoder Representations)

### 1. 从 Feature-based 到 Fine-tuning (From Feature-based to Fine-tuning)

在 4.1 节中，我们看到 ELMo 的典型用法是冻结预训练双向语言模型、学习层权重，并把表示加入下游模型。OpenAI GPT (基于 Decoder) 和 Google BERT (基于 Encoder) 则系统展示了端到端 **微调 (Fine-tuning)** 预训练 Transformer 的路线。

<span style="background-color: #DAE8FC; color: black; padding: 2px 4px; border-radius: 4px;">微调范式</span>：不仅是输出层，**整个预训练模型** 的参数都会在下游任务中进行更新。这意味着模型可以针对特定任务进行端到端的适配。

BERT (**B**idirectional **E**ncoder **R**epresentations from **T**ransformers) 的出现常被称为 NLP 历史上的“ImageNet 时刻”：它系统展示了 **深度双向架构** 配合大规模预训练在多类语言理解基准上的强迁移能力。

### 2. BERT 架构概览 (BERT Architecture Overview)

BERT 仅使用了 Transformer 的 **Encoder** 部分（详见 3.2 节）。与 GPT 的单向（从左到右）不同，BERT 的 Encoder 允许每个 token 同时“看到”其左边和右边的所有 token。

#### 2.1 输入表示 (Input Representations)

BERT 的输入表示由三部分相加：
1.  **Token Embeddings**: WordPiece 词向量。
2.  **Segment Embeddings**: 区分句子对（句子 A vs 句子 B）。
3.  **Position Embeddings**: 学习到的位置编码（非正弦）。

```mermaid
graph TD
    %% 样式定义
    classDef token fill:#FFF2CC,stroke:#D6B656,color:#000000;
    classDef segment fill:#DAE8FC,stroke:#6C8EBF,color:#000000;
    classDef pos fill:#F8CECC,stroke:#B85450,color:#000000;
    classDef sum fill:#E1D5E7,stroke:#9673A6,color:#000000;

    subgraph Input_Representation ["BERT 输入构成"]
        T1["Token: [CLS]"]:::token
        T2["Token: my"]:::token
        T3["Token: dog"]:::token
        T4["Token: [SEP]"]:::token

        S1["Segment: A"]:::segment
        S2["Segment: A"]:::segment
        S3["Segment: A"]:::segment
        S4["Segment: A"]:::segment

        P1["Pos: 0"]:::pos
        P2["Pos: 1"]:::pos
        P3["Pos: 2"]:::pos
        P4["Pos: 3"]:::pos
    end

    T1 --> Sum1((+)):::sum
    S1 --> Sum1
    P1 --> Sum1

    T2 --> Sum2((+)):::sum
    S2 --> Sum2
    P2 --> Sum2

    T3 --> Sum3((+)):::sum
    S3 --> Sum3
    P3 --> Sum3

    T4 --> Sum4((+)):::sum
    S4 --> Sum4
    P4 --> Sum4

    Sum1 --> E1[Encoder Input 1]
    Sum2 --> E2[Encoder Input 2]
    Sum3 --> E3[Encoder Input 3]
    Sum4 --> E4[Encoder Input 4]
```

### 3. 预训练任务 (Pre-training Tasks)

BERT 原论文联合使用两个自监督任务：**掩码语言模型 (Masked Language Model, MLM)** 和 **下一句预测 (Next Sentence Prediction, NSP)**。后续消融表明 NSP 并非普遍必要，因此不能把 BERT 的迁移效果简单归因于两个目标缺一不可。

#### 3.1 掩码语言模型 (Masked Language Model, MLM)

传统的语言模型是单向的（预测下一个词），这限制了对上下文的理解。BERT 采用“完形填空”的方式：
*   随机 Mask 掉输入中 15% 的 Token。
*   模型需要利用 **双向上下文** 来预测被 Mask 掉的词。

<span style="background-color: #F5F5F5; color: black; padding: 2px 4px; border-radius: 4px; border: 1px solid #999;">掩码策略细节</span>
在被选中的 15% Token 中：
*   80% 替换为 `[MASK]` 符号。
*   10% 替换为随机词（迫使模型关注语义而非仅依赖 `[MASK]` 标记）。
*   10% 保持不变（缩小预训练与微调阶段的分布差异）。

<img src="chapter_04/images/mlm_masking_strategy.png" width="75%" />

**训练目标（最小数学形式）**：令 $\mathcal{M}$ 为被选中 mask 的位置集合，MLM 的损失可以写为：

$$ \mathcal{L}_{\text{MLM}} = - \sum_{i \in \mathcal{M}} \log P_\theta(x_i \mid \tilde{x}) $$

其中 $x_i$ 是原 token，$\tilde{x}$ 是按 80/10/10 规则破坏后的输入序列；保持原词的 10% 位置也仍计入预测损失。

#### 3.2 下一句预测 (Next Sentence Prediction, NSP)

BERT 设计 NSP 的原意，是给句对任务提供跨句训练信号：模型判断句子 B 是否紧接在句子 A 之后。它并不直接监督一般“逻辑关系”，随机负例也可能让任务依赖主题差异等捷径。
*   **正例 (Positive)**: 50% 概率选择真实的下一句。
*   **负例 (Negative)**: 50% 概率从语料库中随机选择一句。

**训练目标（最小数学形式）**：令标签 $y\in\{0,1\}$ 表示 IsNext（1 为真），模型给出 $P_\theta(y=1\mid A,B)$，则二分类交叉熵为

$$ \mathcal{L}_{\text{NSP}} = -\left[y\log P_\theta + (1-y)\log(1-P_\theta)\right] $$

（后续工作如 RoBERTa 在改变数据与训练配方后移除了 NSP 并取得更好结果；因此这里应把它理解为 BERT 原始配方中的句对训练信号，而不是对齐或推理的必要条件。）

```mermaid
graph LR
    %% 样式定义
    classDef mask fill:#F8CECC,stroke:#B85450,color:#000000;
    classDef context fill:#D5E8D4,stroke:#82B366,color:#000000;
    classDef model fill:#DAE8FC,stroke:#6C8EBF,color:#000000;

    subgraph Masked_Input ["输入序列"]
        I1("[CLS] The man went to [MASK] store"):::context
        I2("[SEP] He bought a gallon of milk"):::context
    end

    I1 --> BERT{BERT Encoder}:::model
    I2 --> BERT

    BERT --> O1("Output: [MASK] = 'the'"):::mask
    BERT --> O2("Output: IsNext = True"):::context
```

### 4. 总结 (Summary)

BERT 推动了 NLP 从任务特定模型走向“预训练 + 微调”的主流范式。
*   **架构**: 纯 Encoder，深度双向。
*   **数据**: 大规模无标注文本（BooksCorpus + Wikipedia）。
*   **影响**: 系统化推广了预训练 Transformer 加下游微调的工作流，并在原论文所测语言理解任务上以较少任务标注取得强结果。

然而，BERT 的双向 MLM 目标并未训练“只看左侧前缀逐 token 生成”的条件分解，因此原始 BERT 不可直接当作自回归生成器。通过增加解码器或采用其他掩码/去噪目标仍可构造生成系统；这里的差别是训练目标与推理接口，而不是编码器在数学上“不能生成”。
<a id="section-4-3"></a>

## 4.3 GPT 系列：生成式预训练变换器 (GPT Series: Generative Pre-trained Transformers)

### 1. 另一条道路：生成式模型 (The Generative Path)

在 BERT 专注于“完形填空”以此理解语言结构的同时，GPT 系列选择了一条更接近自然语言顺序生成的道路：**自回归生成 (Autoregressive Generation)**。

**GPT (Generative Pre-trained Transformer)** 用下一 token 预测作为统一训练目标。大规模实验表明，该目标可以学到可迁移的语法、语义、知识模式和上下文适配能力；能否可靠完成某类推理或事实任务仍需独立评测。

**训练目标（最小数学形式）**：GPT 采用 **自回归语言建模 (Causal Language Modeling, CLM)**。给定序列 $x_{1:T}$，最大化似然等价于最小化负对数似然：

$$ \mathcal{L}_{\text{CLM}} = -\sum_{t=1}^{T} \log P(x_t \mid x_{<t}) $$

它之所以能被并行训练，是因为训练时我们一次性喂入整段文本，并用 **因果掩码 (Causal Mask)** 在注意力层里严格禁止“偷看未来”。

#### 1.1 架构对比：Decoder-only (Architecture Comparison)

GPT 采用了 Transformer 的 **Decoder** 部分（去掉了 Encoder-Decoder Attention 层）。

<span style="background-color: #DAE8FC; color: black; padding: 2px 4px; border-radius: 4px;">关键区别</span>：**因果掩码 (Causal Mask)**。
*   **BERT (Encoder)**: 能够看到未来的 Token（双向）。适合判别式任务。
*   **GPT (Decoder)**: 严格只能看到当前及过去的 Token（单向）。适合生成式任务。

为了把“能看见哪些 token”变成可视化对象，下图对比了两者的注意力可见性矩阵：

<img src="chapter_04/images/bert_vs_gpt_attention_mask.png" width="80%" />

```mermaid
graph TD
    %% 样式定义
    classDef input fill:#F5F5F5,stroke:#666666,color:#000000;
    classDef block fill:#DAE8FC,stroke:#6C8EBF,color:#000000;
    classDef output fill:#D5E8D4,stroke:#82B366,color:#000000;

    subgraph GPT_Process ["GPT 生成过程"]
        I1("Input: 'The AI'"):::input --> B1[Transformer Block]:::block
        B1 --> O1("Output: 'is'"):::output

        I2("Input: 'The AI is'"):::input --> B2[Transformer Block]:::block
        B2 --> O2("Output: 'learning'"):::output

        I3("Input: 'The AI is learning'"):::input --> B3[Transformer Block]:::block
        B3 --> O3("Output: 'fast'"):::output
    end
```

### 2. GPT 的演进之路 (Evolution of GPT)

GPT 系列的发展可以理解为模型规模（Scale）、数据分布、训练工程与能力（Capability）共同演进的历史。一些评测曲线看起来有 **涌现 (Emergence)** 或跃迁，但这种形态会受到指标离散化、任务选择和采样方法影响，不等同于存在已知的普适能力临界点。

#### 2.1 GPT-1: 预训练 + 微调 (Pre-training + Fine-tuning)
*   **规模**: 1.17亿参数。
*   **贡献**: 验证了在无标注数据上预训练 Decoder 模型，再在下游任务上有监督微调（SFT）的有效性。此时它和 BERT 的思路类似。

#### 2.2 GPT-2: 零样本学习者 (Zero-shot Learner)
*   **规模**: 15亿参数。
*   **洞察**: "Language Models are Unsupervised Multitask Learners"。
*   GPT-2 的论文在不做任务梯度更新的设置下，用任务特定输入格式评估翻译、摘要等零样本行为；结果展示了迁移潜力，也明显依赖任务与评测格式。
    *   *Prompt*: "English: Hello. French: " -> 模型自动补全 "Bonjour"。

#### 2.3 GPT-3: 上下文学习 (In-context Learning)
*   **规模**: 1750亿参数。
*   **核心突破**: 即使不进行任何梯度更新（No Gradient Updates），模型也能通过输入中的少量示例（Few-shot）在上下文内适配新任务。

<span style="background-color: #FFF2CC; color: black; padding: 2px 4px; border-radius: 4px;">In-context Learning 示例</span>

```text
Input to GPT-3:
Task: Translate English to Spanish.
apple -> manzana
car -> coche
book -> ???

GPT-3 Output:
libro
```

### 3. 缩放定律 (Scaling Laws)

Kaplan 等人（2020）系统刻画了语言模型的经验 **Scaling Laws**：
模型性能（Loss）与计算量（Compute）、数据集大小（Data Size）、参数量（Parameters）之间存在 **幂律关系 (Power Law)**。

这意味着：在相似数据分布和训练设定下，**增加算力、数据和参数通常会带来可预测的损失下降**。但缩放不是免费午餐：高质量数据会枯竭，训练与推理成本会上升，评测也会受到数据污染和任务选择影响。2024 年之后的近期系统越来越依赖后训练、工具使用和测试时计算，而不只是扩大预训练规模。

<img src="chapter_04/images/scaling_law_plot.png" width="85%" />

### 4. 总结 (Summary)

GPT 系列通过下一 token 预测目标，并在数据、模型规模和训练工程上持续扩展，获得了广泛的上下文学习与生成能力。它推动 NLP 从“每个任务单独微调”扩展到“通过上下文描述任务”，也为后续指令微调、工具调用和推理后训练提供了统一的自回归接口。

然而，早期 GPT 基座模型主要优化文本似然，并不保证稳定遵循用户指令或满足安全策略。后续 InstructGPT 等工作因此研究指令微调、偏好学习与安全评测。
<a id="section-4-4"></a>

## 4.4 统一框架：T5 与 BART (Unified Frameworks: T5 & BART)

### 1. 编码器与解码器的再融合 (Reuniting Encoder & Decoder)

BERT（仅编码器）擅长理解，GPT（仅解码器）擅长生成。那么，是否存在一种架构能够统一处理这两类任务？
答案是回归本源：使用完整的 **编码器-解码器 (Encoder-Decoder)** 架构（即原始 Transformer 架构）。

这一领域的代表作是 Google 的 **T5** 和 Facebook (Meta) 的 **BART**。

### 2. T5: 文本到文本转换器 (Text-to-Text Transfer Transformer)

T5 提出了一个极具统一性的视角：**许多常见 NLP 任务都可以被改写为“文本到文本” (Text-to-Text) 的转换问题。**

#### 2.1 统一接口 (Unified Interface)

在 T5 之前，不同的任务需要不同的模型头部（Head）：
*   分类任务需要全连接层输出类别概率。
*   回归任务需要输出实数。
*   生成任务需要解码文本。

T5 将一切统一为文本生成：
*   **翻译**: Input: "translate English to German: That is good." $\rightarrow$ Output: "Das ist gut."
*   **分类**: Input: "cola sentence: The course is jumping well." $\rightarrow$ Output: "not acceptable."
*   **回归**: Input: "stsb sentence1: ... sentence2: ..." $\rightarrow$ Output: "3.8" (以字符串形式输出数字)。

```mermaid
graph LR
    %% 样式定义
    classDef task fill:#DAE8FC,stroke:#6C8EBF,color:#000000;
    classDef model fill:#E1D5E7,stroke:#9673A6,color:#000000;
    classDef output fill:#D5E8D4,stroke:#82B366,color:#000000;

    subgraph Inputs ["输入提示"]
        I1("translate En to De: Good morning"):::task
        I2("summarize: The scientists found..."):::task
        I3("classify sentiment: I hate this"):::task
    end

    I1 --> T5{T5 Model}:::model
    I2 --> T5
    I3 --> T5

    T5 --> O1("Guten Morgen"):::output
    T5 --> O2("Scientists discovered..."):::output
    T5 --> O3("negative"):::output
```

#### 2.2 预训练目标：Span Corruption

T5 使用了一种类似 BERT MLM 但更适合生成任务的目标：**Span Corruption (片段破坏)**。
它不是 Mask 单个词，而是 Mask 掉一段连续的文本，并用唯一的哨兵符（Sentinel Token, 如 `<X>`, `<Y>`）代替。模型需要生成被 Mask 掉的内容。

*   **Original**: "The cute dog runs in the park."
*   **Input**: "The `<X>` runs in the `<Y>`."
*   **Target**: "`<X>` cute dog `<Y>` park `<Z>`"

<img src="chapter_04/images/span_corruption_example.png" width="85%" />

**训练目标（最小数学形式）**：把被破坏后的输入记为 $\tilde{x}$，目标序列（需要模型生成出来的 spans）记为 $y_{1:T}$，则 T5 的生成式去噪目标就是标准序列到序列的负对数似然：

$$ \mathcal{L}_{\text{span}} = -\sum_{t=1}^{T} \log P(y_t \mid y_{<t}, \tilde{x}) $$

它与 BERT-MLM 的差别在于：BERT 在被选位置独立产生 token 分类损失；T5 的解码器自回归生成由哨兵符分隔的一个或多个被删 span，因此目标之间也存在序列条件依赖，更直接对应 seq2seq 生成接口。

### 3. BART: 去噪自编码器 (Denoising Autoencoder)

BART (Bidirectional and Auto-Regressive Transformers) 同样采用了 Encoder-Decoder 架构。它的预训练任务更加多样化，旨在恢复被破坏的文档。

<span style="background-color: #FFF2CC; color: black; padding: 2px 4px; border-radius: 4px;">噪声策略</span>
1.  **Token Masking**: 类似 BERT。
2.  **Token Deletion**: 删除 Token。
3.  **Text Infilling**: 类似 T5 Span Corruption。
4.  **Sentence Permutation**: 随机打乱句序。
5.  **Document Rotation**: 随机选择一个 Token 作为开头，旋转文档。

BART 在 **文本摘要 (Summarization)** 等生成任务上表现尤为出色。

### 4. 本章总结 (Chapter Summary)

本章比较了三类常见 Transformer 架构：

|流派 (Paradigm)|代表模型 (Model)|架构 (Arch)|优势 (Pros)|劣势 (Cons)|
|:---|:---|:---|:---|:---|
|**Encoder-only**|BERT, RoBERTa|Bidirectional|高效形成整段输入表示，适合分类/抽取|原始 MLM 接口不能直接左到右生成|
|**Decoder-only**|GPT-2, GPT-3|Autoregressive|统一条件生成与上下文学习接口|每个位置只能读取左侧前缀；长输出需串行解码|
|**Encoder-Decoder**|T5, BART|Full Transformer|输入可双向编码，输出自回归，适合 seq2seq|同时维护编码器与解码器，成本取决于输入/输出长度和规模|

在接下来的章节中，我们将进入 **大模型时代 (The Era of LLMs)**，探讨如何通过指令微调 (Instruction Tuning) 和人类反馈强化学习 (RLHF) 将这些基座模型转化为更可用的指令跟随系统。
