# 资料源

本表优先收录原始论文、规范和标准。正文中的状态对象与伪代码是跨实现抽象；具体产品的 chat template、kernel、调度器、streaming 协议和工具 API 仍须绑定其不可变版本文档。来源支持算法或接口定义，不把某个实验结果外推为所有系统的性质。

## 输入、Unicode 与 Tokenization {#source-input}

- Unicode Consortium, [*Unicode Standard Annex #15: Unicode Normalization Forms*](https://www.unicode.org/reports/tr15/)。用于 NFC、NFD、NFKC、NFKD 及规范化一致性的标准定义。
- F. Yergeau, [*RFC 3629: UTF-8, a transformation format of ISO 10646*](https://www.rfc-editor.org/rfc/rfc3629), 2003。用于 UTF-8 字节与 Unicode 标量值的编码边界。
- Rico Sennrich, Barry Haddow, and Alexandra Birch, [*Neural Machine Translation of Rare Words with Subword Units*](https://aclanthology.org/P16-1162/), ACL 2016。用于 BPE 子词建模的原始神经机器翻译工作。
- Taku Kudo and John Richardson, [*SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing*](https://aclanthology.org/D18-2012/), EMNLP 2018。用于直接从原始句子训练、分词与反分词的具体接口背景。

聊天模板不是单一学术标准。正文因此只规定确定性序列化契约，不声称不同框架的角色标记或特殊 token 可以互换。

## Transformer 前向与 KV {#source-transformer}

- Ashish Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762), 2017。用于 scaled dot-product attention、multi-head attention、残差和位置级前馈网络的基础定义。
- Joshua Ainslie et al., [*GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*](https://arxiv.org/abs/2305.13245), 2023。用于 query 头与较少 KV 头之间的 grouped-query 接口。
- Tri Dao et al., [*FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*](https://arxiv.org/abs/2205.14135), NeurIPS 2022。用于“不显式物化完整 attention 矩阵但保持精确逻辑结果”的实现背景。

## 解码、约束与随机数 {#source-decoding}

- Ari Holtzman et al., [*The Curious Case of Neural Text Degeneration*](https://arxiv.org/abs/1904.09751), ICLR 2020。用于 nucleus/top-p sampling 的提出与开放式文本解码背景。
- Brandon T. Willard and Rémi Louf, [*Efficient Guided Generation for Large Language Models*](https://arxiv.org/abs/2307.09702), 2023。用于把正则或上下文无关约束转成有限状态引导生成的算法背景。
- John K. Salmon et al., [*Parallel Random Numbers: As Easy as 1, 2, 3*](https://dl.acm.org/doi/10.1145/2063384.2063405), SC 2011。用于 Philox 等 counter-based PRNG 的原始来源；第七章使用 Philox 名称只为固定教学夹具。

正文给出的处理器顺序是本卷参考配置，不是上述论文规定的跨库标准。复现具体服务必须保存它自己的有序处理器列表。

## Cache 管理与连续批处理 {#source-serving}

- Woosuk Kwon et al., [*Efficient Memory Management for Large Language Model Serving with PagedAttention*](https://arxiv.org/abs/2309.06180), SOSP 2023。用于分页 KV cache、请求动态增长与连续批处理的系统背景。

卷二只依赖这些优化应保持逻辑 token/cache 语义；吞吐、量化和并行策略的工程展开见[卷一第六章](../vol-01/ch06_inference_efficiency_serving.md)。

## 连续扩散、潜变量与引导 {#source-diffusion}

- Jonathan Ho, Ajay Jain, and Pieter Abbeel, [*Denoising Diffusion Probabilistic Models*](https://arxiv.org/abs/2006.11239), NeurIPS 2020。用于高斯前向扰动与学习逆过程。
- Jiaming Song, Chenlin Meng, and Stefano Ermon, [*Denoising Diffusion Implicit Models*](https://arxiv.org/abs/2010.02502), ICLR 2021。用于 DDIM 更新及确定性采样路径。
- Robin Rombach et al., [*High-Resolution Image Synthesis with Latent Diffusion Models*](https://arxiv.org/abs/2112.10752), CVPR 2022。用于潜空间去噪、VAE 与 cross-attention 条件化。
- Jonathan Ho and Tim Salimans, [*Classifier-Free Diffusion Guidance*](https://arxiv.org/abs/2207.12598), 2022。用于有条件与无条件预测的线性组合。

## Flow 与数值积分接口 {#source-flow}

- Yaron Lipman et al., [*Flow Matching for Generative Modeling*](https://arxiv.org/abs/2210.02747), ICLR 2023。用于条件概率路径、向量场学习与 ODE 采样接口。

Euler 与 Heun 公式是标准数值分析算法；正文只用它们说明 solver、步数与网络评估次数是生成配置的一部分，不试图替代数值分析教材。

## 离散扩散与掩码修订 {#source-discrete-diffusion}

- Jacob Austin et al., [*Structured Denoising Diffusion Models in Discrete State-Spaces*](https://arxiv.org/abs/2107.03006), NeurIPS 2021。用于 D3PM 转移矩阵与离散逆过程。
- Huiwen Chang et al., [*MaskGIT: Masked Generative Image Transformer*](https://arxiv.org/abs/2202.04200), CVPR 2022。用于并行预测、置信确认与多轮掩码修订的代表性实例。
- Shen Nie et al., [*Large Language Diffusion Models*](https://arxiv.org/abs/2502.09992), 2025。作为掩码扩散语言模型实例；本卷不据此宣称该路线已经取代自回归语言模型。

## JSON 与 Schema {#source-schema}

- Tim Bray, [*RFC 8259: The JavaScript Object Notation (JSON) Data Interchange Format*](https://www.rfc-editor.org/rfc/rfc8259), 2017。用于 JSON 文本、对象、数组、数字与字符串的标准语法。
- JSON Schema project, [*JSON Schema Core, Draft 2020-12*](https://json-schema.org/draft/2020-12/json-schema-core) 与 [*Validation Vocabulary*](https://json-schema.org/draft/2020-12/json-schema-validation), 2022。用于 schema dialect、vocabulary、结构断言与 `format` 行为。

## 幂等、重试与提交语义 {#source-transactions}

- R. Fielding, M. Nottingham, and J. Reschke, [*RFC 9110: HTTP Semantics*](https://www.rfc-editor.org/rfc/rfc9110), 2022，尤其是 9.2.2 节。用于 HTTP 方法幂等性的标准定义。

RFC 9110 不保证任意工具写调用“恰好一次”。正文的幂等键、规范请求摘要、outcome unknown 与提交边界是应用协议必须额外实现并记录的契约。

## 分布式事件关联 {#source-tracing}

- W3C, [*Trace Context*](https://www.w3.org/TR/trace-context/), W3C Recommendation, 2021。用于跨 HTTP 服务传播 trace ID、parent ID 与供应商状态的标准格式。

Trace context 负责关联事件，不自动保存模型输入、Invocation 参数或外部提交证据；这些仍由正文定义的事件 payload 提供。

## 章节—来源对应

| 章节 | 主要来源组 |
|---|---|
| 第一章 | [输入、Unicode 与 Tokenization](#source-input) |
| 第二章 | [Transformer 前向与 KV](#source-transformer)、[Cache 管理](#source-serving) |
| 第三至四章 | [解码、约束与随机数](#source-decoding)、[Cache 管理](#source-serving) |
| 第五章 | [连续扩散](#source-diffusion)、[Flow](#source-flow)、[离散扩散](#source-discrete-diffusion) |
| 第六章 | [JSON 与 Schema](#source-schema)、[幂等与提交](#source-transactions)、[分布式事件](#source-tracing) |
| 第七章 | 上述全部来源组；数值与 ID 是本卷合成夹具 |
