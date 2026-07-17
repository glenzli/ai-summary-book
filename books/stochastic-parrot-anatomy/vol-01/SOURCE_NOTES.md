# 参考书目与文献索引 (References and Bibliography)

本列表收录书中引用的经典论文、核心书籍、技术报告与协议规范，按主题分类整理。2024 年以后仍可能变化的模型、产品和协议条目校准至 **2026 年 7 月 12 日**；动态文档的日期表示本书核验口径，而不等同于论文发表时间。

## 1. 基础理论 (Foundations)

*   **[McCulloch & Pitts, 1943]** *A Logical Calculus of the Ideas Immanent in Nervous Activity*. (M-P 神经元模型的提出)
*   **[Turing, 1950]** *Computing Machinery and Intelligence*. (图灵测试)
*   **[Rosenblatt, 1958]** *The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain*. (感知机)
*   **[Minsky & Papert, 1969]** *Perceptrons*. (指出了感知机的 XOR 局限性)
*   **[Rumelhart, Hinton & Williams, 1986]** *Learning representations by back-propagating errors*. (反向传播算法的复兴)
*   **[Cybenko, 1989]** [*Approximation by Superpositions of a Sigmoidal Function*](https://doi.org/10.1007/BF02551274). (连续 sigmoidal/discriminatory 激活的一致逼近定理)
*   **[Hornik, 1991]** [*Approximation Capabilities of Multilayer Feedforward Networks*](https://doi.org/10.1016/0893-6080(91)90009-T). (前馈网络逼近能力的推广)
*   **[Leshno et al., 1993]** [*Multilayer Feedforward Networks with a Nonpolynomial Activation Function Can Approximate Any Function*](https://doi.org/10.1016/S0893-6080(05)80131-5). (非多项式激活条件的刻画)

## 2. 卷积神经网络 (CNNs)

*   **[LeCun et al., 1998]** *Gradient-based learning applied to document recognition*. (LeNet-5)
*   **[Krizhevsky et al., 2012]** *ImageNet Classification with Deep Convolutional Neural Networks*. (AlexNet: 深度学习爆发的原点)
*   **[Simonyan & Zisserman, 2014]** *Very Deep Convolutional Networks for Large-Scale Image Recognition*. (VGGNet)
*   **[He et al., 2016]** *Deep Residual Learning for Image Recognition*. (ResNet: 残差连接)

## 3. 序列模型与 RNN (Sequence Models)

*   **[Hochreiter & Schmidhuber, 1997]** *Long Short-Term Memory*. (LSTM 的提出)
*   **[Cho et al., 2014]** *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation*. (GRU 与 Seq2Seq)
*   **[Bahdanau, Cho & Bengio, 2014]** [*Neural Machine Translation by Jointly Learning to Align and Translate*](https://arxiv.org/abs/1409.0473). (加性注意力)
*   **[Luong, Pham & Manning, 2015]** [*Effective Approaches to Attention-based Neural Machine Translation*](https://arxiv.org/abs/1508.04025). (乘性注意力)
<a id="ref-peters-2018"></a>
*   **[Peters et al., 2018]** [*Deep Contextualized Word Representations*](https://arxiv.org/abs/1802.05365). (ELMo: 动态词向量)

## 4. Transformer 与预训练语言模型 (Transformer & PLMs)

*   **[Vaswani et al., 2017]** [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762). (Transformer 架构)
*   **[Ba, Kiros & Hinton, 2016]** [*Layer Normalization*](https://arxiv.org/abs/1607.06450). (LayerNorm)
*   **[Shazeer, 2019]** [*Fast Transformer Decoding: One Write-Head is All You Need*](https://arxiv.org/abs/1911.02150). (Multi-Query Attention)
*   **[Ainslie et al., 2023]** [*GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*](https://arxiv.org/abs/2305.13245). (Grouped-Query Attention)
*   **[Su et al., 2021]** [*RoFormer: Enhanced Transformer with Rotary Position Embedding*](https://arxiv.org/abs/2104.09864). (RoPE)
*   **[Chen et al., 2023]** [*Extending Context Window of Large Language Models via Positional Interpolation*](https://arxiv.org/abs/2306.15595). (位置插值与上下文扩展)
*   **[Liu et al., 2023]** [*Ring Attention with Blockwise Transformers for Near-Infinite Context*](https://arxiv.org/abs/2310.01889). (分布式长上下文注意力)
*   **[Child et al., 2019]** [*Generating Long Sequences with Sparse Transformers*](https://arxiv.org/abs/1904.10509). (稀疏注意力与长序列生成)
*   **[Dao et al., 2022]** [*FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*](https://arxiv.org/abs/2205.14135). (IO-aware 精确注意力)
*   **[Dao, 2023]** [*FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*](https://arxiv.org/abs/2307.08691). (更高并行度的注意力 kernel)
*   **[Gu & Dao, 2023]** [*Mamba: Linear-Time Sequence Modeling with Selective State Spaces*](https://arxiv.org/abs/2312.00752). (选择性状态空间模型)
*   **[Dao & Gu, 2024]** [*Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality*](https://arxiv.org/abs/2405.21060). (Mamba-2 与 SSM/Attention 统一视角)
<a id="ref-devlin-2018"></a>
*   **[Devlin et al., 2018]** [*BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*](https://arxiv.org/abs/1810.04805). (BERT)
<a id="ref-liu-roberta-2019"></a>
*   **[Liu et al., 2019]** [*RoBERTa: A Robustly Optimized BERT Pretraining Approach*](https://arxiv.org/abs/1907.11692). (移除 NSP 并联合调整数据与训练配方)
<a id="ref-radford-2018"></a>
*   **[Radford et al., 2018]** [*Improving Language Understanding by Generative Pre-Training*](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf). (GPT-1)
<a id="ref-radford-2019"></a>
*   **[Radford et al., 2019]** [*Language Models are Unsupervised Multitask Learners*](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf). (GPT-2)
<a id="ref-brown-2020"></a>
*   **[Brown et al., 2020]** [*Language Models are Few-Shot Learners*](https://arxiv.org/abs/2005.14165). (GPT-3)
<a id="ref-raffel-2020"></a>
*   **[Raffel et al., 2020]** [*Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*](https://arxiv.org/abs/1910.10683). (T5)
<a id="ref-lewis-bart-2019"></a>
*   **[Lewis et al., 2019]** [*BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension*](https://arxiv.org/abs/1910.13461). (BART)

## 5. 大模型对齐与优化 (Alignment & Optimization)

*   **[Wei et al., 2021]** *Finetuned Language Models Are Zero-Shot Learners*. (FLAN: 指令微调)
<a id="ref-ouyang-2022"></a>
*   **[Ouyang et al., 2022]** [*Training language models to follow instructions with human feedback*](https://arxiv.org/abs/2203.02155). (InstructGPT: RLHF 的应用)
*   **[Schulman et al., 2017]** *Proximal Policy Optimization Algorithms*. (PPO 算法)
*   **[Amodei et al., 2016]** [*Concrete Problems in AI Safety*](https://arxiv.org/abs/1606.06565). (reward hacking、负副作用与安全问题)
*   **[Rafailov et al., 2023]** *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*. (DPO)
*   **[Li & Liang, 2021]** [*Prefix-Tuning: Optimizing Continuous Prompts for Generation*](https://arxiv.org/abs/2101.00190). (每层 prefix 参数)
*   **[Lester, Al-Rfou & Constant, 2021]** [*The Power of Scale for Parameter-Efficient Prompt Tuning*](https://arxiv.org/abs/2104.08691). (输入层 soft prompt)
*   **[Hu et al., 2021]** *LoRA: Low-Rank Adaptation of Large Language Models*. (LoRA)
*   **[Dettmers et al., 2023]** *QLoRA: Efficient Finetuning of Quantized LLMs*. (QLoRA)
*   **[Frantar et al., 2022]** [*GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*](https://arxiv.org/abs/2210.17323). (GPTQ 权重量化)
*   **[Xiao et al., 2022]** [*SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models*](https://arxiv.org/abs/2211.10438). (激活/权重量化平滑)
*   **[Lin et al., 2023]** [*AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*](https://arxiv.org/abs/2306.00978). (AWQ 混合/保护式量化)
<a id="ref-kaplan-2020"></a>
*   **[Kaplan et al., 2020]** [*Scaling Laws for Neural Language Models*](https://arxiv.org/abs/2001.08361). (缩放定律)
*   **[Kwon et al., 2023]** *Efficient Memory Management for Large Language Model Serving with PagedAttention*. (vLLM)
*   **[DeepSeek-AI, 2024]** [*DeepSeek-V3 Technical Report*](https://arxiv.org/abs/2412.19437). (MLA、MoE 与高效训练)
*   **[DeepSeek-AI, 2025]** [*Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention*](https://arxiv.org/abs/2502.11089). (NSA: 硬件对齐、可训练稀疏注意力)
*   **[DeepSeek-AI, 2025]** [*DeepSeek-V3.2-Exp: Boosting Long-Context Efficiency with DeepSeek Sparse Attention*](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp). (DSA 实验性长上下文效率模型)
*   **[Zhou et al., 2026]** [*MISA: Mixture of Indexer Sparse Attention for Long-Context LLM Inference*](https://arxiv.org/abs/2605.07363). (独立后续研究：DSA 索引器的 head 轴路由加速，并非 DeepSeek 团队论文)
<a id="ref-shao-deepseekmath-2024"></a>
*   **[Shao et al., 2024]** [*DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*](https://arxiv.org/abs/2402.03300). (GRPO 的提出)
<a id="ref-deepseek-r1-2025"></a>
*   **[DeepSeek-AI, 2025]** [*DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*](https://arxiv.org/abs/2501.12948). (推理模型、GRPO 与 RL 后训练)
*   **[DeepSeek-AI, 2025]** [*DeepSeek-OCR: Contexts Optical Compression*](https://arxiv.org/abs/2510.18234). (作者所称的 initial investigation：视觉 token 压缩与 OCR 重建，不代表通用长上下文压缩已解决)
*   **[Cheng et al., 2026]** [*Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models*](https://arxiv.org/abs/2601.07372). (Engram：哈希寻址 N-gram embedding 与层内条件记忆)
<a id="ref-openai-o1-2024"></a>
*   **[OpenAI, 2024]** [*Learning to Reason with LLMs*](https://openai.com/index/learning-to-reason-with-llms/). (o1、强化学习与测试时计算的公开说明)
*   **[Dubey et al., 2024]** [*The Llama 3 Herd of Models*](https://arxiv.org/abs/2407.21783). (开放权重模型、预训练与后训练技术报告)
*   **[OpenAI, 2026-07-09]** [*Introducing GPT-5.6*](https://openai.com/index/gpt-5-6/). (GPT-5.6 Sol / Terra / Luna 已 GA；图像输入等能力以发布页为准；核验于 2026-07-12)
*   **[DeepSeek, 2026]** [*DeepSeek API Change Log*](https://api-docs.deepseek.com/updates) 与 [*Models and Pricing*](https://api-docs.deepseek.com/quick_start/pricing). (`deepseek-v4-pro` / `deepseek-v4-flash` 官方 API 标识；核验于 2026-07-12)
*   **[Hinton et al., 2015]** [*Distilling the Knowledge in a Neural Network*](https://arxiv.org/abs/1503.02531). (知识蒸馏)
*   **[Wang et al., 2022]** [*Self-Instruct: Aligning Language Models with Self-Generated Instructions*](https://arxiv.org/abs/2212.10560). (合成指令数据)
<a id="ref-tofu-unlearning-2024"></a>
*   **[Maini et al., 2024]** [*TOFU: A Task of Fictitious Unlearning for LLMs*](https://openreview.net/forum?id=q0eyIBnE2t). (面向个体虚构资料的遗忘集/保留集、多指标评测与重训参照；基准结果不证明任意观察下的精确遗忘)
*   **[Bai et al., 2022]** [*Constitutional AI: Harmlessness from AI Feedback*](https://arxiv.org/abs/2212.08073). (AI 反馈与安全后训练)
*   **[Korbak et al., 2025]** [*Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety*](https://arxiv.org/abs/2507.11473). (CoT 监控与安全训练风险)
*   **[Wortsman et al., 2022]** [*Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time*](https://arxiv.org/abs/2203.05482). (权重平均与模型合并)
*   **[Ilharco et al., 2022]** [*Editing Models with Task Arithmetic*](https://arxiv.org/abs/2212.04089). (任务向量与模型编辑)
*   **[Yadav et al., 2023]** [*TIES-Merging: Resolving Interference When Merging Models*](https://arxiv.org/abs/2306.01708). (模型合并干扰处理)
*   **[Yu et al., 2023]** [*Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch*](https://arxiv.org/abs/2311.03099). (DARE 与模型合并)
*   **[Leviathan et al., 2023]** [*Fast Inference from Transformers via Speculative Decoding*](https://arxiv.org/abs/2211.17192). (投机解码)
*   **[Cai et al., 2024]** [*Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads*](https://arxiv.org/abs/2401.10774). (多解码头推理加速)
*   **[Li et al., 2024]** [*EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty*](https://arxiv.org/abs/2401.15077). (特征级投机解码)

## 6. 多模态与智能体 (Multimodal & Agents)

<a id="ref-dosovitskiy-2020"></a>
*   **[Dosovitskiy et al., 2020]** [*An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*](https://arxiv.org/abs/2010.11929). (ViT)
<a id="ref-radford-clip-2021"></a>
*   **[Radford et al., 2021]** [*Learning Transferable Visual Models From Natural Language Supervision*](https://arxiv.org/abs/2103.00020). (CLIP)
<a id="ref-liu-llava-2023"></a>
*   **[Liu et al., 2023]** [*Visual Instruction Tuning*](https://arxiv.org/abs/2304.08485). (LLaVA)
<a id="ref-rombach-2022"></a>
*   **[Rombach et al., 2022]** [*High-Resolution Image Synthesis with Latent Diffusion Models*](https://arxiv.org/abs/2112.10752). (Latent Diffusion / Stable Diffusion 基础)
<a id="ref-peebles-xie-2023"></a>
*   **[Peebles & Xie, 2023]** [*Scalable Diffusion Models with Transformers*](https://arxiv.org/abs/2212.09748). (DiT)
<a id="ref-lipman-2022"></a>
*   **[Lipman et al., 2022]** [*Flow Matching for Generative Modeling*](https://arxiv.org/abs/2210.02747). (Flow Matching 与连续生成路径)
*   **[Betker et al., 2023]** [*Improving Image Generation with Better Captions*](https://cdn.openai.com/papers/dall-e-3.pdf). (DALL-E 3 与高质量图文描述)
<a id="ref-openai-gpt4o-2024"></a>
*   **[OpenAI, 2024]** [*GPT-4o System Card*](https://cdn.openai.com/gpt-4o-system-card.pdf). (文本、视觉、音频的端到端多模态评测与安全分析)
<a id="ref-openai-sora-2024"></a>
*   **[OpenAI, 2024]** [*Video Generation Models as World Simulators*](https://openai.com/index/video-generation-models-as-world-simulators/). (Sora：时空 patch 上的扩散 Transformer；技术细节披露有限)
*   **[Google DeepMind, 核验于 2026-07-12]** [*Veo*](https://deepmind.google/models/veo/). (视频生成模型族的官方能力材料；不据此推断未公开架构)
<a id="ref-kondratyuk-2023"></a>
*   **[Kondratyuk et al., 2023]** [*VideoPoet: A Large Language Model for Zero-Shot Video Generation*](https://arxiv.org/abs/2312.14125). (视频生成的语言模型路线)
<a id="ref-reid-2024"></a>
*   **[Reid et al., 2024]** [*Gemini 1.5: Unlocking Multimodal Understanding Across Millions of Tokens of Context*](https://arxiv.org/abs/2403.05530). (长上下文多模态理解)
<a id="ref-google-gemini-35-flash-2026"></a>
*   **[Google, 核验于 2026-07-12]** [*Gemini 3.5 Flash Model Documentation*](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash). (stable/GA；text/image/video/audio/PDF 输入、text 输出；不支持 Live API)
<a id="ref-ha-schmidhuber-2018"></a>
*   **[Ha & Schmidhuber, 2018]** [*World Models*](https://arxiv.org/abs/1803.10122). (世界模型经典工作)
<a id="ref-hafner-2019"></a>
*   **[Hafner et al., 2019]** [*Dream to Control: Learning Behaviors by Latent Imagination*](https://arxiv.org/abs/1912.01603). (Dreamer 与潜在想象)
<a id="ref-assran-ijepa-2023"></a>
*   **[Assran et al., 2023]** [*Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*](https://arxiv.org/abs/2301.08243). (I-JEPA)
<a id="ref-bruce-genie-2024"></a>
*   **[Bruce et al., 2024]** [*Genie: Generative Interactive Environments*](https://arxiv.org/abs/2402.15391). (从视频学习可交互环境)
<a id="ref-rt2-2023"></a>
*   **[Zitkovich et al., 2023]** [*RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*](https://proceedings.mlr.press/v229/zitkovich23a.html). (把视觉、语言和机器人动作接入同一 VLA 接口；具体实验不外推为开放环境安全保证)
<a id="ref-gemini-robotics-2025"></a>
*   **[Google DeepMind et al., 2025]** [*Gemini Robotics: Bringing AI into the Physical World*](https://arxiv.org/abs/2503.20020). (VLA 通用模型、机器人形态适配与物理行动研究入口)
<a id="ref-wei-cot-2022"></a>
*   **[Wei et al., 2022]** [*Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*](https://arxiv.org/abs/2201.11903). (CoT)
<a id="ref-kojima-2022"></a>
*   **[Kojima et al., 2022]** [*Large Language Models are Zero-Shot Reasoners*](https://arxiv.org/abs/2205.11916). (Zero-shot CoT)
<a id="ref-yao-react-2022"></a>
*   **[Yao et al., 2022]** [*ReAct: Synergizing Reasoning and Acting in Language Models*](https://arxiv.org/abs/2210.03629). (ReAct 框架)
<a id="ref-lewis-rag-2020"></a>
*   **[Lewis et al., 2020]** [*Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*](https://arxiv.org/abs/2005.11401). (RAG)
<a id="ref-mcp-2025-11-25"></a>
*   **[Model Context Protocol, 2025-11-25]** [*Specification*](https://modelcontextprotocol.io/specification/2025-11-25). (模型应用与工具、资源、提示模板等上下文集成；`latest` 核验于 2026-07-12)
<a id="ref-a2a-2026"></a>
*   **[A2A Project, 2026]** [*A2A Protocol Specification v1.0.0*](https://a2a-protocol.org/latest/specification/). (2026 年首个 stable 规范：Agent 间消息、任务与 artifact 协调；Google 发起，现为 Linux Foundation 项目；核验于 2026-07-12)
*   **[OpenAI, 核验于 2026-07-12]** [*OpenAI Agents SDK Documentation*](https://openai.github.io/openai-agents-python/). (Agent 运行时、工具、handoff、guardrails 与 tracing 的实现文档)
*   **[LangChain, 核验于 2026-07-12]** [*LangGraph Overview*](https://docs.langchain.com/oss/python/langgraph/overview). (持久化执行、状态图与多 Agent 编排的实现文档)
*   **[Google, 核验于 2026-07-12]** [*Agent Development Kit Documentation*](https://google.github.io/adk-docs/). (Agent 开发、会话、记忆、工具与上下文管理的实现文档)

## 7. 推荐书籍 (Recommended Books)

*   **Ian Goodfellow, Yoshua Bengio, Aaron Courville**. *Deep Learning*. MIT Press, 2016. (深度学习教材)
*   **Sutton & Barto**. *Reinforcement Learning: An Introduction*. MIT Press, 2018. (强化学习入门)
*   **Daniel Kahneman**. *Thinking, Fast and Slow*. (对 System 1 / System 2 表述的普及；双过程理论来自更广泛研究传统)
