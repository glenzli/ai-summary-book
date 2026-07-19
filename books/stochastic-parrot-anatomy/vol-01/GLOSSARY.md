# 术语与缩略语表 (Glossary)

本表用于统一本书中的核心术语。它给出阅读口径，不替代正文中的机制解释、数学推导或参考文献。

| 缩写 / 术语 | 全称 | 中文口径 | 本书中的使用边界 |
| :--- | :--- | :--- | :--- |
| AI | Artificial Intelligence | 人工智能，研究让机器表现出感知、推理、学习、规划或行动能力的技术总称。 | 是总领域名，不等同于某一种模型或产品。 |
| ML | Machine Learning | 机器学习，从数据中估计规则、表示或决策函数的方法族。 | 是 AI 的重要分支，但 AI 还包括搜索、规划、知识表示等路线。 |
| DL | Deep Learning | 深度学习，使用多层可微模型学习表示与任务映射。 | 是 ML 的一个主要分支，不覆盖所有机器学习方法。 |
| Neural Network | Neural Network | 神经网络，由参数化层、非线性和训练目标组成的函数近似器。 | 本书关注其工程与数学机制，不把它解释为生物神经系统的等同物。 |
| Foundation Model | Foundation Model | 基座模型，经过大规模预训练后可迁移到多任务的模型。 | 需要后训练、工具、检索和系统约束后，才可能成为稳定应用。 |
| LLM | Large Language Model | 大语言模型，主要在文本或 token 序列上训练的大规模生成模型。 | “语言模型”不自动意味着事实可靠、可行动或已对齐。 |
| Token | Token | 模型处理的离散基本单元，可以是词、子词、字符片段或多模态 patch。 | token 数不是语义长度的直接等价物。 |
| Embedding | Embedding | 将离散对象映射为连续向量的表示。 | 向量相近通常表示训练目标下的统计相似，不保证人类语义完全一致。 |
| Transformer | Transformer | 以注意力、前馈层、残差连接和归一化为核心的序列架构。 | 不是所有现代模型都只用标准 Transformer，长序列和多模态系统常有变体。 |
| Attention | Attention | 通过 query-key 匹配对 value 加权汇聚的信息选择机制。 | “注意力权重”可辅助解释，但不能直接当作完整因果解释。 |
| KV Cache | Key-Value Cache | 自回归推理中缓存历史 key/value，以避免重复计算。 | 降低生成成本，但会占用显存并影响长上下文服务设计。 |
| MoE | Mixture of Experts | 混合专家，让路由器为每个 token 或样本选择部分专家子网络。 | 总参数量与每次激活参数量不同；MoE 不等同于免费提升能力。 |
| SSM | State Space Model | 状态空间模型，用隐状态递推描述序列动力学的一类模型。 | Mamba 等方法提供注意力之外的长序列路线，但适用性取决于任务与实现。 |
| RAG | Retrieval-Augmented Generation | 检索增强生成，先从外部语料取回证据，再把证据交给生成模型。 | 不能保证正确引用；仍需检索质量、证据约束和输出验证。 |
| SFT | Supervised Fine-Tuning | 监督微调，用示范输入输出训练模型遵循任务格式或指令。 | 能改善行为接口，但不能单独解决偏好、安全和事实性问题。 |
| RLHF | Reinforcement Learning from Human Feedback | 基于人类反馈的强化学习，通常用偏好数据训练奖励模型，再优化策略。 | 学到的是标注口径下的偏好近似，不是完整价值函数。 |
| PPO | Proximal Policy Optimization | 近端策略优化，RLHF 中常见的策略优化算法。 | 是一种优化方法，不等同于 RLHF 全流程。 |
| DPO | Direct Preference Optimization | 直接偏好优化，用成对偏好直接约束策略相对参考模型的概率。 | 常作为 RLHF/PPO 的替代或补充；效果依赖偏好数据质量和参考模型。 |
| PEFT | Parameter-Efficient Fine-Tuning | 参数高效微调，只训练少量新增或低秩参数。 | 降低适配成本，但容量、兼容性和安全边界仍需单独评估。 |
| LoRA | Low-Rank Adaptation | 低秩适配，在权重旁加入可训练低秩更新。 | 是 PEFT 的一种常见实现，不保证只学习风格或总能保留底模能力。 |
| Quantization | Quantization | 量化，用较低比特表示权重、激活或 KV Cache。 | 可降低成本，但会引入精度、稳定性和任务退化风险。 |
| Contrastive Learning | Contrastive Learning | 对比学习，通过正对与负对的相对分数学习表示。 | 相似度是训练采样与标注制度下的匹配量，不是真值概率。 |
| Connector / Resampler | Connector / Resampler | 把视觉或音频特征变成语言骨干可读取 token 的接口。 | 连接器既可能做逐 token 投影，也可能以固定查询压缩信息；二者瓶颈不同。 |
| Diffusion | Diffusion Model | 扩散模型，以已知前向扰动和学习到的反向核或 score 生成样本。 | 训练网络、参数化、噪声路径和采样器是不同对象。 |
| Score | Score Function | 密度对输入的对数梯度 $\nabla_x\log p(x)$。 | 不是分类分数或主观评分；在 diffusion 章专指概率密度的局部梯度。 |
| Flow Matching | Flow Matching | 通过回归时间依赖速度场，把基分布连续输运到数据分布。 | 与 diffusion 有连续时间联系，但路径、回归目标和求解器不能混同。 |
| Machine Unlearning | Machine Unlearning | 机器遗忘，使模型相对于指定遗忘集接近未使用该数据训练的参考模型。 | 必须声明参考重训、观察族、保留集和攻击者；有限基准通过不等于精确遗忘。 |
| Agent | Agent | 智能体，在目标、状态、工具和环境反馈之间形成行动闭环的软件系统。 | 不只是“提示词 + LLM”；权限、运行时、记忆、审批和评测同样重要。 |
| Tool Calling | Tool / Function Calling | 工具调用，让模型产生结构化请求，由系统执行外部函数或 API。 | 模型建议调用，真实副作用应由运行时权限和审批控制。 |
| MCP | Model Context Protocol | 模型上下文协议，用于连接 LLM 应用与外部工具、资源、提示模板等上下文。 | 主要解决模型应用到工具/数据的接口标准化，不直接提升模型智能。 |
| A2A | Agent2Agent / Agent-to-Agent | Agent 间通信与任务协作协议，用于描述能力、任务状态、消息和 artifact。 | 主要解决跨 Agent 协调，不替代 MCP 这类工具上下文协议。 |
| World Model | World Model | 世界模型，学习环境状态及其随时间、行动变化的模型。 | 高保真视频生成可能相关，但不能单独证明模型具备可规划的行动后果预测。 |
| Belief State | Belief State | 部分可观测环境中，给定历史对隐藏状态的条件分布。 | 学习到的潜状态通常只是近似充分统计量，不自动等于真实环境状态。 |
| Multimodal Model | Multimodal Model | 多模态模型，处理文本、图像、音频、视频或动作等多种信号。 | 模态输入丰富不等于行动可靠，也不等于具备世界模型。 |
| VLA | Vision-Language-Action Model | 视觉—语言—动作模型，把感知和语言目标映射到机器人动作接口。 | 真实动作仍经过规划器、控制器、安全门和执行器，不能把模型输出等同于物理结果。 |
| ANN | Approximate Nearest Neighbor | 近似最近邻索引，以召回损失换取向量检索的延迟和内存效率。 | ANN 漏召回属于索引误差，不应直接归因于 embedding 或生成模型。 |
| Artifact | Artifact | 长任务产生并可独立寻址的结构化制品，如 diff、查询结果、计划或测试日志。 | 应有类型、版本和 digest；自然语言声称不等于制品已提交。 |
| Idempotency Key | Idempotency Key | 让服务端识别同一副作用请求重试的稳定键。 | 只有服务端持久化键与结果并校验参数时才有语义；不提供任意系统的 exactly-once。 |
| Provenance / Lineage | Provenance / Lineage | 工件或数据从哪些父对象经何种过程产生的可追溯关系。 | 来源记录不证明内容正确，但决定复现、删除、审计和派生影响分析是否可能。 |
| Canary Release | Canary Release | 将少量真实流量送入新版本的渐进发布。 | 不等同于随机 A/B；要预设流量单位、门槛、监控与回滚条件。 |
| Alignment | Alignment | 对齐，使模型行为更符合人类意图、规则、偏好和安全边界的训练与系统过程。 | 既包括训练，也包括产品策略、权限控制、监控和评测。 |
| Benchmark | Benchmark / Evaluation | 基准与评测，用固定任务或数据集衡量模型行为。 | 单一分数不能代表完整能力；要结合任务、数据污染、成本和部署约束阅读。 |
