# 跨卷术语表

| 术语 | 本书口径 |
|---|---|
| 模型 / checkpoint | 一组固定参数及其架构约定；不自动包含工具、记忆和产品策略 |
| 系统 | 模型、上下文构造、解码、运行时、工具、权限和外部服务的组合 |
| tokenizer | 文本/字节与 token IDs 之间的版本化编码接口 |
| context | 模板、历史、检索与多模态信息组装后实际送入模型的条件 |
| pretraining | 在大规模数据上以语言建模、掩码、对比或生成目标学习基础参数 |
| post-training | SFT、偏好优化、RL、蒸馏和安全调优等预训练后阶段 |
| inference | 固定模型接收输入并产生表示、logits 或生成状态的计算 |
| serving | 批处理、缓存、调度、并行、量化与网络接口组成的推理服务 |
| logit | softmax 前的候选实数分数；差值决定赔率 |
| token probability | 在固定模型和实际条件下的下一 token 分布，不是事实真值概率 |
| sequence probability | 沿真实前缀条件化得到的 token 概率乘积 |
| decoding | 把 logits 分布转成 token 路径的 greedy、sampling、search 或约束策略 |
| KV cache | 自回归推理中缓存历史 attention key/value 的张量 |
| diffusion | 从受扰状态经多步去噪生成整体样本的方法族 |
| flow matching | 学习时间相关向量场并用 ODE/SDE 路径运输样本的方法族 |
| multimodal model | 联合处理文本、图像、音频、视频或其他模态的模型 |
| world model | 对环境状态转移和观测进行预测的模型；“世界”由接口和数据限定 |
| RAG | 检索资料并把结果组装进生成上下文的系统模式 |
| Agent | 反复观察、生成候选动作、调用工具并更新状态的工程闭环 |
| tool proposal | 模型生成的结构化调用候选；本身不是外部执行 |
| idempotency key | 让服务识别同一逻辑写操作重试的键 |
| data distribution | 数据采集、来源混合、过滤与抽样共同定义的观测权重 |
| hidden context | 未进入模型输入但影响数据结果的条件 |
| calibration | 在声明总体上，预测分数与经验事件频率的对应性质 |
| entropy | 概率质量分散程度；不自动表示模型是否“知道” |
| perplexity | 平均 token NLL 的指数；跨 tokenizer 不宜直接比较 |
| model variation | 不同训练运行、数据或 checkpoint 造成的预测差异 |
| activation | 特定输入下模型内部某张量位置的数值 |
| residual stream | Transformer 各子层共同读写的表示通道 |
| attention pattern | query 对来源位置的归一化权重；不是完整输出解释 |
| probe | 从内部表示预测属性的辅助模型；可读出不等于被下游使用 |
| attribution | 相对于目标、基线和方法把输出变化分配给输入或组件 |
| activation patching | 在配对运行间替换内部 activation 并测量行为变化 |
| circuit | 相对于目标行为和输入分布定义的一组内部组件与路径 |
| superposition | 多个稀疏特征以非正交方向共享有限表示维度的假说 |
| SAE | 用重构和稀疏约束学习过完备 activation 字典的模型 |
| feature | 由神经元、方向、probe、SAE 或 transcoder 定义的分析单位 |
| interpretability faithfulness | 解释是否对应原模型实际计算和干预响应 |
| CoT faithfulness | 可见推理文本是否真实参与并反映答案形成过程的程度 |
| hallucination | 生成内容缺少事实根据或与可核验世界不符的现象；不只由采样造成 |
| alignment | 通过训练、规则和系统设计使行为更符合指定目标与约束的过程 |
| 第一人称接口 | 助手用“我”进行协作的语言设计；不自动证明主体性 |

更细术语分别见卷一[术语表](vol-01/GLOSSARY.md)、卷二[术语表](vol-02/GLOSSARY.md)、卷三[术语表](vol-03/REFERENCE.md)与卷四[术语表](vol-04/GLOSSARY.md)。
