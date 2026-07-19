# 符号与术语

| 术语 | 本卷含义 |
|---|---|
| activation | 特定输入、层与位置在前向传播中产生的张量值 |
| residual stream | Transformer 子层共同读写的 $d_{model}$ 维状态通道 |
| neuron | MLP 中间层的架构坐标；不天然对应单一概念 |
| attention pattern | query 对来源位置的归一化权重；只描述路由的一部分 |
| QK / OV circuit | attention 的位置选择机制 / value 读取与写入机制 |
| feature | 由神经元、方向、probe、SAE 或 transcoder 定义的分析单位 |
| logit difference | 两个候选 logits 之差，常作为连续行为指标 |
| direct logit attribution | residual contribution 到 unembedding 方向的直接线性投影 |
| saliency | 目标对输入或 activation 的局部敏感性归因 |
| Integrated Gradients | 从基线到输入路径上积分梯度所得归因 |
| probe | 从 activation 预测标签的辅助读出器 |
| decodability | 属性可被指定 probe 类从表示中读出的性质 |
| selectivity | 真实任务 probe 相对 control task 的性能增量 |
| ablation | 用零、均值、重采样值等删除组件信息的干预 |
| activation patching | 把一个 run 的内部状态替换到另一个 run |
| causal tracing | 扫描 patch site 以定位行为恢复路径的方法族 |
| steering | 沿内部方向或 feature decoder 修改 activation |
| circuit | 相对于目标行为定义的模型计算子图 |
| path patching | 只允许 source effect 经指定路径到达 receiver 的干预 |
| faithfulness | 解释是否对应原模型实际计算与干预响应 |
| completeness | 解释覆盖目标行为中多少相关计算效应 |
| superposition | 多个稀疏特征以非正交方向共享有限表示维度的假说 |
| SAE | 用重构损失与稀疏约束学习过完备 activation 字典的模型 |
| transcoder | 用稀疏 latent 预测模型子模块输出的替代模型 |
| attribution graph | 对固定目标和输入组织 feature、边及 logit effect 的图 |
| polysemantic | 一个分析单位在多个难以统一的输入模式上起作用 |
| monosemanticity | 单位可由相对统一且可预测的语义/计算规则描述的程度 |
| reconstruction error | 原 activation/模块输出与稀疏替代重构之差 |
| CoT faithfulness | 可见推理文本是否真实参与并反映答案形成过程的程度 |
| internal monitor | 从 activation 预测风险、状态或未来行为的辅助模型 |

这些术语不是本体类别。每项主张仍须注明模型、输入分布、hook site、目标行为和验证协议。
