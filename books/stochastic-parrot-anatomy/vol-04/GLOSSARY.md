# 符号与术语

本表固定卷四内部用法。术语不是模型本体类别；每项主张仍须注明模型、输入分布、hook site、目标指标和验证协议。

## 通用符号

| 符号 | 含义 |
|---|---|
| $M_\theta$ | 权重冻结在 $\theta$ 的模型 |
| $\tau$ | tokenizer、序列化与 chat template |
| $\mathcal D_{\mathrm{eval}}$ | 可解释性主张针对的输入分布；不默认等于训练分布或部署流量 |
| $S$ | 标量行为或内部目标 |
| $\mathcal U$ | 可观察/干预的内部单位集合 |
| $\mathcal I$ | 允许的干预族 |
| $x_{\ell,p}$ | 层 $\ell$、位置 $p$ 的 residual state |
| $A_{p,j}^{h}$ | head $h$ 从目标位置 $p$ 到 source $j$ 的 attention weight |
| $u_{a,b}$ | 候选 token $a,b$ 的 unembedding difference |
| $F_M,F_E$ | 原模型与解释/替代模型的目标输出 |
| $I$ | 一个明确赋值规则的 intervention |
| $\Delta_I$ | 指定干预造成的目标差 |

## 内部对象

| 术语 | 本卷含义与边界 |
|---|---|
| parameter / weight | 训练后保存、跨输入固定的模型参数 |
| activation | 特定输入、层、位置和前向时刻产生的张量值 |
| residual stream | Transformer 子层共同读写的 $d_{model}$ 维状态通道 |
| hook site | 读取或修改张量的精确计算位置，含 norm/addition 前后约定 |
| attention pattern | query 对 source positions 的归一化路由权重；不含 value 内容 |
| head result | value 混合经 output projection 后写入 residual 的向量 |
| QK / OV | attention 的 source 选择双线性形式 / value 读取与写入变换 |
| neuron | MLP 中间层的架构坐标；不天然对应单一概念 |
| direction | 表示空间中的一维向量及其投影约定 |
| subspace | 允许内部换基的一组方向张成空间 |
| feature | 由 neuron、direction、probe、SAE 或 transcoder 明确定义的分析单位 |
| latent | 分析器学习出的隐藏坐标；不是原模型天然存在的同义词 |
| read/write | 下游 weight 对方向敏感 / 上游模块向方向贡献 activation |
| Jacobian | 一个内部变量对另一变量的局部导数矩阵 |

## 行为与归因

| 术语 | 本卷含义与边界 |
|---|---|
| logit difference | 两候选 logits 之差，常用连续行为指标 |
| minimal pair | 尽可能只改变目标因素的一对输入；自然语言中仍需检查伴随变化 |
| behavioral signature | 机制假说预先推出的一组外部行为预测 |
| invariance / equivariance | 输入变换后输出不变 / 按指定表示同步变化 |
| saliency | 目标对输入或 activation 的局部敏感性归因 |
| Gradient × Input | 相对零缩放方向的一阶局部贡献近似 |
| Integrated Gradients | 从指定基线沿指定路径积分梯度所得归因 |
| attribution completeness | 归因项按定义加总为端点 score 差；不等于机制完整性 |
| occlusion | 删除、mask 或替换输入单位的有限反事实 |
| Shapley value | 对给定 coalition value function 的公理化效应分配；不是唯一物理原因 |
| direct logit attribution | residual contribution 对目标 logit difference 的局部直接读出；含 final norm 时一阶式为 $u^\top J_{\operatorname{Norm}_f}(x_L)c$，只有忽略或冻结 normalization 后才约化为 $u^\top c$ |

## 读出、几何与说明

| 术语 | 本卷含义与边界 |
|---|---|
| probe | 从冻结 activation 预测研究标签的辅助读出器 |
| decodability | 属性可由指定 probe 类在指定测试分布读出的性质 |
| control task | 保留部分输入统计、随机化标签关系以检测 probe 记忆的任务 |
| selectivity | 真实任务相对 control task 的读出性能增量；依控制构造 |
| MDL probing | 以在线编码长度操作化读出样本复杂度 |
| CKA / CCA | 带不同不变性的表示相似性度量 |
| probe equivalence | 多个读出方向在观察数据上近似同预测，因而不可区分 |
| amnesic probing | 投影删除 probe 子空间后测行为与剩余可读性的 causal-probing 方法 |
| input-side semantics | 什么输入条件使单位激活 |
| output-side effect | 单位经 decoder/output weight 向下游写入什么 |
| auto-interpretation | 用另一模型生成并在独立样本上评分内部单位说明 |
| simulation score | 自然语言说明预测单位 activation 的性能；不含下游机制保证 |
| polysemantic | 一个分析单位对多个难由单一规则统一的模式起作用 |
| monosemanticity | 单位能否由相对统一、可预测规则描述的程度；不是天然二元标签 |

## 干预与回路

| 术语 | 本卷含义与边界 |
|---|---|
| ablation | 用零、均值、重采样值或投影等删除组件信息的指定干预 |
| activation patching | 把 source run 的内部状态替换进 base run |
| causal tracing | 扫描 patch sites 以定位行为恢复路径的方法族 |
| path patching | 只允许 sender effect 经指定 receiver/路径传播的多 run 干预 |
| steering | 沿内部方向或 feature decoder 修改 activation |
| off-manifold | 干预后状态偏离自然 activation 支持集；没有唯一通用距离 |
| necessity | 删除组件后目标受损；只相对于删除规则、分布与目标成立 |
| sufficiency | 只保留/加入组件可恢复目标；只相对于基线与保留规则成立 |
| causal abstraction | 低层模型的 interchange interventions 与受限高层因果模型相容 |
| identifiability | 给定假说类、数据和干预族后，解释是否只剩一个等价类 |
| circuit | 相对于行为和补图处理规则定义的模型计算子图 |
| edge | 一个计算节点对下游节点输入的明确贡献路径 |
| causal scrubbing | 按高层假说对无关差异重采样并测试行为是否保持 |
| replacement model | 近似原 activation 或模块输出的辅助模型 |
| attribution graph | 对固定目标组织 feature、边、误差节点与 logit effect 的局部图 |
| behavioral faithfulness | 解释模型是否复现原模型自然运行输出 |
| intervention faithfulness | 解释与原模型是否对对应内部干预作同类响应 |
| circuit completeness | 候选 circuit 是否覆盖目标相关计算；须由 keep/remove/error 协议定义 |
| minimality | 删除候选 circuit 的元素会显著降低所声明保真度 |

## 稀疏表示与稳定性

| 术语 | 本卷含义与边界 |
|---|---|
| superposition | 多个稀疏 feature 以非正交方向共享有限表示维度的假说 |
| SAE | 以重构与稀疏目标学习过完备 activation 字典的模型 |
| encoder direction | 决定 SAE latent 如何由 activation 触发的方向 |
| decoder direction | 决定 latent 如何重构原空间的方向 |
| scale degeneracy | latent 与 decoder 反向缩放保持重构却改变稀疏惩罚的退化 |
| permutation symmetry | 重排 latent 并相应重排 decoder 不改变重构的等价性 |
| dead feature | 在目标数据上几乎从不越过 firing threshold 的 latent |
| feature splitting | 一个较宽模式被多个条件 latent 分担 |
| feature absorption | 某模式在关键上下文未由预期 latent 表示，而被其他 feature 吸收 |
| reconstruction error | 原 activation/模块输出与稀疏替代重构之差 |
| transcoder | 从模块输入以稀疏 latent 预测模块输出的替代模型 |
| cross-layer transcoder | 允许早层 features 直接预测多个后续模块输出的 transcoder |
| feature matching | 按 decoder、firing 与干预签名跨 run 对齐 features |
| functional role | 由行为签名、输入条件、输出方向和 intervention effect 定义的组件角色 |
| cross-seed stability | 同训练配方不同随机 seed 上功能或子空间的可重复程度 |
| cross-scale stability | 不同模型规模中经显式角色对齐后的机制可重复程度 |

## 推理过程与监测

| 术语 | 本卷含义与边界 |
|---|---|
| chain of thought (CoT) | 答案前生成并通常进入后续上下文的步骤化 token 序列 |
| trace intervention | 截断、替换或重排过程 token 后测答案变化 |
| CoT causal sensitivity | 最终答案对指定 trace 因素的反事实依赖 |
| factor disclosure | 真正影响行为的外部 cue 是否在 CoT 中被承认 |
| process completeness | 解决任务所需关键步骤是否可由 trace/monitor 恢复 |
| monitor | 从 CoT、动作、输出或 activation 预测行为属性的辅助系统 |
| monitorability | 特定 Agent、monitor、观察面、数据和目标共同决定的检测性能 |
| internal monitor | 从 activation 预测风险、状态或未来行为的 probe/classifier |
| obfuscation pressure | 促使模型隐藏、压缩或改写可监测过程的提示或训练压力 |
| lead time | monitor 发出可靠预警到目标行为发生之间的时间/步骤裕量 |

同一个英文词在不同文献中可能有不同定义。特别是 `completeness`、`faithfulness`、`feature` 和 `causal`，引用时应复述对应论文的操作定义，而不是依赖词面一致。
