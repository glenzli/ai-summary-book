# 术语与符号

| 术语 | 本书口径 |
|---|---|
| target system | 模型版本、输入/环境/输出可测空间、确定可测映射或随机核、可测指标和乘积空间目标分布组成的研究对象 |
| explanandum / 被解释项 | 目标系统中范围明确的待解释现象、规律或计算结构；它不是研究者向该现象提出的问题 |
| explanation question / 解释问题 | 针对被解释项提出的“为什么/如何”问题；应声明所寻求的功能、因果、机制或语义关系，以及评价候选回答的证据标准 |
| kernel-averaged metric | 随机系统中对输出核积分得到的标量 $\overline m_\theta(x,e)=\int m(y)K_\theta((x,e),\mathrm{d}y)$；不把概率测度直接代入 $m$ |
| evidence operator | 把候选假说映到指定协议可观察总体量的映射 $\mathcal O_P$ |
| identifiability | 全局识别指证据算子在假说类上单射；点识别指指定假说的纤维为单点；商空间识别只确定等价类 |
| observational equivalence | 两系统在指定测试集或观察协议上不可区分；不等于全域函数或机制相同 |
| attribution | 在固定目标标量、基线、路径或合作规则下，把差异分配给坐标/组件的规则 |
| saliency | 局部敏感度或显著性数值，不预设因果语义 |
| completeness | 路径归因总和等于端点输出差；是守恒恒等式，不是语义唯一性 |
| attention weight | 指定头、查询位置和掩码下值向量的归一化混合系数 |
| direct value readout | 固定注意力权重与线性读出时的项 $\alpha_j u^\top W_Ov_j$ |
| probe | 从内部表示预测外部目标属性的辅助模型，必须连同探针类、训练算法和分布说明；命题 5.3 专指全体仿射分数探针类 $\mathcal P_{\mathrm{aff}}$ |
| decodability | 某属性在指定探针类、分布、损失和阈值下可预测 |
| control task | 与真实任务共享接口、但按设计主要测量探针记忆或容量的辅助任务 |
| intervention | 用新方程替换计算图中一个或多个结构方程，再重算后代节点 |
| source/base | patching 中提供激活的运行/接收替换激活的运行；不必分别等于正确/错误输入 |
| activation patching | 用 source 运行的激活替换 base 运行的指定节点，并测量结果变化 |
| interchange intervention | 在低层模型中交换表示，以检验其是否具有对齐高层变量的干预行为 |
| causal tracing | 通过一组内部替换定位对指定输出指标有局部模型内因果作用的状态 |
| component granularity | 电路研究预先选择的节点或已类型化边消息；两种粒度不得在同一 $\Gamma$ 中含混混用 |
| $A_b(S)$ | 同时按冻结替换规则 $b$ 消融组件集 $S\subseteq\Gamma$ 并重算后代的算子；$A_b(C)$ 测候选集删除响应，$A_b(\Gamma\setminus C)$ 测只保留候选集时的充分性 |
| circuit | 在声明分布、指标、组件粒度、替换规则与干预族内满足保留充分性、删除响应和未见干预预测要求的组件假说 |
| feature | 内部激活方向、稀疏坐标或可操作属性的建模单位；不预设自然概念身份 |
| SAE | sparse autoencoder，按重构与稀疏代理目标学习字典的模型 |
| feature absorption | 语义上应触发某特征的样本由其他更专门特征吸收，导致该特征漏触发的经验现象 |
| faithfulness | 解释对已声明计算量或干预响应的保真关系；必须给判据和评估分布 |
| stability | 解释对基线、seed、输入、checkpoint 或等价参数化变化的保持程度 |
| plausibility | 解释对人类是否自然或有说服力；不等同于 faithfulness |
| metric emergence | 报告指标在有限观测尺度间出现跃升 |
| mechanism transition | 内部计算假说随尺度发生可识别的定性变化，证据责任强于指标跃升 |
| operational label | 由明确情境、指标、干预和判据支持的心理词简称，不自动成为内在属性 |
| visible CoT | 会进入后续自回归上下文的可见 token 序列；其因果作用、报告忠实性和隐藏计算对应关系是不同问题 |
| hallucination | 输出与选定事实标准、来源或任务约束不一致的工程类别；报告时仍需细分检索失败、引用伪造、实体混淆等机制 |
| optimization / strategy / intention | 训练算法使用的优化标量、从行为拟合的策略目标与涉及主体归属的意图；三者不得互换 |
| deception protocol | 同时检查错误陈述、真值区分能力以及对受众、监督或后果的策略性反事实依赖；通过协议不自动裁决主体本体论 |
| $F_\theta$ | 固定参数确定模型的可测输入输出映射 |
| $h_l(x)$ | 输入 $x$ 在层 $l$ 的内部表示 |
| $m(y)$ | 作用于输出 $y\in\mathcal Y$ 的可测标量指标，如 logit 差 |
| $I_i$ | 对第 $i$ 个变量的干预 |

“表示”“概念”“机制”“理解”“意图”和“欺骗”都不是无条件的本体标签。正文使用时会说明操作定义、证据层级和未支持的升级。
