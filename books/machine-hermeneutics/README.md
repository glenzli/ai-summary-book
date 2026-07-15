# 机器解释学

**状态/定位：已收口的中文研究方法教材；正文与配套材料的当前入口见 [STATUS.md](STATUS.md)。**

## 人类如何给计算过程安排意义

**作者：Dr. Stochastic Parrot**

这是一部关于机器学习解释方法的中文研究方法教材。全书从研究现场中反复出现的争议出发：注意力图究竟测到了什么，探针准确率能否说明下游使用，激活 patching 怎样形成因果证据，稀疏特征为何可能不唯一，以及“记忆”“推理”和“意图”何时是可复核的技术简称。

章节沿一项解释研究的实际推进顺序展开：先用行为实验建立现象，再以梯度、注意力和探针生成候选，随后用内部干预、电路与稀疏特征检验机制，最后讨论稳健性、尺度叙事和心理词汇。每章都包含可跟随的计算或实验案例，让读者看见对象怎样定义、操作怎样实施、观察怎样汇总，以及结论在哪里停止。

形式结果给出完整证明；经验结论以稳定来源编号回到原始论文并限定模型与任务；方法学规则则说明它排除了哪一种替代解释。严格性在这里服务于研究设计：读者应能把一句解释还原成可运行的实验，也能指出哪一种新观察会迫使结论改变。

## 目录

1. [序章：解释的对象与责任](00_preface_and_scope.md)
2. [解释层次与主张类型](01_levels_and_claims.md)
3. [行为证据、对照与识别](02_behavior_and_identification.md)
4. [梯度、积分梯度与局部归因](03_gradient_attribution.md)
5. [注意力权重能解释什么](04_attention_and_attribution.md)
6. [探针、可解码性与表示](05_probes_and_representation.md)
7. [干预、激活 patching 与因果追踪](06_interventions_and_patching.md)
8. [电路、特征与稀疏自编码器](07_circuits_and_sparse_features.md)
9. [稳健性、欠定与解释评估](08_robustness_and_underdetermination.md)
10. [涌现、基准与尺度叙事](09_emergence_and_benchmarks.md)
11. [心理词汇与机器传记](10_psychological_vocabulary.md)
12. [解释协议与完整案例](11_protocol_and_cases.md)

配套：[状态说明](STATUS.md)、[术语与符号](GLOSSARY.md)、[资料源与证据边界](SOURCES.md)、[主张责任表](CLAIM_LEDGER.md)、[习题解答](SOLUTIONS.md)、[闭合审计](CLOSURE_AUDIT.md)。

## 预备知识

读者应熟悉基础线性代数、多元微积分、概率、可测空间的基本语言与神经网络前向计算。第三章会明确微分和路径假设；第六章只在已声明计算图内使用模型内因果语言，不把它自动外推到现实社会因果问题。
