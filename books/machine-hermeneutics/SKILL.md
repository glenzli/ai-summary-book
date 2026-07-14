---
name: machine-hermeneutics
description: Use when writing or reviewing books/machine-hermeneutics. Requires a research-method textbook narrative built from concrete disputes and experiments, explicit explananda, separated mathematical/empirical/methodological/philosophical claims, causal identification boundaries, primary-source traceability, complete proofs for formal results, and calibrated language about representations, circuits, attention, concepts, agency, and understanding.
---

# 《机器解释学》写作规范

本书服从上级 [OET 本体严格性标准](../OET_RIGOR_STANDARD.md) 与
[人类可读教材叙事标准](../TEXTBOOK_NARRATIVE_STANDARD.md)，并针对经验型机器学习研究增加以下约束。

## 1. 先声明解释对象

每项研究必须区分三类对象：目标系统 $T$ 是被研究的模型或部署系统；被解释项（explanandum）$\Xi$ 是其中范围明确的现象、规律或结构；解释问题 $Q$ 是针对 $\Xi$ 提出的、带评价标准的“为什么/如何”问题。不得把目标系统本身、待解释现象和研究者的问题混成一个名词，也不得用一个输入上的发现无条件代表整个模型。

- 若目标系统确定，声明输入、环境、输出可测空间和可测映射；若随机，声明随机核 $K((x,e),B)$。
- 指标只作用于其已声明定义域中的输出；不得把 $m$ 无说明地直接作用于输入、干预或组件。
- “全局可识别”用于证据算子在假说类上单射；“点可识别”用于指定假说纤维为单点；只识别等价类时写明商关系。

## 2. 主张类型

- **形式结论**：给出对象、假设、量词与完整证明。
- **经验结果**：给出论文、模型、数据、干预或测量，并限制外推范围。
- **方法学规则**：说明它用于控制何种混杂或识别风险。
- **解释性论证**：列出前提与结论，不能用“证明”措辞冒充数学定理。
- **术语约定**：说明本书如何使用“表示”“机制”“电路”“概念”等多义词。

## 3. 识别纪律

- 相关性、可解码性、必要性、充分性、因果中介和语义同一性必须分开。
- attention 权重、梯度、积分梯度、探针、激活 patching、消融、稀疏自编码器各自回答不同问题。
- 每个干预要说明处理变量、对照、结果变量、干预分布、离分布风险和聚合方式。
- “神经元编码概念”至少要区分可预测、被使用、因果相关和跨上下文稳定四种主张。

## 4. 资料与引用

- 核心经验主张优先引用原始论文、作者技术报告或正式会议版本。
- 综述只用于组织线索，不作为争议结论的唯一证据。
- 不把工具作者的展示样例写成普遍定理。
- 近期结果需说明版本日期和是否经过同行评审。
- 正文经验句使用 `SOURCES.md` 的稳定编号，并写清该来源实际覆盖的模型、数据、指标与干预。

## 5. 完成要求

- 每章在第一个分节前从真实方法争议、失败解释、模型观察或可计算例子进入，不设置固定的“本章目标”“依赖”“主线”“本章小结”栏目。
- 每章至少包含一个可跟随的研究案例、计算或实验方案。方法案例必须明确对象、操作、观察、推断与边界；不能用原则清单替代案例。
- 章节之间应形成行为证据、局部归因、结构测量、探针、干预、电路/特征、稳健性与语义语言的推进关系。多数章节直接在最后一个实质小节中收尾；只有确有跨节综合内容时才另设收束标题。结尾应交替使用案例结果、竞争假说、待做实验或反例，不批量预告下一章。
- 定理、定义和边界声明嵌入研究问题：定理前说明它解决哪一识别缺口，证明后说明可复用机制；同一“不蕴含”边界在章内只完整说明一次。
- 练习保留独立分节，解答统一进入 `SOLUTIONS.md`。
- 全书主张状态进入 `CLAIM_LEDGER.md`，来源进入 `SOURCES.md`。
- 修改后运行 `validate.py`、OET 严格审计与 `git diff --check`。
- `SOLUTIONS.md` 对每个问题中的全部分问逐项作答；练习不得承担正文主线的未证命题。
