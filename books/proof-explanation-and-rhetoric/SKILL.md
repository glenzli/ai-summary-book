---
name: proof-explanation-and-rhetoric
description: Use when writing or reviewing books/proof-explanation-and-rhetoric. Requires typed claims, explicit premises and quantifiers, valid proof rules, countermodels for invalid inferences, distinction among deductive proof, mathematical explanation, causal explanation, statistical evidence, analogy, rhetoric, and citation; complete proofs for formal results and source-backed empirical claims.
---

# 《证明、解释与漂亮话》写作规范

本书同时服从上级 [OET 本体严格性标准](../OET_RIGOR_STANDARD.md) 与
[人类可读教材叙事标准](../TEXTBOOK_NARRATIVE_STANDARD.md)。前者约束对象、定理与证明责任，后者约束这些内容如何进入正文；两者不能互相替代。

## 章节推进

1. 章首先给出一段可逐句审查的证明、统计说法、科学解释、图示、引用或 AI 推理，不用“本章目标”“依赖”“主线”代替问题。
2. 保留原始文本足够久，使读者能看到失败发生在哪一句、哪一项类型判断或哪一条依赖边。
3. 概念从修复需要中出现：先定位缺口，再引入定义、反模型、定理、证据协议或来源核对。
4. 每章至少完成一个贯穿案例或完整重写。案例必须给出输入、诊断、修复步骤和最终可支持表述，不能只列场景名称。
5. 定理前说明它解决案例中的哪个问题；证明后回到案例解释结论和剩余边界。
6. 章末用自然段交代修订结果并引向下一章，不设置固定“本章小结”。练习保留独立分节。

## 严格性规则

1. 先写可判定命题，再讨论证明；疑问、命令、隐喻和价值判断不得伪装成真假命题。
2. 每个论证列出前提、推理规则和结论；省略前提必须能被明确补回。
3. 有效性与前提真实性分开；反模型只反驳有效性或全称命题，不自动建立替代理论。
4. 定义先给类型、域和边界；必要条件、充分条件、双条件不得混用。
5. 数学证明、数学解释、因果解释、统计证据、机制叙述和修辞说服分别标记。
6. 图、类比和例子不得承担未写出的普遍证明。
7. 引用给出来源与确切用途；权威、引用数量和论文标题不替代论证。
8. AI 生成的推理按外部文本审查，不以流畅度、长度或 chain of thought 形式视为正确性证据。
9. 语言、解释/模型、语义蕴涵和推导系统分层声明；健全论证与演算可靠性使用不同术语。
10. 直接证明写任意对象，经典反证声明双重否定原则，归纳写良基对象，极限写收敛模式与交换条件。
11. 外部输入给出精确版本、正文用途和未重证边界；经验论文只支持其模型、任务和协议。
12. AI 章节分别定义正确性关系、理由、证书、验证器可靠性和过程忠实性，除非有定理或实验，不写无条件等价。

## 完成要求

每章必须有自然章首、可跟随案例、完整正文过渡和练习；所有练习有逐题解答。书内形式命题完整证明，深逻辑结果和经验论文精确引用。正文不得出现固定标题 `本章目标`、`依赖`、`主线`、`本章小结`。修改后运行 `validate.py`、作用域严格 OET 审计、裸 TeX 宏扫描与 `git diff --check`。
