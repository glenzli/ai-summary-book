# 计算理论、类型论与程序语义

作者：Dr. Stochastic Parrot

状态：严格中文 MD 教材有限范围审定版；正文、索引、来源和闭合审计按同一证明责任维护

范围：`books/computation-types-semantics/` 内部文件

核心口径：所有程序均先被视为有限语法对象，再分别解释为计算过程、类型判断和语义对象。

本书把三个常被分开教授的主题放在同一条证明链中：计算理论说明哪些任务存在算法边界，类型论说明哪些程序形状可在语法层排除错误，程序语义说明程序语句在状态、值域或断言上的含义。全书不把“程序能运行”“程序有类型”“程序满足规格”混为一谈；每个结论都标明所在层级、使用的输入定理和证明状态。

## 阅读对象与前提

默认读者熟悉集合、关系、归纳定义、一阶逻辑和基础离散数学。附录 A-C 重述本书会实际使用的集合、序、归纳、共归纳和证明规则模板；正文不依赖未登记的外部技术。遇到大型经典定理时，正文只使用精确陈述，并在 [SOURCES.md](SOURCES.md) 和 [THEOREM_INDEX.md](THEOREM_INDEX.md) 登记为外部输入。

## 章节

1. [第 0 章：范围、对象层级与证明状态](00_preface_scope_and_metatheory.md)
2. [第 1 章：有效过程、机器与可计算函数](01_effective_procedures_and_machines.md)
3. [第 2 章：不可判定性、归约与 Rice 边界](02_undecidability_and_reductions.md)
4. [第 3 章：λ 演算、替换与归约](03_lambda_calculus_and_combinatory_computation.md)
5. [第 4 章：简单类型、类型安全与正规化](04_simple_types_and_normalization.md)
6. [第 5 章：依赖类型与构造性逻辑](05_dependent_types_and_constructive_logic.md)
7. [第 6 章：多态、递归类型与计算效应](06_polymorphism_recursion_and_effects.md)
8. [第 7 章：操作语义、求值上下文与抽象机](07_operational_semantics_and_abstract_machines.md)
9. [第 8 章：指称语义、域与不动点](08_denotational_semantics_domains_and_fixed_points.md)
10. [第 9 章：公理语义、程序逻辑与验证](09_axiomatic_semantics_logics_and_verification.md)
11. [第 10 章：表达力、完全抽象与可靠性边界](10_expressivity_full_abstraction_and_synthesis.md)

## 附录与账本

- [附录 A：集合、逻辑与序理论前提](A_set_logic_and_order_prerequisites.md)
- [附录 B：归纳、共归纳与不动点模板](B_induction_coinduction_and_fixed_points.md)
- [附录 C：推导规则和元定理证明模板](C_proof_rules_and_metatheorem_templates.md)
- [NOTATION.md](NOTATION.md)：符号、判断和层级约定。
- [SOURCES.md](SOURCES.md)：外部输入来源与定位状态。
- [THEOREM_INDEX.md](THEOREM_INDEX.md)：定义、定理、外部输入和证明状态索引。
- [DEPENDENCY_GRAPH.md](DEPENDENCY_GRAPH.md)：章节依赖、定理依赖和外部输入边界。
- [CONTENT_CLOSURE_AUDIT.md](CONTENT_CLOSURE_AUDIT.md)：本体闭合审计。
- [SOLUTIONS.md](SOLUTIONS.md)：正文练习的解题要点。

## 本书的证明状态标签

- **内部证明**：正文给出完整证明，并覆盖相关规则或构造子的全部情形。
- **外部输入**：本书使用但不重证的深结果；必须固定演算、假设和结论，并给出版本及章节或定理定位。
- **边界说明**：非定理性的限制、反例或研究接口，不作为后续证明的前提。

本书不把提纲性论证计作终态。外部输入之后可以附证明路线以帮助阅读，但路线不增加任何已证结论。

当前版本的目标不是替代经典专著，而是给出一条闭合的教材主线：从有限语法到可计算性边界，再到类型安全、语义等价和程序验证。
