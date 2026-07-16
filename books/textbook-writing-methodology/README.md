# 如何写一本可收口的严格教材

作者：Dr. Stochastic Parrot

状态：严格 MD 教材内容收口版

范围：教材写作方法论、结构审定、证明责任、外部输入、例题习题、索引和收口流程。

本书讨论的不是文采，也不是出版排版，而是一个更窄也更硬的问题：怎样把一个巨大领域写成一部读者可以沿主线读完、作者可以用证据审定、后续维护者可以继续修订而不破坏逻辑的教材。这里的“严格”不是每个外部学科都重证一遍，而是所有对象、命题、证明责任和外部输入都能被定位、检查和维护；文件齐全只算机械闭合，不能自动推出内容收口。

## 目录

0. [序章：一本书何以收口](00_preface_what_closure_means.md)
1. [第一章：选题、范围和反范围](01_scope_and_anti_scope.md)
2. [第二章：读者模型与前置知识合同](02_reader_model_and_prerequisite_contract.md)
3. [第三章：主线、依赖图和章节分解](03_mainline_dependency_graph_and_chapters.md)
4. [第四章：定义、符号和术语压缩](04_definitions_notation_and_terms.md)
5. [第五章：命题、证明责任和外部输入](05_propositions_proofs_and_external_inputs.md)
6. [第六章：例子、反例和计算](06_examples_counterexamples_and_calculations.md)
7. [第七章：习题、解答和读者检验](07_exercises_solutions_and_reader_checks.md)
8. [第八章：来源、引用和版本边界](08_sources_citations_and_version_boundaries.md)
9. [第九章：索引、账本和机械审计](09_indices_ledgers_and_mechanical_audits.md)
10. [第十章：修订、收口和系列化维护](10_revision_closure_and_series_maintenance.md)

附录：

- [附录 A：文件模板和账本格式](A_templates_and_ledgers.md)
- [附录 B：审计清单和验证脚本模型](B_audit_checklists_and_validation.md)
- [附录 C：失败模式和重写案例](C_failure_modes_and_rewrites.md)

配套文件：

- [SKILL.md](SKILL.md)：本书写作约束。
- [NOTATION.md](NOTATION.md)：术语、状态和账本符号。
- [SOURCES.md](SOURCES.md)：方法论来源和本仓库内标准。
- [THEOREM_INDEX.md](THEOREM_INDEX.md)：原则、命题和证明状态。
- [DEPENDENCY_GRAPH.md](DEPENDENCY_GRAPH.md)：章节依赖图。
- [SOLUTIONS.md](SOLUTIONS.md)：核心练习解答。
- [CONTENT_CLOSURE_AUDIT.md](CONTENT_CLOSURE_AUDIT.md)：内容收口审定。

## 核心口径

本书采用四条原则：

1. 教材首先是一条依赖链，不是资料堆。
2. 严格性来自证明责任的诚实分配，不来自把外部理论伪装成短证明。
3. 可读性来自动机、例子和过渡，不来自降低定义精度。
4. 收口来自范围边界、内容审读和可复现证据；索引账本与机械审计只承担其中可自动检查的部分。
