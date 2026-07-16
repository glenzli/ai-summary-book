# 数学物理基础：从几何、表示论到量子场论

作者：Dr. Stochastic Parrot
状态：可审定内容收口版；主线范围按 00-10 章与 A-C 附录冻结，外部边界见审计文件
主资料源及版本定位见 [SOURCES.md](SOURCES.md)

本书面向已经熟悉高等微积分、线性代数、常微分方程和基础复分析的读者。正文把数学物理中反复出现的三条语言线索放在同一体系内：微分几何给出相空间、场和规范自由度；表示论给出对称性、谱分解和粒子标签；泛函分析与分布论给出量子态、算符和场的最低限度严谨框架。量子场论部分只建立 perturbative 与 axiomatic 接口所必需的核心结构，不把重整化群、规范异常或构造性 QFT 写成已经完全内部证明的定理。

## 写作约束

本目录写作必须遵守 [SKILL.md](SKILL.md)。核心规则如下：

- 使用中文；首次出现的标准英文术语括注。
- 每个核心对象先给定义、约定和使用边界，再进入命题或计算。
- 非平凡结论必须标记为 `P`、`S` 或 `E`：`P` 只表示正文已经给出完整证明；`S` 表示带正规化、微扰阶数或有效能区边界的标准物理推导；`E` 表示精确陈述并定位来源的外部输入定理。
- 解释外部定理机制的文字只能称为“证明路线（外部输入）”，不能承担 `P` 的证明责任。
- 外部输入必须能在 [SOURCES.md](SOURCES.md) 中追溯。
- 全书符号以 [NOTATION.md](NOTATION.md) 为准；新增符号必须同步登记。
- 正文自足到“主线可闭合”：大型定理可作为外部输入，但不得被伪装成已在正文证明。

## 目录

0. [序章：对象、证明状态与闭合范围](00_preface_and_scope.md)
1. [第一章：流形、张量与变分语言](01_smooth_manifolds_tensors_and_variational_language.md)
2. [第二章：辛几何与 Hamilton 系统](02_symplectic_geometry_and_hamiltonian_systems.md)
3. [第三章：Lie 群、Lie 代数与表示](03_lie_groups_lie_algebras_and_representations.md)
4. [第四章：纤维丛、联络与规范场](04_fiber_bundles_connections_and_gauge_fields.md)
5. [第五章：泛函分析、分布与谱理论](05_functional_analysis_distributions_and_spectral_theory.md)
6. [第六章：量子力学、对称性与自旋](06_quantum_mechanics_symmetry_and_spin.md)
7. [第七章：经典场论、Noether 定理与 Lagrange 几何](07_classical_field_theory_noether_and_lagrangian_geometry.md)
8. [第八章：路径积分、重整化与有效场论](08_path_integrals_renormalization_and_effective_fields.md)
9. [第九章：量子场、Wightman 公理与 Fock 空间](09_quantum_fields_wightman_canonical_and_fock.md)
10. [第十章：规范量子场论、异常与几何接口](10_gauge_theory_anomalies_and_qft_interfaces.md)

附录：

- [附录 A：线性代数、拓扑与测度工具](A_linear_algebra_topology_and_measure.md)
- [附录 B：范畴、同调与指标工具](B_category_homological_and_index_tools.md)
- [附录 C：公式、约定与常用表](C_formulae_conventions_and_tables.md)

全书交叉文件：

- [NOTATION.md](NOTATION.md)：符号与约定。
- [SOURCES.md](SOURCES.md)：资料源与使用边界。
- [THEOREM_INDEX.md](THEOREM_INDEX.md)：主要定义、命题、定理状态索引。
- [DEPENDENCY_GRAPH.md](DEPENDENCY_GRAPH.md)：章节依赖图。
- [CONTENT_CLOSURE_AUDIT.md](CONTENT_CLOSURE_AUDIT.md)：内容闭合审计。
- [SOLUTIONS.md](SOLUTIONS.md)：核心练习解答。

## 内容范围

本书不试图替代微分几何、表示论、泛函分析或量子场论的完整专著。它的闭合目标较窄：读者读完后应能把经典力学、量子力学和量子场论中的基本结构写成统一的几何与表示论语言，并知道哪些结论已经在正文中证明，哪些依赖外部大定理，哪些只是标准物理推导。

## 先修知识合同

本书把有限维线性代数（含复矩阵特征值理论）、多元微积分、常微分方程的局部存在唯一性、Lebesgue 积分的基本收敛定理以及基础复分析视为先修知识。正文中的 `P` 可以直接调用这些结果；超出这一合同的 Darboux 定理、Peter-Weyl 定理、无界自伴算子谱定理、双曲 PDE 基本解、Wightman 重构和指标定理等均逐项标为 `E`。因此“书内完整证明”是相对于这份明示先修合同而言，而不是宣称从集合论开始重建全部数学。
