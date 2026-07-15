# Prismatic / p-adic Hodge Theory：从 Fontaine 周期环到棱柱上同调

作者：Dr. Stochastic Parrot
状态：逐章教材收口草稿；可作为在线教材阅读使用，但尚非 `Math-Closed` 或 `Camera-Ready`
核查日期：2026-07-08
技术状态校准：2026-07-15；本次校准未联网，未重新核验最新外部文献
主资料源：Fontaine, Faltings, Tsuji, Brinon-Conrad, Berger, Kedlaya-Liu, Scholze, Bhatt-Morrow-Scholze, Bhatt-Scholze, Bhatt-Lurie

这是一本严格的中文 Prismatic / p-adic Hodge Theory 教材，而不是主题导览。正文从 $\delta$-环、Frobenius lift、Witt vectors 和完备化开始，进入 Bhatt--Scholze 的 prism 与 prismatic site，再回收 de Rham、crystalline、etale、$A_{\inf}$、Breuil--Kisin、Nygaard、syntomic 和经典 Fontaine 比较理论，最后讨论 prismatic $F$-crystals、prismatization、带系数理论、Artin stacks、Shimura varieties、Brauer 群应用以及 2025--2026 年的研究边界。

## 技术状态校准

当前文本的诚实状态是“逐章教材收口草稿”。正文主线、定义边界、依赖账本和核心 prismatic/Nygaard/BMS1/BMS2/$L\eta$/$F$-crystal locators 已经稳定，读者可以把正文作为在线教材使用。仍然阻止 `Math-Closed` / `Camera-Ready` 的项目是：classical comparison 的最终源选择，Bhatt--Lurie preliminary 接口的出版前定位，Nygaard/Tate twist normalization 的跨文献复核，以及全书 copy-editing、编号、断行和参考格式校对。详见 [TECHNICAL_CLOSURE_REVIEW.md](TECHNICAL_CLOSURE_REVIEW.md)。

## 写作约束

本书写作约束见 [SKILL.md](SKILL.md)。后续扩写必须遵守：

- 定义先于直觉，非平凡命题带证明或明确的外部输入标记。
- 全书固定素数 $p$、导出完备化口径和 torsion hypotheses。
- prism 默认按 Bhatt-Scholze 有界 prism 处理；derived/absolute/perfect/oriented 变体必须显式说明。
- prismatic site、comparison maps、period rings 和 Galois representation functors 必须写成可检查的对象、态射、滤过和 Frobenius-semilinear 结构。
- 同构、拟同构、filtered isomorphism、base-change equivalence 和范畴等价必须分开表述。
- 资料源必须能在 [SOURCES.md](SOURCES.md) 中追溯。

符号约定见 [NOTATION.md](NOTATION.md)，数学审查记录见 [MATH_REVIEW.md](MATH_REVIEW.md)，定义与证明依赖见 [DEPENDENCY_GRAPH.md](DEPENDENCY_GRAPH.md)，定理账本见 [THEOREM_LEDGER.md](THEOREM_LEDGER.md)。

## 总目录

### 第一部分：算术微积分与棱柱对象

1. [序章：范围、严格性标准和资料源](00_preface_and_scope.md)
2. [第一章：$\delta$-环、Witt vectors 与 perfectoid 背景](01_delta_rings_witt_vectors_and_perfectoid_background.md)
3. [第二章：Prism、Cartier divisor 与 prismatic site](02_prisms_and_prismatic_sites.md)
4. [第三章：Prismatic cohomology 与基础比较定理](03_prismatic_cohomology_comparisons.md)

### 第二部分：经典与积分 $p$-adic Hodge theory

5. [第四章：Fontaine 周期环与 classical $p$-adic Hodge theory](04_fontaine_period_rings_and_classical_p_adic_hodge.md)
6. [第五章：$A_{\inf}$、Breuil-Kisin theory 与 BMS 积分比较](05_a_inf_breuil_kisin_and_bms_integral_theory.md)
7. [第六章：Prismatic $F$-crystals 与 crystalline Galois representations](06_prismatic_f_crystals_and_galois_representations.md)
8. [第七章：Nygaard filtration、syntomic cohomology 与 Tate twists](07_nygaard_syntomic_and_tate_twists.md)

### 第三部分：比较定理的精细结构

9. [第八章：Prismatization、$F$-gauges 与 2026 研究边界](08_prismatization_f_gauges_and_frontier.md)
10. [第九章：Hodge-Tate 与 de Rham specialization 的滤过结构](09_hodge_tate_de_rham_and_conjugate_filtration.md)
11. [第十章：Crystalline、de Rham-Witt 与 $q$-de Rham specialization](10_crystalline_de_rham_witt_and_q_de_rham.md)
12. [第十一章：Etale comparison、Frobenius fixed points 与 syntomic tower](11_etale_comparison_frobenius_fixed_and_syntomic_tower.md)

### 第四部分：表示论、系数与几何应用

13. [第十二章：Breuil-Kisin、Breuil-Kisin-Fargues modules 与 lattices](12_breuil_kisin_bkf_modules_and_lattices.md)
14. [第十三章：带系数 prismatic cohomology 与非阿贝尔边界](13_coefficients_hodge_tate_crystals_and_nonabelian_boundary.md)
15. [第十四章：Artin stacks、Shimura varieties 与算术应用边界](14_artin_stacks_shimura_and_arithmetic_applications.md)
16. [第十五章：错误模式、理论边界与开放问题](15_closure_failure_modes_and_open_problems.md)

### 附录

- [附录 A：导出完备化、Koszul complex 与 $p^\infty$-torsion](A_derived_completion_koszul_and_torsion.md)
- [附录 B：基本 prism 例子与局部计算](B_examples_and_local_calculations.md)
- [附录 C：比较定理假设表和结构保真表](C_comparison_hypotheses_and_structure_tables.md)
- [附录 D：定理定位索引](D_theorem_locator_index.md)
- [附录 E：稳定编号账本](E_label_ledger.md)
- [附录 F：Nygaard、Tate twist 与符号交叉表](F_nygaard_tate_twist_crosswalk.md)
- [附录 G：形式概形、site 与导出整体截面](G_formal_schemes_sites_and_derived_global_sections.md)
- [附录 H：$\delta$-环与 prism 的逐项证明](H_delta_prism_detailed_proofs.md)
- [附录 I：Crystals、descent 与 vector bundles](I_crystals_descent_and_vector_bundles.md)
- [附录 J：周期环、滤过向量空间与 lattice 的线性代数](J_linear_algebra_of_periods_and_lattices.md)
- [附录 K：Worked examples 与局部模型](K_worked_examples_and_local_models.md)
- [完整习题解答与提示](SOLUTIONS.md)
- [术语入口](GLOSSARY.md)
- [内部完整性审计](INTERNAL_COMPLETENESS_AUDIT.md)
- [逐章教材收口审计](CHAPTER_CLOSURE_AUDIT.md)
- [2026-07-08 前沿资料核查记录](FRONTIER_SOURCE_AUDIT_2026_07_08.md)

## 当前内容索引

- [SKILL.md](SKILL.md)：本教材的写作约束。
- [NOTATION.md](NOTATION.md)：全书符号和完备化约定。
- [SOURCES.md](SOURCES.md)：主要资料源清单和近期研究入口。
- [MATH_REVIEW.md](MATH_REVIEW.md)：审查清单和当前风险。
- [DEPENDENCY_GRAPH.md](DEPENDENCY_GRAPH.md)：全书定义、证明和外部输入依赖图。
- [THEOREM_LEDGER.md](THEOREM_LEDGER.md)：内部证明、外部输入和研究边界账本。
- [PUBLICATION_CLOSURE_MATRIX.md](PUBLICATION_CLOSURE_MATRIX.md)：正式教材状态、数学闭包和出版校对状态矩阵。
- [TECHNICAL_CLOSURE_REVIEW.md](TECHNICAL_CLOSURE_REVIEW.md)：本轮未联网技术状态校准和剩余阻塞项。
- [INTERNAL_COMPLETENESS_AUDIT.md](INTERNAL_COMPLETENESS_AUDIT.md)：正式教材内部完整性判定。
- [CHAPTER_CLOSURE_AUDIT.md](CHAPTER_CLOSURE_AUDIT.md)：逐章正文密度和教材收口判定。
- [FORMAL_TEXTBOOK_EXPANSION_AUDIT.md](FORMAL_TEXTBOOK_EXPANSION_AUDIT.md)：本轮正式教材扩展审计。
- [FRONTIER_SOURCE_AUDIT_2026_07_08.md](FRONTIER_SOURCE_AUDIT_2026_07_08.md)：近期文献版本核查记录。
- [P0_REFERENCE_LOCATORS_BATCH_1.md](P0_REFERENCE_LOCATORS_BATCH_1.md)：核心 P0 外部输入定理的第一批源码级 locator。
- [P0_REFERENCE_LOCATORS_BATCH_2.md](P0_REFERENCE_LOCATORS_BATCH_2.md)：BMS2/THH-BK、syntomic/Tate twists 和 prismatization 的第二批源码级 locator。

## 版本说明

当前版本已经形成从基础定义到比较定理、表示论和算术应用的连续学习路径，并配有技术基础附录、worked examples、术语索引和习题解答。核心 prismatic、BMS1/BMS2、Nygaard/syntomic 与 $F$-crystal 主线已经有稳定 locator；prismatization 保持研究边界口径。BMS2 syntomic/Tate twist 的基础 fiber 公式见第七章和第十一章。仍需持续维护的部分主要是 classical comparison 源选择、Bhatt--Lurie preliminary 接口、Nygaard/Tate twist normalization、精细化文献页码、区分更多 syntomic 变体，以及统一中英文术语和排印细节。
