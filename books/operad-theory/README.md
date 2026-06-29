# Operad Theory：从代数运算到 infinity-operad

作者：Dr. Stochastic Parrot
状态：operad theory 数学收口已达到，出版社级最终出版校对未完成
核查日期：2026-06-30
主资料源：May, Boardman-Vogt, Markl-Shnider-Stasheff, Loday-Vallette, Fresse, Moerdijk-Weiss, Cisinski-Moerdijk, Lurie, Ayala-Francis, Hoffbeck-Moerdijk

本书目标是写成一部严格的中文 Operad Theory 教材，而不是主题导览。正文从有限集上的对称序列、代入乘积和 operad 的单子/幺半对象定义开始，逐步进入经典例子、colored operads、自由 operad、树公式、代数、Koszul 对偶、bar-cobar 构造、模型范畴中的 operad、dendroidal sets 和 infinity-operads。

## 写作约束

本书写作约束见 [SKILL.md](SKILL.md)。后续扩写必须遵守：

- 定义先于直觉，非平凡命题带证明或明确的外部输入标记。
- 全书固定集合论宇宙和小性约定。
- 基础 operad 默认含 arity $0$ 且带对称群作用；变体必须显式说明。
- 代入、公理和代数结构必须写成可检查的有限集分块、树代入、自然变换或 operad morphism。
- 同伦与 infinity 内容必须区分严格等式、同构、弱等价、Quillen 等价和 infinity-范畴中的等价。
- 资料源必须能在 [SOURCES.md](SOURCES.md) 中追溯。

符号约定见 [NOTATION.md](NOTATION.md)，数学审查记录见 [MATH_REVIEW.md](MATH_REVIEW.md)，第二轮严格化路线见 [SECOND_PASS_STRICTIFICATION_PLAN.md](SECOND_PASS_STRICTIFICATION_PLAN.md)。

## 建议总目录

### 第一部分：普通 operad

1. [序章：范围、严格性标准和资料源](00_preface_and_scope.md)
2. [第一章：对称序列、代入乘积与 operad](01_symmetric_sequences_and_operads.md)
3. [第二章：Operad 代数、自由代数与单子](02_operad_algebras_free_algebras_and_monads.md)
4. [第三章：非对称 operad、偏复合与树](03_nonsymmetric_operads_partial_compositions_and_trees.md)
5. [第四章：自由 operad、生成元与关系](04_free_operads_generators_and_relations.md)
6. [第五章：Colored operad、多范畴与带类型代数](05_colored_operads_multicategories_and_typed_algebras.md)
7. [第六章：线性 operad、Schur 函子与经典例子](06_linear_operads_schur_functors_and_classical_examples.md)
8. [第七章：PROP、properad 与 wheeled 变体](07_props_properads_and_wheeled_variants.md)

### 第二部分：代数与同伦代数

9. [第八章：二次 operad 与 Koszul 对偶](08_quadratic_operads_and_koszul_duality.md)
10. [第九章：bar-cobar 构造与 twisting morphism](09_bar_cobar_constructions_and_twisting_morphisms.md)
11. [第十章：$A_\infty$、$L_\infty$ 与 $E_n$-operad](10_a_infinity_l_infinity_and_e_n_operads.md)
12. [第十一章：Gerstenhaber、BV 与 Deligne 猜想](11_gerstenhaber_bv_and_deligne_conjecture.md)
13. [第十二章：brace operad 与 Hochschild cochains](12_brace_operad_and_hochschild_cochains.md)
14. [第十三章：同伦转移定理与最小模型](13_homotopy_transfer_and_minimal_models.md)

### 第三部分：模型范畴与 infinity-operad

15. [第十四章：模型范畴中的 operad](14_operads_in_model_categories.md)
16. [第十五章：simplicial operad 与 topological operad](15_simplicial_and_topological_operads.md)
17. [第十六章：dendroidal sets 与树范畴 $\Omega$](16_dendroidal_sets_and_tree_category.md)
18. [第十七章：dendroidal inner Kan 条件与 homotopy operads](17_dendroidal_inner_kan_and_homotopy_operads.md)
19. [第十八章：Lurie-style infinity-operads 与 operadic fibration](18_lurie_infinity_operads_and_operadic_fibrations.md)
20. [第十九章：模型比较、straightening 与 operadic localization](19_model_comparison_straightening_and_operadic_localization.md)
21. [第二十章：factorization algebra、Fukaya categories 与几何应用](20_factorization_algebras_fukaya_categories_and_geometry.md)
22. [第二十一章：2025-2026 研究边界与开放问题目录](21_research_frontier_2026.md)

### 附录

- [附录 A：集合论宇宙、有限集骨架与 symmetric group 约定](A_set_theory_universes_finite_sets_and_symmetric_groups.md)
- [附录 B：树、分块、代入乘积和 coinvariants 公式](B_trees_partitions_substitution_and_coinvariants.md)
- [附录 C：模型范畴与 Quillen adjunction 复习](C_model_categories_and_quillen_adjunctions.md)
- [附录 D：资料源定理索引](D_source_theorem_index.md)
- [附录 E：符号、悬挂与分次约定](E_signs_suspensions_and_graded_conventions.md)
- [附录 F：经典 operad 的逐项验算](F_classical_operads_and_checked_examples.md)
- [附录 G：模型结构假设、admissibility 与 rectification 检查表](G_model_structure_hypotheses_and_rectification.md)
- [附录 H：树约定、叶标号与自由 operad 的群胚商](H_tree_conventions_and_free_operad_quotients.md)
- [附录 I：Koszul、bar-cobar 与 twisting 的严格约定](I_koszul_bar_cobar_strict_conventions.md)
- [附录 J：同伦转移的树公式与最小模型约定](J_homotopy_transfer_tree_formulas.md)
- [附录 K：Colored operad、模结构与 enriched 版本](K_colored_operads_modules_and_enrichment.md)
- [附录 L：$\mathcal P_\infty$-代数、$A_\infty/L_\infty$ 与 $E_n$ 约定](L_infinity_algebras_and_en_operad_conventions.md)
- [附录 M：Dendroidal、Lurie 与模型范畴比较图](M_dendroidal_lurie_and_model_comparison_map.md)
- [附录 N：Factorization homology、excision 与几何计算](N_factorization_homology_examples_and_geometry.md)
- [附录 O：失败模式、反例边界与不可混用约定](O_failure_modes_counterexamples_and_boundary_cases.md)
- [附录 P：低阶计算、逐项验算与小模型](P_low_arity_checks_and_worked_computations.md)
- [附录 Q：Koszul complex、bar-cobar 谱序列与计算样例](Q_koszul_complexes_and_bar_cobar_examples.md)
- [附录 R：模型范畴、admissibility 与 rectification 案例](R_model_category_case_studies.md)
- [附录 S：同伦转移低阶计算与最小模型样例](S_homotopy_transfer_worked_examples.md)
- [附录 T：Dendroidal horns、Segal core 与 normality 样例](T_dendroidal_horns_segal_and_normality_examples.md)
- [附录 U：PROP、properad 与 wheeled 图计算样例](U_props_properads_graphical_calculus_examples.md)
- [附录 V：带边界与分层 factorization homology 样例](V_stratified_and_boundary_factorization_examples.md)
- [附录 W：符号、悬挂与总次数交叉核对表](W_sign_convention_crosswalk.md)
- [附录 X：具体代数例子、反例与边界计算](X_concrete_algebraic_examples_and_counterexamples.md)
- [附录 Y：Infinity-operadic homology 与 Koszul 对偶前沿接口](Y_infinity_operadic_homology_and_koszul_frontier.md)
- [附录 Z：Operadic categories、relative Rezk nerve 与 Fukaya 前沿接口](Z_operadic_categories_relative_rezk_and_fukaya_frontier.md)

## 当前内容索引

- [SKILL.md](SKILL.md)：本教材的写作约束。
- [NOTATION.md](NOTATION.md)：全书符号和小性约定。
- [SOURCES.md](SOURCES.md)：主要资料源清单和近期研究入口。
- [MATH_REVIEW.md](MATH_REVIEW.md)：审查清单和当前风险。
- [SECOND_PASS_STRICTIFICATION_PLAN.md](SECOND_PASS_STRICTIFICATION_PLAN.md)：第二轮严格化路线图。
- [DEPENDENCY_GRAPH.md](DEPENDENCY_GRAPH.md)：全书定义、证明和外部输入依赖图。
- [THEOREM_LEDGER.md](THEOREM_LEDGER.md)：内部证明、外部输入和研究边界账本。
- [INTERNAL_OPERAD_CLOSURE_AUDIT.md](INTERNAL_OPERAD_CLOSURE_AUDIT.md)：operad theory 主体的内部定义闭合审计。
- [INTERNAL_NUMBERING_AND_CROSSREF_AUDIT.md](INTERNAL_NUMBERING_AND_CROSSREF_AUDIT.md)：第一至第七章编号和交叉引用审计。
- [LABEL_LEDGER_CH01_07.md](LABEL_LEDGER_CH01_07.md)：第一至第七章稳定 label 表。
- [LABEL_LEDGER_CORE_APPENDICES.md](LABEL_LEDGER_CORE_APPENDICES.md)：核心附录 A/B/H/K/P/U/X 稳定 label 表。
- [LABEL_LEDGER_CH08_21.md](LABEL_LEDGER_CH08_21.md)：第八至第二十一章稳定 label 表。
- [LABEL_LEDGER_REMAINING_APPENDICES.md](LABEL_LEDGER_REMAINING_APPENDICES.md)：剩余附录 C/D/E/F/G/I/J/L/M/N/O/Q/R/S/T/V/W/Y/Z 稳定 label 表。
- [CROSSREF_REWRITE_AUDIT.md](CROSSREF_REWRITE_AUDIT.md)：两轮散文交叉引用替换审计，覆盖主体章节、高级章节、主要附录和元文档。
- [PUBLICATION_CLOSURE_MATRIX.md](PUBLICATION_CLOSURE_MATRIX.md)：基本完本、数学收口和 camera-ready 出版状态闭包矩阵。
- [PUBLICATION_PROOFING_LEDGER.md](PUBLICATION_PROOFING_LEDGER.md)：最终出版校对账本，记录已修正项和 production/copy-editing 剩余包。
- [REFERENCE_LOCATOR_LEDGER.md](REFERENCE_LOCATOR_LEDGER.md)：最终出版前的 P0/P1/P2/R 引用定位账本。
- [FINAL_OPERAD_THEORY_CLOSURE.md](FINAL_OPERAD_THEORY_CLOSURE.md)：operad theory 自身的最终数学收口判定。
- [P0_REFERENCE_LOCATORS_BATCH_1.md](P0_REFERENCE_LOCATORS_BATCH_1.md)：P0 外部输入第一批精确定位。
- [P0_REFERENCE_LOCATORS_BATCH_2.md](P0_REFERENCE_LOCATORS_BATCH_2.md)：P0 外部输入第二批精确定位。
- [P0_REFERENCE_LOCATORS_BATCH_3.md](P0_REFERENCE_LOCATORS_BATCH_3.md)：P0 外部输入第三批精确定位。
- [P0_REFERENCE_LOCATORS_BATCH_4.md](P0_REFERENCE_LOCATORS_BATCH_4.md)：P0 外部输入第四批精确定位。
- [P0_REFERENCE_LOCATORS_BATCH_5.md](P0_REFERENCE_LOCATORS_BATCH_5.md)：P0 外部输入第五批精确定位。
- [P0_REFERENCE_LOCATORS_BATCH_6.md](P0_REFERENCE_LOCATORS_BATCH_6.md)：P0 外部输入第六批精确定位。
- [P0_REFERENCE_LOCATORS_BATCH_7.md](P0_REFERENCE_LOCATORS_BATCH_7.md)：P0 外部输入第七批精确定位。
- [P0_REFERENCE_LOCATORS_BATCH_8.md](P0_REFERENCE_LOCATORS_BATCH_8.md)：P0 外部输入第八批精确定位。
- [P0_REFERENCE_LOCATORS_BATCH_9.md](P0_REFERENCE_LOCATORS_BATCH_9.md)：P0 外部输入第九批精确定位。
- [P0_REFERENCE_LOCATORS_BATCH_10.md](P0_REFERENCE_LOCATORS_BATCH_10.md)：P0 外部输入第十批精确定位。
- [P1_REFERENCE_LOCATORS_FINAL_SWEEP.md](P1_REFERENCE_LOCATORS_FINAL_SWEEP.md)：P1 外部输入与几何边界最终定位批。
- [FRONTIER_SOURCE_AUDIT_2026_06_30.md](FRONTIER_SOURCE_AUDIT_2026_06_30.md)：近期前沿文献版本核查记录。
- [FRONTIER_SOURCE_AUDIT_2026_06_29.md](FRONTIER_SOURCE_AUDIT_2026_06_29.md)：前一轮近期文献核查记录。
- [00_preface_and_scope.md](00_preface_and_scope.md)：序章。
- [01_symmetric_sequences_and_operads.md](01_symmetric_sequences_and_operads.md)：第一章严格草稿。
- [02_operad_algebras_free_algebras_and_monads.md](02_operad_algebras_free_algebras_and_monads.md)：第二章严格草稿。
- [03_nonsymmetric_operads_partial_compositions_and_trees.md](03_nonsymmetric_operads_partial_compositions_and_trees.md)：第三章严格草稿。
- [04_free_operads_generators_and_relations.md](04_free_operads_generators_and_relations.md)：第四章严格草稿。
- [05_colored_operads_multicategories_and_typed_algebras.md](05_colored_operads_multicategories_and_typed_algebras.md)：第五章严格草稿。
- [06_linear_operads_schur_functors_and_classical_examples.md](06_linear_operads_schur_functors_and_classical_examples.md)：第六章严格草稿。
- [07_props_properads_and_wheeled_variants.md](07_props_properads_and_wheeled_variants.md)：第七章严格草稿。
- [08_quadratic_operads_and_koszul_duality.md](08_quadratic_operads_and_koszul_duality.md)：第八章严格草稿。
- [09_bar_cobar_constructions_and_twisting_morphisms.md](09_bar_cobar_constructions_and_twisting_morphisms.md)：第九章严格草稿。
- [10_a_infinity_l_infinity_and_e_n_operads.md](10_a_infinity_l_infinity_and_e_n_operads.md)：第十章严格草稿。
- [11_gerstenhaber_bv_and_deligne_conjecture.md](11_gerstenhaber_bv_and_deligne_conjecture.md)：第十一章严格草稿。
- [12_brace_operad_and_hochschild_cochains.md](12_brace_operad_and_hochschild_cochains.md)：第十二章严格草稿。
- [13_homotopy_transfer_and_minimal_models.md](13_homotopy_transfer_and_minimal_models.md)：第十三章严格草稿。
- [14_operads_in_model_categories.md](14_operads_in_model_categories.md)：第十四章严格草稿。
- [15_simplicial_and_topological_operads.md](15_simplicial_and_topological_operads.md)：第十五章严格草稿。
- [16_dendroidal_sets_and_tree_category.md](16_dendroidal_sets_and_tree_category.md)：第十六章严格草稿。
- [17_dendroidal_inner_kan_and_homotopy_operads.md](17_dendroidal_inner_kan_and_homotopy_operads.md)：第十七章严格草稿。
- [18_lurie_infinity_operads_and_operadic_fibrations.md](18_lurie_infinity_operads_and_operadic_fibrations.md)：第十八章严格草稿。
- [19_model_comparison_straightening_and_operadic_localization.md](19_model_comparison_straightening_and_operadic_localization.md)：第十九章严格草稿。
- [20_factorization_algebras_fukaya_categories_and_geometry.md](20_factorization_algebras_fukaya_categories_and_geometry.md)：第二十章严格草稿。
- [21_research_frontier_2026.md](21_research_frontier_2026.md)：近期研究边界索引。
- [A_set_theory_universes_finite_sets_and_symmetric_groups.md](A_set_theory_universes_finite_sets_and_symmetric_groups.md)：附录 A 严格草稿。
- [B_trees_partitions_substitution_and_coinvariants.md](B_trees_partitions_substitution_and_coinvariants.md)：附录 B 严格草稿。
- [C_model_categories_and_quillen_adjunctions.md](C_model_categories_and_quillen_adjunctions.md)：附录 C 严格草稿。
- [D_source_theorem_index.md](D_source_theorem_index.md)：附录 D，外部输入定理索引和引用包账本。
- [E_signs_suspensions_and_graded_conventions.md](E_signs_suspensions_and_graded_conventions.md)：附录 E 严格草稿。
- [F_classical_operads_and_checked_examples.md](F_classical_operads_and_checked_examples.md)：附录 F 严格草稿。
- [G_model_structure_hypotheses_and_rectification.md](G_model_structure_hypotheses_and_rectification.md)：附录 G 严格草稿。
- [H_tree_conventions_and_free_operad_quotients.md](H_tree_conventions_and_free_operad_quotients.md)：附录 H 严格草稿。
- [I_koszul_bar_cobar_strict_conventions.md](I_koszul_bar_cobar_strict_conventions.md)：附录 I 严格草稿。
- [J_homotopy_transfer_tree_formulas.md](J_homotopy_transfer_tree_formulas.md)：附录 J 严格草稿。
- [K_colored_operads_modules_and_enrichment.md](K_colored_operads_modules_and_enrichment.md)：附录 K 严格草稿。
- [L_infinity_algebras_and_en_operad_conventions.md](L_infinity_algebras_and_en_operad_conventions.md)：附录 L 严格草稿。
- [M_dendroidal_lurie_and_model_comparison_map.md](M_dendroidal_lurie_and_model_comparison_map.md)：附录 M 严格草稿。
- [N_factorization_homology_examples_and_geometry.md](N_factorization_homology_examples_and_geometry.md)：附录 N 严格草稿。
- [O_failure_modes_counterexamples_and_boundary_cases.md](O_failure_modes_counterexamples_and_boundary_cases.md)：附录 O 严格草稿。
- [P_low_arity_checks_and_worked_computations.md](P_low_arity_checks_and_worked_computations.md)：附录 P 严格草稿。
- [Q_koszul_complexes_and_bar_cobar_examples.md](Q_koszul_complexes_and_bar_cobar_examples.md)：附录 Q 严格草稿。
- [R_model_category_case_studies.md](R_model_category_case_studies.md)：附录 R 严格草稿。
- [S_homotopy_transfer_worked_examples.md](S_homotopy_transfer_worked_examples.md)：附录 S 严格草稿。
- [T_dendroidal_horns_segal_and_normality_examples.md](T_dendroidal_horns_segal_and_normality_examples.md)：附录 T 严格草稿。
- [U_props_properads_graphical_calculus_examples.md](U_props_properads_graphical_calculus_examples.md)：附录 U 严格草稿。
- [V_stratified_and_boundary_factorization_examples.md](V_stratified_and_boundary_factorization_examples.md)：附录 V 严格草稿。
- [W_sign_convention_crosswalk.md](W_sign_convention_crosswalk.md)：附录 W 严格草稿。
- [X_concrete_algebraic_examples_and_counterexamples.md](X_concrete_algebraic_examples_and_counterexamples.md)：附录 X 严格草稿。
- [Y_infinity_operadic_homology_and_koszul_frontier.md](Y_infinity_operadic_homology_and_koszul_frontier.md)：附录 Y 严格草稿。
- [Z_operadic_categories_relative_rezk_and_fukaya_frontier.md](Z_operadic_categories_relative_rezk_and_fukaya_frontier.md)：附录 Z 严格草稿。
