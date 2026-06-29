# Operad Theory：从代数运算到 infinity-operad

作者：Dr. Stochastic Parrot  
状态：主体第一轮起草完成，第二轮严格化审校中  
核查日期：2026-06-29  
主资料源：May, Boardman-Vogt, Markl-Shnider-Stasheff, Loday-Vallette, Fresse, Moerdijk-Weiss, Cisinski-Moerdijk, Lurie, Hoffbeck-Moerdijk

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

## 当前已起草内容

- [SKILL.md](SKILL.md)：本教材的写作约束。
- [NOTATION.md](NOTATION.md)：全书符号和小性约定。
- [SOURCES.md](SOURCES.md)：主要资料源清单和近期研究入口。
- [MATH_REVIEW.md](MATH_REVIEW.md)：审查清单和当前风险。
- [SECOND_PASS_STRICTIFICATION_PLAN.md](SECOND_PASS_STRICTIFICATION_PLAN.md)：第二轮严格化路线图。
- [FRONTIER_SOURCE_AUDIT_2026_06_29.md](FRONTIER_SOURCE_AUDIT_2026_06_29.md)：近期前沿文献版本核查记录。
- [00_preface_and_scope.md](00_preface_and_scope.md)：序章。
- [01_symmetric_sequences_and_operads.md](01_symmetric_sequences_and_operads.md)：第一章初稿。
- [02_operad_algebras_free_algebras_and_monads.md](02_operad_algebras_free_algebras_and_monads.md)：第二章初稿。
- [03_nonsymmetric_operads_partial_compositions_and_trees.md](03_nonsymmetric_operads_partial_compositions_and_trees.md)：第三章初稿。
- [04_free_operads_generators_and_relations.md](04_free_operads_generators_and_relations.md)：第四章初稿。
- [05_colored_operads_multicategories_and_typed_algebras.md](05_colored_operads_multicategories_and_typed_algebras.md)：第五章初稿。
- [06_linear_operads_schur_functors_and_classical_examples.md](06_linear_operads_schur_functors_and_classical_examples.md)：第六章初稿。
- [07_props_properads_and_wheeled_variants.md](07_props_properads_and_wheeled_variants.md)：第七章初稿。
- [08_quadratic_operads_and_koszul_duality.md](08_quadratic_operads_and_koszul_duality.md)：第八章初稿。
- [09_bar_cobar_constructions_and_twisting_morphisms.md](09_bar_cobar_constructions_and_twisting_morphisms.md)：第九章初稿。
- [10_a_infinity_l_infinity_and_e_n_operads.md](10_a_infinity_l_infinity_and_e_n_operads.md)：第十章初稿。
- [11_gerstenhaber_bv_and_deligne_conjecture.md](11_gerstenhaber_bv_and_deligne_conjecture.md)：第十一章初稿。
- [12_brace_operad_and_hochschild_cochains.md](12_brace_operad_and_hochschild_cochains.md)：第十二章初稿。
- [13_homotopy_transfer_and_minimal_models.md](13_homotopy_transfer_and_minimal_models.md)：第十三章初稿。
- [14_operads_in_model_categories.md](14_operads_in_model_categories.md)：第十四章初稿。
- [15_simplicial_and_topological_operads.md](15_simplicial_and_topological_operads.md)：第十五章初稿。
- [16_dendroidal_sets_and_tree_category.md](16_dendroidal_sets_and_tree_category.md)：第十六章初稿。
- [17_dendroidal_inner_kan_and_homotopy_operads.md](17_dendroidal_inner_kan_and_homotopy_operads.md)：第十七章初稿。
- [18_lurie_infinity_operads_and_operadic_fibrations.md](18_lurie_infinity_operads_and_operadic_fibrations.md)：第十八章初稿。
- [19_model_comparison_straightening_and_operadic_localization.md](19_model_comparison_straightening_and_operadic_localization.md)：第十九章初稿。
- [20_factorization_algebras_fukaya_categories_and_geometry.md](20_factorization_algebras_fukaya_categories_and_geometry.md)：第二十章初稿。
- [21_research_frontier_2026.md](21_research_frontier_2026.md)：近期研究边界索引。
- [A_set_theory_universes_finite_sets_and_symmetric_groups.md](A_set_theory_universes_finite_sets_and_symmetric_groups.md)：附录 A 初稿。
- [B_trees_partitions_substitution_and_coinvariants.md](B_trees_partitions_substitution_and_coinvariants.md)：附录 B 初稿。
- [C_model_categories_and_quillen_adjunctions.md](C_model_categories_and_quillen_adjunctions.md)：附录 C 初稿。
- [D_source_theorem_index.md](D_source_theorem_index.md)：附录 D 初稿。
- [E_signs_suspensions_and_graded_conventions.md](E_signs_suspensions_and_graded_conventions.md)：附录 E 初稿。
- [F_classical_operads_and_checked_examples.md](F_classical_operads_and_checked_examples.md)：附录 F 初稿。
- [G_model_structure_hypotheses_and_rectification.md](G_model_structure_hypotheses_and_rectification.md)：附录 G 初稿。
- [H_tree_conventions_and_free_operad_quotients.md](H_tree_conventions_and_free_operad_quotients.md)：附录 H 初稿。
- [I_koszul_bar_cobar_strict_conventions.md](I_koszul_bar_cobar_strict_conventions.md)：附录 I 初稿。
- [J_homotopy_transfer_tree_formulas.md](J_homotopy_transfer_tree_formulas.md)：附录 J 初稿。
