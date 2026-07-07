# Homological Mirror Symmetry：Fukaya 范畴、导出几何与镜像等价

作者：Dr. Stochastic Parrot  
状态：完整在线教材内容本体收口版，尚未出版级校对  
核查日期：2026-07-08  
主资料源：Kontsevich, Seidel, Fukaya-Oh-Ohta-Ono, Auroux, Keller, Lefevre-Hasegawa, Huybrechts, Bondal-Orlov, Ganatra-Pardon-Shende, Nadler, Abouzaid-Auroux, Lekili-Ueda

本书目标是写成一部严格的中文 Homological Mirror Symmetry 教材，而不是镜像对称导览。正文从 dg category 与 $A_\infty$-category 的增强语言开始，逐步建立 B-side 的导出范畴、A-side 的 Fukaya 范畴、wrapped/partially wrapped 技术、HMS 断言的精确定式、标准例子、生成准则、microlocal/sheaf-theoretic 模型，以及 2025-2026 研究边界。

## 写作约束

本书写作约束见 [SKILL.md](SKILL.md)。后续扩写必须遵守：

- 定义先于直觉，非平凡命题带证明或明确的外部输入标记。
- 全书固定集合论宇宙、基域、分次和增强范畴口径。
- 不把 triangulated equivalence 当作最终 HMS 形式；必须说明 dg、$A_\infty$、Morita 或 stable $\infty$ enhancement。
- A-side 必须说明 brane data、orientation、grading、transversality、compactness、Novikov 系数或 exact/monotone 假设。
- B-side 必须说明 $\operatorname{Perf}$、$\mathrm D^b\operatorname{Coh}$、Fourier-Mukai kernel、matrix factorization 或 singularity category 的几何假设。
- 资料源必须能在 [SOURCES.md](SOURCES.md) 中追溯。

符号约定见 [NOTATION.md](NOTATION.md)，数学审查记录见 [MATH_REVIEW.md](MATH_REVIEW.md)，闭合矩阵见 [PUBLICATION_CLOSURE_MATRIX.md](PUBLICATION_CLOSURE_MATRIX.md)，近期资料核查见 [FRONTIER_SOURCE_AUDIT_2026_07_08.md](FRONTIER_SOURCE_AUDIT_2026_07_08.md)。

## 建议总目录

### 第一部分：增强范畴语言

1. [序章：范围、严格性标准和 HMS 的数学形态](00_preface_and_scope.md)
2. [第一章：dg 范畴、$A_\infty$ 范畴与预三角化](01_dg_and_a_infinity_categories.md)
3. [第二章：导出范畴、完美复形与 B-side 增强](02_derived_categories_and_b_side_enhancements.md)

### 第二部分：A-side 基础

4. [第三章：辛流形、Lagrangian brane 与 exact Floer 口径](03_symplectic_lagrangian_and_floer_foundations.md)
5. [第四章：holomorphic polygon、$A_\infty$ 结构与 Fukaya category](04_holomorphic_polygons_and_fukaya_categories.md)
6. [第五章：obstruction、bounding cochains、Novikov 系数与 curved $A_\infty$ 结构](05_obstruction_bounding_cochains_and_novikov_coefficients.md)
7. [第六章：Liouville manifolds、sectors 与 wrapped Fukaya categories](06_liouville_sectors_and_wrapped_fukaya_categories.md)
8. [第七章：stops、partially wrapped categories 与 localization](07_stops_partially_wrapped_categories_and_localization.md)

### 第三部分：HMS 断言与标准模型

9. [第八章：HMS 断言、增强等价与必要不变量](08_hms_statement_enhancements_and_invariants.md)
10. [第九章：椭圆曲线、复环面与 SYZ 的第一模型](09_elliptic_curves_complex_tori_and_syz_first_model.md)
11. [第十章：toric Fano、Landau-Ginzburg potential 与 Jacobian ring](10_toric_fano_landau_ginzburg_potentials_and_jacobian_rings.md)
12. [第十一章：Fukaya-Seidel category 与 Picard-Lefschetz theory](11_fukaya_seidel_categories_and_picard_lefschetz_theory.md)
13. [第十二章：K3 曲面、四次曲面与 Calabi-Yau hypersurfaces](12_k3_quartics_and_calabi_yau_hypersurfaces.md)
14. [第十三章：pairs of pants、tropical degeneration 与 hypersurfaces in $(\mathbb C^\ast)^n$](13_pairs_of_pants_tropical_degeneration_and_hypersurfaces.md)

### 第四部分：生成、局部化与 sheaf 模型

15. [第十四章：split-generation、open-closed map 与 Abouzaid criterion](14_split_generation_open_closed_and_abouzaid_criterion.md)
16. [第十五章：wrapped Fukaya categories 的 sectorial descent](15_sectorial_descent_for_wrapped_fukaya_categories.md)
17. [第十六章：Nadler-Zaslow、microlocal sheaves 与 cotangent bundles](16_nadler_zaslow_microlocal_sheaves_and_cotangent_bundles.md)
18. [第十七章：stop removal、Viterbo functor 与 functorial HMS](17_stop_removal_viterbo_functors_and_functorial_hms.md)
19. [第十八章：Hochschild invariants、closed-open maps 与 categorical enumerative checks](18_hochschild_closed_open_and_categorical_enumerative_checks.md)

### 第五部分：研究边界

20. [第十九章：Rabinowitz Fukaya categories、singularities 与 matrix factorizations](19_rabinowitz_fukaya_singularities_and_matrix_factorizations.md)
21. [第二十章：functorial HMS、wall-crossing、BPS categories 与 2026 研究边界](20_functorial_wall_crossing_bps_and_2026_research_boundary.md)

### 附录

- [附录 A：集合论宇宙、基域、分次与符号约定](A_universes_coefficients_gradings_and_signs.md)
- [附录 B：$A_\infty$ 符号、bar construction 与 Yoneda embedding](B_a_infinity_signs_bar_construction_and_yoneda.md)
- [附录 C：dg quotient、Morita localization 与 perfect modules](C_dg_quotients_morita_localization_and_perfect_modules.md)
- [附录 D：Fourier-Mukai transforms 与导出代数几何接口](D_fourier_mukai_transforms_and_derived_geometry_interface.md)
- [附录 E：Floer analytic inputs、compactness、orientation 与 gluing](E_floer_analytic_inputs_compactness_orientation_and_gluing.md)
- [附录 F：Liouville sectors、stops 与 wrapped examples](F_liouville_stops_and_wrapped_examples.md)
- [附录 G：HMS theorem locator 与外部输入索引](G_hms_theorem_locator_and_external_input_index.md)
- [附录 H：标准例子的可计算模型与反例边界](H_computable_models_counterexamples_and_boundary_cases.md)
- [附录 I：低阶 $A_\infty$、curvature 与 Maurer-Cartan 计算](I_low_arity_a_infinity_and_curvature_calculations.md)
- [附录 J：椭圆曲线、toric Fano 与 Fukaya-Seidel 计算模型](J_elliptic_toric_and_fukaya_seidel_worked_models.md)
- [附录 K：生成、descent 与 localization 证明模板](K_generation_descent_and_localization_templates.md)

## 当前内容索引

- [SKILL.md](SKILL.md)：本教材的写作约束。
- [NOTATION.md](NOTATION.md)：全书符号和小性约定。
- [SOURCES.md](SOURCES.md)：主要资料源清单和近期研究入口。
- [MATH_REVIEW.md](MATH_REVIEW.md)：初始数学审查账本。
- [FRONTIER_SOURCE_AUDIT_2026_07_08.md](FRONTIER_SOURCE_AUDIT_2026_07_08.md)：近期文献版本核查记录。
- [THEOREM_LEDGER.md](THEOREM_LEDGER.md)：内部命题、外部输入和研究边界账本。
- [DEPENDENCY_GRAPH.md](DEPENDENCY_GRAPH.md)：定义与证明依赖图。
- [PUBLICATION_CLOSURE_MATRIX.md](PUBLICATION_CLOSURE_MATRIX.md)：正式教材主体草稿闭合矩阵。
- [INTERNAL_CONTENT_CLOSURE_AUDIT.md](INTERNAL_CONTENT_CLOSURE_AUDIT.md)：教材本体内容收口审计。
- [SOLUTIONS.md](SOLUTIONS.md)：主体章节练习解答与提示。
- [00_preface_and_scope.md](00_preface_and_scope.md)：序章。
- [01_dg_and_a_infinity_categories.md](01_dg_and_a_infinity_categories.md)：第一章严格草稿。
- [02_derived_categories_and_b_side_enhancements.md](02_derived_categories_and_b_side_enhancements.md)：第二章严格草稿。
- [03_symplectic_lagrangian_and_floer_foundations.md](03_symplectic_lagrangian_and_floer_foundations.md)：第三章严格草稿。
- [04_holomorphic_polygons_and_fukaya_categories.md](04_holomorphic_polygons_and_fukaya_categories.md)：第四章严格草稿。
- [05_obstruction_bounding_cochains_and_novikov_coefficients.md](05_obstruction_bounding_cochains_and_novikov_coefficients.md)：第五章严格草稿。
- [06_liouville_sectors_and_wrapped_fukaya_categories.md](06_liouville_sectors_and_wrapped_fukaya_categories.md)：第六章严格草稿。
- [07_stops_partially_wrapped_categories_and_localization.md](07_stops_partially_wrapped_categories_and_localization.md)：第七章严格草稿。
- [08_hms_statement_enhancements_and_invariants.md](08_hms_statement_enhancements_and_invariants.md)：HMS 断言与不变量检查的严格草稿。
- [09_elliptic_curves_complex_tori_and_syz_first_model.md](09_elliptic_curves_complex_tori_and_syz_first_model.md)：第九章严格草稿。
- [10_toric_fano_landau_ginzburg_potentials_and_jacobian_rings.md](10_toric_fano_landau_ginzburg_potentials_and_jacobian_rings.md)：第十章严格草稿。
- [11_fukaya_seidel_categories_and_picard_lefschetz_theory.md](11_fukaya_seidel_categories_and_picard_lefschetz_theory.md)：第十一章严格草稿。
- [12_k3_quartics_and_calabi_yau_hypersurfaces.md](12_k3_quartics_and_calabi_yau_hypersurfaces.md)：第十二章严格草稿。
- [13_pairs_of_pants_tropical_degeneration_and_hypersurfaces.md](13_pairs_of_pants_tropical_degeneration_and_hypersurfaces.md)：第十三章严格草稿。
- [14_split_generation_open_closed_and_abouzaid_criterion.md](14_split_generation_open_closed_and_abouzaid_criterion.md)：第十四章严格草稿。
- [15_sectorial_descent_for_wrapped_fukaya_categories.md](15_sectorial_descent_for_wrapped_fukaya_categories.md)：第十五章严格草稿。
- [16_nadler_zaslow_microlocal_sheaves_and_cotangent_bundles.md](16_nadler_zaslow_microlocal_sheaves_and_cotangent_bundles.md)：第十六章严格草稿。
- [17_stop_removal_viterbo_functors_and_functorial_hms.md](17_stop_removal_viterbo_functors_and_functorial_hms.md)：第十七章严格草稿。
- [18_hochschild_closed_open_and_categorical_enumerative_checks.md](18_hochschild_closed_open_and_categorical_enumerative_checks.md)：第十八章严格草稿。
- [19_rabinowitz_fukaya_singularities_and_matrix_factorizations.md](19_rabinowitz_fukaya_singularities_and_matrix_factorizations.md)：第十九章严格草稿。
- [20_functorial_wall_crossing_bps_and_2026_research_boundary.md](20_functorial_wall_crossing_bps_and_2026_research_boundary.md)：第二十章严格草稿。
- [A_universes_coefficients_gradings_and_signs.md](A_universes_coefficients_gradings_and_signs.md)：附录 A。
- [B_a_infinity_signs_bar_construction_and_yoneda.md](B_a_infinity_signs_bar_construction_and_yoneda.md)：附录 B。
- [C_dg_quotients_morita_localization_and_perfect_modules.md](C_dg_quotients_morita_localization_and_perfect_modules.md)：附录 C。
- [D_fourier_mukai_transforms_and_derived_geometry_interface.md](D_fourier_mukai_transforms_and_derived_geometry_interface.md)：附录 D。
- [E_floer_analytic_inputs_compactness_orientation_and_gluing.md](E_floer_analytic_inputs_compactness_orientation_and_gluing.md)：附录 E。
- [F_liouville_stops_and_wrapped_examples.md](F_liouville_stops_and_wrapped_examples.md)：附录 F。
- [G_hms_theorem_locator_and_external_input_index.md](G_hms_theorem_locator_and_external_input_index.md)：附录 G。
- [H_computable_models_counterexamples_and_boundary_cases.md](H_computable_models_counterexamples_and_boundary_cases.md)：附录 H。
- [I_low_arity_a_infinity_and_curvature_calculations.md](I_low_arity_a_infinity_and_curvature_calculations.md)：附录 I。
- [J_elliptic_toric_and_fukaya_seidel_worked_models.md](J_elliptic_toric_and_fukaya_seidel_worked_models.md)：附录 J。
- [K_generation_descent_and_localization_templates.md](K_generation_descent_and_localization_templates.md)：附录 K。
