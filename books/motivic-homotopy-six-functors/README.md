# Motivic Homotopy and Six Functors：从 A1-局部化到 Grothendieck 六操作

作者：Dr. Stochastic Parrot
状态：完整教材可读版；核心基础、六操作及第 09-18 章 P0 主线完成 OET 复核，扩展/realization locator 和终校继续推进
核查日期：2026-07-11
主资料源：Lurie, Morel-Voevodsky, Robalo, Ayoub, Cisinski-Deglise, Hoyois, Drew-Gallauer, Bachmann-Hoyois, Elmanto-Hoyois-Khan-Sosnilo-Yakerson, Deglise-Jin-Khan

本书目标是写成一部严格的中文 Motivic Homotopy and Six Functors 教材，而不是主题导览。正文从光滑站点、Nisnevich descent、space-valued sheaves 和 `\mathbb A^1`-局部化开始，随后进入 `T`/`\mathbb P^1` 稳定化、`\mathbf{SH}(S)`、Grothendieck 六操作、纯性、基本类、转移、范数、framed correspondences、motivic cohomology、realization functors、stacky/equivariant/log/perfect/analytic 扩展及 2025-2026 研究边界。

按当前标准，本书已经达到“可完整阅读和教学使用”的教材闭合：各章包含定义、主要命题、证明或外部输入标记、边界说明和练习。宇宙/局部化、`T`-稳定化、六操作方差、base change/projection formula、purity、compactness、三角翻译，以及第 09-18 章 `HZ/DM/KGL/MGL/slice/transfers/framed/Gysin/norm/MW` 主线均完成 OET 级修订和 P0 locator 闭合。尚未完成的是第 19-23 章扩展/realization 的剩余 P0 locator、全书自动交叉引用和长篇习题详解。

## 写作约束

本书写作约束见 [SKILL.md](SKILL.md)。后续扩写必须遵守：

- 定义先于直觉，非平凡命题带证明或明确的外部输入标记。
- 全书固定集合论宇宙、小骨架、基概形类别和默认有限性假设。
- `\mathbf H(S)`、`\mathbf H_*(S)`、`\mathbf{SH}(S)`、`\mathbf{DM}(S)`、motivic sheaves 和 bivariant theories 不得混写。
- 六操作存在性、smooth/absolute purity、Atiyah duality、framed
  recognition、norms、stacky extension 和 analytic extension 必须作为
  可追溯外部输入处理；由这些输入形式推出的 ambidexterity 仍须给出书内
  伴随证明。
- 资料源必须能在 [SOURCES.md](SOURCES.md) 和 [THEOREM_LEDGER.md](THEOREM_LEDGER.md) 中追溯。

符号约定见 [NOTATION.md](NOTATION.md)，数学审查记录见 [MATH_REVIEW.md](MATH_REVIEW.md)，教材内容闭合审计见 [TEACHING_CLOSURE_AUDIT.md](TEACHING_CLOSURE_AUDIT.md)，内部闭合矩阵见 [INTERNAL_CLOSURE_MATRIX.md](INTERNAL_CLOSURE_MATRIX.md)，近期文献边界见 [FRONTIER_SOURCE_AUDIT_2026_07_08.md](FRONTIER_SOURCE_AUDIT_2026_07_08.md)。

## 建议总目录

### 第一部分：Motivic spaces 与稳定化

1. [序章：范围、严格性标准和资料源](00_preface_and_scope.md)
2. [第一章：基概形、光滑站点与 Nisnevich descent](01_base_schemes_smooth_sites_and_nisnevich_descent.md)
3. [第二章：A1-局部化与 motivic spaces](02_a1_localization_and_motivic_spaces.md)
4. [第三章：Tate sphere、P1-稳定化与 SH(S)](03_tate_sphere_p1_stabilization_and_sh.md)

### 第二部分：六操作形式主义

5. [第四章：六操作的抽象形式主义](04_six_functor_formalism.md)
6. [第五章：Motivic 六操作、proper compatibility 与 localization](05_motivic_six_operations_proper_and_localization.md)
7. [第六章：Homotopy purity、Thom spaces 与 purity transformations](06_homotopy_purity_thom_spaces_and_purity_transformations.md)
8. [第七章：Smooth/proper ambidexterity、duality 与 trace](07_smooth_proper_ambidexterity_duality_and_trace.md)
9. [第八章：Base change、projection formula 与 Beck-Chevalley 相干](08_base_change_projection_formula_and_beck_chevalley.md)

### 第三部分：Motivic cohomology、motives 与计算接口

10. [第九章：Eilenberg-Mac Lane spectra、motivic cohomology 与 `H\mathbb Z`](09_eilenberg_mac_lane_spectra_and_motivic_cohomology.md)
11. [第十章：Voevodsky motives、Cisinski-Deglise motives 与 `H\mathbb Z`-modules](10_motives_hz_modules_and_dm.md)
12. [第十一章：Algebraic K-theory、homotopy K-theory 与 cdh descent](11_kgl_kh_and_cdh_descent.md)
13. [第十二章：Algebraic cobordism、orientations 与 formal group laws](12_mgl_orientations_and_formal_group_laws.md)
14. [第十三章：Slice filtration、effective categories 与 cellular methods](13_slice_filtration_effective_categories_and_cellular_methods.md)

### 第四部分：转移、范数与 framed homotopy

15. [第十四章：Finite correspondences、presheaves with transfers 与 motivic complexes](14_finite_correspondences_transfers_and_motivic_complexes.md)
16. [第十五章：Framed correspondences 与 motivic infinite loop spaces](15_framed_correspondences_and_motivic_infinite_loop_spaces.md)
17. [第十六章：Fundamental classes、Gysin maps 与 bivariant theory](16_fundamental_classes_gysin_maps_and_bivariant_theory.md)
18. [第十七章：Norm functors、normed spectra 与 multiplicative transfers](17_norm_functors_normed_spectra_and_multiplicative_transfers.md)
19. [第十八章：Milnor-Witt refinements、quadratic refinements 与 enumerative applications](18_milnor_witt_quadratic_refinements_and_enumerative_applications.md)

### 第五部分：扩展、比较与前沿

20. [第十九章：Equivariant motivic homotopy 与 quotient stacks](19_equivariant_motivic_homotopy_and_quotient_stacks.md)
21. [第二十章：Algebraic stacks 上的 motivic homotopy 与六操作](20_motivic_homotopy_of_algebraic_stacks_and_stacky_six_operations.md)
22. [第二十一章：Log schemes、perfect schemes 与 universal homeomorphisms](21_log_perfect_schemes_and_universal_homeomorphisms.md)
23. [第二十二章：Betti、etale、real etale 与 analytic realization](22_realization_functors_betti_etale_real_etale_and_analytic.md)
24. [第二十三章：Universal six-functor formalisms 与 pullback formalisms](23_universal_six_functor_and_pullback_formalisms.md)
25. [第二十四章：2025-2026 研究边界、开放问题与资料源定位](24_research_frontier_2026_open_problems_and_source_boundaries.md)

### 附录

- [附录 A：集合论宇宙、小骨架、presentability 与 accessible localization](A_universes_presentability_and_localization.md)
- [附录 B：Grothendieck topologies、points、Nisnevich squares 与 cd-structures](B_grothendieck_topologies_nisnevich_squares_and_cd_structures.md)
- [附录 C：Pointed presentable categories、stabilization 与 symmetric monoidal spectra](C_pointed_stabilization_and_symmetric_monoidal_spectra.md)
- [附录 D：六操作相干图、mate calculus 与 Beck-Chevalley 记号](D_mate_calculus_beck_chevalley_and_coherence.md)
- [附录 E：代数几何最小背景：smooth、etale、proper、closed/open immersion](E_algebraic_geometry_background_for_six_functors.md)
- [附录 F：三角范畴和稳定 infinity-范畴翻译表](F_stable_infinity_vs_triangulated_translation.md)
- [附录 G：资料源定理索引与 locator ledger](G_source_theorem_index.md)
- [附录 H：低阶例子、对象等价与基本计算](H_worked_examples_and_basic_computations.md)

## 当前内容索引

- [SKILL.md](SKILL.md)：本教材的写作约束。
- [NOTATION.md](NOTATION.md)：全书符号和小性约定。
- [SOURCES.md](SOURCES.md)：主要资料源清单和近期研究入口。
- [THEOREM_LEDGER.md](THEOREM_LEDGER.md)：内部证明、外部输入和研究边界账本。
- [REFERENCE_LOCATOR_LEDGER.md](REFERENCE_LOCATOR_LEDGER.md)：外部输入精确定位账本。
- [P0_REFERENCE_LOCATORS_BATCH_1.md](P0_REFERENCE_LOCATORS_BATCH_1.md)：framed、norm、fundamental class 与 universal formalism 定位。
- [P0_REFERENCE_LOCATORS_BATCH_2.md](P0_REFERENCE_LOCATORS_BATCH_2.md)：基础范畴论、稳定化、六操作、purity 与 triangulated shadow 定位。
- [P0_REFERENCE_LOCATORS_BATCH_3.md](P0_REFERENCE_LOCATORS_BATCH_3.md)：第九至第十八章 cohomology、motives、谱、transfers、norms 与 Milnor-Witt 主线定位。
- [MATH_REVIEW.md](MATH_REVIEW.md)：当前严格性审查与风险记录。
- [TEACHING_CLOSURE_AUDIT.md](TEACHING_CLOSURE_AUDIT.md)：教材内容闭合审计，判断内容、证明和引用是否足以支撑完整教学使用。
- [TYPESETTING_AND_NUMBERING.md](TYPESETTING_AND_NUMBERING.md)：统一编号、排版、交叉引用和证明格式规范。
- [INDEX.md](INDEX.md)：全书主题索引。
- [EXERCISE_SOLUTIONS.md](EXERCISE_SOLUTIONS.md)：全书习题解答要点。
- [INTERNAL_CLOSURE_MATRIX.md](INTERNAL_CLOSURE_MATRIX.md)：正式教材内部闭合状态矩阵。
- [CHAPTER_DENSITY_AUDIT.md](CHAPTER_DENSITY_AUDIT.md)：逐章密度审计，检查是否仍为大纲态。
- [FRONTIER_SOURCE_AUDIT_2026_07_08.md](FRONTIER_SOURCE_AUDIT_2026_07_08.md)：2025-2026 研究边界核查。
- [00_preface_and_scope.md](00_preface_and_scope.md)：序章。
- [01_base_schemes_smooth_sites_and_nisnevich_descent.md](01_base_schemes_smooth_sites_and_nisnevich_descent.md)：第一章严格草稿。
- [02_a1_localization_and_motivic_spaces.md](02_a1_localization_and_motivic_spaces.md)：第二章严格草稿。
- [03_tate_sphere_p1_stabilization_and_sh.md](03_tate_sphere_p1_stabilization_and_sh.md)：第三章严格草稿。
- [04_six_functor_formalism.md](04_six_functor_formalism.md)：第四章严格草稿。
- [05_motivic_six_operations_proper_and_localization.md](05_motivic_six_operations_proper_and_localization.md)：第五章严格草稿。
- [06_homotopy_purity_thom_spaces_and_purity_transformations.md](06_homotopy_purity_thom_spaces_and_purity_transformations.md)：第六章严格草稿。
- [07_smooth_proper_ambidexterity_duality_and_trace.md](07_smooth_proper_ambidexterity_duality_and_trace.md)：第七章严格草稿。
- [08_base_change_projection_formula_and_beck_chevalley.md](08_base_change_projection_formula_and_beck_chevalley.md)：第八章严格草稿。
- [09_eilenberg_mac_lane_spectra_and_motivic_cohomology.md](09_eilenberg_mac_lane_spectra_and_motivic_cohomology.md)：第九章严格草稿。
- [10_motives_hz_modules_and_dm.md](10_motives_hz_modules_and_dm.md)：第十章严格草稿。
- [11_kgl_kh_and_cdh_descent.md](11_kgl_kh_and_cdh_descent.md)：第十一章严格草稿。
- [12_mgl_orientations_and_formal_group_laws.md](12_mgl_orientations_and_formal_group_laws.md)：第十二章严格草稿。
- [13_slice_filtration_effective_categories_and_cellular_methods.md](13_slice_filtration_effective_categories_and_cellular_methods.md)：第十三章严格草稿。
- [14_finite_correspondences_transfers_and_motivic_complexes.md](14_finite_correspondences_transfers_and_motivic_complexes.md)：第十四章严格草稿。
- [15_framed_correspondences_and_motivic_infinite_loop_spaces.md](15_framed_correspondences_and_motivic_infinite_loop_spaces.md)：第十五章严格草稿。
- [16_fundamental_classes_gysin_maps_and_bivariant_theory.md](16_fundamental_classes_gysin_maps_and_bivariant_theory.md)：第十六章严格草稿。
- [17_norm_functors_normed_spectra_and_multiplicative_transfers.md](17_norm_functors_normed_spectra_and_multiplicative_transfers.md)：第十七章严格草稿。
- [18_milnor_witt_quadratic_refinements_and_enumerative_applications.md](18_milnor_witt_quadratic_refinements_and_enumerative_applications.md)：第十八章严格草稿。
- [19_equivariant_motivic_homotopy_and_quotient_stacks.md](19_equivariant_motivic_homotopy_and_quotient_stacks.md)：第十九章严格草稿。
- [20_motivic_homotopy_of_algebraic_stacks_and_stacky_six_operations.md](20_motivic_homotopy_of_algebraic_stacks_and_stacky_six_operations.md)：第二十章严格草稿。
- [21_log_perfect_schemes_and_universal_homeomorphisms.md](21_log_perfect_schemes_and_universal_homeomorphisms.md)：第二十一章严格草稿。
- [22_realization_functors_betti_etale_real_etale_and_analytic.md](22_realization_functors_betti_etale_real_etale_and_analytic.md)：第二十二章严格草稿。
- [23_universal_six_functor_and_pullback_formalisms.md](23_universal_six_functor_and_pullback_formalisms.md)：第二十三章严格草稿。
- [24_research_frontier_2026_open_problems_and_source_boundaries.md](24_research_frontier_2026_open_problems_and_source_boundaries.md)：第二十四章严格草稿。
- [A_universes_presentability_and_localization.md](A_universes_presentability_and_localization.md)：附录 A 严格草稿。
- [B_grothendieck_topologies_nisnevich_squares_and_cd_structures.md](B_grothendieck_topologies_nisnevich_squares_and_cd_structures.md)：附录 B 严格草稿。
- [C_pointed_stabilization_and_symmetric_monoidal_spectra.md](C_pointed_stabilization_and_symmetric_monoidal_spectra.md)：附录 C 严格草稿。
- [D_mate_calculus_beck_chevalley_and_coherence.md](D_mate_calculus_beck_chevalley_and_coherence.md)：附录 D 严格草稿。
- [E_algebraic_geometry_background_for_six_functors.md](E_algebraic_geometry_background_for_six_functors.md)：附录 E 严格草稿。
- [F_stable_infinity_vs_triangulated_translation.md](F_stable_infinity_vs_triangulated_translation.md)：附录 F 严格草稿。
- [G_source_theorem_index.md](G_source_theorem_index.md)：附录 G 严格草稿。
- [H_worked_examples_and_basic_computations.md](H_worked_examples_and_basic_computations.md)：附录 H 严格草稿。
