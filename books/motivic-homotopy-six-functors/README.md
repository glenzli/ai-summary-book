# Motivic Homotopy and Six Functors：从 `\mathbb A^1`-局部化到 Grothendieck 六操作

作者：Dr. Stochastic Parrot

本书从一个具体问题展开：怎样把代数几何中的局部粘合、仿射直线同伦、Tate 悬挂和
紧支撑推前放进同一套稳定同伦语言。正文从光滑站点、Nisnevich descent 和空间值层
开始，经 `\mathbb A^1`-局部化与 `T`/`\mathbb P^1`-稳定化构造
`\mathbf H(S)` 和 `\mathbf{SH}(S)`；随后建立 Grothendieck 六操作、纯性、对偶、
基本类、转移、范数、motivic cohomology 与 realization。最后五章讨论 equivariant、
stacky、log、perfect 和 analytic 扩展，并把已知比较与开放问题严格分开。

预备知识包括基本概形论、Grothendieck 拓扑、范畴论与经典稳定同伦论。读者若尚未
熟悉 presentable 或 stable infinity-范畴，可以在阅读前三章时并行查阅附录 A、C、F；
六操作的 mate calculus 可随第四、八章查阅附录 D。每条深定理均标为外部输入，
正文只证明由这些输入形式推出的结论。

## 阅读约定

本书的完整写作与校订约束见 [SKILL.md](SKILL.md)。阅读正文时应留意：

- 定义先于直觉，非平凡命题带证明或明确的外部输入标记。
- 全书固定集合论宇宙、小骨架、基概形类别和默认有限性假设。
- `\mathbf H(S)`、`\mathbf H_*(S)`、`\mathbf{SH}(S)`、`\mathbf{DM}(S)`、motivic sheaves 和 bivariant theories 不得混写。
- 六操作存在性、smooth/absolute purity、Atiyah duality、framed
  recognition、norms、stacky extension 和 analytic extension 必须作为
  可追溯外部输入处理；由这些输入形式推出的 ambidexterity 仍须给出书内
  伴随证明。
- 资料源必须能在 [SOURCES.md](SOURCES.md) 和 [THEOREM_LEDGER.md](THEOREM_LEDGER.md) 中追溯。

符号约定见 [NOTATION.md](NOTATION.md)，外部输入见
[THEOREM_LEDGER.md](THEOREM_LEDGER.md) 与 [SOURCES.md](SOURCES.md)，近期文献的
版本边界见 [FRONTIER_SOURCE_AUDIT_2026_07_08.md](FRONTIER_SOURCE_AUDIT_2026_07_08.md)。

## 目录

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

### 第三部分：Motivic cohomology、motives 与计算工具

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
25. [第二十四章：研究边界、比较问题与开放方向](24_research_frontier_2026_open_problems_and_source_boundaries.md)

### 附录

- [附录 A：集合论宇宙、小骨架、presentability 与 accessible localization](A_universes_presentability_and_localization.md)
- [附录 B：Grothendieck topologies、points、Nisnevich squares 与 cd-structures](B_grothendieck_topologies_nisnevich_squares_and_cd_structures.md)
- [附录 C：Pointed presentable categories、stabilization 与 symmetric monoidal spectra](C_pointed_stabilization_and_symmetric_monoidal_spectra.md)
- [附录 D：六操作相干图、mate calculus 与 Beck-Chevalley 记号](D_mate_calculus_beck_chevalley_and_coherence.md)
- [附录 E：代数几何最小背景：smooth、etale、proper、closed/open immersion](E_algebraic_geometry_background_for_six_functors.md)
- [附录 F：三角范畴和稳定 infinity-范畴翻译表](F_stable_infinity_vs_triangulated_translation.md)
- [附录 G：资料源定理索引与 locator ledger](G_source_theorem_index.md)
- [附录 H：低阶例子、对象等价与基本计算](H_worked_examples_and_basic_computations.md)

## 阅读辅助与资料

- [NOTATION.md](NOTATION.md)：全书符号、宇宙与小性约定。
- [INDEX.md](INDEX.md)：主题索引。
- [EXERCISE_SOLUTIONS.md](EXERCISE_SOLUTIONS.md)：习题解答与证明提示。
- [SOURCES.md](SOURCES.md)：主要一手资料及其用途。
- [THEOREM_LEDGER.md](THEOREM_LEDGER.md)：内部证明、外部输入与研究边界。
- [REFERENCE_LOCATOR_LEDGER.md](REFERENCE_LOCATOR_LEDGER.md)：外部定理的版本和精确定位。
- [TYPESETTING_AND_NUMBERING.md](TYPESETTING_AND_NUMBERING.md)：公式、编号与交叉引用规范。
- [MATH_REVIEW.md](MATH_REVIEW.md)：数学审校记录；不属于连续阅读正文。
