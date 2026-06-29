# Langlands 纲领：从 `GL(1)` 到几何 Langlands

作者：Dr. Stochastic Parrot  
状态：严格教材草稿，证明补强版  
主资料源：Tate, Weil, Langlands, Gelbart, Bump, Goldfeld-Hundley, Jacquet-Langlands, Godement-Jacquet, Arthur, Milne, Serre, Silverman, Diamond-Shurman, Cornell-Silverman-Stevens, Bushnell-Henniart, Frenkel, Gaitsgory

本书目标是写成一部数学化、专业化、成体系的 Langlands 纲领教材，而不是导览文章。正文从整体域、局部域、adeles 和 Tate thesis 开始，逐步进入类域论、`GL(1)` Langlands、模形式和椭圆曲线、`GL(n)` 自守表示、Galois 表示、L 群、函子性和几何 Langlands。费马大定理的证明作为单独应用章处理。

## 写作约束

本书写作约束见 [SKILL.md](SKILL.md)。后续扩写必须遵守：

- 定义先于直觉，命题带证明或明确的外部输入标记。
- 全书固定局部/整体域、赋值、Haar 测度和表示论符号。
- 所有 L 函数必须说明局部因子、Euler 乘积区域、解析延拓和函数方程来源。
- Langlands 对应必须写成 Galois/Weil 参数、自守表示和 L 群同态之间的结构性陈述。
- 资料源必须能在 [SOURCES.md](SOURCES.md) 中追溯。

符号约定见 [NOTATION.md](NOTATION.md)，数学审查记录见 [MATH_REVIEW.md](MATH_REVIEW.md)。主要定理状态见 [THEOREM_INDEX.md](THEOREM_INDEX.md)，章节依赖见 [DEPENDENCY_GRAPH.md](DEPENDENCY_GRAPH.md)，核心习题解答见 [SOLUTIONS.md](SOLUTIONS.md)。

## 建议总目录

### 第一部分：局部-整体语言

0. [序章：范围、严格性和 Langlands 主线](00_preface_and_scope.md)
1. [第一章：整体域、局部域与 adeles](01_global_fields_and_adeles.md)
2. [第二章：Tate thesis、Hecke 特征与 `GL(1)` L 函数](02_tate_thesis_and_gl1.md)
3. [第三章：类域论作为 `GL(1)` Langlands](03_class_field_theory_as_gl1.md)
4. [第四章：局部紧群、Haar 测度与光滑表示](04_local_groups_haar_and_smooth_representations.md)
5. [第五章：Weil 群、Weil-Deligne 数据与局部参数](05_weil_groups_and_local_parameters.md)

### 第二部分：`GL(2)`、模形式和椭圆曲线

6. [第六章：上半平面上的模形式与 Hecke 算子](06_modular_forms_and_hecke_operators.md)
7. [第七章：adelic 模形式与 `GL(2)` 自守表示](07_adelic_modular_forms_and_gl2.md)
8. [第八章：椭圆曲线、导子和 Hasse-Weil L 函数](08_elliptic_curves_conductors_l_functions.md)
9. [第九章：Galois 表示与模性定理](09_galois_representations_and_modularity.md)
10. [第十章：局部-整体相容性和降层](10_local_global_compatibility_and_level_lowering.md)

### 第三部分：一般 Langlands 纲领

11. [第十一章：还原群、对偶群和 L 群](11_reductive_groups_dual_groups_l_groups.md)
12. [第十二章：局部 Langlands 猜想](12_local_langlands_conjecture.md)
13. [第十三章：全局自守表示和标准 L 函数](13_global_automorphic_representations_and_l_functions.md)
14. [第十四章：`GL(n)` 的 Langlands 对应与已知定理](14_gl_n_correspondence_and_known_theorems.md)
15. [第十五章：函子性原理](15_functoriality_principle.md)
16. [第十六章：trace formula 与 endoscopy](16_trace_formula_and_endoscopy.md)
17. [第十七章：Arthur 参数和谱分解](17_arthur_parameters_and_spectral_decomposition.md)

### 第四部分：几何 Langlands

18. [第十八章：曲线、`G`-bundles 和 Hecke 修改](18_curves_g_bundles_and_hecke_modifications.md)
19. [第十九章：几何 Satake](19_geometric_satake.md)
20. [第二十章：Hecke eigensheaves](20_hecke_eigensheaves.md)
21. [第二十一章：谱侧、局部系统和范畴化对应](21_spectral_side_local_systems_and_categorical_correspondence.md)
22. [第二十二章：函数域类比和数论-几何桥梁](22_function_field_bridge_and_arithmetic_geometry.md)

### 应用与附录

90. [应用章：费马大定理作为 `GL(2)/\mathbb Q` 模性的实例](90_fermat_last_theorem_application.md)
- [附录 A：代数数论复习](A_algebraic_number_theory_review.md)
- [附录 B：局部紧群与 Haar 测度](B_locally_compact_groups_and_haar.md)
- [附录 C：smooth admissible representations](C_smooth_admissible_representations.md)
- [附录 D：模曲线和维数公式](D_modular_curves_and_dimension_formulas.md)
- [附录 E：资料源定理索引](E_external_input_theorem_index.md)
- [附录 F：Fourier 分析、Pontryagin 对偶和 Poisson 求和](F_fourier_analysis_and_poisson.md)
- [附录 G：根资料、对偶群和 L 群计算表](G_root_data_and_dual_group_tables.md)
- [附录 H：Hecke 双陪集、Fourier 系数和 Adelic 比较](H_hecke_double_cosets_and_adelic_comparison.md)
- [附录 I：Godement-Jacquet 与 Rankin-Selberg 积分](I_godement_jacquet_rankin_selberg_integrals.md)

## 当前已起草内容

- [SKILL.md](SKILL.md)：本教材的写作约束。
- [NOTATION.md](NOTATION.md)：全书符号和局部-整体约定。
- [SOURCES.md](SOURCES.md)：主要资料源清单。
- [MATH_REVIEW.md](MATH_REVIEW.md)：审查清单和当前风险。
- [THEOREM_INDEX.md](THEOREM_INDEX.md)：全书主要定理、命题、外部输入和猜想的状态索引。
- [DEPENDENCY_GRAPH.md](DEPENDENCY_GRAPH.md)：章节依赖图、阅读路径和证明依赖层级。
- [SOLUTIONS.md](SOLUTIONS.md)：核心习题解答与提示。
- [00_preface_and_scope.md](00_preface_and_scope.md)：序章。
- [01_global_fields_and_adeles.md](01_global_fields_and_adeles.md)：第一章初稿。
- [02_tate_thesis_and_gl1.md](02_tate_thesis_and_gl1.md)：第二章初稿。
- [03_class_field_theory_as_gl1.md](03_class_field_theory_as_gl1.md)：第三章初稿。
- [04_local_groups_haar_and_smooth_representations.md](04_local_groups_haar_and_smooth_representations.md)：第四章初稿。
- [05_weil_groups_and_local_parameters.md](05_weil_groups_and_local_parameters.md)：第五章初稿。
- [06_modular_forms_and_hecke_operators.md](06_modular_forms_and_hecke_operators.md)：第六章初稿。
- [07_adelic_modular_forms_and_gl2.md](07_adelic_modular_forms_and_gl2.md)：第七章初稿。
- [08_elliptic_curves_conductors_l_functions.md](08_elliptic_curves_conductors_l_functions.md)：第八章初稿。
- [09_galois_representations_and_modularity.md](09_galois_representations_and_modularity.md)：第九章初稿。
- [10_local_global_compatibility_and_level_lowering.md](10_local_global_compatibility_and_level_lowering.md)：第十章初稿。
- [11_reductive_groups_dual_groups_l_groups.md](11_reductive_groups_dual_groups_l_groups.md)：第十一章初稿。
- [12_local_langlands_conjecture.md](12_local_langlands_conjecture.md)：第十二章初稿。
- [13_global_automorphic_representations_and_l_functions.md](13_global_automorphic_representations_and_l_functions.md)：第十三章初稿。
- [14_gl_n_correspondence_and_known_theorems.md](14_gl_n_correspondence_and_known_theorems.md)：第十四章初稿。
- [15_functoriality_principle.md](15_functoriality_principle.md)：第十五章初稿。
- [16_trace_formula_and_endoscopy.md](16_trace_formula_and_endoscopy.md)：第十六章初稿。
- [17_arthur_parameters_and_spectral_decomposition.md](17_arthur_parameters_and_spectral_decomposition.md)：第十七章初稿。
- [18_curves_g_bundles_and_hecke_modifications.md](18_curves_g_bundles_and_hecke_modifications.md)：第十八章初稿。
- [19_geometric_satake.md](19_geometric_satake.md)：第十九章初稿。
- [20_hecke_eigensheaves.md](20_hecke_eigensheaves.md)：第二十章初稿。
- [21_spectral_side_local_systems_and_categorical_correspondence.md](21_spectral_side_local_systems_and_categorical_correspondence.md)：第二十一章初稿。
- [22_function_field_bridge_and_arithmetic_geometry.md](22_function_field_bridge_and_arithmetic_geometry.md)：第二十二章初稿。
- [A_algebraic_number_theory_review.md](A_algebraic_number_theory_review.md)：附录 A，含乘积公式、ray class、idele class 和导子补充。
- [B_locally_compact_groups_and_haar.md](B_locally_compact_groups_and_haar.md)：附录 B，含 Haar 测度、卷积、商测度和 restricted product 积分补充。
- [C_smooth_admissible_representations.md](C_smooth_admissible_representations.md)：附录 C，含 Hecke 作用、Schur 引理、smooth dual 和可容许性补充。
- [D_modular_curves_and_dimension_formulas.md](D_modular_curves_and_dimension_formulas.md)：附录 D，含 $X_0(2)$ genus 计算细节和权 2 微分形式补充。
- [E_external_input_theorem_index.md](E_external_input_theorem_index.md)：附录 E 初稿。
- [F_fourier_analysis_and_poisson.md](F_fourier_analysis_and_poisson.md)：附录 F，含 LCA Fourier 分析、adeles 自对偶和 Poisson 求和接口。
- [G_root_data_and_dual_group_tables.md](G_root_data_and_dual_group_tables.md)：附录 G，含 root datum、dual group 和 L homomorphism 计算表。
- [H_hecke_double_cosets_and_adelic_comparison.md](H_hecke_double_cosets_and_adelic_comparison.md)：附录 H，含 Hecke 双陪集、Fourier 系数和经典-adelic Hecke 比较。
- [I_godement_jacquet_rankin_selberg_integrals.md](I_godement_jacquet_rankin_selberg_integrals.md)：附录 I，含 Godement-Jacquet、Rankin-Selberg 和 converse theorem 积分接口。
- [90_fermat_last_theorem_application.md](90_fermat_last_theorem_application.md)：费马大定理应用章初稿。

## 当前教材化补强层

- 全书主要结果已由 [THEOREM_INDEX.md](THEOREM_INDEX.md) 标记为 `P`、`S`、`E`、`C` 四类，分别对应已证、证明草图、外部输入和猜想。
- 全书阅读路径已由 [DEPENDENCY_GRAPH.md](DEPENDENCY_GRAPH.md) 分成 `GL(1)`、费马应用、一般数论 Langlands 和几何 Langlands 四条路径。
- 核心习题解答已覆盖第 1 至 5 章、若干 `GL(2)` 计算、一般 Langlands 基础、几何 Langlands 入门和费马应用章。
- 附录 A-D、F-I 已从接口复习扩展为带关键证明、计算表和积分接口的参考附录，但还不是可替代专著的完整证明卷。
