# Homological Mirror Symmetry：Fukaya 范畴、导出几何与镜像等价

作者：Dr. Stochastic Parrot

版本：在线阅读稿，2026-07-15 校订

这是一部以增强范畴为基本语言的中文 Homological Mirror Symmetry 教材。
全书从 dg category 与 $A_\infty$-category 出发，分别建立 B-side 的导出几何
和 A-side 的 Fukaya 理论，再进入 wrapped/stopped 范畴、标准镜像例子、生成与
局部化、microlocal sheaves、奇点模型以及函子化问题。读者不会在这里看到把
“两个导出范畴相等”当作终点的写法：每一项 HMS 断言都要说明系数、增强模型、
完成方式、等价强度与深层外部输入。

建议先读[序章](00_preface_and_scope.md)和[符号约定](NOTATION.md)。熟悉导出范畴
的读者仍不宜跳过第一章，因为后文区分 raw $H^0$、quasi-equivalence 与 Morita
equivalence；熟悉辛几何的读者也应从第三章开始核对本书采用的 brane、分次和
系数口径。分析基础和大型 HMS 定理不在书内伪造证明，而在正文中以“外部输入
定理”标出，并可由[资料源](SOURCES.md)和[定理定位表](ONLINE_THEOREM_LOCATOR.md)
追溯。

## 阅读约定

- 定义先于使用；非平凡结论给出书内证明，或明确标为外部输入。
- dg、$A_\infty$、三角、stable $\infty$ 与 Morita 层级不互相省略。
- A-side 的 exact/monotone/Novikov、brane data、紧致性、正则性和取向假设
  必须与所用 Fukaya 模型同时出现。
- B-side 的 $\operatorname{Perf}$、$\mathrm D^b\operatorname{Coh}$、
  Fourier--Mukai 与 matrix-factorization 模型必须说明几何假设。
- 深定理的引用给出本书实际使用的结论范围；来源只证明 fully faithful 或
  quasi-embedding 时，正文不会把它改写成等价。

完整写作约束见 [SKILL.md](SKILL.md)，练习答案见 [SOLUTIONS.md](SOLUTIONS.md)。

## 目录

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
11. [第十章：toric Fano、Landau--Ginzburg potential 与 Jacobian ring](10_toric_fano_landau_ginzburg_potentials_and_jacobian_rings.md)
12. [第十一章：Fukaya--Seidel category 与 Picard--Lefschetz theory](11_fukaya_seidel_categories_and_picard_lefschetz_theory.md)
13. [第十二章：K3 曲面、四次曲面与 Calabi--Yau hypersurfaces](12_k3_quartics_and_calabi_yau_hypersurfaces.md)
14. [第十三章：pairs of pants、tropical degeneration 与 algebraic-torus hypersurfaces](13_pairs_of_pants_tropical_degeneration_and_hypersurfaces.md)

### 第四部分：生成、局部化与 sheaf 模型

15. [第十四章：split-generation、open--closed map 与 Abouzaid criterion](14_split_generation_open_closed_and_abouzaid_criterion.md)
16. [第十五章：wrapped Fukaya categories 的 sectorial descent](15_sectorial_descent_for_wrapped_fukaya_categories.md)
17. [第十六章：Nadler--Zaslow、microlocal sheaves 与 cotangent bundles](16_nadler_zaslow_microlocal_sheaves_and_cotangent_bundles.md)
18. [第十七章：stop removal、Viterbo functor 与 functorial HMS](17_stop_removal_viterbo_functors_and_functorial_hms.md)
19. [第十八章：Hochschild invariants、closed--open maps 与 categorical checks](18_hochschild_closed_open_and_categorical_enumerative_checks.md)

### 第五部分：奇点与函子化边界

20. [第十九章：Rabinowitz Fukaya categories、singularities 与 matrix factorizations](19_rabinowitz_fukaya_singularities_and_matrix_factorizations.md)
21. [第二十章：函子化、墙穿越与尚未统一的范畴结构](20_functorial_wall_crossing_bps_and_2026_research_boundary.md)

### 附录与参考表

附录 A--F 固定集合论、$A_\infty$ 符号、dg 商、Fourier--Mukai 技术、Floer
分析输入和 wrapped 例子；附录 G--L 汇集外部定理、反例、低阶计算、标准模型
与生成/局部化论证。阅读时还可查阅[术语表](GLOSSARY.md)、
[依赖图](DEPENDENCY_GRAPH.md)、[定理账本](THEOREM_LEDGER.md)、
[外部输入使用表](EXTERNAL_INPUT_USAGE_TABLE.md)和[练习答案](SOLUTIONS.md)。
