# 凝聚数学讲义：第一卷

作者：Dr. Stochastic Parrot  
状态：严格教材草稿，第一卷基本完本草稿
主资料源：Peter Scholze, *Lectures on Condensed Mathematics*；Asgeirsson 等，*Categorical Foundations of Formalized Condensed Mathematics*

这是《凝聚数学讲义》的第一卷。目标不是写一篇“什么是凝聚数学”的介绍文章，而是按严格教材方式，从站点、sheaf、紧 Hausdorff 空间、profinite 空间、凝聚集合、凝聚阿贝尔群、solid 对象和 analytic rings 逐步建立理论。

## 写作约束

本书的写作约束见 [SKILL.md](SKILL.md)。后续扩写必须遵守：

- 以正式资料为依据。
- 定义、命题、证明、例子、练习齐全。
- 不用直觉类比替代数学定义。
- 不跳过关键公式和等化子条件。
- 不把尚未证明的高级结果当作基础事实使用。

资料源见 [SOURCES.md](SOURCES.md)，符号约定见 [NOTATION.md](NOTATION.md)，数学审查记录见 [MATH_REVIEW.md](MATH_REVIEW.md)。

## 建议阅读顺序

1. [序章：为什么需要凝聚数学](00_preface_and_plan.md)
2. [第一章：站点、覆盖与 sheaf 条件](01_sites_and_sheaves.md)
3. [第二章：紧 Hausdorff 空间与 profinite 空间](02_compact_hausdorff_and_profinite.md)
4. [第三章：凝聚集合](03_condensed_sets.md)
5. [第四章：凝聚阿贝尔群](04_condensed_abelian_groups.md)
6. [第五章：测试站点的比较](05_comparison_of_test_sites.md)
7. [第六章：极不连通空间](06_extremally_disconnected_spaces.md)
8. [第七章：自由对象与投射生成元](07_free_objects_and_projectives.md)
9. [第八章：正合性检测与第一层 Ext](08_exactness_and_first_ext.md)
10. [第九章：张量积与凝聚环](09_tensor_products_and_condensed_rings.md)
11. [第十章：凝聚模](10_condensed_modules.md)
12. [第十一章：派生张量积与 Tor](11_derived_tensor_and_tor.md)
13. [第十二章：固体阿贝尔群](12_solid_abelian_groups.md)
14. [第十三章：固体张量积](13_solid_tensor_products.md)
15. [第十四章：解析环](14_analytic_rings.md)
16. [第十五章：全局化与相干对偶纲要](15_globalization_and_duality.md)
17. [附录 A：集合论宇宙与小性约定](A_universes_and_size.md)
18. [附录 B：站点比较定理](B_site_comparison_theorem.md)
19. [附录 C：阿贝尔群值 sheaf 的范畴性质](C_sheaves_of_abelian_groups.md)
20. [附录 D：Stone 对偶与 Gleason cover](D_stone_gleason.md)
21. [附录 E：sheaf 模、内部 Hom 与派生张量](E_sheaf_modules_and_internal_hom.md)
22. [附录 F：Nöbeling 定理与 solid 计算](F_nobeling_and_solid_calculations.md)
23. [附录 G：基本 Ext 与 Tor 计算](G_basic_ext_and_tor_calculations.md)
24. [附录 H：正合 sheafification 与派生工具](H_exact_sheafification_and_derived_tools.md)
25. [附录 I：Horseshoe 引理与导出函子形式](I_horseshoe_and_derived_functor_formalism.md)
26. [附录 J：Regular open 代数与 Gleason cover 细节](J_regular_open_and_gleason_cover_details.md)
27. [附录 K：ED 覆盖检测与有效满射](K_ed_cover_detection_and_effective_epimorphisms.md)
28. [附录 L：边界例子与反例](L_boundary_examples_and_counterexamples.md)
29. [附录 M：Ext 与 Tor 工作例题](M_worked_ext_tor_examples.md)
30. [附录 N：Stone 对偶的完整证明链](N_stone_duality_full_proof.md)
31. [附录 O：Gleason 投射性定理的证明模块](O_gleason_projectivity_modules.md)
32. [附录 P：Nöbeling 定理的证明模块](P_nobeling_proof_modules.md)

## 第一卷结构

- 第 0 章：问题背景、资料源和全书路线。
- 第 1 章：站点、覆盖族、预层、sheaf 条件、可表 sheaf。
- 第 2 章：紧 Hausdorff 空间、profinite 空间、有限联合满射覆盖。
- 第 3 章：凝聚集合的定义与基本例子。
- 第 4 章：凝聚阿贝尔群与阿贝尔范畴性质。
- 第 5 章：测试站点的比较。
- 第 6 章：极不连通空间与计算 sheaf 的简化。
- 第 7 章：自由对象与投射生成元。
- 第 8 章：正合性检测与第一层 Ext。
- 第 9 章：张量积与凝聚环。
- 第 10 章：凝聚模。
- 第 11 章：派生张量积与 Tor。
- 第 12 章：固体阿贝尔群。
- 第 13 章：固体张量积。
- 第 14 章：解析环。
- 第 15 章：全局化与相干对偶纲要。
- 附录 A：集合论宇宙与小性问题。
- 附录 B：站点比较定理。
- 附录 C：阿贝尔群值 sheaf 的范畴性质。
- 附录 D：Stone 对偶与 Gleason cover。
- 附录 E：sheaf 模、内部 Hom 与派生张量。
- 附录 F：Nöbeling 定理与 solid 计算。
- 附录 G：基本 Ext 与 Tor 计算。
- 附录 H：正合 sheafification、Grothendieck 阿贝尔范畴和 K-flat 派生工具。
- 附录 I：Horseshoe 引理、投射分解比较、长正合列和维数平移。
- 附录 J：regular open 代数、Stone 空间到紧 Hausdorff 空间的 Gleason cover 映射。
- 附录 K：ED 覆盖检测 sheaf 单射、满射、同构和阿贝尔 sheaf 正合性。
- 附录 L：sheaf 满射、separated presheaf、站点比较、无限乘积张量和拓扑阿贝尔群的边界例子。
- 附录 M：有限离散自由对象、两项投射分解、乘以 $n$ 的 Ext/Tor 工作例题。
- 附录 N：滤子、超滤子、Stone 空间紧性、开闭代数同构和 profinite 逆极限表示。
- 附录 O：regular open 完备 Boolean algebra、Sikorski extension、Gleason lifting 的证明模块。
- 附录 P：Nöbeling 定理的有限、可数与超限过滤证明模块。

## 当前范围

当前版本完成第一卷：基础部分给出书内证明，新增附录 H 将 sheafification 正合性和派生张量的同调代数基础补齐，附录 I-K 补投射分解、Gleason cover 形式细节和 ED 覆盖检测正合性的证明链，附录 L 补关键假设的边界例子，附录 M 补 Ext/Tor 工作例题，附录 N 补 Stone 对偶的完整证明链，附录 O-P 补 Gleason 投射性和 Nöbeling 定理的证明模块；solid、analytic rings 和相干对偶部分给出严格定义、核心定理、输入定理和引用边界。Gleason lifting、Nöbeling 一般情形和 Scholze 的 solid/analytic 结构定理仍作为外部输入定理使用；第二卷继续展开这些高阶结构。

分卷答案见 [SOLUTIONS.md](SOLUTIONS.md)。总目录见 [凝聚数学讲义](../)。续卷见 [凝聚数学讲义：第二卷](../volume-2/)。
