# 凝聚数学讲义：第二卷

作者：Dr. Stochastic Parrot  
状态：solid/analytic/liquid 主线输入定理型闭合草稿
副题：Solid、Analytic 与 Liquid 结构

本卷接续 [第一卷](../volume-1/)。第一卷建立站点、凝聚集合、凝聚阿贝尔群、投射生成元、基本同调代数、solid 与 analytic 的入口；第二卷把这些入口发展成 solid、analytic 与 liquid 主线。本卷不是应用附录，而是全书的核心理论卷之一。

## 写作边界

第二卷不重新证明第一卷已经完成的基础事实。默认读者已经熟悉：

- $\mathbf{CondSet}$ 与 $\mathbf{CondAb}$。
- 极不连通紧 Hausdorff 空间作为投射测试对象。
- $\mathbb Z^\square[S]$、solidification 和 solid tensor product 的定义。
- 基本 $\operatorname{Ext}$、$\operatorname{Tor}$ 和派生张量。

本卷的目标是把 Scholze 讲义后半部分写成中文教材：定义要完整，定理要标清输入来源，证明能展开的就展开，不能在本卷完全展开的深定理必须说明依赖位置。solid、analytic 和 liquid 的分工见 [../MAINLINE_ARCHITECTURE.md](../MAINLINE_ARCHITECTURE.md)。

## 建议阅读顺序

1. [序章：第二卷的主题和边界](00_preface_and_scope.md)
2. [第一章：solid 派生范畴](01_solid_derived_categories.md)
3. [第二章：solid 环与 solid 模](02_solid_rings_and_modules.md)
4. [第三章：解析环的正式条件](03_analytic_rings_formal_conditions.md)
5. [第四章：解析化与 Bousfield localization](04_analyticization_and_bousfield_localization.md)
6. [第五章：liquid 向量空间入口](05_liquid_vector_spaces.md)
7. [第六章：离散 Huber pair 与解析环](06_discrete_huber_pairs_and_analytic_rings.md)
8. [第七章：$f_!$、投影公式与相干对偶](07_coherent_duality_and_f_shriek.md)
9. [第八章：复几何应用的范畴语言](08_complex_geometry_language.md)
10. [附录 A：输入定理与证明路线](A_input_theorems_and_proof_roadmap.md)
11. [附录 B：例子与类型检查](B_worked_examples_and_type_checks.md)
12. [附录 C：Bousfield localization 与解析化的形式定理](C_bousfield_localization_formalism.md)
13. [附录 D：第二卷输入定理登记表](D_precise_input_theorem_register.md)
14. [附录 E：局部化技术引理](E_localization_technical_lemmas.md)
15. [附录 F：伴随函子与投影公式的形式骨架](F_adjoint_functor_and_projection_formula.md)
16. [附录 G：Cech 下降与 totalization](G_cech_descent_and_totalization.md)
17. [附录 H：紧生成、局部化子范畴与生成元检验](H_compact_generation_and_generator_tests.md)
18. [附录 I：解析环公理检查表与失败模式](I_analytic_ring_axioms_and_failure_modes.md)
19. [附录 J：Liquid、Banach 与 Fréchet 的边界](J_liquid_banach_frechet_boundaries.md)
20. [附录 K：幺半 Bousfield 局部化细节](K_monoidal_bousfield_localization_details.md)
21. [附录 L：闭幺半局部化与内部 Hom](L_closed_monoidal_localization_and_internal_hom.md)
22. [附录 M：Solid localization 的生成、反射与完备化口径](M_solid_localization_generation_and_completion.md)
23. [附录 N：Analytic descent 与 rational localization 的证明义务](N_analytic_descent_and_rational_localization_obligations.md)
24. [附录 O：可展示稳定局部化与正合性](O_presentable_localization_and_exactness.md)
25. [附录 P：Liquid、Fréchet 复形与闭值域](P_liquid_frechet_complexes_and_closed_range.md)
26. [附录 Q：Solid 主定理包](Q_solid_main_theorem_package.md)
27. [附录 R：Analytic 主定理包](R_analytic_main_theorem_package.md)
28. [附录 S：Liquid 主定理包](S_liquid_main_theorem_package.md)
29. [附录 T：Solid、Analytic 与 Liquid 的统一闭包](T_mainline_closure_theorem.md)
30. [附录 U：第二卷出版级闭包审查](U_publication_closure_audit.md)
31. [附录 V：Solidification 反射存在性的证明模块](V_solidification_reflection_proof.md)
32. [附录 W：Solid 核张量理想性的证明模块](W_solid_tensor_ideal_proof_modules.md)
33. [附录 X：Analytic localization 的证明模块](X_analytic_localization_proof_modules.md)
34. [附录 Y：Rational descent 的证明模块](Y_rational_descent_proof_modules.md)
35. [附录 Z：Liquid realization 的证明模块](Z_liquid_realization_proof_modules.md)
36. [附录 AA：Scholze 与 Clausen-Scholze 核心定理图谱](AA_scholze_clausen_scholze_core_theorem_atlas.md)

## 当前范围

当前版本完成第二卷主线输入定理型闭合草稿：它给出 solid 派生范畴、solid 环与模、解析环、解析化、liquid 入口、离散 Huber pair、$f_!$ 与复几何应用的范畴语言。Scholze 和 Clausen-Scholze 的深层结构定理以输入定理形式标注；附录 A 记录证明路线，附录 B 做类型检查，附录 C-D 把 localization 形式定理和输入定理颗粒度写清，附录 E 证明局部等价、局部化核、张量理想、幺半下降和相对张量积的技术引理，附录 F 证明伴随函子、投影公式和内部 Hom 相容的形式骨架，附录 G 补 Cech nerve、totalization、稳定范畴值 descent 和 rational Cech 下降的形式推论，附录 H 补紧生成、localizing subcategory 和生成元检验的形式证明，附录 I 补 analytic ring 公理检查表、cone 判别和失败模式，附录 J 补 liquid 与 Banach/Fréchet 的边界，附录 K 补幺半 Bousfield 局部化和相对张量积的下降判别，附录 L 补闭幺半局部化与内部 Hom 的类型边界，附录 M-N 补 solid localization 的生成核口径和 analytic/rational descent 的证明义务，附录 O-P 补可展示稳定局部化的正合形式和 Fréchet/liquid 闭值域类型检查，附录 Q-T 把 solid、analytic、liquid 和三者统一接口整理成主定理包，附录 U 给出出版级闭包审查，附录 V-Z 进一步把 solidification 反射存在性、solid 核张量理想性、analytic localization、rational descent 和 liquid realization 拆成教材内部证明模块，附录 AA 把 Scholze 与 Clausen-Scholze 的核心定理作为主线图谱集中列出。真正证明 compact complex manifolds 的 finiteness、Serre duality、GAGA 和 Riemann-Roch 留给第三卷。

若后续扩写到出版级篇幅，第二卷是最自然的拆分对象：solid 可独立成卷，analytic 与 liquid 可独立成卷。当前仓库暂不移动文件。

续卷见 [凝聚数学讲义：第三卷](../volume-3/)。

## 资料

资料源见 [SOURCES.md](SOURCES.md)，符号约定见 [NOTATION.md](NOTATION.md)，审查记录见 [MATH_REVIEW.md](MATH_REVIEW.md)，分卷答案见 [SOLUTIONS.md](SOLUTIONS.md)。
