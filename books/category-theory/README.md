# 范畴论：从普通范畴到 $\infty$-范畴

作者：Dr. Stochastic Parrot  
状态：严格教材草稿，多轮扩写中  
主资料源：Mac Lane, Borceux, Kelly, Riehl, Leinster, Awodey, Jacobs, Adamek-Rosicky, Dwyer-Kan, Rezk, Bergner, Goodwillie, Morel-Voevodsky, Lurie, Riehl-Verity, Kerodon, BBD, Ravenel, Hopkins-Smith, Hovey-Palmieri-Strickland, Balmer, Nikolaus-Scholze, Blumberg-Mandell, Dundas-Goodwillie-McCarthy, Kashiwara-Schapira, Gaitsgory-Rozenblyum, Toën-Vezzosi, Saavedra, Deligne-Milne, Keller, Toën, Tabuada, Goresky-MacPherson, Francis, Ayala-Francis, Ayoub, Cisinski-Déglise, Voevodsky, Hofmann-Streicher, Shulman, Clausen-Scholze

本书目标是写成一部严格的范畴论教材，而不是主题导览。正文从范畴、函子和自然变换开始，逐步进入 Yoneda 引理、极限、伴随、Kan 延拓、幺半与富范畴、可表现范畴、topos、同伦范畴论和 $\infty$-范畴。

## 写作约束

本书写作约束见 [SKILL.md](SKILL.md)。后续扩写必须遵守：

- 定义先于直觉，命题带证明或明确的外部输入标记。
- 全书固定集合论宇宙和小性约定。
- 泛性质必须写成可检查的自然同构、终对象/始对象或表示性语句。
- 高阶内容必须区分严格范畴、2-范畴、同伦范畴、simplicial category 和 $\infty$-范畴。
- 后续扩写范围控制在范畴论本体；外部领域深定理只作为外部输入，不在本书内部闭合。
- 资料源必须能在 [SOURCES.md](SOURCES.md) 中追溯。

符号约定见 [NOTATION.md](NOTATION.md)，术语索引见 [TERM_INDEX.md](TERM_INDEX.md)，数学审查记录见 [MATH_REVIEW.md](MATH_REVIEW.md)。
练习答案见 [SOLUTIONS.md](SOLUTIONS.md)。跨章节综合题见 [COMPREHENSIVE_EXERCISES.md](COMPREHENSIVE_EXERCISES.md)，答案见 [COMPREHENSIVE_SOLUTIONS.md](COMPREHENSIVE_SOLUTIONS.md)。
章节来源注释见 [CHAPTER_SOURCE_NOTES.md](CHAPTER_SOURCE_NOTES.md)，外部输入依赖图见 [THEOREM_DEPENDENCIES.md](THEOREM_DEPENDENCIES.md)。

## 总目录

### 第一部分：普通范畴论基础

0. [序章：范围、资料源和严格性标准](00_preface_and_scope.md)
1. [第一章：范畴、函子与自然变换](01_categories_functors_natural_transformations.md)
2. [第二章：泛性质与 Yoneda 引理](02_universal_properties_and_yoneda.md)
3. [第三章：极限与余极限](03_limits_and_colimits.md)
4. [第四章：伴随函子](04_adjoint_functors.md)
5. [第五章：可表函子、密度与生成元](05_representables_density_generators.md)
6. [第六章：Kan 延拓](06_kan_extensions.md)
7. [第七章：单子、余单子与代数](07_monads_and_algebras.md)

### 第二部分：结构性范畴论

8. [第八章：幺半范畴与相干性](08_monoidal_categories.md)
9. [第九章：闭范畴、张量-Hom 伴随与 Day 卷积](09_closed_categories_and_day_convolution.md)
10. [第十章：富范畴、加权极限与 enriched Yoneda](10_enriched_categories.md)
11. [第十一章：end、coend 与 Fubini 定理](11_ends_and_coends.md)
12. [第十二章：可表现范畴与可达范畴](12_presentable_and_accessible_categories.md)
13. [第十三章：正规、正合、阿贝尔和 Grothendieck 范畴](13_exact_abelian_grothendieck_categories.md)
14. [第十四章：站点、sheaf 与 Grothendieck topos](14_sites_sheaves_and_topoi.md)

### 第三部分：高阶与同伦范畴论

15. [第十五章：2-范畴、双范畴与弱相干性](15_two_categories_and_bicategories.md)
16. [第十六章：模型范畴与同伦范畴](16_model_categories_and_homotopy_categories.md)
17. [第十七章：单纯集与 quasi-category](17_simplicial_sets_and_quasicategories.md)
18. [第十八章：$\infty$-范畴中的等价、极限和伴随](18_limits_adjunctions_in_infinity_categories.md)
19. [第十九章：Cartesian fibration 与 straightening](19_cartesian_fibrations_and_straightening.md)
20. [第二十章：稳定 $\infty$-范畴与谱](20_stable_infinity_categories_and_spectra.md)
21. [第二十一章：高阶 topos](21_higher_topos_theory.md)
22. [第二十二章：高阶代数、$\infty$-operad 与幺半 $\infty$-范畴](22_higher_algebra_and_infinity_operads.md)
23. [第二十三章：可表现 $\infty$-范畴、可达局部化与 $\operatorname{Pr}^L$](23_presentable_infinity_categories_and_localizations.md)
24. [第二十四章：Profunctor、Cauchy 完备化与 Correspondence](24_profunctors_cauchy_completion_and_correspondences.md)
25. [第二十五章：富 Profunctor、Equipment 与 Beck-Chevalley 条件](25_enriched_profunctors_equipments_and_base_change.md)
26. [第二十六章：紧生成、Brown 表示性与 Bousfield 局部化](26_compact_generation_brown_representability_and_bousfield_localization.md)
27. [第二十七章：dg 范畴、稳定增强与导出 Morita 理论](27_dg_categories_enhancements_and_derived_morita_theory.md)
28. [第二十八章：六操作形式主义、基变换与投影公式](28_six_functor_formalism_base_change_and_projection_formula.md)
29. [第二十九章：相对范畴、单纯局部化与模型比较](29_relative_categories_simplicial_localization_and_model_comparisons.md)
30. [第三十章：dg 商、局部化不变量与非交换 motives](30_dg_quotients_localizing_invariants_and_noncommutative_motives.md)
31. [第三十一章：Perverse sheaves、recollement 与 t-结构](31_perverse_sheaves_recollement_and_t_structures.md)
32. [第三十二章：Chromatic homotopy、Bousfield lattice 与 telescope conjecture](32_chromatic_homotopy_bousfield_lattices_and_telescope_conjecture.md)
33. [第三十三章：$D$-modules、Riemann-Hilbert 与 de Rham 函子](33_d_modules_riemann_hilbert_and_de_rham_functors.md)
34. [第三十四章：导出代数几何、cotangent complex 与 spectral stacks](34_derived_algebraic_geometry_cotangent_complexes_and_spectral_stacks.md)
35. [第三十五章：Barr-Beck-Lurie 单子性、余单子下降与 descent](35_barr_beck_lurie_monadicity_and_descent.md)
36. [第三十六章：Tannaka duality、仿射群概形与高阶重构](36_tannaka_duality_affine_group_schemes_and_higher_reconstruction.md)
37. [第三十七章：Tensor triangular geometry、Balmer spectrum 与支撑理论](37_tensor_triangular_geometry_balmer_spectra_and_support.md)
38. [第三十八章：Topological Hochschild homology、cyclotomic trace 与 $TC$](38_topological_hochschild_homology_cyclotomic_trace_and_tc.md)
39. [第三十九章：Goodwillie calculus、excisive functors 与函子导数](39_goodwillie_calculus_excisive_functors_and_derivatives.md)
40. [第四十章：Motivic homotopy、$\mathbb A^1$-局部化与六操作](40_motivic_homotopy_a1_localization_and_six_operations.md)
41. [第四十一章：范畴逻辑、依赖类型论与 Univalence](41_categorical_logic_dependent_type_theory_and_univalence.md)
42. [第四十二章：因子化同调、$E_n$-代数与非阿贝尔 Poincare 对偶](42_factorization_homology_en_algebras_and_nonabelian_poincare_duality.md)
43. [第四十三章：Condensed sets、Solid modules 与解析范畴](43_condensed_sets_solid_modules_and_analytic_categories.md)
44. [第四十四章：语法范畴、分类 Topos 与 Tripos](44_syntactic_categories_classifying_toposes_and_tripos.md)
45. [第四十五章：正合完成、关系、Allegory 与 Regular 逻辑](45_exact_completions_relations_allegories_and_regular_logic.md)
46. [第四十六章：Cohesive Topos、模态与微分凝聚](46_cohesive_toposes_modalities_and_differential_cohesion.md)
47. [第四十七章：层化同伦、Exit-path 范畴与可构造 Sheaves](47_stratified_homotopy_exit_path_categories_and_constructible_sheaves.md)
48. [第四十八章：高阶 Morita、Trace 与 $E_n$-Koszul 对偶](48_higher_morita_traces_and_en_koszul_duality.md)
49. [第四十九章：Derivator、同伦 Kan 延拓与稳定 Derivator](49_derivators_homotopy_kan_extensions_and_stable_derivators.md)
50. [第五十章：Stacks、Gerbes 与非阿贝尔上同调](50_stacks_gerbes_and_nonabelian_cohomology.md)
51. [第五十一章：范畴 Galois 理论、Descent 与有效下降](51_categorical_galois_theory_descent_and_effective_descent.md)
52. [第五十二章：多项式函子、Species、解析函子与 W-types](52_polynomial_functors_species_analytic_functors_and_w_types.md)
53. [第五十三章：$\infty$-Cosmos 与模型无关的高阶范畴论](53_infinity_cosmoi_model_independent_infinity_category_theory.md)
54. [第五十四章：正交性、因子化系统与弱因子化系统](54_orthogonality_factorization_systems_and_weak_factorization.md)
55. [第五十五章：Sketches、Doctrines 与范畴化理论](55_sketches_doctrines_and_categorical_theories.md)
56. [第五十六章：幂等分裂、Karoubi 包络与绝对余极限](56_idempotents_karoubi_envelopes_and_absolute_colimits.md)

### 附录

- [附录 A：集合论宇宙与大小问题](A_universes_and_size.md)
- [附录 B：单纯形范畴 $\Delta$ 与单纯恒等式](B_simplex_category_and_nerve_details.md)
- [附录 C：常用泛性质证明模板](C_universal_property_templates.md)
- [附录 D：资料源定理索引](D_theorem_source_index.md)
- [附录 E：高阶范畴的技术模型](E_higher_categorical_technical_models.md)
- [附录 F：范围边界与外部输入政策](F_scope_boundary_and_external_input_policy.md)
- [附录 G：终稿化审查标准](G_final_textbookization_audit.md)

## 当前范围

当前版本完成全目录多轮致密草稿：每章均包含目标、前置知识、核心定义、基本命题或外部输入定理、本章小结和练习。普通范畴论基础章节已补齐主要证明，并进一步补入终稿阅读约定、骨架、等价边界例子、可表性边界、Yoneda 计算原则、极限的表示性刻画、Set 余等化子、创造极限、有限极限反例、共尾性定理、偏序伴随、对角函子伴随、伴随保持性反例、伴随的全忠实判别、反射子范畴、生成族忠实判别、稠密与本质满边界、右 Kan 点态公式完整证明、Kan 点态公式的共尾缩小、有伴随时 Kan 点态公式退化、Kan 延拓存在性边界、Kleisli 伴随、Eilenberg-Moore 自由-遗忘伴随、幂等单子和反射单子；第八至第十一章已补入辫/对称幺半范畴、松幺半函子传代数对象、单子作为端函子范畴中的代数对象、非辫性例子、闭结构单位内部 Hom、指数律、非闭幺半结构反例、Day 卷积单位计算、enriched Yoneda 的书内证明、富 Yoneda 全忠实、张量/余张量、集合值 coend 商公式、end/coend 形式 Yoneda 和存在性边界；第十二至第十四章已补入预层范畴局部可表现性、强生成子、紧生成对象检测自然同构、Set 有限生成与基数边界、局部可表现范畴伴随函子定理、image/coimage、核/余核判别单满、正合函子保持 image/coimage、模范畴 Grothendieck 性、阿贝尔非 Grothendieck 边界、separated 预层、plus 构造、sheaf 极限创建、sheaf 化反射泛性质和几何态射复合；第十七至第十九章已吸收 join、slice、左右映射空间模型、correspondence 表示性口径、adjunction data 低维展开、walking adjunction、scaled nerve 低维口径、Joyal 模型结构、marked/scaled simplicial sets、Cartesian model structure、标准单纯形计算、ordinary pullback 恢复、普通 Grothendieck construction、基为 $[1]$ 和 $[2]$ 的 straightening 低维模型、Cartesian 传输函子和 Cartesian sections；第二十一章已补入 effective epimorphism、groupoid object、Postnikov tower、hypercompletion、$\infty$-几何态射和点；第二十三章已补入 presentable $\infty$-categories、$\operatorname{Ind}_\kappa$、accessible localization、Bousfield localization、left exact/exact localization 和 $\operatorname{Pr}^L$；第二十四至第二十五章已补入 profunctor、富 profunctor、coend 复合、Cauchy completion、加权余极限、equipment、Beck-Chevalley 条件和 $\infty$-correspondence；第二十六至第五十六章已补入 compact generation、Brown representability、Verdier quotient、Bousfield localization、smashing localization、Neeman-Thomason 型定理、dg category、dg modules、pretriangulated enhancement、derived Morita equivalence、dg bimodules、perfect modules、Hochschild 型 Morita 不变量、六操作形式主义、基变换、投影公式、recollement、Verdier 对偶、relative categories、Dwyer-Kan localization、simplicial categories、coherent nerve、complete Segal spaces、模型比较、dg quotient、localizing invariants、noncommutative motives、perverse sheaves、中间延拓、BBD gluing、nearby cycles、vanishing cycles、chromatic homotopy、Bousfield lattice、Morava $K$-theory、thick subcategory theorem、telescope conjecture、chromatic fracture square、$D$-modules、Riemann-Hilbert correspondence、de Rham functor、derived stacks、$\operatorname{QCoh}$、cotangent complex、formal moduli problems、$\operatorname{IndCoh}$、Barr-Beck-Lurie 单子性、comonadic descent、Tannaka duality、高阶重构、tensor triangular geometry、Balmer spectrum、$THH$、cyclotomic spectra、$TC$、cyclotomic trace、Goodwillie calculus、excisive functors、functor derivatives、motivic homotopy、$\mathbb A^1$-localization、stable motivic homotopy category、范畴逻辑、依赖类型论、univalence、语法范畴、分类 topos、tripos、正合完成、allegory、cohesive topos、modalities、微分凝聚、exit-path $\infty$-categories、constructible sheaves、层化因子化同调、高阶 Morita traces、$E_n$-Koszul duality、derivators、stacks、gerbes、nonabelian cohomology、categorical Galois theory、polynomial functors、species、W-types、$\infty$-cosmoi、orthogonality、factorization systems、sketches、doctrines、Karoubi envelopes、absolute colimits、condensed sets、solid modules 和 analytic rings；第二十至第二十二章已补入 sequential prespectrum、$\Omega$-谱、映射谱、smash product、悬挂-环路互逆、正合函子、t-结构 heart 核余核、heart 加性、cohomology 长正合列、exact couple、有限滤过与完备滤过谱序列收敛、离散 sheaf 与 ordinary sheaf 比较、超覆盖、超下降、active/inert 分解、Segal 条件、多重映射空间、模 $\infty$-范畴、bar 构造、相对张量积、Morita、单位双模、矩阵代数 Morita 等价、smooth/proper 可对偶性判别、Frobenius 代数二维 TFT 影子、中心、因子化同调、fully dualizable objects 和 cobordism hypothesis 等内容。高阶大型结构定理保留外部输入标记，并在 [SOURCES.md](SOURCES.md) 与 [D_theorem_source_index.md](D_theorem_source_index.md) 中记录来源边界。终稿化审查标准见 [G_final_textbookization_audit.md](G_final_textbookization_audit.md)。全部现有章末练习均在 [SOLUTIONS.md](SOLUTIONS.md) 中给出答案或解题要点；综合题另有独立答案。

## 审稿辅助文件

- [TERM_INDEX.md](TERM_INDEX.md)：核心术语索引。
- [CHAPTER_SOURCE_NOTES.md](CHAPTER_SOURCE_NOTES.md)：逐章资料源、书内证明范围和外部输入边界。
- [THEOREM_DEPENDENCIES.md](THEOREM_DEPENDENCIES.md)：外部输入定理依赖图。
- [MATH_REVIEW.md](MATH_REVIEW.md)：数学审查清单和下一轮风险。
- [G_final_textbookization_audit.md](G_final_textbookization_audit.md)：终稿化审查标准。
- [validate.py](validate.py)：本目录的内部链接、章节结构、占位标记和习题答案覆盖检查。
