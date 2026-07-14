# 凝聚数学讲义：第三卷

作者：Dr. Stochastic Parrot  
状态：输入定理型严格教材稿
副题：复几何与相干对偶

本卷接续 [第一卷](../volume-1/) 与 [第二卷](../volume-2/)。第一卷建立凝聚基础，第二卷建立 solid、analytic、liquid 和 $f_!$ 的范畴语言；第三卷把这些工具放入复几何。

## 建议阅读顺序

1. [序章：从函数空间到复几何定理](00_preface_and_goals.md)
2. [第一章：复解析空间的凝聚语言](01_complex_analytic_spaces_condensed_language.md)
3. [第二章：相干层、perfect 复形与点支撑例子](02_coherent_sheaves_and_derived_categories.md)
4. [第三章：Dolbeault resolution 与 liquid 复形](03_dolbeault_complexes_and_liquid_modules.md)
5. [第四章：相干上同调有限性](04_finiteness_of_coherent_cohomology.md)
6. [第五章：Serre 对偶](05_serre_duality.md)
7. [第六章：GAGA](06_gaga.md)
8. [第七章：Grothendieck-Hirzebruch-Riemann-Roch](07_hirzebruch_riemann_roch.md)
9. [第八章：六函子关系与开放问题](08_six_functors_and_outlook.md)
10. [附录 A：复几何定理的证明路线](A_proof_roadmap_for_complex_geometry.md)
11. [附录 B：经典语言与凝聚语言对照](B_classical_to_condensed_dictionary.md)
12. [附录 C：Stein、Cech 与相干分解](C_stein_cech_and_coherent_resolutions.md)
13. [附录 D：Dolbeault 与 Serre 对偶计算模型](D_dolbeault_and_serre_calculations.md)
14. [附录 E：GAGA 与 Riemann-Roch 例子](E_gaga_and_riemann_roch_examples.md)
15. [附录 F：经典复几何输入定理的精确形式](F_classical_complex_geometry_prerequisites.md)
16. [附录 G：复几何主定理的依赖链](G_dependency_chain_for_complex_geometry.md)
17. [附录 H：$\mathbb P^1$ 上线丛上同调的 Čech 计算](H_p1_line_bundle_cech_calculation.md)
18. [附录 I：Čech 超上同调与谱序列](I_cech_hypercohomology_and_spectral_sequences.md)
19. [附录 J：Serre 对偶的形式证明层](J_serre_duality_formalism.md)
20. [附录 K：GAGA 与 Riemann-Roch 的形式推论](K_gaga_and_riemann_roch_formal_consequences.md)
21. [附录 L：Fredholm-Hodge 有限性的形式证明层](L_fredholm_hodge_finiteness.md)
22. [附录 M：有限分解、谱序列与有限性边界](M_finite_resolutions_and_spectral_sequence_finiteness.md)
23. [附录 N：Fine sheaf 与 Dolbeault resolution 细节](N_fine_sheaves_and_dolbeault_resolution_details.md)
24. [附录 O：从向量丛对偶到相干层 Ext-Serre 形式](O_vector_bundle_to_coherent_serre_formal_reduction.md)
25. [附录 P：特征类、Chern character 与 Riemann-Roch 的形式代数](P_characteristic_classes_and_riemann_roch_algebra.md)
26. [附录 Q：GAGA 的 properness、反例与导出比较细节](Q_gaga_properness_and_derived_comparison_details.md)
27. [附录 R：Dolbeault 局部正合的解析骨架](R_dolbeault_local_poincare_details.md)
28. [附录 S：射影空间上线丛上同调的单项式计算](S_projective_space_cohomology_bott_basic.md)
29. [附录 T：射影空间上线丛的 Serre 对偶](T_projective_space_serre_duality.md)
30. [附录 U：射影空间线丛的 Hirzebruch-Riemann-Roch](U_hrr_for_projective_space_line_bundles.md)
31. [附录 V：Stein、Cartan 定理与相干层工具](V_stein_cartan_and_coherent_sheaf_tools.md)
32. [附录 W：正则局部环与相干层有限分解](W_regular_local_rings_and_coherent_resolutions.md)
33. [附录 X：从向量丛情形传播相干上同调有限性](X_finiteness_from_vector_bundle_cases.md)
34. [附录 Y：Projective GAGA 的证明结构](Y_projective_gaga_proof_architecture.md)
35. [附录 Z：椭圆复形与 Hodge 定理接口](Z_elliptic_complexes_and_hodge_theorem.md)
36. [附录 AA：由 Hodge 理论推出向量丛 Serre 对偶](AA_vector_bundle_serre_duality_from_hodge.md)
37. [附录 AB：Cartan A/B 的证明模块](AB_cartan_theorems_proof_modules.md)
38. [附录 AC：Grauert 直接像定理与有限性](AC_grauert_direct_image_and_finiteness.md)
39. [附录 AD：Dualizing complex 与一般 Serre 对偶](AD_dualizing_complex_and_general_serre_duality.md)
40. [附录 AE：Grothendieck-Riemann-Roch 的一般形式](AE_grothendieck_riemann_roch_general_formalism.md)
41. [附录 AF：Weierstrass 与 Oka coherence 的局部代数](AF_weierstrass_oka_coherence_local_algebra.md)
42. [附录 AG：Runge、Cousin 与 Cartan B 的机制](AG_runge_cousin_and_cartan_b_mechanism.md)
43. [附录 AH：Hörmander L2 方法与 Stein 消没](AH_hormander_l2_and_stein_vanishing.md)
44. [附录 AI：Projective GAGA 的 graded module 细节](AI_projective_gaga_graded_module_details.md)
45. [附录 AJ：Grothendieck duality 的构造义务](AJ_grothendieck_duality_construction_obligations.md)
46. [附录 AK：Deformation to the normal cone 与 GRR 证明模块](AK_deformation_to_normal_cone_and_grr_modules.md)
47. [附录 AL：Weierstrass 除法的估计形式](AL_weierstrass_division_estimates.md)
48. [附录 AM：Hörmander 基本估计与闭值域步骤](AM_hormander_basic_estimate_and_closed_range.md)
49. [附录 AN：Grauert 定理的 Banach 复形证明模块](AN_grauert_banach_complex_and_direct_images.md)
50. [附录 AO：形式函数、形式 GAGA 与代数化](AO_formal_functions_and_gaga_algebraization.md)
51. [附录 AP：GRR 的局部化与推前相容](AP_grr_localization_and_pushforward_compatibility.md)
52. [附录 AQ：复几何主定理包与凝聚闭包](AQ_main_theorem_package_and_condensed_closure.md)
53. [附录 AR：Clausen-Scholze 复几何核心定理图谱](AR_clausen_scholze_complex_geometry_core_theorem_atlas.md)

## 正文与附录分工

数字章承担连续理解所需的定义、形式证明和 worked examples：第一至三章给出复解析
局部模型、Fréchet/凝聚函数族、点层 resolution 与 fine Dolbeault 计算；第四至七章
证明有限性传播、链级 Serre 配对、derived GAGA 和 $K$-理论/HRR 形式后果；第八章以
六函子关系和开放问题收束。Grauert、一般 Serre duality、GAGA、HRR/GRR 及
Clausen--Scholze analytic 建模仍作为精确外部输入，不在正文中压缩成伪证明。

附录保留技术展开与参考功能。正文中的 $\mathbb P^1$、Čech、Dolbeault、有限性、
对偶、GAGA 和特征类计算分别回指附录 H--P；附录 Q--AA 记录 properness、射影空间、
Stein/Cartan、局部 resolution、椭圆 Hodge 与射影 GAGA 的完整版本；附录 AB--AP
保存深层经典输入的证明模块与构造义务；附录 AQ--AR 集中登记主定理包和
Clausen--Scholze 定理图谱。读者可以先沿数字章掌握论证，再在相应附录核对更细的
符号、边界和来源。

工具卷见 [凝聚数学讲义：第四卷](../volume-4/)。

## 资料

资料源见 [SOURCES.md](SOURCES.md)，符号约定见 [NOTATION.md](NOTATION.md)，审查记录见 [MATH_REVIEW.md](MATH_REVIEW.md)，分卷答案见 [SOLUTIONS.md](SOLUTIONS.md)。
