# 凝聚数学讲义：第三卷

作者：Dr. Stochastic Parrot  
状态：应用导读草稿
副题：复几何与相干对偶

本卷接续 [第一卷](../volume-1/) 与 [第二卷](../volume-2/)。第一卷建立凝聚基础，第二卷建立 solid、analytic、liquid 和 $f_!$ 的范畴语言；第三卷把这些工具放入复几何。

## 建议阅读顺序

1. [序章：第三卷的目标](00_preface_and_goals.md)
2. [第一章：复解析空间的凝聚语言](01_complex_analytic_spaces_condensed_language.md)
3. [第二章：相干层与导出范畴](02_coherent_sheaves_and_derived_categories.md)
4. [第三章：Dolbeault 复形与 liquid 模](03_dolbeault_complexes_and_liquid_modules.md)
5. [第四章：相干上同调有限性](04_finiteness_of_coherent_cohomology.md)
6. [第五章：Serre 对偶](05_serre_duality.md)
7. [第六章：GAGA](06_gaga.md)
8. [第七章：Grothendieck-Hirzebruch-Riemann-Roch](07_hirzebruch_riemann_roch.md)
9. [第八章：六函子形式与后续方向](08_six_functors_and_outlook.md)
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

## 当前范围

本卷给出复几何应用的应用导读草稿：它解释 compact complex manifolds、coherent analytic sheaves、Dolbeault 复形、Serre duality、GAGA 和 Riemann-Roch 在 condensed/analytic 语言中的位置。深层定理以 Clausen-Scholze 输入定理标注；本卷给出证明策略、范畴翻译、术语对照、局部计算模型和基础例子。附录 F-G 将经典输入定理和依赖链拆细，附录 H 给出 $\mathbb P^1$ 上线丛上同调的完整 Čech 计算，附录 I 给出 Čech-to-derived 谱序列和超上同调计算的同调代数证明，附录 J-K 补 Serre 对偶、GAGA 和 Riemann-Roch 在接受输入定理后的形式证明层，附录 L 补 Fredholm-Hodge 有限性的形式证明层，附录 M 补有限分解和谱序列传播有限性的严格边界，附录 N 补 fine sheaf、acyclic resolution 和 Dolbeault cohomology 计算的形式证明，附录 O 补从向量丛 Serre 对偶到相干层 Ext 形式的条件性同调代数推导，附录 P 补 Chern character、Todd class 和 Riemann-Roch 的形式代数，附录 Q 补 GAGA properness 反例和导出比较细节，附录 R 补 Dolbeault 局部正合的解析骨架，附录 S 补 $\mathbb P^n$ 上 $\mathcal O(d)$ 的 Čech 单项式计算，附录 T-U 补射影空间线丛的 Serre 对偶和 HRR 公式证明，避免把复几何深定理伪装成书内已证结论。

工具卷见 [凝聚数学讲义：第四卷](../volume-4/)。

## 资料

资料源见 [SOURCES.md](SOURCES.md)，符号约定见 [NOTATION.md](NOTATION.md)，审查记录见 [MATH_REVIEW.md](MATH_REVIEW.md)。
