# Langlands 纲领：从 `GL(1)` 到几何 Langlands

作者：Dr. Stochastic Parrot
状态：严格教材，审定前闭合版，待出版级审定
主资料源：Tate, Weil, Langlands, Gelbart, Bump, Goldfeld-Hundley, Jacquet-Langlands, Godement-Jacquet, Arthur, Milne, Serre, Silverman, Diamond-Shurman, Cornell-Silverman-Stevens, Bushnell-Henniart, Frenkel, Gaitsgory

本书目标是写成一部数学化、专业化、成体系的 Langlands 纲领教材，而不是导览文章。正文从整体域、局部域、adeles 和 Tate thesis 开始，逐步进入类域论、`GL(1)` Langlands、模形式和椭圆曲线、`GL(n)` 自守表示、Galois 表示、L 群、函子性和几何 Langlands。费马大定理的证明作为单独应用章处理。

## 写作约束

本书写作约束见 [SKILL.md](SKILL.md)。后续写作必须遵守：

- 定义先于直觉，命题带证明或明确的外部输入标记。
- 全书固定局部/整体域、赋值、Haar 测度和表示论符号。
- 所有 L 函数必须说明局部因子、Euler 乘积区域、解析延拓和函数方程来源。
- Langlands 对应必须写成 Galois/Weil 参数、自守表示和 L 群同态之间的结构性陈述。
- 资料源必须能在 [SOURCES.md](SOURCES.md) 中追溯。
- 外部定理按“核心结构、支撑接口、卫星理论”分级；只有直接服务 Langlands 对象、参数、L 因子、Hecke 作用、局部-整体相容或应用链闭环的内容才在本书展开，其他深层理论保留为外部输入或另卷。

符号约定见 [NOTATION.md](NOTATION.md)，归一化总表见 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md)，概念审定见 [CONCEPTUAL_AUDIT.md](CONCEPTUAL_AUDIT.md)，数学审查记录见 [MATH_REVIEW.md](MATH_REVIEW.md)，逐章收口台账见 [CHAPTER_CLOSURE_AUDIT.md](CHAPTER_CLOSURE_AUDIT.md)，编号审计见 [NUMBERING_CROSSREF_AUDIT.md](NUMBERING_CROSSREF_AUDIT.md)，收口标准见 [CLOSURE_STATUS.md](CLOSURE_STATUS.md)。主要定理状态见 [THEOREM_INDEX.md](THEOREM_INDEX.md)，章节依赖见 [DEPENDENCY_GRAPH.md](DEPENDENCY_GRAPH.md)，主线最短证明链见 [MAINLINE_PROOF_CHAINS.md](MAINLINE_PROOF_CHAINS.md)，核心习题解答见 [SOLUTIONS.md](SOLUTIONS.md)，习题覆盖审查见 [EXERCISE_COVERAGE.md](EXERCISE_COVERAGE.md)。

## 当前收口判定

本书已经达到主线广度基本固定：`GL(1)`、`GL(2)` 与费马应用、一般算术 Langlands、几何 Langlands 四条路径均已建立，附录 A-AE 已承担主要支撑接口。

本书已经进入审定前闭合版。第一轮收口完成高风险主章的归一化回指，并建立主线最短证明链和习题覆盖表；第二轮收口完成索引一致性审计和高风险附录的归一化回指；第三轮收口完成逐章风险清理并建立收口缺口台账；第四轮收口完成重点外部输入来源拆分；第五轮收口完成编号和交叉引用审计；第六轮补入第 3、7、10、14、16、19、22、90 章的接口检查表和最小模型说明；第七轮补入第 1、2、5、8、12、17 章的使用边界表并完成附录层精校状态审稿；第八轮修正 $\ell$-adic 记法、旧状态措辞和若干接口表述；第九轮收紧主体章节的高风险假设、归一化和版本选择；第十轮收紧主体证明链直接引用的附录接口，并抽查外部输入索引和资料源索引；第十一轮统一最终收口口径；第十二轮完成最终概念审定；第十三轮完成出版前文字、排版和局部数学口径维护。当前剩余工作不应继续横向扩张新理论分支，而应集中在来源页码、排版审稿和终校索引维护。判据和任务分级见 [CLOSURE_STATUS.md](CLOSURE_STATUS.md)。

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
- [附录 J：Newforms、Atkin-Lehner 理论和局部 Newvectors](J_newforms_atkin_lehner_and_local_newvectors.md)
- [附录 K：Galois 变形、Selmer 群和 Taylor-Wiles 接口](K_galois_deformation_and_taylor_wiles_interface.md)
- [附录 L：Eisenstein Series、常数项和残余谱](L_eisenstein_series_constant_terms_and_residual_spectrum.md)
- [附录 M：Langlands-Shahidi 局部系数、Gamma 因子和局部 L 因子](M_langlands_shahidi_gamma_factors_and_local_coefficients.md)
- [附录 N：局部 Packets、Endoscopy 和内形式例子](N_local_packets_endoscopy_and_inner_forms_examples.md)
- [附录 O：几何 Langlands 的 D-modules、IndCoh 和奇异支撑](O_geometric_langlands_d_modules_indcoh_and_singular_support.md)
- [附录 P：球 Hecke 代数、Cartan 分解和 Satake 同构](P_spherical_hecke_algebras_and_satake_isomorphism.md)
- [附录 Q：Bernstein-Zelevinsky 理论、Langlands 商和 `GL(n)` 局部分类](Q_bernstein_zelevinsky_langlands_classification_gl_n.md)
- [附录 R：Trace Formula 的项、截断、稳定化和应用接口](R_trace_formula_terms_stabilization_and_applications.md)
- [附录 S：函数域、Shtukas、Excursion Operators 和 Lafforgue 接口](S_function_field_shtukas_and_lafforgue_interface.md)
- [附录 T：模曲线上同调、Eichler-Shimura 和 Deligne 表示](T_modular_curves_eichler_shimura_and_deligne_representations.md)
- [附录 U：p-adic Hodge、Shimura Varieties 和 Cohomological Automorphic Galois 表示](U_p_adic_hodge_shimura_and_cohomological_automorphic_galois_representations.md)
- [附录 V：Class Formations、Artin Reciprocity 和导子接口](V_class_formations_artin_reciprocity_and_conductors.md)
- [附录 W：模曲线、Hecke Correspondences 和 Atkin-Lehner-Li 理论接口](W_modular_curves_hecke_correspondences_atkin_lehner_li_theory.md)
- [附录 X：Arthur 分类、Classical Groups 和 Mok 的 Unitary Groups 接口](X_arthur_classification_classical_groups_and_mok_unitary.md)
- [附录 Y：Factorization、Beilinson-Drinfeld Grassmannian 和几何 Satake 技术层](Y_factorization_bd_grassmannian_and_geometric_satake_technical_layer.md)
- [附录 Z：局部调和分析、Harish-Chandra Characters 和 Plancherel 接口](Z_local_harmonic_analysis_harish_chandra_plancherel.md)
- [附录 AA：Bruhat-Tits 建筑、Parahoric、Hyperspecial 和非分歧群](AA_bruhat_tits_buildings_parahoric_hyperspecial_unramified_groups.md)
- [附录 AB：Derived Stacks、QCoh/IndCoh 和 Six Functors 技术接口](AB_derived_stacks_qcoh_indcoh_and_six_functors.md)
- [附录 AC：Fargues-Fontaine 曲线、Diamonds、Local Shtukas 和几何局部 Langlands](AC_fargues_fontaine_curve_diamonds_local_shtukas_geometric_local_langlands.md)
- [附录 AD：椭圆曲线约化、Neron 模型、Kodaira 符号和 Tate Algorithm](AD_elliptic_curves_reduction_neron_kodaira_tate_algorithm.md)
- [附录 AE：`GL(2)` 局部 Langlands 的 Principal Series、Steinberg 和 Supercuspidal 例子](AE_local_gl2_principal_steinberg_supercuspidal_examples.md)

## 当前文件清单

- [SKILL.md](SKILL.md)：本教材的写作约束。
- [NOTATION.md](NOTATION.md)：全书符号和局部-整体约定。
- [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md)：Frobenius、reciprocity、Haar/Fourier、Satake、Galois 表示和 L 函数变量归一化总表。
- [CONCEPTUAL_AUDIT.md](CONCEPTUAL_AUDIT.md)：最终概念审定，固定参数、表示、L 函数、函子性、几何 Langlands 和费马应用的概念边界。
- [SOURCES.md](SOURCES.md)：主要资料源清单。
- [MATH_REVIEW.md](MATH_REVIEW.md)：审查清单和逐章收口状态。
- [CHAPTER_CLOSURE_AUDIT.md](CHAPTER_CLOSURE_AUDIT.md)：逐章收口缺口审查台账。
- [NUMBERING_CROSSREF_AUDIT.md](NUMBERING_CROSSREF_AUDIT.md)：编号一致性、习题解答回指和 Markdown 链接审计。
- [CLOSURE_STATUS.md](CLOSURE_STATUS.md)：收口标准、当前状态、准入规则和后置另卷清单。
- [MAINLINE_PROOF_CHAINS.md](MAINLINE_PROOF_CHAINS.md)：`GL(1)`、费马应用、一般算术 Langlands 和几何 Langlands 的最短证明链。
- [EXERCISE_COVERAGE.md](EXERCISE_COVERAGE.md)：四条主线的习题覆盖矩阵和收口用新增题目建议。
- [THEOREM_INDEX.md](THEOREM_INDEX.md)：全书主要定理、命题、外部输入和猜想的状态索引。
- [DEPENDENCY_GRAPH.md](DEPENDENCY_GRAPH.md)：章节依赖图、阅读路径和证明依赖层级。
- [SOLUTIONS.md](SOLUTIONS.md)：核心习题解答与提示。
- [00_preface_and_scope.md](00_preface_and_scope.md)：序章。
- [01_global_fields_and_adeles.md](01_global_fields_and_adeles.md)：第一章正文稿。
- [02_tate_thesis_and_gl1.md](02_tate_thesis_and_gl1.md)：第二章正文稿。
- [03_class_field_theory_as_gl1.md](03_class_field_theory_as_gl1.md)：第三章正文稿。
- [04_local_groups_haar_and_smooth_representations.md](04_local_groups_haar_and_smooth_representations.md)：第四章正文稿。
- [05_weil_groups_and_local_parameters.md](05_weil_groups_and_local_parameters.md)：第五章正文稿。
- [06_modular_forms_and_hecke_operators.md](06_modular_forms_and_hecke_operators.md)：第六章正文稿。
- [07_adelic_modular_forms_and_gl2.md](07_adelic_modular_forms_and_gl2.md)：第七章正文稿。
- [08_elliptic_curves_conductors_l_functions.md](08_elliptic_curves_conductors_l_functions.md)：第八章正文稿。
- [09_galois_representations_and_modularity.md](09_galois_representations_and_modularity.md)：第九章正文稿。
- [10_local_global_compatibility_and_level_lowering.md](10_local_global_compatibility_and_level_lowering.md)：第十章正文稿。
- [11_reductive_groups_dual_groups_l_groups.md](11_reductive_groups_dual_groups_l_groups.md)：第十一章正文稿。
- [12_local_langlands_conjecture.md](12_local_langlands_conjecture.md)：第十二章正文稿。
- [13_global_automorphic_representations_and_l_functions.md](13_global_automorphic_representations_and_l_functions.md)：第十三章正文稿。
- [14_gl_n_correspondence_and_known_theorems.md](14_gl_n_correspondence_and_known_theorems.md)：第十四章正文稿。
- [15_functoriality_principle.md](15_functoriality_principle.md)：第十五章正文稿。
- [16_trace_formula_and_endoscopy.md](16_trace_formula_and_endoscopy.md)：第十六章正文稿。
- [17_arthur_parameters_and_spectral_decomposition.md](17_arthur_parameters_and_spectral_decomposition.md)：第十七章正文稿。
- [18_curves_g_bundles_and_hecke_modifications.md](18_curves_g_bundles_and_hecke_modifications.md)：第十八章正文稿。
- [19_geometric_satake.md](19_geometric_satake.md)：第十九章正文稿。
- [20_hecke_eigensheaves.md](20_hecke_eigensheaves.md)：第二十章正文稿。
- [21_spectral_side_local_systems_and_categorical_correspondence.md](21_spectral_side_local_systems_and_categorical_correspondence.md)：第二十一章正文稿。
- [22_function_field_bridge_and_arithmetic_geometry.md](22_function_field_bridge_and_arithmetic_geometry.md)：第二十二章正文稿。
- [A_algebraic_number_theory_review.md](A_algebraic_number_theory_review.md)：附录 A，含乘积公式、素理想分解、分解群、惯性群、高阶分歧群、ray class、idele class 和导子补充。
- [B_locally_compact_groups_and_haar.md](B_locally_compact_groups_and_haar.md)：附录 B，含 Haar 测度、卷积、商测度和 restricted product 积分补充。
- [C_smooth_admissible_representations.md](C_smooth_admissible_representations.md)：附录 C，含 Hecke 作用、Schur 引理、smooth dual 和可容许性补充。
- [D_modular_curves_and_dimension_formulas.md](D_modular_curves_and_dimension_formulas.md)：附录 D，含 $X_0(2)$ genus 计算细节和权 2 微分形式补充。
- [E_external_input_theorem_index.md](E_external_input_theorem_index.md)：附录 E，外部输入定理索引，已拆细 Frey、Satake、Arthur、几何 Satake 和 Fargues-Scholze 来源。
- [F_fourier_analysis_and_poisson.md](F_fourier_analysis_and_poisson.md)：附录 F，含 LCA Fourier 分析、有限 Abel 群 Fourier 反演、非 Archimedean 紧开陪集计算、$\mathbb A_\mathbb Q/\mathbb Q$ 基本域、adeles 自对偶、Poisson 求和和 Tate theta 恒等式接口。
- [G_root_data_and_dual_group_tables.md](G_root_data_and_dual_group_tables.md)：附录 G，含 root datum、dual group 和 L homomorphism 计算表。
- [H_hecke_double_cosets_and_adelic_comparison.md](H_hecke_double_cosets_and_adelic_comparison.md)：附录 H，含 Hecke 双陪集、Fourier 系数和经典-adelic Hecke 比较。
- [I_godement_jacquet_rankin_selberg_integrals.md](I_godement_jacquet_rankin_selberg_integrals.md)：附录 I，含 Godement-Jacquet、Rankin-Selberg 和 converse theorem 积分接口。
- [J_newforms_atkin_lehner_and_local_newvectors.md](J_newforms_atkin_lehner_and_local_newvectors.md)：附录 J，含 old/new 分解、Atkin-Lehner 算子、Casselman newvector 和导子接口。
- [K_galois_deformation_and_taylor_wiles_interface.md](K_galois_deformation_and_taylor_wiles_interface.md)：附录 K，含 Galois deformation、Selmer 群、$R=T$ 和 Taylor-Wiles patching 接口。
- [L_eisenstein_series_constant_terms_and_residual_spectrum.md](L_eisenstein_series_constant_terms_and_residual_spectrum.md)：附录 L，含 Eisenstein series、常数项、intertwining operators、残余谱和 Arthur 参数接口。
- [M_langlands_shahidi_gamma_factors_and_local_coefficients.md](M_langlands_shahidi_gamma_factors_and_local_coefficients.md)：附录 M，含 Langlands-Shahidi local coefficients、局部 $\gamma$ 因子、全局 Eisenstein 函数方程和函子性接口。
- [N_local_packets_endoscopy_and_inner_forms_examples.md](N_local_packets_endoscopy_and_inner_forms_examples.md)：附录 N，含 tori、$\operatorname{SL}_2$、Jacquet-Langlands、endoscopic transfer 和基本引理例子。
- [O_geometric_langlands_d_modules_indcoh_and_singular_support.md](O_geometric_langlands_d_modules_indcoh_and_singular_support.md)：附录 O，含 D-modules、六运算、QCoh/IndCoh、奇异支撑和范畴几何 Langlands 技术层。
- [P_spherical_hecke_algebras_and_satake_isomorphism.md](P_spherical_hecke_algebras_and_satake_isomorphism.md)：附录 P，含球 Hecke 代数、Cartan 分解、Satake 变换、非分歧表示和 `GL(n)` 显式公式。
- [Q_bernstein_zelevinsky_langlands_classification_gl_n.md](Q_bernstein_zelevinsky_langlands_classification_gl_n.md)：附录 Q，含 segments、multisegments、Langlands quotient、tempered/generic 分类和 `GL(n)` 局部因子相容。
- [R_trace_formula_terms_stabilization_and_applications.md](R_trace_formula_terms_stabilization_and_applications.md)：附录 R，含紧商核公式、Arthur truncation、几何侧、谱侧、invariant trace formula、稳定化和应用接口。
- [S_function_field_shtukas_and_lafforgue_interface.md](S_function_field_shtukas_and_lafforgue_interface.md)：附录 S，含函数域双商、Hecke correspondences、shtukas、Drinfeld/Lafforgue 定理、excursion operators 和几何桥梁。
- [T_modular_curves_eichler_shimura_and_deligne_representations.md](T_modular_curves_eichler_shimura_and_deligne_representations.md)：附录 T，含模曲线局部系统、Hecke correspondences、Eichler-Shimura、Deligne 表示、weight two 和 residual representations。
- [U_p_adic_hodge_shimura_and_cohomological_automorphic_galois_representations.md](U_p_adic_hodge_shimura_and_cohomological_automorphic_galois_representations.md)：附录 U，含 regular algebraic automorphic representations、Shimura varieties、p-adic Hodge theory、局部-整体相容和 automorphy lifting 接口。
- [V_class_formations_artin_reciprocity_and_conductors.md](V_class_formations_artin_reciprocity_and_conductors.md)：附录 V，含 class formations、局部/全局 Artin reciprocity、norm subgroup theorem、ray class fields 和 `GL(1)` Langlands 重述。
- [W_modular_curves_hecke_correspondences_atkin_lehner_li_theory.md](W_modular_curves_hecke_correspondences_atkin_lehner_li_theory.md)：附录 W，含模曲线代数化、权二微分、genus formula、Hecke correspondences、old/new 分解和 Atkin-Lehner signs。
- [X_arthur_classification_classical_groups_and_mok_unitary.md](X_arthur_classification_classical_groups_and_mok_unitary.md)：附录 X，含 classical groups、Arthur parameters、multiplicity formula、standard transfer、inner forms 和 Mok unitary groups 接口。
- [Y_factorization_bd_grassmannian_and_geometric_satake_technical_layer.md](Y_factorization_bd_grassmannian_and_geometric_satake_technical_layer.md)：附录 Y，含 Ran space、factorization、BD Grassmannian、fusion、几何 Satake 和 Hecke action 技术层。
- [Z_local_harmonic_analysis_harish_chandra_plancherel.md](Z_local_harmonic_analysis_harish_chandra_plancherel.md)：附录 Z，含 Harish-Chandra characters、temperedness、Plancherel、Bernstein center、Paley-Wiener 和局部字符展开接口。
- [AA_bruhat_tits_buildings_parahoric_hyperspecial_unramified_groups.md](AA_bruhat_tits_buildings_parahoric_hyperspecial_unramified_groups.md)：附录 AA，含 Bruhat-Tits building、parahoric group schemes、hyperspecial subgroups、Iwahori-Hecke 和 Moy-Prasad filtrations。
- [AB_derived_stacks_qcoh_indcoh_and_six_functors.md](AB_derived_stacks_qcoh_indcoh_and_six_functors.md)：附录 AB，含 derived stacks、cotangent complex、QCoh/IndCoh、singular support、six functors、kernel formalism 和 renormalized D-modules。
- [AC_fargues_fontaine_curve_diamonds_local_shtukas_geometric_local_langlands.md](AC_fargues_fontaine_curve_diamonds_local_shtukas_geometric_local_langlands.md)：附录 AC，含 perfectoid/diamonds、Fargues-Fontaine curve、$G$-bundles、local Shimura varieties、Fargues-Scholze 几何局部 Langlands。
- [AD_elliptic_curves_reduction_neron_kodaira_tate_algorithm.md](AD_elliptic_curves_reduction_neron_kodaira_tate_algorithm.md)：附录 AD，含 Neron models、Kodaira symbols、Tate algorithm、Ogg conductor formula、Tate curve 和 Frey 曲线局部导子。
- [AE_local_gl2_principal_steinberg_supercuspidal_examples.md](AE_local_gl2_principal_steinberg_supercuspidal_examples.md)：附录 AE，含 `GL(2)` principal series、Steinberg twists、supercuspidals、Weil-Deligne 参数和局部 L 因子。
- [90_fermat_last_theorem_application.md](90_fermat_last_theorem_application.md)：费马大定理应用章正文稿。

## 当前教材化补强层

- 全书主要结果已由 [THEOREM_INDEX.md](THEOREM_INDEX.md) 标记为 `P`、`S`、`E`、`C` 四类，分别对应已证、外部输入的证明路线、外部输入和猜想。
- 全书阅读路径已由 [DEPENDENCY_GRAPH.md](DEPENDENCY_GRAPH.md) 分成 `GL(1)`、费马应用、一般数论 Langlands 和几何 Langlands 四条路径。
- 核心习题解答已覆盖第 1 至 5 章、若干 `GL(2)` 计算、一般 Langlands 基础、几何 Langlands 入门和费马应用章。
- 附录 A-D、F-AE 已从接口复习扩展为带关键证明、Fourier/Poisson 计算、积分接口、谱分解接口、局部 packet 例子、几何范畴技术层、`GL(n)` 局部分类、`GL(2)` 局部 LLC 例子、trace formula 稳定化、函数域 shtuka 接口、模曲线上同调、p-adic Hodge/Shimura 接口、class formation、Atkin-Lehner-Li、Arthur 分类、factorization/BD Grassmannian、局部调和分析、Bruhat-Tits、derived stacks、Fargues-Fontaine 和椭圆曲线局部约化接口的参考附录；这些附录已达到主体可引用接口深度，但仍不是可替代专著的完整证明卷。

## 最终收口型审定结论

本书当前状态为审定前闭合版：四条主线、应用链、概念边界、外部输入边界、编号索引、交叉引用、习题回指和资料源大类均已闭合。后续进入出版前审定时，只应接受数学错误修正、来源补强、排版统一、术语统一和索引维护；新增大块理论、附录群或第五条主线应另列为新版本或另卷目标。
