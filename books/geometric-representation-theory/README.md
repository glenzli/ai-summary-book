# Geometric Representation Theory：几何、层与表示

作者：Dr. Stochastic Parrot

状态：连续教材稿；大型结论按外部输入边界使用

核查日期：2026-07-15

主资料源：Borel, Springer, Kazhdan-Lusztig, Beilinson-Bernstein-Deligne, Brylinski-Kashiwara, Beilinson-Bernstein, Ginzburg, Chriss-Ginzburg, Kashiwara-Schapira, Hotta-Takeuchi-Tanisaki, Mirkovic-Vilonen, Lusztig, Braverman-Finkelberg-Nakajima, Gaitsgory-Raskin and collaborators

本书目标是写成一部严格的中文 Geometric Representation Theory 教材，而不是主题导览。正文从 reductive algebraic groups、flag varieties、weights、category $\mathcal O$ 和 equivariant sheaves 开始，逐步进入 Schubert 几何、Hecke categories、Kazhdan-Lusztig 理论、Springer correspondence、D-modules、Beilinson-Bernstein localization、geometric Satake、affine Grassmannian、quiver varieties、categorification、Coulomb branches、symplectic duality 和 geometric Langlands 的研究边界。

## 写作约束

本书写作约束见 [SKILL.md](SKILL.md)。后续扩写必须遵守：

- 定义先于直觉，非平凡命题带证明或明确的外部输入标记。
- 全书固定底域、系数域、层论模型和大小约定。
- 不把代数表示、Lie 代数表示、D-module、perverse sheaf、constructible complex 和 infinity category 混同。
- 卷积、基变换、Verdier duality、t-structure、localization 和 Satake 等价必须写出类型和假设。
- 资料源必须能在 [SOURCES.md](SOURCES.md) 中追溯。

符号约定见 [NOTATION.md](NOTATION.md)，当前审查记录见 [MATH_REVIEW.md](MATH_REVIEW.md)，内部和外部定理分类见 [THEOREM_LEDGER.md](THEOREM_LEDGER.md)。

## 全书目录

### 第一部分：代数群、表示和 flag 几何

1. [序章：范围、严格性标准和资料源](00_preface_and_scope.md)
2. [第一章：Reductive groups、flag varieties 与权格](01_reductive_groups_flag_varieties_and_weights.md)
3. [第二章：Lie 代数表示、category O 与中心 character](02_representations_category_o_and_harish_chandra_modules.md)
4. [第三章：Equivariant sheaves、六函子与 perverse t-structure](03_equivariant_sheaves_six_functors_and_perversity.md)
5. [第四章：Schubert 几何、Hecke categories 与 Kazhdan-Lusztig 基](04_schubert_geometry_hecke_categories_and_kazhdan_lusztig.md)
6. [第五章：Springer resolution、Steinberg variety 与 Weyl group action](05_springer_resolution_steinberg_and_weyl_action.md)
7. [第六章：Nilpotent orbits、generalized Springer correspondence 与 character sheaves](06_nilpotent_orbits_generalized_springer_and_character_sheaves.md)

### 第二部分：D-modules、localization 和经典几何表示论

8. [第七章：D-modules、Riemann-Hilbert 对应与 regular holonomic 条件](07_d_modules_riemann_hilbert_and_regular_holonomic.md)
9. [第八章：Beilinson-Bernstein localization 与 category O 的几何化](08_beilinson_bernstein_localization_and_category_o.md)
10. [第九章：Borel-Weil-Bott、translation functors 与 wall crossing](09_borel_weil_bott_translation_and_wall_crossing.md)
11. [第十章：Harish-Chandra bimodules、primitive ideals 与 characteristic cycles](10_harish_chandra_bimodules_primitive_ideals_and_characteristic_cycles.md)
12. [第十一章：Soergel bimodules、Hodge theory 与 Hecke categorification](11_soergel_bimodules_hodge_theory_and_hecke_categorification.md)

### 第三部分：仿射与 Langlands 方向

13. [第十二章：Affine Grassmannian、loop groups 与 convolution](12_affine_grassmannian_loop_groups_and_convolution.md)
14. [第十三章：Geometric Satake 等价与 Tannakian reconstruction](13_geometric_satake_and_tannakian_reconstruction.md)
15. [第十四章：Affine flag varieties、Iwahori-Hecke categories 与 affine Kazhdan-Lusztig theory](14_affine_flag_iwahori_hecke_and_affine_kazhdan_lusztig.md)
16. [第十五章：Kac-Moody localization、chiral algebras 与 factorization categories](15_kac_moody_localization_chiral_and_factorization_categories.md)
17. [第十六章：Geometric Langlands 的局部和全局接口](16_geometric_langlands_local_global_interface.md)

### 第四部分：辛几何、量子化和范畴化

18. [第十七章：Quiver varieties 与 Nakajima 表示构造](17_quiver_varieties_and_nakajima_representations.md)
19. [第十八章：KLR/Rouquier 代数、canonical bases 与 categorification](18_klr_rouquier_categorification_and_canonical_bases.md)
20. [第十九章：Symplectic resolutions、category O 与 symplectic duality](19_symplectic_resolutions_category_o_and_symplectic_duality.md)
21. [第二十章：Coulomb branches、BFN 构造与量子化](20_coulomb_branches_bfn_construction_and_quantization.md)
22. [第二十一章：Hall algebras、cohomological Hall algebras 与 Donaldson-Thomas 接口](21_hall_coha_and_donaldson_thomas_interfaces.md)
23. [第二十二章：Quantum groups、crystals 与 canonical bases 的几何模型](22_quantum_groups_crystals_and_canonical_bases_geometric_models.md)
24. [第二十三章：研究边界与开放障碍](23_research_frontier_2026_and_open_problem_map.md)

### 附录

- [附录 A：代数几何、导出范畴和商栈基础约定](A_foundations_algebraic_geometry_and_stacks.md)
- [附录 B：Coxeter groups、root data 与 Bruhat order](B_coxeter_root_data_and_bruhat_order.md)
- [附录 C：六函子、Verdier duality 和 perverse sheaves 技术细节](C_six_functors_perverse_and_ic_technicalities.md)
- [附录 D：资料源定理索引与 theorem locator](D_source_theorem_index.md)
- [附录 E：D-module convention、left/right 转换和 twist](E_d_module_conventions_and_twists.md)
- [附录 F：卷积 correspondence、proper base change 和 associativity 检查](F_convolution_correspondences_and_associativity.md)
- [附录 G：低秩例子：$SL_2$、$SL_3$、Springer fibers 和 Schubert singularities](G_low_rank_examples_sl2_sl3_springer_and_schubert.md)
- [附录 H：Kazhdan-Lusztig 多项式和 Soergel bimodule 计算](H_kazhdan_lusztig_and_soergel_computations.md)
- [附录 I：Geometric Satake 的 Tannakian 细节](I_geometric_satake_tannakian_details.md)
- [附录 J：前沿结果进入正文的验证流程](J_frontier_result_entry_protocol.md)

## 阅读与参考文件

- [SKILL.md](SKILL.md)：本教材的写作约束与 OET 边界。
- [NOTATION.md](NOTATION.md)、[TERMINOLOGY.md](TERMINOLOGY.md)：全书符号、大小约定与术语表。
- [INDEX.md](INDEX.md)、[SYMBOL_INDEX.md](SYMBOL_INDEX.md)、[EXAMPLE_INDEX.md](EXAMPLE_INDEX.md)：术语、符号与低秩计算入口。
- [EXERCISE_SOLUTIONS.md](EXERCISE_SOLUTIONS.md)：各章习题答案与提示。
- [SOURCES.md](SOURCES.md)、[D_source_theorem_index.md](D_source_theorem_index.md)、[P0_REFERENCE_LOCATORS_BATCH_1.md](P0_REFERENCE_LOCATORS_BATCH_1.md)：外部输入的资料源与 locator。
- [THEOREM_LEDGER.md](THEOREM_LEDGER.md)、[DEPENDENCY_GRAPH.md](DEPENDENCY_GRAPH.md)、[MODEL_HYPOTHESES_MATRIX.md](MODEL_HYPOTHESES_MATRIX.md)：定理分类、依赖方向与模型假设。
- [K_internal_proof_kernels_foundations.md](K_internal_proof_kernels_foundations.md)、[L_low_rank_and_calculation_kernels.md](L_low_rank_and_calculation_kernels.md)：内部证明核与低阶计算核；这些文件保持账本式结构。
- [MATH_REVIEW.md](MATH_REVIEW.md)、[CHAPTER_COMPLETENESS_AUDIT.md](CHAPTER_COMPLETENESS_AUDIT.md)、[CONTENT_CLOSURE_AUDIT.md](CONTENT_CLOSURE_AUDIT.md)、[FORMAL_COMPLETION_MATRIX.md](FORMAL_COMPLETION_MATRIX.md)：数学复核与维护记录。
- [PUBLISHING_STYLE.md](PUBLISHING_STYLE.md)：排版与阅读样式。
- [FRONTIER_SOURCE_AUDIT_2026_07_08.md](FRONTIER_SOURCE_AUDIT_2026_07_08.md)、[J_frontier_result_entry_protocol.md](J_frontier_result_entry_protocol.md)：前沿资料核验记录与维护流程；该流程不属于第二十三章的数学叙事。

## 阅读路径与研究边界

第一至六章建立有限维旗簇、层与 Springer 主线；第七至十一章进入 localization、characteristic cycles 与 Soergel 范畴化；第十二至十六章处理 affine Grassmannian、Satake 和 geometric Langlands 接口；第十七至二十二章以 quiver varieties、KLR、辛分辨率、BFN、Hall/CoHA 和 crystals 形成第二条连续主线。第二十三章不罗列时效性结果，而用局部到全局、Poisson 到辛几何、模系数 torsion、decategorification 与 derived models 五类障碍收束全书。所有大型结论仍须按正文中的“外部输入定理”标记及定理账本使用。
