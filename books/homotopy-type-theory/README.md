# 同伦类型论与单值基础

作者：Dr. Stochastic Parrot
状态：连续教材修订版
最近资料核查：2026-07-15
主资料源：*Homotopy Type Theory: Univalent Foundations of Mathematics*；Egbert Rijke, *Introduction to Homotopy Type Theory*；cubical type theory、simplicial model、单值范畴论、合成同伦论和经典代数拓扑文献。

这是一本中文 HoTT 教材。目标不是科普介绍，而是按严格教材方式，从依赖类型论的判断规则、恒等类型和路径代数开始，进入等价、函数外延性、单值性、高阶归纳类型、截断、同伦层级、合成同伦论、单值范畴论和当前研究边界。

本书采用严格教材口径：每个核心定义必须有规则或精确定义；每个非平凡断言必须具有书内定理、条件化推导、精确外部输入或研究边界之一的身份；解释证明思路的文字本身不算证明。

资料源见 [SOURCES.md](SOURCES.md)，符号约定见 [NOTATION.md](NOTATION.md)，证明身份规则见 [B_proof_status_blueprints.md](B_proof_status_blueprints.md)，关键依赖边界见 [K_remaining_obligations.md](K_remaining_obligations.md)，出版审计见 [PUBLICATION_CLOSURE_AUDIT.md](PUBLICATION_CLOSURE_AUDIT.md)。

本地验收入口为 [`validate.py`](validate.py)；它检查第 0-17 章导言、旧模板回归、Banach/Rezk 关键边界，并调用严格叙事审计、严格 OET 审计与 `git diff --check`。

## 建议阅读顺序

1. [序章：范围、验证等级与路线](00_preface_and_scope.md)
2. [第一章：依赖类型论的判断与结构规则](01_dependent_type_theory_and_judgments.md)
3. [第二章：恒等类型、路径归纳与路径代数](02_identity_types_and_paths.md)
4. [第三章：基础归纳类型与命题作为类型](03_basic_inductive_types.md)
5. [第四章：可收缩性、命题、集合与同伦层级](04_contractibility_and_hlevels.md)
6. [第五章：Fiber、等价与等价的等价定义](05_equivalences_and_fibers.md)
7. [第六章：函数外延性、命题外延性与单值性](06_function_extensionality_and_univalence.md)
8. [第七章：单值性的基本后果](07_univalence_consequences.md)
9. [第八章：截断、商类型与集合层数学](08_truncations_sets_quotients.md)
10. [第九章：高阶归纳类型的规则格式](09_higher_inductive_types.md)
11. [第十章：圆、悬挂、Pushout 与同伦余极限](10_circle_suspension_pushouts.md)
12. [第十一章：基本群、覆盖空间与圆的计算](11_fundamental_group_and_coverings.md)
13. [第十二章：从环路到稳定现象](12_synthetic_homotopy_and_advanced_interfaces.md)
14. [第十三章：预范畴、单值范畴与结构等同](13_univalent_categories.md)
15. [第十四章：Yoneda、极限、伴随与 Rezk 完备化](14_yoneda_limits_adjunctions_rezk.md)
16. [第十五章：模型语义、可靠性与相对一致性](15_models_sources_and_boundaries.md)
17. [第十六章：Cubical Type Theory、计算单值性与模型](16_cubical_type_theory_and_models.md)
18. [第十七章：研究边界中的语言与定理](17_research_frontier_and_open_problems.md)

## 核心附录

- [附录 A：路径代数参考表](A_path_algebra_reference.md)
- [附录 B：证明身份与使用规则](B_proof_status_blueprints.md)
- [附录 C：练习提示与解题路线](C_exercise_hints_and_solutions.md)
- [附录 D：基础证明核](D_foundational_proof_kernel.md)
- [附录 E：等价证明核](E_equivalence_proof_kernel.md)
- [附录 F：外延性与截断证明核](F_extensionality_truncation_kernel.md)
- [附录 G：等价定义与同伦层级证明核](G_equivalence_definitions_hlevels.md)
- [附录 H：布尔类型与 Universe 非集合性](H_bool_universe_not_set.md)
- [附录 I：结构等同性原则证明核](I_structure_identity_principle.md)
- [附录 J：一元代数签名的结构等同性](J_algebraic_signature_sip.md)
- [附录 K：关键依赖与不可逆边界](K_remaining_obligations.md)
- [附录 L：高阶归纳类型输入规则表](L_HIT_input_rules.md)
- [附录 M：整数对象与 Successor 等价](M_integers_and_successor.md)
- [附录 N：圆的 Encode-Decode 证明核](N_circle_encode_decode.md)
- [附录 O：同伦层级性质的命题性](O_hlevel_property_kernel.md)
- [附录 P：预范畴与单值范畴证明核](P_univalent_category_kernel.md)
- [附录 Q：Yoneda 引理证明核](Q_yoneda_kernel.md)
- [附录 R：Rezk 完备化的构造输入](R_rezk_completion_input.md)
- [附录 S：来源定位索引](S_source_locator_index.md)
- [附录 T：单值性推出函数外延性的外部输入](T_univalence_funext_external_input.md)
- [附录 U：预层范畴与 Yoneda 嵌入](U_presheaf_category_yoneda_embedding.md)
- [附录 V：圆的基本群同构](V_circle_fundamental_group_isomorphism.md)
- [附录 W：整数加法群律证明核](W_integer_addition_group_laws.md)
- [附录 X：函子范畴、自然同构与单值性](X_functor_categories_and_univalence.md)
- [附录 Y：合成上同调证明核与高级输入](Y_synthetic_cohomology_kernel.md)
- [附录 Z：Cubical 与 HIT 元理论边界](Z_cubical_hit_metatheory_boundary.md)

## 高级接口附录

- [附录 AA：Weak equivalence 与 Rezk 泛性质的外部输入](AA_rezk_universal_property_schema.md)
- [附录 AB：同伦层级向上闭包证明核](AB_hlevel_upward_closure.md)
- [附录 AC：Eckmann-Hilton 与高阶同伦群交换性](AC_eckmann_hilton_and_higher_homotopy.md)
- [附录 AD：二点类型悬挂与圆的等价](AD_suspension_bool_circle.md)
- [附录 AE：自然数与和类型的离散性证明核](AE_discrete_natural_numbers_and_coproducts.md)
- [附录 AF：终对象唯一性与伴随形式证明核](AF_limits_and_adjunctions_kernel.md)
- [附录 AG：结构 Transport 与代数 SIP 证明核](AG_structure_transport_and_sip.md)
- [附录 AH：Full Subcategory 与本质像证明核](AH_full_subcategories_and_essential_images.md)
- [附录 AI：Pushout 的等价不变性证明核](AI_pushout_equivalence_invariance.md)
- [附录 AJ：模态、局部化与正交分解系统](AJ_modalities_localization_and_factorization.md)
- [附录 AK：Cauchy 实数与构造性分析证明核](AK_cauchy_reals_and_constructive_analysis.md)
- [附录 AL：Blakers-Massey、Freudenthal 与 Hopf Fibration](AL_synthetic_homotopy_core_theorems.md)
- [附录 AM：Smash Product、对称幺半结构与上同调运算](AM_smash_products_and_cohomology_operations.md)
- [附录 AN：Directed / Simplicial Type Theory 与高阶范畴接口](AN_directed_and_simplicial_type_theory.md)
- [附录 AO：Cubical 模型、弱单值性与 2026 边界](AO_cubical_models_and_2026_boundaries.md)
- [附录 AP：Fiber Sequence 与同伦群长正合列](AP_fiber_sequences_and_long_exact_sequence.md)
- [附录 AQ：Exact Couples、谱序列与收敛接口](AQ_exact_couples_and_spectral_sequences.md)
- [附录 AR：Cauchy 实数的环、序与完备有序域接口](AR_cauchy_real_field_and_order_kernel.md)
- [附录 AS：Directed / Simplicial Type Theory 的规则核](AS_directed_simplicial_rule_kernel.md)
- [附录 AT：Left Exact 模态、Cohesive HoTT 与使用边界](AT_lex_modalities_and_cohesive_hott.md)
- [附录 AU：Join Connectivity、Flattening Lemma 与 Blakers-Massey 证明接口](AU_join_connectivity_and_flattening.md)
- [附录 AV：Serre、Atiyah-Hirzebruch 与 Adams 谱序列接口](AV_serre_ahss_adams_spectral_sequences.md)
- [附录 AW：Dedekind 实数、Locatedness 与 Cauchy 实数比较](AW_dedekind_reals_and_cauchy_comparison.md)
- [附录 AX：Directed / Simplicial Type Theory 的语义接口](AX_directed_semantics_interface.md)
- [附录 AY：Pushout path-code 的内部接口与外部边界](AY_pushout_path_encode_decode_kernel.md)
- [附录 AZ：谱、稳定范畴与收敛证明接口](AZ_spectra_stable_category_and_convergence.md)
- [附录 BA：构造性分析中的连续性、紧致性与典型定理](BA_constructive_analysis_continuity_compactness.md)
- [附录 BB：Rezk 类型、Complete Segal 对象与合成无穷范畴](BB_rezk_types_and_synthetic_infinity_categories.md)
- [附录 BC：HIIT、QIIT 与计算 HIT 语义](BC_hiit_qiit_and_computational_hit_semantics.md)
- [附录 BD：Cohesive HoTT、合成微分几何与 Zariski 接口](BD_cohesive_sdg_and_zariski_hott.md)
- [附录 BE：Displayed Categories、Bicategories 与高阶单值性](BE_displayed_categories_bicategories_and_univalence.md)
- [附录 BF：Higher Groups、Deloopings 与 Classifying Types](BF_higher_groups_deloopings_and_classifying_types.md)
- [附录 BG：Two-Level Type Theory、Strict Equality 与半单纯形](BG_two_level_type_theory_and_strict_equality.md)
- [附录 BH：集合层代数层级、商结构与局部化](BH_set_level_algebra_hierarchy_and_localization.md)
- [附录 BI：有限集、基数、序数与选择原则](BI_finite_sets_cardinals_ordinals_and_choice.md)
- [附录 BJ：Postnikov Towers、Whitehead 定理与障碍理论接口](BJ_postnikov_whitehead_and_obstruction_theory.md)
- [附录 BK：Cofiber、Puppe 序列与 Mayer-Vietoris](BK_cofiber_puppe_and_mayer_vietoris.md)
- [附录 BL：逻辑原则、Resizing、选择与构造性边界](BL_logic_resizing_choice_and_constructivity.md)
- [附录 BM：局部系数、扭曲上同调与 Postnikov 系数系统](BM_local_coefficients_and_twisted_cohomology.md)
- [附录 BN：Steenrod 代数、Ext 与 Adams 计算接口](BN_steenrod_algebra_ext_and_adams_calculations.md)
- [附录 BO：构造性度量空间、级数与积分](BO_constructive_metric_spaces_series_and_integration.md)

## 收口文件

- [收口范围与封稿门槛](CLOSURE_SCOPE.md)
- [依赖分层与外部输入边界](DEPENDENCY_LAYERS.md)
- [数学审查记录](MATH_REVIEW.md)
- [出版收口审计](PUBLICATION_CLOSURE_AUDIT.md)
- [写作约束](SKILL.md)
- [资料源](SOURCES.md)
- [符号表](NOTATION.md)

## 当前范围

按文本教材口径，L0-L5 的内部语言、等价、单值性、HIT、圆的基本群、单值范畴论与 Yoneda 均有书内证明核或明确公理输入；Rezk 完备化的构造在书内完成，其限制函子泛性质精确采用外部定理。L6-L9 的构造性分析、稳定同伦论、directed/cohesive/2LTT、高阶 Rezk/Segal 相干和模型论按各自假设作为高级接口或研究边界保留。
