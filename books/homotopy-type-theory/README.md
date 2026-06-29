# 同伦类型论与单值基础

作者：Dr. Stochastic Parrot  
状态：严格教材草稿，完整教材第一版  
最近资料核查：2026-06-29  
主资料源：*Homotopy Type Theory: Univalent Foundations of Mathematics*；Egbert Rijke, *Introduction to Homotopy Type Theory*；Coq-HoTT；UniMath；Cubical Agda；Agda 官方文档。

这是一本中文 HoTT 教材草稿。目标不是写“什么是 HoTT”的科普介绍，而是按严格教材方式，从依赖类型论的判断规则、恒等类型和路径代数开始，逐步进入等价、函数外延性、单值性、高阶归纳类型、截断、同伦层级、合成同伦论、单值范畴论、形式化数学库和当前研究边界。

## 写作约束

本书的写作约束见 [SKILL.md](SKILL.md)。后续扩写必须遵守：

- 每个核心定义必须给出类型论规则或精确定义。
- 每个非平凡命题必须有证明、证明说明、外部输入或机器形式化状态。
- 不把同伦直觉当作证明。
- 不偷用函数外延性、单值性、高阶归纳类型、截断或选择原则。
- 涉及最新研究、形式化库状态或软件版本时必须联网核查。

资料源见 [SOURCES.md](SOURCES.md)，符号约定见 [NOTATION.md](NOTATION.md)，数学审查记录见 [MATH_REVIEW.md](MATH_REVIEW.md)。

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
13. [第十二章：合成同伦论、谱序列入口与近期形式化](12_synthetic_homotopy_and_recent_formalization.md)
14. [第十三章：预范畴、单值范畴与结构等同](13_univalent_categories.md)
15. [第十四章：Yoneda、极限、伴随与 Rezk 完备化](14_yoneda_limits_adjunctions_rezk.md)
16. [第十五章：形式化库比较](15_formalization_libraries.md)
17. [第十六章：Cubical Type Theory、计算单值性与模型](16_cubical_type_theory_and_models.md)
18. [第十七章：研究边界、开放问题与版本化阅读](17_research_frontier_and_open_problems.md)
19. [附录 A：路径代数参考表](A_path_algebra_reference.md)
20. [附录 B：形式化蓝图](B_formalization_blueprints.md)
21. [附录 C：练习提示与解答草图](C_exercise_hints_and_solution_sketches.md)
22. [附录 D：基础证明核](D_foundational_proof_kernel.md)
23. [附录 E：等价证明核](E_equivalence_proof_kernel.md)
24. [附录 F：外延性与截断证明核](F_extensionality_truncation_kernel.md)
25. [附录 G：等价定义与同伦层级证明核](G_equivalence_definitions_hlevels.md)
26. [附录 H：布尔类型与 Universe 非集合性](H_bool_universe_not_set.md)
27. [附录 I：结构等同性原则证明核](I_structure_identity_principle.md)
28. [附录 J：一元代数签名的结构等同性](J_algebraic_signature_sip.md)
29. [附录 K：剩余证明义务登记](K_remaining_obligations.md)
30. [附录 L：高阶归纳类型输入规则表](L_HIT_input_rules.md)
31. [附录 M：整数对象与 Successor 等价](M_integers_and_successor.md)
32. [附录 N：圆的 Encode-Decode 证明核](N_circle_encode_decode.md)
33. [附录 O：同伦层级性质的命题性](O_hlevel_property_kernel.md)
34. [附录 P：预范畴与单值范畴证明核](P_univalent_category_kernel.md)
35. [附录 Q：Yoneda 引理证明核](Q_yoneda_kernel.md)
36. [附录 R：Rezk 完备化的构造输入](R_rezk_completion_input.md)
37. [附录 S：形式化库索引](S_formalization_library_index.md)
38. [附录 T：单值性推出函数外延性的形式化输入](T_univalence_funext_formal_input.md)
39. [附录 U：预层范畴与 Yoneda 嵌入](U_presheaf_category_yoneda_embedding.md)
40. [附录 V：圆的基本群同构](V_circle_fundamental_group_isomorphism.md)
41. [附录 W：整数加法群律证明核](W_integer_addition_group_laws.md)
42. [附录 X：函子范畴、自然同构与单值性](X_functor_categories_and_univalence.md)
43. [附录 Y：合成上同调证明核与形式化入口](Y_synthetic_cohomology_kernel.md)
44. [附录 Z：Cubical 与 HIT 元理论边界](Z_cubical_hit_metatheory_boundary.md)
45. [附录 AA：Rezk 完备化泛性质证明架构](AA_rezk_universal_property_schema.md)
46. [附录 AB：同伦层级向上闭包证明核](AB_hlevel_upward_closure.md)
47. [附录 AC：Eckmann-Hilton 与高阶同伦群交换性](AC_eckmann_hilton_and_higher_homotopy.md)
48. [附录 AD：二点类型悬挂与圆的等价](AD_suspension_bool_circle.md)
49. [附录 AE：自然数与和类型的离散性证明核](AE_discrete_natural_numbers_and_coproducts.md)
50. [附录 AF：终对象唯一性与伴随形式证明核](AF_limits_and_adjunctions_kernel.md)
51. [附录 AG：结构 Transport 与代数 SIP 证明核](AG_structure_transport_and_sip.md)
52. [附录 AH：Full Subcategory 与本质像证明核](AH_full_subcategories_and_essential_images.md)
53. [附录 AI：Pushout 的等价不变性证明核](AI_pushout_equivalence_invariance.md)

## 全书规划

第一部：内部语言与基础规则

- 第 0 章：范围、资料源和验证纪律。
- 第 1 章：判断、语境、替换、宇宙、$\Pi$ 型与 $\Sigma$ 型。
- 第 2 章：恒等类型、路径归纳、transport、路径代数。
- 第 3 章：自然数、和类型、空类型、单位类型与命题作为类型。
- 第 4 章：可收缩类型、命题、集合与同伦层级。

第二部：等价与单值基础

- 第 5 章：fiber、isEquiv、半伴随等价与等价的稳定性。
- 第 6 章：函数外延性、命题外延性和单值性。
- 第 7 章：univalence 的基本后果与结构等同性。
- 第 8 章：截断、商类型和集合层基础数学。

第三部：高阶归纳类型与合成同伦论

- 第 9 章：高阶归纳类型的规则格式。
- 第 10 章：圆、悬挂、pushout 与同伦余极限。
- 第 11 章：基本群、覆盖空间和 $\pi_1(S^1)$。
- 第 12 章：合成同伦论、Eilenberg-Mac Lane 型、cohomology 与近期形式化结果入口。

第四部：单值范畴论与形式化数学

- 第 13 章：预范畴、单值范畴、同构与等同性。
- 第 14 章：Yoneda、极限、伴随和 Rezk completion。
- 第 15 章：UniMath、Coq-HoTT、Cubical Agda、1Lab 的口径比较。
- 第 16 章：cubical type theory、计算单值性和模型论边界。
- 第 17 章：研究边界、开放问题与版本化阅读。

## 当前范围

当前版本给出完整教材第一版，并开始向致密教材形态推进：全书主章节、附录、资料源、符号表和审查记录均已建立；附录 D 把路径归纳、$\Sigma$ 路径、fiber 收缩等基础证明核展开为可复用引理，附录 E 展开 contractible total space、复合 fiber 与等价复合的证明，附录 F 展开命题外延性、子类型外延性和命题截断泛性质，附录 G 展开等价定义比较、逆等价和同伦层级保持，附录 H 展开布尔类型的非平凡自等价和 universe 非集合性，附录 I-J 给出结构等同性原则、一元代数签名和群对象实例，附录 K 登记剩余证明义务，附录 L 精确列出当前使用的 HIT 输入规则，附录 M 展开归纳整数、商整数与 successor 自等价，附录 N 展开圆的 encode-decode 证明核，附录 O 证明同伦层级性质的命题性，附录 P-Q 展开单值范畴和 Yoneda 引理证明核，附录 R 给出 Rezk 完备化的构造输入与边界，附录 S 固定 Coq-HoTT、UniMath 和 Cubical Agda 的版本化形式化入口，附录 T 把单值性推出函数外延性精确外部化，附录 U 补齐预层范畴和 Yoneda 嵌入 fully faithful 的函子级证明核，附录 V 把圆的 loop space 等价提升为基本群同构，附录 W 补全整数加法交换群律，附录 X 补全一般函子范畴、自然同构和目标单值推出函子范畴单值的证明核，附录 Y 补入 Eilenberg-Mac Lane 型、上同调群、悬挂同构、球面计算和 cup product 的证明核与形式化入口，附录 Z 精确区分对象语言、元语言、实现语言、cubical 计算单值性、HIT 语义和 canonicity 的边界，附录 AA 给出 Rezk 完备化泛性质的 weak equivalence 限制函子证明架构，附录 AB 补全同伦层级向上闭包，附录 AC 补全 Eckmann-Hilton 与高阶同伦群交换性，附录 AD 补全 $\mathsf{susp}(\mathbf 2)\simeq\mathbb S^1$，附录 AE 补全自然数 no-confusion、自然数集合性和和类型集合性，附录 AF 补全终对象唯一性和伴随两种形式的等价，附录 AG 补全结构 transport 与命题性公理代数 SIP，附录 AH 补全 full subcategory 与本质像的单值性、fully faithful 和 essentially surjective 证明，附录 AI 补全 pushout 等价不变性证明核。基础章节给出书内证明；高阶归纳类型、合成同伦论、cubical metatheory 和形式化库章节对长证明采用“证明说明 / 外部输入 / 机器形式化 / 研究边界”分层标注。
