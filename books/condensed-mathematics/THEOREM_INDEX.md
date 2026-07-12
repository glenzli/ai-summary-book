# 凝聚数学讲义定理索引

作者：Dr. Stochastic Parrot

本索引只列本套书中承担结构作用的定义、定理、命题和输入定理。它不是术语表，而是帮助读者追踪证明依赖的地图。

## 使用约定

- “书内证明”表示本书给出完整证明或足够详细的同调代数证明。
- “输入定理”表示本书明确引用外部正式资料，不把证明伪装成正文内容。
- “形式推论”表示在接受输入定理后，由范畴论或同调代数推出。

## 卷一

| 条目 | 位置 | 类型 | 依赖 |
| --- | --- | --- | --- |
| \(\kappa\)-截断、小骨架与层级改变 | `volume-1/A_universes_and_size.md` | 书内证明 + 输入定理 | 强极限基数 / S26 Proposition 2.9, Definition 2.11 |
| sheaf 条件的等化子形式 | `volume-1/01_sites_and_sheaves.md` | 书内证明 | 覆盖和纤维积 |
| compact Hausdorff 纤维积闭性 | `volume-1/02_compact_hausdorff_and_profinite.md` | 书内证明 | Hausdorff 对角线闭 |
| 凝聚集合定义 | `volume-1/03_condensed_sets.md` | 定义 | compact Hausdorff 站点 |
| 凝聚阿贝尔群范畴 | `volume-1/04_condensed_abelian_groups.md` | 定义/形式推论 | 阿贝尔群值 sheaf |
| profinite/CHaus 站点比较 | `volume-1/05_comparison_of_test_sites.md`, `volume-1/B_site_comparison_theorem.md` | 部分书内证明 | 稳定基与共同细化 |
| Gleason 投射性 | `volume-1/06_extremally_disconnected_spaces.md`, `volume-1/D_stone_gleason.md`, `volume-1/O_gleason_projectivity_modules.md` | 输入定理 + 书内证明模块 | Gleason, Stone spaces, Sikorski extension |
| Gleason cover 构造 | `volume-1/D_stone_gleason.md`, `volume-1/J_regular_open_and_gleason_cover_details.md` | 书内证明 + 输入定理 | regular open algebra; 投射性仍输入 |
| ED 自由对象投射性 | `volume-1/07_free_objects_and_projectives.md` | 书内证明 | ED 满射分裂 |
| ED 测试正合性 | `volume-1/08_exactness_and_first_ext.md` | 书内证明 | sheaf 满射局部提升 |
| 凝聚张量积 | `volume-1/09_tensor_products_and_condensed_rings.md` | 形式推论 | sheaf 模张量泛性质 |
| 凝聚模范畴 | `volume-1/10_condensed_modules.md` | 形式推论 | Grothendieck 阿贝尔范畴 |
| 派生张量与 Tor | `volume-1/11_derived_tensor_and_tor.md`, `volume-1/H_exact_sheafification_and_derived_tools.md` | 书内证明 + 输入定理 | ringed-site K-flat 替换（Stacks `06YL`） |
| Horseshoe 与导出函子形式 | `volume-1/I_horseshoe_and_derived_functor_formalism.md` | 书内证明 | 阿贝尔范畴同调代数 |
| ED 覆盖检测与有效满射 | `volume-1/K_ed_cover_detection_and_effective_epimorphisms.md` | 书内证明 + 输入定理 | ED 覆盖存在性与投射性 |
| solid 阿贝尔群判别 | `volume-1/12_solid_abelian_groups.md` | 输入定理 | Scholze solid 理论 |
| solid 张量积 | `volume-1/13_solid_tensor_products.md` | 输入定理 | solid 核张量理想性 |
| analytic ring 入口 | `volume-1/14_analytic_rings.md` | 输入定理 | Scholze analytic rings |
| $f_!$ 与相干对偶入口 | `volume-1/15_globalization_and_duality.md` | 输入定理 | Scholze 第八讲 |
| 正合 sheafification | `volume-1/H_exact_sheafification_and_derived_tools.md` | 书内证明 | plus 构造 |
| 边界例子与反例 | `volume-1/L_boundary_examples_and_counterexamples.md` | 书内证明 | 基础 sheaf 理论 / 普通张量积 / TopAb |
| Ext/Tor 工作例题 | `volume-1/M_worked_ext_tor_examples.md` | 书内证明 | 两项投射分解 / 乘以 $n$ |
| Stone 对偶完整证明链 | `volume-1/N_stone_duality_full_proof.md` | 书内证明 + 输入定理 | Boolean prime ideal theorem |
| Gleason 投射性证明模块 | `volume-1/O_gleason_projectivity_modules.md` | 书内证明 + 输入定理 | regular open algebra / Sikorski / lifting |
| Nöbeling 定理证明模块 | `volume-1/P_nobeling_proof_modules.md` | 书内证明 + 输入定理 | finite quotient / transfinite filtration |

## 卷二

| 条目 | 位置 | 类型 | 依赖 |
| --- | --- | --- | --- |
| solid 派生范畴 $D_\square(\mathbb Z)$ | `volume-2/01_solid_derived_categories.md`, `volume-2/C_bousfield_localization_formalism.md` | 输入定理 + 形式推论 | Bousfield localization |
| solid 环与 solid 模 | `volume-2/02_solid_rings_and_modules.md` | 形式推论 | 幺半局部化 |
| analytic ring 公理、结构定理与 cone 判别 | `volume-2/03_analytic_rings_formal_conditions.md` | 定义 + 输入定理 + 书内证明 | S26 Definition 7.4 / Proposition 7.5 / Warning 7.6 |
| analyticization 泛性质 | `volume-2/04_analyticization_and_bousfield_localization.md` | 形式推论 | 反射局部化 |
| \(p\)-liquid 定义与经典空间 membership | `volume-2/05_liquid_vector_spaces.md` | 输入定理 + 书内证明 | S26 Theorem 7.11 / CS26 Theorems 2.14, 3.11 |
| 离散 Huber pair 解析化 | `volume-2/06_discrete_huber_pairs_and_analytic_rings.md` | 输入定理 | Scholze Huber pair 构造 |
| $f_!$、$f^!$、投影公式 | `volume-2/07_coherent_duality_and_f_shriek.md` | 输入定理 | Scholze six functor 入口 |
| Bousfield localization 形式定理 | `volume-2/C_bousfield_localization_formalism.md` | 形式推论 | presentable localization |
| 局部化技术引理 | `volume-2/E_localization_technical_lemmas.md` | 书内证明 | 稳定范畴与张量理想 |
| 伴随函子与投影公式形式骨架 | `volume-2/F_adjoint_functor_and_projection_formula.md` | 书内证明 + 输入定理 | presentable adjoint theorem / Brown representability |
| Cech descent 与 totalization | `volume-2/G_cech_descent_and_totalization.md` | 书内证明 + 输入定理 | ordinary sheaf descent / rational Cech descent |
| 紧生成与生成元检验 | `volume-2/H_compact_generation_and_generator_tests.md` | 书内证明 + 输入定理 | compact generation / Brown representability 边界 |
| Analytic ring 检查表与失败模式 | `volume-2/I_analytic_ring_axioms_and_failure_modes.md` | 书内证明 + 输入定理 | Dirac cone / 张量理想 / rational descent |
| Liquid 与 Banach/Fréchet 边界 | `volume-2/J_liquid_banach_frechet_boundaries.md` | 书内证明 + 输入定理 | liquid membership / 凝聚 epimorphism 的局部提升 |
| 幺半 Bousfield 局部化 | `volume-2/K_monoidal_bousfield_localization_details.md` | 书内证明 + 输入定理 | presentable localization / 张量理想 |
| 闭幺半局部化与内部 Hom | `volume-2/L_closed_monoidal_localization_and_internal_hom.md` | 书内证明 | closed monoidal category / dualizable |
| Solid localization 生成核 | `volume-2/M_solid_localization_generation_and_completion.md` | 书内证明 + 输入定理 | Dirac-to-measure cone / solidification |
| Analytic rational descent 义务 | `volume-2/N_analytic_descent_and_rational_localization_obligations.md` | 书内证明 + 输入定理 | rational localization / Čech totalization |
| 可展示稳定局部化正合性 | `volume-2/O_presentable_localization_and_exactness.md` | 书内证明 + 输入定理 | reflective localization / stable exactness |
| Liquid-Fréchet 严格 cohomology 比较 | `volume-2/P_liquid_frechet_complexes_and_closed_range.md` | 书内证明 + 输入定理 | 局部提升 / 连续 Hodge splitting |
| Solid 主定理包 | `volume-2/Q_solid_main_theorem_package.md` | 书内证明 + 输入定理 | solid 反射局部化 / 张量理想 / profinite 测度 |
| Analytic 主定理包 | `volume-2/R_analytic_main_theorem_package.md` | 书内证明 + 输入定理 | analyticization / Huber pair / rational descent |
| Liquid 主定理包 | `volume-2/S_liquid_main_theorem_package.md` | 书内证明 + 输入定理 | \(p\)-liquid analytic ring / classical-space membership / Fredholm-Hodge |
| solid/analytic/liquid 统一闭包 | `volume-2/T_mainline_closure_theorem.md` | 书内证明 + 输入定理 | 附录 Q/R/S 主定理包 |
| 第二卷出版级闭包审查 | `volume-2/U_publication_closure_audit.md` | 审查矩阵 | 附录 Q/R/S/T 与输入定理登记 |
| Solidification 反射存在性证明模块 | `volume-2/V_solidification_reflection_proof.md` | 书内证明 + 输入定理 | presentable localization / Scholze 识别 |
| Solid 核张量理想性证明模块 | `volume-2/W_solid_tensor_ideal_proof_modules.md` | 书内证明 + 输入定理 | localizing subcategory / profinite 测度张量公式 |
| Analytic localization 证明模块 | `volume-2/X_analytic_localization_proof_modules.md` | 书内证明 + 输入定理 | analytic cone / analytic ring localization |
| Rational descent 证明模块 | `volume-2/Y_rational_descent_proof_modules.md` | 书内证明 + 输入定理 | Čech nerve / compact generation / rational acyclicity |
| 经典空间的 liquid 接口证明模块 | `volume-2/Z_liquid_realization_proof_modules.md` | 书内证明 + 输入定理 | 凝聚化 / membership / 局部提升 / Fredholm-Hodge |
| Scholze/Clausen-Scholze 核心定理图谱 | `volume-2/AA_scholze_clausen_scholze_core_theorem_atlas.md` | 输入闭包图谱 | condensed / solid / analytic / liquid / complex geometry |

## 卷三

| 条目 | 位置 | 类型 | 依赖 |
| --- | --- | --- | --- |
| 复解析空间的凝聚语言 | `volume-3/01_complex_analytic_spaces_condensed_language.md` | 输入定理 | Clausen-Scholze |
| 相干层进入 analytic/liquid 派生范畴 | `volume-3/02_coherent_sheaves_and_derived_categories.md` | 输入定理 | Clausen-Scholze |
| Dolbeault 复形计算 | `volume-3/03_dolbeault_complexes_and_liquid_modules.md`, `volume-3/F_classical_complex_geometry_prerequisites.md` | 输入定理 + 形式推论 | Dolbeault lemma |
| Dolbeault 的严格 liquid cohomology 比较 | `volume-3/AQ_main_theorem_package_and_condensed_closure.md`, `volume-3/AR_clausen_scholze_complex_geometry_core_theorem_atlas.md` | 输入定理 + 书内条件推论 | Fréchet liquid membership / 局部提升 / 连续 Hodge splitting |
| coherent cohomology finiteness | `volume-3/04_finiteness_of_coherent_cohomology.md` | 输入定理 | Grauert/Hodge-Fredholm |
| Serre duality | `volume-3/05_serre_duality.md` | 输入定理 | Hodge theory/Clausen-Scholze |
| GAGA | `volume-3/06_gaga.md` | 输入定理 | Serre GAGA |
| HRR | `volume-3/07_hirzebruch_riemann_roch.md` | 输入定理 | Hirzebruch-Riemann-Roch |
| Stein-Čech 计算 | `volume-3/C_stein_cech_and_coherent_resolutions.md`, `volume-3/I_cech_hypercohomology_and_spectral_sequences.md` | 书内证明 + 输入定理 | Cartan B |
| $\mathbb P^1$ 线丛上同调 | `volume-3/H_p1_line_bundle_cech_calculation.md` | 书内证明 | 两开覆盖 Čech 计算 |
| Serre 对偶形式证明层 | `volume-3/J_serre_duality_formalism.md` | 书内证明 + 输入定理 | Serre perfectness |
| GAGA/RR 形式推论 | `volume-3/K_gaga_and_riemann_roch_formal_consequences.md` | 书内证明 + 输入定理 | GAGA / Riemann-Roch |
| Fredholm-Hodge 有限性形式层 | `volume-3/L_fredholm_hodge_finiteness.md` | 书内证明 + 输入定理 | elliptic Fredholm/Hodge decomposition |
| 有限分解与谱序列有限性传播 | `volume-3/M_finite_resolutions_and_spectral_sequence_finiteness.md` | 书内证明 | 有限过滤、谱序列收敛、有限维线性代数 |
| Fine sheaf 与 Dolbeault resolution | `volume-3/N_fine_sheaves_and_dolbeault_resolution_details.md` | 书内证明 + 输入定理 | partition of unity / Dolbeault lemma |
| 有限 resolution 下的 Ext-Serre 形式 | `volume-3/O_vector_bundle_to_coherent_serre_formal_reduction.md` | 书内证明 + 输入定理 | 向量丛 Serre duality / 有限局部自由 resolution |
| Chern character 与 Todd class 形式代数 | `volume-3/P_characteristic_classes_and_riemann_roch_algebra.md` | 书内证明 + 输入定理 | Chern 类 / splitting principle / HRR 输入 |
| GAGA properness 与导出比较 | `volume-3/Q_gaga_properness_and_derived_comparison_details.md` | 书内证明 + 输入定理 | Serre GAGA / exact equivalence |
| Dolbeault 局部正合骨架 | `volume-3/R_dolbeault_local_poincare_details.md` | 书内证明 + 输入定理 | Cauchy-Green / polydisc 同伦 |
| $\mathbb P^n$ 线丛上同调 | `volume-3/S_projective_space_cohomology_bott_basic.md` | 书内证明 + 输入定理 | Cartan B / Čech 单项式 |
| $\mathbb P^n$ 线丛 Serre 对偶 | `volume-3/T_projective_space_serre_duality.md` | 书内证明 | Čech residue / canonical bundle |
| $\mathbb P^n$ 线丛 HRR | `volume-3/U_hrr_for_projective_space_line_bundles.md` | 书内证明 | Euler sequence / Todd class / residue |
| Stein/Cartan 工具 | `volume-3/V_stein_cartan_and_coherent_sheaf_tools.md` | 书内证明 + 输入定理 | Cartan A/B / Stein acyclicity |
| 相干层有限局部自由分解 | `volume-3/W_regular_local_rings_and_coherent_resolutions.md` | 书内证明 + 输入定理 | 正则局部环 / 有限整体维数 |
| 有限性传播 | `volume-3/X_finiteness_from_vector_bundle_cases.md` | 书内证明 | hypercohomology spectral sequence |
| Projective GAGA 证明结构 | `volume-3/Y_projective_gaga_proof_architecture.md` | 书内证明 + 输入定理 | Serre twisting / finite presentation |
| 椭圆复形 Hodge 接口 | `volume-3/Z_elliptic_complexes_and_hodge_theorem.md` | 书内证明 + 输入定理 | Fredholm-Hodge / harmonic representatives |
| 向量丛 Serre 对偶 Hodge 证明 | `volume-3/AA_vector_bundle_serre_duality_from_hodge.md` | 书内证明 + 输入定理 | Hodge star / Dolbeault pairing |
| Cartan A/B 证明模块 | `volume-3/AB_cartan_theorems_proof_modules.md` | 书内证明 + 输入定理 | Weierstrass / Oka / Cousin |
| Grauert 直接像与有限性 | `volume-3/AC_grauert_direct_image_and_finiteness.md` | 书内证明 + 输入定理 | proper direct image |
| Dualizing complex 与 Serre 对偶 | `volume-3/AD_dualizing_complex_and_general_serre_duality.md` | 书内证明 + 输入定理 | Grothendieck-Serre duality |
| 一般 GRR 形式 | `volume-3/AE_grothendieck_riemann_roch_general_formalism.md` | 书内证明 + 输入定理 | Chern character / Todd / pushforward |
| Weierstrass-Oka coherence | `volume-3/AF_weierstrass_oka_coherence_local_algebra.md` | 书内证明 + 输入定理 | Weierstrass division / Oka |
| Runge-Cousin 到 Cartan B | `volume-3/AG_runge_cousin_and_cartan_b_mechanism.md` | 书内证明 + 输入定理 | Runge approximation / Cousin |
| Hörmander L2 与 Stein 消没 | `volume-3/AH_hormander_l2_and_stein_vanishing.md` | 书内证明 + 输入定理 | L2 estimate / elliptic regularity |
| Projective GAGA graded module | `volume-3/AI_projective_gaga_graded_module_details.md` | 书内证明 + 输入定理 | Serre correspondence / twisting |
| Grothendieck duality 构造义务 | `volume-3/AJ_grothendieck_duality_construction_obligations.md` | 书内证明 + 输入定理 | smooth / closed immersion / trace |
| GRR deformation-to-normal-cone | `volume-3/AK_deformation_to_normal_cone_and_grr_modules.md` | 书内证明 + 输入定理 | projective bundle / regular immersion |
| Weierstrass division 估计形式 | `volume-3/AL_weierstrass_division_estimates.md` | 书内证明 + 输入定理 | Banach 截断估计 / Neumann 级数 |
| Hörmander 基本估计与闭值域 | `volume-3/AM_hormander_basic_estimate_and_closed_range.md` | 书内证明 + 输入定理 | Bochner-Kodaira / Hilbert 复形 |
| Grauert Banach 复形模块 | `volume-3/AN_grauert_banach_complex_and_direct_images.md` | 书内证明 + 输入定理 | privileged covering / finite presentation |
| 形式 GAGA 代数化 | `volume-3/AO_formal_functions_and_gaga_algebraization.md` | 书内证明 + 输入定理 | formal functions / Grothendieck existence |
| GRR 局部化与推前相容 | `volume-3/AP_grr_localization_and_pushforward_compatibility.md` | 书内证明 + 输入定理 | \(K\)-theory localization / Chern character |
| 复几何主定理包 | `volume-3/AQ_main_theorem_package_and_condensed_closure.md` | 书内证明 + 输入定理 | finite cohomology / duality / GAGA / GRR |
| Clausen-Scholze 复几何核心定理图谱 | `volume-3/AR_clausen_scholze_complex_geometry_core_theorem_atlas.md` | 输入闭包图谱 | 建模 / Dolbeault / 有限性 / 对偶 / GAGA / HRR-GRR |

## 卷四

| 条目 | 位置 | 类型 | 依赖 |
| --- | --- | --- | --- |
| 形式化对象分类 | `volume-4/01_formalized_condensed_mathematics.md` | 形式说明 | Lean/formalized foundations |
| sheaf 计算模板 | `volume-4/02_site_and_sheaf_computations.md` | 书内证明 | 等化子 sheaf 条件 |
| Ext/Tor 计算模板 | `volume-4/03_ext_tor_computation_templates.md` | 书内证明 + 输入定理 | Grothendieck 阿贝尔范畴 |
| solid 张量积例子 | `volume-4/04_solid_tensor_examples.md` | 形式推论 | Scholze solid tensor |
| analytic ring 例子 | `volume-4/05_analytic_ring_examples.md` | 形式推论 | analytic ring 输入定理 |
| liquid 函数分析例子 | `volume-4/06_liquid_functional_analysis_examples.md` | 输入定理 + 例子 | liquid theory |
| pro-etale 比较 | `volume-4/07_pro_etale_and_condensed.md`, `volume-4/D_pro_etale_comparison_details.md` | 比较性材料 | Bhatt-Scholze |
| pyknotic/凝聚同伦入口 | `volume-4/E_current_directions_pyknotic_and_homotopy.md` | 当代方向 | Barwick-Haine |
| 凝聚基础形式化证明义务 | `volume-4/F_formal_proof_obligations_for_condensed_basics.md` | 书内证明模块 | sites / sheaves / Ext-Tor |
| 凝聚谱与 pyknotic 接口 | `volume-4/G_condensed_spectra_and_pyknotic_interfaces.md` | 书内证明 + 输入定理 | spectra-valued sheaves / hyperdescent |

## 输入定理集中清单

1. Gleason 投射性与 Gleason cover。
2. Nöbeling 定理：profinite 空间上整数值连续函数群自由。
3. Scholze solid abelian groups 的结构定理。
4. Scholze solid tensor product 和派生 solid 范畴的幺半结构。
5. Scholze analytic rings、analyticization 和 Huber pair rational localization。
6. Scholze liquid vector spaces 的 analytic ring 构造。
7. Clausen-Scholze 的 condensed/analytic complex geometry 建模。
8. Cartan A/B、Dolbeault lemma、Grauert finiteness、Serre duality、GAGA、HRR。
9. Pyknotic objects 与 condensed/pyknotic homotopy 的基础定义。
10. Rational Cech descent、Grauert/Fredholm-Hodge 有限性等应用层输入。
11. Weierstrass division estimates、Bochner-Kodaira-Nakano identity、Grauert privileged covering、Grothendieck existence 和 localized Chern character compatibility。
12. Sikorski extension theorem、Gleason lifting theorem、Nöbeling-Asgeirsson transfinite filtration 和谱值 solid/analytic localization。
