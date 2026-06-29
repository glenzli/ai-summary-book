# 外部输入定理依赖图

本文档列出本书当前版本中明确作为外部输入使用的大型定理、来源和下游影响。若将来要把某个外部输入改为书内证明，应先在这里修改依赖状态。

## 依赖分级

- **核心依赖**：后续章节直接使用，若移除会改变正文结论。
- **结构依赖**：用于说明理论全貌或相干性，移除后正文仍可保留定义但会弱化结构。
- **背景依赖**：用于定位理论，不直接参与书内证明。

## 普通与结构性范畴论

| 编号 | 外部输入 | 位置 | 分级 | 主要来源 | 下游影响 |
|---|---|---|---|---|---|
| EXT-01 | Beck 单子性定理 | [第七章](07_monads_and_algebras.md) | 结构依赖 | Mac Lane, Borceux, Riehl | 单子理论与代数范畴恢复 |
| EXT-02 | Mac Lane 相干性定理 | [第八章](08_monoidal_categories.md) | 核心依赖 | Mac Lane, Kelly | 幺半范畴中省略括号、Day 卷积相干性 |
| EXT-03 | Day 卷积完整相干性 | [第九章](09_closed_categories_and_day_convolution.md) | 结构依赖 | Day, Kelly | 预层范畴上的幺半结构 |
| EXT-04 | Fubini for ends/coends | [第十一章](11_ends_and_coends.md) | 结构依赖 | Mac Lane, Kelly, Riehl | Day 卷积计算、富范畴公式变换 |
| EXT-05 | Ind/可达范畴结构定理 | [第十二章](12_presentable_and_accessible_categories.md) | 结构依赖 | Adámek-Rosický, Borceux | presentable category 的高级应用 |
| EXT-06 | Gabriel-Popescu 定理 | [第十三章](13_exact_abelian_grothendieck_categories.md) | 背景依赖 | Popescu, Borceux | Grothendieck 范畴结构理论 |
| EXT-07 | sheaf 化存在且左正合 | [第十四章](14_sites_sheaves_and_topoi.md) | 核心依赖 | SGA 4, Johnstone | sheaf topos 构造、有限极限 |
| EXT-08 | Giraud 定理 | [第十四章](14_sites_sheaves_and_topoi.md) | 结构依赖 | Giraud, SGA 4, Johnstone | Grothendieck topos 内在刻画 |

## 同伦与高阶范畴论

| 编号 | 外部输入 | 位置 | 分级 | 主要来源 | 下游影响 |
|---|---|---|---|---|---|
| EXT-09 | 2-范畴等价严格化为伴随等价 | [第十五章](15_two_categories_and_bicategories.md) | 结构依赖 | Bénabou, Lack | 2-范畴中等价的伴随形式 |
| EXT-10 | 模型范畴同伦范畴计算 | [第十六章](16_model_categories_and_homotopy_categories.md) | 核心依赖 | Quillen, Hovey | cofibrant-fibrant 对象计算 Hom |
| EXT-11 | Quillen 伴随诱导导出函子 | [第十六章](16_model_categories_and_homotopy_categories.md) | 核心依赖 | Quillen, Hovey, Hirschhorn | 导出同伦论 |
| EXT-12 | Dwyer-Kan localization | [第十六章](16_model_categories_and_homotopy_categories.md) | 结构依赖 | Dwyer-Kan, Cisinski | 从模型范畴到 $\infty$-范畴 |
| EXT-13 | Joyal 模型结构 | [第十七章](17_simplicial_sets_and_quasicategories.md), [附录 E](E_higher_categorical_technical_models.md) | 核心依赖 | Joyal, Lurie, Cisinski | quasi-category 的同伦理论 |
| EXT-14 | quasi-category 的同伦范畴构造 | [第十八章](18_limits_adjunctions_in_infinity_categories.md) | 核心依赖 | Lurie, Riehl-Verity | 等价边、复合良定义 |
| EXT-15 | 映射空间是 Kan 复形 | [第十八章](18_limits_adjunctions_in_infinity_categories.md) | 核心依赖 | Lurie, Kerodon | 映射空间判别、伴随保持性 |
| EXT-16 | $\infty$-伴随的映射空间刻画 | [第十八章](18_limits_adjunctions_in_infinity_categories.md) | 核心依赖 | Lurie, Riehl-Verity | 左伴随保持余极限 |
| EXT-17 | Cartesian model structure on marked simplicial sets | [第十九章](19_cartesian_fibrations_and_straightening.md), [附录 E](E_higher_categorical_technical_models.md) | 核心依赖 | Lurie HTT, Kerodon | Cartesian fibration 的模型结构 |
| EXT-18 | ordinary fibration nerve comparison | [第十九章](19_cartesian_fibrations_and_straightening.md), [附录 E](E_higher_categorical_technical_models.md) | 背景依赖 | Lurie HTT, Kerodon | 普通 Grothendieck fibration 与高阶版本比较 |
| EXT-19 | straightening/unstraightening | [第十九章](19_cartesian_fibrations_and_straightening.md) | 核心依赖 | Lurie HTT, Riehl-Verity | $\infty$-范畴族与 Cartesian fibrations 等价 |
| EXT-20 | presentable $\infty$-categories 中 Kan 延拓存在性 | [第十九章](19_cartesian_fibrations_and_straightening.md) | 结构依赖 | Lurie HTT | 高阶 Kan 延拓 |
| EXT-21 | 稳定 $\infty$-范畴的三角同伦范畴 | [第二十章](20_stable_infinity_categories_and_spectra.md) | 核心依赖 | Lurie HA | 从 stable $\infty$-category 到 triangulated category |
| EXT-22 | 谱范畴稳定化泛性质 | [第二十章](20_stable_infinity_categories_and_spectra.md) | 核心依赖 | Lurie HA | spectra 与稳定化 |
| EXT-23 | t-结构 heart 阿贝尔性 | [第二十章](20_stable_infinity_categories_and_spectra.md) | 核心依赖 | Lurie HA | 稳定 $\infty$-范畴与阿贝尔范畴连接 |
| EXT-24 | 高阶 Giraud 定理 | [第二十一章](21_higher_topos_theory.md) | 结构依赖 | Lurie HTT, Rezk | $\infty$-topos 内在刻画 |
| EXT-25 | $\infty$-topos 的 0-截断 topos | [第二十一章](21_higher_topos_theory.md) | 核心依赖 | Lurie HTT | ordinary topos 与 higher topos 比较 |
| EXT-26 | Dunn additivity | [第二十二章](22_higher_algebra_and_infinity_operads.md) | 背景依赖 | Dunn, Lurie, May | $E_n$-代数结构 |
| EXT-27 | $\operatorname{Alg}_{\mathcal O}(C)$ 的 presentability | [第二十二章](22_higher_algebra_and_infinity_operads.md) | 结构依赖 | Lurie HA | 高阶代数范畴存在性 |
| EXT-28 | stable $\infty$-categories 的谱富化 | [第二十章](20_stable_infinity_categories_and_spectra.md) | 核心依赖 | Lurie HA | 映射谱、稳定高阶信息 |
| EXT-29 | $\mathbf{Sp}$ 的闭对称幺半 smash product | [第二十章](20_stable_infinity_categories_and_spectra.md) | 核心依赖 | Lurie HA, Hovey-Schwede-Shipley | 环谱、模谱、高阶代数 |
| EXT-30 | correspondence/表示性伴随刻画 | [第十八章](18_limits_adjunctions_in_infinity_categories.md) | 结构依赖 | Lurie HTT, Riehl-Verity | $\infty$-伴随的模型无关表述 |
| EXT-31 | 模 $\infty$-范畴和 bar 相对张量积存在性 | [第二十二章](22_higher_algebra_and_infinity_operads.md) | 核心依赖 | Lurie HA | 模谱、相对张量积、导出代数 |
| EXT-32 | adjunction data 与其他伴随定义等价 | [第十八章](18_limits_adjunctions_in_infinity_categories.md) | 结构依赖 | Lurie HTT, Riehl-Verity | 单位余单位和高阶三角相干 |
| EXT-33 | t-结构 heart 阿贝尔性、cohomology 长正合列和滤过谱序列 | [第二十章](20_stable_infinity_categories_and_spectra.md) | 核心依赖 | Lurie HA, Beilinson-Bernstein-Deligne, Boardman | heart 正合性、长正合列、谱序列计算和收敛 |
| EXT-34 | 高阶 Deligne 中心定理 | [第二十二章](22_higher_algebra_and_infinity_operads.md) | 结构依赖 | Lurie HA, Francis | $E_n$-代数中心与 $E_{n+1}$-结构 |
| EXT-35 | 因子化同调和 excision | [第二十二章](22_higher_algebra_and_infinity_operads.md) | 结构依赖 | Lurie HA, Ayala-Francis | 拓扑场论、$E_n$-代数的流形积分 |
| EXT-36 | cobordism hypothesis | [第二十二章](22_higher_algebra_and_infinity_operads.md) | 背景依赖 | Baez-Dolan, Lurie | fully dualizable objects 与 extended TFT |
| EXT-37 | smooth/proper Morita 可对偶性判别 | [第二十二章](22_higher_algebra_and_infinity_operads.md) | 结构依赖 | Toën, Lurie HA, Francis | Morita 理论中的 fully dualizable objects 和 TFT 点值 |
| EXT-38 | scaled simplicial sets 的 $(\infty,2)$ 模型结构 | [第十八章](18_limits_adjunctions_in_infinity_categories.md), [附录 E](E_higher_categorical_technical_models.md) | 结构依赖 | Lurie HTT, Gagna-Harpaz-Lanari, Riehl-Verity | walking adjunction、2-态射相干和高阶 Morita 结构 |
| EXT-39 | Cartesian sections as limits | [第十九章](19_cartesian_fibrations_and_straightening.md) | 核心依赖 | Lurie HTT, Kerodon | descent data、层化对象和参数化对象的极限描述 |
| EXT-40 | 局部可表现范畴伴随函子定理 | [第十二章](12_presentable_and_accessible_categories.md) | 核心依赖 | Adámek-Rosický, Borceux, Riehl | 可达且保持余极限的函子有右伴随 |
| EXT-41 | plus 构造给出 sheaf 化 | [第十四章](14_sites_sheaves_and_topoi.md) | 结构依赖 | SGA 4, Johnstone, Mac Lane-Moerdijk | sheaf 化存在性、separated 化和左正合性 |
| EXT-42 | $\infty$-topos 中 groupoid objects 有效 | [第二十一章](21_higher_topos_theory.md) | 核心依赖 | Lurie HTT, Rezk | effective epimorphisms、descent、higher Giraud |
| EXT-43 | hypercompletion 和 Postnikov 收敛 | [第二十一章](21_higher_topos_theory.md) | 结构依赖 | Lurie HTT | hyperdescent、截断塔和 left exact localization |
| EXT-44 | $\infty$-Yoneda 与预层 $\infty$-范畴 | [第二十三章](23_presentable_infinity_categories_and_localizations.md) | 核心依赖 | Lurie HTT, Riehl-Verity, Kerodon | 可表紧性、预层生成、presentable $\infty$-categories |
| EXT-45 | presentable $\infty$-category 伴随函子定理和 accessible localizations | [第二十三章](23_presentable_infinity_categories_and_localizations.md) | 核心依赖 | Lurie HTT | 左伴随判别、Bousfield localization、topos 和稳定局部化 |
| EXT-46 | $\operatorname{Pr}^L$ 的闭对称幺半结构 | [第二十三章](23_presentable_infinity_categories_and_localizations.md) | 结构依赖 | Lurie HA | presentable 幺半 $\infty$-范畴、高阶代数和模范畴 |
| EXT-47 | $\mathbf{Prof}$ 双范畴与 Cauchy completion 相干性 | [第二十四章](24_profunctors_cauchy_completion_and_correspondences.md) | 结构依赖 | Bénabou, Street, Kelly | profunctor 复合、加权余极限、Morita 观点 |
| EXT-48 | $\infty$-correspondence 的 $(\infty,2)$-范畴结构 | [第二十四章](24_profunctors_cauchy_completion_and_correspondences.md) | 结构依赖 | Lurie HTT, HA, Gaitsgory-Rozenblyum | 高阶 profunctor、Morita $\infty$-范畴、span/correspondence |
| EXT-49 | 富 profunctor 双范畴与 equipment 相干性 | [第二十五章](25_enriched_profunctors_equipments_and_base_change.md) | 结构依赖 | Kelly, Street, Shulman | 富 coend 复合、companion/conjoint、Beck-Chevalley |
| EXT-50 | 高阶 equipment/framed bicategory 模型 | [第二十五章](25_enriched_profunctors_equipments_and_base_change.md) | 结构依赖 | Lurie HA, Gaitsgory-Rozenblyum, Haugseng, Shulman | 高阶 correspondence、six functors、Morita $(\infty,2)$-结构 |
| EXT-51 | Brown 表示性 | [第二十六章](26_compact_generation_brown_representability_and_bousfield_localization.md) | 核心依赖 | Brown, Neeman, Lurie HA | 伴随存在性、表示对象、稳定同伦论 |
| EXT-52 | 稳定 presentable Verdier quotient 和 Bousfield localization 存在性 | [第二十六章](26_compact_generation_brown_representability_and_bousfield_localization.md) | 核心依赖 | Lurie HA, Neeman | localizing subcategory、稳定商、同调局部化 |
| EXT-53 | Neeman-Thomason 紧对象商定理 | [第二十六章](26_compact_generation_brown_representability_and_bousfield_localization.md) | 结构依赖 | Thomason, Neeman | compact objects、Verdier quotient、幂等完备化 |
| EXT-54 | dg 模范畴的模型结构和导出模 $\infty$-范畴 | [第二十七章](27_dg_categories_enhancements_and_derived_morita_theory.md) | 核心依赖 | Keller, Tabuada, Lurie HA | $D(\mathcal A)$、perfect modules、导出 Morita 理论 |
| EXT-55 | dg nerve 与 pretriangulated dg category 的稳定性 | [第二十七章](27_dg_categories_enhancements_and_derived_morita_theory.md) | 核心依赖 | Lurie HA, Keller | dg enhancement、稳定 $\infty$-范畴和三角范畴比较 |
| EXT-56 | $D(\mathcal A)$ 的 compact generation 与 perfect objects | [第二十七章](27_dg_categories_enhancements_and_derived_morita_theory.md) | 核心依赖 | Keller, Neeman, Lurie HA | $D(\mathcal A)^\omega\simeq\operatorname{Perf}(\mathcal A)$ |
| EXT-57 | 导出 Morita 定理 | [第二十七章](27_dg_categories_enhancements_and_derived_morita_theory.md) | 结构依赖 | Toën, Tabuada, Keller | dg categories 的 Morita 局部化和 bimodule 复合 |
| EXT-58 | Hochschild 型不变量的 Morita 不变性 | [第二十七章](27_dg_categories_enhancements_and_derived_morita_theory.md) | 结构依赖 | Keller, Toën, Blumberg-Gepner-Tabuada | 非交换不变量、trace、THH 和 cyclic homology |
| EXT-59 | 具体几何理论中的六操作存在性 | [第二十八章](28_six_functor_formalism_base_change_and_projection_formula.md) | 核心依赖 | Grothendieck, Verdier, Deligne, Ayoub, Cisinski-Déglise | sheaf、étale sheaf、motivic sheaf、$D$-module 的六操作 |
| EXT-60 | 六操作基变换定理 | [第二十八章](28_six_functor_formalism_base_change_and_projection_formula.md) | 核心依赖 | SGA 4, Verdier, Deligne, Lurie, Gaitsgory-Rozenblyum | proper base change、smooth base change、非常基变换 |
| EXT-61 | 投影公式和 $\mathcal D(Y)$-线性 | [第二十八章](28_six_functor_formalism_base_change_and_projection_formula.md) | 核心依赖 | Grothendieck, Verdier, Ayoub, Cisinski-Déglise | tensor 与推前相容、dualizable object 计算 |
| EXT-62 | Recollement 和局部化三角 | [第二十八章](28_six_functor_formalism_base_change_and_projection_formula.md) | 结构依赖 | BBD, Verdier, Kashiwara-Schapira, Lurie | 开闭分解、extension by zero、局部-整体检测 |
| EXT-63 | Verdier 对偶、exceptional pullback 和 purity | [第二十八章](28_six_functor_formalism_base_change_and_projection_formula.md) | 结构依赖 | Verdier, Deligne, Ayoub, Cisinski-Déglise | $f^!$、dualizing object、proper duality、smooth purity |
| EXT-64 | Dwyer-Kan simplicial localization 和 hammock localization | [第二十九章](29_relative_categories_simplicial_localization_and_model_comparisons.md) | 核心依赖 | Dwyer-Kan, Hirschhorn, Cisinski | relative categories 到映射空间丰富模型 |
| EXT-65 | 模型范畴 underlying $\infty$-category 的映射空间计算 | [第二十九章](29_relative_categories_simplicial_localization_and_model_comparisons.md) | 核心依赖 | Dwyer-Kan, Hovey, Lurie HTT | cofibrant-fibrant derived mapping spaces |
| EXT-66 | Coherent nerve 与 simplicial categories | [第二十九章](29_relative_categories_simplicial_localization_and_model_comparisons.md) | 核心依赖 | Cordier-Porter, Lurie HTT, Riehl-Verity | simplicial category 与 quasi-category 比较 |
| EXT-67 | Bergner-Joyal Quillen 等价 | [第二十九章](29_relative_categories_simplicial_localization_and_model_comparisons.md) | 结构依赖 | Bergner, Joyal, Lurie HTT | $\mathbf{sCat}$ 与 quasi-categories 的模型比较 |
| EXT-68 | Rezk complete Segal space 模型与相对范畴模型比较 | [第二十九章](29_relative_categories_simplicial_localization_and_model_comparisons.md) | 结构依赖 | Rezk, Barwick-Kan, Bergner | CSS、relative categories 和 homotopy theories of homotopy theories |
| EXT-69 | Drinfeld dg quotient | [第三十章](30_dg_quotients_localizing_invariants_and_noncommutative_motives.md) | 核心依赖 | Drinfeld, Keller | dg quotient、对象收缩、Verdier quotient 的 dg 模型 |
| EXT-70 | dg quotient 与稳定 Verdier quotient 的比较 | [第三十章](30_dg_quotients_localizing_invariants_and_noncommutative_motives.md) | 核心依赖 | Drinfeld, Lurie HA, Keller | dg enhancement、稳定商、Morita 理论 |
| EXT-71 | 非连通代数 $K$-理论的局部化定理 | [第三十章](30_dg_quotients_localizing_invariants_and_noncommutative_motives.md) | 核心依赖 | Thomason-Trobaugh, Waldhausen, Blumberg-Gepner-Tabuada | exact sequences 到谱纤维序列 |
| EXT-72 | Universal additive/localizing noncommutative motives | [第三十章](30_dg_quotients_localizing_invariants_and_noncommutative_motives.md) | 结构依赖 | Tabuada, Blumberg-Gepner-Tabuada | additive/localizing invariants 的普遍表示 |
| EXT-73 | Hochschild 与 THH 的 localizing 性 | [第三十章](30_dg_quotients_localizing_invariants_and_noncommutative_motives.md) | 结构依赖 | Keller, Blumberg-Mandell, Blumberg-Gepner-Tabuada | trace 型不变量、Morita 不变性、局部化序列 |
| EXT-74 | Perverse t-structure 存在性 | [第三十一章](31_perverse_sheaves_recollement_and_t_structures.md) | 核心依赖 | BBD, Goresky-MacPherson, Kashiwara-Schapira | perverse heart、支撑/余支撑条件 |
| EXT-75 | BBD recollement gluing of t-structures | [第三十一章](31_perverse_sheaves_recollement_and_t_structures.md) | 核心依赖 | BBD | 层化归纳、perverse t-结构构造 |
| EXT-76 | 中间延拓的刻画 | [第三十一章](31_perverse_sheaves_recollement_and_t_structures.md) | 结构依赖 | BBD, Kashiwara-Schapira | simple perverse sheaves、闭支撑子商对象 |
| EXT-77 | Verdier 对偶对 perverse t-结构的 t-exactness | [第三十一章](31_perverse_sheaves_recollement_and_t_structures.md) | 核心依赖 | Verdier, BBD | perverse heart 反等价、中间延拓对偶 |
| EXT-78 | Nearby cycles 与 vanishing cycles 的 perverse t-exactness | [第三十一章](31_perverse_sheaves_recollement_and_t_structures.md) | 结构依赖 | Deligne, BBD, SGA 7, Kashiwara-Schapira | 奇异退化、monodromy、消失循环 |
| EXT-79 | Morava $K$-theories 存在性和 graded field 性质 | [第三十二章](32_chromatic_homotopy_bousfield_lattices_and_telescope_conjecture.md) | 核心依赖 | Morava, Ravenel, Hovey-Strickland | chromatic height、$K(n)$-localization |
| EXT-80 | Hopkins-Smith thick subcategory theorem | [第三十二章](32_chromatic_homotopy_bousfield_lattices_and_telescope_conjecture.md) | 核心依赖 | Hopkins-Smith, Ravenel | finite spectra 的 chromatic type 分类 |
| EXT-81 | 周期性定理与 $v_n$-self maps | [第三十二章](32_chromatic_homotopy_bousfield_lattices_and_telescope_conjecture.md) | 核心依赖 | Hopkins-Smith, Devinatz-Hopkins-Smith | telescope 谱和有限周期性 |
| EXT-82 | Telescope conjecture | [第三十二章](32_chromatic_homotopy_bousfield_lattices_and_telescope_conjecture.md) | 背景依赖 | Ravenel, Bousfield | finite localization 与 telescope localization 比较 |
| EXT-83 | Chromatic fracture square | [第三十二章](32_chromatic_homotopy_bousfield_lattices_and_telescope_conjecture.md) | 结构依赖 | Ravenel, Hovey-Palmieri-Strickland | 高度分层和拉回粘合 |
| EXT-84 | Bernstein inequality 和 holonomic $D$-modules | [第三十三章](33_d_modules_riemann_hilbert_and_de_rham_functors.md) | 核心依赖 | Bernstein, Kashiwara, Hotta-Takeuchi-Tanisaki | characteristic variety、holonomic 有限性 |
| EXT-85 | Riemann-Hilbert correspondence | [第三十三章](33_d_modules_riemann_hilbert_and_de_rham_functors.md) | 核心依赖 | Kashiwara, Mebkhout, Borel, Beilinson-Bernstein | regular holonomic $D$-modules 与 constructible/perverse sheaves |
| EXT-86 | $D$-module 六操作 | [第三十三章](33_d_modules_riemann_hilbert_and_de_rham_functors.md) | 结构依赖 | Kashiwara-Schapira, Gaitsgory-Rozenblyum | proper direct image、duality、de Rham 相容 |
| EXT-87 | Kashiwara equivalence | [第三十三章](33_d_modules_riemann_hilbert_and_de_rham_functors.md) | 核心依赖 | Kashiwara, Kashiwara-Schapira | 闭嵌入、支撑在闭子空间的 $D$-modules |
| EXT-88 | Derived/spectral stack representability | [第三十四章](34_derived_algebraic_geometry_cotangent_complexes_and_spectral_stacks.md) | 结构依赖 | Toën-Vezzosi, Lurie DAG/SAG | derived Artin/DM stacks、atlases、下降 |
| EXT-89 | $\operatorname{QCoh}$ compact generation and perfect objects | [第三十四章](34_derived_algebraic_geometry_cotangent_complexes_and_spectral_stacks.md) | 核心依赖 | Lurie SAG, Gaitsgory-Rozenblyum, Thomason-Trobaugh | perfect complexes、stable presentable categories |
| EXT-90 | Lurie-Pridham formal moduli theorem | [第三十四章](34_derived_algebraic_geometry_cotangent_complexes_and_spectral_stacks.md) | 核心依赖 | Lurie, Pridham, Hinich | formal moduli problems 与 dg/spectral Lie algebras |
| EXT-91 | IndCoh 和 singular support | [第三十四章](34_derived_algebraic_geometry_cotangent_complexes_and_spectral_stacks.md) | 结构依赖 | Gaitsgory-Rozenblyum, Arinkin-Gaitsgory | Grothendieck duality、奇异支撑、几何表示论 |
| EXT-92 | Barr-Beck-Lurie monadicity theorem | [第三十五章](35_barr_beck_lurie_monadicity_and_descent.md) | 核心依赖 | Lurie HA, Barr-Beck, Riehl-Verity | monadic 重构、模范畴、代数对象 |
| EXT-93 | Comonadic Barr-Beck 和 Cech descent | [第三十五章](35_barr_beck_lurie_monadicity_and_descent.md) | 核心依赖 | Lurie HA, Gaitsgory-Rozenblyum | descent data、余单子、覆盖重构 |
| EXT-94 | Faithfully flat descent for QCoh | [第三十五章](35_barr_beck_lurie_monadicity_and_descent.md) | 核心依赖 | SGA, Lurie SAG, Gaitsgory-Rozenblyum | QCoh 下降、fpqc descent |
| EXT-95 | 经典 Tannaka duality | [第三十六章](36_tannaka_duality_affine_group_schemes_and_higher_reconstruction.md) | 核心依赖 | Saavedra, Deligne-Milne | neutral Tannakian categories、仿射群概形 |
| EXT-96 | 高阶 Tannaka duality for derived stacks | [第三十六章](36_tannaka_duality_affine_group_schemes_and_higher_reconstruction.md) | 核心依赖 | Lurie SAG, Gaitsgory-Rozenblyum, Toën-Vezzosi | $\operatorname{QCoh}(X)$ 重构 derived stacks |
| EXT-97 | $\operatorname{QCoh}(BG)\simeq\operatorname{Rep}(G)$ | [第三十六章](36_tannaka_duality_affine_group_schemes_and_higher_reconstruction.md) | 结构依赖 | Gaitsgory-Rozenblyum, Lurie SAG | classifying stacks、表示范畴 |
| EXT-98 | Balmer spectrum 分类定理 | [第三十七章](37_tensor_triangular_geometry_balmer_spectra_and_support.md) | 核心依赖 | Balmer, Thomason | radical thick tensor ideals 与 Thomason subsets |
| EXT-99 | $\operatorname{Spc}(\operatorname{Perf}(R))\cong\operatorname{Spec}R$ | [第三十七章](37_tensor_triangular_geometry_balmer_spectra_and_support.md) | 核心依赖 | Balmer, Thomason | perfect complexes 的 tt-geometry |
| EXT-100 | 有限谱的 Balmer spectrum 与 chromatic primes | [第三十七章](37_tensor_triangular_geometry_balmer_spectra_and_support.md) | 结构依赖 | Hopkins-Smith, Balmer, Hovey-Palmieri-Strickland | thick tensor ideals 和 chromatic type |
| EXT-101 | $THH$ 的 localizing invariance | [第三十八章](38_topological_hochschild_homology_cyclotomic_trace_and_tc.md) | 核心依赖 | Blumberg-Mandell, Blumberg-Gepner-Tabuada | exact sequences 到 THH 纤维序列 |
| EXT-102 | $THH$ 的 cyclotomic refinement | [第三十八章](38_topological_hochschild_homology_cyclotomic_trace_and_tc.md) | 核心依赖 | Bökstedt, Nikolaus-Scholze | cyclotomic spectra、Frobenius、Tate construction |
| EXT-103 | Cyclotomic trace | [第三十八章](38_topological_hochschild_homology_cyclotomic_trace_and_tc.md) | 核心依赖 | Bökstedt-Hsiang-Madsen, Dundas-McCarthy | $K\to TC$ 自然变换 |
| EXT-104 | Dundas-Goodwillie-McCarthy theorem | [第三十八章](38_topological_hochschild_homology_cyclotomic_trace_and_tc.md) | 结构依赖 | Dundas-Goodwillie-McCarthy | nilpotent extensions 的相对 $K$ 与相对 $TC$ |
| EXT-105 | Goodwillie $n$-excisive approximation | [第三十九章](39_goodwillie_calculus_excisive_functors_and_derivatives.md) | 核心依赖 | Goodwillie, Lurie HA | Taylor tower、$P_nF$ 泛性质 |
| EXT-106 | Goodwillie derivatives and homogeneous functor classification | [第三十九章](39_goodwillie_calculus_excisive_functors_and_derivatives.md) | 核心依赖 | Goodwillie, Arone-Ching | $\partial_nF$、$\Sigma_n$-spectra、homogeneous layers |
| EXT-107 | Goodwillie chain rule and operad structure | [第三十九章](39_goodwillie_calculus_excisive_functors_and_derivatives.md) | 结构依赖 | Arone-Ching, Lurie HA, Heuts | derivatives of composite functors、spectral Lie operad |
| EXT-108 | Goodwillie tower convergence | [第三十九章](39_goodwillie_calculus_excisive_functors_and_derivatives.md) | 结构依赖 | Goodwillie | analytic functors、connectivity estimates |
| EXT-109 | Morel-Voevodsky motivic homotopy category | [第四十章](40_motivic_homotopy_a1_localization_and_six_operations.md) | 核心依赖 | Morel-Voevodsky, Hoyois | $\mathbb A^1$-localization、motivic spaces、$\mathbf{SH}(S)$ |
| EXT-110 | Motivic six functor formalism | [第四十章](40_motivic_homotopy_a1_localization_and_six_operations.md) | 核心依赖 | Ayoub, Cisinski-Déglise, Hoyois | $\mathbf{SH}(-)$、base change、projection formula、purity |
| EXT-111 | Homotopy purity theorem | [第四十章](40_motivic_homotopy_a1_localization_and_six_operations.md) | 核心依赖 | Morel-Voevodsky | Thom spaces、closed immersions、purity |
| EXT-112 | Motivic Eilenberg-Mac Lane spectra and motives | [第四十章](40_motivic_homotopy_a1_localization_and_six_operations.md) | 结构依赖 | Voevodsky, Cisinski-Déglise, Robalo | motives as modules、motivic cohomology |
| EXT-113 | Compact generation of $\mathbf{SH}(S)$ | [第四十章](40_motivic_homotopy_a1_localization_and_six_operations.md) | 结构依赖 | Morel-Voevodsky, Neeman, Hoyois | smooth schemes and Tate twists as generators |
| EXT-114 | Elementary topos internal logic | [第四十一章](41_categorical_logic_dependent_type_theory_and_univalence.md) | 核心依赖 | Lawvere, Johnstone, Mac Lane-Moerdijk | Heyting-valued semantics、higher-order intuitionistic logic |
| EXT-115 | Identity types from weak factorization systems | [第四十一章](41_categorical_logic_dependent_type_theory_and_univalence.md) | 结构依赖 | Awodey-Warren, Garner, Gambino | Martin-Lof identity types、path objects、comprehension categories |
| EXT-116 | Univalent universes and HoTT models | [第四十一章](41_categorical_logic_dependent_type_theory_and_univalence.md) | 核心依赖 | Voevodsky, Kapulkin-Lumsdaine, Shulman | univalence、simplicial sets、$\infty$-topos semantics |
| EXT-117 | Factorization homology excision | [第四十二章](42_factorization_homology_en_algebras_and_nonabelian_poincare_duality.md) | 核心依赖 | Lurie HA, Ayala-Francis | collar-gluing、relative tensor products、local-to-global computation |
| EXT-118 | Circle factorization homology and Hochschild homology | [第四十二章](42_factorization_homology_en_algebras_and_nonabelian_poincare_duality.md) | 核心依赖 | Lurie HA, Francis, Ayala-Francis | trace、$HH(A)$、circle action |
| EXT-119 | Nonabelian Poincare duality | [第四十二章](42_factorization_homology_en_algebras_and_nonabelian_poincare_duality.md) | 结构依赖 | Segal, May, Lurie, Ayala-Francis | grouplike $E_n$-spaces、mapping spaces、delooping |
| EXT-120 | Locally constant factorization algebras and $E_n$-algebras | [第四十二章](42_factorization_homology_en_algebras_and_nonabelian_poincare_duality.md) | 结构依赖 | Costello-Gwilliam, Lurie, Francis | factorization algebras、Weiss descent、$E_n$-structures |
| EXT-121 | Condensed topos and condensed abelian groups | [第四十三章](43_condensed_sets_solid_modules_and_analytic_categories.md) | 核心依赖 | Clausen-Scholze, Johnstone | condensed sets、Grothendieck topos、Grothendieck abelian category |
| EXT-122 | Topological spaces embedded in condensed sets | [第四十三章](43_condensed_sets_solid_modules_and_analytic_categories.md) | 结构依赖 | Clausen-Scholze, Barwick-Haine | compactly generated spaces、finite limits、continuous maps |
| EXT-123 | Solidification and solid tensor product | [第四十三章](43_condensed_sets_solid_modules_and_analytic_categories.md) | 核心依赖 | Clausen-Scholze | solid abelian groups、reflective localization、solid modules |
| EXT-124 | Analytic rings and solid derived categories | [第四十三章](43_condensed_sets_solid_modules_and_analytic_categories.md) | 结构依赖 | Clausen-Scholze, Scholze | analytic rings、complete modules、stable presentable categories |
| EXT-125 | Syntactic category universal property | [第四十四章](44_syntactic_categories_classifying_toposes_and_tripos.md) | 核心依赖 | Lawvere, Makkai-Reyes | finite-limit theories、models as lex functors |
| EXT-126 | Existence of classifying toposes | [第四十四章](44_syntactic_categories_classifying_toposes_and_tripos.md) | 核心依赖 | Johnstone, Mac Lane-Moerdijk | geometric theories、generic models、sheaves on syntactic sites |
| EXT-127 | Tripos-to-topos construction | [第四十四章](44_syntactic_categories_classifying_toposes_and_tripos.md) | 核心依赖 | Hyland-Johnstone-Pitts, Johnstone | PER objects、elementary toposes、realizability |
| EXT-128 | Regular completion | [第四十五章](45_exact_completions_relations_allegories_and_regular_logic.md) | 结构依赖 | Carboni-Vitale, Makkai-Reyes | lex categories、regular categories、images |
| EXT-129 | Exact completion | [第四十五章](45_exact_completions_relations_allegories_and_regular_logic.md) | 核心依赖 | Carboni-Vitale, Johnstone | effective equivalence relations、quotients、exact categories |
| EXT-130 | Allegories and exact categories | [第四十五章](45_exact_completions_relations_allegories_and_regular_logic.md) | 结构依赖 | Freyd-Scedrov, Carboni-Walters | relation calculus、tabulations、regular logic |
| EXT-131 | Examples of cohesive $\infty$-toposes | [第四十六章](46_cohesive_toposes_modalities_and_differential_cohesion.md) | 结构依赖 | Lawvere, Schreiber, Shulman | shape、discrete/codiscrete objects、cohesive homotopy types |
| EXT-132 | Differential cohesion and de Rham stacks | [第四十六章](46_cohesive_toposes_modalities_and_differential_cohesion.md) | 结构依赖 | Schreiber, Shulman | infinitesimal shape、differential cohomology、smooth stacks |
| EXT-133 | Modal HoTT semantics | [第四十六章](46_cohesive_toposes_modalities_and_differential_cohesion.md) | 核心依赖 | Shulman, Rijke-Shulman-Spitters | left exact modalities、modal type theory、identity types |
| EXT-134 | Exit-path simplicial set is a quasi-category | [第四十七章](47_stratified_homotopy_exit_path_categories_and_constructible_sheaves.md) | 核心依赖 | Lurie, Treumann | conically stratified spaces、exit paths、quasi-categories |
| EXT-135 | Constructible sheaves classified by exit paths | [第四十七章](47_stratified_homotopy_exit_path_categories_and_constructible_sheaves.md) | 核心依赖 | Lurie, Treumann, MacPherson | constructible sheaves、functors from exit categories |
| EXT-136 | Stratified factorization homology | [第四十七章](47_stratified_homotopy_exit_path_categories_and_constructible_sheaves.md) | 结构依赖 | Ayala-Francis-Rozenblyum | stratified disks、constructible bundles、excision |
| EXT-137 | Higher Morita $(\infty,n)$-categories | [第四十八章](48_higher_morita_traces_and_en_koszul_duality.md) | 核心依赖 | Lurie HA, Haugseng, Calaque-Scheimbauer | $E_n$-algebras、iterated bimodules、relative tensor product |
| EXT-138 | Smooth/proper fully dualizable criterion | [第四十八章](48_higher_morita_traces_and_en_koszul_duality.md) | 结构依赖 | Toën, Lurie HA, Francis | Morita dualizability、TFT point values |
| EXT-139 | Higher Morita traces and factorization homology | [第四十八章](48_higher_morita_traces_and_en_koszul_duality.md) | 核心依赖 | Lurie HA, Francis, Ayala-Francis | traces、higher Hochschild objects、annular factorization homology |
| EXT-140 | $E_n$-Koszul duality | [第四十八章](48_higher_morita_traces_and_en_koszul_duality.md) | 核心依赖 | Lurie HA, Francis-Gaitsgory, Ginot | augmented $E_n$-algebras、bar/cobar、double duality |
| EXT-141 | Factorization/Koszul Poincare duality | [第四十八章](48_higher_morita_traces_and_en_koszul_duality.md) | 结构依赖 | Francis, Ayala-Francis, Lurie HA | factorization homology/cohomology、Koszul dual coalgebras |
| EXT-142 | Derivators from homotopy theories | [第四十九章](49_derivators_homotopy_kan_extensions_and_stable_derivators.md) | 核心依赖 | Grothendieck, Heller, Groth, Cisinski | model categories、relative categories、$\infty$-categories to derivators |
| EXT-143 | Derivator pointwise Kan extension formulas | [第四十九章](49_derivators_homotopy_kan_extensions_and_stable_derivators.md) | 核心依赖 | Groth, Maltsiniotis | homotopy Kan extensions、comma categories、homotopy limits |
| EXT-144 | Stable derivators induce triangulated categories | [第四十九章](49_derivators_homotopy_kan_extensions_and_stable_derivators.md) | 结构依赖 | Heller, Groth, Franke | triangulated structure、cofiber/fiber sequences |
| EXT-145 | Classifying stack and torsor cohomology | [第五十章](50_stacks_gerbes_and_nonabelian_cohomology.md) | 核心依赖 | Giraud, Jardine, Lurie HTT | $BG$、torsors、$H^1$ |
| EXT-146 | Gerbes classified by $H^2$ | [第五十章](50_stacks_gerbes_and_nonabelian_cohomology.md) | 核心依赖 | Giraud, Breen | banded gerbes、nonabelian cohomology |
| EXT-147 | Higher stack hyperdescent | [第五十章](50_stacks_gerbes_and_nonabelian_cohomology.md) | 结构依赖 | Jardine, Lurie HTT | space-valued sheaves、hypercompletion、higher stacks |
| EXT-148 | Effective descent in regular categories and toposes | [第五十一章](51_categorical_galois_theory_descent_and_effective_descent.md) | 核心依赖 | Grothendieck, Johnstone, Janelidze-Kelly | regular epis、epimorphisms、slice descent |
| EXT-149 | Categorical Galois structures | [第五十一章](51_categorical_galois_theory_descent_and_effective_descent.md) | 结构依赖 | Janelidze-Kelly, Borceux-Janelidze | coverings、normal extensions、admissibility |
| EXT-150 | Normal extensions and groupoid actions | [第五十一章](51_categorical_galois_theory_descent_and_effective_descent.md) | 结构依赖 | Janelidze-Kelly | Galois groupoids、descent equivalences |
| EXT-151 | Existence of W-types | [第五十二章](52_polynomial_functors_species_analytic_functors_and_w_types.md) | 核心依赖 | Moerdijk-Palmgren, Gambino-Hyland | polynomial functors、initial algebras、type theory |
| EXT-152 | Polynomial monads and operads | [第五十二章](52_polynomial_functors_species_analytic_functors_and_w_types.md) | 结构依赖 | Gambino-Kock, Kock | trees、operads、polynomial monads |
| EXT-153 | Analytic functors and species | [第五十二章](52_polynomial_functors_species_analytic_functors_and_w_types.md) | 结构依赖 | Joyal | species、symmetric group actions、combinatorial structures |
| EXT-154 | $\infty$-cosmos examples | [第五十三章](53_infinity_cosmoi_model_independent_infinity_category_theory.md) | 核心依赖 | Riehl-Verity, Joyal, Rezk, Bergner | quasi-categories、complete Segal spaces、simplicial categories |
| EXT-155 | $\infty$-cosmos adjunction equivalence theorem | [第五十三章](53_infinity_cosmoi_model_independent_infinity_category_theory.md) | 核心依赖 | Riehl-Verity | homotopy 2-category adjunctions、quasi-categorical adjunctions |
| EXT-156 | Modules and weighted limits in an $\infty$-cosmos | [第五十三章](53_infinity_cosmoi_model_independent_infinity_category_theory.md) | 结构依赖 | Riehl-Verity | collages、modules、Kan extensions、weighted limits |
| EXT-157 | Cofibrantly generated weak factorization systems | [第五十四章](54_orthogonality_factorization_systems_and_weak_factorization.md) | 结构依赖 | Quillen, Hovey, Riehl | small object argument、weak factorization systems |
| EXT-158 | Sketch model categories are locally presentable | [第五十五章](55_sketches_doctrines_and_categorical_theories.md) | 核心依赖 | Ehresmann, Adámek-Rosický, Makkai-Reyes | sketches、accessible categories、models of theories |
| EXT-159 | Enriched Cauchy completion and absolute weights | [第五十六章](56_idempotents_karoubi_envelopes_and_absolute_colimits.md) | 结构依赖 | Kelly, Street, Bénabou | absolute weighted colimits、Cauchy completion、Morita theory |

## 依赖链示例

### 从 quasi-category 到 Cartesian fibration

1. EXT-13 给出 quasi-category 的同伦理论。
2. EXT-14 和 EXT-15 给出同伦范畴与映射空间。
3. EXT-17 给出 marked simplicial sets over $S$ 的 Cartesian model structure。
4. EXT-19 把 Cartesian fibrations 与 $\operatorname{Fun}(S^{op},\mathcal{Cat}_\infty)$ 连接。
5. EXT-39 把 Cartesian sections 解释为 straightened functor 的极限。

### 从幺半范畴到高阶代数

1. EXT-02 允许幺半范畴中进行相干括号省略。
2. EXT-03 把小幺半范畴的结构延拓到预层。
3. 第十章的 enriched Yoneda 提供富范畴工具。
4. EXT-27 保证合适 presentable 幺半 $\infty$-范畴中代数对象范畴存在且可控。
5. EXT-29 和 EXT-31 给出谱上的环对象、模对象和相对张量积。
6. EXT-34 和 EXT-35 进入 $E_n$-中心、Morita 理论和因子化同调。
7. EXT-37 给出 Morita 语境下 fully dualizable objects 的可检验有限性条件。
8. EXT-36 把 fully dualizable objects 与 fully extended TFT 联系起来。

### 从 sheaf 到 higher topos

1. EXT-07 构造 ordinary sheaf topos。
2. EXT-08 给出 ordinary topos 的内在刻画。
3. 第 21 章把集合值 sheaf 条件替换为 space 值 descent。
4. EXT-24 给出 $\infty$-topos 的内在刻画，EXT-25 连接回 ordinary topos。
5. EXT-42 和 EXT-43 控制高阶覆盖、groupoid objects 与 hyperdescent 的技术边界。

### 从 presentable $\infty$-category 到 higher algebra

1. EXT-44 给出预层 $\infty$-范畴和 Yoneda 控制。
2. EXT-45 允许用 accessible localization 构造 sheaves、local objects 和稳定局部化。
3. EXT-46 给出 $\operatorname{Pr}^L$ 的张量积，从而支撑 presentable 幺半 $\infty$-category 和模范畴存在性。
4. EXT-47 和 EXT-48 把 profunctor/correspondence 复合连接到 Morita 理论的双模复合。
5. EXT-49 和 EXT-50 加入 companion/conjoint 与 Beck-Chevalley，支持 indexed categories 和高阶 base change。
6. EXT-51 至 EXT-53 控制稳定 presentable 范畴的局部化、商和紧对象行为。
7. EXT-54 至 EXT-58 把紧生成稳定 $k$-线性范畴落实为 dg 模范畴、dg enhancement 和 Morita 不变量。
8. EXT-59 至 EXT-63 把 base change、投影公式和对偶性组织为 sheaf 型几何理论中的六操作形式主义。
9. EXT-64 至 EXT-68 比较 relative categories、simplicial categories、quasi-categories 和 complete Segal spaces 的等价模型。
10. EXT-69 至 EXT-73 组织 dg quotient、localizing invariants 和 noncommutative motives。
11. EXT-74 至 EXT-78 把 recollement、t-结构和 Verdier 对偶落实为 perverse sheaves 理论。
12. EXT-79 至 EXT-83 把 Bousfield localization 专门化为 chromatic homotopy 的高度分层。
13. EXT-84 至 EXT-87 把 perverse sheaves 与 regular holonomic $D$-modules 通过 Riemann-Hilbert 对应起来。
14. EXT-88 至 EXT-91 把 presentable/stable $\infty$-范畴工具推进到 derived 和 spectral algebraic geometry。
15. EXT-92 至 EXT-94 给出 monadic/comonadic descent 的高阶重构机制。
16. EXT-95 至 EXT-97 给出经典和高阶 Tannaka 重构。
17. EXT-98 至 EXT-100 把 compact tensor triangulated categories 几何化为 Balmer spectra。
18. EXT-101 至 EXT-104 给出 trace methods 中 $THH$、$TC$ 和 $K$-theory 的桥梁。
19. EXT-105 至 EXT-108 把函子范畴中的非线性近似组织为 Goodwillie tower 和 derivatives。
20. EXT-109 至 EXT-113 把 $\mathbb A^1$-局部化、稳定化和六操作组织为 motivic homotopy theory。
21. EXT-114 至 EXT-116 把子对象纤维化、comprehension categories 和 $\infty$-topos 语义连接到依赖类型论与 univalence。
22. EXT-117 至 EXT-120 把 $E_n$-代数的局部结构通过因子化同调扩展到流形、trace 和非阿贝尔 Poincare 对偶。
23. EXT-121 至 EXT-124 把 sheaf/topos 技术推进到 condensed mathematics、solid tensor products 和解析 derived categories。
24. EXT-125 至 EXT-127 把逻辑语法、分类 topos 和 tripos-to-topos 组织为模型分类理论。
25. EXT-128 至 EXT-130 把 regular/exact completion 与关系 allegory 连接到 regular 逻辑。
26. EXT-131 至 EXT-133 把 cohesive topos、left exact modalities 和 modal type theory 接入高阶 topos 语义。
27. EXT-134 至 EXT-136 把层化空间的方向性编码为 exit-path $\infty$-categories，并分类 constructible sheaves。
28. EXT-137 至 EXT-141 把 higher Morita、trace、factorization homology 和 $E_n$-Koszul duality 连接为高阶代数的对偶性框架。
29. EXT-142 至 EXT-144 把模型范畴和 $\infty$-范畴的图形同伦范畴系统化为 derivator。
30. EXT-145 至 EXT-147 把 torsors、gerbes 和 higher stacks 接入非阿贝尔上同调。
31. EXT-148 至 EXT-150 把 effective descent、Galois structures 和 normal extensions 连接到抽象 Galois 对应。
32. EXT-151 至 EXT-153 把 polynomial functors、species、W-types 和 operads 连接到组合与类型论。
33. EXT-154 至 EXT-156 把 quasi-categories、Segal spaces 等模型统一到 $\infty$-cosmos 的模型无关语言。
34. EXT-157 至 EXT-159 只记录范畴论内部大型构造：小对象论证、sketch 可表现性和富 Cauchy completion。
