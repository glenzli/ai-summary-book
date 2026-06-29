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
