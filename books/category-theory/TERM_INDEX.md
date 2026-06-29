# 术语索引

本文档记录《范畴论》中核心术语的中文名、英文名、主要出现位置和用途。它不是定义替代；正式定义仍以正文为准。

## 集合论与基础大小

| 中文术语 | English | 位置 | 用途 |
|---|---|---|---|
| Grothendieck 宇宙 | Grothendieck universe | [附录 A](A_universes_and_size.md) | 控制集合、范畴和范畴的范畴的大小层级 |
| 小范畴 | small category | [第一章](01_categories_functors_natural_transformations.md), [附录 A](A_universes_and_size.md) | 对象和态射集合属于固定 universe 的范畴 |
| 局部小范畴 | locally small category | [附录 A](A_universes_and_size.md) | Hom 为集合但对象类可大的范畴 |

## 普通范畴论

| 中文术语 | English | 位置 | 用途 |
|---|---|---|---|
| 范畴 | category | [第一章](01_categories_functors_natural_transformations.md) | 对象、态射、复合、恒等态射 |
| 函子 | functor | [第一章](01_categories_functors_natural_transformations.md) | 保持对象、态射、复合和恒等态射的映射 |
| 自然变换 | natural transformation | [第一章](01_categories_functors_natural_transformations.md) | 比较两个函子 |
| 反范畴 | opposite category | [第一章](01_categories_functors_natural_transformations.md) | 反转态射方向 |
| 完全忠实 | fully faithful | [第一章](01_categories_functors_natural_transformations.md) | Hom 映射为双射 |
| 本质满 | essentially surjective | [第一章](01_categories_functors_natural_transformations.md) | 目标对象均同构于像中对象 |
| 范畴等价 | equivalence of categories | [第一章](01_categories_functors_natural_transformations.md) | 完全忠实且本质满的函子 |

## 表示性、极限与伴随

| 中文术语 | English | 位置 | 用途 |
|---|---|---|---|
| 终对象 | terminal object | [第二章](02_universal_properties_and_yoneda.md) | 到其态射唯一的对象 |
| 始对象 | initial object | [第二章](02_universal_properties_and_yoneda.md) | 从其出发态射唯一的对象 |
| 可表函子 | representable functor | [第二章](02_universal_properties_and_yoneda.md) | 由 Hom 函子表示的集合值函子 |
| 泛元素 | universal element | [第二章](02_universal_properties_and_yoneda.md) | 表示性对应的元素形式 |
| Yoneda 引理 | Yoneda lemma | [第二章](02_universal_properties_and_yoneda.md) | 元素与自然变换的自然双射 |
| 极限 | limit | [第三章](03_limits_and_colimits.md) | 锥范畴的终对象 |
| 余极限 | colimit | [第三章](03_limits_and_colimits.md) | 余锥范畴的始对象 |
| 伴随 | adjunction | [第四章](04_adjoint_functors.md) | Hom 自然同构、单位、余单位 |
| Kan 延拓 | Kan extension | [第六章](06_kan_extensions.md) | 沿函子改变定义域的泛构造 |
| 单子 | monad | [第七章](07_monads_and_algebras.md) | 自函子上的单位和乘法结构 |

## 结构性范畴论

| 中文术语 | English | 位置 | 用途 |
|---|---|---|---|
| 幺半范畴 | monoidal category | [第八章](08_monoidal_categories.md) | 内部张量积和单位对象 |
| 闭幺半范畴 | closed monoidal category | [第九章](09_closed_categories_and_day_convolution.md) | 张量-Hom 伴随 |
| Day 卷积 | Day convolution | [第九章](09_closed_categories_and_day_convolution.md) | 预层范畴上的幺半结构 |
| 富范畴 | enriched category | [第十章](10_enriched_categories.md) | Hom 集替换为 Hom 对象 |
| 富自然变换 | enriched natural transformation | [第十章](10_enriched_categories.md) | 由 end 给出的富 Hom 对象 |
| enriched Yoneda | enriched Yoneda lemma | [第十章](10_enriched_categories.md) | 富可表预层的 Yoneda 引理 |
| end | end | [第十一章](11_ends_and_coends.md) | 满足自然性条件的积 |
| coend | coend | [第十一章](11_ends_and_coends.md) | 按自然性关系商掉的余积 |
| 可表现范畴 | presentable category | [第十二章](12_presentable_and_accessible_categories.md) | 由紧对象和滤过余极限控制的大范畴 |
| 强生成子 | strong generator | [第十二章](12_presentable_and_accessible_categories.md) | 通过 Hom 集检测同构的小对象族 |
| Grothendieck 范畴 | Grothendieck category | [第十三章](13_exact_abelian_grothendieck_categories.md) | AB5 且有生成元的阿贝尔范畴 |
| coimage | coimage | [第十三章](13_exact_abelian_grothendieck_categories.md) | $\operatorname{coker}(\ker f)$，阿贝尔范畴中与 image 同构 |
| 正合函子 | exact functor | [第十三章](13_exact_abelian_grothendieck_categories.md), [第二十章](20_stable_infinity_categories_and_spectra.md) | 保持短正合列或稳定正合结构的函子 |
| Grothendieck topos | Grothendieck topos | [第十四章](14_sites_sheaves_and_topoi.md) | 小站点上的 sheaf 范畴 |
| separated 预层 | separated presheaf | [第十四章](14_sites_sheaves_and_topoi.md) | 局部相等推出全局相等的预层 |
| plus 构造 | plus construction | [第十四章](14_sites_sheaves_and_topoi.md) | 由覆盖匹配族构造 separated/sheaf 化的步骤 |
| 几何态射 | geometric morphism | [第十四章](14_sites_sheaves_and_topoi.md) | topoi 之间 inverse image 左正合的伴随对 |

## 同伦与高阶范畴论

| 中文术语 | English | 位置 | 用途 |
|---|---|---|---|
| 2-范畴 | 2-category | [第十五章](15_two_categories_and_bicategories.md) | 对象、1-态射、2-态射 |
| 双范畴 | bicategory | [第十五章](15_two_categories_and_bicategories.md) | 复合在相干同构下结合 |
| 模型范畴 | model category | [第十六章](16_model_categories_and_homotopy_categories.md) | 弱等价、纤维化、余纤维化 |
| 单纯集 | simplicial set | [第十七章](17_simplicial_sets_and_quasicategories.md) | $\Delta$ 上的预层 |
| 标准单纯形 | standard simplex | [第十七章](17_simplicial_sets_and_quasicategories.md) | 可表单纯集 $\Delta^n=\Delta(-,[n])$ |
| 内角 | inner horn | [第十七章](17_simplicial_sets_and_quasicategories.md) | quasi-category 中表达可复合性的 horn |
| quasi-category | quasi-category | [第十七章](17_simplicial_sets_and_quasicategories.md) | 满足 inner horn 填充的单纯集 |
| Kan 复形 | Kan complex | [第十七章](17_simplicial_sets_and_quasicategories.md) | 所有 horn 可填的单纯集，建模 spaces |
| Joyal 模型结构 | Joyal model structure | [第十七章](17_simplicial_sets_and_quasicategories.md), [附录 E](E_higher_categorical_technical_models.md) | 以 quasi-category 为 fibrant object 的模型结构 |
| join | join | [第十八章](18_limits_adjunctions_in_infinity_categories.md), [附录 E](E_higher_categorical_technical_models.md) | 定义 slice 和锥 |
| slice quasi-category | slice quasi-category | [第十八章](18_limits_adjunctions_in_infinity_categories.md), [附录 E](E_higher_categorical_technical_models.md) | 高阶锥和逗号对象 |
| 映射空间 | mapping space | [第十八章](18_limits_adjunctions_in_infinity_categories.md) | Hom 集的高阶替代 |
| 右映射空间 | right mapping space | [第十八章](18_limits_adjunctions_in_infinity_categories.md) | 用 $\Delta^{n+1}$ 和常值末面建模 $\operatorname{Map}_C(x,y)$ |
| 左映射空间 | left mapping space | [第十八章](18_limits_adjunctions_in_infinity_categories.md) | 右映射空间的对偶模型 |
| correspondence | correspondence | [第十八章](18_limits_adjunctions_in_infinity_categories.md) | 用 $C^{op}\times D\to\mathcal S$ 表示广义态射空间 |
| adjunction data | adjunction data | [第十八章](18_limits_adjunctions_in_infinity_categories.md) | 由 $\Delta^1$ 上双纤维对象编码伴随、单位和余单位 |
| walking adjunction | walking adjunction | [第十八章](18_limits_adjunctions_in_infinity_categories.md) | 由一个形式伴随生成的 2-范畴/高阶图形 |
| scaled nerve | scaled nerve | [第十八章](18_limits_adjunctions_in_infinity_categories.md), [附录 E](E_higher_categorical_technical_models.md) | 用标记 $2$-单纯形记录 $(\infty,2)$-相干数据 |
| Grothendieck construction | Grothendieck construction | [第十九章](19_cartesian_fibrations_and_straightening.md) | 由反变范畴值函子构造 ordinary fibration |
| straightening | straightening | [第十九章](19_cartesian_fibrations_and_straightening.md) | 把 Cartesian fibration 转换为 $\infty$-范畴值函子 |
| Cartesian fibration | Cartesian fibration | [第十九章](19_cartesian_fibrations_and_straightening.md) | 反变 $\infty$-范畴族的几何模型 |
| marked simplicial set | marked simplicial set | [第十九章](19_cartesian_fibrations_and_straightening.md), [附录 E](E_higher_categorical_technical_models.md) | 标记等价边或 Cartesian edges |
| Cartesian section | Cartesian section | [第十九章](19_cartesian_fibrations_and_straightening.md) | Cartesian fibration 中与限制函子相容的 section |
| 稳定 $\infty$-范畴 | stable $\infty$-category | [第二十章](20_stable_infinity_categories_and_spectra.md) | 有限极限和余极限兼容的 pointed $\infty$-范畴 |
| sequential prespectrum | sequential prespectrum | [第二十章](20_stable_infinity_categories_and_spectra.md) | pointed spaces 序列及结构映射 $\Sigma E_n\to E_{n+1}$ |
| $\Omega$-谱 | $\Omega$-spectrum | [第二十章](20_stable_infinity_categories_and_spectra.md) | 结构伴随 $E_n\to\Omega E_{n+1}$ 为等价的谱模型 |
| 映射谱 | mapping spectrum | [第二十章](20_stable_infinity_categories_and_spectra.md) | 稳定 $\infty$-范畴中的谱值 Hom 对象 |
| smash product | smash product | [第二十章](20_stable_infinity_categories_and_spectra.md) | $\mathbf{Sp}$ 的闭对称幺半乘法 |
| 环谱 | ring spectrum | [第二十章](20_stable_infinity_categories_and_spectra.md), [第二十二章](22_higher_algebra_and_infinity_operads.md) | $\mathbf{Sp}$ 中的 $E_1$-代数 |
| 悬挂 | suspension | [第二十章](20_stable_infinity_categories_and_spectra.md) | $\Sigma X=\operatorname{cofib}(X\to0)$ |
| 环路对象 | loop object | [第二十章](20_stable_infinity_categories_and_spectra.md) | $\Omega X=\operatorname{fib}(0\to X)$ |
| t-结构 | t-structure | [第二十章](20_stable_infinity_categories_and_spectra.md) | 稳定 $\infty$-范畴中的截断和 heart 结构 |
| heart | heart | [第二十章](20_stable_infinity_categories_and_spectra.md) | t-结构的阿贝尔范畴核心 |
| 长正合列 | long exact sequence | [第二十章](20_stable_infinity_categories_and_spectra.md) | t-结构 cohomology 对纤维-余纤维序列的正合输出 |
| 谱序列 | spectral sequence | [第二十章](20_stable_infinity_categories_and_spectra.md) | 滤过对象的 associated graded 到 cohomology 的计算工具 |
| exact couple | exact couple | [第二十章](20_stable_infinity_categories_and_spectra.md) | 产生谱序列的正合三角数据 |
| 有限滤过 | finite filtration | [第二十章](20_stable_infinity_categories_and_spectra.md) | 谱序列强收敛的基本充分条件 |
| $\infty$-topos | $\infty$-topos | [第二十一章](21_higher_topos_theory.md) | space 值 sheaf 理论 |
| Čech descent | Čech descent | [第二十一章](21_higher_topos_theory.md) | 用 Čech nerve 的同伦极限表达 sheaf 条件 |
| 超覆盖 | hypercover | [第二十一章](21_higher_topos_theory.md) | 逐维覆盖 matching object 的增广单纯对象 |
| 超下降 | hyperdescent | [第二十一章](21_higher_topos_theory.md) | 对所有超覆盖要求同伦极限下降 |
| $0$-截断对象 | 0-truncated object | [第二十一章](21_higher_topos_theory.md) | 映射对象离散的 higher topos 对象 |
| effective epimorphism | effective epimorphism | [第二十一章](21_higher_topos_theory.md) | Čech nerve 的几何实现恢复目标的覆盖态射 |
| Postnikov 塔 | Postnikov tower | [第二十一章](21_higher_topos_theory.md) | 由截断对象逐层近似高阶对象 |
| hypercompletion | hypercompletion | [第二十一章](21_higher_topos_theory.md) | 强制对象由 Postnikov 塔恢复的左正合局部化 |
| $\infty$-几何态射 | geometric morphism of $\infty$-topoi | [第二十一章](21_higher_topos_theory.md) | inverse image 左正合的 $\infty$-topos 伴随 |
| 预层 $\infty$-范畴 | presheaf $\infty$-category | [第二十三章](23_presentable_infinity_categories_and_localizations.md) | $\mathcal P(C)=\operatorname{Fun}(C^{op},\mathcal S)$ |
| presentable $\infty$-category | presentable $\infty$-category | [第二十三章](23_presentable_infinity_categories_and_localizations.md) | 可达且余完备的 $\infty$-范畴 |
| accessible localization | accessible localization | [第二十三章](23_presentable_infinity_categories_and_localizations.md) | 由可达局部化函子给出的反射子范畴 |
| Bousfield localization | Bousfield localization | [第二十三章](23_presentable_infinity_categories_and_localizations.md) | 由一族态射指定局部对象的局部化 |
| $\operatorname{Pr}^L$ | $\operatorname{Pr}^L$ | [第二十三章](23_presentable_infinity_categories_and_localizations.md) | presentable $\infty$-categories 与左伴随组成的 $\infty$-范畴 |
| profunctor | profunctor | [第二十四章](24_profunctors_cauchy_completion_and_correspondences.md) | 从 $\mathcal C$ 到 $\mathcal D$ 的广义态射 $\mathcal C^{op}\times\mathcal D\to\mathbf{Set}$ |
| Cauchy completion | Cauchy completion | [第二十四章](24_profunctors_cauchy_completion_and_correspondences.md) | 通过分裂幂等得到的 Karoubi 完备化 |
| 加权余极限 | weighted colimit | [第二十四章](24_profunctors_cauchy_completion_and_correspondences.md) | 由权重 $W$ 加权的余极限，常用 coend 表示 |
| $\infty$-correspondence | $\infty$-correspondence | [第二十四章](24_profunctors_cauchy_completion_and_correspondences.md) | space 值 profunctor 或高阶 span 型广义态射 |
| 富 profunctor | enriched profunctor | [第二十五章](25_enriched_profunctors_equipments_and_base_change.md) | $\mathcal V$-富范畴之间的 $\mathcal V$-值广义态射 |
| equipment | equipment | [第二十五章](25_enriched_profunctors_equipments_and_base_change.md) | 同时含垂直函子、水平 profunctor 和二重胞腔的双范畴结构 |
| companion | companion | [第二十五章](25_enriched_profunctors_equipments_and_base_change.md) | 垂直函子诱导的同向水平 profunctor |
| conjoint | conjoint | [第二十五章](25_enriched_profunctors_equipments_and_base_change.md) | 垂直函子诱导的反向水平 profunctor |
| Beck-Chevalley 条件 | Beck-Chevalley condition | [第二十五章](25_enriched_profunctors_equipments_and_base_change.md) | base change 比较态射为同构的 exact square 条件 |
| compact object | compact object | [第二十六章](26_compact_generation_brown_representability_and_bousfield_localization.md) | 映射函子保持滤过余极限的对象 |
| compactly generated | compactly generated | [第二十六章](26_compact_generation_brown_representability_and_bousfield_localization.md) | 由一小集紧对象检测并生成的稳定 presentable 范畴 |
| localizing subcategory | localizing subcategory | [第二十六章](26_compact_generation_brown_representability_and_bousfield_localization.md) | 稳定且对小余积封闭的全子范畴 |
| Verdier quotient | Verdier quotient | [第二十六章](26_compact_generation_brown_representability_and_bousfield_localization.md) | 把稳定子范畴对象强制为零的稳定商 |
| Brown 表示性 | Brown representability | [第二十六章](26_compact_generation_brown_representability_and_bousfield_localization.md) | cohomological functor 或伴随存在性的表示定理 |
| smashing localization | smashing localization | [第二十六章](26_compact_generation_brown_representability_and_bousfield_localization.md) | 由张量某个对象给出的 Bousfield 局部化 |
| dg 范畴 | dg category | [第二十七章](27_dg_categories_enhancements_and_derived_morita_theory.md) | 链复形范畴 $\operatorname{Ch}(k)$ 上的富范畴 |
| $H^0(\mathcal A)$ | homotopy category of a dg category | [第二十七章](27_dg_categories_enhancements_and_derived_morita_theory.md) | Hom 取 $0$ 次同调得到的普通范畴 |
| quasi-equivalence | quasi-equivalence | [第二十七章](27_dg_categories_enhancements_and_derived_morita_theory.md) | Hom 复形 quasi-isomorphism 且 $H^0$ 本质满的 dg 函子 |
| dg 模 | dg module | [第二十七章](27_dg_categories_enhancements_and_derived_morita_theory.md) | dg 函子 $\mathcal A^{op}\to\operatorname{Ch}(k)$ |
| perfect module | perfect module | [第二十七章](27_dg_categories_enhancements_and_derived_morita_theory.md) | 由可表 dg 模经有限稳定操作和 retract 生成的紧对象 |
| pretriangulated dg category | pretriangulated dg category | [第二十七章](27_dg_categories_enhancements_and_derived_morita_theory.md) | 可表模对悬挂和锥封闭的 dg 范畴 |
| dg enhancement | dg enhancement | [第二十七章](27_dg_categories_enhancements_and_derived_morita_theory.md) | 由 pretriangulated dg category 给出的三角或稳定 $\infty$-范畴增强 |
| Morita equivalence | Morita equivalence | [第二十七章](27_dg_categories_enhancements_and_derived_morita_theory.md) | 诱导导出模范畴等价的 dg 函子 |
| dg bimodule | dg bimodule | [第二十七章](27_dg_categories_enhancements_and_derived_morita_theory.md) | dg 函子 $\mathcal A^{op}\otimes\mathcal B\to\operatorname{Ch}(k)$ |
| Hochschild chains | Hochschild chains | [第二十七章](27_dg_categories_enhancements_and_derived_morita_theory.md) | 恒等 bimodule 的导出 trace |
| 六操作形式主义 | six functor formalism | [第二十八章](28_six_functor_formalism_base_change_and_projection_formula.md) | $f^*,f_*,f_!,f^!$ 与张量、内 Hom 的相干系统 |
| 稳定系数系统 | stable coefficient system | [第二十八章](28_six_functor_formalism_base_change_and_projection_formula.md) | $\mathcal B^{op}\to\operatorname{CAlg}(\operatorname{Pr}^L_{\operatorname{st}})$ |
| 基变换 | base change | [第二十八章](28_six_functor_formalism_base_change_and_projection_formula.md) | Cartesian 方块上 $g^*f_*\simeq f'_*g'^*$ 或 $g^*f_!\simeq f'_!g'^*$ |
| 投影公式 | projection formula | [第二十八章](28_six_functor_formalism_base_change_and_projection_formula.md) | $f_!(A\otimes f^*B)\simeq f_!A\otimes B$ |
| extension by zero | extension by zero | [第二十八章](28_six_functor_formalism_base_change_and_projection_formula.md) | 开嵌入 $j$ 的非常推前 $j_!$ |
| recollement | recollement | [第二十八章](28_six_functor_formalism_base_change_and_projection_formula.md) | 开闭分解给出的局部化和粘合结构 |
| dualizing object | dualizing object | [第二十八章](28_six_functor_formalism_base_change_and_projection_formula.md) | $\omega_X=p_X^!\mathbb 1$ |
| Verdier 对偶 | Verdier duality | [第二十八章](28_six_functor_formalism_base_change_and_projection_formula.md) | $\mathbb D_X(K)=\underline{\operatorname{Hom}}_X(K,\omega_X)$ |
| purity | purity | [第二十八章](28_six_functor_formalism_base_change_and_projection_formula.md) | 把 $f^!$ 与 $f^*$、相对 dualizing object 和平移联系的定理 |
| 相对范畴 | relative category | [第二十九章](29_relative_categories_simplicial_localization_and_model_comparisons.md) | 范畴 $\mathcal C$ 加指定 weak equivalences 子范畴 $W$ |
| $\infty$-局部化 | $\infty$-categorical localization | [第二十九章](29_relative_categories_simplicial_localization_and_model_comparisons.md) | 把 $W$ 变为等价且满足 $\infty$-范畴泛性质的局部化 |
| saturated weak equivalences | saturated weak equivalences | [第二十九章](29_relative_categories_simplicial_localization_and_model_comparisons.md) | 正是在局部化后成为同构或等价的弱等价类 |
| 单纯范畴 | simplicial category | [第二十九章](29_relative_categories_simplicial_localization_and_model_comparisons.md) | $\mathbf{sSet}$-富范畴 |
| Dwyer-Kan equivalence | Dwyer-Kan equivalence | [第二十九章](29_relative_categories_simplicial_localization_and_model_comparisons.md) | 映射空间弱等价且同伦范畴本质满的单纯函子 |
| simplicial localization | simplicial localization | [第二十九章](29_relative_categories_simplicial_localization_and_model_comparisons.md) | relative category 的单纯范畴值局部化 |
| coherent nerve | coherent nerve | [第二十九章](29_relative_categories_simplicial_localization_and_model_comparisons.md) | 从 simplicial categories 到 simplicial sets 的同伦相干 nerve |
| complete Segal space | complete Segal space | [第二十九章](29_relative_categories_simplicial_localization_and_model_comparisons.md) | 满足 Segal 条件和 completeness 条件的 simplicial space |
| Rezk nerve | Rezk nerve | [第二十九章](29_relative_categories_simplicial_localization_and_model_comparisons.md) | relative category 到 complete Segal space 模型的 nerve |
| $\infty$-operad | $\infty$-operad | [第二十二章](22_higher_algebra_and_infinity_operads.md) | 高阶代数运算和相干性 |
| Segal 条件 | Segal condition | [第二十二章](22_higher_algebra_and_infinity_operads.md) | 把 $\langle n\rangle$ 上对象识别为 $n$ 个颜色列表 |
| 多重映射空间 | multimorphism space | [第二十二章](22_higher_algebra_and_infinity_operads.md) | $\infty$-operad 中多输入一输出运算的空间 |
| active 态射 | active morphism | [第二十二章](22_higher_algebra_and_infinity_operads.md) | $\mathbf{Fin}_*$ 中保留所有非基点输入的运算型态射 |
| inert 态射 | inert morphism | [第二十二章](22_higher_algebra_and_infinity_operads.md) | $\mathbf{Fin}_*$ 中选取输入槽的投影型态射 |
| $E_n$-代数 | $E_n$-algebra | [第二十二章](22_higher_algebra_and_infinity_operads.md) | 介于结合与同伦交换之间的代数结构 |
| 模 $\infty$-范畴 | module $\infty$-category | [第二十二章](22_higher_algebra_and_infinity_operads.md) | $E_1$-代数上的左模、右模和双模 |
| bar 构造 | bar construction | [第二十二章](22_higher_algebra_and_infinity_operads.md) | 构造相对张量积的单纯对象 |
| 相对张量积 | relative tensor product | [第二十二章](22_higher_algebra_and_infinity_operads.md) | $M\otimes_A N$ 的同伦平衡张量积 |
| Morita $\infty$-范畴 | Morita $\infty$-category | [第二十二章](22_higher_algebra_and_infinity_operads.md) | 以代数为对象、双模为态射、相对张量积为复合 |
| 单位双模 | unit bimodule | [第二十二章](22_higher_algebra_and_infinity_operads.md) | Morita $\infty$-范畴中的恒等 1-态射 |
| 中心 | center | [第二十二章](22_higher_algebra_and_infinity_operads.md) | 代数作为双模的 endomorphism object |
| 因子化同调 | factorization homology | [第二十二章](22_higher_algebra_and_infinity_operads.md) | 将 $E_n$-代数沿 $n$-流形局部到整体粘合 |
| smooth/proper | smooth/proper | [第二十二章](22_higher_algebra_and_infinity_operads.md) | Morita 理论中刻画可对偶性的有限性条件 |
| Frobenius 代数 | Frobenius algebra | [第二十二章](22_higher_algebra_and_infinity_operads.md) | 普通二维 TFT 的代数数据 |
| fully dualizable | fully dualizable | [第二十二章](22_higher_algebra_and_infinity_operads.md) | 高维场论中可赋给点的完全可对偶对象 |
| cobordism hypothesis | cobordism hypothesis | [第二十二章](22_higher_algebra_and_infinity_operads.md) | fully extended TFT 由点值分类的定理 |
