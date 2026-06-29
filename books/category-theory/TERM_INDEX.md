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
| 骨架 | skeleton | [第一章](01_categories_functors_natural_transformations.md) | 每个同构类选一个代表的全子范畴 |

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
| 创造极限 | creates limits | [第三章](03_limits_and_colimits.md) | 底层极限唯一提升为源范畴极限的性质 |
| 共尾函子 | final functor | [第三章](03_limits_and_colimits.md) | 不改变余极限的指标函子 |
| 始函子 | initial functor | [第三章](03_limits_and_colimits.md) | 不改变极限的指标函子 |
| 伴随 | adjunction | [第四章](04_adjoint_functors.md) | Hom 自然同构、单位、余单位 |
| Galois connection | Galois connection | [第四章](04_adjoint_functors.md) | 偏序范畴中的伴随 |
| 反射子范畴 | reflective subcategory | [第四章](04_adjoint_functors.md) | 包含函子有左伴随的全子范畴 |
| 余反射子范畴 | coreflective subcategory | [第四章](04_adjoint_functors.md) | 包含函子有右伴随的全子范畴 |
| Kan 延拓 | Kan extension | [第六章](06_kan_extensions.md) | 沿函子改变定义域的泛构造 |
| 单子 | monad | [第七章](07_monads_and_algebras.md) | 自函子上的单位和乘法结构 |
| 幂等单子 | idempotent monad | [第七章](07_monads_and_algebras.md) | 乘法为自然同构的单子，常来自反射子范畴 |
| Kleisli 范畴 | Kleisli category | [第七章](07_monads_and_algebras.md) | 单子的效应态射范畴 |
| Eilenberg-Moore 范畴 | Eilenberg-Moore category | [第七章](07_monads_and_algebras.md) | 单子代数及其同态范畴 |

## 结构性范畴论

| 中文术语 | English | 位置 | 用途 |
|---|---|---|---|
| 幺半范畴 | monoidal category | [第八章](08_monoidal_categories.md) | 内部张量积和单位对象 |
| 辫幺半范畴 | braided monoidal category | [第八章](08_monoidal_categories.md) | 带相干交换约束的幺半范畴 |
| 对称幺半范畴 | symmetric monoidal category | [第八章](08_monoidal_categories.md) | 辫子平方为恒等的幺半范畴 |
| 松幺半函子 | lax monoidal functor | [第八章](08_monoidal_categories.md) | 保持张量到指定结构态射的函子 |
| 代数对象 | algebra object | [第八章](08_monoidal_categories.md), [第二十二章](22_higher_algebra_and_infinity_operads.md) | 幺半范畴或幺半 $\infty$-范畴内部的含幺结合乘法对象 |
| 闭幺半范畴 | closed monoidal category | [第九章](09_closed_categories_and_day_convolution.md) | 张量-Hom 伴随 |
| 笛卡尔闭范畴 | cartesian closed category | [第九章](09_closed_categories_and_day_convolution.md) | 有有限积且积函子有指数右伴随的范畴 |
| 评价态射 | evaluation morphism | [第九章](09_closed_categories_and_day_convolution.md) | 内部 Hom 的伴随余单位型态射 |
| Day 卷积 | Day convolution | [第九章](09_closed_categories_and_day_convolution.md) | 预层范畴上的幺半结构 |
| 富范畴 | enriched category | [第十章](10_enriched_categories.md) | Hom 集替换为 Hom 对象 |
| 富满忠实 | enriched fully faithful | [第十章](10_enriched_categories.md) | Hom 对象比较态射均为同构的富函子 |
| 富自然变换 | enriched natural transformation | [第十章](10_enriched_categories.md) | 由 end 给出的富 Hom 对象 |
| enriched Yoneda | enriched Yoneda lemma | [第十章](10_enriched_categories.md) | 富可表预层的 Yoneda 引理 |
| 加权极限 | weighted limit | [第十章](10_enriched_categories.md) | 富范畴中由权重表示的极限 |
| 张量对象 | tensor of an enriched object | [第十章](10_enriched_categories.md) | 富范畴中由 $V\odot A$ 表示的加权余极限特例 |
| 余张量对象 | cotensor of an enriched object | [第十章](10_enriched_categories.md) | 富范畴中由 $A^V$ 表示的加权极限特例 |
| end | end | [第十一章](11_ends_and_coends.md) | 满足自然性条件的积 |
| coend | coend | [第十一章](11_ends_and_coends.md) | 按自然性关系商掉的余积 |
| co-Yoneda 引理 | co-Yoneda lemma | [第十一章](11_ends_and_coends.md) | 把预层表示为可表预层的 coend |
| Fubini for ends/coends | Fubini theorem for ends/coends | [第十一章](11_ends_and_coends.md) | 在存在性条件下重排迭代 end 或 coend |
| $\kappa$-滤过范畴 | $\kappa$-filtered category | [第十二章](12_presentable_and_accessible_categories.md) | 控制少于 $\kappa$ 大小图形的共同上界 |
| $\kappa$-紧对象 | $\kappa$-presentable object | [第十二章](12_presentable_and_accessible_categories.md) | Hom 函子保持 $\kappa$-滤过余极限的对象 |
| 可达范畴 | accessible category | [第十二章](12_presentable_and_accessible_categories.md) | 由小的紧对象族经滤过余极限生成的范畴 |
| 可表现范畴 | presentable category | [第十二章](12_presentable_and_accessible_categories.md) | 由紧对象和滤过余极限控制的大范畴 |
| 生成族 | generating family | [第五章](05_representables_density_generators.md) | 通过从小对象出发的态射检测平行态射 |
| 生成元 | generator | [第五章](05_representables_density_generators.md) | 单对象生成族 |
| 投射对象 | projective object | [第五章](05_representables_density_generators.md) | 对满射具有提升性质的对象 |
| 强生成子 | strong generator | [第十二章](12_presentable_and_accessible_categories.md) | 通过 Hom 集检测同构的小对象族 |
| 阿贝尔范畴 | abelian category | [第十三章](13_exact_abelian_grothendieck_categories.md) | 有核余核且 image/coimage 同构的加性范畴 |
| 核 | kernel | [第十三章](13_exact_abelian_grothendieck_categories.md) | 等化态射与零态射的泛对象 |
| 余核 | cokernel | [第十三章](13_exact_abelian_grothendieck_categories.md) | 余等化态射与零态射的泛对象 |
| Grothendieck 范畴 | Grothendieck category | [第十三章](13_exact_abelian_grothendieck_categories.md) | AB5 且有生成元的阿贝尔范畴 |
| coimage | coimage | [第十三章](13_exact_abelian_grothendieck_categories.md) | $\operatorname{coker}(\ker f)$，阿贝尔范畴中与 image 同构 |
| 正合函子 | exact functor | [第十三章](13_exact_abelian_grothendieck_categories.md), [第二十章](20_stable_infinity_categories_and_spectra.md) | 保持短正合列或稳定正合结构的函子 |
| Grothendieck 拓扑 | Grothendieck topology | [第十四章](14_sites_sheaves_and_topoi.md) | 用覆盖筛公理化覆盖关系 |
| 站点 | site | [第十四章](14_sites_sheaves_and_topoi.md) | 带 Grothendieck 拓扑的小范畴 |
| sheaf 化 | sheafification | [第十四章](14_sites_sheaves_and_topoi.md) | 从预层到 sheaf 子范畴的左正合反射 |
| Grothendieck topos | Grothendieck topos | [第十四章](14_sites_sheaves_and_topoi.md) | 小站点上的 sheaf 范畴 |
| separated 预层 | separated presheaf | [第十四章](14_sites_sheaves_and_topoi.md) | 局部相等推出全局相等的预层 |
| plus 构造 | plus construction | [第十四章](14_sites_sheaves_and_topoi.md) | 由覆盖匹配族构造 separated/sheaf 化的步骤 |
| 几何态射 | geometric morphism | [第十四章](14_sites_sheaves_and_topoi.md) | topoi 之间 inverse image 左正合的伴随对 |

## 同伦与高阶范畴论

| 中文术语 | English | 位置 | 用途 |
|---|---|---|---|
| 2-范畴 | 2-category | [第十五章](15_two_categories_and_bicategories.md) | 对象、1-态射、2-态射 |
| 双范畴 | bicategory | [第十五章](15_two_categories_and_bicategories.md) | 复合在相干同构下结合 |
| 2-函子 | 2-functor | [第十五章](15_two_categories_and_bicategories.md) | 严格保持 1-复合和 2-复合的高阶函子 |
| 伪函子 | pseudofunctor | [第十五章](15_two_categories_and_bicategories.md) | 在相干同构下保持复合的双范畴函子 |
| 模型范畴 | model category | [第十六章](16_model_categories_and_homotopy_categories.md) | 弱等价、纤维化、余纤维化 |
| 相对范畴 | relative category | [第十六章](16_model_categories_and_homotopy_categories.md), [第二十九章](29_relative_categories_simplicial_localization_and_model_comparisons.md) | 范畴 $\mathcal C$ 加指定 weak equivalences 子范畴 $W$ |
| 离散模型结构 | discrete model structure | [第十六章](16_model_categories_and_homotopy_categories.md) | 弱等价为同构、所有态射同时为纤维化和余纤维化 |
| 单纯集 | simplicial set | [第十七章](17_simplicial_sets_and_quasicategories.md) | $\Delta$ 上的预层 |
| nerve | nerve | [第十七章](17_simplicial_sets_and_quasicategories.md) | 把普通范畴嵌入单纯集的全忠实构造 |
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
| 局部等价 | local equivalence | [第二十三章](23_presentable_infinity_categories_and_localizations.md) | 被局部化函子送为等价的态射 |
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
| $\infty$-局部化 | $\infty$-categorical localization | [第二十九章](29_relative_categories_simplicial_localization_and_model_comparisons.md) | 把 $W$ 变为等价且满足 $\infty$-范畴泛性质的局部化 |
| saturated weak equivalences | saturated weak equivalences | [第二十九章](29_relative_categories_simplicial_localization_and_model_comparisons.md) | 正是在局部化后成为同构或等价的弱等价类 |
| 单纯范畴 | simplicial category | [第二十九章](29_relative_categories_simplicial_localization_and_model_comparisons.md) | $\mathbf{sSet}$-富范畴 |
| Dwyer-Kan equivalence | Dwyer-Kan equivalence | [第二十九章](29_relative_categories_simplicial_localization_and_model_comparisons.md) | 映射空间弱等价且同伦范畴本质满的单纯函子 |
| simplicial localization | simplicial localization | [第二十九章](29_relative_categories_simplicial_localization_and_model_comparisons.md) | relative category 的单纯范畴值局部化 |
| coherent nerve | coherent nerve | [第二十九章](29_relative_categories_simplicial_localization_and_model_comparisons.md) | 从 simplicial categories 到 simplicial sets 的同伦相干 nerve |
| complete Segal space | complete Segal space | [第二十九章](29_relative_categories_simplicial_localization_and_model_comparisons.md) | 满足 Segal 条件和 completeness 条件的 simplicial space |
| Rezk nerve | Rezk nerve | [第二十九章](29_relative_categories_simplicial_localization_and_model_comparisons.md) | relative category 到 complete Segal space 模型的 nerve |
| exact sequence of stable categories | exact sequence of stable categories | [第三十章](30_dg_quotients_localizing_invariants_and_noncommutative_motives.md) | $A\to B\to C$ 中 $C$ 为 $B/A$ 幂等完备化的序列 |
| flasque stable category | flasque stable category | [第三十章](30_dg_quotients_localizing_invariants_and_noncommutative_motives.md) | 满足 $\operatorname{id}\oplus T\simeq T$ 的 Eilenberg swindle 条件 |
| dg quotient | dg quotient | [第三十章](30_dg_quotients_localizing_invariants_and_noncommutative_motives.md) | 把 dg 子范畴对象收缩为零的 dg 局部化 |
| Drinfeld quotient | Drinfeld quotient | [第三十章](30_dg_quotients_localizing_invariants_and_noncommutative_motives.md) | 通过添加收缩元构造的 dg quotient 模型 |
| additive invariant | additive invariant | [第三十章](30_dg_quotients_localizing_invariants_and_noncommutative_motives.md) | 把 split-exact sequences 送为直和分解的不变量 |
| localizing invariant | localizing invariant | [第三十章](30_dg_quotients_localizing_invariants_and_noncommutative_motives.md) | 把 exact sequences 送为纤维序列的不变量 |
| noncommutative motive | noncommutative motive | [第三十章](30_dg_quotients_localizing_invariants_and_noncommutative_motives.md) | 普遍表示 additive 或 localizing invariants 的 motive 对象 |
| 可构造导出范畴 | constructible derived category | [第三十一章](31_perverse_sheaves_recollement_and_t_structures.md) | 层化空间上 cohomology sheaves 沿 stratum 局部常值的导出范畴 |
| perverse t-结构 | perverse t-structure | [第三十一章](31_perverse_sheaves_recollement_and_t_structures.md) | 由支撑与余支撑维数条件定义的 t-结构 |
| perverse sheaf | perverse sheaf | [第三十一章](31_perverse_sheaves_recollement_and_t_structures.md) | perverse t-结构 heart 中的对象 |
| 中间延拓 | intermediate extension | [第三十一章](31_perverse_sheaves_recollement_and_t_structures.md) | $j_{!*}P=\operatorname{im}({}^pj_!P\to{}^pj_*P)$ |
| nearby cycles | nearby cycles | [第三十一章](31_perverse_sheaves_recollement_and_t_structures.md) | 记录一般纤维靠近特殊纤维行为的函子 |
| vanishing cycles | vanishing cycles | [第三十一章](31_perverse_sheaves_recollement_and_t_structures.md) | 记录退化时消失同调的函子 |
| Bousfield class | Bousfield class | [第三十二章](32_chromatic_homotopy_bousfield_lattices_and_telescope_conjecture.md) | 由 $E$-acyclic spectra 组成的等价类 $\langle E\rangle$ |
| Bousfield lattice | Bousfield lattice | [第三十二章](32_chromatic_homotopy_bousfield_lattices_and_telescope_conjecture.md) | Bousfield classes 按可见性形成的偏序结构 |
| Morava $K$-theory | Morava $K$-theory | [第三十二章](32_chromatic_homotopy_bousfield_lattices_and_telescope_conjecture.md) | 检测 chromatic height 的谱 $K(n)$ |
| chromatic type | chromatic type | [第三十二章](32_chromatic_homotopy_bousfield_lattices_and_telescope_conjecture.md) | 有限 $p$-local 谱第一个非零 Morava $K$ 高度 |
| telescope | telescope | [第三十二章](32_chromatic_homotopy_bousfield_lattices_and_telescope_conjecture.md) | $v_n$-self map 迭代余极限得到的周期谱 |
| telescope conjecture | telescope conjecture | [第三十二章](32_chromatic_homotopy_bousfield_lattices_and_telescope_conjecture.md) | 有限局部化与 telescope 局部化比较的猜想 |
| chromatic fracture square | chromatic fracture square | [第三十二章](32_chromatic_homotopy_bousfield_lattices_and_telescope_conjecture.md) | 用低高度和 $K(n)$-local 部分粘合 $L_nX$ 的拉回方块 |
| $D$-module | $D$-module | [第三十三章](33_d_modules_riemann_hilbert_and_de_rham_functors.md) | 微分算子层 $D_X$ 上的模 |
| flat connection | flat connection | [第三十三章](33_d_modules_riemann_hilbert_and_de_rham_functors.md) | 曲率为零的联络，对应适当条件下的左 $D_X$-模 |
| characteristic variety | characteristic variety | [第三十三章](33_d_modules_riemann_hilbert_and_de_rham_functors.md) | coherent $D_X$-module 的 associated graded 支撑 |
| holonomic $D$-module | holonomic $D$-module | [第三十三章](33_d_modules_riemann_hilbert_and_de_rham_functors.md) | characteristic variety 维数最小的 coherent $D_X$-module |
| Riemann-Hilbert correspondence | Riemann-Hilbert correspondence | [第三十三章](33_d_modules_riemann_hilbert_and_de_rham_functors.md) | regular holonomic $D$-modules 与可构造 sheaves/perverse sheaves 的等价 |
| de Rham functor | de Rham functor | [第三十三章](33_d_modules_riemann_hilbert_and_de_rham_functors.md) | 把 $D$-module 送到 de Rham complex 的函子 |
| Kashiwara equivalence | Kashiwara equivalence | [第三十三章](33_d_modules_riemann_hilbert_and_de_rham_functors.md) | 闭嵌入上 $D_Z$-modules 与支撑在 $Z$ 的 $D_X$-modules 的等价 |
| derived affine scheme | derived affine scheme | [第三十四章](34_derived_algebraic_geometry_cotangent_complexes_and_spectral_stacks.md) | connective $E_\infty$-ring 的反对象 $\operatorname{Spec}A$ |
| derived stack | derived stack | [第三十四章](34_derived_algebraic_geometry_cotangent_complexes_and_spectral_stacks.md) | 满足超下降的 derived affine 上 space 值 functor |
| $\operatorname{QCoh}$ | $\operatorname{QCoh}$ | [第三十四章](34_derived_algebraic_geometry_cotangent_complexes_and_spectral_stacks.md) | derived stack 上的 quasi-coherent complexes 稳定 $\infty$-范畴 |
| cotangent complex | cotangent complex | [第三十四章](34_derived_algebraic_geometry_cotangent_complexes_and_spectral_stacks.md) | 表示导子空间的模对象 |
| formal moduli problem | formal moduli problem | [第三十四章](34_derived_algebraic_geometry_cotangent_complexes_and_spectral_stacks.md) | Artinian derived algebras 上满足 Schlessinger 型拉回条件的 functor |
| $\operatorname{IndCoh}$ | $\operatorname{IndCoh}$ | [第三十四章](34_derived_algebraic_geometry_cotangent_complexes_and_spectral_stacks.md) | coherent sheaves 的 Ind 型增强，适合 Grothendieck duality |
| singular support | singular support | [第三十四章](34_derived_algebraic_geometry_cotangent_complexes_and_spectral_stacks.md) | IndCoh 对象在奇异空间中的微局部支撑条件 |
| Barr-Beck-Lurie 定理 | Barr-Beck-Lurie theorem | [第三十五章](35_barr_beck_lurie_monadicity_and_descent.md) | 用保守性和几何实现保持性判定 monadicity 的高阶定理 |
| comparison functor | comparison functor | [第三十五章](35_barr_beck_lurie_monadicity_and_descent.md) | 从伴随右侧范畴到 monad 代数范畴的比较函子 |
| monadic functor | monadic functor | [第三十五章](35_barr_beck_lurie_monadicity_and_descent.md) | 通过 monad 代数恢复源范畴的右伴随 |
| comonadic descent | comonadic descent | [第三十五章](35_barr_beck_lurie_monadicity_and_descent.md) | 通过余单子 coalgebras 或 Cech totalization 恢复对象 |
| Cech nerve | Cech nerve | [第三十五章](35_barr_beck_lurie_monadicity_and_descent.md) | 覆盖 $U\to X$ 的迭代纤维积单纯对象 |
| Tannakian category | Tannakian category | [第三十六章](36_tannaka_duality_affine_group_schemes_and_higher_reconstruction.md) | 刚性 $k$-线性阿贝尔张量范畴配 faithful exact fiber functor |
| fiber functor | fiber functor | [第三十六章](36_tannaka_duality_affine_group_schemes_and_higher_reconstruction.md) | 到有限维向量空间的 faithful exact 对称幺半函子 |
| Tannaka duality | Tannaka duality | [第三十六章](36_tannaka_duality_affine_group_schemes_and_higher_reconstruction.md) | 由表示范畴及 fiber functor 重构群或栈 |
| matrix coefficient coalgebra | matrix coefficient coalgebra | [第三十六章](36_tannaka_duality_affine_group_schemes_and_higher_reconstruction.md) | coend $\int^X\omega(X)^\vee\otimes\omega(X)$ |
| classifying stack | classifying stack | [第三十六章](36_tannaka_duality_affine_group_schemes_and_higher_reconstruction.md) | 把 $A$ 送到 $G$-torsors 的栈 $BG$ |
| tt-category | tensor triangulated category | [第三十七章](37_tensor_triangular_geometry_balmer_spectra_and_support.md) | 带精确对称幺半结构的小幂等完备三角范畴 |
| thick tensor ideal | thick tensor ideal | [第三十七章](37_tensor_triangular_geometry_balmer_spectra_and_support.md) | 对张量任意对象封闭的厚子范畴 |
| prime tensor ideal | prime tensor ideal | [第三十七章](37_tensor_triangular_geometry_balmer_spectra_and_support.md) | 满足 $x\otimes y\in\mathfrak p$ 蕴含 $x\in\mathfrak p$ 或 $y\in\mathfrak p$ 的 thick tensor ideal |
| Balmer spectrum | Balmer spectrum | [第三十七章](37_tensor_triangular_geometry_balmer_spectra_and_support.md) | prime thick tensor ideals 的拓扑空间 $\operatorname{Spc}(T)$ |
| Thomason subset | Thomason subset | [第三十七章](37_tensor_triangular_geometry_balmer_spectra_and_support.md) | quasi-compact opens 的补的并 |
| $THH$ | topological Hochschild homology | [第三十八章](38_topological_hochschild_homology_cyclotomic_trace_and_tc.md) | 稳定范畴或环谱的谱值 Hochschild trace |
| cyclotomic spectrum | cyclotomic spectrum | [第三十八章](38_topological_hochschild_homology_cyclotomic_trace_and_tc.md) | 带圆作用和 Frobenius/Tate 结构的谱 |
| $TC$ | topological cyclic homology | [第三十八章](38_topological_hochschild_homology_cyclotomic_trace_and_tc.md) | 从 cyclotomic spectrum 取出的固定点型不变量 |
| cyclotomic trace | cyclotomic trace | [第三十八章](38_topological_hochschild_homology_cyclotomic_trace_and_tc.md) | 自然变换 $K\to TC$ |
| trace methods | trace methods | [第三十八章](38_topological_hochschild_homology_cyclotomic_trace_and_tc.md) | 用 $THH/TC$ 和 cyclotomic trace 研究 $K$-理论的方法 |
| Goodwillie calculus | Goodwillie calculus | [第三十九章](39_goodwillie_calculus_excisive_functors_and_derivatives.md) | 用 excisive functors 和 Taylor tower 近似同伦函子的理论 |
| $n$-excisive functor | $n$-excisive functor | [第三十九章](39_goodwillie_calculus_excisive_functors_and_derivatives.md) | 把 strongly cocartesian $(n+1)$-cubes 送到 cartesian cubes 的函子 |
| Goodwillie tower | Goodwillie tower | [第三十九章](39_goodwillie_calculus_excisive_functors_and_derivatives.md) | $\cdots\to P_nF\to P_{n-1}F\to\cdots$ 的多项式近似塔 |
| cross-effect | cross-effect | [第三十九章](39_goodwillie_calculus_excisive_functors_and_derivatives.md) | 由多变量楔和立方的全纤维提取的非线性部分 |
| Goodwillie derivative | Goodwillie derivative | [第三十九章](39_goodwillie_calculus_excisive_functors_and_derivatives.md) | 控制 homogeneous layer 的带对称群作用谱 $\partial_nF$ |
| motivic space | motivic space | [第四十章](40_motivic_homotopy_a1_localization_and_six_operations.md) | Nisnevich sheaves 关于 $\mathbb A^1$-投影局部化后的对象 |
| $\mathbb A^1$-invariance | $\mathbb A^1$-invariance | [第四十章](40_motivic_homotopy_a1_localization_and_six_operations.md) | $F(X)\simeq F(X\times\mathbb A^1)$ 的同伦不变性 |
| stable motivic homotopy category | stable motivic homotopy category | [第四十章](40_motivic_homotopy_a1_localization_and_six_operations.md) | $T$-spectra 形成的稳定范畴 $\mathbf{SH}(S)$ |
| Thom space | Thom space | [第四十章](40_motivic_homotopy_a1_localization_and_six_operations.md) | 向量丛 $V$ 的商 $V/(V\setminus X)$ |
| motivic Eilenberg-Mac Lane spectrum | motivic Eilenberg-Mac Lane spectrum | [第四十章](40_motivic_homotopy_a1_localization_and_six_operations.md) | 表示 motivic cohomology 的谱 $H\mathbb Z$ |
| 子对象纤维化 | subobject fibration | [第四十一章](41_categorical_logic_dependent_type_theory_and_univalence.md) | $X\mapsto\operatorname{Sub}(X)$ 的反变谓词语义 |
| regular category | regular category | [第四十一章](41_categorical_logic_dependent_type_theory_and_univalence.md) | 有有限极限、image factorization 且 regular epis pullback 稳定的范畴 |
| Heyting category | Heyting category | [第四十一章](41_categorical_logic_dependent_type_theory_and_univalence.md) | 子对象格带 Heyting implication 并与替换相容的范畴 |
| locally Cartesian closed category | locally Cartesian closed category | [第四十一章](41_categorical_logic_dependent_type_theory_and_univalence.md) | 每个 slice 都 Cartesian closed 的有限极限范畴 |
| comprehension category | comprehension category | [第四十一章](41_categorical_logic_dependent_type_theory_and_univalence.md) | 用 fibration 和上下文扩张解释依赖类型的范畴结构 |
| univalent universe | univalent universe | [第四十一章](41_categorical_logic_dependent_type_theory_and_univalence.md) | 满足相等类型等价于等价类型的 universe |
| 几何逻辑 | geometric logic | [第四十一章](41_categorical_logic_dependent_type_theory_and_univalence.md) | 由有限合取、任意析取和存在量词生成的逻辑片段 |
| $\operatorname{Disk}_n$ | little disks category | [第四十二章](42_factorization_homology_en_algebras_and_nonabelian_poincare_duality.md) | 有限个 $\mathbb R^n$ 不交并及嵌入组成的对称幺半 $\infty$-范畴 |
| 因子化同调 | factorization homology | [第四十二章](42_factorization_homology_en_algebras_and_nonabelian_poincare_duality.md) | $E_n$-代数沿流形的对称幺半左 Kan 延拓 |
| excision | excision | [第四十二章](42_factorization_homology_en_algebras_and_nonabelian_poincare_duality.md) | 把 collar-gluing 下的因子化同调表达为相对张量积 |
| 非阿贝尔 Poincare 对偶 | nonabelian Poincare duality | [第四十二章](42_factorization_homology_en_algebras_and_nonabelian_poincare_duality.md) | $\int_MA$ 与 $\operatorname{Map}_c(M,B^nA)$ 的等价 |
| factorization algebra | factorization algebra | [第四十二章](42_factorization_homology_en_algebras_and_nonabelian_poincare_duality.md) | 对不交开集多重乘法并满足 Weiss descent 的局部到整体结构 |
| condensed set | condensed set | [第四十三章](43_condensed_sets_solid_modules_and_analytic_categories.md) | profinite 站点上的 set-valued sheaf |
| condensed abelian group | condensed abelian group | [第四十三章](43_condensed_sets_solid_modules_and_analytic_categories.md) | profinite 站点上的 abelian group-valued sheaf |
| solidification | solidification | [第四十三章](43_condensed_sets_solid_modules_and_analytic_categories.md) | 从 condensed abelian groups 到 solid objects 的反射性对称幺半局部化 |
| solid tensor product | solid tensor product | [第四十三章](43_condensed_sets_solid_modules_and_analytic_categories.md) | $M\otimes^\solid N=(M\otimes N)^\solid$ |
| solid module | solid module | [第四十三章](43_condensed_sets_solid_modules_and_analytic_categories.md) | solid commutative algebra 在 solid objects 中的模 |
| analytic ring | analytic ring | [第四十三章](43_condensed_sets_solid_modules_and_analytic_categories.md) | condensed/solid 语境中编码解析完备性和模范畴的环状对象 |
| 语法范畴 | syntactic category | [第四十四章](44_syntactic_categories_classifying_toposes_and_tripos.md) | 由理论的上下文、公式和函数式关系组成的分类范畴 |
| 分类 topos | classifying topos | [第四十四章](44_syntactic_categories_classifying_toposes_and_tripos.md) | 表示某几何理论模型 2-函子的 Grothendieck topos |
| 泛模型 | generic model | [第四十四章](44_syntactic_categories_classifying_toposes_and_tripos.md) | 分类 topos 中由恒等几何态射对应的模型 |
| tripos | tripos | [第四十四章](44_syntactic_categories_classifying_toposes_and_tripos.md) | 带 Heyting 纤维、量词和 generic predicate 的谓词纤维化 |
| generic predicate | generic predicate | [第四十四章](44_syntactic_categories_classifying_toposes_and_tripos.md) | 通过 classifying maps 拉回分类全部谓词的泛谓词 |
| tripos-to-topos | tripos-to-topos | [第四十四章](44_syntactic_categories_classifying_toposes_and_tripos.md) | 从 tripos 构造 elementary topos 的过程 |
| 关系 | relation | [第四十五章](45_exact_completions_relations_allegories_and_regular_logic.md) | 乘积 $X\times Y$ 的子对象 $R:X\nrightarrow Y$ |
| regular completion | regular completion | [第四十五章](45_exact_completions_relations_allegories_and_regular_logic.md) | 泛地把有限极限范畴嵌入 regular category 的完成 |
| exact completion | exact completion | [第四十五章](45_exact_completions_relations_allegories_and_regular_logic.md) | 泛地加入有效等价关系商的正合完成 |
| effective equivalence relation | effective equivalence relation | [第四十五章](45_exact_completions_relations_allegories_and_regular_logic.md) | 为某态射 kernel pair 的内部等价关系 |
| allegory | allegory | [第四十五章](45_exact_completions_relations_allegories_and_regular_logic.md) | 抽象关系演算的 locally posetal 2-category |
| cohesive topos | cohesive topos | [第四十六章](46_cohesive_toposes_modalities_and_differential_cohesion.md) | 带 shape、离散、全局截面和余离散伴随串的高阶 topos |
| shape modality | shape modality | [第四十六章](46_cohesive_toposes_modalities_and_differential_cohesion.md) | $\int=\operatorname{Disc}\Pi$，提取对象同伦形状的模态 |
| flat modality | flat modality | [第四十六章](46_cohesive_toposes_modalities_and_differential_cohesion.md) | $\flat=\operatorname{Disc}\Gamma$，离散化全局截面的模态 |
| sharp modality | sharp modality | [第四十六章](46_cohesive_toposes_modalities_and_differential_cohesion.md) | $\sharp=\operatorname{Codisc}\Gamma$，余离散化全局截面的模态 |
| differential cohesion | differential cohesion | [第四十六章](46_cohesive_toposes_modalities_and_differential_cohesion.md) | 含 de Rham 或 infinitesimal shape 模态的 cohesive 结构 |
| exit path | exit path | [第四十七章](47_stratified_homotopy_exit_path_categories_and_constructible_sheaves.md) | 层标号只能沿偏序增大的层化路径 |
| exit-path $\infty$-category | exit-path $\infty$-category | [第四十七章](47_stratified_homotopy_exit_path_categories_and_constructible_sheaves.md) | 由点和 exit simplices 组成的层化空间高阶范畴 |
| constructible sheaf | constructible sheaf | [第四十七章](47_stratified_homotopy_exit_path_categories_and_constructible_sheaves.md) | 在每个 stratum 上局部常值的 sheaf |
| conically stratified space | conically stratified space | [第四十七章](47_stratified_homotopy_exit_path_categories_and_constructible_sheaves.md) | 局部形如 $\mathbb R^k\times C(L)$ 的层化空间 |
| 高阶 Morita 范畴 | higher Morita category | [第四十八章](48_higher_morita_traces_and_en_koszul_duality.md) | 以 $E_n$-代数和迭代双模为对象与态射的 $(\infty,n)$-范畴 |
| Morita trace | Morita trace | [第四十八章](48_higher_morita_traces_and_en_koszul_duality.md) | Morita $(\infty,2)$-范畴中恒等双模的 trace |
| higher Hochschild object | higher Hochschild object | [第四十八章](48_higher_morita_traces_and_en_koszul_duality.md) | 由球面或环形因子化同调表达的高阶 trace |
| $E_n$-Koszul duality | $E_n$-Koszul duality | [第四十八章](48_higher_morita_traces_and_en_koszul_duality.md) | 增广 $E_n$-代数与对偶余代数/代数之间的 bar-cobar 对偶 |
| derivator | derivator | [第四十九章](49_derivators_homotopy_kan_extensions_and_stable_derivators.md) | 记录所有图形同伦范畴及同伦 Kan 延拓的 2-函子 |
| stable derivator | stable derivator | [第四十九章](49_derivators_homotopy_kan_extensions_and_stable_derivators.md) | pushout 与 pullback 方块一致的 pointed derivator |
| stack | stack | [第五十章](50_stacks_gerbes_and_nonabelian_cohomology.md) | 满足对象与同构 descent 的 groupoid-valued prestack |
| torsor | torsor | [第五十章](50_stacks_gerbes_and_nonabelian_cohomology.md) | 局部同构于群 sheaf 正则作用的带作用 sheaf |
| gerbe | gerbe | [第五十章](50_stacks_gerbes_and_nonabelian_cohomology.md) | 局部非空且局部任意两对象同构的 stack |
| nonabelian cohomology | nonabelian cohomology | [第五十章](50_stacks_gerbes_and_nonabelian_cohomology.md) | 用 torsors、gerbes 和 higher stacks 描述的非交换上同调 |
| effective descent morphism | effective descent morphism | [第五十一章](51_categorical_galois_theory_descent_and_effective_descent.md) | 使全局对象范畴等价于 descent data 范畴的态射 |
| categorical Galois structure | categorical Galois structure | [第五十一章](51_categorical_galois_theory_descent_and_effective_descent.md) | 由反射、覆盖类和下降条件组成的抽象 Galois 理论框架 |
| normal extension | normal extension | [第五十一章](51_categorical_galois_theory_descent_and_effective_descent.md) | 自身拉回后 trivial 且满足有效下降的 covering |
| polynomial functor | polynomial functor | [第五十二章](52_polynomial_functors_species_analytic_functors_and_w_types.md) | 形如 $\Sigma_t\Pi_ps^*$ 的函子 |
| species | species | [第五十二章](52_polynomial_functors_species_analytic_functors_and_w_types.md) | 有限集合与双射 groupoid 上的集合值函子 |
| analytic functor | analytic functor | [第五十二章](52_polynomial_functors_species_analytic_functors_and_w_types.md) | 由 species 通过 $\sum_nF[n]\times_{\Sigma_n}X^n$ 生成的函子 |
| W-type | W-type | [第五十二章](52_polynomial_functors_species_analytic_functors_and_w_types.md) | 多项式函子的初代数 |
| $\infty$-cosmos | $\infty$-cosmos | [第五十三章](53_infinity_cosmoi_model_independent_infinity_category_theory.md) | 支持模型无关高阶范畴论的 simplicially enriched 结构 |
| homotopy 2-category | homotopy 2-category | [第五十三章](53_infinity_cosmoi_model_independent_infinity_category_theory.md) | Hom 为映射 quasi-category 同伦范畴的 2-范畴 |
| 正交性 | orthogonality | [第五十四章](54_orthogonality_factorization_systems_and_weak_factorization.md) | 交换方块存在唯一对角填充的态射关系 $f\perp g$ |
| 正交因子化系统 | orthogonal factorization system | [第五十四章](54_orthogonality_factorization_systems_and_weak_factorization.md) | 每个态射按两类相互正交态射分解的系统 |
| 弱因子化系统 | weak factorization system | [第五十四章](54_orthogonality_factorization_systems_and_weak_factorization.md) | 只要求提升存在、不要求唯一的因子化系统 |
| 局部对象 | local object | [第二十三章](23_presentable_infinity_categories_and_localizations.md), [第五十四章](54_orthogonality_factorization_systems_and_weak_factorization.md) | 对指定态射类取 Hom 后为等价或双射的对象 |
| sketch | sketch | [第五十五章](55_sketches_doctrines_and_categorical_theories.md) | 小范畴配指定极限锥和余极限余锥的数据 |
| doctrine | doctrine | [第五十五章](55_sketches_doctrines_and_categorical_theories.md) | 指定结构和保持结构函子的理论口径 |
| essentially algebraic theory | essentially algebraic theory | [第五十五章](55_sketches_doctrines_and_categorical_theories.md) | 允许由有限极限定义域控制的部分运算理论 |
| 幂等分裂 | idempotent splitting | [第五十六章](56_idempotents_karoubi_envelopes_and_absolute_colimits.md) | 把幂等 $e$ 写作 $sr$ 且 $rs=1$ 的 retract 数据 |
| Karoubi 包络 | Karoubi envelope | [第五十六章](56_idempotents_karoubi_envelopes_and_absolute_colimits.md) | 自由加入所有幂等分裂的范畴 |
| 绝对余极限 | absolute colimit | [第五十六章](56_idempotents_karoubi_envelopes_and_absolute_colimits.md) | 被所有函子保持的余极限 |
| Cauchy 完备 | Cauchy complete | [第五十六章](56_idempotents_karoubi_envelopes_and_absolute_colimits.md) | 存在所有绝对余极限的完备性条件 |
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
