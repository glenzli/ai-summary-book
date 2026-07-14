# 附录 D：资料源定理索引

本附录索引正文中作为“外部输入”的定理。它的作用不是替代引用，而是给后续严格化提供检查清单：每个大型定理最终都应补充精确版本、定理编号、模型约定和证明来源。

## D.0 状态标签和引用包

**定义 D.0.1（外部输入状态）.** 本书给外部结论使用以下状态标签。

1. **外部可用**：该结论是经典或已出版来源中的标准结论；本书可在满足假设后作为证明步骤使用，但必须在本附录记录来源和假设。
2. **外部候选**：该结论看似可作为正文证明步骤使用，但本书尚未精确定位版本、编号或模型约定；在定位前不得用于推出新命题。
3. **研究边界**：该结论属于近期预印本或模型尚未并入本书主线；只能在第二十一章或前沿审计中描述。
4. **禁用为证明步骤**：该结论只说明失败模式、反例或不可混用边界，不能作为正向证明输入。

**定义 D.0.2（引用包）.** 一个外部输入的可出版引用包由以下数据组成：

- 一个数学断言 $T$；
- 一个模型语境，例如 dg operad、simplicial operad、dendroidal set、marked simplicial set、factorization algebra；
- 一组假设 $H$，包括底环、特征、cofibrancy、fibrancy、smallness、left properness、monoid axiom、conilpotence 或几何横截性；
- 一个来源定位 $L$，至少包含作者、标题、版本或出版信息、定理编号或章节编号；
- 一个转换说明 $C$，解释来源中的符号和本书符号之间的差异。

只有当 $H,L,C$ 都被记录时，正文才可写“由外部输入定理”。若缺少任意一项，正文必须写成“研究边界”“证明边界”或“待验证”。

**规则 D.0.3（不倒用原则）.** 外部比较定理不能倒用为定义。例如：

- dendroidal-Lurie 比较不能用来定义 dendroidal infinity-operad；
- $E_n$-algebra 与 locally constant factorization algebra 的等价不能用来定义 factorization homology；
- rectification theorem 不能用来定义 strict algebra；
- Fukaya gluing theorem 不能用来替代 Fukaya category 的分析构造。

## D.1 基础 operad 与自由构造

| 位置 | 外部输入 | 主要来源 | 后续需补 |
| --- | --- | --- | --- |
| 第七章 | Properad 到自由 PROP 的图公式 | Markl-Shnider-Stasheff；Fresse | directed graph groupoids、connectedness 条件 |
| 附录 U | 自由 properad/PROP 的图群胚构造 | Markl-Shnider-Stasheff；Loday-Vallette；Fresse | directed graph groupoids、对称群商 |
| 附录 U | Frobenius algebra 的 PROP 表示 | Markl-Shnider-Stasheff；Kock；Frobenius algebra 文献 | pairing convention、有限维假设 |

## D.2 Koszul 对偶与 bar-cobar

| 位置 | 外部输入 | 主要来源 | 后续需补 |
| --- | --- | --- | --- |
| 第八章 | Ass、Com、Lie 的 Koszul 性 | Ginzburg-Kapranov；Loday-Vallette | 特征假设、reduced 约定 |
| 第九章 | Koszul twisting morphism 判别 | Ginzburg-Kapranov；Loday-Vallette；Fresse | quasi-isomorphism 模型结构 |
| 第九章 | Bar-cobar resolution 的 weak equivalence | Loday-Vallette；Fresse | conilpotence 和 filtration 条件 |
| 附录 I | Koszul 判别等价条件 | Ginzburg-Kapranov；Loday-Vallette；Fresse | weight spectral sequence、有限型假设 |
| 附录 I | $\Omega\mathcal P^¡\to\mathcal P$ quasi-isomorphism criterion | Ginzburg-Kapranov；Loday-Vallette；Fresse | reduced/conilpotent convention |
| 附录 Q | $\operatorname{Ass}_{ns}$ Koszul 性 | Loday--Vallette LV-3 = Theorem 8.1.1 + following $\operatorname{As}$ example | reduced/nonunital、非对称 convention；正文内部检查终止性与唯一临界对 |
| 附录 Q | Bar-cobar counit 是 quasi-isomorphism/cofibrant resolution | Loday-Vallette；Fresse；Hinich | conilpotence、模型结构、滤过收敛 |
| 附录 Q | Koszul 谱序列收敛与判别 | Ginzburg-Kapranov；Loday-Vallette；Fresse | boundedness、weight filtration convention |

## D.3 同伦代数与 Deligne 型定理

| 位置 | 外部输入 | 主要来源 | 后续需补 |
| --- | --- | --- | --- |
| 第十章 | May recognition principle | May；Boardman-Vogt | 基点、连通性、group completion 条件 |
| 第十章 | $H_\*(E_n)\cong\operatorname{Pois}_n$ | Cohen；Fresse | 系数环、degree convention |
| 第十章 | $E_n$-operad 形式性 | Kontsevich；Tamarkin；Fresse | 特征 $0$、链模型 |
| 第十一章 | Deligne conjecture | McClure--Smith MS-1--MS-3；Berger--Fresse BF-1--BF-4；Kontsevich-Soibelman | 使用的 $E_2$ 链模型和 suspended sign convention |
| 第十一章 | Framed $E_2$ 同调为 BV | Getzler；Fresse | circle action 和 BV operator convention |
| 第十二章 | Brace operad 与 $E_2$ 链模型弱等价 | Berger--Fresse BF-1--BF-3；McClure--Smith MS-1--MS-3 | brace/surjection operad 比较；附录 W 符号转换 |
| 第十三章 | Homological perturbation lemma | Gugenheim-Lambe-Stasheff；Markl；Loday-Vallette | side conditions、filtration |
| 第十三章 | Homotopy transfer theorem | Markl, arXiv:math/9907138v3, MHT-1--MHT-8；Kadeishvili；Merkulov | sign conventions |
| 第十三章 | Minimal model 存在唯一性 | Kadeishvili；Loday-Vallette | $A_\infty/L_\infty$ quasi-isomorphism 类型 |
| 附录 J | Contraction side condition normalization | Homological perturbation literature；Markl | normalized contraction 公式 |
| 附录 J | $A_\infty$ transfer tree formulas with signs | Markl MHT-6 for existence；Kadeishvili；Loday-Vallette | suspended sign convention |
| 附录 J | $L_\infty$ transfer tree formulas with signs | Markl MHT-6 for existence；Merkulov；Loday-Vallette | shuffle signs、反对称化 |
| 附录 J | Minimal model uniqueness | Kadeishvili；Loday-Vallette；Fresse | $\infty$-isomorphism 类型 |
| 附录 S | Homotopy transfer theorem 的低阶公式与完整高阶延拓 | Markl MHT-1--MHT-8；Kadeishvili；Gugenheim-Lambe-Stasheff；Merkulov；Loday-Vallette | side conditions、suspended signs |
| 附录 S | Minimal model formality 与 strict dg formality 比较 | Kadeishvili；Loday-Vallette；Hinich；Fresse | $A_\infty$ quasi-isomorphism 与 rectification |

## D.4 模型范畴中的 operad

| 位置 | 外部输入 | 主要来源 | 后续需补 |
| --- | --- | --- | --- |
| 第十四章 | 对称序列 projective 模型结构 | 命题 14.7 内部归约为 $\prod_n\mathcal M^{\Sigma_n}$ | 各作用范畴 projective 模型结构的存在仍须在所用底范畴中验证 |
| 第十四章 | Operad transferred 模型结构 | Berger-Moerdijk, arXiv:math/0206094v3, Theorem 3.1；Fresse | monoid axiom、cofibrant unit |
| 第十四章 | Operad algebra admissibility | Berger-Moerdijk, arXiv:math/0206094v3, Theorem 3.2；Pavlov--Scholbach PSAR-1--PSAR-3；PSP-1--PSP-2 | symmetric h-monoidality、symmetroidality、smallness、tractability |
| 第十四章 | Rectification criterion | Berger-Moerdijk, arXiv:math/0206094v3, Theorem 4.4 and Corollary 4.5；Hinich HIN-2；Pavlov--Scholbach PSAR-4--PSAR-6 | cofibrancy、admissibility、symmetric flatness 与底范畴条件 |
| 第十四章 | Boardman-Vogt resolution | Boardman-Vogt；Berger-Moerdijk；当前仅作外部边界 14.31 | 尚缺覆盖 interval object、well-pointedness、$\Sigma$-cofibrancy 和目标模型结构的统一 locator，不得直接调用 |
| 附录 G | Operad transferred model structure schema | Berger-Moerdijk；Fresse FRE-1--FRE-6；Pavlov--Scholbach PSAR-1--PSAR-3 | 小性、树形 filtration、等变 cofibration |
| 附录 G | Colored admissibility schema | Berger-Moerdijk；Pavlov--Scholbach PSAR-1--PSAR-3；PSP-1--PSP-2 | symmetric h-monoidality、symmetric flatness、tractability |
| 附录 G | Rectification schema | Hinich HIN-2；Berger-Moerdijk；Pavlov--Scholbach PSAR-4--PSAR-6；Lurie HA-ALG-1--HA-ALG-3 | Quillen equivalence 条件 |
| 附录 G | 正特征中 $E_\infty$ 与 strict commutative dg algebra 的边界 | Hinich；Mandell；Pavlov-Scholbach | 具体反例和 power operations |
| 附录 K | Enriched colored operad admissibility | Berger-Moerdijk；Pavlov--Scholbach PSAR-1--PSAR-3 | colored symmetric flatness 条件 |
| 附录 R | $\mathbf{sSet}$、$\mathbf{Top}$、$\mathbf{Ch}_k$ 中 operad transferred/admissibility 案例 | Berger-Moerdijk；Hinich HIN-1--HIN-2；Fresse FRE-1--FRE-6；Pavlov--Scholbach PSAR-1--PSAR-6；PSP-1--PSP-2 | 逐底范畴假设翻译 |
| 附录 R | 特征 $0$ 中 $E_\infty$ 到 $\operatorname{Com}$ 的 rectification 案例 | Hinich HIN-2；Berger-Moerdijk；Fresse；Pavlov--Scholbach PSAR-4 | cofibrant operad、admissibility、symmetric flatness |
| 附录 R | 正特征或一般底环中 rectification 不可用边界 | Hinich；Mandell；Pavlov--Scholbach PSAR-4 and symmetric-flatness hypotheses | power operations、对称幂不 exact |
| 附录 X | $\operatorname{Sym}^p$ 不保持 acyclic complex 的内部计算 | 本书命题 X.15 和推论 X.16 | 用作正特征 rectification 风险的内部例子，不替代 Mandell/Hinich/Pavlov-Scholbach 定理 |
| 附录 X | Hochschild homology 的 Morita invariance 边界说明 | Keller；Loday；Weibel | 本书只使用 $HH_0$ 低阶检查 |

## D.5 空间、dendroidal 与 infinity-operad

| 位置 | 外部输入 | 主要来源 | 后续需补 |
| --- | --- | --- | --- |
| 第十五章 | Kan-Quillen 模型结构 | Quillen；Goerss-Jardine | weak homotopy equivalence convention |
| 第十五章 | $\mathbf{sSet}$ 与 $\mathbf{Top}$ 的底范畴 Quillen equivalence | Quillen；Goerss-Jardine | Top 采用的点集范畴；不得自动提升到 operad categories |
| 第十五章 | Simplicial/topological operad 模型结构 | BM-1 分别应用于 $\Delta^1$ 与 $[0,1]$ Hopf intervals | Operad-level realization--Sing Quillen equivalence 尚作外部边界 15.18 |
| 第十五章 | Eilenberg-Zilber 相干性 | Eilenberg-Zilber；Mac Lane | lax symmetric monoidal 选择 |
| 第十六章 | Dendroidal nerve fully faithful | Moerdijk-Weiss, arXiv:math/0701293v2, Example 4.2；MW-2 | strict Segal 条件 |
| 第十六章 | $\Delta\hookrightarrow\Omega$ fully faithful | Moerdijk-Weiss, arXiv:math/0701293v2, Section 3；MW-1 | 根方向 convention |
| 第十七章 | Normal monomorphism 引理 | Cisinski-Moerdijk, arXiv:0902.1954v2, Proposition 1.4 | automorphism 稳定子检查 |
| 第十七章 | Inner anodyne/horn calculus | Cisinski-Moerdijk, arXiv:0902.1954v2, Proposition 1.5 and Corollaries 1.6--1.8 | weakly saturated closure、lifting 性质 |
| 第十七章 | Operadic model structure on dSet | Cisinski-Moerdijk, arXiv:0902.1954v2, Theorem 2.4；Proposition 2.6 for fibrant weak equivalences | weak equivalence 定义、erratum 影响 |
| 第十八章 | Lurie-style infinity-operad definition technology | Lurie | marked simplicial sets |
| 第十八章 | Dendroidal-Lurie model comparison | Heuts--Hinich--Moerdijk HHM-1--HHM-5 | Quillen equivalence zig-zag；open/no-constants restriction |
| 附录 T | Inner horn inclusions 是 normal monomorphisms | Cisinski-Moerdijk；Moerdijk-Weiss | elementary face/degeneracy 分解 |
| 附录 T | Strict Segal 条件刻画 dendroidal nerve 本质像 | Moerdijk-Weiss MW-2；MW-4 | fully faithfulness、自然变换恢复 operad |
| 附录 M | Category of operators nerve gives Lurie-style infinity-operad | Lurie HA-OP-1--HA-OP-3；Hinich | active/inert convention |
| 附录 M | Dendroidal-Lurie comparison | Heuts--Hinich--Moerdijk HHM-1--HHM-5 | 模型结构版本与 open/no-constants restriction |
| 附录 M | Algebra localization comparison | White WHT-1--WHT-4；White--Yau WY-1--WY-3 for model-category preservation；Pavlov--Scholbach PSAR-5--PSAR-6；Lurie HA-ALG-1--HA-ALG-3；Hinich DKR-7 | preservation 与 infinity-categorical comparison 分离；admissibility/cofibrancy 条件 |

## D.6 Localization 与 straightening

| 位置 | 外部输入 | 主要来源 | 后续需补 |
| --- | --- | --- | --- |
| 第十九章 | Dwyer-Kan localization | Hinich DKR-1--DKR-4；Dwyer--Kan | hammock model and infinity-categorical localization |
| 第十九章 | Simplicial model category coherent nerve comparison | Hinich DKR-3--DKR-6；Lurie；Dwyer--Kan | fibrant-cofibrant subcategory |
| 第十九章 | Quillen equivalence induces infinity-equivalence | Hinich DKR-7；Lurie；Hovey | combinatorial/size 条件 |
| 第十九章 | Straightening/unstraightening | Lurie, *Higher Topos Theory*, Theorem 3.2.0.1；Pratali PRA-1--PRA-5 for spaces-valued operadic straightening | coCartesian convention；operadic straightening preprint boundary |
| 第十九章 | Monoidal localization | White, arXiv:1404.5197, Section 4 and Theorems 4.5--4.6 for model-category criteria；Lurie HA；Hinich | tensor preserves weak equivalences；model-category 与 infinity-categorical versions 分离 |
| 第十九章 | Algebra localization comparison | White, arXiv:1404.5197, Definition 3.1, Theorem 3.2, Corollary 3.4；White--Yau, arXiv:1503.06720, Definition 7.2.1, Theorem 7.2.3, Theorems 7.4.1--7.4.3；Pavlov--Scholbach PSAR-5--PSAR-6；Lurie HA-ALG-1--HA-ALG-3；Hinich DKR-7 | operad/colored-operad preservation 与 strict-to-infinity algebra comparison 分离 |

## D.7 几何应用

| 位置 | 外部输入 | 主要来源 | 后续需补 |
| --- | --- | --- | --- |
| 第二十章 | Locally constant multiplicative factorization algebras 等价于 $E_n$-algebras | 当前仅作外部边界 20.6/N.26；Costello--Gwilliam、Lurie 为候选来源 | 尚无精确 theorem locator；AF-3 不能替代该比较 |
| 第二十章 | Factorization homology excision | Ayala-Francis, arXiv:1206.5522v6, Lemma 3.18 | collar/gluing 条件、stratified/Fukaya 版本另行定位 |
| 第二十章 | Dunn additivity | Lurie DUNN-1；Dunn | tensor product of infinity-operads；strict topological tensor product另行引用 |
| 第二十章 | Fukaya category 构造 | Seidel；Fukaya-Oh-Ohta-Ono | transversality/obstruction setup |
| 第二十章 | Wrapped Fukaya gluing | Ganatra-Pardon-Shende | Liouville sector hypotheses |
| 附录 N | Framed/tangential factorization homology 的定义与 functoriality | Ayala-Francis, arXiv:1206.5522v6, Theorem 1.2 and Theorem 3.24；Lurie；Costello-Gwilliam | tangential structure、disk category 模型 |
| 附录 N | Excision 与 derived relative tensor product 公式 | Ayala-Francis, arXiv:1206.5522v6, Lemma 3.18 | collar、边界版本、bar construction 模型 |
| 附录 N | $\int_{S^1}A\simeq HH_\*(A)$ | Ayala-Francis, arXiv:1206.5522v6, Theorem 3.19；Hochschild/cyclic bar 文献 | $E_1$ 模型、cyclic bar convention |
| 附录 N | Manifold homology theories 与 Disk$_n$-algebras 的刻画 | Ayala-Francis, arXiv:1206.5522v6, Theorem 3.24 | 与 locally constant factorization algebra 等价分开处理 |
| 附录 N | 圆柱/Fubini 与 $E_{n-1}$-Hochschild object | 当前仅作外部边界 N.21 | 尚缺 product/Fubini theorem locator；AF-2 与 DUNN-1 单独均不足 |
| 附录 N | 交换系数下的 $B\otimes M$ 识别 | Lurie；Ayala-Francis | tensoring over spaces、higher Hochschild chains |
| 附录 N | Fukaya 型 cosheaf/sectorial descent | Ganatra-Pardon-Shende；Seidel；Fukaya-Oh-Ohta-Ono | 几何设置、横截性、紧性、orientation |
| 附录 V | 带边界 factorization homology 的区间计算 | Ayala-Francis, arXiv:1206.5522v6, Theorem 3.26；Lurie；Costello-Gwilliam | boundary disk category、module 标记 |
| 附录 V | Stratified factorization homology 与 defect gluing | Ayala-Francis-Tanaka；Lurie | conically smooth stratified spaces |
| 附录 V | Sectorial/skeletal Fukaya descent | Ganatra-Pardon-Shende；Nadler；Seidel | Liouville sector、stops、compactness |

## D.8 研究边界

| 位置 | 外部输入 | 主要来源 | 后续需补 |
| --- | --- | --- | --- |
| 第二十一章 | Homology of infinity-operads | Hoffbeck-Moerdijk | 版本、定理编号 |
| 第二十一章 | Koszul duality for algebras over infinity-operads | Hoffbeck-Moerdijk 2026 | 预印本版本核查 |
| 第二十一章 | Relative dendroidal Rezk nerve | Arakawa-Carmona-Pratali 2026 | 模型结构与应用边界 |
| 第二十一章 | Higher operad structure for Fukaya categories | Yuan 2026 | 几何模型和技术假设 |
| 第二十一章 | Operadic categories as simplicial groupoids | Batanin-Kock-Weber 2026 | operadic nerve 定义 |
| 附录 Y | Infinity-operadic homology 与 Koszul duality 的前沿接口 | Hoffbeck-Moerdijk | linear infinity-operad 模型、特化到 classical Koszul 的比较 |
| 附录 Z | Operadic categories、relative dendroidal Rezk nerve、Fukaya 高阶结构接口 | Pavlova；Arakawa-Carmona-Pratali；Batanin-Kock-Weber；Yuan | operadic nerve、Rezk nerve、几何 gluing 的定理编号和模型假设 |

## D.9 经典例子附录

| 位置 | 外部输入 | 主要来源 | 后续需补 |
| --- | --- | --- | --- |
| 附录 F | 自由 Lie 代数的 Lyndon word/tree/primitive 元素模型 | Reutenauer；Loday-Vallette | 底环和 PBW 假设 |
| 附录 F | $H_\*(\mathcal C_n)\cong\operatorname{Pois}_n$ | Cohen；May；Fresse | 系数环、degree convention |
| 附录 F | PBW/Shirshov-Witt 型自由 Lie 代数定理 | PBW；Shirshov-Witt；Reutenauer | 一般底环版本 |

## D.10 Infinity-algebra 与 $E_n$ 附录

| 位置 | 外部输入 | 主要来源 | 后续需补 |
| --- | --- | --- | --- |
| 附录 L | $H_\*(\mathcal C_n)\cong\operatorname{Pois}_n$ | Cohen；May；Fresse | 系数和 degree convention |
| 附录 L | $E_n$ 形式性 | Kontsevich；Tamarkin；Fresse | 特征 $0$ 和链模型 |
| 附录 L | Dunn/Lurie additivity | Lurie DUNN-1；Dunn | tensor product 模型 |
| 附录 L | $E_\infty$ rectification boundary | Hinich；Pavlov-Scholbach；Mandell | 正特征和一般底环反例 |

## D.11 使用规则

后续扩写若使用本附录中的外部输入，必须在相应章节旁补充：

1. 精确文献；
2. 定理编号或章节编号；
3. 模型范畴或 infinity-operad 模型；
4. 底环、特征、cofibrancy、fibrancy、smallness 条件；
5. 与本书符号约定的转换。

若无法补齐这些信息，该结论只能保留为研究边界或说明，不得进入核心证明链。

## D.12 最小可出版引用包

本节把全书最常用的外部输入压缩成引用包。每个引用包给出允许用法和禁止用法。最终 operad theory 数学收口见 [FINAL_OPERAD_THEORY_CLOSURE.md](FINAL_OPERAD_THEORY_CLOSURE.md)；后续出版社级版本只需继续做 bibliography/page/tag 核查。

### D.12.1 Koszul/bar-cobar 引用包

**断言.** 对满足 reduced、weight graded、有限型和 conilpotence 假设的二次 operad $\mathcal P$，Koszul twisting morphism $\kappa:\mathcal P^¡\to\mathcal P$ 的 acyclicity 与 bar-cobar resolution 的准同构性质等价；若 $\mathcal P$ Koszul，则
$$
\Omega\mathcal P^¡\longrightarrow \mathcal P
$$
是 quasi-isomorphism。

**模型语境.** 同调分次 dg operads 与 conilpotent dg cooperads。

**当前来源定位.** Ginzburg--Kapranov classical core 已由 [P0_REFERENCE_LOCATORS_BATCH_4.md](P0_REFERENCE_LOCATORS_BATCH_4.md) 中 GK-1--GK-7 定位。Loday--Vallette author-hosted draft v0.99 的 Theorems 6.6.2、7.4.6 和 8.1.1 已在同一批次登记为 LV-1--LV-3：前两项分别精确支撑 connected weight-graded twisting-morphism 四项等价与 $\mathcal P^¡=\mathcal C(sE,s^2R)$ 的 quadratic Koszul criterion；LV-3 连同定理后的 $\operatorname{As}$ 例子支撑附录 Q 的 nonsymmetric rewriting/Koszul 步骤。Fresse modern cobar/cofibrant replacement 由 [P0_REFERENCE_LOCATORS_BATCH_5.md](P0_REFERENCE_LOCATORS_BATCH_5.md) 的 FRE-1--FRE-6 定位；Hinich dg-operad model context 由 HIN-1--HIN-2 定位。Quasi-isomorphism criterion 与模型范畴中的 cofibrant-resolution 结论分开使用。

**允许用法.** 定义 $A_\infty$、$L_\infty$、$C_\infty$ 为 cofibrant resolution；证明 classical operad algebra 的同伦版本由 bar-cobar 控制。

**禁止用法.** 不得推出 arbitrary infinity-operad 的 Koszul duality；不得在没有 conilpotence 或 filtration 条件时使用。

### D.12.2 同伦转移引用包

**断言.** 给定满足 side conditions 的 contraction
$$
(H,d_H)\xrightarrow{i}(A,d_A)\xrightarrow{p}(H,d_H),
\qquad
ip-\operatorname{id}_A=d_Ah+hd_A,
$$
若 $A$ 带有 $\mathcal P_\infty$-代数结构，则 $H$ 获得转移的 $\mathcal P_\infty$-代数结构，并且 $i,p$ 可提升为相应的 $\infty$-morphism。

**模型语境.** 链复形上的 quasi-free operadic algebra，通常以 rooted tree 或 bar-cobar coalgebra coderivation 表达。

**当前来源定位.** Markl 的 strongly homotopy transfer over chain homotopy equivalence 已由 [P0_REFERENCE_LOCATORS_BATCH_6.md](P0_REFERENCE_LOCATORS_BATCH_6.md) 中 MHT-1--MHT-8 定位：Definition 17、Theorem 19、Lemma 20、Theorem 27、Proposition 31、Theorem 33、Proposition 34、Proposition 35 和 Proposition 36。Gugenheim--Lambe--Stasheff/Huebschmann 的 basic perturbation lemma、Kadeishvili minimal model、Merkulov/Loday--Vallette tree signs 和 minimal model uniqueness 已在 [FINAL_OPERAD_THEORY_CLOSURE.md](FINAL_OPERAD_THEORY_CLOSURE.md) 中关闭为 HPT/sign convention package；正文使用 suspended/coderivation 定义和 Markl existence theorem，不把 unsuspended 全公式作为未定位证明步骤。

**允许用法.** 构造 homology 上的 $A_\infty/L_\infty$ 结构；解释 $m_3$ 与 Massey product 的边界关系；给 minimal model 存在性提供输入。

**禁止用法.** 不得把低阶公式当作完整转移定理；不得在未固定 suspension convention 时写 unsuspended 全公式。

### D.12.3 模型范畴中 operad 引用包

**断言.** 在满足适当 cofibrant generation、monoid axiom、pushout-product/unit axiom、小性、等变 cofibration 或 symmetric flatness/h-monoidality 假设的对称幺半模型范畴 $\mathcal M$ 中，operad 或 colored operad 的代数范畴可获得 transferred model structure；entrywise weak equivalence 的 operad map 在额外 cofibrancy/admissibility 假设下诱导 Quillen equivalence。

**模型语境.** $\mathcal M$ 中的 symmetric sequences、colored symmetric sequences、operads 和 algebras over operads。

**当前来源定位.** Berger--Moerdijk early transferred/rectification results 已由 [P0_REFERENCE_LOCATORS_BATCH_1.md](P0_REFERENCE_LOCATORS_BATCH_1.md) 定位；Fresse cobar/cofibrant replacement 和 Hinich dg-operad model context 已由 [P0_REFERENCE_LOCATORS_BATCH_5.md](P0_REFERENCE_LOCATORS_BATCH_5.md) 中 FRE-1--FRE-6 与 HIN-1--HIN-2 定位；Pavlov--Scholbach modern colored admissibility/rectification 已由 [P0_REFERENCE_LOCATORS_BATCH_9.md](P0_REFERENCE_LOCATORS_BATCH_9.md) 中 PSAR-1--PSAR-6 与 PSP-1--PSP-2 定位；Lurie associative/commutative strict-to-infinity comparison 已由 HA-ALG-1--HA-ALG-3 定位。

**允许用法.** 在第十四章和附录 R 中说明 transferred model structure、admissibility 和 rectification 的使用条件。

**禁止用法.** 不得把 $\mathbf{Ch}_k$ 特征 $0$ 中的 rectification 推广到正特征或一般底环；不得把 $\Sigma$-cofibrant operad 的结论推广到任意 operad。

### D.12.4 Dendroidal model structure 引用包

**断言.** Dendroidal sets 上存在 Cisinski-Moerdijk operadic model structure；fibrant objects 是 dendroidal infinity-operads 的模型；dendroidal nerve of strict operads 与 strict Segal dendroidal sets 的比较由 fully faithful nerve 和模型结构给出。

**模型语境.** presheaves on Moerdijk-Weiss tree category $\Omega$。

**当前来源定位.** Moerdijk--Weiss strict dendroidal nerve core 已由 [P0_REFERENCE_LOCATORS_BATCH_7.md](P0_REFERENCE_LOCATORS_BATCH_7.md) 中 MW-1--MW-6 定位：Section 3、Example 4.2、Section 4 after Example 4.1、Example 7.1、Proposition 7.2 和 Theorem 7.5。Cisinski--Moerdijk operadic model structure 已由 [P0_REFERENCE_LOCATORS_BATCH_1.md](P0_REFERENCE_LOCATORS_BATCH_1.md) 中 CM-1--CM-4 定位。Heuts--Hinich--Moerdijk dendroidal-Lurie comparison 已由 [P0_REFERENCE_LOCATORS_BATCH_10.md](P0_REFERENCE_LOCATORS_BATCH_10.md) 中 HHM-1--HHM-5 定位；使用时须保留来源中的 open/no-constants 限制。

**允许用法.** 第十六至十七章可用来区分 strict operads、strict Segal dendroidal sets 和 inner Kan dendroidal sets。

**禁止用法.** 不得把一个 inner horn filler 的存在误写成 strict composition；不得把 strict nerve 的唯一 filler 性推广到任意 dendroidal infinity-operad。

### D.12.5 Lurie-style infinity-operad 与模型比较引用包

**断言.** Lurie-style infinity-operads 可由适当的 marked simplicial sets over $N(\mathbf{Fin}_*)$ 或相近模型描述；dendroidal infinity-operads 与 Lurie-style infinity-operads 之间存在模型比较；在适当假设下，operadic algebra categories 在比较下对应。

**模型语境.** quasi-categories、marked simplicial sets、preoperads、dendroidal sets、category of operators nerve。

**当前来源定位.** Category-of-operators/Lurie-style entry 已由 [P0_REFERENCE_LOCATORS_BATCH_10.md](P0_REFERENCE_LOCATORS_BATCH_10.md) 中 HA-OP-1--HA-OP-3 定位；dendroidal-Lurie Quillen-equivalence comparison 已由 HHM-1--HHM-5 定位；simplicial colored operad strict-to-infinity algebra comparison 已由 [P0_REFERENCE_LOCATORS_BATCH_9.md](P0_REFERENCE_LOCATORS_BATCH_9.md) 中 PSAR-5 定位。

**允许用法.** 第十八至十九章可用来说明不同 infinity-operad 模型之间的可比较性。

**禁止用法.** 不得把 dendroidal set 定义和 Lurie-style definition 混写；不得在没有 active/inert convention 的情况下比较 algebra objects。

### D.12.6 Localization 与 straightening 引用包

**断言.** Relative category 有 Dwyer-Kan localization；simplicial model category 的 cofibrant-fibrant 子范畴经 coherent nerve 给出 underlying infinity-category；Quillen equivalence 在适当条件下诱导 infinity-categorical equivalence；straightening/unstraightening 分类 coCartesian fibrations。

**模型语境.** relative categories、simplicial categories、quasi-categories、coCartesian fibrations。

**当前来源定位.** Ordinary straightening/unstraightening 已由 [P0_REFERENCE_LOCATORS_BATCH_2.md](P0_REFERENCE_LOCATORS_BATCH_2.md) 中 HTT-1 定位。模型范畴的 underlying symmetric monoidal infinity-category 已由 [P0_REFERENCE_LOCATORS_BATCH_9.md](P0_REFERENCE_LOCATORS_BATCH_9.md) 中 HA-MON-1--HA-MON-2 定位；Bousfield localization 是否继续 monoidal 以及是否保持 operad algebras 则由 WHT-1--WHT-4 与 WY-1--WY-3 分开控制。Dwyer--Kan localization、underlying infinity-category、fibrant-cofibrant subcategory comparison 和 Quillen-pair passage 由 DKR-1--DKR-7 定位。Spaces-valued operadic straightening 由 PRA-1--PRA-5 作 2025 preprint/P1 locator 定位，不扩张到任意 $\mathcal C$-值。Strict-to-infinity algebra comparison 由 PSAR-5--PSAR-6 与 HA-ALG-1--HA-ALG-3 定位。

**允许用法.** 第十九章可用来连接模型范畴、localization 和 functorial families。

**禁止用法.** 不得把 ordinary localization 当作 hammock localization；不得假设“先代数后 localization”和“先 localization 后代数”无条件交换。

### D.12.7 Factorization homology 引用包

**断言.** Locally constant multiplicative factorization algebras on $\mathbb R^n$ 与 $E_n$-algebras 等价；factorization homology 满足 excision；在 $E_1$ 情形中
$$
\int_{S^1} A\simeq HH_\*(A)
$$
在适当模型和系数条件下成立。

**模型语境.** Disk categories、symmetric monoidal infinity-categories、factorization algebras、factorization homology。

**当前来源定位.** Ayala--Francis topological-manifold factorization homology、excision、circle Hochschild calculation、homology-theory characterization、boundary version 和 commutative-coefficient calculation 已由 [P0_REFERENCE_LOCATORS_BATCH_3.md](P0_REFERENCE_LOCATORS_BATCH_3.md) 中 AF-0--AF-5 定位；其中 AF-5 是 arXiv:1206.5522v6, Proposition 5.1，支撑 $\int_MA\simeq M\otimes A$。Dunn additivity 已由 [P1_REFERENCE_LOCATORS_FINAL_SWEEP.md](P1_REFERENCE_LOCATORS_FINAL_SWEEP.md) 中 DUNN-1 定位。Costello--Gwilliam/Lurie locally constant factorization algebra equivalence、stratified factorization 和 Fukaya geometry 保留为外部边界 locator。

**允许用法.** 第二十章和附录 N、V 可用来计算标准低维例子并解释 excision。

**禁止用法.** 不得把无边界 disk 归一化用于带边界或分层空间；不得把 Hochschild 识别推广到所有 $E_n$-algebras 和所有流形；不得把 AF-5 的 commutative algebra hypothesis 弱化成未指定的“足够交换”。

### D.12.8 Fukaya category 引用包

**断言.** 在给定 brane data、transversality、compactness、orientation 和 obstruction theory 的几何设置中，pseudo-holomorphic polygon counts 产生 $A_\infty$-category；在若干 Liouville sector 或 skeletal 设置中存在 descent/gluing 型结果。

**模型语境.** $A_\infty$-categories、wrapped Fukaya categories、Liouville sectors、sectorial descent。

**当前来源定位.** Seidel；Fukaya--Oh--Ohta--Ono；Ganatra--Pardon--Shende；相关局部模型文献。状态：已由 [FINAL_OPERAD_THEORY_CLOSURE.md](FINAL_OPERAD_THEORY_CLOSURE.md) 关闭为外部几何边界；不进入 operad theory 内部证明链。

**允许用法.** 第二十章可把 Fukaya category 当作 operadic/algebraic structures 的几何来源。

**禁止用法.** 不得仅由 operad 公理推出 Fukaya category 的存在；不得把 holomorphic curve compactness 或 orientation 当作纯代数事实。

### D.12.9 2026 前沿引用包

**断言.** Hoffbeck-Moerdijk、Pavlova、Yuan、Arakawa-Carmona-Pratali、Batanin-Kock-Weber 的近期条目分别提供 infinity-operadic Koszul duality、operadic categories、relative dendroidal Rezk nerve 和 Fukaya 高阶 operadic structure 的前沿入口。

**模型语境.** 视具体论文而定：linear infinity-operads、operadic categories、relative infinity-operads、dg $\mathbf{fc}$-multicategories。

**当前来源定位.** [FRONTIER_SOURCE_AUDIT_2026_06_30.md](FRONTIER_SOURCE_AUDIT_2026_06_30.md)。状态：研究边界。

**允许用法.** 第二十一章只把这些资料作为开放问题的背景；任何具体新结论仍须按定义 D.0.2 单独登记模型、假设与定理定位。

**禁止用法.** 不得把这些条目的新结论写成第八至二十章的基础定理；不得在未比较模型前把它们和 classical operad 结论合并。

## D.13 当前最终边界与生产校对包

以下项目不阻止本书作为 operad theory 数学收口版本使用。它们只说明哪些内容属于 convention package、几何边界或出版社级 production work。

1. Koszul/bar-cobar 定理的 Ginzburg--Kapranov classical core、Loday--Vallette LV-1--LV-2 四项判别、LV-3 nonsymmetric rewriting criterion 和 Fresse/Hinich modern cobar/model context 已分别定位；不得把 quasi-isomorphism criterion 自动升级为 cofibrant-resolution 结论。
2. Homotopy transfer theorem 的 Markl existence version 已定位；basic perturbation lemma、Kadeishvili minimal model、tree signs 和 minimal model uniqueness 作为 HPT/sign convention package 处理。
3. Operad transferred model structure、admissibility、rectification、Bousfield localization preservation、Dwyer--Kan localization、dendroidal-Lurie comparison、category-of-operators entry、ordinary/operadic straightening 和 strict-to-infinity algebra comparison 均已有 locator；后续只需逐底范畴记录假设转换、模型限制和 bibliography 版本。
4. Factorization homology 的 topological-manifold excision、圆周计算、带边界版本和 Dunn additivity 已有 locator；locally constant factorization algebra equivalence、stratified factorization 和 Fukaya geometry 保留为外部几何边界。
5. Fukaya category 和 wrapped Fukaya gluing 需要几何假设表；纯 operad 章节不得承担这些分析证明。
6. 2026 前沿预印本只能保留为研究边界，除非未来完成版本、定理编号、模型比较和独立依赖链。
