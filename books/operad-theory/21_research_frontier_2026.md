# 第二十一章：2025-2026 研究边界与开放问题目录

## 本章目标

本章记录截至 2026-06-30 已联网核查的近期 operad theory 相关研究入口。它不是前沿结果的教材化证明。本章的作用是把新文献放入本书的逻辑地图中，并说明哪些主题需要后续验证后才能进入正文定理链。具体版本边界见 [FRONTIER_SOURCE_AUDIT_2026_06_30.md](FRONTIER_SOURCE_AUDIT_2026_06_30.md)。

## 依赖前置知识

需要 operad、colored operad、dendroidal set、infinity-operad、bar-cobar 构造、Koszul 对偶、模型范畴和 localization 的基本语言。本书前二十章已经给出第一轮主体草稿；本章仍作为研究索引阅读，不能替代对具体预印本的逐条验证。

## 21.1 收录标准

**约定 21.1.** 本章只收录满足以下条件的资料：

- 来源是一手资料，例如 arXiv、作者主页、期刊版本或出版商页面；
- 与 operad、operadic category、dendroidal set、infinity-operad、operadic localization、同伦代数或几何中的 operadic structure 直接相关；
- 题名、作者、提交日期或更新时间已经核查；
- 本章不把未验证的新结果表述为本书定理。

**约定 21.2.** 若后续章节需要使用本章某篇论文中的定理，必须执行以下步骤：

1. 记录具体版本号、定理编号和页码。
2. 检查该定理依赖的定义是否与本书约定一致。
3. 判断结论是否已有独立来源、后续修正或 erratum。
4. 在正文中标注为“外部输入定理”，除非本书给出完整证明。

**版本表 21.3.** 本章采用下表作为 2026-06-30 的前沿入口清单。

| 资料 | 核查版本 | 主要模型 | 本书允许用法 |
| --- | --- | --- | --- |
| Hoffbeck--Moerdijk, “Homology of infinity-operads” | arXiv:2105.11943, v1 submitted 2021-05-25 | infinity-operad homology；bar-cobar 型结构 | 第十八、十九章后的背景入口 |
| Hoffbeck--Moerdijk, “Koszul duality for algebras over infinity-operads” | arXiv:2602.08851, v1 submitted 2026-02-09 | linear infinity-operads；infinity-cooperads | Koszul 前沿边界 |
| Pavlova, “Boardman-Vogt tensor product and wreath product of operadic categories” | arXiv:2601.03985, v2 revised 2026-05-28 | operadic categories；wreath product；Boardman--Vogt tensor product | operadic categories 前沿入口 |
| Arakawa--Carmona--Pratali, “Relative dendroidal Rezk nerve and applications” | arXiv:2606.11895, v1 submitted 2026-06-10 | relative infinity-operads；dendroidal Rezk nerve | localization 前沿边界 |
| Batanin--Kock--Weber, “Operadic categories as (pseudo)-simplicial groupoids” | arXiv:2606.15671, v1 submitted 2026-06-14 | operadic categories；pseudo-simplicial groupoids | higher nerve 前沿入口 |
| Yuan, “Higher operad structure for Fukaya categories” | arXiv:2603.08039, v1 submitted 2026-03-09 | dg $\mathbf{fc}$-multicategories；Fukaya categories | 几何应用前沿入口 |

**规则 21.4.** 表 21.3 中的条目均不得作为本书前二十章的证明步骤。允许的引用语句形如：

> 某方向在近期文献中有进一步发展；本书只记录其对象和验证义务。

不允许的引用语句形如：

> 由某 2026 预印本可知本章定理成立。

## 21.2 Infinity-operad 的同调与 Koszul 对偶

**资料 21.5.** Hoffbeck 和 Moerdijk 的 “Homology of infinity-operads” 建立了 infinity-operad 的同调理论，并把 bar-cobar 型构造推广到 dendroidal/infinity-operadic 语境。它适合放在本书第十八至第十九章之后，作为 infinity-operad 的同调代数入口。

**资料 21.6.** Hoffbeck 和 Moerdijk 的 “Koszul duality for algebras over infinity-operads” 是 2026 年预印本。其核心对象是 linear infinity-operad 上的代数与 infinity-cooperad 上的余代数，并把此前的 infinity-operadic Koszul 对偶推进到代数和余代数范畴。该文目前只能作为研究边界资料；若要纳入第九至第十章的 Koszul 叙述，需要先完成 classical operad、linear infinity-operad 和树范畴预层之间的定义比较。

**待验证问题 21.7.** 后续需要检查：

- linear infinity-operad 的定义与本书 dendroidal set 口径如何对应；
- 该 Koszul 对偶是否给出 Quillen 型结构、派生等价，还是更一般的 bar-cobar 对偶；
- classical quadratic Koszul duality 在该框架中作为何种特例出现。

## 21.3 Operadic categories 与 Boardman-Vogt tensor product

**资料 21.8.** Pavlova 的 “Boardman-Vogt tensor product and wreath product of operadic categories” 研究 operadic categories 的 wreath product，并把它与 `Set` 中 colored operads 的 Boardman-Vogt tensor product 联系起来。它属于本书第六、七和十四章之后的内容，因为读者需要先掌握 colored operad、operadic Grothendieck construction 和 Boardman-Vogt tensor product。

**资料 21.9.** Batanin、Kock 和 Weber 的 “Operadic categories as (pseudo)-simplicial groupoids” 为 operadic category 构造 operadic nerve，把 chosen local terminals、fiber functor 和 cardinality functor 汇入 pseudo-simplicial groupoid 的结构中。该文适合连接本书第六章的 colored/multicategory 语言、第十六章的树范畴语言和第十八章以后的 higher nerve 主题。

**待验证问题 21.10.** 后续需要检查：

- operadic nerve 是否能作为本书定义 operadic category 的替代公理化；
- pseudo-simplicial groupoid 的相干性是否需要单独的 2-范畴背景；
- IKEO map 与已有 operad-over-operadic-category 定义的精确等价条件。

## 21.4 Dendroidal Rezk nerve 与 operadic localization

**资料 21.11.** Arakawa、Carmona 和 Pratali 的 “Relative dendroidal Rezk nerve and applications” 把 dendroidal Rezk nerve 推广到 relative infinity-operads，并将其与 infinity-operad 的 localization 联系起来。该文还给出 operadic localization、cyclic operads、operadic modules 和 factorization algebras 相关应用。

**待验证问题 21.12.** 后续需要检查：

- relative infinity-operad 的模型与本书第十八章采用的模型是否一致；
- dendroidal Rezk nerve 的 fibrancy 和 localization 条件需要哪些模型结构假设；
- 该结果与 Mazel-Gee 型 localization 定理的精确包含关系。

## 21.5 几何中的高阶 operadic 结构

**资料 21.13.** Yuan 的 “Higher operad structure for Fukaya categories” 把 pseudo-holomorphic polygons 的模空间组织成 $\mathbf{fc}$-multicategory 结构，并以 dg $\mathbf{fc}$-multicategory 统一表达 $A_\infty$ algebra、module、bimodule 和 category 型结构。该文适合放在本书第二十章的几何应用之后。

**待验证问题 21.14.** 后续需要检查：

- $\mathbf{fc}$-multicategory 与 ordinary operad、colored operad、double category 之间的精确定义关系；
- Fukaya category 的分析输入是否能在本书中作为黑箱引用；
- curved $A_\infty$ 结构的符号约定是否与本书同伦代数章节一致。

## 21.6 模型差异表

下表记录本章前沿对象与本书主体对象之间的最小差异。差异未消除前，不能把相应前沿结果移入正文定理链。

| 前沿对象 | 本书已有对象 | 缺失比较 |
| --- | --- | --- |
| linear infinity-operad | dg operad；dendroidal infinity-operad | 线性化方式、bar/cobar 构造、代数/余代数范畴模型 |
| operadic category | colored operad；multicategory | chosen local terminals、fiber functor、cardinality functor 与 colored substitution 的关系 |
| operadic nerve of an operadic category | dendroidal nerve；category of operators nerve | pseudo-simplicial coherence 与 active/inert 或 tree-face structure 的比较 |
| relative infinity-operad | relative category；Lurie-style infinity-operad；dendroidal infinity-operad | weak equivalence 标记、Rezk nerve、localization universal property |
| dg $\mathbf{fc}$-multicategory | $A_\infty$-category；colored dg operad | curve-counting operations、module/bimodule slots、curvature 与符号约定 |

**判定 21.15.** 本章五类方向均被本书逻辑地图覆盖，但不都被本书证明体系吸收。覆盖的意思是：每个前沿对象都能定位到已有章节之后的自然位置；未吸收的意思是：对应定义、模型比较和深定理尚未在本书内部证明。

**证明.** Infinity-operadic Koszul 对偶依赖第八、九、十七至十九章；operadic categories 依赖第五、七、十六和十八章；relative dendroidal Rezk nerve 依赖第十七至十九章；Fukaya 高阶结构依赖第十、十三和二十章。因此它们都有前置章节位置。另一方面，表中每一行都含至少一个本书尚未定义或尚未证明的比较：linear infinity-operad 的线性模型、pseudo-simplicial groupoid 的相干性、relative infinity-operad 的 localization universal property、以及 Fukaya theory 的分析输入均不在前二十章内部推出。故只能称为逻辑覆盖，不能称为定理吸收。$\square$

## 21.7 进入正文的验证流程

**流程 21.16.** 若未来版本准备把本章某一前沿条目的某个结论写成正文定理，应按以下顺序执行。

1. **版本冻结.** 记录 arXiv 版本、提交或修订日期、出版状态和可能的 erratum。
2. **定义翻译.** 把论文中的基本对象翻译到本书已定义对象，或新增定义并给出与旧对象的比较。
3. **假设拆分.** 把结论所需假设分成集合论小性、代数底环、模型结构、同伦完备性、几何分析输入五类。
4. **定理定位.** 记录 theorem/proposition/lemma 编号和证明中引用的上游结果。
5. **符号转换.** 若涉及链复形、suspension、brace、$A_\infty$ 或 $L_\infty$ 结构，必须通过附录 E 和 W。
6. **账本登记.** 在附录 D 和 [THEOREM_LEDGER.md](THEOREM_LEDGER.md) 中把该结论从研究边界改为外部输入。

**命题 21.17.** 流程 21.16 是防止前沿结果污染基础定理链的充分检查。

**证明.** 基础定理链可能被污染的方式只有三类：使用未固定版本的结论、使用模型不一致的结论、使用假设不足的结论。步骤 1 排除版本不确定性；步骤 2 和 5 排除模型与符号不一致；步骤 3 和 4 排除假设不足和来源不明；步骤 6 保证结论的状态能被全书后续引用检查发现。因此若流程完成，该结论作为外部输入进入正文时不会被误认为内部证明或无条件基础事实。$\square$

## 本章小结

截至 2026-06-30，operad theory 的近期研究集中在 infinity-operadic Koszul 对偶、operadic categories 的 higher nerve、relative dendroidal Rezk nerve、operadic localization 和几何中的高阶 operadic structure。它们都与本书主体相关，但目前不应直接进入基础章节的定理链。后续扩写时，应在现有主体草稿基础上逐条验证这些前沿论文的定义、定理编号、版本和依赖。

## 练习

**练习 21.1.** 选择资料 21.6，列出其中 “algebra over a linear infinity-operad” 的定义依赖，并说明它与 classical operad algebra 的差异。

**练习 21.2.** 选择资料 21.9，解释 operadic category 的哪些结构被 operadic nerve 编码为 simplicial identities。

**练习 21.3.** 选择资料 21.11，写出 relative infinity-operad localization 与普通 category localization 的至少两个结构性差异。

**练习 21.4.** 对表 21.3 中任一条目，写出一个完整引用包：断言、模型语境、假设、来源定位和符号转换。

**练习 21.5.** 说明为什么表 21.3 中的某个 2026 条目不能直接替代第十四章或第十九章的外部输入定理。
