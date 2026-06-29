# 第二十一章：2025-2026 研究边界与开放问题目录

## 本章目标

本章记录截至 2026-06-29 已联网核查的近期 operad theory 相关研究入口。它不是前沿结果的教材化证明。本章的作用是把新文献放入本书的逻辑地图中，并说明哪些主题需要后续验证后才能进入正文定理链。

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

## 21.2 Infinity-operad 的同调与 Koszul 对偶

**资料 21.3.** Hoffbeck 和 Moerdijk 的 “Homology of infinity-operads” 建立了 infinity-operad 的同调理论，并把 bar-cobar 型构造推广到 dendroidal/infinity-operadic 语境。它适合放在本书第十八至第十九章之后，作为 infinity-operad 的同调代数入口。

**资料 21.4.** Hoffbeck 和 Moerdijk 的 “Koszul duality for algebras over infinity-operads” 是 2026 年预印本。其核心对象是 linear infinity-operad 上的代数与 infinity-cooperad 上的余代数，并把此前的 infinity-operadic Koszul 对偶推进到代数和余代数范畴。该文目前只能作为研究边界资料；若要纳入第九至第十章的 Koszul 叙述，需要先完成 classical operad、linear infinity-operad 和树范畴预层之间的定义比较。

**待验证问题 21.5.** 后续需要检查：

- linear infinity-operad 的定义与本书 dendroidal set 口径如何对应；
- 该 Koszul 对偶是否给出 Quillen 型结构、派生等价，还是更一般的 bar-cobar 对偶；
- classical quadratic Koszul duality 在该框架中作为何种特例出现。

## 21.3 Operadic categories 与 Boardman-Vogt tensor product

**资料 21.6.** Pavlova 的 “Boardman-Vogt tensor product and wreath product of operadic categories” 研究 operadic categories 的 wreath product，并把它与 `Set` 中 colored operads 的 Boardman-Vogt tensor product 联系起来。它属于本书第六、七和十四章之后的内容，因为读者需要先掌握 colored operad、operadic Grothendieck construction 和 Boardman-Vogt tensor product。

**资料 21.7.** Batanin、Kock 和 Weber 的 “Operadic categories as (pseudo)-simplicial groupoids” 为 operadic category 构造 operadic nerve，把 chosen local terminals、fiber functor 和 cardinality functor 汇入 pseudo-simplicial groupoid 的结构中。该文适合连接本书第六章的 colored/multicategory 语言、第十六章的树范畴语言和第十八章以后的 higher nerve 主题。

**待验证问题 21.8.** 后续需要检查：

- operadic nerve 是否能作为本书定义 operadic category 的替代公理化；
- pseudo-simplicial groupoid 的相干性是否需要单独的 2-范畴背景；
- IKEO map 与已有 operad-over-operadic-category 定义的精确等价条件。

## 21.4 Dendroidal Rezk nerve 与 operadic localization

**资料 21.9.** Arakawa、Carmona 和 Pratali 的 “Relative dendroidal Rezk nerve and applications” 把 dendroidal Rezk nerve 推广到 relative infinity-operads，并将其与 infinity-operad 的 localization 联系起来。该文还给出 operadic localization、cyclic operads、operadic modules 和 factorization algebras 相关应用。

**待验证问题 21.10.** 后续需要检查：

- relative infinity-operad 的模型与本书第十八章采用的模型是否一致；
- dendroidal Rezk nerve 的 fibrancy 和 localization 条件需要哪些模型结构假设；
- 该结果与 Mazel-Gee 型 localization 定理的精确包含关系。

## 21.5 几何中的高阶 operadic 结构

**资料 21.11.** Yuan 的 “Higher operad structure for Fukaya categories” 把 pseudo-holomorphic polygons 的模空间组织成 $\mathbf{fc}$-multicategory 结构，并以 dg $\mathbf{fc}$-multicategory 统一表达 $A_\infty$ algebra、module、bimodule 和 category 型结构。该文适合放在本书第二十章的几何应用之后。

**待验证问题 21.12.** 后续需要检查：

- $\mathbf{fc}$-multicategory 与 ordinary operad、colored operad、double category 之间的精确定义关系；
- Fukaya category 的分析输入是否能在本书中作为黑箱引用；
- curved $A_\infty$ 结构的符号约定是否与本书同伦代数章节一致。

## 本章小结

截至 2026-06-29，operad theory 的近期研究集中在 infinity-operadic Koszul 对偶、operadic categories 的 higher nerve、dendroidal localization 和几何中的高阶 operadic structure。它们都与本书主体相关，但目前不应直接进入基础章节的定理链。后续扩写时，应在现有主体草稿基础上逐条验证这些前沿论文的定义、定理编号、版本和依赖。

## 练习

**练习 21.1.** 选择资料 21.4，列出其中 “algebra over a linear infinity-operad” 的定义依赖，并说明它与 classical operad algebra 的差异。

**练习 21.2.** 选择资料 21.7，解释 operadic category 的哪些结构被 operadic nerve 编码为 simplicial identities。

**练习 21.3.** 选择资料 21.9，写出 relative infinity-operad localization 与普通 category localization 的至少两个结构性差异。
