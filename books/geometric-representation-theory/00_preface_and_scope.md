# 序章：范围、严格性标准和资料源

## 本章目标

本章说明本书研究的数学对象、默认约定、严格性标准和资料源边界。读者在进入第一章前，应当知道本书中“geometric representation theory”不是一个单一公理化对象，而是一组把表示范畴转化为几何范畴并用几何操作研究表示的技术。

## 依赖前置知识

需要熟悉基本范畴论、交换代数、代数簇、线性代数群、Lie 代数和有限群表示。导出范畴、perverse sheaves、D-modules、stack 和 infinity category 不预设，会在正文或附录中逐步引入。

## 0.1 本书研究的对象

**约定 0.1.** 本书固定 Grothendieck universes
$$
\mathcal U\in\mathcal V\in\mathcal W.
$$
若不特别说明，小范畴、小集合和代数簇均取 $\mathcal U$-小模型；由所有 $\mathcal U$-小对象构成的范畴视为 $\mathcal V$-小。

**约定 0.2.** 本书默认底域 $k$ 为代数闭域。涉及复解析 topology、D-modules、Riemann-Hilbert 对应和 BGG category $\mathcal O$ 时，默认 $k=\mathbb C$。sheaf 的默认系数域为特征 $0$ 域 $E$。

**约定 0.3.** 默认 $G$ 为连通 reductive algebraic group，$B\subset G$ 为 Borel subgroup，$T\subset B$ 为 maximal torus，$W=N_G(T)/T$ 为 Weyl group，$\mathcal B=G/B$ 为 flag variety。

**定义 0.4.** 本书中“几何表示论（geometric representation theory）”指如下类型的数学过程：给定一个代数或拓扑表示论问题，构造带群作用、分层、辛结构、导出结构或 stack 结构的几何对象 $X$，再用 $X$ 上的 sheaves、D-modules、coherent sheaves、K-theory、Borel-Moore homology 或其范畴化版本描述原表示范畴或其不变量。

这个定义不是公理化定理，而是本书的范围约定。它允许同一本书同时讨论 category $\mathcal O$、Hecke categories、Springer theory、geometric Satake、quiver varieties 和 geometric Langlands，但每个方向都必须有自己的对象、范畴和 functor。

## 0.2 严格性标准

**约定 0.5.** 本书采用如下证明标准。

1. 一个定义必须指定底域、对象所在范畴、结构映射和公理。
2. 一个几何对象必须说明是 scheme、variety、ind-scheme、stack、derived stack 还是 analytic space。
3. 一个 sheaf-theoretic construction 必须说明使用的 topology、系数、constructibility 条件和 functor 类型。
4. 一个命题、引理或定理必须给出证明；若证明依赖大型外部结果，则标注“外部输入定理”。
5. 一个前沿方向可以写入研究边界，但不得作为基础证明链的步骤。

**例 0.6.** “Kazhdan-Lusztig 多项式由 Schubert variety 的 intersection cohomology 给出”不能只作为口号。严格写法必须至少说明：

- Coxeter group 或 Weyl group $W$；
- Schubert variety $\overline X_w\subset G/B$；
- intersection complex $\operatorname{IC}(\overline X_w)$ 的 normalization；
- stalk cohomology 取在哪个点或 stratum；
- Hecke algebra 中标准基和 Kazhdan-Lusztig 基的 convention；
- 该识别是外部输入定理，除非正文重证 decomposition theorem 和相关 purity。

## 0.3 为什么从 flag variety 开始

许多几何表示论对象都以 flag geometry 为原型。

1. Bruhat decomposition 把 $\mathcal B=G/B$ 分成由 $W$ 标号的 affine cells。
2. Schubert varieties 的奇点控制 Hecke algebra 的 canonical basis 和 Kazhdan-Lusztig 多项式。
3. $G/B$ 上的 twisted D-modules 几何化了带固定中心 character 的 $\mathfrak g$-模。
4. Springer resolution 把 nilpotent cone 与 flag variety 联系起来，并产生 Weyl group 表示。
5. affine Grassmannian 是 loop group 版本的 flag geometry，并通向 geometric Satake。

因此本书先建立 $G$、$B$、$T$、$W$、$G/B$、Schubert cells、category $\mathcal O$ 和 equivariant sheaves，再进入高级方向。

## 0.4 外部输入和研究边界

本书会使用一些大型定理作为外部输入。包括但不限于：

- Borel fixed point theorem 和 Bruhat decomposition；
- PBW theorem、Harish-Chandra isomorphism 和 highest weight theory；
- BBD perverse sheaf formalism 和 decomposition theorem；
- Kazhdan-Lusztig conjecture 及其几何证明；
- Springer correspondence；
- Beilinson-Bernstein localization；
- Riemann-Hilbert correspondence；
- geometric Satake equivalence；
- Soergel categorification 和 Hodge theory；
- Nakajima quiver variety construction；
- BFN Coulomb branch construction；
- 2024 geometric Langlands proof series。

其中前八类会进入正文定理链，但必须定位来源和假设。最后三类在第一版中主要作为研究边界，除非后续章节补足模型和 locator。

## 本章小结

本书把几何表示论处理为一套可检查的对象和 functor，而不是若干漂亮类比。基础章节固定代数群、flag variety、表示范畴和 sheaf theory 约定；高级章节只在外部输入和模型假设明确之后调用大型定理。当前书稿已经进入主体教材化收口阶段，后续目标是完成出版级 locator、交叉引用和模型假设校对。

## 练习

**练习 0.1.** 写出一个例子，说明同一个符号 $G/B$ 可以表示复代数簇、复解析空间或 quotient stack 的 atlas，并说明三种语境中 sheaf category 的差异。

**练习 0.2.** 解释为什么“$D^b_B(G/B)$ 是 Hecke category”不是完整定义。要求写出卷积 correspondence。

**练习 0.3.** 给出一个大型外部输入定理，并列出把它纳入教材正文需要检查的五项假设。
