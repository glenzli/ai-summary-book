# 序章：范围、严格性标准和资料源

同一个 Weyl group 表示既可能藏在 Springer fiber 的上同调里，也可能由 Steinberg variety 上的卷积产生；同一个 Hecke 代数既能从双陪集写出，又能在 Schubert variety 的层范畴中被范畴化。这些现象并不来自一个包罗万象的定义，而来自一套反复出现的转换：先把代数问题放到带群作用和分层的几何空间上，再以层、D-module、同调或导出范畴读取表示。本书沿这些转换逐层建立对象和函子，并在每次调用大型定理时保留其假设与来源边界。读者需要基本范畴论、交换代数、代数簇、线性代数群、Lie 代数和有限群表示；导出范畴、perverse sheaf、D-module、stack 与高阶范畴所需的部分则随正文引入。

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

其中前八类进入后续数学论证时，都以精确陈述的外部输入出现。Nakajima 表示构造、BFN Coulomb branch 与 geometric Langlands 的一般形式需要更重的几何模型；本书只在相应章节已经声明的版本中使用它们，不从方向名称本身推出结论。

于是全书的阅读顺序由对象之间的需要决定：flag variety 提供有限维原型，equivariant sheaf 与 D-module 使几何操作进入表示范畴，仿射 Grassmannian 和卷积把这一结构提升到 loop group，辛几何与范畴化再展示同一机制的其他实现。第一章从最小的一组 Borel 数据出发，把这条路线落实到可以计算的 Schubert 分层。

## 练习

**练习 0.1.** 写出一个例子，说明同一个符号 $G/B$ 可以表示复代数簇、复解析空间或 quotient stack 的 atlas，并说明三种语境中 sheaf category 的差异。

**练习 0.2.** 解释为什么“$D^b_B(G/B)$ 是 Hecke category”不是完整定义。要求写出卷积 correspondence。

**练习 0.3.** 给出一个大型外部输入定理，并列出把它纳入教材正文需要检查的五项假设。
