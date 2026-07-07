# 序章：范围、严格性标准和资料源

## 本章目标

本章说明本书研究的数学对象、严格性标准和资料源边界。读者在进入第一章前，应当知道本书中“prism”“prismatic cohomology”“$p$-adic Hodge theory”分别处于哪一层级，哪些陈述会在书内证明，哪些陈述必须作为外部输入定理。

## 依赖前置知识

需要熟悉交换代数、模、张量积、导出范畴的基本语言、scheme/formal scheme 的基本定义、Galois 表示和同调代数。Perfectoid spaces、almost mathematics、pro-etale topology 和 prismatic cohomology 不预设，会在后续章节逐步引入；但一些大型比较定理只作为外部输入使用。

## 0.1 本书研究的对象

**约定 0.1.** 全书固定素数 $p$。所有环默认是交换含幺环。除非特别说明，$p$-完备、$(p,I)$-完备和 completed tensor product 均按导出完备化理解。

**约定 0.2.** 本书中单独出现的“prism”默认指 Bhatt-Scholze 意义下的有界 prism。更具体地说，它是二元组 $(A,I)$，其中 $A$ 是 $\delta$-环，$I\subset A$ 定义 Cartier divisor，$A$ 是导出 $(p,I)$-完备，并且 $p\in I+\phi_A(I)A$；有界性指 $A/I$ 的 $p^\infty$-torsion 有界。

**约定 0.3.** 本书中 $p$-adic Hodge theory 分三层处理。

1. **Classical layer.** Fontaine period rings、$B$-admissible representations、Hodge-Tate/de Rham/crystalline/semistable representations。
2. **Integral layer.** $A_{\inf}$-cohomology、Breuil-Kisin modules、BMS integral comparison、Nygaard filtration 和 syntomic complexes。
3. **Prismatic layer.** Prism、prismatic site、prismatic cohomology、prismatic $F$-crystals、prismatization 和 $F$-gauges。

不同层级之间由比较定理连接，而不是由术语相似性连接。

## 0.2 严格性标准

**约定 0.4.** 本书采用如下证明标准。

- 一个定义必须指定所在范畴、完备性、拓扑或 site、结构映射和公理。
- 一个例子必须验证定义中的非零因子条件、完备性、Frobenius 条件或滤过条件。
- 一个命题、引理或定理必须给出证明；若证明依赖大型外部结果，则标注“外部输入定理”。
- 一个比较定理必须列出假设，并说明比较对象之间是同构、拟同构、filtered isomorphism、$\varphi$-equivariant isomorphism 还是范畴等价。

**例 0.5.** “Prismatic cohomology unifies crystalline, de Rham and etale cohomology”不是可检查的数学陈述。严格写法必须固定 bounded prism $(A,I)$ 和 smooth $p$-adic formal scheme $X$ over $A/I$，定义
$$
R\Gamma_\Delta(X/A)=R\Gamma((X/A)_\Delta,\mathcal O_\Delta),
$$
然后分别说明 crystalline specialization、Hodge-Tate specialization、de Rham specialization、perfect prism 上的 etale comparison 等定理的假设和结论。

## 0.3 本书不做什么

**约定 0.6.** 本书不尝试重证以下大型结果：

- Faltings almost purity theorem；
- Fontaine-Faltings-Tsuji comparison theorems；
- Scholze perfectoid tilting equivalence；
- Bhatt-Morrow-Scholze $A_{\inf}$ integral comparison theorem；
- Bhatt-Scholze prismatic comparison theorem；
- Bhatt-Scholze prismatic $F$-crystal classification theorem；
- Bhatt-Lurie prismatization 的主解释定理。

这些结果会被明确标为外部输入。书内可以证明定义层、线性代数层、局部例子层和形式推论层的命题。

## 0.4 为什么先讲 $\delta$-环

Prism 的核心不是先给一个新 cohomology theory，而是先把“Frobenius lift 的 infinitesimal arithmetic calculus”编码在 $\delta$-环中。若 $A$ 是 $p$-torsionfree，给定 $\delta:A\to A$ 等价于给定一个 Frobenius lift $\phi:A\to A$，公式为
$$
\phi(x)=x^p+p\delta(x).
$$
Prism 的条件 $p\in I+\phi(I)A$ 正是要求 Frobenius 对 Cartier divisor 的相对位置足够横截，从而能把 de Rham、crystalline、Hodge-Tate 与 etale specialization 放在同一个对象中。

## 0.5 资料源和研究边界

本书基础部分以 Bhatt-Scholze 的 prisms and prismatic cohomology 为主源，以 Bhatt-Morrow-Scholze 的 integral $p$-adic Hodge theory 为积分层主源，以 Fontaine、Faltings、Tsuji、Brinon-Conrad、Berger、Kedlaya-Liu 和 Scholze 为 classical and perfectoid background。

截至 2026-07-08，prismatic theory 的近期研究仍在活跃发展，尤其在以下方向：

- prismatic cohomology with coefficients；
- rational Hodge-Tate prismatic crystals and non-abelian $p$-adic Hodge theory；
- syntomic Steenrod operations and spectral prismatic $F$-gauges；
- prismatic realization of Shimura varieties；
- finite flat group schemes and prismatic $F$-gauges。

这些方向会进入第八章的研究边界。除非完成独立核验，本书不会把近期预印本的新结论纳入基础定理链。

## 0.6 教材阅读路线

**说明 0.7.** 本书有两条阅读路线。

第一条是基础路线：
$$
\delta\text{-rings}\longrightarrow
prisms\longrightarrow
prismatic\ site\longrightarrow
R\Gamma_\Delta\longrightarrow
comparison\ theorems.
$$
这条路线对应第一至三章、第九至十一章和附录 G、H、K。

第二条是表示论路线：
$$
period\ rings\longrightarrow
integral\ cohomology\longrightarrow
BK/BKF\ modules\longrightarrow
prismatic\ F\text{-crystals}\longrightarrow
crystalline\ lattices.
$$
这条路线对应第四至六章、第十二章和附录 I、J。

**命题 0.8（内部学习闭包）.** 读者若掌握附录 G、H、I、J 的技术词典，则可以在不查外部资料的情况下理解本书所有基础定义和内部证明；需要外部资料的只剩大型比较定理和前沿应用。

**证明.** 附录 G 给出 formal schemes、site 和 derived global sections；附录 H 给出 $\delta$-环和 prism 条件的代数计算；附录 I 给出 crystals 和 descent；附录 J 给出 period/lattice 线性代数。正文基础定义均依赖这些工具。大型比较定理在相应章节均标为外部输入，因此不属于内部证明闭包。证毕。

## 本章小结

本书把 prismatic theory 作为严格数学对象处理：先定义 $\delta$-环和 prism，再构造 prismatic site 和 cohomology，随后通过比较定理连接 classical and integral $p$-adic Hodge theory。大型比较定理作为外部输入；书内证明集中在定义层、例子层、形式推论和可局部检查的代数计算。

## 练习

**练习 0.1.** 说明为什么“$A$ 是 $p$-完备环”在 prismatic theory 中通常不足够：还必须指定 ordinary completion 还是 derived completion。

**练习 0.2.** 给出一个陈述，其中“同构”和“filtered isomorphism”不能互换使用。

**练习 0.3.** 查阅 `SOURCES.md`，把本书中至少三个外部输入定理按 classical layer、integral layer、prismatic layer 分类。
