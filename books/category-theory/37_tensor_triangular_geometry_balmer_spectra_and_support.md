# 第三十七章：Tensor triangular geometry、Balmer spectrum 与支撑理论

## 本章目标

本章介绍 tensor triangular geometry。它把小的幂等完备稳定对称幺半范畴看作“非交换或高阶空间”，并用 prime tensor ideals 构造 Balmer spectrum。该理论统一了交换环的 Zariski 谱、代数簇上的 perfect complexes、稳定同伦论中的 chromatic primes，以及表示论中的 support varieties。

## 依赖前置知识

需要三角范畴、稳定 $\infty$-范畴、幂等完备化、厚子范畴、紧对象、对称幺半结构、Bousfield localization、chromatic homotopy 和 perfect complexes。

## 37.1 Tensor triangulated categories

**定义 37.1.** 一个 tensor triangulated category，简称 tt-category，是一个本质小的幂等完备三角范畴 $T$，配有精确的对称幺半结构

$$
\otimes:T\times T\to T,\qquad \mathbb 1\in T,
$$

且 $\otimes$ 对每个变量都保持 distinguished triangles。

在稳定 $\infty$-范畴语境中，通常从小幂等完备稳定对称幺半 $\infty$-范畴 $C$ 出发，取 $T=hC$。

**定义 37.2.** $T$ 的 thick tensor ideal 是 full triangulated subcategory $I\subseteq T$，满足：

1. $I$ 对 direct summands 封闭；
2. 若 $x\in I$ 且 $t\in T$，则 $x\otimes t\in I$。

**命题 37.3.** 任意对象族 $S\subseteq T$ 生成最小 thick tensor ideal，记作 $\langle S\rangle_\otimes$。

**证明.** 取所有包含 $S$ 的 thick tensor ideals 的交。交仍对平移、锥、有限直和、直和项和与任意对象张量封闭，因此是 thick tensor ideal，并显然是最小者。$\square$

## 37.2 Prime ideals 与 Balmer spectrum

**定义 37.4.** Proper thick tensor ideal $\mathfrak p\subsetneq T$ 称为 prime，若

$$
x\otimes y\in\mathfrak p
$$

蕴含 $x\in\mathfrak p$ 或 $y\in\mathfrak p$。

**定义 37.5.** Balmer spectrum 定义为 prime thick tensor ideals 的集合

$$
\operatorname{Spc}(T)=\{\mathfrak p\subset T\mid \mathfrak p\text{ prime}\}.
$$

对 $x\in T$，定义其支撑

$$
\operatorname{supp}(x)=\{\mathfrak p\in\operatorname{Spc}(T)\mid x\notin\mathfrak p\}.
$$

这些集合生成 $\operatorname{Spc}(T)$ 的拓扑。

**命题 37.6.** 对 $x,y\in T$，有

$$
\operatorname{supp}(0)=\varnothing,\qquad
\operatorname{supp}(\mathbb 1)=\operatorname{Spc}(T),
$$

$$
\operatorname{supp}(x\oplus y)=\operatorname{supp}(x)\cup\operatorname{supp}(y),
$$

且

$$
\operatorname{supp}(x\otimes y)=\operatorname{supp}(x)\cap\operatorname{supp}(y).
$$

**证明.** $0$ 属于所有 thick tensor ideals，故支撑为空。若 $\mathbb 1\in\mathfrak p$，则对任意 $t$ 有 $t\simeq\mathbb 1\otimes t\in\mathfrak p$，从而 $\mathfrak p=T$，与 proper 矛盾，因此 $\mathbb 1$ 不在任何 prime 中。对直和，thick 子范畴对直和与直和项封闭，所以 $x\oplus y\in\mathfrak p$ 当且仅当 $x,y\in\mathfrak p$，取补得到并。对张量，prime 条件给出 $x\otimes y\in\mathfrak p$ 当且仅当 $x\in\mathfrak p$ 或 $y\in\mathfrak p$；取补得到交。$\square$

## 37.3 Thomason subsets 与分类定理

**定义 37.7.** 拓扑空间 $X$ 的 Thomason subset 是可写为 quasi-compact open subsets 的补的并的子集。若 $X$ spectral，这等价于 specialization-closed 且满足合适 quasi-compactness 条件的子集。

**外部输入定理 37.8（Balmer 分类定理）.** 对 rigid tt-category $T$，radical thick tensor ideals 与 $\operatorname{Spc}(T)$ 的 Thomason subsets 之间存在包含保持对应：

$$
I\mapsto\bigcup_{x\in I}\operatorname{supp}(x).
$$

在许多 noetherian 情形下，这给出 thick tensor ideals 与 specialization-closed subsets 的分类。

**定义 37.9.** Thick tensor ideal $I$ 的 radical 定义为

$$
\sqrt I=\{x\in T\mid x^{\otimes n}\in I\text{ for some }n\ge1\}.
$$

**命题 37.10.** 若 $I$ 为 radical thick tensor ideal，则

$$
x\in I\Rightarrow \operatorname{supp}(x)\subseteq\bigcup_{y\in I}\operatorname{supp}(y).
$$

反过来，Balmer 分类定理说明支撑包含关系可在刚性假设下检测 radical membership。

**证明.** 第一部分由并的定义立即得到。反向不是形式推论，而是定理 37.8 的内容：支撑落在 $I$ 的 Thomason 支撑中时，$x$ 属于与该 Thomason 子集对应的 radical thick tensor ideal。$\square$

## 37.4 例子：交换环与 perfect complexes

**外部输入定理 37.11.** 若 $R$ 是交换环，则

$$
\operatorname{Spc}(\operatorname{Perf}(R))\cong\operatorname{Spec}R.
$$

在该同胚下，perfect complex $P$ 的 Balmer support 对应于

$$
\{\mathfrak p\in\operatorname{Spec}R\mid P_{\mathfrak p}\not\simeq0\}.
$$

**命题 37.12.** 若 $P$ 是 perfect $R$-complex，且 $P\simeq0$，则其支撑为空。若 $R$ noetherian，反过来支撑为空蕴含 $P\simeq0$。

**证明.** $P=0$ 时局部化 $P_{\mathfrak p}=0$ 对所有 $\mathfrak p$ 成立，支撑为空。反向在 noetherian 情形中，perfect complex 有有限生成 cohomology，若所有局部化为零，则所有 cohomology modules 局部化为零，从而 cohomology 为零，故 $P\simeq0$。$\square$

## 37.5 例子：有限谱与 chromatic primes

**外部输入定理 37.13.** $p$-local finite spectra 的 tt-geometry 与 chromatic thick subcategory theorem 相容。其 prime thick tensor ideals 由 chromatic type 给出，Balmer spectrum 是按高度排列的空间。

**命题 37.14.** 若有限 $p$-local 谱 $F$ 的 type 至少为 $n$，则它属于由 type $n$ spectra 生成的 thick tensor ideal。

**证明.** 这是厚子范畴定理的直接应用。有限 $p$-local spectra 的 thick subcategories 按 type 线性排列；type 至少为 $n$ 的对象正构成相应厚子范畴。该子范畴对 smash product 封闭，因此是 thick tensor ideal。$\square$

## 37.6 Tensor triangular geometry 与局部化

**命题 37.15.** 若 $I\subseteq T$ 是 thick tensor ideal，则 Verdier quotient $T/I$ 继承张量结构，当张量能良好下降时，商函子 $T\to T/I$ 是 tt-functor。

**证明.** 因 $I$ 是 tensor ideal，若 $x\to x'$ 的锥在 $I$ 中，则对任意 $y$，$x\otimes y\to x'\otimes y$ 的锥也在 $I$ 中。故张量把被倒置的态射送到被倒置的态射，从而由 Verdier quotient 泛性质下降到 $T/I$。对称幺半相干由 $T$ 中相干下降得到。$\square$

**注 37.16.** 这与第二十六章的稳定 presentable quotient 和第三十二章的 chromatic localization 对应：小的 compact 层面由 tt-geometry 分类，大的 presentable 层面由 Bousfield localization 实现。

## 37.7 本章小结

Tensor triangular geometry 用 prime thick tensor ideals 构造 $\operatorname{Spc}(T)$，把稳定同伦论、代数几何和表示论中的支撑理论统一起来。Balmer spectrum 把“对象在哪里非零”变成拓扑支撑；分类定理把 radical thick tensor ideals 与 Thomason subsets 对应。它是 compact tensor triangulated categories 的几何化语言。

## 练习

**练习 37.1.** 定义 tt-category。

**练习 37.2.** 定义 thick tensor ideal。

**练习 37.3.** 证明对象族生成最小 thick tensor ideal。

**练习 37.4.** 定义 prime thick tensor ideal。

**练习 37.5.** 定义 Balmer spectrum 和对象支撑。

**练习 37.6.** 证明 $\operatorname{supp}(x\otimes y)=\operatorname{supp}(x)\cap\operatorname{supp}(y)$。

**练习 37.7.** 定义 Thomason subset。

**练习 37.8.** 陈述 Balmer 分类定理。

**练习 37.9.** 说明 $\operatorname{Spc}(\operatorname{Perf}(R))$ 与 $\operatorname{Spec}R$ 的关系。

**练习 37.10.** 证明 noetherian 情形中 perfect complex 支撑为空则对象为零。

**练习 37.11.** 说明有限谱的 tt-geometry 与 chromatic type 的关系。

**练习 37.12.** 解释 thick tensor ideal 为什么允许张量结构下降到 Verdier quotient。
