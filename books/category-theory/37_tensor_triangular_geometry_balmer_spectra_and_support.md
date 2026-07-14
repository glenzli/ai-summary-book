# 第三十七章：Tensor triangular geometry、Balmer spectrum 与支撑理论

交换环的素理想由乘法检测，稳定张量范畴中的 prime thick tensor ideals 扮演同样角色。Balmer spectrum 把这些素理想组成拓扑空间，对象的支撑则记录它在哪些素点不消失；在 $\operatorname{Perf}(R)$ 上可恢复 $\operatorname{Spec}R$，在稳定同伦和表示论中则产生 chromatic 或 support-variety 几何。本章建立 prime、support datum 与分类定理的精确关系，并区分小的紧对象范畴与其 presentable completion。

所需背景是稳定/三角范畴、幂等完备、厚子范畴、紧对象和对称幺半结构。Thomason subset 分类与具体谱识别作为带假设的外部输入；rigidity、essential smallness 和 tensor ideal 条件会在每条定理中保留。

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

**证明.** 取所有包含 $S$ 的 thick tensor ideals 的交。交仍对平移、锥、有限直和、直和项和与任意对象张量封闭，因此是 thick tensor ideal。按交的定义，它包含于任意一个包含 $S$ 的 thick tensor ideal，所以是最小者。$\square$

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

Balmer 拓扑规定每个 $\operatorname{supp}(x)$ 为闭集；它们构成闭集的一组基，并且补集 $U(x)$ 是 quasi-compact open。

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

**定义 37.7.** 拓扑空间 $X$ 的 Thomason subset 是形如

$$
\bigcup_{\lambda\in\Lambda}Z_\lambda
$$

的子集，其中每个 $Z_\lambda$ 为闭集且 $X\setminus Z_\lambda$ quasi-compact open。若 $X$ Noetherian，这恰好是 specialization-closed subsets；对一般 spectral space，不能删去 quasi-compact-complement 条件。

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

**命题 37.12.** 对任意交换环 $R$ 与 perfect $R$-complex $P$，有

$$
P\simeq0\quad\Longleftrightarrow\quad\operatorname{supp}(P)=\varnothing.
$$

**证明.** 正向显然。反向假设 $P_{\mathfrak p}\simeq0$ 对所有素理想成立。局部化与有限 perfect complex 的 cohomology 交换，故每个 $H^i(P)$ 在所有素理想处局部化为零。任意非零模都有非空支撑：取非零元，其 annihilator 包含于某个素理想，该元在相应局部化中仍非零。因此 $H^i(P)=0$ 对所有 $i$ 成立，故 $P\simeq0$。此论证不需要 Noetherian 假设。$\square$

## 37.5 例子：有限谱与 chromatic primes

**外部输入定理 37.13.** $p$-local finite spectra 的 tt-geometry 与 chromatic thick subcategory theorem 相容。其 prime thick tensor ideals 由 chromatic type 给出，Balmer spectrum 是按高度排列的空间。

**命题 37.14.** 若有限 $p$-local 谱 $F$ 的 type 至少为 $n$，则它属于由 type $n$ spectra 生成的 thick tensor ideal。

**证明.** 这是厚子范畴定理的直接应用。有限 $p$-local spectra 的 thick subcategories 按 type 线性排列；type 至少为 $n$ 的对象正构成相应厚子范畴。该子范畴对 smash product 封闭，因此是 thick tensor ideal。$\square$

## 37.6 Tensor triangular geometry 与局部化

**命题 37.15.** 若 $I\subseteq T$ 是 thick tensor ideal，则 Verdier quotient $T/I$ 典范继承张量结构，商函子 $T\to T/I$ 是 tt-functor。

**证明.** 因 $I$ 是 tensor ideal，若 $x\to x'$ 的锥在 $I$ 中，则对任意 $y$，$x\otimes y\to x'\otimes y$ 的锥也在 $I$ 中。故张量把被倒置的态射送到被倒置的态射，从而由 Verdier quotient 泛性质下降到 $T/I$。对称幺半相干由 $T$ 中相干下降得到。$\square$

**注 37.16.** 这与第二十六章的稳定 presentable quotient 和第三十二章的 chromatic localization 对应：小的 compact 层面由 tt-geometry 分类，大的 presentable 层面由 Bousfield localization 实现。

## 37.7 支撑的形式性质与函子性

**命题 37.17.** 对 $T$ 中任意 distinguished triangle

$$
x\to y\to z\to \Sigma x
$$

有

$$
\operatorname{supp}(\Sigma x)=\operatorname{supp}(x),
\qquad
\operatorname{supp}(y)\subseteq\operatorname{supp}(x)\cup\operatorname{supp}(z).
$$

**证明.** Thick 子范畴对平移封闭，所以 $x\in\mathfrak p$ 当且仅当 $\Sigma x\in\mathfrak p$，第一式成立。若 $\mathfrak p$ 不属于 $\operatorname{supp}(x)\cup\operatorname{supp}(z)$，则 $x,z\in\mathfrak p$。由于 $\mathfrak p$ 是 triangulated subcategory，三角中两项属于 $\mathfrak p$ 蕴含第三项 $y$ 也属于 $\mathfrak p$。故 $\mathfrak p\notin\operatorname{supp}(y)$，得到包含关系。$\square$

**命题 37.18.** 对任意 $n\ge1$，

$$
\operatorname{supp}(x^{\otimes n})=\operatorname{supp}(x).
$$

因此 Balmer 支撑只看对象的 tensor-nilpotence radical。

**证明.** 由命题 37.6，

$$
\operatorname{supp}(x^{\otimes n})
=\underbrace{\operatorname{supp}(x)\cap\cdots\cap\operatorname{supp}(x)}_{n\text{ 次}}
=\operatorname{supp}(x).
$$

这说明把对象替换为正张量幂不改变其支撑。$\square$

**命题 37.19（谱的函子性）.** 若 $F:T\to T'$ 是 tt-functor，则对任意 prime $\mathfrak q\in\operatorname{Spc}(T')$，原像

$$
F^{-1}(\mathfrak q)=\{x\in T\mid F(x)\in\mathfrak q\}
$$

是 $T$ 的 prime thick tensor ideal。因而 $F$ 诱导连续映射

$$
\operatorname{Spc}(T')\to\operatorname{Spc}(T).
$$

**证明.** 因 $F$ 保三角、直和项和张量，$F^{-1}(\mathfrak q)$ 是 thick tensor ideal。它 proper，因为 $F(\mathbb 1_T)\simeq\mathbb 1_{T'}$，而 prime ideal $\mathfrak q$ 不含单位。若 $x\otimes y\in F^{-1}(\mathfrak q)$，则

$$
F(x)\otimes F(y)\simeq F(x\otimes y)\in\mathfrak q.
$$

由 $\mathfrak q$ prime，$F(x)\in\mathfrak q$ 或 $F(y)\in\mathfrak q$，即 $x\in F^{-1}(\mathfrak q)$ 或 $y\in F^{-1}(\mathfrak q)$。连续性由

$$
F^{-1}_{\operatorname{Spc}}(\operatorname{supp}(x))=\operatorname{supp}(F(x))
$$

对支撑基开集的计算得到。$\square$

## 37.8 稳定张量范畴的素谱

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

**练习 37.13.** 证明三角 $x\to y\to z\to\Sigma x$ 给出 $\operatorname{supp}(y)\subseteq\operatorname{supp}(x)\cup\operatorname{supp}(z)$。

**练习 37.14.** 证明 $\operatorname{supp}(x^{\otimes n})=\operatorname{supp}(x)$。

**练习 37.15.** 证明 tt-functor $F:T\to T'$ 诱导 $\operatorname{Spc}(T')\to\operatorname{Spc}(T)$。
