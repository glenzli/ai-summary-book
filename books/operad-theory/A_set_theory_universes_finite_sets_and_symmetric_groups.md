# 附录 A：集合论宇宙、有限集骨架与 symmetric group 约定

本附录固定全书的大小约定和对称群作用约定。Operad theory 中许多公式可用 arity $n$ 表示，也可用有限集 $S$ 表示。本书以有限集口径为主，arity 公式只作为骨架化写法。

## A.1 Universes

**约定 A.1.** 全书固定 Grothendieck universes
$$
\mathcal U\in\mathcal V\in\mathcal W.
$$
若不特别说明，“集合”指 $\mathcal U$-小集合。

**定义 A.2.** $\mathbf{Set}_{\mathcal U}$ 是 $\mathcal U$-小集合构成的范畴。一个范畴 $\mathcal C$ 称为 $\mathcal U$-小，若其对象集和态射集均属于 $\mathcal U$。称 $\mathcal C$ 为 locally $\mathcal U$-small，若任意 Hom 集属于 $\mathcal U$。

**约定 A.3.** 本书中基础代数对象通常在 $\mathcal U$ 中取值；由这些对象形成的范畴通常是 $\mathcal V$-小或 locally $\mathcal U$-small。谈论这些范畴的范畴时，默认进入 $\mathcal W$。

**命题 A.4.** 若 $\mathcal C$ 是 $\mathcal U$-小范畴，则 presheaf category
$$
\operatorname{Fun}(\mathcal C^{\operatorname{op}},\mathbf{Set}_{\mathcal U})
$$
是 locally $\mathcal V$-small。

**证明.** 一个 presheaf 是从 $\mathcal C^{\operatorname{op}}$ 到 $\mathbf{Set}_{\mathcal U}$ 的函子，其数据由对 $\mathcal C$ 的对象和态射的函数赋值组成。由于 $\mathcal C$ 的对象集和态射集属于 $\mathcal U$，所有这类赋值组成的集合属于更大 universe $\mathcal V$。两个 presheaves 之间的自然变换是逐对象函数族并满足对每个态射的自然性等式；这是一个 $\mathcal V$-小集合的子集。因此 Hom 集属于 $\mathcal V$。$\square$

## A.2 Finite sets and skeletons

**定义 A.5.** $\mathbf{Fin}_{\mathcal U}$ 是 $\mathcal U$-小有限集和函数构成的范畴。$\mathbf B_{\mathcal U}$ 是 $\mathcal U$-小有限集和双射构成的群胚。

**定义 A.6.** 对 $n\ge0$，记
$$
[n]=\{1,\ldots,n\},\qquad [0]=\varnothing.
$$
对称群记为
$$
\Sigma_n=\operatorname{Aut}_{\mathbf B_{\mathcal U}}([n]).
$$

**命题 A.7.** 群胚 $\mathbf B_{\mathcal U}$ 等价于群胚并
$$
\coprod_{n\ge0}B\Sigma_n.
$$

**证明.** 每个有限集 $S$ 与唯一的 $[|S|]$ 同构，因此 $[n]$ 给出 $\mathbf B_{\mathcal U}$ 的一个骨架。骨架中不同 $[m]$ 与 $[n]$ 在 $m\ne n$ 时无同构；$[n]$ 的自同构群为 $\Sigma_n$。因此骨架正是 $\coprod_{n\ge0}B\Sigma_n$。包含骨架的函子本质满、全忠实，故为群胚等价。$\square$

**推论 A.8.** 给出函子
$$
X:\mathbf B_{\mathcal U}\to\mathcal C
$$
等价于给出对象族 $X(n)\in\mathcal C$ 以及每个 $\Sigma_n$ 在 $X(n)$ 上的作用，并且该等价依赖骨架选择。

**证明.** 由命题 A.7，函子 $\mathbf B_{\mathcal U}\to\mathcal C$ 等价于从 $\coprod B\Sigma_n$ 到 $\mathcal C$ 的函子。后一数据正是对每个 $n$ 给出对象 $X(n)$ 和群同态 $\Sigma_n\to\operatorname{Aut}_{\mathcal C}(X(n))$。$\square$

## A.3 Left and right actions

本书的基本定义使用协变函子
$$
X:\mathbf B_{\mathcal U}\to\mathcal C.
$$
因此 $\sigma\in\Sigma_n$ 给出自同构
$$
X(\sigma):X([n])\to X([n]).
$$
这是左作用约定。

许多 operad 文献使用右 $\Sigma_n$-作用。为互译，定义
$$
x\cdot\sigma=X(\sigma^{-1})(x).
$$

**命题 A.9.** 上式定义右作用：
$$
(x\cdot\sigma)\cdot\tau=x\cdot(\sigma\tau),\qquad x\cdot e=x.
$$

**证明.** 由定义，
$$
(x\cdot\sigma)\cdot\tau
=X(\tau^{-1})(X(\sigma^{-1})(x))
=X(\tau^{-1}\sigma^{-1})(x).
$$
由于 $X$ 是函子，
$$
X(\tau^{-1}\sigma^{-1})=X((\sigma\tau)^{-1}).
$$
故
$$
(x\cdot\sigma)\cdot\tau=X((\sigma\tau)^{-1})(x)=x\cdot(\sigma\tau).
$$
单位元情形由 $X(e)=\operatorname{id}$ 得到。$\square$

**约定 A.10.** 正文中若使用有限集 $S$，采用函子性左作用口径；若使用 arity $n$ 公式并出现 $x\cdot\sigma$，采用右作用口径。两者按命题 A.9 互译。

## A.4 Coinvariants and invariants

设 $G$ 是有限群，$R$ 是交换环，$M$ 是右 $R[G]$-模。

**定义 A.11.** Coinvariants 定义为
$$
M_G=M/\langle m\cdot g-m: m\in M,\ g\in G\rangle.
$$
若 $N$ 是左 $R[G]$-模，则
$$
M\otimes_{R[G]}N
$$
是 usual tensor product over group algebra。

**定义 A.12.** Invariants 定义为
$$
N^G=\{n\in N: gn=n\ \text{for all }g\in G\}.
$$

**命题 A.13.** 若 $R$ 是特征 $0$ 的域，且 $G$ 有限，则 invariants functor 和 coinvariants functor 都 exact，并且 norm map 给出自然同构
$$
M_G\cong M^G
$$
对每个右 $G$-表示 $M$ 成立，其中
$$
M^G=\{m\in M:m\cdot g=m\ \text{for all }g\in G\}.
$$

**证明.** Maschke 定理说明 $R[G]$ 是半单代数，因此 $R[G]$-模范畴半单。半单范畴中所有短正合列分裂，任意加性函子 exact。Invariants 和 coinvariants 均为加性函子，故 exact。Norm map 由
$$
\overline m\mapsto \frac1{|G|}\sum_{g\in G}m\cdot g
$$
给出；它良定义，因为若 $m$ 改变为 $m-m\cdot h$，则平均和为零。复合 $M^G\to M\to M_G\to M^G$ 为恒等，反向复合在 coinvariants 中也为恒等。$\square$

**警告 A.14.** 命题 A.13 在一般交换环或正特征域上不成立。第十四章和第十九章中 commutative dg algebras 与 $E_\infty$-algebras 的区别，正是由这类对称群 coinvariants 问题造成。

## A.5 Coends over finite groupoids

**定义 A.15.** 设 $\mathcal G$ 是小群胚，$F:\mathcal G^{\operatorname{op}}\times\mathcal G\to\mathcal C$。若 $\mathcal C$ 有相应 colimits，则 coend 记为
$$
\int^{g\in\mathcal G}F(g,g).
$$

在 $\mathbf{Set}$ 或 $\mathbf{Mod}_R$ 中，它可表示为 coproduct 或 direct sum 对关系取商：
$$
\coprod_g F(g,g)\big/\sim
$$
其中关系由 $\mathcal G$ 的态射生成。

**命题 A.16.** 若 $\mathcal G=B G$ 是单对象群胚，则 coend 等同于 coinvariants。

**证明.** $BG$ 只有一个对象 $*$，态射为 $G$。函子 $F$ 的值 $F(*,*)$ 带有由左右变量诱导的 $G$-作用。Coend 对所有态射 $g\in G$ 施加关系
$$
F(g,\operatorname{id})(x)\sim F(\operatorname{id},g)(x).
$$
这正是把相应 $G$-作用取 coinvariants 的关系。因此 coend 为 coinvariants。$\square$

**说明 A.17.** 代入乘积中的公式
$$
(X\circ Y)(n)
=
\coprod_{k\ge0}X(k)\times_{\Sigma_k}
\left(\coprod_{n_1+\cdots+n_k=n}
Y(n_1)\times\cdots\times Y(n_k)\times_{\Sigma_{n_1}\times\cdots\times\Sigma_{n_k}}\Sigma_n
\right)
$$
可理解为有限集分块群胚上的 coend。有限集口径优先，是为了避免在该公式中反复处理左右作用和商关系。

## A.6 本附录小结

全书的安全口径是：用有限集和双射群胚定义对称序列；用骨架 $[n]$ 和 $\Sigma_n$ 只做计算；遇到右作用公式时用 $x\cdot\sigma=X(\sigma^{-1})(x)$ 转换；在一般底环上谨慎处理 coinvariants。这个约定贯穿自由 operad、Schur functor、Koszul 对偶、模型结构和 rectification。
