# 第四十七章：层化同伦、Exit-path 范畴与可构造 Sheaves

局部常值 sheaf 由基本 $\infty$-groupoid 的表示分类；层化空间上，sheaf 只要求在每个 stratum 内局部常值，并允许跨层发生单向 specialization。Exit-path $\infty$-category 把可从较奇异层“退出”的层化路径及其高阶同伦编码起来，constructible sheaves 因而可描述为其到目标范畴的函子。这个对应把 strata 上的局部系统和跨层粘合数据统一为一个范畴对象。

本章使用 sheaves、$\infty$-categories、singular complexes 和 stratified spaces。Exit-path 分类需要 conical、locally contractible 等温和性条件；方差取决于 sheaf 还是 cosheaf，正文会固定方向而不把二者混同。

## 47.1 层化空间

**定义 47.1.** 一个层化空间是拓扑空间 $X$ 配连续映射

$$
\pi:X\to P
$$

到偏序集 $P$ 的 Alexandrov 拓扑，使每个纤维 $X_p=\pi^{-1}(p)$ 称为 stratum。

**定义 47.2.** 层化空间称为 conically stratified，若每点邻域同胚于

$$
\mathbb R^k\times C(L)
$$

并且层化由 $L$ 的层化和 cone point 诱导。

**例子 47.3.** 带有限 Whitney stratification 的复代数簇在合适拓扑口径下给出 conically stratified spaces 的主要来源。

## 47.2 Exit paths

**定义 47.4.** 层化空间 $X\to P$ 中的 exit path 是路径 $\gamma:[0,1]\to X$，使其所在 stratum 随时间只能沿偏序向“较大”层移动：若 $t\le t'$，则 $\pi(\gamma(t))\le\pi(\gamma(t'))$。

**定义 47.5.** Exit-path $\infty$-category $\operatorname{Exit}(X)$ 的对象为 $X$ 的点。其 $n$-单纯形不是任意连续映射 $\Delta^n\to X$：先给 $\Delta^n$ 赋由顶点次序诱导的标准层化，并选非降链

$$
p_0\le p_1\le\cdots\le p_n
$$

作为到 $P$ 的层标号；一个 exit-simplex 是与这些层化相容的 stratified map

$$
\Delta^n\longrightarrow X.
$$

面与退化由标准单纯形的 stratified maps 诱导。

**外部输入定理 47.6.** 若 $X$ conically stratified、paracompact，且 strata 与 links 满足通常局部可缩性条件，则 exit simplicial set 是 quasi-category。

**命题 47.7.** 若 $X$ 只有一个 stratum，则 $\operatorname{Exit}(X)$ 等价于 $X$ 的 fundamental $\infty$-groupoid。

**证明.** 单层情形中偏序条件自动满足，任何 singular simplex 都是 exit-simplex。因此 exit simplicial set 与 $X$ 的 singular complex 相同。Singular complex 是 Kan complex，表示 $X$ 的 fundamental $\infty$-groupoid。$\square$

## 47.3 两层空间的方向性

**命题 47.8.** 设 $X$ 有闭层 $Z$ 和开层 $U$，偏序取 $Z<U$。Exit path 可从 $Z$ 进入 $U$，但不可从 $U$ 进入 $Z$。

**证明.** 若路径从 $Z$ 到 $U$，则层标号从 $Z$ 增大到 $U$，满足 $Z\le U$。若路径从 $U$ 到 $Z$，则存在 $t<t'$ 使标号从 $U$ 变为 $Z$，这要求 $U\le Z$，与偏序 $Z<U$ 矛盾。因此后者不是 exit path。$\square$

**命题 47.9.** 把 $\operatorname{Exit}(X)$ 限制到某个单独 stratum $X_p$ 内的 exit simplices，得到的 simplicial set 等于 $X_p$ 的 singular complex，因此表示 $\Pi_\infty(X_p)$。

**证明.** 若 simplex $\Delta^n\to X$ 的像落在 $X_p$，则层标号恒为 $p$，exit 条件自动满足。因此这些 exit simplices 正是 $X_p$ 的 ordinary singular simplices。由 singular complex 对 fundamental $\infty$-groupoid 的表示，所得对象为 $\Pi_\infty(X_p)$。$\square$

## 47.4 可构造 Sheaves

**定义 47.10.** 设 $X$ 层化。空间值 sheaf $F$ 称为 constructible，若对每个 stratum $X_p$，限制 $F|_{X_p}$ 是 locally constant sheaf。

**外部输入定理 47.11（Exit-path 分类）.** 在定理 47.6 的假设下，有自然等价

$$
\operatorname{Shv}_{cbl}(X;\mathcal S)\simeq
\operatorname{Fun}(\operatorname{Exit}(X),\mathcal S).
$$

更一般地，若 presentable $\infty$-category $C$ 的极限满足该 sheaf 理论所需的可交换性，则 $C$-值 constructible sheaves 由 $\operatorname{Fun}(\operatorname{Exit}(X),C)$ 分类。本章采用 sheaf 的这一协变 exit 约定；constructible cosheaves 对应相反的方差。

**命题 47.12.** 单层情形中，该定理退化为 locally constant sheaves 与 fundamental $\infty$-groupoid 表示的等价。

**证明.** 由命题 47.7，$\operatorname{Exit}(X)\simeq\Pi_\infty(X)$。Constructible 条件在单层情形正是 locally constant。因此定理 47.11 给出

$$
\operatorname{Loc}(X;\mathcal S)\simeq\operatorname{Fun}(\Pi_\infty(X),\mathcal S),
$$

这就是局部常值 sheaves 的单值化分类。$\square$

## 47.5 Recollement 与 Exit 分类

**命题 47.13.** 对开闭分解 $j:U\hookrightarrow X$、$i:Z\hookrightarrow X$，constructible sheaf 由 $U$ 上对象、$Z$ 上对象和从 $Z$ 的 exit-link 到 $U$ 的相容传输数据粘合。

**证明.** Exit-path 分类把 constructible sheaf 化为函子

$$
F:\operatorname{Exit}(X)\to C.
$$

限制到 full subcategories $\operatorname{Exit}(U)$ 与 $\operatorname{Exit}(Z)$ 给出两部分数据。跨越开闭分解的非恒等信息来自从 $Z$ 中点出发并进入 $U$ 的 exit morphisms；没有从 $U$ 回到 $Z$ 的 morphisms。函子性要求这些跨层 morphisms 给出从闭层数据到开层附近数据的相容映射。$\square$

## 47.6 Perverse Sheaves 的范畴论影子

**定义 47.14.** 层化空间上的 perverse sheaf 可视为 constructible derived sheaf，在支撑和余支撑条件下落入 perverse t-结构的 heart。

**命题 47.15.** Exit-path 分类说明 perverse sheaf 的底层可构造信息由有向高阶范畴控制。

**证明.** Perverse sheaf 首先是 constructible derived sheaf。因此遗忘 t-结构条件后，它的可构造局部系统和跨层 monodromy 数据由 $\operatorname{Exit}(X)$ 上的函子编码。Perverse 条件再对这些数据施加同调维数和 exactness 限制。故 exit-path 范畴给出底层有向高阶组合骨架。$\square$

## 47.7 层化因子化同调

**外部输入定理 47.16.** 对 conically stratified manifolds，有层化版本的 $\operatorname{Disk}$-范畴和因子化同调。其系数由适合各层和链接的代数数据给出，并满足层化 excision。

**命题 47.17.** 单层流形上的层化因子化同调恢复普通因子化同调。

**证明.** 单层情形中允许的局部模型只有普通 $\mathbb R^n$，层化 $\operatorname{Disk}$-范畴退化为 $\operatorname{Disk}_n$，层化 open embeddings 退化为普通 open embeddings。于是左 Kan 延拓定义与第四十二章的定义相同。$\square$

## 47.8 由 exit paths 分类可构造层

Exit-path $\infty$-category 把层化空间中的方向性和高阶同伦合为一个范畴对象。Constructible sheaves 被分类为 exit-path 范畴上的函子；perverse sheaves 在此基础上加入 t-结构条件；层化因子化同调则把同一局部到整体思想推广到带奇异层的流形。

## 练习

**练习 47.1.** 定义层化空间。

**练习 47.2.** 定义 conically stratified space。

**练习 47.3.** 定义 exit path。

**练习 47.4.** 定义 $\operatorname{Exit}(X)$。

**练习 47.5.** 证明单层情形中 $\operatorname{Exit}(X)$ 是 fundamental $\infty$-groupoid。

**练习 47.6.** 说明两层空间中 exit path 的方向限制。

**练习 47.7.** 定义 constructible sheaf。

**练习 47.8.** 陈述 exit-path 分类定理。

**练习 47.9.** 说明单层情形如何恢复局部常值 sheaf 的分类。

**练习 47.10.** 用 exit-path 描述开闭分解中的粘合数据。

**练习 47.11.** 说明 perverse sheaf 与 constructible sheaf 的关系。

**练习 47.12.** 解释 exit-path 范畴为何是 perverse sheaf 的底层组合骨架。

**练习 47.13.** 陈述层化因子化同调。

**练习 47.14.** 证明单层情形恢复普通因子化同调。

**练习 47.15.** 证明单个 stratum 上的 exit simplices 正是该 stratum 的 singular simplices。
