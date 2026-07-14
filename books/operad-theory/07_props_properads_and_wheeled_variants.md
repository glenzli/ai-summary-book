# 第七章：PROP、properad 与 wheeled 变体

Operad 的每个顶点只有一条输出边，因此能直接编码乘法，却不能把余乘法 $\Delta:X\to X\otimes X$ 当作同类生成运算。双代数兼容式更同时使用纵向复合、横向并排和输入置换；若再取 trace，还要把一条输出接回输入形成环。有限有向图由此取代有根树成为正确的记账对象。PROP 允许全部多输入多输出图和并排张量，properad 把基本复合限制为连通图，wheeled 变体再允许内部环路。以下构造建立在对称幺半范畴、colored operad、张量积和有限有向图的通常语言上，并把需要刚性或有限性的 contraction 单独标出。

## 7.1 PROP 的范畴定义

**定义 7.1.** 一个集合值 PROP 是一个严格对称幺半范畴 $\mathsf P$，满足：

- 对象集为自然数 $\mathbb N=\{0,1,2,\ldots\}$；
- 张量积在对象上为加法：
  $$
  m\otimes n=m+n;
  $$
- 幺元对象为 $0$。

本书采用约定
$$
\mathsf P(m,n)=\operatorname{Hom}_{\mathsf P}(n,m),
$$
其元素称为从 $n$ 个输入到 $m$ 个输出的运算。

**展开 7.2.** 一个 PROP 等价于以下数据：

- 集合 $\mathsf P(m,n)$，其中 $m,n\ge0$；
- 垂直复合
  $$
  \circ:\mathsf P(\ell,m)\times\mathsf P(m,n)\to\mathsf P(\ell,n);
  $$
- 水平张量
  $$
  \otimes:\mathsf P(m,n)\times\mathsf P(m',n')\to
  \mathsf P(m+m',n+n');
  $$
- 单位 $\operatorname{id}_n\in\mathsf P(n,n)$；
- 对称群元素 $\sigma\in\Sigma_n$ 给出的置换态射 $\sigma\in\mathsf P(n,n)$；

并满足范畴公理、严格幺半公理、对称幺半相干性和 interchange law：
$$
(f_1\otimes f_2)\circ(g_1\otimes g_2)
=
(f_1\circ g_1)\otimes(f_2\circ g_2),
$$
只要两边的输入输出数匹配。

**命题 7.3.** 定义 7.1 与展开 7.2 等价。

**证明.** 从严格对称幺半范畴出发，取 hom 集、范畴复合、幺半张量和对称结构即可得到展开数据。Interchange law 是幺半范畴中张量积为双函子的公理。

反过来，给定展开数据，以自然数为对象，以 $\mathsf P(m,n)$ 为从 $n$ 到 $m$ 的态射集合。垂直复合给出范畴复合，水平张量给出严格幺半结构，置换态射给出对称结构。所列公理正是严格对称幺半范畴公理。$\square$

## 7.2 $\mathbb S$-双模与 endomorphism PROP

**定义 7.4.** 一个 $\mathbb S$-双模（symmetric bimodule）是集合族
$$
M(m,n),\qquad m,n\ge0,
$$
其中 $\Sigma_m$ 从左作用，$\Sigma_n$ 从右作用，并且两种作用交换：
$$
\sigma\cdot(x\cdot\tau)=(\sigma\cdot x)\cdot\tau.
$$
一个 PROP 的底层 $\mathbb S$-双模由输出和输入置换作用给出。

**定义 7.5.** 设 $X$ 是集合。其 endomorphism PROP 定义为
$$
\operatorname{End}_X(m,n)=\mathbf{Set}_{\mathcal U}(X^n,X^m).
$$
垂直复合为函数复合，水平张量为笛卡尔积函数：
$$
(f\otimes g)(x_1,\ldots,x_{n+n'})
=
\big(f(x_1,\ldots,x_n),g(x_{n+1},\ldots,x_{n+n'})\big).
$$
输入和输出置换分别由预复合和后复合给出。

**命题 7.6.** $\operatorname{End}_X$ 是 PROP。

**证明.** 函数复合给出范畴公理，笛卡尔积给出严格幺半结构，有限积坐标置换给出对称结构。Interchange law 对任意输入元逐坐标计算即为函数复合与笛卡尔积配对的相容性。$\square$

**定义 7.7.** 设 $\mathsf P$ 是 PROP。一个集合值 $\mathsf P$-代数是 PROP morphism
$$
\mathsf P\to\operatorname{End}_X.
$$
若工作在 $R$-模范畴中，则使用 $R$-线性 endomorphism PROP
$$
\operatorname{End}_V(m,n)=\operatorname{Hom}_R(V^{\otimes n},V^{\otimes m}).
$$

## 7.3 从 operad 到 PROP

**定义 7.8.** 设 $\mathcal O$ 是单色 operad。由 $\mathcal O$ 生成的 PROP，记为 $\operatorname{Prop}(\mathcal O)$，是满足如下泛性质的 PROP：
对任意 PROP $\mathsf P$，给出 PROP morphism
$$
\operatorname{Prop}(\mathcal O)\to\mathsf P
$$
等价于给出 operad morphism
$$
\mathcal O\to U_1(\mathsf P),
$$
其中 $U_1(\mathsf P)(n)=\mathsf P(1,n)$ 是 $\mathsf P$ 的一输出部分。

**命题 7.9.** 若 $\mathcal O$ 是 operad，则 $\operatorname{Prop}(\mathcal O)$ 存在。

**证明.** 可以用生成元与关系构造。取 $\mathcal O$ 的每个元素 $o\in\mathcal O(n)$ 作为一个生成的 $(1,n)$-运算，并加入 $\mathcal O$ 中的单位、代入和对称群关系。再在 PROP 语境中闭合于水平张量、垂直复合和输入输出置换。所得商 PROP 满足定义 7.8 的泛性质，因为任何到 $\mathsf P$ 的 PROP morphism 必须把生成的 $(1,n)$-运算送到 $\mathsf P(1,n)$ 中满足同样 operad 关系的元素，反过来这些元素由 PROP 的复合和张量唯一延拓。$\square$

**例 7.10.** $\operatorname{Prop}(\operatorname{Ass})$ 的代数仍然是幺半群或结合代数；PROP 只是允许把多个乘法表达式并排输出。例如 $X^4\to X^2$ 的运算
$$
(x_1,x_2,x_3,x_4)\mapsto(x_1x_2,x_3x_4)
$$
来自两个二元乘法的水平张量。

## 7.4 双代数的 PROP

**定义 7.11.** 双代数 PROP $\mathsf{Bialg}$ 是由以下生成元和关系给出的 PROP：

生成元为
$$
\mu\in\mathsf{Bialg}(1,2),\qquad
\eta\in\mathsf{Bialg}(1,0),
$$
$$
\Delta\in\mathsf{Bialg}(2,1),\qquad
\epsilon\in\mathsf{Bialg}(0,1).
$$
这里 $\mu$ 是乘法，$\eta$ 是单位，$\Delta$ 是余乘法，$\epsilon$ 是余单位。

关系包括：

1. $(\mu,\eta)$ 满足结合和单位关系。
2. $(\Delta,\epsilon)$ 满足余结合和余单位关系。
3. 兼容关系
   $$
   \Delta\circ\mu
   =
   (\mu\otimes\mu)\circ
   (\operatorname{id}\otimes\tau\otimes\operatorname{id})
   \circ
   (\Delta\otimes\Delta),
   $$
   其中 $\tau\in\mathsf{Bialg}(2,2)$ 交换中间两个张量因子。
4. 单位和余单位兼容：
   $$
   \Delta\circ\eta=\eta\otimes\eta,\qquad
   \epsilon\circ\mu=\epsilon\otimes\epsilon,\qquad
   \epsilon\circ\eta=\operatorname{id}_0.
   $$

**命题 7.12.** $R$-线性 $\mathsf{Bialg}$-代数等价于 $R$ 上含单位、含余单位的双代数。

**证明.** 一个 PROP morphism
$$
\mathsf{Bialg}\to\operatorname{End}_V
$$
把 $\mu,\eta,\Delta,\epsilon$ 分别送到
$$
V\otimes V\to V,\quad R\to V,\quad V\to V\otimes V,\quad V\to R.
$$
前两类关系给出含单位结合代数结构，余结合和余单位关系给出余代数结构。兼容关系在 $\operatorname{End}_V(2,2)$ 中正是
$$
\Delta(xy)=\sum x_{(1)}y_{(1)}\otimes x_{(2)}y_{(2)}
$$
的无 Sweedler 记号版本；单位和余单位兼容关系给出通常双代数公理。反向由任意双代数的结构映射通过生成元关系的泛性质得到 PROP morphism。$\square$

## 7.5 Properad

**定义 7.13.** 一个 directed $(m,n)$-graph 是有限有向图，带有 $n$ 条输入外腿和 $m$ 条输出外腿；每个内部顶点 $v$ 有有限个输入半边和输出半边。若图连通且无有向环，则称为连通无环 directed graph。

**定义 7.14.** 一个 properad $\mathcal P$ 由 $\mathbb S$-双模 $\mathcal P(m,n)$、单位 $\mathbf 1\in\mathcal P(1,1)$ 和如下图复合组成：对任意连通无环 directed $(m,n)$-graph $G$，若每个顶点 $v$ 装饰为
$$
p_v\in\mathcal P(\operatorname{out}(v),\operatorname{in}(v)),
$$
则给出元素
$$
\mu_G((p_v)_{v\in V(G)})\in\mathcal P(m,n).
$$
这些映射要对输入输出重标号等变，并满足：

- 单顶点图复合为该顶点装饰；
- 插入单位边不改变复合；
- 把图的顶点替换为连通无环子图时，先对子图复合再对外图复合，等于一次性对替换后的总图复合。

**说明 7.15.** Properad 与 PROP 的区别在于：properad 的基本复合只沿连通图进行。PROP 还允许不连通图，因为水平张量可以把两个互不连接的运算并排放置。

**命题 7.16.** 每个 PROP $\mathsf P$ 给出 properad，其图复合由 PROP 的垂直复合、水平张量和置换组合而成。

**证明.** 给定连通无环 directed graph，选择一个拓扑排序，把同一层顶点的装饰水平张量起来，再用置换把输出线接到下一层输入线，最后垂直复合各层。若选择不同拓扑排序，相邻可交换的独立顶点只改变水平张量的括号和置换；这些由 PROP 的对称幺半相干性和 interchange law 保证给出相同元素。图替换相干性同样由垂直复合结合律、水平张量结合律和 interchange law 推出。$\square$

**外部输入定理 7.17.** 每个 properad $\mathcal P$ 有自由生成的 PROP $\operatorname{Prop}(\mathcal P)$，其不连通图由 $\mathcal P$ 的连通图复合和水平张量生成。该构造的完整证明依赖 directed graph groupoids 的商和相干性检查；调用本定理时应引用 Markl-Shnider-Stasheff、Loday-Vallette 或 Fresse。

## 7.6 Wheeled 变体

**定义 7.18.** 一个 wheeled properad 是 properad 的变体，其中图复合允许有向环；等价地，它除了无环图复合外，还允许把一个输出腿接回一个输入腿，形成 contraction 或 trace 型操作
$$
\operatorname{tr}_{i}^{j}:\mathcal P(m,n)\to\mathcal P(m-1,n-1),
$$
其中第 $j$ 个输出与第 $i$ 个输入被相接并消去。这些 contraction 必须与重标号、properad 复合和彼此的迭代相容。

**例 7.19.** 若 $V$ 是有限生成投射 $R$-模，则 $\operatorname{End}_V(m,n)=\operatorname{Hom}_R(V^{\otimes n},V^{\otimes m})$ 具有 wheeled 结构。Contraction 由评价配对
$$
V^\vee\otimes V\to R
$$
或等价的 trace 操作给出。有限生成投射条件保证 trace 不依赖基的选择。

**警告 7.20.** 一般 $R$-模没有自然 trace。因此 wheeled endomorphism properad 不是无条件存在的结构；必须指定双对偶、trace、有限维性或刚性条件。

## 7.7 从有根树到有向图

一输出部分仍能恢复 operad，但多输出使图演算出现三种彼此不同的结构：沿边的垂直复合、分量并排的水平张量，以及对输入输出腿的置换。PROP 同时保留三者，properad 先隔离连通图复合，wheeled 结构则再加入 contraction。双代数的兼容式展示了前两种复合如何交织；有限生成投射模的 endomorphism 例子则说明闭环为什么需要对偶性。下一部分回到一输出线性 operad，研究生成关系具有二次权重时如何产生 Koszul 对偶和同伦分解。

## 练习

**练习 7.1.** 写出 $\operatorname{End}_X$ 中 interchange law 的逐元素证明。

**练习 7.2.** 证明一个 PROP 的一输出部分 $U_1(\mathsf P)(n)=\mathsf P(1,n)$ 带有自然 operad 结构。

**练习 7.3.** 在 $\mathsf{Bialg}$ 中画出兼容关系
$$
\Delta\circ\mu
=
(\mu\otimes\mu)\circ(\operatorname{id}\otimes\tau\otimes\operatorname{id})\circ(\Delta\otimes\Delta)
$$
对应的线路图。

**练习 7.4.** 给出一个 properad 中可复合的连通无环图，其既不是单纯 operad 树，也不是 PROP 中两个运算的水平张量。

**练习 7.5.** 解释为什么 wheeled contraction 在无限维向量空间上通常没有基无关定义。
