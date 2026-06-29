# 附录 V：带边界与分层 factorization homology 样例

本附录补充第二十章的定义 20.3--外部输入定理 20.23 和附录 N 的定义 N.3--外部输入定理 N.30。目标是说明为什么带边界、带角和分层空间不能用无边界 manifold 的公式直接处理。完整 stratified factorization homology 和 sectorial/Fukaya descent 仍作为外部输入。

## V.1 半空间与边界条件

令
$$
\mathbb H^n=\{(x_1,\ldots,x_n)\in\mathbb R^n\mid x_n\ge0\}
$$
为标准半空间。带边界 factorization theory 的局部模型不只包括 $\mathbb R^n$，还包括 $\mathbb H^n$ 及其边界 $\mathbb R^{n-1}$。

**定义 V.1（形式边界系数）.** 一个带边界局部系数系统至少包含：

1. 内部 $E_n$-algebra $A$；
2. 边界对象 $M$；
3. $M$ 上与 $A$ 相容的 module 或 $E_{n-1}$-module 型结构。

具体结构依赖采用 framed、oriented、stratified 或 $\infty$-categorical 模型。

**警告 V.2.** 不能把
$$
\int_{\mathbb H^n}A
$$
无条件写成 $A$。半空间带有边界 strata；若未指定边界条件，积分对象未定义。

## V.2 一维区间

令 $A$ 是 $E_1$-algebra。若区间 $[0,1]$ 的左端点标记为右 $A$-module $M$，右端点标记为左 $A$-module $N$，则期望公式为
$$
\int_{[0,1]}(M,A,N)\simeq M\otimes_A^{\mathbf L}N.
$$

**外部输入定理 V.3（区间计算；AF-4 边界来源）.** 在带边界或分层 factorization homology 的标准模型中，上述公式成立。

**证明边界.** 区间内部由 $A$ 标记，两个端点由 modules 标记。把区间分解为左端 collar、内部小区间和右端 collar，excision 给出 two-sided bar construction
$$
B(M,A,N).
$$
Ayala--Francis, arXiv:1206.5522v6, Theorem 3.26 给出带边界版本的基本定位；严格区间公式还需要把所选边界 disk category 与 module 标记模型逐项对齐。$\square$

**说明 V.4.** 若取 $M=N=A$，则
$$
A\otimes_A^{\mathbf L}A\simeq A.
$$
这解释了为什么某些带边界区间计算看起来像 $A$，但原因是选择了特定边界条件，而不是无边界 disk 归一化。

## V.3 圆周作为 trace

把圆周看作把区间两端 glue：
$$
S^1\simeq [0,1]/(0\sim1).
$$
对 $E_1$-algebra $A$，闭合端点使左、右 $A$-作用配对，得到 trace：
$$
\int_{S^1}A\simeq A\otimes^{\mathbf L}_{A\otimes A^{op}}A\simeq HH_\*(A).
$$

**说明 V.5.** 这不是普通 homology with coefficients。它是 algebra object 的 trace 或 factorization homology 的一维闭合计算。若 $A$ 非交换，$A^{op}$ 和双模结构不可省略。

## V.4 分层区间

考虑区间
$$
[-1,1]
$$
并在 $0$ 处分层：
$$
[-1,0),\quad \{0\},\quad (0,1].
$$
左右一维 strata 可标记为 $E_1$-algebras $A,B$，点 strata 可标记为 $(A,B)$-bimodule $M$。

**外部输入定理 V.6（点缺陷计算）.** 在一维分层 factorization homology 中，
$$
\int_{[-1,1]}(A,M,B)
$$
由 $M$ 及两侧 collar action 决定；若再给左右端点 modules，则整体计算为相应 iterated derived tensor product。

**说明 V.7.** 该模型是“defect”或“domain wall”的最小例子。点 strata 不是普通点系数，而是连接左右 $E_1$-algebras 的双模数据。

## V.5 高维 hypersurface defect

设 $M$ 是 $n$-manifold，被一条 cooriented hypersurface $N$ 分成 $M_-$ 与 $M_+$。可用如下标记：

1. $M_-$ 上的 $E_n$-algebra $A_-$；
2. $M_+$ 上的 $E_n$-algebra $A_+$；
3. $N$ 上的 defect algebra/module $D$，它应是连接 $A_-$ 与 $A_+$ 的 $E_{n-1}$-型双模对象。

**外部输入定理 V.8.** 在适当分层 factorization homology 模型中，沿 hypersurface 的 gluing 由 $E_{n-1}$-algebra/module 的 relative tensor product 控制。

**证明边界.** 该结论需要 stratified disk category、constructible factorization algebra 和 conically smooth stratified spaces 的技术。不能由无边界 excision 单独推出。$\square$

## V.6 Corners 与多模块结构

带角流形的局部模型包括
$$
[0,\infty)^k\times\mathbb R^{n-k}.
$$
codimension $k$ 的 corner 通常要求 iterated module 或 higher bimodule 数据。

**警告 V.9.** 在有 corners 的 gluing 中，边界的边界也有结构。只指定每个 codimension $1$ face 的 module 数据不足以保证 corner 处相容。

**例 V.10.** 矩形 $[0,1]^2$ 若四条边有边界条件，则四个角需要边界条件之间的交互数据。否则 factorization homology 的局部-to-global gluing 没有完整输入。

## V.7 与 Fukaya skeleta 的关系

Wrapped Fukaya categories 的 cosheaf 或 sheaf-theoretic gluing 常在 Liouville sectors、skeleta 或 stopped Weinstein manifolds 上表达。形式上类似：

1. 对局部 sector 分配 category；
2. 对包含或 gluing 分配 functor；
3. 通过 descent 重构全局 category。

**外部输入定理 V.11.** 在指定的 Liouville sector 或 wrapped Fukaya 设置中，sectorial descent 或 skeletal descent 可把全局 Fukaya category 表示为局部数据的 colimit 或 limit。

**证明边界.** 这属于辛几何与 Floer theory 的外部定理，需要 compactness、transversality、wrapping、stops、orientation 和 functoriality。Operad theory 只组织 gluing 形状。$\square$

## V.8 使用检查表

使用带边界或分层 factorization homology 前，必须说明：

1. 空间是 manifold with boundary、manifold with corners，还是 conically smooth stratified space；
2. 每个 stratum 的维度和切结构；
3. 内部 strata 的 $E_n$-algebra 标记；
4. 边界或 defect strata 的 module/bimodule/higher module 标记；
5. gluing collar 或 conical neighborhood；
6. relative tensor product 是否 derived；
7. 是否引用 stratified factorization homology 或 sectorial descent 定理。

## V.9 小结

无边界公式
$$
\int_{\mathbb R^n}A\simeq A
$$
不能直接迁移到半空间、区间、带角流形或分层空间。边界和缺陷需要 module 数据；角需要更高相容数据；Fukaya 型 gluing 还需要独立分析定理。外部输入定理 N.15 的 excision 是入口，本附录记录它在带边界和分层情形中的使用边界。
