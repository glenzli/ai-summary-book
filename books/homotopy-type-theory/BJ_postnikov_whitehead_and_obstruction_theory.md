# 附录 BJ：Postnikov Towers、Whitehead 定理与障碍理论接口

本附录补入合成同伦论中连接截断、同伦群和分类问题的核心骨架：Postnikov tower、Whitehead theorem、Eilenberg-Mac Lane fiber 和 obstruction theory。它们是 HoTT 中从“单个空间计算”走向“同伦类型分类”的必要部分。

## BJ.1 截断塔

**定义 BJ.1（Postnikov 截断塔）。** 对类型 $X$，其 Postnikov tower 是由截断泛性质给出的反向系统
$$
\cdots\to\|X\|_{n+1}\to\|X\|_n\to\cdots\to\|X\|_0\to\|X\|_{-1}.
$$
映射 $\|X\|_{n+1}\to\|X\|_n$ 由 $\|X\|_n$ 是 $n$-type 和 $|-|_n:X\to\|X\|_n$ 的泛性质诱导。

**命题 BJ.2（塔映射相容性，书内证明核）。** 对 $m\le n$，存在规范映射
$$
\tau_{n,m}:\|X\|_n\to\|X\|_m
$$
且
$$
\tau_{m,k}\circ\tau_{n,m}=\tau_{n,k}
$$
在函数外延性下成立。

**证明.** $\tau_{n,m}$ 由 $\|X\|_m$ 的 $m$-type 性和 $X\to\|X\|_m$ 的泛性质从 $X\to\|X\|_n$ 因子化得到。两个复合与 $\tau_{n,k}$ 预合成 $X\to\|X\|_n$ 后都等于 $X\to\|X\|_k$；由截断泛性质的唯一性和函数外延性得到相等。$\square$

## BJ.2 同伦群作为截断塔 fiber

**定义 BJ.3（homotopy fiber of Postnikov stage）。** 对 pointed connected type $(X,x_0)$，定义第 $n$ 层 fiber
$$
F_n(X)\coloneqq
\mathsf{fib}_{\|X\|_n\to\|X\|_{n-1}}(|x_0|_{n-1}).
$$

**命题 BJ.4（fiber 的连通与截断范围，证明架构）。** 若 $X$ 连通且 $n\ge1$，则 $F_n(X)$ 是 $n$-type，且在适当连通假设下其唯一非平凡同伦群为 $\pi_n(X)$。

**证明架构.** 由 fiber of map into $(n-1)$-type 的路径空间计算和截断塔相容性证明 $F_n$ 的截断范围。通过 long exact sequence of homotopy groups applied to
$$
F_n(X)\to\|X\|_n\to\|X\|_{n-1}
$$
识别同伦群。长正合列证明核见附录 AP。

**输入 BJ.5（Eilenberg-Mac Lane fiber）。** 在有足够 EM 型的口径中，若 $\pi_n(X)=G$，则第 $n$ 个 Postnikov fiber 等价于
$$
K(G,n).
$$
该结果依赖 EM 型存在性、连通性计算和截断塔 fiber 的唯一性。

## BJ.3 Whitehead 定理

**定义 BJ.6（weak equivalence on homotopy groups）。** 对 pointed connected types 的 pointed map
$$
f:X\to_\ast Y,
$$
称 $f$ 是 homotopy-group weak equivalence，若对每个 $n\ge0$，
$$
\pi_n(f):\pi_n(X)\to\pi_n(Y)
$$
是群同构，其中 $n=0$ 解释为 connected component 层。

**定理 BJ.7（Whitehead theorem，外部输入 / 证明架构）。** 对足够好的 HoTT 类型类，例如 CW-like HIT generated types 或具有收敛 Postnikov tower 的类型，若 $f:X\to Y$ 在所有同伦群上诱导同构，则 $f$ 是等价。

**证明架构.** 对每个 $n$ 证明
$$
\|f\|_n:\|X\|_n\to\|Y\|_n
$$
是等价。归纳步使用 Postnikov fiber sequence 和五引理式的同伦群长正合列比较。最后用 Postnikov tower 收敛或 Whitehead 原理把所有截断层等价提升为 $X\simeq Y$。

**边界 BJ.8.** 普通 HoTT 中“所有同伦群同构推出等价”需要类型类或收敛假设。对任意类型无条件声明该定理会把未证明的 hypercompleteness/Whitehead principle 偷渡进系统。

## BJ.4 Obstruction classes

**定义 BJ.9（lifting problem）。** 给定 Postnikov stage map
$$
p:E_{n+1}\to E_n
$$
和映射 $g:A\to E_n$，lifting problem 是 fiber
$$
\sum_{\tilde g:A\to E_{n+1}}(p\circ\tilde g=g).
$$

**定义 BJ.10（obstruction class，接口）。** 若 fiber of $p$ 等价于 $K(G,n+1)$，则 lift 的主要障碍属于
$$
H^{n+2}(A;G)
$$
或带 local coefficient 的相应上同调群。该类通常由 $k$-invariant
$$
k:E_n\to K(G,n+2)
$$
拉回得到。

**命题 BJ.11（障碍消失给出 merely lift，证明架构）。** 若 obstruction class $g^\ast k$ 为零，则 lifting type 的命题截断
$$
\left\|\sum_{\tilde g:A\to E_{n+1}}(p\circ\tilde g=g)\right\|
$$
成立。

**证明架构.** $p$ 由 $k$ 的 homotopy fiber 给出。给出 $g^\ast k=0$ 等价于给出 $g$ 到该 homotopy fiber 的 lift。若 $k$ 只在截断或局部系数口径下给出，则结论相应位于命题截断中。

## BJ.5 Postnikov classification

**定义 BJ.12（Postnikov data）。** 一个 connected type 的 Postnikov data 包含：

1.  同伦群族 $\pi_n$；
2.  每层的 group action 或 local coefficient data；
3.  $k$-invariants
    $$
    k_n:\|X\|_n\to K(\pi_{n+1},n+2);
    $$
4.  相干条件，使下一层是 $k_n$ 的 fiber。

**事实 BJ.13（分类接口）。** 在有 EM 型、局部系数和 tower 收敛的口径中，connected types 可由 Postnikov data 重建到相应等价。

**外部边界.** 本书只把该事实作为 obstruction theory 接口。其定理身份依赖 local coefficient、twisted EM types、tower limit 和 Whitehead principle 的精确外部版本。

## BJ.6 塔收敛与分类边界

有限 Postnikov stage 与由全部 stages 重建原类型是不同结论。后者需要 EM 型、局部系数、twisted cohomology、tower convergence 与 Whitehead 原理；本附录没有从基础 HIT 规则构造这些数据，因此只允许在定理逐项列出这些输入后使用分类结论。
