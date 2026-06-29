# 附录 AB：同伦层级向上闭包证明核

本附录补全命题 4.13。全篇使用第四章的递归定义：
$$
\mathsf{isOfHLevel}_0(A)\coloneqq\mathsf{isContr}(A),
$$
$$
\mathsf{isOfHLevel}_{n+1}(A)
\coloneqq
\prod_{x,y:A}\mathsf{isOfHLevel}_n(x=y).
$$

## AB.1 可收缩类型的路径空间可收缩

**引理 AB.1.** 若 $A$ 可收缩，则对任意 $x,y:A$，路径类型 $x=y$ 可收缩。

**证明.** 设 $A$ 的收缩数据为 $(c,H)$，其中
$$
H:\prod_{z:A}(c=z).
$$
先由命题 4.7 得到 $A$ 是命题。由命题 4.11 或附录 F.1，命题是集合，因此每个路径类型 $x=y$ 是命题。另一方面，$x=y$ 有点：
$$
H(x)^{-1}\cdot H(y):x=y.
$$
命题且有点推出可收缩（命题 4.8），故 $x=y$ 可收缩。$\square$

## AB.2 向上闭包

**定理 AB.2（同伦层级向上闭包）.** 对任意 $n:\mathbb N$，
$$
\mathsf{isOfHLevel}_n(A)\to\mathsf{isOfHLevel}_{n+1}(A).
$$

**证明.** 对 $n$ 作自然数归纳。

当 $n\equiv0$ 时，假设
$$
h:\mathsf{isContr}(A).
$$
需证明
$$
\prod_{x,y:A}\mathsf{isContr}(x=y),
$$
这正是引理 AB.1。

归纳步：假设对任意类型 $X$ 有
$$
\mathsf{isOfHLevel}_n(X)\to\mathsf{isOfHLevel}_{n+1}(X).
$$
设
$$
h:\mathsf{isOfHLevel}_{n+1}(A).
$$
按定义，
$$
h:\prod_{x,y:A}\mathsf{isOfHLevel}_n(x=y).
$$
需证明
$$
\mathsf{isOfHLevel}_{n+2}(A),
$$
即
$$
\prod_{x,y:A}\mathsf{isOfHLevel}_{n+1}(x=y).
$$
给定 $x,y:A$，把归纳假设应用到类型 $x=y$ 和证明 $h(x,y)$，得到
$$
\mathsf{isOfHLevel}_{n+1}(x=y).
$$
这正是所需函数值。$\square$

**推论 AB.3（命题是集合的层级形式）.** 若 $A$ 是 h-level $1$，则 $A$ 是 h-level $2$；按第四章术语，命题是集合。

**证明.** 取 AB.2 在 $n=1$ 的实例。它与命题 4.11 的直接路径代数证明一致。$\square$

**依赖说明。** 本附录使用命题 4.11 / 附录 F.1 来证明 AB.1。若希望完全避免该引用，也可直接对收缩路径作路径归纳，证明可收缩类型中的平行路径唯一；两种证明等价。
