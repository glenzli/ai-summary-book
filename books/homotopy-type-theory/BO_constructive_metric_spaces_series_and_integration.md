# 附录 BO：构造性度量空间、级数与积分接口

附录 BA 给出连续性、紧致性和典型定理接口；本附录补上构造性分析中更基础的工作层：度量空间、Cauchy 完备化、级数、一致收敛、Banach 不动点和 Riemann 积分。它服务于实数章节从“有序域接口”走向分析教材。

## BO.1 Premetric space

**定义 BO.1（有理值预度量）。** 预度量空间由集合 $X$ 和关系
$$
d(x,y)<q\qquad(q:\mathbb Q_{>0})
$$
组成，满足：

1.  反身性：$d(x,x)<q$；
2.  对称性；
3.  三角性：若 $d(x,y)<q$ 且 $d(y,z)<r$，则 $d(x,z)<q+r$；
4.  分离性：若对所有 $q>0$ 有 $d(x,y)<q$，则 $x=y$。

**定义 BO.2（Cauchy sequence）。** 序列 $a:\mathbb N\to X$ 是 Cauchy，若
$$
\prod_{\varepsilon:\mathbb Q_{>0}}
\left\|\sum_{N:\mathbb N}\prod_{m,n\ge N}d(a_m,a_n)<\varepsilon\right\|.
$$

**定义 BO.3（complete）。** $X$ 完备，若每个 Cauchy sequence merely has a limit：
$$
\prod_a \mathsf{isCauchy}(a)\to
\left\|\sum_{x:X}\mathsf{lim}(a,x)\right\|.
$$

## BO.2 Uniform convergence

**定义 BO.4（一致收敛）。** 函数列 $f_n:X\to Y$ 一致收敛到 $f:X\to Y$，若
$$
\prod_{\varepsilon>0}
\left\|\sum_N\prod_{n\ge N}\prod_{x:X}
d_Y(f_n(x),f(x))<\varepsilon\right\|.
$$

**定理 BO.5（一致极限保持连续，证明核）。** 若每个 $f_n$ 连续，且 $f_n$ 一致收敛到 $f$，则 $f$ 连续。

**证明.** 给定 $x$ 和 $\varepsilon>0$。由一致收敛取 $N$ 使 $d(f_N(y),f(y))<\varepsilon/3$ 对所有 $y$ 成立，并有 $d(f_N(x),f(x))<\varepsilon/3$。由 $f_N$ 在 $x$ 连续，取 $\delta$ 使 $d(x,y)<\delta$ 推出 $d(f_N(x),f_N(y))<\varepsilon/3$。三角不等式给出
$$
d(f(x),f(y))<\varepsilon.
$$
若选择 $N$ 位于命题截断中，结论连续性为命题时可用截断消去。$\square$

## BO.3 Series

**定义 BO.6（级数收敛）。** 在 normed Abelian group $V$ 中，级数 $\sum a_n$ 收敛到 $s$，若部分和
$$
S_N=\sum_{n<N}a_n
$$
收敛到 $s$。

**定理 BO.7（Cauchy criterion for series，证明核）。** 若 $V$ 完备，则 $\sum a_n$ 收敛当且仅当 tails Cauchy：
$$
\prod_{\varepsilon>0}
\left\|\sum_N\prod_{m\ge n\ge N}
\left\|\sum_{k=n}^{m}a_k\right\|<\varepsilon\right\|.
$$

**证明.** 级数收敛等价于部分和序列 Cauchy，因为完备性把 Cauchy 序列提升为极限，分离性保证极限唯一。部分和 Cauchy 条件展开后正是 tail estimate。$\square$

**定理 BO.8（Weierstrass M-test，证明核）。** 若 $\|f_n(x)\|\le M_n$ 对所有 $x$ 成立，且实数级数 $\sum M_n$ 收敛，则 $\sum f_n$ 一致收敛。

**证明.** 对 tails 使用三角不等式：
$$
\left\|\sum_{k=n}^{m}f_k(x)\right\|
\le
\sum_{k=n}^{m}\|f_k(x)\|
\le
\sum_{k=n}^{m}M_k.
$$
由 $\sum M_n$ 的 Cauchy tail 对任意 $\varepsilon$ 给出统一 $N$。$\square$

## BO.4 Banach fixed point

**定义 BO.9（contraction）。** 自映射 $T:X\to X$ 是 contraction，若存在有理数 $0<c<1$，使
$$
d(Tx,Ty)\le c\,d(x,y)
$$
以预度量的有理近似形式成立。

**定理 BO.10（Banach fixed point，证明架构）。** 若 $X$ 完备且 $T$ 是 contraction，则 $T$ 有唯一不动点。

**证明架构.** 取任意 $x_0:X$，定义 $x_{n+1}=T(x_n)$。由 contraction 得
$$
d(x_n,x_{n+k})\le c^n(1+c+\cdots+c^{k-1})d(x_0,x_1),
$$
右侧由几何级数 tail 控制，故 $(x_n)$ Cauchy。完备性给出极限 $x$。连续性由 contraction 推出，故 $T(x)=x$。若 $x,y$ 都是不动点，则
$$
d(x,y)=d(Tx,Ty)\le c\,d(x,y)
$$
推出 $d(x,y)=0$，由分离性得 $x=y$。

## BO.5 Riemann integral interface

**定义 BO.11（partition）。** 闭区间 $[a,b]$ 的划分是有限序列
$$
a=x_0\le x_1\le\cdots\le x_n=b.
$$

**定义 BO.12（mesh）。** 划分 $P$ 的 mesh 是最大长度
$$
\max_i(x_{i+1}-x_i).
$$
构造性定义可用有限集最大值或上界加 locatedness。

**定义 BO.13（Riemann integrable）。** 函数 $f:[a,b]\to\mathbb R_C$ Riemann integrable，若存在 $I:\mathbb R_C$，使任意 tagged partition $P$ 在 mesh 足够小时，其 Riemann sum
$$
S(f,P)
$$
满足 $|S(f,P)-I|<\varepsilon$。

**定理 BO.14（uniform continuous implies Riemann integrable，证明架构）。** 若 $f:[a,b]\to\mathbb R_C$ 一致连续，则 $f$ Riemann integrable。

**证明架构.** 用一致连续给出振幅控制：当子区间长度小于 $\delta$，任意两点函数值相差小于 $\varepsilon/(b-a+1)$。任意两组足够细 Riemann sums 之间距离小于 $\varepsilon$，故 Riemann sums 形成 Cauchy net。由 $\mathbb R_C$ 完备性得到积分值。构造性细节需要 finite partition 的最大值、区间长度有界性和有理误差预算。

## BO.6 Fundamental theorem boundary

**事实 BO.15（微积分基本定理边界）。** 在构造性分析中，微积分基本定理需要精确定义 differentiability、uniform differentiability 或 located derivative，并区分点态导数和强导数。它不应从 classical 证明直接移植。

**接口 BO.16.** 若 $F$ 有强导数 $f$ 且 $f$ 连续，则
$$
\int_a^b f=F(b)-F(a)
$$
可通过 Riemann sums 的 telescoping estimate 证明。完整证明需要 BO.14 的积分存在性和导数误差的统一控制。

## BO.7 本附录关闭的缺口

本附录把构造性度量空间、Cauchy 完备、统一收敛、级数判别、Weierstrass M-test、Banach fixed point 和 Riemann 积分接口加入教材。剩余义务是全部有理误差预算、partition 有限组合学、积分线性性、变量替换和微积分基本定理的完整构造性证明。
