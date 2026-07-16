# 独立性、乘积与随机核

两个随机量“互不影响”不是数学定义。概率论里的独立性说的是联合事件概率分解，或者等价地，联合分布分解为乘积分布。随机核则把“给定当前状态后下一步的分布”写成可测对象，使条件转移可以组合。独立性描述并列结构，核描述先后结构；随机过程需要两者都能被精确计算。

## 4.1 独立性

**定义 4.1（事件族与随机变量独立）。** 子 $\sigma$-代数 $\mathcal G_1,\ldots,\mathcal G_m\subseteq\mathcal F$ 称为独立，若对任意 $A_i\in\mathcal G_i$，

$$
\mathbb P\left(\bigcap_{i=1}^m A_i\right)=\prod_{i=1}^m\mathbb P(A_i).
$$

随机变量 $X_i:\Omega\to E_i$ 独立，若 $\sigma(X_1),\ldots,\sigma(X_m)$ 独立。

**定理 4.1（有限族独立性的函数刻画）。** 设 $X_i$ 是取有限集合 $E_i$ 值的随机变量。则 $X_1,\ldots,X_m$ 独立当且仅当对任意函数 $g_i:E_i\to\mathbb R$，

$$
\mathbb E\left[\prod_{i=1}^m g_i(X_i)\right]=\prod_{i=1}^m\mathbb E[g_i(X_i)].
$$

**证明.** 若函数恒等式成立，取 $g_i=\mathbf 1_{B_i}$ 得到事件分解，因此独立。反过来，有限集合上的任意函数可写为 $g_i=\sum_{x_i\in E_i}g_i(x_i)\mathbf 1_{\{x_i\}}$。展开乘积并用独立性：

$$
\mathbb E\prod_i g_i(X_i)
=\sum_{x_1,\ldots,x_m}\prod_i g_i(x_i)\mathbb P(X_1=x_1,\ldots,X_m=x_m)
$$

$$
=\sum_{x_1,\ldots,x_m}\prod_i g_i(x_i)\prod_i\mathbb P(X_i=x_i)
=\prod_i\sum_{x_i}g_i(x_i)\mathbb P(X_i=x_i).
$$

最后一项就是 $\prod_i\mathbb E[g_i(X_i)]$。证毕。

## 4.2 乘积与核

**定义 4.2（Markov 核）。** 从 $(E,\mathcal E)$ 到 $(F,\mathcal H)$ 的 Markov 核是函数 $K:E\times\mathcal H\to[0,1]$，满足：

1. 对每个 $x\in E$，$B\mapsto K(x,B)$ 是 $(F,\mathcal H)$ 上的概率测度；
2. 对每个 $B\in\mathcal H$，$x\mapsto K(x,B)$ 是 $\mathcal E$-可测函数。

若 $\mu$ 是 $E$ 上概率测度，定义输出分布

$$
(\mu K)(B)=\int_E K(x,B)\,\mu(dx).
$$

**定理 4.2（核复合给出概率测度）。** 若 $\mu$ 是 $E$ 上概率测度，$K$ 是从 $E$ 到 $F$ 的 Markov 核，则 $\mu K$ 是 $F$ 上概率测度。

**证明.** 对空集，$(\mu K)(\varnothing)=\int_E0\,d\mu=0$。对全集，$(\mu K)(F)=\int_E1\,d\mu=1$。若 $B_n$ 两两不交，则对每个 $x$，

$$
K\left(x,\bigcup_nB_n\right)=\sum_nK(x,B_n).
$$

右边为非负函数列的和，由单调收敛可交换积分与求和，得到

$$
(\mu K)\left(\bigcup_nB_n\right)=\sum_n(\mu K)(B_n).
$$

故 $\mu K$ 是概率测度。证毕。

**外部输入定理 4.3（Tonelli--Fubini，EI-4）。** 设 $(E,\mathcal E,\mu)$ 与 $(F,\mathcal H,\nu)$ 为 $\sigma$-有限测度空间，$f:E\times F\to[-\infty,\infty]$ 对乘积 $\sigma$-代数可测。

1. 若 $f\ge0$，则两种次序的迭代积分均有定义于 $[0,\infty]$，且
   $$
   \int f\,d(\mu\otimes\nu)
   =\int_E\left(\int_Ff(x,y)\,\nu(dy)\right)\mu(dx)
   =\int_F\left(\int_Ef(x,y)\,\mu(dx)\right)\nu(dy).
   $$
2. 若 $\int|f|\,d(\mu\otimes\nu)<\infty$，则内层积分几乎处处有限，两种迭代积分可积，并等于乘积积分。

本书用非负版本处理核复合与 Chapman--Kolmogorov，用可积版本处理独立乘积期望。来源与未重证边界见 [SOURCES.md](SOURCES.md) 的 EI-4。

## 4.3 例子：二状态转移核

**例 4.1（二状态核的一步输出）。**

令 $E=\{0,1\}$，转移矩阵

$$
P=\begin{pmatrix}
1-b & b\\
1-a & a
\end{pmatrix}
$$

表示 $P(i,j)=\mathbb P(X_{n+1}=j\mid X_n=i)$。若初始分布为 $\mu=(q,1-q)$，则下一步分布为 $\mu P$。这不是矩阵记号的偶然，而是有限核积分的具体形式。

## 练习

**练习 4.1.** 设 $X,Y$ 独立且均为 Bernoulli 随机变量。计算 $X+Y$ 的分布。

**练习 4.2.** 证明若 $X,Y$ 独立，则 $f(X),g(Y)$ 独立，其中 $f,g$ 可测。

**练习 4.3.** 对二状态核 $P$，写出 $P^2(0,1)$。
