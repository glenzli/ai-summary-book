# 随机过程与 Markov 链

一个随机过程不是一张时间序列表，而是一族定义在同一概率空间上的随机变量。只给每个时刻的分布还不够；还必须给出跨时刻的联合结构。Markov 链把这种结构压缩为初始分布和一步转移核。这样一来，路径概率、长期分布和熵率都能从同一转移对象中读出，并且每一步计算都有可检查的概率含义。

## 7.1 随机过程

**定义 7.1（随机过程）。** 设 $T$ 是索引集，$(E,\mathcal E)$ 是状态空间。$E$ 值随机过程是随机变量族 $(X_t)_{t\in T}$，其中 $X_t:\Omega\to E$。

**定义 7.2（有限维分布）。** 对 $t_1,\ldots,t_m\in T$，随机向量 $(X_{t_1},\ldots,X_{t_m})$ 在 $E^m$ 上的分布称为该过程的有限维分布。

**外部输入定理 7.3（Kolmogorov 扩张，EI-5）。** 设 $T$ 为任意索引集，$(E,\mathcal E)$ 为标准 Borel 空间。对每个非空有限子集 $J\subseteq T$，给定 $E^J$ 上概率测度 $\mu_J$。若对所有有限 $J\subseteq K\subseteq T$，坐标投影 $\pi_{K,J}:E^K\to E^J$ 满足

$$
(\pi_{K,J})_\#\mu_K=\mu_J,
$$

则在 $(E^T,\mathcal E^{\otimes T})$ 上存在唯一概率测度 $\mu$，使每个有限坐标投影的分布为 $\mu_J$。本书用该结果从一致有限维分布构造过程测度；它不保证连续、右连续或任何其他路径正则性。来源与未重证边界见 [SOURCES.md](SOURCES.md) 的 EI-5。

## 7.2 Markov 链

**定义 7.3（离散时间 Markov 链）。** 设 $(E,\mathcal E)$ 为可测状态空间，$P$ 是从 $E$ 到 $E$ 的 Markov 核。过程 $(X_n)_{n\ge0}$ 称为初始分布 $\mu$、转移核 $P$ 的 Markov 链，若 $X_0\sim\mu$，且对所有 $n\ge0$、$B\in\mathcal E$，

$$
\mathbb E[\mathbf 1_{\{X_{n+1}\in B\}}\mid
\sigma(X_0,\ldots,X_n)]=P(X_n,B)
$$

几乎处处成立。右端可测且有界，因此定义良好；这个定义不要求先选择逐点正则条件分布。

**定理 7.1（Chapman--Kolmogorov 方程）。** 对 Markov 核 $P$，令 $P^0(x,B)=\mathbf 1_B(x)$，并递归定义

$$
P^{n+1}(x,B)=\int_E P^n(y,B)P(x,dy).
$$

则每个 $P^n$ 都是 Markov 核，且对所有 $m,n\ge0$，

$$
P^{m+n}(x,B)=\int_E P^n(y,B)P^m(x,dy).
$$

**证明.** 先记录核积分的可测性。若 $f:E\to[0,\infty]$ 可测，则

$$
x\longmapsto\int_Ef(y)P(x,dy)
$$

可测：对示性函数这是核定义的第二条，对非负简单函数由有限线性成立，对一般 $f$ 取递增简单逼近并使用点态极限的可测性。现在 $P^0$ 是恒等核。若 $P^n$ 是核，则对固定 $x$，$B\mapsto P^{n+1}(x,B)$ 是核 $P(x,\cdot)$ 作用于 $P^n$ 后的概率测度；对固定 $B$，刚证的核积分可测性给出 $x\mapsto P^{n+1}(x,B)$ 可测。因此归纳得到每个 $P^n$ 都是核。

当 $m=0$ 时公式由恒等核定义成立；当 $m=1$ 时正是递归定义。若公式对某个 $m\ge1$ 成立，则

$$
P^{m+1+n}(x,B)=\int_E P^{m+n}(z,B)P(x,dz)
$$

$$
=\int_E\left[\int_E P^n(y,B)P^m(z,dy)\right]P(x,dz).
$$

被积函数非负。由 Tonelli 定理 EI-4 交换两次核积分，并按 $P^{m+1}$ 的定义组合内层，得到

$$
P^{m+1+n}(x,B)=\int_E P^n(y,B)P^{m+1}(x,dy).
$$

归纳完成。证毕。

**定义 7.4（不变分布）。** 概率测度 $\pi$ 称为 $P$ 的不变分布，若 $\pi P=\pi$。

**定理 7.2（不变分布在时间演化下保持不变）。** 若 $X_0\sim\pi$ 且 $\pi P=\pi$，则 Markov 链满足 $X_n\sim\pi$ 对所有 $n\ge0$。

**证明.** 对 $n=0$ 成立。若 $X_n\sim\pi$，则对任意 $B\in\mathcal E$，由全期望和 Markov 性，

$$
\begin{aligned}
\mathbb P(X_{n+1}\in B)
&=\mathbb E\!\left[
\mathbb E[\mathbf 1_{\{X_{n+1}\in B\}}\mid
\sigma(X_0,\ldots,X_n)]
\right]\\
&=\mathbb E[P(X_n,B)]
=\int_EP(x,B)\,\pi(dx)
=(\pi P)(B)=\pi(B).
\end{aligned}
$$

所以 $X_{n+1}\sim\pi$，归纳完成。证毕。

## 7.3 二状态链

**例 7.1（二状态链的不变分布）。**

对第 4 章矩阵

$$
P=\begin{pmatrix}1-b&b\\1-a&a\end{pmatrix},
$$

若 $0<a<1$、$0<b<1$，不变分布 $\pi=(\pi_0,\pi_1)$ 解方程 $\pi P=\pi$ 与 $\pi_0+\pi_1=1$。计算得

$$
\pi_1=\frac b{1-a+b},\qquad \pi_0=\frac{1-a}{1-a+b}.
$$

当 $a=b=p$ 且初始分布也是 Bernoulli$(p)$ 时，每一步条件分布都与过去无关，故过程是独立同分布的 Bernoulli$(p)$ 过程。若初始分布不是 Bernoulli$(p)$，则只有 $X_1,X_2,\ldots$ 独立同分布，$X_0$ 可能有不同分布。当 $a$ 与 $b$ 不同，单时刻平稳分布不能决定相邻联合分布。

## 练习

**练习 7.1.** 对有限 Markov 链证明 $n$ 步转移矩阵为普通矩阵幂 $P^n$。

**练习 7.2.** 在例 7.1 的参数范围 $0<a,b<1$ 内，判断二状态链在什么条件下 $\pi_1=1/2$。

**练习 7.3.** 令隐藏变量 $\Theta$ 等概率取 $1/4$ 与 $3/4$；给定 $\Theta=\theta$ 后，令 $(X_n)_{n\ge1}$ 独立同分布为 Bernoulli$(\theta)$。证明每个 $X_n$ 都服从 Bernoulli$(1/2)$，过程平稳，但它不是一阶 Markov 链。
