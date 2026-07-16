# 附录 A：测度扩张与条件分布接口

本附录集中说明正文所用的四个存在性输入。完整书目定位见 [SOURCES.md](SOURCES.md)；这里强调各输入的对象类型以及它们不能互相替代的边界。

## A.1 从预测度到测度

**外部输入 A.1（Caratheodory 扩张，EI-1）。** 设 $\mathcal A$ 是集合 $\Omega$ 上的代数。函数 $\mu_0:\mathcal A\to[0,\infty]$ 称为预测度，若 $\mu_0(\varnothing)=0$，并且对每个两两不交序列 $(A_n)\subseteq\mathcal A$，只要 $\bigcup_nA_n\in\mathcal A$，就有

$$
\mu_0\left(\bigcup_nA_n\right)=\sum_n\mu_0(A_n).
$$

若 $\Omega=\bigcup_nE_n$，其中 $E_n\in\mathcal A$ 且 $\mu_0(E_n)<\infty$，则 $\mu_0$ 在 $\sigma(\mathcal A)$ 上存在唯一测度延拓。

第 1 章用这一版本解释从区间代数或其他生成代数上的一致赋值进入完整可测结构。EI-1 本身不判断一族有限维分布是否彼此一致。

## A.2 从有限维分布到过程

**外部输入 A.2（Kolmogorov 扩张，EI-5）。** 设 $T$ 为索引集，$(E,\mathcal E)$ 为标准 Borel 空间。对每个非空有限 $J\subseteq T$ 给定 $E^J$ 上概率测度 $\mu_J$，并假设对有限 $J\subseteq K$，

$$
(\pi_{K,J})_\#\mu_K=\mu_J.
$$

则在 $(E^T,\mathcal E^{\otimes T})$ 上存在唯一概率测度 $\mu$，使 $(\pi_J)_\#\mu=\mu_J$ 对每个有限 $J$ 成立。

这一定理承担过程测度存在性。若要求连续路径、右连续路径或可分版本，还需额外矩估计或正则性定理；本书不调用这些强化。

## A.3 有限条件分布

**命题 A.1（有限条件分布核）。** 设 $(X,Y)$ 取有限集合 $\mathcal X\times\mathcal Y$ 值。固定任意 $\mathcal X$ 上概率分布 $\rho$，定义

$$
K(y,A)=
\begin{cases}
\mathbb P(X\in A,Y=y)/\mathbb P(Y=y),
&\mathbb P(Y=y)>0,\\
\rho(A),&\mathbb P(Y=y)=0.
\end{cases}
$$

则 $K$ 是从 $\mathcal Y$ 到 $\mathcal X$ 的 Markov 核，并且对所有 $A\subseteq\mathcal X$、$B\subseteq\mathcal Y$，

$$
\mathbb P(X\in A,Y\in B)
=\sum_{y\in B}K(y,A)\mathbb P(Y=y).
$$

**证明.** 对固定 $y$，正概率情形是通常的条件概率分布，零概率情形由 $\rho$ 给出概率分布；有限空间上的可测性自动成立。若 $\mathbb P(Y=y)>0$，则

$$
K(y,A)\mathbb P(Y=y)=\mathbb P(X\in A,Y=y).
$$

若 $\mathbb P(Y=y)=0$，等式两边都为零，与 $\rho$ 的选择无关。对 $y\in B$ 求和即得联合分布恒等式。证毕。

## A.4 正则条件分布

**外部输入 A.3（正则条件分布，EI-6）。** 设 $S,T$ 为标准 Borel 空间，$X:\Omega\to S$、$Y:\Omega\to T$ 可测。则存在从 $T$ 到 $S$ 的 Markov 核 $K$，使

$$
\mathbb P(X\in A,Y\in B)
=\int_BK(y,A)\,\mathcal L(Y)(dy)
$$

对所有 $A\in\mathcal B(S)$、$B\in\mathcal B(T)$ 成立。两个满足该式的核对 $\mathcal L(Y)$-几乎每个 $y$ 给出同一概率测度。

命题 A.1 是这一结论的有限版本。标准 Borel 假设是本书的明确边界；在任意可测空间上，抽象条件期望仍存在，但未必能选择成以条件值 $y$ 为参数的概率核。

## A.5 Radon--Nikodym 与条件期望

**外部输入 A.4（Radon--Nikodym，EI-3）。** 设 $\mu$ 为 $\sigma$-有限正测度，$\nu$ 为 $\sigma$-有限正测度且 $\nu\ll\mu$。则存在可测 $f:S\to[0,\infty]$，使

$$
\nu(A)=\int_Af\,d\mu
$$

对所有可测 $A$ 成立；$f$ 在 $\mu$-几乎处处意义下唯一。若 $\nu$ 有限，则 $f\in L^1(\mu)$。

第 5 章对有限正测度

$$
\nu^\pm(A)=\int_AX^\pm\,d\mathbb P,\qquad A\in\mathcal G,
$$

分别应用 EI-3，再取两个密度之差。这样得到条件期望的存在；几乎处处唯一性、线性、提出因子与塔性质均在正文证明。
