# 第十五章：相同粒子、对称化与 Fock 空间

## 本章目标

本章建立相同粒子的对称化公设，定义玻色子、费米子、Slater 行列式和 Fock 空间。

## 依赖前置知识

需要张量积、置换群、行列式和正交投影。

## 15.1 对称与反对称子空间

**定义 15.1.** 设单粒子 Hilbert 空间为 $\mathcal H$。置换 $\pi\in S_n$ 在 $\mathcal H^{\otimes n}$ 上作用为
$$
U_\pi(\psi_1\otimes\cdots\otimes\psi_n)
=\psi_{\pi^{-1}(1)}\otimes\cdots\otimes\psi_{\pi^{-1}(n)}.
$$
对称子空间为 $U_\pi\Psi=\Psi$ 的向量集合，反对称子空间为 $U_\pi\Psi=\operatorname{sgn}(\pi)\Psi$ 的向量集合。

**定义 15.2.** 对称化和反对称化投影为
$$
P_+=\frac1{n!}\sum_{\pi\in S_n}U_\pi,\qquad
P_-=\frac1{n!}\sum_{\pi\in S_n}\operatorname{sgn}(\pi)U_\pi.
$$

**命题 15.3.** $P_+$ 与 $P_-$ 是正交投影。

**证明.** 由 $U_\pi^*=U_{\pi^{-1}}$。对 $P_+$，
$$
P_+^*=\frac1{n!}\sum_\pi U_{\pi^{-1}}=P_+.
$$
并且
$$
P_+^2=\frac1{(n!)^2}\sum_{\pi,\sigma}U_{\pi\sigma}
=\frac1{n!}\sum_\tau U_\tau=P_+,
$$
因为每个 $\tau$ 有 $n!$ 种表示 $\pi\sigma=\tau$。$P_-$ 同理，使用符号同态 $\operatorname{sgn}(\pi\sigma)=\operatorname{sgn}(\pi)\operatorname{sgn}(\sigma)$。$\square$

## 15.2 Pauli 原理

**命题 15.4.** 若 $\psi_1,\dots,\psi_n$ 线性相关，则反对称张量
$$
\psi_1\wedge\cdots\wedge\psi_n
$$
为零。

**证明.** 反对称张量的坐标系数是由 $\psi_j$ 坐标组成的行列式。若向量线性相关，行列式为零，因此该反对称张量为零。$\square$

这给出 Pauli 不相容原理的数学形式：两个费米子不能占据同一个单粒子态。

## 15.3 Fock 空间

**定义 15.5.** 玻色 Fock 空间和费米 Fock 空间分别为
$$
\mathcal F_+(\mathcal H)=\bigoplus_{n=0}^\infty \operatorname{Sym}^n\mathcal H,\qquad
\mathcal F_-(\mathcal H)=\bigoplus_{n=0}^\infty \wedge^n\mathcal H.
$$
其中 $n=0$ 项为真空线 $\mathbb C\Omega$。

**边界 15.6.** 产生湮灭算符、二次量子化 Hamiltonian 和无穷自由度场论需要更强的算子代数工具。本书只使用 Fock 空间作为相同粒子多体态空间。

## 15.4 占有数记号

**定义 15.7.** 设 $(e_j)_{j\in J}$ 是单粒子正交归一基。玻色 Fock 空间中的占有数向量写作
$$
|n_1,n_2,\dots\rangle,
$$
其中 $n_j\in\mathbb N$ 且只有有限多个非零。费米情形要求 $n_j\in\{0,1\}$。

**例子 15.8.** 两个玻色子同处于 $e_1$ 的态为 $|2,0,\dots\rangle$。两个费米子不能给出该态，因为反对称化使 $e_1\wedge e_1=0$。两个费米子分别处于 $e_1,e_2$ 的态为
$$
e_1\wedge e_2=\frac{1}{\sqrt2}(e_1\otimes e_2-e_2\otimes e_1).
$$

**命题 15.9.** 固定有限个单粒子模式时，玻色 $n$ 粒子态数等于
$$
\binom{n+r-1}{r-1},
$$
费米 $n$ 粒子态数为
$$
\binom r n.
$$

**证明.** 玻色情形是非负整数解 $n_1+\cdots+n_r=n$ 的数目，由 stars and bars 得第一式。费米情形每个 $n_j$ 只能为 $0$ 或 $1$，故只需选择被占据的 $n$ 个模式。$\square$

**定义 15.10.** 在占有数基上，粒子数算符定义为
$$
N|n_1,n_2,\dots\rangle=\left(\sum_jn_j\right)|n_1,n_2,\dots\rangle.
$$
固定 $n$ 粒子子空间正是 $N$ 的本征值 $n$ 子空间。

## 本章小结

相同粒子的态空间不是普通张量积，而是其对称或反对称部分。玻色子使用对称张量，费米子使用反对称张量。Fock 空间把不同粒子数的态组织为直和。

## 练习

**练习 15.1.** 对两个粒子写出 $P_+$ 与 $P_-$ 的显式公式。

**练习 15.2.** 证明两个相同费米子处于同一单粒子态时反对称态为零。
