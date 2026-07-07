# 第一章：Hilbert 空间、态与射线

## 本章目标

本章建立量子态的线性代数基础：复 Hilbert 空间、单位向量、射线、转移概率和有限维纯态的几何。

## 依赖前置知识

需要复向量空间、内积空间、正交投影和有限维矩阵计算。

## 1.1 Hilbert 空间

**定义 1.1.** 一个复 Hilbert 空间（Hilbert space）是带内积 $\langle-,-\rangle$ 的复向量空间 $\mathcal H$，满足由范数
$$
\|\psi\|=\sqrt{\langle\psi,\psi\rangle}
$$
诱导的度量是完备的。本书约定内积对第二变量线性。

**例子 1.2.** $\mathbb C^n$ 带
$$
\langle z,w\rangle=\sum_{j=1}^n\overline{z_j}w_j
$$
是 Hilbert 空间。$L^2(\mathbb R^d)$ 中向量是平方可积函数的等价类，内积为
$$
\langle f,g\rangle=\int_{\mathbb R^d}\overline{f(x)}g(x)\,dx.
$$

**命题 1.3（Cauchy-Schwarz 不等式）.** 对任意 $\psi,\phi\in\mathcal H$，
$$
|\langle\psi,\phi\rangle|\le \|\psi\|\,\|\phi\|.
$$

**证明.** 若 $\psi=0$ 则成立。设 $\psi\ne0$，令
$$
\lambda=\frac{\langle\psi,\phi\rangle}{\|\psi\|^2}.
$$
由正性，
$$
0\le \|\phi-\lambda\psi\|^2
=\|\phi\|^2-\overline{\lambda}\langle\psi,\phi\rangle-\lambda\langle\phi,\psi\rangle+|\lambda|^2\|\psi\|^2.
$$
代入 $\lambda$ 得
$$
0\le \|\phi\|^2-\frac{|\langle\psi,\phi\rangle|^2}{\|\psi\|^2}.
$$
整理即得结论。$\square$

## 1.2 态矢量与射线

**定义 1.4.** 一个归一化态矢量是满足 $\|\psi\|=1$ 的向量。两个单位向量 $\psi,\phi$ 表示同一纯态，若存在 $\theta\in\mathbb R$ 使
$$
\phi=e^{i\theta}\psi.
$$
等价类 $[\psi]$ 称为射线（ray）。

**定义 1.5.** 两个纯态 $[\psi]$ 与 $[\phi]$ 的转移概率定义为
$$
p([\psi],[\phi])=|\langle\psi,\phi\rangle|^2.
$$

**命题 1.6.** 转移概率与代表元选择无关。

**证明.** 若 $\psi'=e^{i\alpha}\psi$、$\phi'=e^{i\beta}\phi$，则
$$
\langle\psi',\phi'\rangle=e^{i(\beta-\alpha)}\langle\psi,\phi\rangle.
$$
取绝对值平方后相位因子消失。$\square$

## 1.3 正交分解

**定义 1.7.** 向量族 $(e_j)_{j\in J}$ 称为正交归一族，若 $\langle e_j,e_k\rangle=\delta_{jk}$。若其闭线性张成为整个 $\mathcal H$，称为正交归一基。

**命题 1.8.** 若 $(e_j)_{j=1}^n$ 是有限维 Hilbert 空间的正交归一基，则
$$
\psi=\sum_{j=1}^n e_j\langle e_j,\psi\rangle,\qquad
\|\psi\|^2=\sum_{j=1}^n|\langle e_j,\psi\rangle|^2.
$$

**证明.** 令 $\eta=\psi-\sum_j e_j\langle e_j,\psi\rangle$。对每个 $k$，
$$
\langle e_k,\eta\rangle=\langle e_k,\psi\rangle-\sum_j\langle e_k,e_j\rangle\langle e_j,\psi\rangle=0.
$$
因基张成全空间，$\eta=0$。范数公式由正交性展开内积得到。$\square$

## 1.4 射线的投影表示

**定义 1.9.** 对单位向量 $\psi$，定义秩一投影
$$
P_\psi=|\psi\rangle\langle\psi|.
$$
它作用在 $\phi\in\mathcal H$ 上为
$$
P_\psi\phi=\psi\langle\psi,\phi\rangle.
$$

**命题 1.10.** 两个单位向量 $\psi,\phi$ 表示同一射线，当且仅当
$$
P_\psi=P_\phi.
$$

**证明.** 若 $\phi=e^{i\theta}\psi$，则
$$
|\phi\rangle\langle\phi|
=e^{i\theta}|\psi\rangle e^{-i\theta}\langle\psi|
=P_\psi.
$$
反过来若 $P_\psi=P_\phi$，则
$$
\psi=P_\psi\psi=P_\phi\psi=\phi\langle\phi,\psi\rangle.
$$
因 $\|\psi\|=\|\phi\|=1$，系数 $\langle\phi,\psi\rangle$ 的模为 $1$，故二者只差相位。$\square$

**说明 1.11.** 用秩一投影表示纯态可自动消去整体相位，并为第十七章的密度算子口径做准备。

## 本章小结

纯态不是单位向量本身，而是单位向量模整体相位后的射线。转移概率由内积绝对值平方给出，并且与相位选择无关。正交归一基给出态的概率振幅展开。

## 练习

**练习 1.1.** 证明若两个单位向量满足 $|\langle\psi,\phi\rangle|=1$，则它们表示同一射线。

**练习 1.2.** 在 $\mathbb C^2$ 中令 $\psi=(1,0)$，$\phi=(1,1)/\sqrt2$。计算转移概率。
