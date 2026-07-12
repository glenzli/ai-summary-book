# 第三章：无界可观测量与谱定理

## 本章目标

本章解释为什么无限维量子力学必须处理无界算子，并固定自伴算子、谱测度和函数演算的使用口径。

## 依赖前置知识

需要 Hilbert 空间、有界算子、有限维谱分解和基本测度论直觉。

## 3.1 无界算子与定义域

**定义 3.1.** Hilbert 空间 $\mathcal H$ 上的无界算子是一个线性映射
$$
A:\mathcal D(A)\to\mathcal H,
$$
其中 $\mathcal D(A)\subseteq\mathcal H$ 是线性子空间，称为定义域。若 $\mathcal D(A)$ 稠密，称 $A$ 为稠定算子。

**定义 3.2.** 稠定算子 $A$ 的伴随 $A^*$ 的定义域由所有 $\psi\in\mathcal H$ 组成，使得存在 $\eta\in\mathcal H$ 满足
$$
\langle\psi,A\phi\rangle=\langle\eta,\phi\rangle
$$
对所有 $\phi\in\mathcal D(A)$ 成立。此时定义 $A^*\psi=\eta$。

**定义 3.3.** 稠定算子 $A$ 称为对称，若
$$
\langle A\psi,\phi\rangle=\langle\psi,A\phi\rangle
$$
对所有 $\psi,\phi\in\mathcal D(A)$ 成立。若 $A=A^*$ 且定义域相同，称 $A$ 为自伴。

**命题 3.4.** 自伴算子必为对称算子。

**证明.** 若 $A=A^*$，则 $\mathcal D(A)=\mathcal D(A^*)$。由伴随定义，对 $\psi,\phi\in\mathcal D(A)$，
$$
\langle A\psi,\phi\rangle=\langle\psi,A^*\phi\rangle=\langle\psi,A\phi\rangle.
$$
故 $A$ 对称。$\square$

## 3.2 位置与动量

**例子 3.5.** 在 $L^2(\mathbb R)$ 中，位置算子可写为
$$
(Xf)(x)=xf(x),
$$
定义域为所有满足 $xf(x)\in L^2(\mathbb R)$ 的 $f$。动量算子形式上为
$$
P=-i\frac{d}{dx}.
$$
其标准自伴 realization 的定义域为 Sobolev 空间
$$
\mathcal D(P)=H^1(\mathbb R).
$$
Fourier 变换把 $P$ 酉等价为动量变量的乘法算子，因此该 realization 自伴。Schwartz 空间 $\mathcal S(\mathbb R)$ 是 $X$ 与 $P$ 的共同不变稠密核，但把算子只限制在 $\mathcal S(\mathbb R)$ 上得到的是对称算子，而不是已经声明完整定义域的自伴算子。
在 Schwartz 空间 $\mathcal S(\mathbb R)$ 上，
$$
[X,P]f=i f.
$$

**证明.** 对 $f\in\mathcal S(\mathbb R)$，
$$
XPf=x(-if'),\qquad PXf=-i(xf)'=-if-ixf'.
$$
相减得 $(XP-PX)f=if$。$\square$

## 3.3 谱定理

**外部输入定理 3.6（谱定理，QM-EXT-1）.** 设 $A$ 为自伴算子。存在唯一投影值测度 $E_A$，使得
$$
A=\int_{\mathbb R}\lambda\,dE_A(\lambda).
$$
对 Borel 函数 $f:\mathbb R\to\mathbb C$，定义
$$
f(A)=\int_{\mathbb R}f(\lambda)\,dE_A(\lambda)
$$
及其定义域
$$
\mathcal D(f(A))
=\left\{\psi\in\mathcal H:
\int_{\mathbb R}|f(\lambda)|^2\,d\mu_\psi^A(\lambda)<\infty
\right\}.
$$
若 $f$ 有界，则 $f(A)$ 是定义在整个 $\mathcal H$ 上的有界算子；若 $f$ 实值，则 $f(A)$ 自伴；一般复值 $f$ 给出闭正规算子。等式 $A=\int\lambda\,dE_A$ 包含定义域陈述，而不只是形式积分。

**定义 3.7.** 在单位态 $\psi$ 中，可观测量 $A$ 的测量结果分布是实线上的概率测度
$$
\mu^A_\psi(\Delta)=\langle\psi,E_A(\Delta)\psi\rangle.
$$

**命题 3.8.** $\mu^A_\psi$ 是概率测度。

**证明.** 谱测度满足 $E_A(\varnothing)=0$、$E_A(\mathbb R)=I$，并对两两不交 Borel 集可列可加。于是
$$
\mu^A_\psi(\mathbb R)=\langle\psi,\psi\rangle=1.
$$
正性来自 $E_A(\Delta)$ 是正交投影：
$$
\langle\psi,E_A(\Delta)\psi\rangle=\|E_A(\Delta)\psi\|^2\ge0.
$$
可列可加由谱测度的强可列可加性与内积连续性得到。$\square$

## 3.4 有限矩与期望值的定义域

**定义 3.9.** 设 $A$ 为自伴算子。单位态 $\psi$ 对 $A$ 有有限一阶矩，若
$$
\int_{\mathbb R}|\lambda|\,d\mu_\psi^A(\lambda)<\infty.
$$
有有限二阶矩，若
$$
\int_{\mathbb R}\lambda^2\,d\mu_\psi^A(\lambda)<\infty.
$$

**命题 3.10.** 若 $\psi\in\mathcal D(A)$，则 $\psi$ 对 $A$ 有有限二阶矩，且
$$
\|A\psi\|^2=\int_{\mathbb R}\lambda^2\,d\mu_\psi^A(\lambda).
$$

**证明.** 由谱定理的函数演算，$A$ 对应函数 $\lambda$，其平方范数对应 $|\lambda|^2$ 的谱积分：
$$
\|A\psi\|^2=\langle A\psi,A\psi\rangle
=\int_{\mathbb R}\lambda^2\,d\mu_\psi^A(\lambda).
$$
定义域 $\mathcal D(A)$ 正由右侧有限的向量组成。$\square$

**说明 3.11.** 并非每个单位向量都有某个无界可观测量的有限期望。写 $\langle\psi,A\psi\rangle$ 前必须知道 $\psi$ 至少在相应 quadratic form 或算子定义域内。

**命题 3.12（一阶矩与二阶矩的区别）.** 对单位向量 $\psi$，若
$$
\int_{\mathbb R}|\lambda|\,d\mu_\psi^A(\lambda)<\infty,
$$
则谱积分
$$
\mathbb E_\psi[A]\coloneqq
\int_{\mathbb R}\lambda\,d\mu_\psi^A(\lambda)
$$
作为有限实数有定义。该条件一般弱于 $\psi\in\mathcal D(A)$；后者等价于有限二阶矩。

**证明.** 一阶绝对矩有限保证正负部分积分都有限，故期望的 Lebesgue 积分存在。命题 3.10 已给出
$$
\psi\in\mathcal D(A)
\Longleftrightarrow
\int_{\mathbb R}\lambda^2\,d\mu_\psi^A(\lambda)<\infty.
$$
由 Cauchy--Schwarz，二阶矩有限蕴含一阶矩有限；反向一般不成立。具体地，在 $\ell^2(\mathbb N)$ 中令
$$
Ae_n=ne_n,
\qquad
p_n=\frac{1}{\zeta(3)n^3},
\qquad
\psi=\sum_{n\ge1}\sqrt{p_n}\,e_n.
$$
这里 $\zeta(3)=\sum_{n\ge1}n^{-3}$。
则 $\|\psi\|=1$，且
$$
\sum_{n\ge1}np_n<\infty,
\qquad
\sum_{n\ge1}n^2p_n=\infty.
$$
故 $\psi$ 有有限一阶矩，却不属于 $\mathcal D(A)$。$\square$

## 本章小结

无限维量子力学中的位置、动量和 Hamiltonian 通常是无界算子。可观测量必须用自伴算子而非仅对称算子表示。谱定理把自伴算子转化为投影值测度，从而给出一般 Born 概率分布。

## 练习

**练习 3.1.** 说明为什么 $X$ 在 $L^2(\mathbb R)$ 上不是有界算子。

**练习 3.2.** 若 $A$ 是有限维自伴矩阵，说明第三章的谱测度如何退化为第二章的谱投影和。
