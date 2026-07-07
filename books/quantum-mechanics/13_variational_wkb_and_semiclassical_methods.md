# 第十三章：变分法、WKB 与半经典近似

## 本章目标

本章介绍基态能量变分原理、Rayleigh-Ritz 方法和一维 WKB 近似。

## 依赖前置知识

需要自伴算子期望值、束缚态和一维 Schrodinger 方程。

## 13.1 变分原理

**命题 13.1（有限维 Rayleigh 原理）.** 设 $H$ 为有限维自伴矩阵，最小本征值为 $E_0$。则
$$
E_0=\min_{\|\psi\|=1}\langle\psi,H\psi\rangle.
$$

**证明.** 取 $H$ 的正交归一本征基 $e_j$，本征值 $E_0\le E_1\le\dots$。若 $\psi=\sum_jc_je_j$ 且 $\sum_j|c_j|^2=1$，则
$$
\langle\psi,H\psi\rangle=\sum_jE_j|c_j|^2\ge E_0\sum_j|c_j|^2=E_0.
$$
等号由 $\psi=e_0$ 达到。$\square$

**定义 13.2.** 给定试探子空间 $M\subset\mathcal H$，Rayleigh-Ritz 方法在 $M$ 中最小化 $\langle\psi,H\psi\rangle$，得到基态能量上界。

## 13.2 WKB 近似

**设定 13.3.** 一维定态方程恢复 $\hbar$：
$$
-\frac{\hbar^2}{2m}\psi''(x)+V(x)\psi(x)=E\psi(x).
$$
在经典允许区 $E>V(x)$，定义
$$
p(x)=\sqrt{2m(E-V(x))}.
$$

**命题 13.4.** 形式代入
$$
\psi(x)=A(x)e^{\frac{i}{\hbar}S(x)}
$$
并按 $\hbar$ 的阶数比较，首阶给出 Hamilton-Jacobi 方程
$$
(S'(x))^2=2m(E-V(x)).
$$

**证明.** 计算
$$
\psi''=\left(A''+\frac{2i}{\hbar}A'S'
\frac{i}{\hbar}AS''-\frac1{\hbar^2}A(S')^2\right)e^{iS/\hbar}.
$$
代入 Schrodinger 方程，$\hbar^0$ 阶为
$$
\frac1{2m}(S')^2+V=E.
$$
整理即得。$\square$

**公式 13.5（WKB 量子化）.** 对两个转折点 $a,b$ 的一维束缚态，WKB 条件为
$$
\int_a^b p(x)\,dx=\pi\hbar\left(n+\frac12\right).
$$
该公式依赖转折点连接公式，作为半经典外部输入定理 QM-EXT-15 使用。

## 13.3 试探函数的具体使用

**例子 13.6.** 对 Hamiltonian
$$
H=\frac{P^2}{2m}+\frac12m\omega^2X^2
$$
取归一 Gaussian 试探态
$$
\psi_\alpha(x)=\left(\frac{2\alpha}{\pi}\right)^{1/4}e^{-\alpha x^2},
\qquad \alpha>0.
$$
则
$$
\langle X^2\rangle_{\psi_\alpha}=\frac1{4\alpha},
\qquad
\langle P^2\rangle_{\psi_\alpha}=\alpha.
$$
因此
$$
E(\alpha)=\frac{\alpha}{2m}+\frac{m\omega^2}{8\alpha}.
$$

**命题 13.7.** 上述 $E(\alpha)$ 的最小值为 $\omega/2$。

**证明.** 求导得
$$
E'(\alpha)=\frac1{2m}-\frac{m\omega^2}{8\alpha^2}.
$$
令其为零，得 $\alpha=m\omega/2$。代回：
$$
E_{\min}=\frac{\omega}{4}+\frac{\omega}{4}=\frac{\omega}{2}.
$$
这与谐振子精确基态能量一致，说明基态正是该 Gaussian。$\square$

## 本章小结

变分法给出基态能量上界，是不可解系统的稳健工具。WKB 方法把 Schrodinger 方程在小 $\hbar$ 极限中联系到经典作用量，适用于缓变势和高量子数。

## 练习

**练习 13.1.** 用 Rayleigh 原理证明任意归一试探态给出的能量期望不低于基态能量。

**练习 13.2.** 对无限深方势阱验证 WKB 量子化给出的能级主项。
