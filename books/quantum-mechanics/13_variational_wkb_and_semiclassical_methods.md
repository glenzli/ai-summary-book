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

**定义 13.2（Rayleigh--Ritz 值）.** 设 $H$ 是下有界自伴算子，
$M\subset\mathcal D(H)$ 是非零线性子空间。定义
$$
E_M=\inf\left\{\langle\psi,H\psi\rangle:
\psi\in M,\ \|\psi\|=1\right\}.
$$
若使用 $H$ 的闭二次型来扩大试探类，则必须把 $M$ 改取在该二次型的
定义域中，并把上式理解为二次型值。

**命题 13.2A（变分上界）.** 在定义 13.2 的假设下，
$$
\inf\sigma(H)\le E_M.
$$
若 $E_0=\inf\sigma(H)$ 是基态本征值，则 $E_M$ 是 $E_0$ 的上界。
当 $M$ 有限维时，上述下确界由 $M$ 中某个单位向量达到。

**证明.** 下有界自伴算子的谱定理给出
$\langle\psi,H\psi\rangle\ge\inf\sigma(H)$，故对试探态取下确界即得
第一式。若 $M$ 有限维，则 $H|_M:M\to\mathcal H$ 连续，Rayleigh
商在 $M$ 的单位球面上连续；该球面紧，所以下确界达到。$\square$

## 13.2 WKB 近似

**设定 13.3.** 取 $m>0$、$\hbar>0$。设实势函数 $V$ 在开区间
$I\subset\mathbb R$ 上至少二次连续可微。一维定态方程为
$$
-\frac{\hbar^2}{2m}\psi''(x)+V(x)\psi(x)=E\psi(x).
$$
在经典允许区 $I_E=\{x\in I:E>V(x)\}$，定义正根
$$
p(x)=\sqrt{2m(E-V(x))}.
$$

**推导 13.4（WKB 首阶）.** 在 $I_E$ 的一个连通分支上，设
$A,S\in C^2$、$A$ 不为零，且二者在形式展开中不依赖 $\hbar$。代入
$$
\psi(x)=A(x)e^{\frac{i}{\hbar}S(x)}
$$
并按 $\hbar$ 的阶数比较，首阶给出 Hamilton-Jacobi 方程
$$
(S'(x))^2=2m(E-V(x)).
$$

**推导.** 计算
$$
\psi''=\left(A''+\frac{2i}{\hbar}A'S'
+\frac{i}{\hbar}AS''-\frac1{\hbar^2}A(S')^2\right)e^{iS/\hbar}.
$$
代入 Schrodinger 方程，$\hbar^0$ 阶为
$$
\frac1{2m}(S')^2+V=E.
$$
整理即得。$\square$

**外部输入公式 13.5（WKB 量子化，QM-EXT-15）.** 设光滑实势阱在
所考虑的正则能区只有两个简单转折点
$a(E)<b(E)$，即 $V(a(E))=V(b(E))=E$ 且两点的 $V'$ 非零，并在二者
之间有 $V<E$。取可随 $\hbar$ 变化的指标
$n=n(\hbar)\in\mathbb Z_{\ge0}$，使半经典束缚态能量
$E_n(\hbar)$ 留在该正则能区，则
首阶 Bohr--Sommerfeld 条件为
$$
\int_{a(E_n)}^{b(E_n)}
\sqrt{2m(E_n-V(x))}\,dx
=\pi\hbar\left(n+\frac12\right)+O(\hbar^2).
$$
余项在固定的紧正则能区内理解；临界能量、并合或高阶转折点以及硬壁
边界不在此公式的假设内。该结论依赖 Airy 转折点连接与半经典谱分析，
本书不重证。

## 13.3 试探函数的具体使用

**例子 13.6.** 回到本书 $\hbar=1$ 的默认单位。取 $m>0$、
$\omega>0$，对 Hamiltonian
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

**练习 13.2.** 对长度为 $L$ 的无限深方势阱，使用 Dirichlet 硬壁的
半经典相位条件验证高能能级的主项；说明为什么不能直接套用公式 13.5
的两个简单转折点相位。
