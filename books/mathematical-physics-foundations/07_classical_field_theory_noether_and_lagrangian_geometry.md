# 第七章：经典场论、Noether 定理与 Lagrange 几何

有限维力学的构型是一条曲线，场论的构型则是时空上的截面。变分仍然给出 Euler-Lagrange 方程，但边界项、守恒流、规范冗余和 Cauchy 数据都变得更重要。本章只建立局部 Lagrange 场论的核心结构：作用量、场方程、Noether 定理、能动张量和自由场的辛形式。

## 7.1 场论变分

**定义 7.1.** 设 $X$ 为 $d$ 维时空，场为丛 $E\to X$ 的截面 $\phi$。一阶 Lagrange 密度局部写为
$$
\mathcal L(\phi,\partial_\mu\phi,x)\,d^dx.
$$
作用量为 $S[\phi]=\int_X\mathcal L\,d^dx$，取紧支撑变分或给定边界条件。

**命题 7.1 (`P`).** 设 $X\subset\mathbb R^d$ 开，$\mathcal L\in C^2$，且 $\phi\in C^2(X,\mathbb R^N)$ 对所有 $\xi\in C_c^1(X,\mathbb R^N)$ 的变分都是临界场。则
$$
\frac{\partial\mathcal L}{\partial\phi^a}
-\partial_\mu\frac{\partial\mathcal L}{\partial(\partial_\mu\phi^a)}=0.
$$

**证明.** 对 $\phi_\epsilon=\phi+\epsilon\xi$ 求一阶变分。$\xi$ 的支撑紧且 $\mathcal L\in C^2$，所以可在积分号下求导：
$$
\delta S=\int\left(
\frac{\partial\mathcal L}{\partial\phi^a}\xi^a+
\frac{\partial\mathcal L}{\partial(\partial_\mu\phi^a)}\partial_\mu\xi^a
\right)d^dx.
$$
分部积分并用紧支撑消去边界项，得到
$$
0=\delta S[\phi](\xi)=\int_X E_a(\mathcal L)(x)\xi^a(x)\,d^dx,
$$
其中 $E_a(\mathcal L)$ 是命题左端。该函数连续。若它在某点严格为正或负，则可在同号邻域中选择非负 bump 函数作为对应分量的 $\xi$，使积分非零，矛盾。因此每个 $E_a(\mathcal L)$ 逐点为零。$\square$

## 7.2 Noether 第一定理

**定义 7.2.** 一个竖直无穷小变换 $\delta_\epsilon\phi^a=\epsilon R^a(\phi,x)$ 是作用量对称，如果存在局部函数 $B^\mu$ 使
$$
\delta\mathcal L=\partial_\mu B^\mu
$$
在任意场上成立。

**命题 7.2 (`P`, Noether 第一定理).** 每个连续作用量对称给出壳上守恒流
$$
j^\mu=\frac{\partial\mathcal L}{\partial(\partial_\mu\phi^a)}R^a-B^\mu,
\qquad
\partial_\mu j^\mu=0
$$
其中“壳上”表示场满足 Euler-Lagrange 方程。

**证明.** 由链式法则与乘积法则，
$$
\delta\mathcal L=E_a(\mathcal L)R^a+\partial_\mu
\left(\frac{\partial\mathcal L}{\partial(\partial_\mu\phi^a)}R^a\right).
$$
另一方面，对称性假设给出 $\delta\mathcal L=\partial_\mu B^\mu$。两式相减即得
$$
\partial_\mu j^\mu=-E_a(\mathcal L)R^a.
$$
在 Euler-Lagrange 方程壳上，右端逐点为零，因此 $j$ 守恒。$\square$

**命题 7.3 (`P`).** 若 $\mathcal L(\phi,\partial\phi)$ 不显含 $x$，则 canonical 能动张量
$$
T^\mu{}_\nu=
\frac{\partial\mathcal L}{\partial(\partial_\mu\phi^a)}\partial_\nu\phi^a
-\delta^\mu{}_\nu\mathcal L.
$$
在每个经典解上满足 $\partial_\mu T^\mu{}_\nu=0$。

**证明.** 记 $\pi_a^\mu=\partial\mathcal L/\partial(\partial_\mu\phi^a)$。直接求散度，并使用偏导可交换，
$$
\begin{aligned}
\partial_\mu T^\mu{}_\nu
&=(\partial_\mu\pi_a^\mu)\partial_\nu\phi^a
+\pi_a^\mu\partial_\mu\partial_\nu\phi^a
-\partial_\nu\mathcal L\\
&=\left(\partial_\mu\pi_a^\mu-
\frac{\partial\mathcal L}{\partial\phi^a}\right)
\partial_\nu\phi^a
=-E_a(\mathcal L)\partial_\nu\phi^a.
\end{aligned}
$$
第二行使用了 $\mathcal L$ 不显含 $x$ 的链式法则。壳上 $E_a(\mathcal L)=0$，故散度为零。$\square$

## 7.3 自由 Klein-Gordon 场

**定义 7.3.** 在 Minkowski 空间上，自由实标量场 Lagrange 密度为
$$
\mathcal L=-\frac12\eta^{\mu\nu}\partial_\mu\phi\partial_\nu\phi-\frac12m^2\phi^2.
$$
以 mostly plus 约定，场方程为
$$
(\Box-m^2)\phi=0,\qquad \Box=-\partial_t^2+\Delta.
$$

**命题 7.4 (`P`).** 在 Minkowski 空间中，设 $\phi_1,\phi_2$ 是 Klein-Gordon 方程的光滑 spacelike-compact 解。若 $\Sigma_0,\Sigma_1$ 是两张光滑 spacelike Cauchy 面，则辛配对
$$
\Omega_\Sigma(\phi_1,\phi_2)=\int_\Sigma
(\phi_1 n^\mu\partial_\mu\phi_2-\phi_2 n^\mu\partial_\mu\phi_1)\,d\Sigma
$$
与 Cauchy 面 $\Sigma$ 的选择无关。

**证明.** 定义流
$J^\mu=\phi_1\partial^\mu\phi_2-\phi_2\partial^\mu\phi_1$。若二者满足 Klein-Gordon 方程，则
$$
\partial_\mu J^\mu=\phi_1\Box\phi_2-\phi_2\Box\phi_1=0.
$$
其中最后一步使用 $\Box\phi_i=m^2\phi_i$。取足够大的紧空间柱，使两解在两 Cauchy 面之间的侧边界邻域为零；spacelike-compact 条件保证可以这样选择。对柱内由 $\Sigma_0$、$\Sigma_1$ 与侧边界围成的区域应用散度定理。侧边界积分为零，两个 Cauchy 面的诱导定向相反，故
$$
0=\int\partial_\mu J^\mu\,d^dx
=\Omega_{\Sigma_1}(\phi_1,\phi_2)-
\Omega_{\Sigma_0}(\phi_1,\phi_2).
$$
因此配对与 Cauchy 面选择无关。$\square$

**定理 7.5 (`E`).** 全局双曲时空上的 Green 双曲算子具有唯一的先进/推迟基本解，并满足因果支撑性质。

**外部输入边界.** 本书只调用 Green 算子的存在唯一性、$G^\pm:C_c^\infty\to C^\infty$ 的因果支撑
$\operatorname{supp}(G^\pm f)\subset J^\pm(\operatorname{supp}f)$，不证明能量估计与全局双曲 PDE 理论；定位见 [SOURCES.md](SOURCES.md) 的 `E-7.5`。

**例 7.6（Klein--Gordon 平面波）.** 在 Minkowski 空间取
$\phi(t,\mathbf x)=A\cos(\mathbf k\cdot\mathbf x-\omega t)$。直接计算
$$
(\Box-m^2)\phi
=(\omega^2-|\mathbf k|^2-m^2)\phi.
$$
因此非零平面波是经典解当且仅当
$\omega^2=|\mathbf k|^2+m^2$。正频支 $\omega=\sqrt{|\mathbf k|^2+m^2}$ 正是第九章一粒子能量 $E_{\mathbf k}$ 的来源；单个平面波不具 spacelike-compact 支撑，故不能直接代入命题 7.4 的 Cauchy 面积分，必须先组成具有适当衰减的波包。

## 练习

**练习 7.1.** 对复标量场
$$
\mathcal L=-\partial_\mu\phi^*\partial^\mu\phi-m^2\phi^*\phi
$$
的全局变换 $\phi\mapsto e^{i\alpha}\phi$，使用 $\delta\phi=i\phi$ 的约定推出守恒流。

**练习 7.2.** 计算自由 Maxwell 场的 Euler-Lagrange 方程，并说明规范变换不改变 $F=dA$。
