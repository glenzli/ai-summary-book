# 第四章 电磁场的协变形式

## 本章目标

本章在固定号差、指标位置与取向后，把四维 Maxwell 方程逐分量还原为三维方程，并核对电荷守恒、Lorentz 力、规范自由和波动方程的全部符号。

## 依赖前置知识

需要第一章的四矢量与张量。本章取 $x^0=t$、$(-,+,+,+)$ 号差、$\epsilon^{123}=+1$ 和 $\epsilon^{0123}=+1$，并使用 Heaviside--Lorentz 单位。

## 4.1 电磁势和场强张量

电磁四势写为

$$
A_\mu=(-\phi,\mathbf{A})
$$

或等价地 $A^\mu=(\phi,\mathbf{A})$，具体分量依赖号差约定。电磁场强张量定义为

$$
F_{\mu\nu}=\partial_\mu A_\nu-\partial_\nu A_\mu.
$$

它自动反对称：

$$
F_{\mu\nu}=-F_{\nu\mu}.
$$

在 $(-,+,+,+)$ 约定下，本章定义

$$
E_i=-\partial_i\phi-\partial_tA_i,
\qquad
B^i=\epsilon^{ijk}\partial_jA_k.
$$

于是

$$
F_{0i}=-E_i,
\qquad
F_{ij}=\epsilon_{ijk}B^k,
\qquad
F^{0i}=E^i,
\qquad
F^{ij}=\epsilon^{ijk}B_k.
$$

## 4.2 Maxwell 方程

电荷电流四矢量为

$$
j^\mu=(\rho,\mathbf{j}).
$$

Maxwell 方程的协变形式为

$$
\partial_\mu F^{\nu\mu}=j^\nu,
$$

以及

$$
\partial_{[\lambda}F_{\mu\nu]}=0.
$$

在给定势 $A$ 的坐标片上，后者由 $F=dA$ 和 $d^2=0$ 得到，称为 Bianchi 恒等式。反过来，闭形式 $F$ 只在可缩坐标片上必可写成 $dA$；全局势的存在还受上同调限制。

**命题 4.1.** 设 $\mathbf E,\mathbf B$ 为 $C^1$ 场，$\rho,\mathbf j$
连续，并按本章 $(-,+,+,+)$ 号差、$\epsilon^{123}=+1$ 与上述指标位置
组装成 $F^{\mu\nu},j^\mu$。则
$\partial_\mu F^{\nu\mu}=j^\nu$ 和
$\partial_{[\lambda}F_{\mu\nu]}=0$ 当且仅当满足三维形式

$$
\nabla\cdot\mathbf{E}=\rho,\qquad
\nabla\times\mathbf{B}-\frac{\partial\mathbf{E}}{\partial t}=\mathbf{j},
$$

和

$$
\nabla\cdot\mathbf{B}=0,\qquad
\nabla\times\mathbf{E}+\frac{\partial\mathbf{B}}{\partial t}=0.
$$

**证明.** 由 $F^{0i}=E^i$、$F^{i0}=-E^i$ 和 $F^{ij}=\epsilon^{ijk}B_k$，取 $\nu=0$ 得
$$
\partial_\mu F^{0\mu}=\partial_iE^i=\rho.
$$
取 $\nu=i$ 得
$$
\partial_\mu F^{i\mu}
=-\partial_tE^i+\epsilon^{ijk}\partial_jB_k=j^i,
$$
即 Ampere--Maxwell 方程。又因 $F_{0i}=-E_i$、$F_{ij}=\epsilon_{ijk}B^k$，Bianchi 恒等式的纯空间分量给出 $\nabla\cdot\mathbf B=0$；$(0,i,j)$ 分量给出
$$
\epsilon_{ijk}\partial_tB^k+\partial_iE_j-\partial_jE_i=0,
$$
与 $\epsilon^{\ell ij}/2$ 缩并即为 $\partial_t\mathbf B+\nabla\times\mathbf E=0$。$\square$

## 4.3 电荷守恒

对 Maxwell 方程取散度：

$$
\partial_\nu j^\nu
=\partial_\nu\partial_\mu F^{\nu\mu}.
$$

因为 $\partial_\mu\partial_\nu$ 对称而 $F^{\mu\nu}$ 反对称，右侧为零。故

$$
\partial_\mu j^\mu=0.
$$

这就是连续性方程

$$
\frac{\partial\rho}{\partial t}+\nabla\cdot\mathbf{j}=0.
$$

## 4.4 Lorentz 力

带电粒子的协变运动方程为

$$
m\frac{du^\mu}{d\tau}=qF^\mu{}_\nu u^\nu.
$$

右侧自动与 $u^\mu$ 正交，因为

$$
u_\mu F^\mu{}_\nu u^\nu=F_{\mu\nu}u^\mu u^\nu=0.
$$

在三维形式中，它给出

$$
\frac{d\mathbf{p}}{dt}=q(\mathbf{E}+\mathbf{v}\times\mathbf{B}),
$$

以及功率方程

$$
\frac{dE}{dt}=q\mathbf{E}\cdot\mathbf{v}.
$$

## 4.5 标量与赝标量不变量

电磁场有两个基本完全缩并量。第一项是整个 Lorentz 群
$O(1,3)$ 下的标量：

$$
F_{\mu\nu}F^{\mu\nu}=2(\mathbf{B}^2-\mathbf{E}^2),
$$

固定时空取向后，第二项为

$$
\tilde{F}_{\mu\nu}F^{\mu\nu}=-4\mathbf{E}\cdot\mathbf{B},
$$

其中

$$
\tilde{F}^{\mu\nu}=\frac12\epsilon^{\mu\nu\rho\sigma}F_{\rho\sigma}.
$$

$\tilde F_{\mu\nu}F^{\mu\nu}$ 在 proper Lorentz 群下不变，但在反转
四维取向的 improper Lorentz 变换下变号，因此严格说是 Lorentz
赝标量，而不是整个 $O(1,3)$ 下的标量。这两个量可用于判断是否存在某
惯性系使磁场或电场消失。例如若 $\mathbf{E}\cdot\mathbf{B}=0$ 且
$\mathbf{E}^2>\mathbf{B}^2$，存在惯性系使 $\mathbf{B}'=0$。

## 4.6 电磁场能动张量

电磁场能动张量为

$$
T^{\mu\nu}_{\mathrm{EM}}
=F^{\mu\alpha}F^\nu{}_\alpha
-\frac14\eta^{\mu\nu}F_{\alpha\beta}F^{\alpha\beta}.
$$

它满足

$$
\partial_\mu T^{\mu\nu}_{\mathrm{EM}}
=-F^\nu{}_\lambda j^\lambda.
$$

这说明电磁场自身能动量不单独守恒；它和带电物质之间交换能量动量。若把物质能动张量加入，总能动张量守恒：

$$
\partial_\mu(T^{\mu\nu}_{\mathrm{matter}}+T^{\mu\nu}_{\mathrm{EM}})=0.
$$

## 4.7 规范变换

势 $A_\mu$ 不是唯一的。若

$$
A_\mu\mapsto A_\mu+\partial_\mu\chi,
$$

则

$$
F_{\mu\nu}\mapsto F_{\mu\nu}.
$$

Lorenz 规范为

$$
\partial_\mu A^\mu=0.
$$

在该规范下，由
$$
\partial_\mu F^{\nu\mu}
=\partial^\nu(\partial_\mu A^\mu)-\Box A^\nu
$$
可见 Maxwell 方程变为波动方程

$$
\Box A^\nu=-j^\nu,
\qquad
\Box=\partial_\mu\partial^\mu=-\partial_t^2+\nabla^2.
$$

这里的负号由本章同时采用 $\partial_\mu F^{\nu\mu}=j^\nu$ 与 mostly-plus d'Alembert 算子决定。若再作规范变换，Lorenz 条件保持当且仅当 $\Box\chi=0$。

## 本章小结

协变 Maxwell 方程只有在 $F$ 的指标次序、四维取向和 $\Box$ 号差同时固定后才能无歧义地还原为三维公式。本章约定给出 $\partial_\mu F^{\nu\mu}=j^\nu$ 和 $\Box A^\nu=-j^\nu$；电荷守恒来自对称导数与反对称场强的缩并为零。

## 习题

1. 写出 $F_{\mu\nu}$ 的矩阵形式。
2. 从 $\partial_\mu F^{\nu\mu}=j^\nu$ 推导电荷守恒。
3. 证明 $F_{\mu\nu}F^{\mu\nu}$ 是 Lorentz 标量。
4. 验证 Lorentz 力四矢量与四速度正交。
5. 在 Lorenz 规范下推导 $\Box A^\nu=-j^\nu$，并说明该符号如何依赖本章的 Maxwell 指标次序与 $\Box$ 约定。
