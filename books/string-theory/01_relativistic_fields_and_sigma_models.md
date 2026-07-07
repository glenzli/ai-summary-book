# 第一章：相对论场论、作用量和 sigma model 语言

## 本章目标

本章建立作用量、变分、Noether current、stress tensor 和 sigma model 的基本语言。后续 string worldsheet action 将被视为二维场论的特殊 sigma model；因此本章的重点不是一般场论百科，而是为世界面理论固定可复用的变分和张量规范。

## 依赖前置知识

需要多变量微积分、特殊相对论和经典场论基础。微分几何记号见附录 A。除非特别说明，本章 target metric 取 mostly plus，世界面 Lorentzian metric 取 $(-,+)$。

## 1.1 局部作用量和 Euler-Lagrange 方程

**定义 1.1A（局部作用量）.** 设 $N$ 是 $d$ 维时空或世界体，$\phi^i$ 是 $N$ 上的场。局部作用量写为
$$
S[\phi]=\int_N d^dx\,\mathcal L(\phi,\partial_\mu\phi,x).
$$
若 $N$ 有边界，则作用量变分一般包含 bulk term 与 boundary term。

**命题 1.1（Euler-Lagrange 方程）.** 若变分 $\delta\phi^i$ 在边界消失，则驻定条件 $\delta S=0$ 等价于
$$
\frac{\partial\mathcal L}{\partial\phi^i}
-
\partial_\mu\frac{\partial\mathcal L}{\partial(\partial_\mu\phi^i)}
=0.
$$

**证明.** 对 $\phi^i\mapsto\phi^i+\epsilon\delta\phi^i$ 求一阶变分：
$$
\delta S
=
\int_N d^dx\left(
\frac{\partial\mathcal L}{\partial\phi^i}\delta\phi^i
+\frac{\partial\mathcal L}{\partial(\partial_\mu\phi^i)}
\partial_\mu\delta\phi^i
\right).
$$
分部积分得
$$
\delta S
=
\int_N d^dx\left(
\frac{\partial\mathcal L}{\partial\phi^i}
-
\partial_\mu\frac{\partial\mathcal L}{\partial(\partial_\mu\phi^i)}
\right)\delta\phi^i
+\int_{\partial N}d\Sigma_\mu\,
\frac{\partial\mathcal L}{\partial(\partial_\mu\phi^i)}\delta\phi^i.
$$
边界项因假设消失。由 $\delta\phi^i$ 任意性得到方程。$\square$

**定义 1.2（良定变分问题）.** 一个作用量与边界条件组成良定变分问题，若所有允许变分下驻定条件等价于 bulk equations 加指定边界条件，且没有未控制的边界项。

**例 1.3（自由标量）.** 对
$$
S=-\frac12\int d^dx\,\partial_\mu\phi\,\partial^\mu\phi
$$
有
$$
\delta S=\int d^dx\,(\Box\phi)\delta\phi
-\int_{\partial N}d\Sigma_\mu\,\partial^\mu\phi\,\delta\phi.
$$
若取 Dirichlet 条件 $\delta\phi|_{\partial N}=0$，或 Neumann 条件 $n_\mu\partial^\mu\phi|_{\partial N}=0$，变分问题良定。

## 1.2 Noether theorem

**定义 1.4A（作用量对称性）.** 若连续变换
$$
\delta_\epsilon\phi^i=\epsilon\Delta\phi^i
$$
使 Lagrangian density 的变化为全导数
$$
\delta_\epsilon\mathcal L=\epsilon\,\partial_\mu K^\mu,
$$
则称其为作用量对称性。

**命题 1.3（Noether current）.** 对每个作用量对称性，存在守恒流
$$
j^\mu=
\frac{\partial\mathcal L}{\partial(\partial_\mu\phi^i)}\Delta\phi^i-K^\mu
$$
满足在运动方程上
$$
\partial_\mu j^\mu=0.
$$

**证明.** 一方面，由对称性有 $\delta\mathcal L=\epsilon\partial_\mu K^\mu$。另一方面，直接变分给出
$$
\delta\mathcal L
=
\epsilon\left(
E_i(\mathcal L)\Delta\phi^i
+\partial_\mu
\left[
\frac{\partial\mathcal L}{\partial(\partial_\mu\phi^i)}
\Delta\phi^i
\right]\right),
$$
其中 $E_i(\mathcal L)$ 是 Euler-Lagrange expression。在运动方程 $E_i=0$ 上比较两式，得 $\partial_\mu j^\mu=0$。$\square$

**例 1.5（平移与 canonical stress tensor）.** 若场论在平坦时空中平移不变，则
$$
T^\mu_{\ \nu,\mathrm{can}}
=
\frac{\partial\mathcal L}{\partial(\partial_\mu\phi^i)}
\partial_\nu\phi^i-\delta^\mu_{\ \nu}\mathcal L
$$
满足 $\partial_\mu T^\mu_{\ \nu,\mathrm{can}}=0$。该张量不一定对称，也不一定是耦合到引力时的 stress tensor。

## 1.3 Hilbert stress tensor

**定义 1.6（Hilbert stress tensor）.** 若场论耦合到背景 metric $g_{\mu\nu}$，定义
$$
T_{\mu\nu}
=-\frac{2}{\sqrt{|g|}}\frac{\delta S}{\delta g^{\mu\nu}}.
$$
等价地，
$$
\delta S=-\frac12\int d^dx\,\sqrt{|g|}\,T_{\mu\nu}\delta g^{\mu\nu}.
$$

**命题 1.4（sigma model stress tensor）.** 对二维 sigma model
$$
S=\frac{1}{4\pi\alpha'}\int d^2\sigma\sqrt{|h|}\,
h^{ab}g_{\mu\nu}(X)\partial_aX^\mu\partial_bX^\nu,
$$
世界面 Hilbert stress tensor 为
$$
T_{ab}
=-\frac1{2\pi\alpha'}\left(
\partial_aX^\mu\partial_bX^\nu g_{\mu\nu}
-\frac12h_{ab}h^{cd}\partial_cX^\mu\partial_dX^\nu g_{\mu\nu}
\right)
$$
在 Euclidean sign convention 下成立。Lorentzian Polyakov 作用量采用本书第二章规范时，约束方程等价于括号内表达式为零。

**证明.** 使用
$$
\delta\sqrt{|h|}=-\frac12\sqrt{|h|}h_{ab}\delta h^{ab}
$$
和
$$
\delta(h^{ab}\partial_aX\cdot\partial_bX)
=\delta h^{ab}\partial_aX\cdot\partial_bX.
$$
代入 Hilbert stress tensor 定义即可。$\square$

**命题 1.7（Weyl invariance 与 tracelessness）.** 若二维作用量在局部 Weyl transformation
$$
h_{ab}\mapsto e^{2\omega}h_{ab}
$$
下不变，则经典 stress tensor 满足
$$
T^a_{\ a}=0.
$$

**证明.** Weyl 变分为
$$
\delta h^{ab}=-2\delta\omega\,h^{ab}.
$$
由 stress tensor 定义，
$$
\delta S=\int d^2\sigma\sqrt{|h|}\,\delta\omega\,T^a_{\ a}.
$$
任意 $\delta\omega$ 下 $\delta S=0$，故 $T^a_{\ a}=0$。$\square$

## 1.4 Sigma model with background fields

**定义 1.8（nonlinear sigma model）.** 二维 nonlinear sigma model 是以映射
$$
X:\Sigma\to M
$$
为场的二维场论。最小作用量为
$$
S_g[X]=\frac{1}{4\pi\alpha'}\int_\Sigma d^2\sigma\sqrt{|h|}\,
h^{ab}g_{\mu\nu}(X)\partial_aX^\mu\partial_bX^\nu.
$$

**定义 1.9（$B$-field 与 dilaton coupling）.** Euclidean worldsheet 上，string sigma model 还可包含
$$
S_B=\frac{i}{4\pi\alpha'}\int_\Sigma
B_{\mu\nu}(X)dX^\mu\wedge dX^\nu,
$$
和
$$
S_\Phi=\frac1{4\pi}\int_\Sigma d^2\sigma\sqrt h\,\Phi(X)R^{(2)}.
$$

**命题 1.10（constant dilaton 与 string coupling）.** 若 $\Phi(X)=\Phi_0$ 为常数，则闭合世界面 genus $g$ 的 dilaton coupling 给出因子
$$
e^{-\Phi_0\chi(\Sigma)}=g_s^{-\chi(\Sigma)}
=g_s^{2g-2},
$$
其中
$$
g_s=e^{\Phi_0},\qquad \chi(\Sigma)=2-2g.
$$

**证明.** Gauss-Bonnet theorem 给出
$$
\frac1{4\pi}\int_\Sigma\sqrt h\,R^{(2)}=\chi(\Sigma).
$$
Euclidean path integral 权重为 $e^{-S_\Phi}$，故得到 $e^{-\Phi_0\chi(\Sigma)}$。$\square$

**注 1.11（背景场作为耦合常数）.** String theory 中的 target-space fields $g_{\mu\nu}$、$B_{\mu\nu}$ 和 $\Phi$ 可被看作 worldsheet sigma model 的耦合常数。量子 Weyl invariance 对这些耦合常数施加 beta function 方程，低阶给出 target-space field equations。

## 本章小结

String worldsheet theory 是二维场论。作用量变分给出 bulk equations 和边界条件；metric 变分给出 stress tensor；Weyl invariance 给出 tracelessness；sigma model 的耦合常数正是 target-space 背景场。

## 练习

**练习 1.1.** 从一维作用量推导 Euler-Lagrange 方程。

**练习 1.2.** 对自由标量场计算 canonical stress tensor 和 Hilbert stress tensor，并说明二者的关系。

**练习 1.3.** 证明常 dilaton coupling 在 genus $g$ 闭合世界面上给出 $g_s^{2g-2}$。

