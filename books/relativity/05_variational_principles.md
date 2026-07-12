# 第五章 作用量、Noether 定理与场论入口

## 本章目标

本章在明确允许变分与边界条件后推导自由粒子、带电粒子和实标量场方程，并证明平移不变场论的规范能动张量在 on-shell 意义下守恒。

## 依赖前置知识

需要第零章的变分基本引理、第三章的四动量和第四章的电磁势。本章所有世界线变分固定端点；场变分取紧支撑，或在边界上为零。

## 5.1 为什么需要变分原理

狭义相对论用 Lorentz 协变性限制物理定律的形式。变分原理进一步给出统一生成机制：粒子轨道、场方程、守恒律和能动张量都可由作用量得到。

一个物理系统的作用量写为

$$
S=\int \mathcal{L}\,d^4x
$$

或对点粒子写为沿世界线的积分。真实运动使 $S$ 在允许变分下取驻值。

## 5.2 自由粒子

设 $m>0$，并令 $x:[\lambda_A,\lambda_B]\to\mathbb R^{1,3}$ 为分段光滑、未来指向类时曲线。自由粒子作用量

$$
S=-m\int d\tau
$$

是 Lorentz 不变量，因为 $d\tau$ 是世界线长度元。

取任意参数 $\lambda$：

$$
S=-m\int\sqrt{-\eta_{\mu\nu}\dot{x}^\mu\dot{x}^\nu}\,d\lambda.
$$

Euler-Lagrange 方程为

$$
\frac{d}{d\lambda}
\left(
\frac{m\eta_{\mu\nu}\dot{x}^\nu}
{\sqrt{-\dot{x}^2}}
\right)=0.
$$

若取 $\lambda=\tau$，则

$$
\frac{d}{d\tau}(m u_\mu)=0,
$$

即四动量守恒。自由粒子世界线是 Minkowski 时空中的直线。

## 5.3 带电粒子与电磁耦合

带电粒子作用量为

$$
S=-m\int d\tau+q\int A_\mu dx^\mu.
$$

第二项在规范变换下变化为

$$
q\int \partial_\mu\chi\,dx^\mu
=q\int d\chi
=q(\chi_B-\chi_A).
$$

端点固定时这不影响运动方程。若端点不固定或世界线有边界自由度，该边界项不能直接丢弃。

变分得到

$$
m\frac{du_\mu}{d\tau}=qF_{\mu\nu}u^\nu,
$$

等价于上一章的 Lorentz 力方程。

## 5.4 标量场例子

实标量场的作用量可写为

$$
S[\phi]=\int
\left(
-\frac12\partial_\mu\phi\,\partial^\mu\phi
-\frac12m^2\phi^2
\right)d^4x.
$$

变分：

$$
\delta S=\int
\left(
-\partial_\mu\phi\,\partial^\mu\delta\phi
-m^2\phi\,\delta\phi
\right)d^4x.
$$

对紧支撑变分 $\delta\phi$ 分部积分，边界项为零，得到

$$
\delta S=\int
(\partial_\mu\partial^\mu\phi-m^2\phi)\delta\phi\,d^4x.
$$

所以场方程为

$$
(\Box-m^2)\phi=0.
$$

这就是 Klein-Gordon 方程。

## 5.5 Noether 定理的最小形式

若场论作用量在连续变换下不变，则在场方程成立时存在守恒流。为固定
整体号差，本书把常参数平移写成
$$
x^\mu\longmapsto x^\mu+\epsilon^\mu,
\qquad
\delta_\epsilon\phi^a(x)=-\epsilon^\nu\partial_\nu\phi^a(x),
$$
并约定相应 Noether 流为
$j^\mu(\epsilon)=\epsilon^\nu\Theta^\mu{}_\nu$。对无显式坐标依赖的
Lagrangian 密度，规范能动张量（canonical stress-energy tensor）定义为
$$
\Theta^\mu{}_\nu
=-\sum_a\frac{\partial\mathcal L}
{\partial(\partial_\mu\phi^a)}\partial_\nu\phi^a
+\delta^\mu{}_\nu\mathcal L.
$$
这个号差使 $\Theta^\mu{}_\nu$ 与本书采用的 Hilbert 能动张量约定相容。
若把主动场变分改写成相反号，则 Noether 流也整体变号，不能只翻转上式
的一部分。

**命题 5.1（平移 Noether 恒等式）.** 设
$\mathcal L(\phi^a,\partial_\mu\phi^a)$ 光滑且无显式 $x$ 依赖，并按
上式定义 $\Theta^\mu{}_\nu$。则恒等式
$$
\partial_\mu \Theta^\mu{}_\nu
=\sum_a\mathcal E_a(\phi)\,\partial_\nu\phi^a
$$
成立，其中
$$
\mathcal E_a(\phi)
=\frac{\partial\mathcal L}{\partial\phi^a}
-\partial_\mu
\frac{\partial\mathcal L}{\partial(\partial_\mu\phi^a)}.
$$
因此场方程 $\mathcal E_a=0$ 成立时
$$
\partial_\mu \Theta^\mu{}_\nu=0.
$$

**证明.** 对定义逐项求导：
$$
\begin{aligned}
\partial_\mu \Theta^\mu{}_\nu
={}&-\sum_a\partial_\mu
\left(\frac{\partial\mathcal L}
{\partial(\partial_\mu\phi^a)}\right)\partial_\nu\phi^a\\
&-\sum_a\frac{\partial\mathcal L}
{\partial(\partial_\mu\phi^a)}\partial_\mu\partial_\nu\phi^a
+\partial_\nu\mathcal L.
\end{aligned}
$$
由于 $\mathcal L$ 无显式坐标依赖，链式法则给出
$$
\partial_\nu\mathcal L
=\sum_a\frac{\partial\mathcal L}{\partial\phi^a}\partial_\nu\phi^a
+\sum_a\frac{\partial\mathcal L}
{\partial(\partial_\mu\phi^a)}\partial_\nu\partial_\mu\phi^a.
$$
后二阶导数项相消，余下正是
$\sum_a\mathcal E_a\partial_\nu\phi^a$。$\square$

对本章实标量场，
$\partial\mathcal L/\partial(\partial_\mu\phi)=-\partial^\mu\phi$，故
$$
\Theta^\mu{}_\nu
=\partial^\mu\phi\,\partial_\nu\phi
+\delta^\mu{}_\nu\mathcal L.
$$
另一方面，把作用量最小耦合到度规，
$$
S_{\mathrm{matter}}[g,\phi]
=\int\left(-\frac12g^{\rho\sigma}\partial_\rho\phi\partial_\sigma\phi
-\frac12m^2\phi^2\right)\sqrt{-g}\,d^4x,
$$
则第 5.6 节采用的 Hilbert 定义给出
$$
T_{\mu\nu}
=\partial_\mu\phi\,\partial_\nu\phi+g_{\mu\nu}\mathcal L.
$$
回到 $g=\eta$ 并升起第一个指标，得到
$T^\mu{}_\nu=\Theta^\mu{}_\nu$。特别地，
$$
T^{00}
=\frac12(\partial_t\phi)^2+\frac12|\nabla\phi|^2+\frac12m^2\phi^2,
$$
这也交叉检查了整体号差。一般场论的规范能动张量不自动对称或规范
不变；它与 Hilbert 张量的比较可能需要 Belinfante--Rosenfeld 改进项，
不能从这个标量场例子无条件推广。

## 5.6 从平直到弯曲

GR 中的关键替换不是机械地把 $\eta_{\mu\nu}$ 改成 $g_{\mu\nu}$。更准确地说：

1. 时空几何由 Lorentz 度规 $g_{\mu\nu}$ 描述。
2. 普通导数替换为与 $g$ 相容的协变导数。
3. 积分体积元替换为 $\sqrt{-g}\,d^4x$。
4. 物质能动张量可由对度规变分定义：

$$
T_{\mu\nu}
=-\frac{2}{\sqrt{-g}}
\frac{\delta S_{\mathrm{matter}}}{\delta g^{\mu\nu}}.
$$

这一公式是 SR 到 GR 的桥。Einstein 方程左边是几何，右边正是物质作用量对几何的响应。

## 5.7 本章小结

本章建立了三条后续主线：

- 粒子世界线可由作用量推导。
- 连续对称性给出守恒律。
- 能动张量是物质对度规的变分响应。

这些概念将在广义相对论中升级为测地线、协变守恒和 Einstein 方程。

## 习题

1. 对自由粒子作用量，在 $\lambda=t$ 下重新推导相对论动量。
2. 对带电粒子作用量做变分，写出得到 Lorentz 力的关键步骤。
3. 推导 Klein-Gordon 方程。
4. 计算实标量场的规范能动张量。
5. 说明规范变换为何不改变带电粒子运动方程。
