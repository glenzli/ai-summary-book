# 第一章 Minkowski 时空与 Lorentz 几何

经典力学先给出空间和绝对时间，再描述物体如何运动；狭义相对论则先问哪些事件之间能够发生因果联系。回答这个问题的不是某个观察者记录的三维距离，而是四维时空间隔及其符号。间隔一旦固定，类时、类光和类空方向便区分开来，Lorentz 变换、固有时和四速度也不再是零散公式，而是保持同一几何结构的不同表现。本章从事件出发建立这套语言，并始终采用 $c=1$ 与 $(-,+,+,+)$ 号差；第零章关于分量、指标和单位的约定在此全部生效。

## 1.1 事件和时空

狭义相对论的基本对象不是“物体的位置随时间变化”，而是 **事件**。事件是时空中的一点，记为

$$
x^\mu=(t,x,y,z).
$$

在一个惯性系中，两个邻近事件的间隔定义为

$$
ds^2=\eta_{\mu\nu}dx^\mu dx^\nu
=-dt^2+dx^2+dy^2+dz^2.
$$

这里采用 $c=1$ 和号差 $(-,+,+,+)$。如果恢复 $c$，则

$$
ds^2=-c^2dt^2+d\mathbf{x}^2.
$$

**定义 1.1 (Minkowski 时空).** 四维实仿射空间连同非退化双线性型

$$
\eta=\operatorname{diag}(-1,1,1,1)
$$

称为 Minkowski 时空。其平移向量空间上的内积写为

$$
\langle u,v\rangle_\eta=\eta_{\mu\nu}u^\mu v^\nu.
$$

时空间隔的核心地位在于：不同惯性系可以给同一对事件分配不同的 $t,x,y,z$，但它们必须同意 $ds^2$。

## 1.2 因果分类

设 $\Delta x^\mu$ 是两个事件的分离向量。

- 若 $\eta_{\mu\nu}\Delta x^\mu\Delta x^\nu<0$，称为 **类时** 分离。
- 若 $\eta_{\mu\nu}\Delta x^\mu\Delta x^\nu=0$，称为 **类光** 分离。
- 若 $\eta_{\mu\nu}\Delta x^\mu\Delta x^\nu>0$，称为 **类空** 分离。

类时分离的事件可以被低于光速的观察者连接；类光分离可被光信号连接；类空分离不能被因果信号连接。

**定义 1.2 (固有时).** 对类时曲线 $x^\mu(\lambda)$，固有时满足

$$
d\tau^2=-ds^2=dt^2-d\mathbf{x}^2.
$$

若用坐标时间参数化，$\mathbf{v}=d\mathbf{x}/dt$，则

$$
d\tau=dt\sqrt{1-\mathbf{v}^2}.
$$

这个公式不是“运动的钟真的被某种力拖慢”，而是 Minkowski 几何中类时曲线长度的表达。

## 1.3 Lorentz 变换

**定义 1.3 (Lorentz 变换).** 线性映射 $\Lambda:\mathbb{R}^{1,3}\to\mathbb{R}^{1,3}$ 若满足

$$
\eta_{\rho\sigma}\Lambda^\rho{}_\mu\Lambda^\sigma{}_\nu=\eta_{\mu\nu},
$$

则称为 Lorentz 变换。矩阵形式为

$$
\Lambda^T\eta\Lambda=\eta.
$$

**命题 1.1 (间隔不变性).** 若 $x'^\mu=\Lambda^\mu{}_\nu x^\nu$ 且 $\Lambda$ 是 Lorentz 变换，则

$$
\eta_{\mu\nu}x'^\mu x'^\nu=\eta_{\mu\nu}x^\mu x^\nu.
$$

**证明.**

直接代入：

$$
\eta_{\mu\nu}x'^\mu x'^\nu
=\eta_{\mu\nu}\Lambda^\mu{}_\rho x^\rho \Lambda^\nu{}_\sigma x^\sigma
=\eta_{\rho\sigma}x^\rho x^\sigma,
$$

其中最后一步正是 $\Lambda^T\eta\Lambda=\eta$。证毕。

Lorentz 变换群有四个连通分支。物理上常取保持时间方向和空间定向的真正规范分支 $SO^+(1,3)$。

## 1.4 四矢量和张量

**定义 1.4 (四矢量).** 在惯性系变换下按

$$
V'^\mu=\Lambda^\mu{}_\nu V^\nu
$$

变换的对象称为反变四矢量。协变分量定义为

$$
V_\mu=\eta_{\mu\nu}V^\nu.
$$

内积

$$
V^\mu W_\mu=\eta_{\mu\nu}V^\mu W^\nu
$$

是 Lorentz 标量。

**定义 1.5 (张量).** 一个 $(r,s)$ 型张量在 Lorentz 变换下按

$$
T'^{\mu_1\cdots\mu_r}{}_{\nu_1\cdots\nu_s}
=
\Lambda^{\mu_1}{}_{\alpha_1}\cdots
\Lambda^{\mu_r}{}_{\alpha_r}
(\Lambda^{-1})^{\beta_1}{}_{\nu_1}\cdots
(\Lambda^{-1})^{\beta_s}{}_{\nu_s}
T^{\alpha_1\cdots\alpha_r}{}_{\beta_1\cdots\beta_s}
$$

变换。

狭义相对论的计算原则是：如果一个方程两侧都是同型张量，且在一个惯性系成立，则在所有惯性系成立。这就是协变性的实际意义。

## 1.5 世界线、四速度和四加速度

类时粒子的世界线可用固有时参数化：

$$
x^\mu=x^\mu(\tau).
$$

**定义 1.6 (四速度).**

$$
u^\mu=\frac{dx^\mu}{d\tau}.
$$

由于 $d\tau^2=-\eta_{\mu\nu}dx^\mu dx^\nu$，有

$$
u^\mu u_\mu=-1.
$$

若三速度为 $\mathbf{v}$，则

$$
u^\mu=\gamma(1,\mathbf{v}),\qquad
\gamma=\frac{1}{\sqrt{1-\mathbf{v}^2}}.
$$

四加速度定义为

$$
a^\mu=\frac{du^\mu}{d\tau}.
$$

由 $u^\mu u_\mu=-1$ 求导得

$$
u_\mu a^\mu=0.
$$

这说明四加速度总与四速度 Minkowski 正交。这里的正交不是欧氏正交，而是 $\eta$ 意义下的正交。

## 1.6 从间隔到相对论运动学

本章把狭义相对论建立为 Minkowski 几何。核心对象是事件、间隔、因果分类、Lorentz 变换和四矢量。后续所有“相对论效应”都将作为这些不变量结构的推论。

## 习题

1. 证明 Lorentz 变换的行列式满足 $\det\Lambda=\pm1$。
2. 设 $V^\mu$ 为类时四矢量，证明存在惯性系使其空间分量为零。
3. 对速度为 $\mathbf{v}$ 的粒子，直接验证 $u^\mu u_\mu=-1$。
4. 证明若 $k^\mu$ 是类光矢量，则任意 Lorentz 变换后仍为类光矢量。
