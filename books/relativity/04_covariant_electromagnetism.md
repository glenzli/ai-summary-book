# 第四章 电磁场的协变形式

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

在 $(-,+,+,+)$ 约定下，可取

$$
F^{0i}=E^i,\qquad F^{ij}=-\epsilon^{ijk}B_k.
$$

## 4.2 Maxwell 方程

电荷电流四矢量为

$$
j^\mu=(\rho,\mathbf{j}).
$$

Maxwell 方程的协变形式为

$$
\partial_\mu F^{\mu\nu}=j^\nu,
$$

以及

$$
\partial_{[\lambda}F_{\mu\nu]}=0.
$$

后者等价于 $F=dA$ 的恒等式，称为 Bianchi 恒等式。

**命题 4.1.** 协变 Maxwell 方程等价于三维形式

$$
\nabla\cdot\mathbf{E}=\rho,\qquad
\nabla\times\mathbf{B}-\frac{\partial\mathbf{E}}{\partial t}=\mathbf{j},
$$

和

$$
\nabla\cdot\mathbf{B}=0,\qquad
\nabla\times\mathbf{E}+\frac{\partial\mathbf{B}}{\partial t}=0.
$$

**证明.** 取 $\nu=0$ 得 Gauss 定律；取 $\nu=i$ 得 Ampere-Maxwell 定律。对 $\partial_{[\lambda}F_{\mu\nu]}=0$ 分别取全空间指标和一个时间两个空间指标，得到无磁单极和 Faraday 定律。证毕。

## 4.3 电荷守恒

对 Maxwell 方程取散度：

$$
\partial_\nu j^\nu
=\partial_\nu\partial_\mu F^{\mu\nu}.
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

## 4.5 两个电磁不变量

电磁场有两个基本 Lorentz 标量：

$$
F_{\mu\nu}F^{\mu\nu}=2(\mathbf{B}^2-\mathbf{E}^2),
$$

和

$$
\tilde{F}_{\mu\nu}F^{\mu\nu}=-4\mathbf{E}\cdot\mathbf{B},
$$

其中

$$
\tilde{F}^{\mu\nu}=\frac12\epsilon^{\mu\nu\rho\sigma}F_{\rho\sigma}.
$$

这两个量可用于判断是否存在某惯性系使磁场或电场消失。例如若 $\mathbf{E}\cdot\mathbf{B}=0$ 且 $\mathbf{E}^2>\mathbf{B}^2$，存在惯性系使 $\mathbf{B}'=0$。

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

在该规范下，Maxwell 方程变为波动方程

$$
\Box A^\nu=j^\nu,
\qquad
\Box=\partial_\mu\partial^\mu=-\partial_t^2+\nabla^2.
$$

## 习题

1. 写出 $F_{\mu\nu}$ 的矩阵形式。
2. 从 $\partial_\mu F^{\mu\nu}=j^\nu$ 推导电荷守恒。
3. 证明 $F_{\mu\nu}F^{\mu\nu}$ 是 Lorentz 标量。
4. 验证 Lorentz 力四矢量与四速度正交。
5. 在 Lorenz 规范下推导 $\Box A^\nu=j^\nu$。
