# 第七章 测地线、等效原理与局部惯性系

第六章给出了联络和曲率，但尚未把它们转化为可观测运动。自由落体提供这一步：测试粒子的世界线由度规作用量驻定，并在仿射参数下满足测地线方程。等效原理说明联络系数可在一点随坐标选择消失，却不能消去由曲率控制的潮汐相对加速度。本章从变分推导测地线，再用局部惯性系、引力红移和 Newton 极限区分“局部看似无引力”与“时空真正平直”。所用工具来自第五章的作用量和第六章的 Levi-Civita 联络。

## 7.1 自由落体的世界线

广义相对论的基本物理判断是：没有非引力外力的测试粒子沿时空测地线运动。对有质量粒子，作用量为

$$
S=-m\int d\tau
=-m\int\sqrt{-g_{\mu\nu}\dot{x}^\mu\dot{x}^\nu}\,d\lambda.
$$

平方根作用量重参数化不变。先对它变分，再沿所得类时驻定曲线选固有时
$\tau$；等价地，可在仿射参数下使用能量 Lagrangian

$$
L_{\mathrm{aff}}
=\frac12g_{\mu\nu}\dot x^\mu\dot x^\nu.
$$

其 Euler-Lagrange 方程为

$$
\frac{d}{d\tau}(g_{\alpha\nu}\dot x^\nu)
-\frac12\partial_\alpha g_{\rho\sigma}
\dot x^\rho\dot x^\sigma=0.
$$

展开第一项、利用 $\dot x^\rho\dot x^\sigma$ 的对称性，再乘以
$g^{\mu\alpha}$，得到

$$
\frac{d^2x^\mu}{d\tau^2}
+\Gamma^\mu{}_{\rho\sigma}
\frac{dx^\rho}{d\tau}
\frac{dx^\sigma}{d\tau}=0.
$$

这就是测地线方程。这里不能在变分前把平方根作用量中的参数机械地
固定为依赖待求曲线的固有时；上面的等价仿射 Lagrangian 避开了这一
循环。

## 7.2 测地线作为极值曲线

测地线首先是长度泛函的驻定曲线。位于凸正规邻域内的类时测地线段
局部极大化固有时；更长的测地线越过共轭点后一般不再具有这一极大性。
类空测地线局部驻定弧长，但不具有无条件的极小或极大结论。类光测地线满足

$$
g_{\mu\nu}\dot{x}^\mu\dot{x}^\nu=0
$$

并用仿射参数满足同一测地线方程。

对光线，不能用 $d\tau$ 作参数；但可用等价作用量

$$
S=\frac12\int e^{-1}g_{\mu\nu}\dot{x}^\mu\dot{x}^\nu\,d\lambda
$$

其中 $e(\lambda)$ 是辅助 einbein。对 $e$ 变分给出类光约束。

## 7.3 等效原理

Einstein 等效原理可表述为：

1. 自由落体测试粒子在局部不可区分于惯性运动。
2. 非引力局部实验在足够小的自由落体实验室中满足狭义相对论。
3. 引力效应可在一点用局部惯性坐标消去，但曲率不能在有限区域消去。

因此，引力不再是 Minkowski 时空中的普通力，而是时空几何本身的表现。

## 7.4 局部惯性系与潮汐力

在点 $p$ 可取正规坐标使

$$
g_{\mu\nu}(p)=\eta_{\mu\nu},
\qquad
\Gamma^\rho{}_{\mu\nu}(p)=0.
$$

此时测地线方程在 $p$ 处退化为

$$
\frac{d^2x^\mu}{d\tau^2}=0.
$$

但邻近测地线之间的相对加速度由曲率控制。测地线偏离方程为

$$
\frac{D^2\xi^\mu}{d\tau^2}
=-R^\mu{}_{\nu\rho\sigma}u^\nu\xi^\rho u^\sigma,
$$

其中 $\xi^\mu$ 是两条邻近测地线之间的分离向量。右侧就是潮汐力的几何形式。

## 7.5 红移

静态时空中若存在时间 Killing 矢量 $\xi^\mu=(\partial_t)^\mu$，光子四动量 $k^\mu$ 满足

$$
E=-\xi_\mu k^\mu
$$

沿测地线守恒。静止观察者四速度为

$$
u^\mu=\frac{\xi^\mu}{\sqrt{-\xi^\nu\xi_\nu}}.
$$

其测得频率为

$$
\omega=-u_\mu k^\mu
=\frac{E}{\sqrt{-g_{tt}}}.
$$

因此两处静止观察者测得频率满足

$$
\frac{\omega_1}{\omega_2}
=
\sqrt{\frac{-g_{tt}(x_2)}{-g_{tt}(x_1)}}.
$$

弱场中 $g_{tt}\approx-(1+2\Phi)$，故

$$
\frac{\Delta\omega}{\omega}\approx \Phi_2-\Phi_1.
$$

这就是引力红移。

## 7.6 Newton 极限中的测地线

设弱场慢速度规为

$$
g_{tt}=-(1+2\Phi),\qquad
g_{ij}\approx\delta_{ij},\qquad
|\Phi|\ll1.
$$

Christoffel 符号

$$
\Gamma^i{}_{tt}
=-\frac12 g^{ij}\partial_j g_{tt}
\approx \partial_i\Phi.
$$

低速测地线方程中 $dt/d\tau\approx1$，得到

$$
\frac{d^2x^i}{dt^2}
+\Gamma^i{}_{tt}\approx0,
$$

即

$$
\frac{d^2\mathbf{x}}{dt^2}=-\nabla\Phi.
$$

Newton 引力成为弱场慢速极限下的测地线方程。

## 习题

1. 从类时粒子作用量推导测地线方程。
2. 证明测地线方程在仿射参数变换 $\lambda\mapsto a\lambda+b$ 下形式不变。
3. 用正规坐标解释为什么 Christoffel 符号可以在一点消去。
4. 从测地线偏离方程解释潮汐力。
5. 推导弱场引力红移公式。
