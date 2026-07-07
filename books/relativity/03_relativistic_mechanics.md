# 第三章 相对论力学与应力能量张量

## 3.1 四动量

质量为 $m>0$ 的自由粒子四动量定义为

$$
p^\mu=mu^\mu.
$$

由 $u^\mu u_\mu=-1$ 得

$$
p^\mu p_\mu=-m^2.
$$

在惯性系中

$$
p^\mu=(E,\mathbf{p})=(\gamma m,\gamma m\mathbf{v}).
$$

于是得到能量动量关系

$$
E^2-\mathbf{p}^2=m^2.
$$

恢复光速为

$$
E^2=p^2c^2+m^2c^4.
$$

静止时 $\mathbf{p}=0$，故 $E=mc^2$。

## 3.2 质量壳和无质量粒子

有质量粒子的四动量位于质量壳

$$
p^2=-m^2,\qquad p^0>0.
$$

无质量粒子满足

$$
p^\mu p_\mu=0,\qquad E=|\mathbf{p}|.
$$

无质量粒子没有静止系，因为类光四矢量不能通过 Lorentz 变换变成纯时间方向。

## 3.3 四力

四力定义为

$$
f^\mu=\frac{dp^\mu}{d\tau}.
$$

若 $m$ 恒定，则

$$
p_\mu f^\mu=m^2 u_\mu a^\mu=0.
$$

四力与四动量 Minkowski 正交。三维力 $\mathbf{F}=d\mathbf{p}/dt$ 与四力的关系为

$$
f^\mu=\gamma\left(\frac{dE}{dt},\frac{d\mathbf{p}}{dt}\right)
=\gamma(\mathbf{F}\cdot\mathbf{v},\mathbf{F}).
$$

## 3.4 粒子作用量和 Euler-Lagrange 方程

自由粒子的作用量为

$$
S=-m\int d\tau
=-m\int \sqrt{-\eta_{\mu\nu}\dot{x}^\mu\dot{x}^\nu}\,d\lambda.
$$

若取 $\lambda=t$，则

$$
L=-m\sqrt{1-\mathbf{v}^2}.
$$

正则动量为

$$
\mathbf{p}=\frac{\partial L}{\partial \mathbf{v}}
=\frac{m\mathbf{v}}{\sqrt{1-\mathbf{v}^2}}
=\gamma m\mathbf{v}.
$$

Hamilton 量为

$$
H=\mathbf{p}\cdot\mathbf{v}-L=\gamma m=E.
$$

因此相对论能量不是另行假设，而是自由粒子作用量的 Hamilton 量。

## 3.5 连续介质和能动张量

孤立粒子的四动量守恒不足以描述场和流体。局域守恒量由能动张量 $T^{\mu\nu}$ 给出。

在平直时空中，局域能量动量守恒写为

$$
\partial_\mu T^{\mu\nu}=0.
$$

总四动量定义为

$$
P^\nu=\int_{\Sigma_t}T^{0\nu}\,d^3x.
$$

若场在空间无穷远足够快衰减，则

$$
\frac{dP^\nu}{dt}
=\int \partial_0T^{0\nu}\,d^3x
=-\int \partial_iT^{i\nu}\,d^3x
=-\int_{\partial\Sigma}T^{i\nu}n_i\,dS
=0.
$$

所以局域守恒推出整体守恒。

## 3.6 完美流体

完美流体在自身局部静止系中有能量密度 $\rho$ 和各向同性压强 $p$。其能动张量为

$$
T^{\mu\nu}=(\rho+p)u^\mu u^\nu+p\eta^{\mu\nu}.
$$

在静止系 $u^\mu=(1,0,0,0)$ 中，

$$
T^{00}=\rho,\qquad T^{ij}=p\delta^{ij},\qquad T^{0i}=0.
$$

这个形式在 GR 中仍成立，只需把 $\eta^{\mu\nu}$ 换成 $g^{\mu\nu}$。

## 3.7 碰撞和阈值

相对论碰撞计算应优先使用不变量。例如两粒子总四动量

$$
P^\mu=p_1^\mu+p_2^\mu
$$

的平方

$$
s=-(P^\mu P_\mu)
$$

是 Lorentz 不变量，称为质心能量平方。在实验室系和质心系之间切换时，使用 $s$ 可以避免繁琐的速度变换。

## 习题

1. 从 $E^2-\mathbf{p}^2=m^2$ 推导 $v=|\mathbf{p}|/E$。
2. 证明无质量粒子的速度大小为 $1$。
3. 对自由粒子 Lagrangian $L=-m\sqrt{1-\mathbf{v}^2}$，求 Hamilton 量。
4. 对完美流体能动张量，验证静止系中的分量。
5. 设两个同质量粒子在质心系中反向等速运动，计算 $s$。
