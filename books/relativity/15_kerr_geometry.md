# 第十五章 Kerr 几何、旋转黑洞与高级黑洞入口

Schwarzschild 解描述静态球对称黑洞。真实天体通常有角动量，因此旋转黑洞必须进入正式广义相对论教材的内容范围。本章不从 Einstein 方程完整求解 Kerr 度规；那需要复杂的代数和对称性理论。但本章会给出 Kerr 几何的基本对象、视界结构、能层、守恒量和测地线可分离性，让读者能读懂旋转黑洞的标准公式。

## 15.1 Boyer-Lindquist 坐标下的 Kerr 度规

取自然单位 $G=c=1$。Kerr 度规在 Boyer-Lindquist 坐标 $(t,r,\theta,\phi)$ 中写为

$$
ds^2
=-\left(1-\frac{2Mr}{\Sigma}\right)dt^2
-\frac{4Mar\sin^2\theta}{\Sigma}dt\,d\phi
+\frac{\Sigma}{\Delta}dr^2
+\Sigma d\theta^2
+\left(
r^2+a^2+\frac{2Ma^2r\sin^2\theta}{\Sigma}
\right)\sin^2\theta\,d\phi^2,
$$

其中

$$
\Sigma=r^2+a^2\cos^2\theta,
\qquad
\Delta=r^2-2Mr+a^2.
$$

参数 $M$ 是质量，$J=Ma$ 是角动量。若 $a=0$，则退化为 Schwarzschild 度规。

## 15.2 视界

**命题 15.1（Kerr 视界位置）.** 在 Boyer-Lindquist 坐标中，Kerr 度规的 Killing 视界候选位置由

$$
\Delta=0
$$

给出，即

$$
r_\pm=M\pm\sqrt{M^2-a^2}.
$$

**证明.**

坐标分量 $g^{rr}$ 满足

$$
g^{rr}=\frac{\Delta}{\Sigma}.
$$

因此 $\Delta=0$ 给出候选视界位置：

$$
r_\pm=M\pm\sqrt{M^2-a^2}.
$$

若 $|a|<M$，存在外视界 $r_+$ 和内视界 $r_-$；若 $|a|=M$，称为极端 Kerr 黑洞；若 $|a|>M$，没有视界，裸奇点出现。

证明完毕。$\square$

**定义 15.1（宇宙审查语境）.** 裸奇点是未被事件视界遮蔽的奇点。经典宇宙审查猜想粗略地说，合理引力塌缩不会产生可从无穷远看见的裸奇点。该猜想不是已完全证明的定理。

## 15.3 能层与静止极限面

Kerr 度规的 $g_{tt}$ 为

$$
g_{tt}
=-\left(1-\frac{2Mr}{\Sigma}\right).
$$

静止观察者需要沿 $\partial_t$ 方向运动。当 $g_{tt}=0$ 时，$\partial_t$ 变成类光方向。解得静止极限面

$$
r_{\mathrm{stat}}(\theta)
=M+\sqrt{M^2-a^2\cos^2\theta}.
$$

外视界 $r_+$ 与静止极限面之间的区域称为能层 ergoregion。在能层内，任何未来指向类时世界线都必须随黑洞旋转方向拖曳。

## 15.4 拖曳惯性系

Kerr 度规含有交叉项 $g_{t\phi}$。这表示时间平移和绕轴旋转不再正交。零角动量观察者的角速度为

$$
\omega
=-\frac{g_{t\phi}}{g_{\phi\phi}}.
$$

在弱场远区，主项为

$$
\omega\sim \frac{2J}{r^3}.
$$

这就是 Lense-Thirring 拖曳效应的几何来源。

## 15.5 Killing 向量与守恒量

**命题 15.2（Killing 守恒量）.** 若 $K^\mu$ 是 Killing 向量场，则 $K_\mu p^\mu$ 沿仿射测地线守恒。

Kerr 时空是定常且轴对称的，因此有两个 Killing 向量：

$$
\xi^\mu=(\partial_t)^\mu,
\qquad
\psi^\mu=(\partial_\phi)^\mu.
$$

对测地线四动量 $p^\mu$，定义

$$
E=-\xi_\mu p^\mu,
\qquad
L_z=\psi_\mu p^\mu.
$$

沿测地线有

$$
\frac{dE}{d\lambda}=0,
\qquad
\frac{dL_z}{d\lambda}=0.
$$

**证明.** 对 Killing 向量 $K^\mu$，

$$
\nabla_{(\mu}K_{\nu)}=0.
$$

沿仿射测地线，

$$
\frac{d}{d\lambda}(K_\mu p^\mu)
=p^\nu\nabla_\nu(K_\mu p^\mu)
=p^\nu p^\mu\nabla_\nu K_\mu
+K_\mu p^\nu\nabla_\nu p^\mu.
$$

第二项由测地线方程为零；第一项中 $p^\nu p^\mu$ 对称，而 $\nabla_\nu K_\mu$ 的对称部分为零，因此为零。$\square$

## 15.6 Carter 常数与可分离性

Kerr 测地线除 $E,L_z$ 和质量壳条件外，还有 Carter 常数 $Q$。这来自隐藏对称性，即一个二阶 Killing 张量。

Hamilton-Jacobi 方程为

$$
g^{\mu\nu}\partial_\mu S\partial_\nu S=-m^2.
$$

在 Kerr 几何中可用分离形式

$$
S=-Et+L_z\phi+S_r(r)+S_\theta(\theta)+\frac12m^2\lambda.
$$

代入后方程分离为只含 $r$ 的部分和只含 $\theta$ 的部分，分离常数就是 Carter 常数。

本书不完整推导所有代数式，但记录测地线方程的标准结构：

$$
\Sigma\frac{dr}{d\lambda}
=\pm\sqrt{R(r)},
\qquad
\Sigma\frac{d\theta}{d\lambda}
=\pm\sqrt{\Theta(\theta)}.
$$

其中 $R$ 和 $\Theta$ 由 $E,L_z,Q,m,M,a$ 决定。

## 15.7 Penrose 过程

在能层中，$\partial_t$ 可变为空间方向，因此粒子能量

$$
E=-p_t
$$

可以为负。设一个粒子进入能层后分裂为两个粒子，其中一个带负能量落入黑洞，另一个逃到无穷远。逃逸粒子的能量可大于原粒子能量，差额来自黑洞自转能。

这个机制称为 Penrose 过程。其数学核心不是“凭空产生能量”，而是定常 Killing 能量在能层中不再对所有未来类时动量正定。

## 15.8 面积定理和不可逆质量

Kerr 黑洞视界面积为

$$
A=8\pi M r_+.
$$

定义不可逆质量 $M_{\mathrm{irr}}$：

$$
A=16\pi M_{\mathrm{irr}}^2.
$$

于是

$$
M_{\mathrm{irr}}^2=\frac12Mr_+.
$$

经典面积定理在适当能量条件下表明黑洞事件视界面积不减。它是黑洞热力学的经典基础之一。本书把面积定理作为外部输入，不证明其全局因果几何细节。

## 15.9 本章边界

Kerr 几何是现代黑洞物理的核心，但完整理论包括：

- Kerr 解的求解和代数分类。
- 主类光方向和 Petrov D 型。
- Teukolsky 方程和黑洞微扰。
- 准正规模和 ringdown 波形。
- Kerr 稳定性。

这些主题超过第一门 GR 教材的内部证明范围，但第十五章的内容足以连接 Schwarzschild 黑洞和现代旋转黑洞物理。

## 习题

1. 证明 $a=0$ 时 Kerr 度规退化为 Schwarzschild 度规。
2. 解 $\Delta=0$ 并讨论 $|a|<M, |a|=M, |a|>M$ 三种情形。
3. 推导静止极限面 $r_{\mathrm{stat}}(\theta)$。
4. 证明 Killing 向量给出沿测地线守恒量。
5. 解释为什么能层内可以出现负 Killing 能量。
