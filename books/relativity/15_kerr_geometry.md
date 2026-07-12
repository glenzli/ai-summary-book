# 第十五章 Kerr 几何、旋转黑洞与高级黑洞入口

Schwarzschild 解描述静态球对称黑洞。真实天体通常有角动量，因此旋转黑洞必须进入正式广义相对论教材的内容范围。本章不从 Einstein 方程完整求解 Kerr 度规；那需要复杂的代数和对称性理论。但本章会给出 Kerr 几何的基本对象、视界结构、能层、守恒量和测地线可分离性，让读者能读懂旋转黑洞的标准公式。

## 本章目标

本章区分 Kerr 度规中的局部代数条件、Killing 视界与全局事件视界，推导显式 Killing 守恒量，并把 Carter 可分离性与面积定理准确列为外部输入。

## 依赖前置知识

需要第六章的 Killing 场、第七章的仿射测地线、第十章的事件视界全局定义以及第十三章的能量条件。

## 15.1 Boyer-Lindquist 坐标下的 Kerr 度规

取自然单位 $G=c=1$，并假设 $M>0$。本章的 $M$ 是几何化质量、
$J=Ma$ 是几何化角动量。若物理质量和角动量记为
$M_{\mathrm{phys}},J_{\mathrm{phys}}$，则

$$
M=\frac{GM_{\mathrm{phys}}}{c^2},
\qquad
J=\frac{GJ_{\mathrm{phys}}}{c^3},
\qquad
a=\frac{J_{\mathrm{phys}}}{M_{\mathrm{phys}}c}.
$$

Kerr 度规在 Boyer-Lindquist 坐标 $(t,r,\theta,\phi)$ 中写为

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

若 $a=0$，则退化为几何质量为 $M$ 的 Schwarzschild 度规，即第九章
公式中的 $GM$ 在本章记成 $M$。

## 15.2 视界

**命题 15.1（Kerr 候选视界根的局部代数）.** 设 $M>0$。在
Boyer-Lindquist 坐标中，方程

$$
\Delta=0
$$

的实代数根存在当且仅当 $|a|\le M$，并由

$$
r_\pm=M\pm\sqrt{M^2-a^2}.
$$

给出。外根 $r=r_+$ 上的 Killing 场
$$
\chi_+=\partial_t+\Omega_+\partial_\phi,
\qquad
\Omega_+=\frac{a}{r_+^2+a^2}
$$
的范数为零。若 $0<|a|<M$，内根 $r_-$ 也满足同一结论，其中
$\chi_-$ 与 $\Omega_-$ 由把 $+$ 换成 $-$ 得到。若 $|a|=M$，两根
重合，只得到一个退化候选视界；若 $a=0$，下方代数根 $r_-=0$ 位于
$\Sigma=0$ 的 Schwarzschild 曲率奇点，不能当作内 Killing 视界。

**证明.** 二次方程 $r^2-2Mr+a^2=0$ 的判别式为 $4(M^2-a^2)$，立即给出根的存在条件与公式。坐标分量 $g^{rr}$ 满足

$$
g^{rr}=\frac{\Delta}{\Sigma}.
$$

故在 $\Sigma\ne0$ 的根上，$r=\mathrm{const}$ 是法余向量 $dr$
变为类光的候选超曲面。又由 $\Delta(r_\pm)=0$ 得
$$
r_\pm^2+a^2=2Mr_\pm.
$$
对 $r_\pm^2+a^2>0$ 的根，把
$\Omega_\pm=a/(r_\pm^2+a^2)$ 代入
$$
g(\chi_\pm,\chi_\pm)
=g_{tt}+2\Omega_\pm g_{t\phi}+\Omega_\pm^2g_{\phi\phi}
$$
并使用上式，得到其在 $r=r_\pm$ 上为零。唯一被这一步排除的
$|a|\le M$ 代数根是 $a=0,r_-=0$，此时分母与 $\Sigma$ 同时退化，
也正是命题已单独说明的曲率奇点。$\square$

**外部输入定理 15.2（Kerr 的全局视界识别）.** 对
$0<|a|<M$ 的标准最大 Kerr 延拓，Boyer-Lindquist 坐标在
$\Delta=0$ 失效，但两个根在 horizon-penetrating 坐标中光滑延拓为
由 $\chi_\pm$ 生成的 Killing 视界；其中 $r=r_+$ 是选定渐近平直端
的未来事件视界，$r=r_-$ 是内 Cauchy 视界。对 $a=0$，只有
$r_+=2M$ 是 Schwarzschild 视界，代数根 $r_-=0$ 是曲率奇点。对
$|a|=M$，重根 $r=M$ 是退化的外事件/Killing 视界，“外根与内根是
两个不同超曲面”的陈述不适用。

命题 15.1 只完成局部代数检查；从 $g^{rr}=0$ 单独不能推出“这是事件视界”，因为事件视界依赖整个因果未来。若 $|a|>M$，$\Delta$ 无实根，标准 Kerr 解析解中的 $\Sigma=0$ 环奇点不被上述外视界遮蔽。

**定义 15.3（宇宙审查语境）.** 裸奇点是未被事件视界遮蔽的奇点。经典宇宙审查猜想粗略地说，合理引力塌缩不会产生可从无穷远看见的裸奇点。该猜想不是已完全证明的定理；$|a|>M$ 的解析 Kerr 参数族本身也不证明动力学塌缩会产生这种终态。

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

**命题 15.4（Killing 守恒量）.** 若 $K^\mu$ 是 Killing 向量场，则 $K_\mu p^\mu$ 沿仿射测地线守恒。

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

**外部输入定理 15.5（Carter 可分离性）.** Kerr 测地线除 $E,L_z$ 和质量壳条件外，还有 Carter 常数 $Q$。它来自一个非平凡二阶 Killing 张量，并使 Hamilton--Jacobi 方程在 Boyer-Lindquist 坐标中可分离。

Hamilton-Jacobi 方程为

$$
g^{\mu\nu}\partial_\mu S\partial_\nu S=-m^2.
$$

在 Kerr 几何中可用分离形式

$$
S=-Et+L_z\phi+S_r(r)+S_\theta(\theta)+\frac12m^2\lambda.
$$

代入后方程分离为只含 $r$ 的部分和只含 $\theta$ 的部分，分离常数就是 Carter 常数。

本书不完整推导 Killing 张量的构造与全部代数式，只记录测地线方程的标准结构：

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

Kerr 黑洞外视界面积为

$$
A=4\pi(r_+^2+a^2)=8\pi M r_+.
$$

定义不可逆质量 $M_{\mathrm{irr}}$：

$$
A=16\pi M_{\mathrm{irr}}^2.
$$

于是

$$
M_{\mathrm{irr}}^2=\frac12Mr_+.
$$

**外部输入定理 15.6（Hawking 面积定理）.** 在 Einstein 方程成立、物质满足类光能量条件（null energy condition），并具备排除裸奇点影响视界生成元的标准强渐近可预测性与正则性假设时，未来事件视界截面的面积向未来不减。它是黑洞热力学的经典基础之一。本书不证明其 Raychaudhuri 方程与全局因果几何结合的完整论证。

## 15.9 本章边界

Kerr 几何是现代黑洞物理的核心，但完整理论包括：

- Kerr 解的求解和代数分类。
- 主类光方向和 Petrov D 型。
- Teukolsky 方程和黑洞微扰。
- 准正规模和 ringdown 波形。
- Kerr 稳定性。

这些主题超过第一门 GR 教材的内部证明范围，但第十五章的内容足以连接 Schwarzschild 黑洞和现代旋转黑洞物理。

## 本章小结

$\Delta=0$、$g(\chi,\chi)=0$ 是 Kerr 视界的局部代数；外事件视界的识别还需要正则延拓和全局因果结构。定常与轴对称 Killing 场在书内给出 $E,L_z$ 守恒，Carter 常数、最大延拓和面积定理则作为带假设的外部输入。

## 习题

1. 证明 $a=0$ 时 Kerr 度规退化为 Schwarzschild 度规。
2. 解 $\Delta=0$ 并讨论 $|a|<M, |a|=M, |a|>M$ 三种情形。
3. 推导静止极限面 $r_{\mathrm{stat}}(\theta)$。
4. 证明 Killing 向量给出沿测地线守恒量。
5. 解释为什么能层内可以出现负 Killing 能量。
