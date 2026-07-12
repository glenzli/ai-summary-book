# 第十二章 线性化引力与引力波

## 本章目标

本章把弱场展开写成有明确阶数的线性理论，推导 Lorenz 规范下的场方程，识别真空平面波的两个规范不变传播自由度，并精确标出四极矩公式与 Isaacson 能流的外部输入边界。

## 依赖前置知识

需要第六章的曲率约定、第七章的测地线偏离和第八章的 Einstein 方程。全章采用 $(-,+,+,+)$ 号差及 $c=1$，令 $\Lambda=0$，并约定 $\Box=\eta^{\mu\nu}\partial_\mu\partial_\nu=-\partial_t^2+\nabla^2$。若 $\Lambda\ne0$，Minkowski 度规不是相应真空方程的背景解，必须改在线性化的 de Sitter 或 anti-de Sitter 背景上展开。

## 12.1 弱场展开

在固定 Minkowski 坐标中引入形式小参数 $\varepsilon$，写

$$
g_{\mu\nu}=\eta_{\mu\nu}+\varepsilon h_{\mu\nu}.
$$

以下公式只保留 $O(\varepsilon)$ 并在写完后令 $\varepsilon=1$；“小”是所选背景和坐标中的微扰陈述，不是 $h_{\mu\nu}$ 单个分量的坐标不变量。定义迹反转扰动

$$
\bar{h}_{\mu\nu}
=h_{\mu\nu}-\frac12\eta_{\mu\nu}h,
\qquad
h=\eta^{\mu\nu}h_{\mu\nu}.
$$

取 Lorenz 规范

$$
\partial^\mu\bar{h}_{\mu\nu}=0.
$$

若物质源也按 $T_{\mu\nu}=O(\varepsilon)$ 计数，Einstein 方程线性化为

$$
\Box\bar{h}_{\mu\nu}
=-16\pi G T_{\mu\nu}.
$$

真空中

$$
\Box\bar{h}_{\mu\nu}=0.
$$

**命题 12.1（线性化 Einstein 张量）.** 在上述约定下，Einstein 张量的一阶项为
$$
G^{(1)}_{\mu\nu}
=-\frac12\Box\bar h_{\mu\nu}
+\partial_{(\mu}\partial^\rho\bar h_{\nu)\rho}
-\frac12\eta_{\mu\nu}\partial_\rho\partial_\sigma \bar h^{\rho\sigma}.
$$
因此 Lorenz 规范下有 $G^{(1)}_{\mu\nu}=-\frac12\Box\bar h_{\mu\nu}$。

**证明.** 先把 Christoffel 符号线性化：

$$
\Gamma^{(1)\rho}_{\mu\nu}
=\frac12\eta^{\rho\sigma}
\left(
\partial_\mu h_{\nu\sigma}
+\partial_\nu h_{\mu\sigma}
-\partial_\sigma h_{\mu\nu}
\right).
$$

因此 Ricci 张量的一阶项为

$$
R^{(1)}_{\mu\nu}
=\frac12
\left(
\partial_\rho\partial_\mu h^\rho{}_\nu
+\partial_\rho\partial_\nu h^\rho{}_\mu
-\Box h_{\mu\nu}
-\partial_\mu\partial_\nu h
\right).
$$

写成迹反转扰动后，Einstein 张量的一阶项可化为

$$
G^{(1)}_{\mu\nu}
=-\frac12\Box\bar h_{\mu\nu}
+\partial_{(\mu}\partial^\rho\bar h_{\nu)\rho}
-\frac12\eta_{\mu\nu}\partial_\rho\partial_\sigma \bar h^{\rho\sigma}.
$$

若取 Lorenz 规范 $\partial^\mu\bar h_{\mu\nu}=0$，后两项消失，得到

$$
G^{(1)}_{\mu\nu}
=-\frac12\Box\bar h_{\mu\nu}.
$$

代入 $G_{\mu\nu}=8\pi G T_{\mu\nu}$ 即得上面的线性化场方程。$\square$

线性化 Bianchi 恒等式给出 $\partial^\mu G^{(1)}_{\mu\nu}=0$，故有源方程要求 $\partial^\mu T_{\mu\nu}=0$ 到所保留阶成立。若源不满足该守恒条件，Lorenz 规范下的波动方程组不相容。

## 12.2 规范自由

与微扰同阶的小坐标变换

$$
x^\mu\mapsto x^\mu+\varepsilon\xi^\mu
$$

使扰动变为

$$
h_{\mu\nu}\mapsto h_{\mu\nu}
-\partial_\mu\xi_\nu-\partial_\nu\xi_\mu.
$$

迹反转变量相应变换为
$$
\bar h_{\mu\nu}\mapsto\bar h_{\mu\nu}
-\partial_\mu\xi_\nu-\partial_\nu\xi_\mu
+\eta_{\mu\nu}\partial_\rho\xi^\rho.
$$
若原扰动满足 Lorenz 规范，则变换后仍满足该规范当且仅当 $\Box\xi^\mu=0$；这称为剩余规范自由。因而 $h_{\mu\nu}$ 的许多分量是坐标自由，而不是物理自由度。

自由度计数可粗略理解如下：对称张量 $h_{\mu\nu}$ 有 $10$ 个分量；小坐标变换提供 $4$ 个规范自由；场方程和约束再去掉非传播分量；最终真空引力波只剩两个传播偏振。这一计数不替代严格 Hamilton 约束分析，下面对单个真空平面波直接验证该结论。

## 12.3 TT 规范

**命题 12.2（真空平面波的两个偏振）.** 设真空 Lorenz 规范解是非零类光波矢 $k$ 的单色平面波。利用满足 $\Box\xi^\mu=0$ 的剩余规范，可将沿 $z$ 方向传播的波写成 transverse-traceless（TT）规范

$$
h_{0\mu}^{TT}=0,\qquad
h^{TT\,i}{}_i=0,\qquad
\partial_i h^{TT\,ij}=0.
$$

非零空间分量可写为

$$
h_{ij}^{TT}
=
\begin{pmatrix}
h_+&h_\times&0\\
h_\times&-h_+&0\\
0&0&0
\end{pmatrix}.
$$

$h_+$ 和 $h_\times$ 是两种偏振。

**证明.** 写 $h_{\mu\nu}=\operatorname{Re}(H_{\mu\nu}e^{ik\cdot x})$。波动方程给出 $k^2=0$，Lorenz 条件给出 $k^\mu\bar H_{\mu\nu}=0$。取 $k$ 沿 $+z$ 方向；剩余平面波规范参数的四个振幅可依次令 $H_{00},H_{01},H_{02},H_{03}$ 为零。把 $\nu=0$ 代入 Lorenz 条件可得迹 $H=0$，故此时 $\bar H_{\mu\nu}=H_{\mu\nu}$。其余 Lorenz 条件再给出 $H_{3\mu}=0$，无迹条件则给出 $H_{22}=-H_{11}$。对称性只留下 $H_{11}$ 与 $H_{12}=H_{21}$，分别记为 $h_+$ 与 $h_\times$。该论证只适用于真空传播区的单个非零频率平面波；有源区不能无条件同时采用 TT 与 Lorenz 的全部简化。$\square$

## 12.4 对测试粒子的作用

一圈自由落体测试粒子在引力波经过时会发生横向拉伸和压缩。测地线偏离方程在线性近似下给出

$$
\frac{d^2\xi^i}{dt^2}
=+\frac12\ddot{h}^{TT}_{ij}\xi^j.
$$

这正是激光干涉仪测量臂长差的理论基础。

## 12.5 四极矩公式

**外部输入定理 12.3（质量四极矩辐射公式）.** 设源孤立、空间紧致、内部速度远小于光速，观测点处于波区 $R$ 远大于源尺度，且弱场迭代与守恒条件成立。定义无迹质量四极矩
$$
Q_{ij}(t)=\int \rho(t,\mathbf x)
\left(x_ix_j-\frac13\delta_{ij}|\mathbf x|^2\right)d^3x.
$$
则最低非消失多极阶的远区辐射为

$$
h_{ij}^{TT}
\sim
\frac{2G}{R}
\frac{d^2Q_{ij}^{TT}(t-R)}{dt^2},
$$

辐射功率主项为

$$
P=\frac{G}{5}
\left\langle
\frac{d^3Q_{ij}}{dt^3}
\frac{d^3Q_{ij}}{dt^3}
\right\rangle.
$$

这里的 $\sim$ 同时表示只取远区 $1/R$ 主项和慢速多极展开的最低
非零阶，并不是无余项的等式。角括号表示对多个周期或适当时间窗平均。
四极矩公式的推导需要延迟 Green 函数、多极展开、源守恒和远区规范
处理，本书将其作为标准物理外部输入；若源强自引力或速度接近光速，
不能把 Newton 质量密度公式直接当作精确结果。

## 12.6 能流和应变量级

在线性化理论中，引力波本身的局域能量密度不能用普通张量完全坐标无关地定义。但在远区、波长远小于背景曲率尺度的短波平均和 TT 规范下，Isaacson 有效能动张量给出

$$
\frac{dE}{dA\,dt}
=\frac{1}{32\pi G}
\left\langle
\dot h^{TT}_{ij}\dot h_{TT}^{ij}
\right\rangle.
$$

对本章的偏振矩阵，$h^{TT}_{ij}h_{TT}^{ij}=2(h_+^2+h_\times^2)$，所以
$$
\frac{dE}{dA\,dt}
=\frac{1}{16\pi G}
\left\langle\dot h_+^2+\dot h_\times^2\right\rangle.
$$
后一式中的 $1/(16\pi G)$ 不能在展开偏振后仍写成 $1/(32\pi G)$；两种写法的差别正来自空间指标求和中的因子 $2$。

这说明探测器测到的无量纲应变虽然很小，但高频变化可以携带有限能量。对双星并合，频率逐渐升高、振幅逐渐增大的 chirp 信号正是轨道能量通过引力波流失的表现。

## 12.7 引力波探测

LIGO/Virgo/KAGRA 类型干涉仪测量的是无量纲应变

$$
h=\frac{\Delta L}{L}.
$$

双黑洞或双中子星并合信号通常分为 inspiral、merger、ringdown 三段。完整模板依赖后牛顿近似、数值相对论和黑洞微扰理论的组合。

## 12.8 线性化计算模板

做弱场引力计算时，可按以下顺序整理：

1. 写出 $g_{\mu\nu}=\eta_{\mu\nu}+h_{\mu\nu}$，明确小量阶数。
2. 保留 $h$ 的一阶项，丢弃 $O(h^2)$。
3. 计算 $\Gamma^{(1)}$、$R^{(1)}_{\mu\nu}$ 和 $G^{(1)}_{\mu\nu}$。
4. 选择 Lorenz 规范或 TT 规范，分清规范条件和物理条件。
5. 真空传播问题用 $\Box\bar h_{\mu\nu}=0$；有源远区辐射用四极矩主项。
6. 最后把可观测量写成应变 $h=\Delta L/L$、能流或频率演化。

## 本章小结

线性化引力是围绕给定背景、按明确小参数截断的规范理论。Lorenz 规范把真空方程化为波动方程，剩余规范在单色真空波上留下两个 TT 偏振。四极矩辐射和 Isaacson 能流还分别需要慢源远区展开与短波平均，不能由一阶场方程不加条件地推出。

## 习题

1. 推导 Lorenz 规范下的线性化 Einstein 方程形式。
2. 证明真空线性化方程支持以光速传播的平面波。
3. 写出沿 $z$ 方向传播的 $+$ 偏振对圆环测试粒子的影响。
4. 解释为什么引力波没有偶极辐射主项。
5. 说明四极矩公式的适用条件。
