# 第十二章 线性化引力与引力波

## 12.1 弱场展开

在线性化引力中写

$$
g_{\mu\nu}=\eta_{\mu\nu}+h_{\mu\nu},
\qquad
|h_{\mu\nu}|\ll1.
$$

只保留 $h$ 的一阶项。定义迹反转扰动

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

Einstein 方程线性化为

$$
\Box\bar{h}_{\mu\nu}
=-16\pi G T_{\mu\nu}.
$$

真空中

$$
\Box\bar{h}_{\mu\nu}=0.
$$

为了看清这个方程从哪里来，先把 Christoffel 符号线性化：

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

代入 $G_{\mu\nu}=8\pi G T_{\mu\nu}$ 即得上面的线性化场方程。

## 12.2 规范自由

小坐标变换

$$
x^\mu\mapsto x^\mu+\xi^\mu
$$

使扰动变为

$$
h_{\mu\nu}\mapsto h_{\mu\nu}
-\partial_\mu\xi_\nu-\partial_\nu\xi_\mu.
$$

因此 $h_{\mu\nu}$ 的许多分量是坐标自由，不是物理自由度。引力波的物理自由度只有两个偏振。

自由度计数可粗略理解如下：对称张量 $h_{\mu\nu}$ 有 $10$ 个分量；小坐标变换提供 $4$ 个规范自由；场方程和约束再去掉非传播分量；最终真空引力波只剩两个传播偏振。这一计数不替代严格 Hamilton 约束分析，但足以解释为什么探测器只寻找 $+$ 和 $\times$ 两种张量偏振。

## 12.3 TT 规范

对沿 $z$ 方向传播的平面波，在 transverse-traceless 规范中可写为

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

## 12.4 对测试粒子的作用

一圈自由落体测试粒子在引力波经过时会发生横向拉伸和压缩。测地线偏离方程在线性近似下给出

$$
\frac{d^2\xi^i}{dt^2}
=+\frac12\ddot{h}^{TT}_{ij}\xi^j.
$$

这正是激光干涉仪测量臂长差的理论基础。

## 12.5 四极矩公式

孤立非相对论源的引力辐射主项由质量四极矩给出：

$$
h_{ij}^{TT}
\sim
\frac{2G}{R}
\frac{d^2Q_{ij}^{TT}}{dt^2}.
$$

辐射功率主项为

$$
P=\frac{G}{5}
\left\langle
\frac{d^3Q_{ij}}{dt^3}
\frac{d^3Q_{ij}}{dt^3}
\right\rangle.
$$

四极矩公式的严格推导需要远区展开和规范处理，本书将其作为线性化理论中的标准输入结果。

## 12.6 能流和应变量级

在线性化理论中，引力波本身的局域能量密度不能用普通张量完全坐标无关地定义。但在远区、短波近似和 TT 规范下，可以给出有效能流：

$$
\frac{dE}{dA\,dt}
=\frac{1}{32\pi G}
\left\langle
\dot h_+^2+\dot h_\times^2
\right\rangle.
$$

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

## 习题

1. 推导 Lorenz 规范下的线性化 Einstein 方程形式。
2. 证明真空线性化方程支持以光速传播的平面波。
3. 写出沿 $z$ 方向传播的 $+$ 偏振对圆环测试粒子的影响。
4. 解释为什么引力波没有偶极辐射主项。
5. 说明四极矩公式的适用条件。
