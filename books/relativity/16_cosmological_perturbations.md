# 第十六章 宇宙学扰动、结构形成与规范问题

第十一章讨论完全齐性各向同性的 FLRW 宇宙。真实宇宙中存在星系、星系团、宇宙微波背景各向异性和大尺度结构，因此必须研究 FLRW 背景上的微小扰动。本章给出宇宙学扰动论的正式入口。

## 16.1 背景加扰动

把度规写成

$$
g_{\mu\nu}
=\bar g_{\mu\nu}+\delta g_{\mu\nu},
\qquad
|\delta g_{\mu\nu}|\ll1.
$$

背景 $\bar g_{\mu\nu}$ 是 FLRW 度规。常用共形时间 $\eta$：

$$
dt=a(\eta)d\eta.
$$

平直背景下，

$$
d\bar s^2
=a^2(\eta)
\left(
-d\eta^2+\delta_{ij}dx^i dx^j
\right).
$$

扰动论的目标是把 Einstein 方程和物质方程按扰动阶数展开。

## 16.2 标量-矢量-张量分解

在空间旋转群下，线性扰动可分为标量、矢量和张量三类。标量扰动写为

$$
ds^2=a^2(\eta)
\left[
-(1+2\Phi)d\eta^2
+2\partial_iB\,d\eta dx^i
+\left((1-2\Psi)\delta_{ij}
+2\partial_i\partial_jE\right)dx^i dx^j
\right].
$$

矢量扰动满足横向条件，例如

$$
\partial_i S^i=0.
$$

张量扰动满足

$$
\partial_i h^{ij}=0,
\qquad
h^i{}_i=0.
$$

线性阶上，标量、矢量、张量模式彼此解耦。这是宇宙学扰动论能被系统处理的原因。

## 16.3 规范问题

扰动的困难在于坐标变换本身也会产生扰动。设无穷小坐标变换为

$$
x^\mu\mapsto x^\mu+\xi^\mu.
$$

则度规扰动变换为

$$
\delta g_{\mu\nu}
\mapsto
\delta g_{\mu\nu}
-\mathcal L_\xi \bar g_{\mu\nu}.
$$

因此某些看似物理的扰动其实只是坐标选择。正式处理必须使用规范固定或规范不变量。

**定义 16.1（Newton 规范）.** 若取

$$
B=0,\qquad E=0,
$$

则标量扰动度规为

$$
ds^2=a^2(\eta)
\left[
-(1+2\Phi)d\eta^2
+(1-2\Psi)\delta_{ij}dx^i dx^j
\right].
$$

在无各向异性应力的情形，Einstein 方程给出

$$
\Phi=\Psi.
$$

## 16.4 密度扰动增长方程

考虑非相对论物质主导、亚视界尺度、Newton 近似有效的情形。定义密度反差

$$
\delta=\frac{\rho-\bar\rho}{\bar\rho}.
$$

连续性方程、Euler 方程和 Poisson 方程线性化为

$$
\dot\delta+\frac{1}{a}\nabla\cdot\mathbf v=0,
$$

$$
\dot{\mathbf v}+H\mathbf v
=-\frac{1}{a}\nabla\Phi,
$$

$$
\nabla^2\Phi=4\pi G a^2\bar\rho\,\delta.
$$

消去 $\mathbf v$ 和 $\Phi$ 得

$$
\ddot\delta+2H\dot\delta
-4\pi G\bar\rho\,\delta=0.
$$

这就是线性结构增长方程。第二项是宇宙膨胀带来的阻尼，第三项是引力聚集。

## 16.5 物质主导宇宙中的增长

**命题 16.1（物质主导宇宙的线性增长）.** 在平直、无 $\Lambda$ 的物质主导宇宙中，线性密度扰动有增长模

$$
\delta\propto a(t).
$$

**证明.**

在平直物质主导宇宙中

$$
a(t)\propto t^{2/3},
\qquad
H=\frac{2}{3t},
\qquad
\bar\rho=\frac{1}{6\pi Gt^2}.
$$

代入增长方程：

$$
\ddot\delta+\frac{4}{3t}\dot\delta-\frac{2}{3t^2}\delta=0.
$$

设 $\delta\propto t^p$，得到

$$
p(p-1)+\frac43p-\frac23=0,
$$

即

$$
3p^2+p-2=0.
$$

解为

$$
p=\frac23,\qquad p=-1.
$$

增长模为

$$
\delta\propto t^{2/3}\propto a(t).
$$

这说明在物质主导时期，线性密度扰动随尺度因子增长。

证明完毕。$\square$

## 16.6 声学振荡与 CMB

重组前，光子和重子强耦合，形成光子-重子流体。密度扰动在引力压缩和辐射压之间振荡，产生声学峰。粗略地，对某个 Fourier 模式可写成受迫振子形式

$$
\ddot\delta_\gamma+c_s^2k^2\delta_\gamma
\approx \text{gravitational forcing}.
$$

声速近似为

$$
c_s^2=\frac{1}{3(1+R)},
\qquad
R=\frac{3\rho_b}{4\rho_\gamma}.
$$

CMB 角功率谱中的峰结构记录了早期宇宙的声学振荡、空间曲率、重子密度、暗物质密度和暗能量参数。

本书不推导 Boltzmann 层级和复合历程，只说明其与 GR 扰动论的连接。

## 16.7 张量扰动与原初引力波

FLRW 背景上的张量扰动写为

$$
ds^2=a^2(\eta)
\left[
-d\eta^2+(\delta_{ij}+h_{ij})dx^i dx^j
\right],
$$

其中

$$
\partial_i h^{ij}=0,
\qquad
h^i{}_i=0.
$$

真空线性方程为

$$
h_{ij}''+2\mathcal H h_{ij}'-\nabla^2 h_{ij}=0,
$$

其中

$$
\mathcal H=\frac{a'}{a}.
$$

与平直时空波动方程相比，多出的 $2\mathcal H h_{ij}'$ 是宇宙膨胀阻尼项。

## 16.8 本章范围

宇宙学扰动论是现代精密宇宙学的基础。完整理论还包括：

- 规范不变量系统，例如 Bardeen 势。
- 多组分 Boltzmann 方程。
- 暴胀产生原初扰动的量子起源。
- 非线性结构形成和 N-body 模拟。
- 弱引力透镜与星系巡天统计。

这些内容超出本书内部完整证明范围，但第十六章给出正式入口和最低计算闭合。

## 习题

1. 解释为什么度规扰动存在规范问题。
2. 写出 Newton 规范下的标量扰动度规。
3. 从连续性、Euler 和 Poisson 方程推导线性增长方程。
4. 求解物质主导宇宙中的增长模和衰减模。
5. 说明 CMB 声学峰为什么能测量宇宙学参数。
