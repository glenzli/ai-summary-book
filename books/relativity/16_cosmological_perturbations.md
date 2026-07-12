# 第十六章 宇宙学扰动、结构形成与规范问题

第十一章讨论完全齐性各向同性的 FLRW 宇宙。真实宇宙中存在星系、星系团、宇宙微波背景各向异性和大尺度结构，因此必须研究 FLRW 背景上的微小扰动。本章给出宇宙学扰动论的正式入口。

## 本章目标

本章固定一阶标量扰动的规范变换约定，构造 Bardeen 规范不变量，在明确的物质主导、亚视界和 Newton 近似下推导密度增长方程，并区分书内流体推导与 Boltzmann/原初扰动等外部理论。

## 依赖前置知识

需要第十一章的 FLRW 方程和第十二章的线性规范变换思想。撇号表示共形时间 $\eta$ 导数，圆点只在明确声明后表示宇宙学固有时 $t$ 导数；全章先取空间平直背景。

## 16.1 背景加扰动

在固定背景分解与坐标图中引入形式小参数 $\varepsilon$，把度规写成

$$
g_{\mu\nu}
=\bar g_{\mu\nu}+\varepsilon\,\delta g_{\mu\nu}.
$$

以下只保留 $O(\varepsilon)$，最后令 $\varepsilon=1$。单个坐标分量的
“小”不是坐标不变量；线性理论依赖所选背景与背景--扰动识别。

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

在背景空间齐性各向同性、线性化方程保持空间旋转对称，且给定 Fourier 模式 $\mathbf k\ne0$ 的条件下，标量、矢量、张量子空间不混合。零模、空间边界和非线性阶需要另行处理，不能由这一线性分解直接推出。

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

把标量型生成元写成
$$
\xi^0=T,
\qquad
\xi^i=\partial^iL.
$$
在本章 $\delta g\mapsto\delta g-\mathcal L_\xi\bar g$ 的约定下，直接计算 Lie 导数得到
$$
\Phi\mapsto\Phi-\mathcal H T-T',
\qquad
\Psi\mapsto\Psi+\mathcal H T,
$$
$$
B\mapsto B+T-L',
\qquad
E\mapsto E-L,
$$
其中 $\mathcal H=a'/a$。这些号差依赖于坐标变换与度规分解的约定；引用其他资料时必须同时转换两者。

**定义 16.1（Bardeen 势）.** 令
$$
\sigma=B-E'.
$$
定义
$$
\Phi_B=\Phi+\mathcal H\sigma+\sigma',
\qquad
\Psi_B=\Psi-\mathcal H\sigma.
$$

**命题 16.2.** $\Phi_B$ 与 $\Psi_B$ 在上述一阶标量规范变换下不变。

**证明.** 由变换公式有 $\sigma\mapsto\sigma+T$。因此
$$
\Phi_B\mapsto
\Phi-\mathcal HT-T'
+\mathcal H(\sigma+T)+(\sigma+T)'=\Phi_B,
$$
且
$$
\Psi_B\mapsto\Psi+\mathcal HT-\mathcal H(\sigma+T)=\Psi_B.
$$
这正是所需不变性。$\square$

**定义 16.3（Newton 规范）.** 若取

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

在该规范中 $\sigma=0$，故坐标函数 $\Phi,\Psi$ 分别等于规范不变量 $\Phi_B,\Psi_B$。对 $\mathbf k\ne0$ 的标量模，配合适当空间边界条件，$B=E=0$ 消除了标量规范自由；齐次零模仍可能对应背景参数的重新定义。

对 $\mathbf k\ne0$ 的标量模，在物质无标量各向异性应力的情形，
线性 Einstein 方程的无迹空间分量给出

$$
\Phi=\Psi.
$$

## 16.4 密度扰动增长方程

考虑无压、非相对论物质主导、亚视界尺度 $k/a\gg H$、Newton 近似有效且各向异性应力可忽略的情形。一般一阶密度扰动满足
$$
\delta\rho\mapsto\delta\rho-\bar\rho'T,
$$
所以密度反差本身依赖规范。本节固定 Newton 规范，并定义

$$
\delta=\frac{\rho-\bar\rho}{\bar\rho}.
$$

以下圆点表示固有时 $t$ 导数，$\mathbf v$ 是物理 peculiar velocity。连续性方程、Euler 方程和 Poisson 方程线性化为

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

**命题 16.4（亚视界无压物质增长方程）.** 在本节全部假设下，消去 $\mathbf v$ 和 $\Phi$ 得

$$
\ddot\delta+2H\dot\delta
-4\pi G\bar\rho\,\delta=0.
$$

**证明.** 记 $\theta=\nabla\cdot\mathbf v$。连续性方程给出 $\theta=-a\dot\delta$。对其求导并用 Euler 方程的散度，得到
$$
\ddot\delta+2H\dot\delta
=\frac{1}{a^2}\nabla^2\Phi.
$$
代入 Poisson 方程即得结论。$\square$

第二项是宇宙膨胀带来的阻尼，第三项是引力聚集。该方程不是任意尺度上的规范无关精确方程；超视界、辐射、多流体或有压情形必须回到完整相对论扰动方程。

## 16.5 物质主导宇宙中的增长

**命题 16.5（物质主导宇宙的线性增长）.** 在平直、无 $\Lambda$ 的物质主导宇宙中，本节亚视界增长方程有增长模

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

重组前，光子和重子强耦合，形成光子-重子流体。密度扰动在引力压缩和辐射压之间振荡，产生声学峰。粗略地，对共动波数为 $k$ 的某个 Fourier 模式，用共形时间可写成受迫振子形式

$$
\delta_\gamma''+c_s^2k^2\delta_\gamma
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

**外部输入定理 16.6（FLRW 上的线性张量模）.** 若物质的 transverse-traceless 各向异性应力在一阶为零，则每个线性张量模满足

$$
h_{ij}''+2\mathcal H h_{ij}'-\nabla^2 h_{ij}=0,
$$

其中

$$
\mathcal H=\frac{a'}{a}.
$$

与平直时空波动方程相比，多出的 $2\mathcal H h_{ij}'$ 是宇宙膨胀阻尼项。该式可由 Einstein 方程的 TT 投影或二阶张量扰动作用量推出；完整的二阶作用量展开作为标准宇宙学扰动论外部输入。若存在 TT 各向异性应力，右侧还会出现相应源项。

## 16.8 本章范围

宇宙学扰动论是现代精密宇宙学的基础。完整理论还包括：

- 规范不变量系统，例如 Bardeen 势。
- 多组分 Boltzmann 方程。
- 暴胀产生原初扰动的量子起源。
- 非线性结构形成和 N-body 模拟。
- 弱引力透镜与星系巡天统计。

这些内容超出本书内部完整证明范围，但第十六章给出正式入口和最低计算闭合。

## 本章小结

宇宙学扰动首先是背景场分解后的规范理论，而不是一组天然可观测的坐标分量。Bardeen 势把标量几何扰动写成规范不变量；Newton 规范在非零 Fourier 模式上给出方便代表。密度增长方程只在无压物质、亚视界和 Newton 近似中成立，声学振荡、张量模和精密 CMB 计算还需要明确的流体或 Boltzmann 外部输入。

## 习题

1. 解释为什么度规扰动存在规范问题。
2. 写出 Newton 规范下的标量扰动度规。
3. 从连续性、Euler 和 Poisson 方程推导线性增长方程。
4. 求解物质主导宇宙中的增长模和衰减模。
5. 说明 CMB 声学峰为什么能测量宇宙学参数。
