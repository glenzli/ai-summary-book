# 第九章 Schwarzschild 几何与经典检验

Einstein 方程只有在具体几何中才能转化为轨道和信号传播的预测。球对称、静态真空区域由 Schwarzschild 解描述；把该深结果作为精确外部输入后，其余任务是书内可检查的测地线计算：Killing 对称给出守恒量，有效势区分圆轨道的存在与稳定，弱场展开则产生近日点进动、光线偏折和 Shapiro 延迟。本章调用第六章的曲率与 Killing 对称、第七章的测地线以及第八章的真空方程，并在每个观测公式处标明所保留的近似阶。

本章取 $c=1$、保留 $G$，并在黑洞与轨道部分假设物理质量 $M>0$。
因此反复出现的几何长度是 $GM$；恢复单位时应整体替换为
$GM/c^2$，不能只给某一个 $M$ 补上 $G$。

## 9.1 Schwarzschild 度规

球对称静态真空解为

$$
ds^2
=-\left(1-\frac{2GM}{r}\right)dt^2
+\left(1-\frac{2GM}{r}\right)^{-1}dr^2
+r^2d\Omega^2,
$$

其中

$$
d\Omega^2=d\theta^2+\sin^2\theta\,d\phi^2.
$$

$r_s=2GM$ 称为 Schwarzschild 半径；恢复单位为
$r_s=2GM/c^2$。$r=2GM$ 处的坐标表达奇异，但曲率不发散；
$r=0$ 才是真曲率奇点。

**外部输入定理 9.1（Birkhoff--Schwarzschild 定理）.** 设四维 Lorentz 流形的一个连通开区域球对称、满足 $R_{\mu\nu}=0$，且面积半径 $r$ 在该区域不是常数。则该区域局部等距于某个质量参数 $M$ 的 Schwarzschild 解。若再选择渐近平直的静态外部区域，并以无穷远处的时间归一化 Killing 场，则得到上式；在该静止渐近系中 $P_i=0$，且参数 $M=E_{\mathrm{ADM}}=m_{\mathrm{ADM}}$。

该定理的全局版本、常面积半径退化情形和带 $\Lambda$ 的推广不在本书内部证明。本章把 Schwarzschild 度规作为外部已知精确解使用，并在其中推导轨道和经典检验。

为了显示局部计算的来源，给出真空外部区域中的求解路线。静态球对称度规可在面积半径满足 $dr\ne0$ 的坐标片中写成

$$
ds^2=-e^{2\Phi(r)}dt^2+e^{2\Lambda(r)}dr^2+r^2d\Omega^2.
$$

真空方程 $R_{\mu\nu}=0$ 的两个独立组合给出

$$
\frac{d}{dr}\left[r(1-e^{-2\Lambda})\right]=0,
$$

以及

$$
\Phi'(r)+\Lambda'(r)=0.
$$

第一式积分得

$$
e^{-2\Lambda}=1-\frac{C}{r}.
$$

第二式说明

$$
e^{2\Phi}=A e^{-2\Lambda}.
$$

无穷远处时间坐标的归一化令 $A=1$，Newton 极限令 $C=2GM$，于是得到 Schwarzschild 度规。这里省略的 Ricci 分量计算只解释静态坐标片中的公式；球对称真空区域必为静态的结论属于定理 9.1 的外部输入，不能由这段静态 ansatz 反向证明。

## 9.2 守恒量

在赤道面 $\theta=\pi/2$ 中运动不失一般性。粒子 Lagrangian 可取

$$
2L
=-\left(1-\frac{2GM}{r}\right)\dot{t}^2
+\left(1-\frac{2GM}{r}\right)^{-1}\dot{r}^2
+r^2\dot{\phi}^2.
$$

设点表示对固有时 $\tau$ 求导。由于 $t,\phi$ 是循环坐标，有每单位静质量的守恒量

$$
E=\left(1-\frac{2GM}{r}\right)\dot{t},
\qquad
\ell=r^2\dot{\phi}.
$$

类时测地线满足

$$
g_{\mu\nu}\dot{x}^\mu\dot{x}^\nu=-1.
$$

代入得径向方程

$$
\dot{r}^2+V_{\mathrm{eff}}(r)=E^2,
$$

其中

$$
V_{\mathrm{eff}}(r)
=\left(1-\frac{2GM}{r}\right)
\left(1+\frac{\ell^2}{r^2}\right).
$$

对类光测地线，改用仿射参数；$E$ 与 $\ell$ 的共同缩放依赖仿射参数归一化，但比值 $b=\ell/E$ 不变。径向方程把括号中的 $1$ 去掉：

$$
V_{\mathrm{eff}}^{\mathrm{null}}(r)
=\left(1-\frac{2GM}{r}\right)
\frac{\ell^2}{r^2}.
$$

## 9.3 圆轨道

**命题 9.2（类时圆轨道及其稳定性）.** 令 $r_0>2GM$。赤道面上半径为 $r_0$ 的类时圆测地线存在当且仅当 $r_0>3GM$；此时
$$
\ell^2=\frac{GMr_0^2}{r_0-3GM},
\qquad
E^2=\frac{(r_0-2GM)^2}{r_0(r_0-3GM)}.
$$
该圆轨道在固定 $\ell$ 的径向扰动下稳定当且仅当 $r_0>6GM$，在 $r_0=6GM$ 边际稳定，在 $3GM<r_0<6GM$ 不稳定。

**证明.** 圆轨道满足

$$
\dot{r}=0,\qquad \frac{dV_{\mathrm{eff}}}{dr}=0.
$$

令 $\mu=GM$。由 $V_{\mathrm{eff}}'(r_0)=0$ 得
$$
\frac{2\mu}{r_0^2}-\frac{2\ell^2}{r_0^3}
+\frac{6\mu\ell^2}{r_0^4}=0,
$$
从而
$$
\ell^2=\frac{\mu r_0^2}{r_0-3\mu}.
$$
右侧为正当且仅当 $r_0>3\mu$。代回 $E^2=V_{\mathrm{eff}}(r_0)$ 得命题中的能量公式。保持 $\ell$ 不变再求一次导数，并代入圆轨道条件，得到
$$
V_{\mathrm{eff}}''(r_0)
=\frac{2\mu(r_0-6\mu)}{r_0^3(r_0-3\mu)}.
$$
因此局部极小、退化驻点和局部极大分别对应 $r_0>6\mu$、$r_0=6\mu$ 和 $3\mu<r_0<6\mu$。$\square$

$r=6GM$ 因而称为最内稳定圆轨道（ISCO）；这不意味着 $3GM<r<6GM$ 内不存在圆轨道，而是这些圆轨道不稳定。

**命题 9.3（光子球）.** 非径向类光圆测地线的半径唯一为
$$
r_0=3GM,
$$
且该轨道径向不稳定；其临界冲击参数为 $b_c=3\sqrt3\,GM$。

**证明.** 除去正因子 $\ell^2$，类光有效势为
$$
W(r)=\frac{1}{r^2}\left(1-\frac{2GM}{r}\right).
$$
直接计算得
$$
W'(r)=\frac{2(3GM-r)}{r^4},
\qquad
W''(3GM)<0.
$$
故唯一驻点为不稳定极大值 $r_0=3GM$。圆轨道上 $E^2=\ell^2W(3GM)=\ell^2/(27G^2M^2)$，所以 $b_c=\ell/E=3\sqrt3\,GM$。$\square$

恢复光速后，光子球半径与临界冲击参数分别为
$3GM/c^2$ 与 $3\sqrt3\,GM/c^2$。

## 9.4 水星近日点进动

令 $u=1/r$。类时轨道方程可写为

$$
\frac{d^2u}{d\phi^2}+u
=\frac{GM}{\ell^2}+3GMu^2.
$$

最后一项是 GR 修正。弱场小偏心率近似给出每圈进动

$$
\Delta\phi
=\frac{6\pi GM}{a(1-e^2)}.
$$

恢复光速为

$$
\Delta\phi
=\frac{6\pi GM}{a(1-e^2)c^2}.
$$

这里 $a$ 是半长轴，$e$ 是偏心率。

这一系数可用摄动频率看出。Newton 椭圆满足

$$
u_0=\frac{1}{p}(1+e\cos\phi),
\qquad
p=a(1-e^2).
$$

GR 修正项使径向振荡相位相对角变量略慢，可写成近似形式

$$
u\simeq\frac{1}{p}\left[1+e\cos((1-\delta)\phi)\right],
$$

其中一阶计算给出

$$
\delta=\frac{3GM}{p}.
$$

径向变量完成一次振荡时，角变量增加

$$
\frac{2\pi}{1-\delta}
\simeq2\pi(1+\delta).
$$

因此每圈额外进动角为

$$
\Delta\phi\simeq2\pi\delta
=\frac{6\pi GM}{p}
=\frac{6\pi GM}{a(1-e^2)}.
$$

这个推导解释了公式中半通径 $p=a(1-e^2)$ 的来源。

## 9.5 光线偏折

类光轨道方程为

$$
\frac{d^2u}{d\phi^2}+u=3GMu^2.
$$

对无扰直线解 $u_0=\sin\phi/b$ 做一阶修正，可得总偏折角

$$
\Delta\phi=\frac{4GM}{b}.
$$

恢复光速为

$$
\Delta\phi=\frac{4GM}{bc^2}.
$$

这里 $b$ 是冲击参数。Newton 式粒子偏折只给出一半，GR 的空间曲率贡献补足另一半。

## 9.6 Shapiro 延迟

径向附近传播的光在 Schwarzschild 坐标时间中满足

$$
0=-\left(1-\frac{2GM}{r}\right)dt^2
+\left(1-\frac{2GM}{r}\right)^{-1}dr^2+\cdots.
$$

与平直时空相比，光信号经过太阳附近会多出时间延迟。弱场近似下，延迟主项具有

$$
\Delta t\sim 2GM\ln\frac{r_Er_R}{b^2}
$$

的对数形式；恢复单位时前因子为 $2GM/c^3$。精确系数和对数中的
常数依赖发射、反射、接收几何；符号 $\sim$ 在此只表示弱场展开的
首个非平凡阶，而不是 $r_E,r_R\to\infty$ 下未经说明的严格渐近等价。

## 9.7 从外部几何到经典检验

Schwarzschild 解本身作为 Birkhoff--Schwarzschild 外部输入使用；给定该度规后，Killing 守恒量和有效势计算在书内闭合。类时圆轨道在 $r>3GM$ 存在，但只有 $r>6GM$ 稳定；$r=3GM$ 的类光圆轨道是不稳定光子球。经典检验公式均是弱场展开的首阶结果。

## 习题

1. 从 Schwarzschild Lagrangian 推导 $E$ 和 $\ell$。
2. 推导类时有效势。
3. 求类光圆轨道半径。
4. 从轨道方程说明为什么存在近日点进动。
5. 给出光偏折公式量纲恢复后的形式。
