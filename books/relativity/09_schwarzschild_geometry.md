# 第九章 Schwarzschild 几何与经典检验

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

$r_s=2GM$ 称为 Schwarzschild 半径。$r=2GM$ 处的坐标表达奇异，但曲率不发散；$r=0$ 才是真曲率奇点。

本章把该解作为外部已知精确解使用，并在其中推导轨道和经典检验。

为了说明它不是凭空写下的，给出求解轮廓。静态球对称度规可在合适坐标中写成

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

渐近平直性要求 $A=1$，Newton 极限要求 $C=2GM$，于是得到 Schwarzschild 度规。完整求解需要逐项计算 Ricci 张量，本书把代数细节放入标准输入边界，但保留这个推导骨架。

## 9.2 守恒量

在赤道面 $\theta=\pi/2$ 中运动不失一般性。粒子 Lagrangian 可取

$$
2L
=-\left(1-\frac{2GM}{r}\right)\dot{t}^2
+\left(1-\frac{2GM}{r}\right)^{-1}\dot{r}^2
+r^2\dot{\phi}^2.
$$

由于 $t,\phi$ 是循环坐标，有守恒量

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

类光测地线则把括号中的 $1$ 去掉：

$$
V_{\mathrm{eff}}^{\mathrm{null}}(r)
=\left(1-\frac{2GM}{r}\right)
\frac{\ell^2}{r^2}.
$$

## 9.3 圆轨道

圆轨道满足

$$
\dot{r}=0,\qquad \frac{dV_{\mathrm{eff}}}{dr}=0.
$$

对类时轨道，可得到稳定圆轨道只存在于

$$
r>6GM
$$

之外。$r=6GM$ 是最内稳定圆轨道 ISCO。类光圆轨道在

$$
r=3GM.
$$

这称为光子球。

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

这里 $b$ 是冲量参数。Newton 式粒子偏折只给出一半，GR 的空间曲率贡献补足另一半。

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

的对数形式。精确系数依赖发射、反射、接收几何。

## 习题

1. 从 Schwarzschild Lagrangian 推导 $E$ 和 $\ell$。
2. 推导类时有效势。
3. 求类光圆轨道半径。
4. 从轨道方程说明为什么存在近日点进动。
5. 给出光偏折公式量纲恢复后的形式。
