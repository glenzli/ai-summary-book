# 第二章 Lorentz 变换、时钟与尺

## 2.1 标准 boost

考虑两个惯性系 $S$ 与 $S'$。$S'$ 以速度 $v$ 沿 $x$ 轴相对 $S$ 运动，并假设两系原点在 $t=t'=0$ 重合。线性、空间均匀性、时间均匀性和间隔不变性要求变换为

$$
\begin{aligned}
t'&=\gamma(t-vx),\\
x'&=\gamma(x-vt),\\
y'&=y,\\
z'&=z,
\end{aligned}
\qquad
\gamma=\frac{1}{\sqrt{1-v^2}}.
$$

**命题 2.1.** 上式保持 Minkowski 间隔。

**证明.**

只需检查 $t,x$ 部分：

$$
\begin{aligned}
-dt'^2+dx'^2
&=-\gamma^2(dt-vdx)^2+\gamma^2(dx-vdt)^2\\
&=\gamma^2\{-(1-v^2)dt^2+(1-v^2)dx^2\}\\
&=-dt^2+dx^2.
\end{aligned}
$$

$y,z$ 不变，故整体间隔不变。证毕。

## 2.2 同时性的相对性

若 $S$ 中两个事件同时，即 $\Delta t=0$，则

$$
\Delta t'=\gamma(\Delta t-v\Delta x)=-\gamma v\Delta x.
$$

当 $\Delta x\ne0$ 时，$S'$ 中它们不同时。所谓“同时”不是两个遥远事件之间的绝对关系，而是依赖惯性系的切片选择。

这并不意味着因果混乱。若两个事件类时或类光相关，其时间顺序在所有保持时间方向的 Lorentz 变换下不变；只有类空分离事件的先后可随惯性系改变。

## 2.3 时间膨胀和长度收缩

设一只钟静止在 $S'$ 中，则 $\Delta x'=0$。由逆变换得

$$
\Delta t=\gamma\Delta t'.
$$

静止在钟自身参考系中的时间 $\Delta t'$ 是固有时，所以运动钟在 $S$ 中经历更长坐标时间才走过同样固有时。

设一根杆静止在 $S'$ 中，固有长度为 $L_0=\Delta x'$。要在 $S$ 中测其长度，必须取同一 $S$ 时刻的两端事件，即 $\Delta t=0$。于是

$$
\Delta x'=\gamma\Delta x,
\qquad
L=\Delta x=\frac{L_0}{\gamma}.
$$

长度收缩的关键是“测量长度”要求同一惯性系中的同时性，因此它不是单纯的视觉透视。

## 2.4 速度合成

由标准 boost

$$
u'_x=\frac{dx'}{dt'}
=\frac{dx-vdt}{dt-vdx}
=\frac{u_x-v}{1-vu_x}.
$$

横向分量为

$$
u'_y=\frac{u_y}{\gamma(1-vu_x)},\qquad
u'_z=\frac{u_z}{\gamma(1-vu_x)}.
$$

若 $|u_x|<1$ 且 $|v|<1$，则

$$
\left|\frac{u_x-v}{1-vu_x}\right|<1.
$$

光速 $|u|=1$ 在 Lorentz 变换下保持为 $1$。这不是额外假设，而是类光间隔不变的直接结果。

## 2.5 快速度

令

$$
v=\tanh\phi,\qquad \gamma=\cosh\phi,\qquad \gamma v=\sinh\phi.
$$

标准 boost 可写为

$$
\begin{pmatrix}
t'\\x'
\end{pmatrix}
=
\begin{pmatrix}
\cosh\phi&-\sinh\phi\\
-\sinh\phi&\cosh\phi
\end{pmatrix}
\begin{pmatrix}
t\\x
\end{pmatrix}.
$$

因此 boost 是 Minkowski 平面中的双曲旋转。速度合成在快速度变量中变成加法：

$$
\phi_{\text{total}}=\phi_1+\phi_2.
$$

这解释了为什么速度本身的合成公式看起来分式化：速度不是 Lorentz 群的一维加法参数，快速度才是。

## 2.6 双生子效应

设一条从事件 $A$ 到事件 $B$ 的类时世界线 $\Gamma$。其固有时为

$$
\tau[\Gamma]=\int_\Gamma \sqrt{dt^2-d\mathbf{x}^2}
=\int_{t_A}^{t_B}\sqrt{1-\mathbf{v}(t)^2}\,dt.
$$

在惯性直线世界线中 $\mathbf{v}=0$ 的参考系下，固有时最大。若另一人以非零速度外出再返回，则积分中的因子小于 $1$，其固有时少于留在惯性系中的人。

这里没有矛盾：往返者的世界线不是单一惯性直线，转向时必须改变惯性系。双生子效应的几何本质是 Minkowski 时空中类时直线在两端固定时给出最大固有时。

## 2.7 匀加速运动

一维恒定固有加速度 $\alpha$ 的世界线可写为

$$
t(\tau)=\alpha^{-1}\sinh(\alpha\tau),\qquad
x(\tau)=\alpha^{-1}\cosh(\alpha\tau).
$$

于是

$$
x^2-t^2=\alpha^{-2}.
$$

四速度和四加速度为

$$
u^\mu=(\cosh\alpha\tau,\sinh\alpha\tau),
\qquad
a^\mu=\alpha(\sinh\alpha\tau,\cosh\alpha\tau).
$$

故

$$
a^\mu a_\mu=\alpha^2.
$$

这说明“恒定加速度”在相对论中应指固有加速度不变，而不是坐标加速度不变。

## 习题

1. 从标准 boost 推导逆变换。
2. 推导横向速度变换公式。
3. 用快速度证明一维速度合成律。
4. 计算恒定固有加速度世界线的坐标速度 $v(t)$。
5. 证明两端固定的惯性类时直线最大化固有时。
