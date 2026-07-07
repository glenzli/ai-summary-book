# 附录 B 详细例题

本附录给出若干完整算例。它们的目的不是增加新理论，而是展示如何把正文中的定义、守恒量和近似原则用于实际计算。

## B.1 从间隔不变性导出标准 boost

考虑一维相对运动。设两个惯性系之间的线性变换为

$$
t'=At+Bx,
\qquad
x'=Ct+Dx.
$$

要求原点 $x'=0$ 在未带撇系中以速度 $v$ 运动，即 $x=vt$。代入得

$$
0=Ct+Dvt,
$$

所以

$$
C=-Dv.
$$

要求 Minkowski 间隔不变：

$$
-dt'^2+dx'^2=-dt^2+dx^2.
$$

代入线性变换，比较 $dt^2$、$dt\,dx$、$dx^2$ 系数：

$$
-A^2+C^2=-1,
$$

$$
-AB+CD=0,
$$

$$
-B^2+D^2=1.
$$

由 $C=-Dv$ 和第二式得

$$
AB=-D^2v.
$$

取保持时间取向的标准 boost，令

$$
A=D=\gamma.
$$

则

$$
C=-\gamma v,
\qquad
B=-\gamma v.
$$

第一式给出

$$
-\gamma^2+\gamma^2v^2=-1,
$$

所以

$$
\gamma=\frac{1}{\sqrt{1-v^2}}.
$$

因此

$$
t'=\gamma(t-vx),
\qquad
x'=\gamma(x-vt).
$$

## B.2 相对论自由粒子的 Hamilton 量

自由粒子 Lagrangian 为

$$
L=-m\sqrt{1-\mathbf v^2}.
$$

共轭动量为

$$
p_i=\frac{\partial L}{\partial v^i}
=\frac{mv_i}{\sqrt{1-\mathbf v^2}}
=\gamma mv_i.
$$

Hamilton 量

$$
H=\mathbf p\cdot\mathbf v-L
=\gamma m\mathbf v^2+m\sqrt{1-\mathbf v^2}.
$$

利用

$$
\sqrt{1-\mathbf v^2}=\frac{1}{\gamma},
$$

得

$$
H=\gamma m\mathbf v^2+\frac{m}{\gamma}
=\gamma m\left(\mathbf v^2+\frac1{\gamma^2}\right).
$$

又

$$
\frac1{\gamma^2}=1-\mathbf v^2,
$$

所以

$$
H=\gamma m.
$$

这就是相对论能量 $E$。由 $\mathbf p=\gamma m\mathbf v$ 得

$$
E^2-\mathbf p^2
=\gamma^2m^2(1-\mathbf v^2)=m^2.
$$

## B.3 电磁场不变量

在本书约定下，电磁场张量满足

$$
F^{0i}=E_i,
\qquad
F^{ij}=-\epsilon^{ijk}B_k.
$$

降指标时 $F_{0i}=-E_i$，$F_{ij}=F^{ij}$。于是

$$
F_{\mu\nu}F^{\mu\nu}
=2F_{0i}F^{0i}+F_{ij}F^{ij}.
$$

第一项为

$$
2F_{0i}F^{0i}=-2\mathbf E^2.
$$

第二项为

$$
F_{ij}F^{ij}
=\epsilon_{ijk}\epsilon_{ij\ell}B_kB_\ell
=2\mathbf B^2.
$$

所以

$$
F_{\mu\nu}F^{\mu\nu}
=2(\mathbf B^2-\mathbf E^2).
$$

另一个不变量为

$$
F_{\mu\nu}{}^\star F^{\mu\nu}
=-4\mathbf E\cdot\mathbf B
$$

在本书取向约定下成立。它区分电场和磁场是否相互正交。

## B.4 Schwarzschild 径向自由落体

Schwarzschild 度规为

$$
ds^2=-f(r)dt^2+f(r)^{-1}dr^2+r^2d\Omega^2,
\qquad
f(r)=1-\frac{2GM}{r}.
$$

考虑径向类时测地线，$d\Omega=0$。由于 $t$ 是循环坐标，

$$
E=f(r)\dot t
$$

守恒，其中点表示对固有时求导。归一化条件为

$$
-f(r)\dot t^2+f(r)^{-1}\dot r^2=-1.
$$

代入 $\dot t=E/f(r)$ 得

$$
-\frac{E^2}{f(r)}+\frac{\dot r^2}{f(r)}=-1.
$$

乘以 $f(r)$：

$$
\dot r^2=E^2-f(r).
$$

若粒子从无穷远静止释放，则 $E=1$，因此

$$
\dot r^2=\frac{2GM}{r}.
$$

取落入方向，

$$
\dot r=-\sqrt{\frac{2GM}{r}}.
$$

这说明自由落体者在有限固有时内穿过 $r=2GM$；视界不是自由落体者感到的局部奇异。

## B.5 Newton 极限中的 Christoffel 主项

弱场静态度规取

$$
g_{00}=-(1+2\Phi),
\qquad
g_{ij}=(1-2\Phi)\delta_{ij},
\qquad
g_{0i}=0.
$$

慢速粒子满足 $dx^i/dt\ll1$，测地线方程空间分量的主项为

$$
\frac{d^2x^i}{dt^2}
\approx
-\Gamma^i{}_{00}.
$$

计算

$$
\Gamma^i{}_{00}
=\frac12g^{ij}
\left(
2\partial_0g_{0j}-\partial_jg_{00}
\right).
$$

静态时 $\partial_0g_{0j}=0$，所以

$$
\Gamma^i{}_{00}
=-\frac12g^{ij}\partial_jg_{00}.
$$

到一阶 $g^{ij}\approx\delta^{ij}$，而

$$
\partial_jg_{00}=-2\partial_j\Phi.
$$

故

$$
\Gamma^i{}_{00}\approx\partial_i\Phi.
$$

于是

$$
\frac{d^2x^i}{dt^2}=-\partial_i\Phi.
$$

## B.6 平直尘埃宇宙的尺度因子

平直、无 $\Lambda$、尘埃宇宙满足

$$
H^2=\frac{8\pi G}{3}\rho,
$$

连续性方程给出

$$
\rho=\rho_0a^{-3}.
$$

于是

$$
\left(\frac{\dot a}{a}\right)^2
=\frac{8\pi G}{3}\rho_0a^{-3}.
$$

令

$$
C=\sqrt{\frac{8\pi G\rho_0}{3}},
$$

则

$$
\dot a=Ca^{-1/2}.
$$

分离变量：

$$
a^{1/2}da=Cdt.
$$

积分得

$$
\frac23a^{3/2}=Ct+\text{const}.
$$

取大爆炸时 $a(0)=0$，

$$
a(t)\propto t^{2/3}.
$$

## B.7 引力波对圆环的作用

沿 $z$ 方向传播的 $+$ 偏振可写为

$$
h_{xx}^{TT}=h_+(t-z),
\qquad
h_{yy}^{TT}=-h_+(t-z),
\qquad
h_{xy}^{TT}=0.
$$

测地线偏离方程为

$$
\frac{d^2\xi^i}{dt^2}
=-\frac12\ddot h^{TT}_{ij}\xi^j.
$$

因此

$$
\frac{d^2\xi^x}{dt^2}
=-\frac12\ddot h_+\xi^x,
$$

$$
\frac{d^2\xi^y}{dt^2}
=+\frac12\ddot h_+\xi^y.
$$

若 $h_+>0$ 从零增长，则 $x$ 方向和 $y$ 方向加速度符号相反，圆环被拉成椭圆；半个周期后反向。这就是干涉仪两臂差分响应的几何基础。

## B.8 线性增长方程的 Fourier 形式

对密度反差作 Fourier 展开

$$
\delta(\mathbf x,t)
=\int \delta_{\mathbf k}(t)e^{i\mathbf k\cdot\mathbf x}\,d^3k.
$$

线性增长方程

$$
\ddot\delta+2H\dot\delta-4\pi G\bar\rho\delta=0
$$

在无压尘埃和 Newton 亚视界近似下不显含 $k$，所以每个模式满足

$$
\ddot\delta_{\mathbf k}
+2H\dot\delta_{\mathbf k}
-4\pi G\bar\rho\delta_{\mathbf k}=0.
$$

这说明在最简单模型中所有线性尺度有相同增长因子；真实宇宙中的尺度依赖来自辐射、暗能量、自由流、重子声学和非线性演化等效应。

## B.9 近日点进动系数的摄动推导

Schwarzschild 类时轨道方程为

$$
u''+u=\frac{GM}{\ell^2}+3GMu^2,
$$

其中撇号表示对 $\phi$ 求导。设

$$
p=\frac{\ell^2}{GM}.
$$

Newton 解为

$$
u_0=\frac1p(1+e\cos\phi).
$$

GR 修正项很小，尝试把椭圆相位改写为

$$
u=\frac1p[1+e\cos(\omega\phi)],
$$

其中 $\omega=1-\delta$，且 $\delta$ 为小量。左边给出

$$
u''+u
=\frac1p
\left[
1+e(1-\omega^2)\cos(\omega\phi)
\right].
$$

右边保留与 $e\cos(\omega\phi)$ 同频的一阶项：

$$
\frac1p+3GMu^2
\simeq
\frac1p
+3GM\frac{1}{p^2}
\left[1+2e\cos(\omega\phi)\right].
$$

常数项会轻微改变平均半径；决定进动的是同频项。比较 $\cos(\omega\phi)$ 系数：

$$
\frac{e}{p}(1-\omega^2)
\simeq
\frac{6GMe}{p^2}.
$$

若 $\omega=1-\delta$，则

$$
1-\omega^2\simeq2\delta.
$$

所以

$$
\delta=\frac{3GM}{p}.
$$

径向周期对应 $\omega\phi=2\pi$，故角变量一圈为

$$
\phi=\frac{2\pi}{\omega}
\simeq2\pi(1+\delta).
$$

额外进动为

$$
\Delta\phi=2\pi\delta
=\frac{6\pi GM}{p}
=\frac{6\pi GM}{a(1-e^2)}.
$$

恢复单位即

$$
\Delta\phi
=\frac{6\pi GM}{a(1-e^2)c^2}.
$$

## B.10 FLRW 曲率分量的最小推导

用共动坐标写 FLRW 度规：

$$
ds^2=-dt^2+a(t)^2\gamma_{ij}dx^idx^j,
$$

其中 $\gamma_{ij}$ 是常曲率三维空间度规，满足

$$
{}^{(3)}R_{ij}=2k\gamma_{ij}.
$$

非零 Christoffel 符号的关键部分为

$$
\Gamma^0{}_{ij}=a\dot a\,\gamma_{ij}=Hg_{ij},
$$

$$
\Gamma^i{}_{0j}=H\delta^i{}_j,
$$

以及三维度规 $\gamma_{ij}$ 自身的 Christoffel 符号。代入 Ricci 张量定义，可得

$$
R_{00}=-3(\dot H+H^2)=-3\frac{\ddot a}{a}.
$$

空间分量为

$$
R_{ij}
=\left(\dot H+3H^2+\frac{2k}{a^2}\right)g_{ij}.
$$

由于

$$
\dot H+H^2=\frac{\ddot a}{a},
$$

也可写成

$$
R_{ij}
=\left(
\frac{\ddot a}{a}+2H^2+\frac{2k}{a^2}
\right)g_{ij}.
$$

缩并得

$$
R=6\left(\frac{\ddot a}{a}+H^2+\frac{k}{a^2}\right).
$$

于是

$$
G_{00}=R_{00}+\frac12R
=3\left(H^2+\frac{k}{a^2}\right),
$$

这就是第一 Friedmann 方程左端。

## B.11 Chirp 质量的出现

近圆双星的 Newton 轨道能量为

$$
E=-\frac{G\mu M}{2r}.
$$

Kepler 关系

$$
\Omega^2=\frac{GM}{r^3}
$$

给出

$$
r=(GM)^{1/3}\Omega^{-2/3}.
$$

代入能量：

$$
E=-\frac12\mu(GM)^{2/3}\Omega^{2/3}.
$$

四极矩辐射功率为

$$
P=-\frac{dE}{dt}
=\frac{32}{5}\frac{G^{7/3}}{c^5}
\mu^2M^{4/3}\Omega^{10/3}.
$$

对 $E(\Omega)$ 求导：

$$
\frac{dE}{d\Omega}
=-\frac13\mu(GM)^{2/3}\Omega^{-1/3}.
$$

由

$$
-\frac{dE}{d\Omega}\frac{d\Omega}{dt}=P
$$

得

$$
\frac{d\Omega}{dt}
=\frac{96}{5}
\frac{G^{5/3}}{c^5}
\mu M^{2/3}\Omega^{11/3}.
$$

定义 chirp 质量

$$
\mathcal M=\mu^{3/5}M^{2/5},
$$

则

$$
\mu M^{2/3}=\mathcal M^{5/3}.
$$

主导引力波频率满足 $f=\Omega/\pi$，所以

$$
\frac{df}{dt}
=\frac{96}{5}\pi^{8/3}
\left(\frac{G\mathcal M}{c^3}\right)^{5/3}
f^{11/3}.
$$

这就是 inspiral 相位对 chirp 质量极其敏感的原因。
