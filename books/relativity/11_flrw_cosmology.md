# 第十一章 FLRW 宇宙学

## 11.1 均匀各向同性

宇宙学原理假设在大尺度上空间均匀且各向同性。对应度规为

$$
ds^2=-dt^2+a(t)^2
\left[
\frac{dr^2}{1-kr^2}+r^2d\Omega^2
\right],
$$

其中

$$
k=0,\pm1.
$$

$a(t)$ 是尺度因子。

## 11.2 完美流体宇宙

物质取完美流体能动张量

$$
T^{\mu\nu}=(\rho+p)u^\mu u^\nu+pg^{\mu\nu}.
$$

共动观察者满足

$$
u^\mu=(1,0,0,0).
$$

## 11.3 Friedmann 方程

将 FLRW 度规代入 Einstein 方程，得到

$$
H^2
=\left(\frac{\dot{a}}{a}\right)^2
=\frac{8\pi G}{3}\rho-\frac{k}{a^2}+\frac{\Lambda}{3},
$$

以及

$$
\frac{\ddot{a}}{a}
=-\frac{4\pi G}{3}(\rho+3p)+\frac{\Lambda}{3}.
$$

能动张量协变守恒给出连续性方程

$$
\dot{\rho}+3H(\rho+p)=0.
$$

若状态方程为

$$
p=w\rho,
$$

则

$$
\rho\propto a^{-3(1+w)}.
$$

尘埃 $w=0$ 给出 $\rho\propto a^{-3}$；辐射 $w=1/3$ 给出 $\rho\propto a^{-4}$；宇宙学常数 $w=-1$ 给出常数密度。

为便于查算，记录核心曲率分量。对 FLRW 度规有

$$
R_{00}=-3\frac{\ddot a}{a},
$$

以及

$$
R_{ij}
=\left(
\frac{\ddot a}{a}+2H^2+\frac{2k}{a^2}
\right)g_{ij}.
$$

标量曲率为

$$
R=6\left(
\frac{\ddot a}{a}+H^2+\frac{k}{a^2}
\right).
$$

因此 Einstein 张量分量为

$$
G_{00}=3\left(H^2+\frac{k}{a^2}\right),
$$

和

$$
G_{ij}
=-\left(
2\frac{\ddot a}{a}+H^2+\frac{k}{a^2}
\right)g_{ij}.
$$

将 $G_{00}+\Lambda g_{00}=8\pi G T_{00}$ 代入 $T_{00}=\rho$，得到第一 Friedmann 方程；再把空间分量与第一方程组合，得到加速度方程。这组分量是宇宙学计算中最常用的检查点。

## 11.4 红移和距离

宇宙学红移满足

$$
1+z=\frac{a(t_0)}{a(t_e)}.
$$

通常取 $a(t_0)=1$，故

$$
a(t_e)=\frac{1}{1+z}.
$$

径向光线满足 $ds^2=0$，因此

$$
\chi=\int_{t_e}^{t_0}\frac{dt}{a(t)}
=\int_0^z \frac{dz'}{H(z')}.
$$

平直空间中，光度距离为

$$
d_L=(1+z)\chi.
$$

角直径距离为

$$
d_A=\frac{\chi}{1+z}.
$$

二者满足 Etherington 关系

$$
d_L=(1+z)^2d_A.
$$

## 11.5 加速膨胀

由

$$
\frac{\ddot{a}}{a}
=-\frac{4\pi G}{3}(\rho+3p)+\frac{\Lambda}{3}
$$

可见，当

$$
\rho+3p<0
$$

时，膨胀加速。宇宙学常数满足 $p=-\rho$，因此能驱动加速膨胀。

## 11.6 地平线问题

粒子视界由

$$
d_{\mathrm{hor}}(t)=a(t)\int_0^t\frac{dt'}{a(t')}
$$

给出。若早期宇宙没有足够长的因果接触，微波背景的大尺度均匀性就需要解释。暴胀理论通过早期近指数膨胀增大可见区域的共同因果来源。

本书不展开暴胀模型，只记录它在标准宇宙学问题中的位置。

## 习题

1. 由连续性方程推导 $\rho\propto a^{-3(1+w)}$。
2. 对 $k=0,\Lambda=0,w=0$，求 $a(t)$ 的幂律。
3. 对 $k=0,\Lambda=0,w=1/3$，求 $a(t)$ 的幂律。
4. 推导宇宙学红移公式。
5. 解释为什么宇宙学常数对应加速膨胀。
