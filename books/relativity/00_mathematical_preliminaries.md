# 第零章 数学预备、单位与证明约定

本章用于固定本书的正式教材口径。相对论的困难不主要在公式多，而在于同一个对象会有坐标形式、几何形式和物理解释三种面貌。若不先固定这些层次，后文很容易把坐标巧合误认为物理结论。

## 0.1 对象、分量与坐标

设 $V$ 是实向量空间。一个向量 $v\in V$ 本身不是一列数；选定基 $e_\mu$ 后才有分量

$$
v=v^\mu e_\mu.
$$

若换基

$$
e'_\mu=A^\nu{}_\mu e_\nu,
$$

则同一个向量的分量满足

$$
v'^\mu=(A^{-1})^\mu{}_\nu v^\nu.
$$

这说明上指标分量和基变换反向变化。相对论中所谓“协变”不是所有公式看起来一样，而是几何对象的等式在坐标变换下仍表达同一个事实。

**定义 0.1（张量）.** 一个 $(r,s)$ 型张量是多重线性映射

$$
T:(V^*)^r\times V^s\to \mathbb R.
$$

在基和对偶基下，它写成分量

$$
T=T^{\mu_1\cdots\mu_r}{}_{\nu_1\cdots\nu_s}
e_{\mu_1}\otimes\cdots\otimes e_{\mu_r}
\otimes
\theta^{\nu_1}\otimes\cdots\otimes\theta^{\nu_s}.
$$

分量变换律来自基和对偶基的变换律，而不是额外假设。

## 0.2 Einstein 求和约定

本书采用 Einstein 求和约定：一个指标在同一项中上下各出现一次时自动求和。例如

$$
a_\mu b^\mu=\sum_{\mu=0}^3 a_\mu b^\mu.
$$

同一项中同一个指标不应出现三次或更多次。若必须临时重复，应先改名虚指标。

**例 0.1.** 表达式

$$
T^\mu{}_\nu v^\nu
$$

表示对 $\nu$ 求和，结果仍带自由指标 $\mu$，因此是一个向量分量。表达式

$$
T^\mu{}_\mu
$$

对 $\mu$ 求和，结果是迹，是标量。

## 0.3 度量、升降指标与号差

本书采用 Lorentz 号差

$$
(-,+,+,+).
$$

Minkowski 度量分量为

$$
\eta_{\mu\nu}=\operatorname{diag}(-1,1,1,1).
$$

给定度量 $g_{\mu\nu}$，可用它建立切向量和余切向量之间的同构：

$$
v_\mu=g_{\mu\nu}v^\nu,
\qquad
\omega^\mu=g^{\mu\nu}\omega_\nu.
$$

必须注意：升降指标会引入号差。例如在 Minkowski 时空中，

$$
v_0=-v^0,\qquad v_i=v^i.
$$

因此 $p^\mu=(E,\mathbf p)$ 时，

$$
p_\mu=(-E,\mathbf p),
\qquad
p_\mu p^\mu=-E^2+\mathbf p^2.
$$

## 0.4 曲线、参数与重参数化

设 $\gamma:I\to M$ 是曲线。局部坐标中写

$$
x^\mu=x^\mu(\lambda).
$$

切向量为

$$
\dot\gamma^\mu=\frac{dx^\mu}{d\lambda}.
$$

若 $\lambda=\lambda(\sigma)$ 且 $d\lambda/d\sigma>0$，则

$$
\frac{dx^\mu}{d\sigma}
=\frac{dx^\mu}{d\lambda}\frac{d\lambda}{d\sigma}.
$$

曲线的几何像不变，但切向量按参数缩放。类时曲线可用固有时 $\tau$ 作为自然参数；类光曲线没有固有时参数，只能选择仿射参数。

## 0.5 变分计算的基本引理

本书多次使用以下基本事实。

**引理 0.1（变分基本引理）.** 设 $f\in C([a,b])$。若对所有
$\varphi\in C_c^\infty((a,b))$ 都有

$$
\int_a^b f(x)\varphi(x)\,dx=0,
$$

则 $f(x)=0$ 对每个 $x\in[a,b]$ 成立。

**证明.** 若存在 $x_0$ 使 $f(x_0)>0$，由连续性可取小邻域 $U$ 使 $f>0$。取非负且支撑在 $U$ 内、不恒为零的 $\varphi$，积分为正，矛盾。$f(x_0)<0$ 同理。故 $f=0$。$\square$

对多维区域和张量分量，也使用同样思想。严格函数空间问题属于泛函分析，本书只在经典光滑变分层面使用。

## 0.6 Euler-Lagrange 方程

**命题 0.1（Euler-Lagrange 方程）.** 设 $L(q,v,\lambda)$ 为 $C^2$
函数，$q:[\lambda_1,\lambda_2]\to\mathbb R^n$ 为 $C^2$ 曲线，且作用量

$$
S[q]=\int_{\lambda_1}^{\lambda_2}
L(q^a,\dot q^a,\lambda)\,d\lambda.
$$

若 $q$ 对每个 $C^1$、端点为零且使变分仍留在 $L$ 定义域内的变分场
都使 $S$ 的一阶变分为零，则

$$
\frac{d}{d\lambda}
\frac{\partial L}{\partial \dot q^a}
-\frac{\partial L}{\partial q^a}=0.
$$

**证明.** 取变分 $q^a\mapsto q^a+\epsilon \xi^a$，并要求端点变分为零：

$$
\xi^a(\lambda_1)=\xi^a(\lambda_2)=0.
$$

一阶变分为

$$
\delta S
=\int
\left(
\frac{\partial L}{\partial q^a}\xi^a
+\frac{\partial L}{\partial \dot q^a}\dot\xi^a
\right)d\lambda.
$$

分部积分得

$$
\delta S
=\int
\left(
\frac{\partial L}{\partial q^a}
-\frac{d}{d\lambda}\frac{\partial L}{\partial \dot q^a}
\right)\xi^a\,d\lambda.
$$

由变分基本引理，驻值曲线满足

$$
\frac{d}{d\lambda}
\frac{\partial L}{\partial \dot q^a}
-\frac{\partial L}{\partial q^a}=0.
$$

这就是 Euler-Lagrange 方程。$\square$

## 0.7 场论变分

设场 $\phi^A(x)$ 的作用量为

$$
S[\phi]=\int_\Omega
\mathcal L(\phi^A,\partial_\mu\phi^A,x)\,d^4x.
$$

在边界变分消失时，

$$
\delta S
=\int_\Omega
\left[
\frac{\partial\mathcal L}{\partial\phi^A}
-\partial_\mu
\left(
\frac{\partial\mathcal L}
{\partial(\partial_\mu\phi^A)}
\right)
\right]\delta\phi^A\,d^4x.
$$

于是场方程为

$$
\partial_\mu
\left(
\frac{\partial\mathcal L}
{\partial(\partial_\mu\phi^A)}
\right)
-\frac{\partial\mathcal L}{\partial\phi^A}=0.
$$

若 $\mathcal L$ 依赖度量 $g_{\mu\nu}$，则能动张量通常定义为

$$
T_{\mu\nu}
=-\frac{2}{\sqrt{-g}}
\frac{\delta S_m}{\delta g^{\mu\nu}}.
$$

这个定义会在第八章连接物质和几何。

## 0.8 单位制与恢复常数

正文多用自然单位

$$
c=1,\qquad \hbar=1
$$

或只令 $c=1$。恢复 $c$ 的基本规则是：

$$
ds^2=-c^2dt^2+d\mathbf x^2
$$

替代

$$
ds^2=-dt^2+d\mathbf x^2.
$$

能量动量关系恢复为

$$
E^2=p^2c^2+m^2c^4.
$$

Schwarzschild 半径恢复为

$$
r_s=\frac{2GM}{c^2}.
$$

Hawking 温度恢复为

$$
T_H=\frac{\hbar c^3}{8\pi G M k_B}.
$$

## 0.9 本书证明等级

本书中结论分为三类：

1. **书内证明。** 定义、命题和推导在正文中完成。
2. **标准输入。** 属于本课程背景但完整证明过长，例如四极矩公式、Schwarzschild/Kerr 解的完整求解过程。
3. **外部深定理。** 需要专门高级课程或研究文献，例如正质量定理、奇点定理、Kerr 稳定性。

正式教材的严谨性并不要求把每个外部深定理都重证一遍，而要求在使用时明确它的假设、结论和依赖位置。

## 习题

1. 证明 $v_\mu w^\mu$ 在基变换下不变。
2. 在 $(-,+,+,+)$ 号差下，计算 $p_\mu p^\mu$ 并恢复 $c$。
3. 对 $L=\frac12 m\dot q^2-V(q)$ 推导 Newton 方程。
4. 对标量场 Lagrangian $\mathcal L=-\frac12\partial_\mu\phi\partial^\mu\phi-\frac12m^2\phi^2$ 推导场方程。
5. 说明为什么类光曲线不能用固有时参数化。
