# 第二章：辛几何与 Hamilton 系统

Lagrange 方程以构型和速度为变量，Hamilton 方程则把位置和动量合成一个几何对象：辛流形。辛形式不是度量；它不测量长度，而把函数微分转换成向量场，从而使“能量生成时间演化”成为内在语句。经典力学中守恒量、Poisson 括号和对称性约化都从这个转换开始。

## 2.1 辛流形

**定义 2.1.** 辛流形是偶数维光滑流形 $M$ 连同闭且非退化的二形式 $\omega\in\Omega^2(M)$。闭性为 $d\omega=0$，非退化性为映射
$$
TM\to T^*M,\qquad v\mapsto\iota_v\omega
$$
在每点为线性同构。

**定义 2.2.** 对 $f\in C^\infty(M)$，Hamilton 向量场 $X_f$ 由
$$
\iota_{X_f}\omega=df
$$
定义。

**命题 2.1 (`P`).** 每个光滑函数 $f$ 存在唯一 Hamilton 向量场 $X_f$。

**证明.** 定义丛映射 $\omega^\flat:TM\to T^*M$，$v\mapsto\iota_v\omega$。非退化性说明每个纤维映射 $\omega_p^\flat$ 都是线性同构。局部坐标中它由可逆光滑矩阵 $(\omega_{ij})$ 表示；逆矩阵的元素是 $\det(\omega_{ij})^{-1}$ 乘余子式，故仍光滑。因此 $(\omega^\flat)^{-1}$ 是光滑丛映射。令
$$
X_f=(\omega^\flat)^{-1}(df).
$$
它是光滑向量场并满足定义式；纤维上的单射性保证唯一性。$\square$

**定义 2.3.** Poisson 括号定义为
$$
\{f,h\}=\omega(X_h,X_f)=X_fh=-X_hf.
$$

**命题 2.2 (`P`).** Poisson 括号满足双线性、反对称、Leibniz 规则和 Jacobi 恒等式。

**证明.** 映射 $f\mapsto X_f$ 线性，故括号双线性；$\omega$ 反对称，故 $\{f,h\}=-\{h,f\}$。又因 $X_f$ 是导子，
$$
\{f,hk\}=X_f(hk)=\{f,h\}k+h\{f,k\}.
$$
为证 Jacobi 恒等式，先证
$$
[X_f,X_h]=X_{\{f,h\}}
$$
推出；该等式用 Cartan 公式和 $d\omega=0$ 验证：
$$
\iota_{[X_f,X_h]}\omega
=\mathcal L_{X_f}\iota_{X_h}\omega-\iota_{X_h}\mathcal L_{X_f}\omega
=d(X_fh)=d\{f,h\}.
$$
这里 $\mathcal L_{X_f}\omega=d\iota_{X_f}\omega+\iota_{X_f}d\omega=d^2f=0$。由 Hamilton 向量场的唯一性即得该等式。于是对任意 $k$，
$$
\begin{aligned}
\{f,\{h,k\}\}+\{h,\{k,f\}\}
&=X_fX_hk-X_hX_fk\\
&=[X_f,X_h]k
=X_{\{f,h\}}k
=\{\{f,h\},k\}.
\end{aligned}
$$
把右端移到左端并用反对称性，便是 Jacobi 恒等式。$\square$

## 2.2 Hamilton 流

**定义 2.4.** Hamiltonian $H\in C^\infty(M)$ 的动力学为
$$
\dot x=X_H(x).
$$

在正则坐标 $(q^i,p_i)$ 且 $\omega=dq^i\wedge dp_i$ 下，
$$
\dot q^i=\frac{\partial H}{\partial p_i},\qquad
\dot p_i=-\frac{\partial H}{\partial q^i}.
$$

**命题 2.3 (`P`).** Hamilton 流保持辛形式：若 $\Phi_t$ 是 $X_H$ 的流，则 $\Phi_t^*\omega=\omega$。

**证明.** 在流存在的时间区间内，拉回沿流的求导公式与 Cartan 公式给出
$$
\frac{d}{dt}\Phi_t^*\omega=\Phi_t^*(\mathcal L_{X_H}\omega)
=\Phi_t^*(d\iota_{X_H}\omega+\iota_{X_H}d\omega)
=\Phi_t^*(d^2H)=0.
$$
故 $t\mapsto\Phi_t^*\omega$ 的导数为零。由 $\Phi_0=\operatorname{id}_M$，其初值为 $\omega$，所以整个存在区间内 $\Phi_t^*\omega=\omega$。$\square$

## 2.3 对称性与 moment map

**定义 2.5.** Lie 群 $G$ 在辛流形 $(M,\omega)$ 上的作用称为 Hamilton 作用，如果存在映射 $\mu:M\to\mathfrak g^*$，使得对每个 $\xi\in\mathfrak g$，由作用生成的向量场 $\xi_M$ 满足
$$
\iota_{\xi_M}\omega=d\langle\mu,\xi\rangle.
$$
$\mu$ 称为 moment map。

**命题 2.4 (`P`).** 若 Hamiltonian $H$ 在 $G$ 作用下不变，则每个 $\langle\mu,\xi\rangle$ 沿 Hamilton 流守恒。

**证明.** 固定 $\xi\in\mathfrak g$。$H$ 的 $G$-不变性对一参数子群 $\exp(t\xi)$ 求导给出 $\xi_MH=0$。由定义，
$$
\frac{d}{dt}\langle\mu,\xi\rangle
=X_H\langle\mu,\xi\rangle
=d\langle\mu,\xi\rangle(X_H)
=\omega(\xi_M,X_H)
=-dH(\xi_M)=-\xi_MH=0.
$$
$\square$

**定理 2.5 (`E`, Darboux 定理).** 任意辛流形每点附近存在坐标 $(q^i,p_i)$，使得 $\omega=\sum_i dq^i\wedge dp_i$。

**外部输入边界.** 本书只使用 Darboux 定理说明辛结构局部无曲率不变量；其证明依赖 Moser 路径法，精确定位见 [SOURCES.md](SOURCES.md) 的 `E-2.5`。

**例 2.6（谐振子的 Hamilton 流）.** 在 $M=\mathbb R^2_{q,p}$ 上取
$\omega=dq\wedge dp$ 和
$$
H(q,p)=\frac{p^2}{2m}+\frac12m\omega_0^2q^2.
$$
定义式给出
$X_H=(p/m)\partial_q-m\omega_0^2q\partial_p$，故
$$
\begin{pmatrix}q(t)\\p(t)\end{pmatrix}
=\begin{pmatrix}
\cos\omega_0t & (m\omega_0)^{-1}\sin\omega_0t\\
-m\omega_0\sin\omega_0t & \cos\omega_0t
\end{pmatrix}
\begin{pmatrix}q(0)\\p(0)\end{pmatrix}.
$$
该矩阵行列式为 $1$，直接给出 $dq(t)\wedge dp(t)=dq(0)\wedge dp(0)$；同时 $X_HH=\{H,H\}=0$，所以轨道位于能量椭圆上。

## 练习

**练习 2.1.** 在 $T^*\mathbb R^n$ 上取 $\omega=dq^i\wedge dp_i$，计算 $X_f$ 的坐标表达。

**练习 2.2.** 设 $SO(3)$ 作用在 $T^*\mathbb R^3$ 上。证明角动量 $L=q\times p$ 是 moment map 的坐标表达。
