# 第三章：二维共形场论和 OPE 语言

## 本章目标

本章建立世界面 CFT 的基本语言：Euclidean continuation、radial quantization、operator product expansion、Ward identity、Virasoro algebra、primary fields、free boson 和 ghost CFT。后续 string spectrum、BRST cohomology 和 scattering amplitudes 都依赖这些工具。

## 依赖前置知识

需要复分析、第一章的 stress tensor 和第二章的 conformal gauge。OPE 归一化见 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md)。

## 3.1 Euclidean worldsheet、复坐标和径向量子化

**定义 3.1（复坐标）.** Wick rotation 后取局部复坐标
$$
z=\sigma^1+i\sigma^2,\qquad
\bar z=\sigma^1-i\sigma^2.
$$
在圆柱到平面的映射中常取
$$
z=e^{\tau_E+i\sigma},\qquad \bar z=e^{\tau_E-i\sigma}.
$$
二维 CFT 的 stress tensor 分解为 holomorphic 与 antiholomorphic 部分：
$$
T(z),\qquad \bar T(\bar z).
$$

**定义 3.2（OPE）.** 两个局部算子的 OPE 是短距离渐近展开
$$
\mathcal O_i(z)\mathcal O_j(w)
\sim
\sum_k C_{ij}^{\ k}(z-w,\bar z-\bar w)\mathcal O_k(w).
$$
符号 $\sim$ 表示只记录在 correlation functions 中决定 $z\to w$ 奇异行为的项。

**定义 3.3（径向序）.** 平面 CFT 中径向量子化把 $|z|$ 视为 Euclidean time。径向序 $R(\mathcal O_1(z)\mathcal O_2(w))$ 按 $|z|>|w|$ 排列算子。

**引理 3.3A（contour deformation 原理）.** 若 current $j(z)$ holomorphic 除了在算子插入处有极点，则
$$
\oint_C dz\,j(z)
$$
的值只由 $C$ 围住的插入点处 OPE residue 决定。

**证明.** 由 Cauchy theorem，contour 可在不穿过极点的区域连续变形。穿过算子插入点时，变化等于该点 OPE 中 $1/(z-w)$ 项的 residue。$\square$

## 3.2 Free boson CFT

**定义 3.4A（free boson OPE）.** $D$ 个 free bosons $X^\mu$ 的基本 OPE 为
$$
X^\mu(z,\bar z)X^\nu(w,\bar w)
\sim
-\frac{\alpha'}{2}\eta^{\mu\nu}\log|z-w|^2.
$$
因此 holomorphic derivatives 满足
$$
\partial X^\mu(z)\partial X^\nu(w)
\sim
-\frac{\alpha'}2\frac{\eta^{\mu\nu}}{(z-w)^2}.
$$

**定义 3.4B（free boson stress tensor）.** Holomorphic stress tensor 为
$$
T(z)=-\frac1{\alpha'}:\partial X^\mu\partial X_\mu:(z).
$$

**命题 3.4（free boson Virasoro OPE）.** Free boson stress tensor 满足
$$
T(z)T(w)\sim
\frac{D/2}{(z-w)^4}
+\frac{2T(w)}{(z-w)^2}
+\frac{\partial T(w)}{z-w}.
$$
因此 central charge 为 $c=D$。

**证明.** 用 Wick theorem 对
$$
:\partial X^\mu\partial X_\mu:(z):\partial X^\nu\partial X_\nu:(w)
$$
作收缩。双收缩给出
$$
\frac1{\alpha'^2}\cdot 2\cdot
\left(\frac{\alpha'}2\right)^2
\frac{D}{(z-w)^4}
=\frac{D/2}{(z-w)^4}.
$$
单收缩给出
$$
\frac{2T(w)}{(z-w)^2}
+\frac{\partial T(w)}{z-w}.
$$
$\square$

**命题 3.7（指数算子的 Wick 规则）.** 对 normal ordered exponentials，
$$
\left\langle\prod_{i=1}^n:e^{ik_i\cdot X(z_i,\bar z_i)}:\right\rangle
=
(2\pi)^D\delta^{(D)}\!\left(\sum_i k_i\right)
\prod_{i<j}|z_i-z_j|^{\alpha' k_i\cdot k_j}.
$$

**证明草图.** 将 $X=x+X'$ 分为零模和非零模。零模积分给出动量守恒；非零模 Gaussian integral 给出 pairwise contractions。$\square$

## 3.3 Primary fields 和 Ward identity

**定义 3.5（primary field）.** 若局部算子 $\mathcal O(w)$ 满足
$$
T(z)\mathcal O(w)\sim
\frac{h\mathcal O(w)}{(z-w)^2}
+\frac{\partial\mathcal O(w)}{z-w},
$$
则称其为 holomorphic conformal weight $h$ 的 primary field。Antiholomorphic weight $\bar h$ 同理定义。

**命题 3.6（tachyon vertex 的 conformal weights）.** Tachyon vertex operator
$$
V_k(z,\bar z)=:e^{ik\cdot X(z,\bar z)}:
$$
的 conformal weights 为
$$
h=\bar h=\frac{\alpha' k^2}{4}.
$$

**证明.** 用
$$
\partial X^\mu(z)V_k(w,\bar w)
\sim
-i\frac{\alpha'}2\frac{k^\mu}{z-w}V_k(w,\bar w)
$$
计算 $T(z)V_k(w)$。双收缩给出二阶极点
$$
\frac{\alpha'k^2/4}{(z-w)^2}V_k,
$$
单收缩给出一阶极点 $\partial V_k/(z-w)$。$\square$

**命题 3.9（conformal Ward identity）.** 对 primary fields $\mathcal O_i(z_i)$，
$$
\left\langle T(z)\prod_i\mathcal O_i(z_i)\right\rangle
=
\sum_i\left(
\frac{h_i}{(z-z_i)^2}
+\frac1{z-z_i}\partial_{z_i}
\right)
\left\langle\prod_i\mathcal O_i(z_i)\right\rangle.
$$

**证明.** 将 $T(z)$ 的 contour 移动到各个插入点附近，并使用 primary OPE 的二阶和一阶极点。$\square$

**外部输入定理 3.8（state-operator correspondence）.** 在满足标准公理的二维 CFT 中，圆柱量子化的态空间与局部算子空间之间存在对应，Hamiltonian 由 $L_0+\bar L_0-c/12$ 控制。

## 3.4 Virasoro algebra 和 modes

**定义 3.9A（Virasoro modes）.** Virasoro generators 定义为
$$
L_n=\frac{1}{2\pi i}\oint dz\,z^{n+1}T(z).
$$
若 $\mathcal O$ 是 primary，则
$$
[L_n,\mathcal O(w)]
=
\left(w^{n+1}\partial_w+h(n+1)w^n\right)\mathcal O(w).
$$

**命题 3.10（Virasoro algebra）.** 若 $T(z)T(w)$ 的 OPE 具有 central charge $c$，则
$$
[L_m,L_n]=(m-n)L_{m+n}
+\frac{c}{12}m(m^2-1)\delta_{m+n,0}.
$$

**证明草图.** 将 commutator 写成嵌套 contour integral：
$$
[L_m,L_n]
=
\oint_0\frac{dw}{2\pi i}w^{n+1}
\oint_w\frac{dz}{2\pi i}z^{m+1}T(z)T(w).
$$
取 $T(z)T(w)$ OPE 的 residues，二阶和一阶极点给出 $(m-n)L_{m+n}$，四阶极点给出中心项。$\square$

## 3.5 Ghost CFT

**定义 3.11A（$bc$ ghosts）.** Reparametrization ghosts 是 conformal weights $(2,-1)$ 的 anticommuting fields $b,c$，满足
$$
b(z)c(w)\sim\frac1{z-w}.
$$
其 stress tensor 可写为
$$
T_{bc}(z)=-2:b\partial c:(z)-:(\partial b)c:(z).
$$

**命题 3.11（$bc$ central charge）.** 权重 $(2,-1)$ 的 $bc$ system central charge 为
$$
c_{bc}=-26.
$$

**证明草图.** 一般 $bc$ system 权重 $(\lambda,1-\lambda)$ 的 central charge 为
$$
c=1-3(2\lambda-1)^2.
$$
代入 $\lambda=2$ 得 $c=-26$。该公式可由 $T_{bc}(z)T_{bc}(w)$ 的 fermionic Wick contraction 直接推出。$\square$

**定义 3.12（ghost number current）.** $bc$ system 的 ghost number current 为
$$
j_{\mathrm{gh}}(z)=-:bc:(z).
$$
其 zero mode $N_{\mathrm{gh}}$ 计数 $c$ 与 $b$ 激发的 ghost number 差。

**注 3.13.** 在 sphere tree amplitude 中，$c$ ghost zero modes 对应 conformal Killing group $PSL_2(\mathbb C)$。这就是第六章未积分顶点需要插入 $c\bar c$ 的原因。

## 3.6 Virasoro 表示和 null states

**定义 3.14（highest-weight state）.** Virasoro 表示中的 highest-weight state $|h\rangle$ 满足
$$
L_0|h\rangle=h|h\rangle,\qquad
L_n|h\rangle=0\quad(n>0).
$$
由负模
$$
L_{-n_1}\cdots L_{-n_k}|h\rangle,\qquad n_i>0,
$$
生成的表示称为 Verma module，level 为 $\sum_i n_i$。

**命题 3.15（低 level Gram matrix）.** 对归一化 $\langle h|h\rangle=1$ 的 highest-weight state，level $1$ descendant 的范数为
$$
\langle h|L_1L_{-1}|h\rangle=2h.
$$
Level $2$ 基
$$
L_{-2}|h\rangle,\qquad L_{-1}^2|h\rangle
$$
的 Gram matrix 为
$$
\begin{pmatrix}
4h+\frac c2 & 6h\\
6h & 4h(2h+1)
\end{pmatrix}.
$$

**证明.** 使用 Virasoro algebra 和 $L_n|h\rangle=0$。例如
$$
\langle h|L_1L_{-1}|h\rangle
=\langle h|[L_1,L_{-1}]|h\rangle
=2h.
$$
Level $2$ 的各项同理由交换关系
$$
[L_2,L_{-2}]=4L_0+\frac c2,\qquad
[L_1,L_{-2}]=3L_{-1}
$$
计算得到。$\square$

**定义 3.16（null state）.** Verma module 中非零向量 $|\chi\rangle$ 若自身为 highest-weight descendant 且范数为零，则称为 null state。Null states 生成的子表示应在 unitary CFT 的物理 Hilbert space 中商去。

**注 3.17.** Minimal models、BRST cohomology 和 string physical state conditions 都会用到 null states。完整 Kac determinant formula 属于二维 CFT 表示论，本书只使用低 level 计算和 null state decoupling 的接口。

## 本章小结

玻色弦量子化的核心是二维 CFT。Matter fields 给出 $c=D$，ghosts 给出 $c=-26$；primary weights 给出 on-shell 条件；contour deformation 和 OPE residues 给出 Ward identities、Virasoro algebra 与 BRST 计算。

## 练习

**练习 3.1.** 用 free boson OPE 计算 $\partial X^\mu(z)e^{ik\cdot X(w)}$ 的奇异部分。

**练习 3.2.** 推导 primary field 在 $L_0$ 和 $L_{-1}$ 下的变换。

**练习 3.3.** 用 $T(z)T(w)$ OPE 的 contour integral 推导 Virasoro algebra 的中心项。

**练习 3.4.** 计算 level $1$ descendant $L_{-1}|h\rangle$ 的范数，并说明 $h=0$ 时发生什么。

