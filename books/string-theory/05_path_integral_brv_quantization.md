# 第五章：路径积分、ghost 和 BRST 量子化

## 本章目标

本章从 Polyakov path integral 出发，说明 gauge fixing 如何产生 $bc$ ghosts，为什么临界维数等价于总 central charge 为零，以及 BRST cohomology 如何定义物理态。相对于第四章 old covariant quantization，BRST 语言更适合描述 gauge equivalence、顶点算子和散射振幅。

## 依赖前置知识

需要第三章的 CFT 和第四章的 Virasoro constraints。BRST convention 见 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md)。

## 5.1 Polyakov path integral

**定义 5.1（形式 Polyakov path integral）.** Euclidean Polyakov path integral 写为
$$
Z=\int\frac{\mathcal D h\,\mathcal D X}
{\operatorname{Diff}(\Sigma)\times\operatorname{Weyl}(\Sigma)}
e^{-S_E[X,h]}.
$$

**注 5.2（形式性）.** 该表达式是物理定义式。严格定义需要选择 regulator、处理 gauge group 体积、处理 Riemann surface moduli 和 operator insertions。本书使用其标准扰动论接口。

**定义 5.3（genus expansion）.** 常 dilaton 背景下，闭弦 genus $g$ 世界面权重为
$$
g_s^{2g-2}.
$$
带 $n$ 个闭弦外态的振幅常整体写成 $g_s^{2g-2+n}$，其中外态归一化可吸收部分 $g_s$ convention。

## 5.2 Gauge fixing、ghost action 和 central charge

**定义 5.4（metric fluctuation 分解）.** 在 conformal gauge 附近，metric fluctuation 可分解为
$$
\delta h_{ab}
=(\mathcal L_v h)_{ab}+2\omega h_{ab}
+\sum_I \delta m^I\,\mu_{I,ab},
$$
其中前三项分别是 diffeomorphism、Weyl 和 moduli 方向。

**定义 5.5A（ghost action）.** Faddeev-Popov determinant 可由 anticommuting ghosts $b_{ab},c^a$ 表示。复坐标中其 action 为
$$
S_{bc}=\frac1{2\pi}\int d^2z\,
\left(b_{zz}\bar\partial c^z+b_{\bar z\bar z}\partial c^{\bar z}\right).
$$
Holomorphic fields 记为 $b(z),c(z)$，权重分别为 $2,-1$。

**命题 5.5（ghost central charge）.** Reparametrization ghost system 的 central charge 为
$$
c_{bc}=-26.
$$

**证明.** 这是第三章命题 3.11 的 $bc$ system central charge 计算，权重为 $(2,-1)$。$\square$

**推论 5.6（玻色弦临界维数）.** 玻色弦量子 Weyl anomaly 消失要求
$$
c_{\mathrm{matter}}+c_{\mathrm{ghost}}=D-26=0.
$$
因此临界维数为
$$
D=26.
$$

**证明.** Weyl invariance 在量子理论中要求总 stress tensor trace anomaly 消失。二维 CFT 中该 anomaly 由 total central charge 控制。Matter free bosons 贡献 $D$，ghosts 贡献 $-26$。$\square$

## 5.3 BRST symmetry

**定义 5.7（BRST current and charge）.** BRST current 可写为
$$
j_B(z)=c(z)\left(T_m(z)+\frac12T_{bc}(z)\right)+\frac32\partial^2c(z)
$$
在标准玻色弦 convention 下成立。BRST charge 为
$$
Q_B=\oint\frac{dz}{2\pi i}\,j_B(z)
$$
加上 antiholomorphic 部分。Mode 形式为
$$
Q_B=\sum_n c_{-n}L_n^{m}
-\frac12\sum_{m,n}(m-n):c_{-m}c_{-n}b_{m+n}:
-a c_0.
$$

**定义 5.8（BRST cohomology）.** BRST 物理态空间定义为 cohomology
$$
\mathcal H_{\mathrm{phys}}=H^\bullet(Q_B)
=\frac{\ker Q_B}{\operatorname{im}Q_B}.
$$
即
$$
Q_B|\psi\rangle=0,\qquad
|\psi\rangle\sim|\psi\rangle+Q_B|\chi\rangle.
$$

**命题 5.9（nilpotency 条件）.** 在玻色弦中，$Q_B^2=0$ 要求总 central charge 为零，并固定 normal ordering constant 为临界值。

**证明草图.** 计算 BRST current 的 OPE $j_B(z)j_B(w)$。若总 central charge 不为零，OPE 中会出现不可写成 total derivative 的高阶极点；contour integration 后给出 $Q_B^2\ne0$。同时，$L_0$ 中的 normal ordering ambiguity 在 mode 计算中表现为 $c_0$ 项，nilpotency 固定 $a=1$。$\square$

**命题 5.10（BRST exact state 的 decoupling）.** 若外态之一为 BRST exact，即 $|\psi\rangle=Q_B|\chi\rangle$，则在无 anomaly 且 moduli 边界项受控的振幅中该外态 decouple。

**证明草图.** 将 $Q_B$ 写成 BRST current 的 contour integral。Contour 可在世界面上移动；穿过 BRST-closed 插入时无贡献，最后收缩为空或化为 moduli space 边界项。对 tree-level on-shell 振幅，边界项给出物理因子化且 exact state 不贡献。$\square$

## 5.4 Ghost zero modes 和顶点插入

**定义 5.11（unintegrated 与 integrated insertions）.** 在具有 conformal Killing vectors 的曲面上，需要用 $c$ ghost 插入吸收 ghost zero modes。Sphere 上闭弦 tree amplitude 通常取三个未积分顶点
$$
c\bar c V
$$
和其余积分顶点
$$
\int d^2z\,V.
$$

**命题 5.11A（sphere ghost zero mode counting）.** Riemann sphere 的 holomorphic conformal Killing vectors 维数为 $3$，因此闭弦 sphere amplitude 需要三个 $c$ 与三个 $\bar c$ zero modes。

**证明.** Sphere 的全纯自同构群为 $PSL_2(\mathbb C)$。其 holomorphic vector fields 由
$$
1,\quad z,\quad z^2
$$
张成，对应三个 $c$ ghost zero modes。Antiholomorphic 部分同理。$\square$

## 5.5 Moduli 分解

**外部输入定理 5.12（Polyakov path integral 的 moduli 分解）.** 固定 genus $g$ 的闭弦扰动论振幅可写成 Riemann surface moduli space $\mathcal M_{g,n}$ 上的积分，积分测度由 matter CFT、ghost determinant、vertex operator insertions 和 Beltrami differentials 给出。

**注 5.13.** 该定理依赖 Riemann surface theory、ghost zero modes 和 gauge slice 的选择。第十五章将回到高 genus 扰动论。

## 5.6 Genus-one measure 示例

**定义 5.14（torus vacuum amplitude）.** 闭弦 genus-one vacuum amplitude 可写为
$$
\mathcal Z_1
=\int_{\mathcal F}\frac{d^2\tau}{2\tau_2^2}\,
Z_X(\tau,\bar\tau)Z_{bc}(\tau,\bar\tau),
$$
其中 $\mathcal F$ 是 $SL(2,\mathbb Z)$ fundamental domain。对 $D$ 个非紧 free bosons，
$$
Z_X=V_D(4\pi^2\alpha'\tau_2)^{-D/2}|\eta(\tau)|^{-2D}.
$$
Ghost contribution 为
$$
Z_{bc}=\tau_2|\eta(\tau)|^4
$$
在常用 zero-mode convention 下成立。

**命题 5.15（critical bosonic torus integrand）.** 对 $D=26$ 的临界玻色弦，
$$
\mathcal Z_1
=\frac{V_{26}}2\int_{\mathcal F}\frac{d^2\tau}{\tau_2}
(4\pi^2\alpha'\tau_2)^{-13}
|\eta(\tau)|^{-48}
$$
至整体规范化 convention 成立。

**证明.** 将定义 5.14 中的 $Z_X$ 与 $Z_{bc}$ 相乘，并取 $D=26$：
$$
|\eta|^{-52}|\eta|^4=|\eta|^{-48},
\qquad
\frac{d^2\tau}{2\tau_2^2}\tau_2
=\frac12\frac{d^2\tau}{\tau_2}.
$$
$\square$

## 本章小结

Gauge fixing 不是无害步骤：它引入 ghost CFT、zero mode selection rules 和 BRST cohomology。临界维数 $26$ 是 quantum Weyl invariance 的结果；物理态不是任意 Fock states，而是 BRST cohomology classes。

## 练习

**练习 5.1.** 解释为什么 $c_{\mathrm{matter}}+c_{\mathrm{ghost}}=0$ 是 Weyl anomaly cancellation 条件。

**练习 5.2.** 说明 BRST exact state 为什么应被视为零物理态。

**练习 5.3.** 解释 sphere tree amplitude 为什么需要三个未积分闭弦顶点。

**练习 5.4.** 从 $Z_X$ 和 $Z_{bc}$ 推导临界玻色弦 torus integrand 中的 $|\eta|^{-48}$。
