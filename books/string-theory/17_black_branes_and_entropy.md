# 第十七章：black branes、BPS states 和黑洞熵

## 本章目标

本章说明 D-branes 如何用于计算某些 supersymmetric black holes 的微观熵。核心逻辑是：

1. BPS 条件保护态数或指数；
2. 弱耦合 D-brane bound states 可被二维 CFT 或量子力学计数；
3. 强耦合下同一电荷态表现为 black brane 或 black hole；
4. 大电荷极限中 microscopic entropy 与 Bekenstein-Hawking entropy 匹配。

## 依赖前置知识

需要第十二章 D-branes、第十四章 dualities、第十一章 supergravity 和第十五章高 genus/factorization 的一致性接口。

## 17.1 BPS states 与 index

**定义 17.1（BPS state）.** BPS state 是饱和 supersymmetry algebra 中质量下界的态：
$$
M=|Z(Q)|,
$$
其中 $Z(Q)$ 是由电荷 $Q$ 决定的 central charge 或 central charge eigenvalue。

**命题 17.2（BPS multiplet shortening）.** BPS state 被部分 supercharges 湮灭，
因此位于短表示中。在存在离散谱、与 continuum 有能隙且不发生短表示重组的参数区，
相应 supersymmetric index 在连续耦合变形下不变；未经加权的绝对 BPS degeneracy
不由 shortening 单独保证不变。

**推导说明（标准物理口径）.** Supersymmetry algebra 的正性给出 $M\ge |Z|$。
若等号成立，某些 supercharge 组合具有零范数并湮灭该态，从而表示比一般 massive
multiplet 短。Index 对可配对的 boson/fermion states 抵消；只有当态进入 continuum、
短表示彼此重组或穿过 stability wall 时，这个论证的假设失效。$\square$

**定义 17.3（BPS index）.** BPS index 是带符号的态数，例如
$$
\Omega(Q)=\operatorname{Tr}_{\mathcal H_Q}(-1)^F
$$
或带角动量插入的 refined index。Index 比绝对 degeneracy 更稳定。

**注 17.4.** 黑洞熵通常比较的是大电荷极限下的增长率。Index 与 degeneracy 的
指数增长一致是许多 supersymmetric 系统中可检验、但并非自动成立的额外命题；
若存在指数级 boson/fermion cancellation，就不能用 $\log|\Omega|$ 代替
$\log d$。

**定义 17.4A（helicity supertrace）.** 在四维理论中，常用 helicity supertrace
$$
B_{2n}(Q)=\frac1{(2n)!}\operatorname{Tr}_{\mathcal H_Q}
(-1)^{2J_3}(2J_3)^{2n}
$$
作为 BPS index。插入角动量因子是为了吸收 broken supersymmetry zero modes。

**命题 17.4B（wall crossing 的必要性）.** BPS index 只在 moduli space 的 chamber 内保持不变；穿过 marginal stability wall 时，bound state 可衰变，index 可按 wall-crossing formula 跳变。

**推导说明（标准物理口径）.** BPS bound state 的稳定性依赖 constituent central charges 的相位。当相位对齐时，结合能可变为零，bound state 达到 marginal decay threshold。此时 Hilbert space 的 BPS sector 发生重组，index 允许跳变。$\square$

## 17.2 Black hole entropy

**标准半经典输入 17.5（Bekenstein--Hawking 公式）.** 对由两导数 Einstein
作用量控制、具有正则 stationary horizon 的 black hole，领先半经典热力学熵为
$$
S_{\mathrm{BH}}=\frac{A_H}{4G_N},
$$
其中取 $\hbar=1$，$A_H$ 是 horizon area。该式由 Euclidean saddle/black-hole
mechanics 等标准引力论证支持，本书不把它定义成任意高导数或量子引力背景中的精确熵。

**注 17.6（修正）.** 高导数 $\alpha'$ corrections 和 quantum corrections 会把面积律修正为 Wald entropy 或更完整的 quantum entropy。Strominger-Vafa 型领先阶匹配使用大电荷、弱曲率极限。

## 17.3 D1-D5-P 系统

考虑 type IIB string theory on
$$
S^1\times X_4,
$$
其中 $X_4=T^4$ 或 $K3$。取

1. $Q_1$ 个 D1-branes 包裹 $S^1$；
2. $Q_5$ 个 D5-branes 包裹 $S^1\times X_4$；
3. 沿 $S^1$ 的 momentum number $n$。

**外部输入定理 17.7（D1-D5 CFT central charge）.** D1-D5 bound state 的低能二维 CFT 在适当模空间点具有 central charge
$$
c=6Q_1Q_5.
$$

**外部输入定理 17.8（Cardy asymptotics）.** 设二维 CFT unitary、modular
invariant，圆柱谱离散、真空唯一，并固定 central charge $c$。若 $n$ 是相对基态的
chiral excitation level，则在 $n\to\infty$ 时
$$
\log d(n)
=2\pi\sqrt{\frac{c_{\mathrm{eff}}n}{6}}
+o(\sqrt n),
$$
其中在上述 unitary vacuum 假设下 $c_{\mathrm{eff}}=c$。本书不重证由 modular
$S$ transformation 与 inverse-Laplace saddle 得到的 Cardy theorem。

**推论 17.8A（D1--D5--P 的严格 Cardy 区）.** 使用外部输入 17.7。对固定
$Q_1,Q_5$ 令 $n\to\infty$；若同时放大 charges，则另假设 Cardy saddle 的误差在
$n/(Q_1Q_5)\to\infty$ 区域一致。弱耦合 D1--D5 CFT 的 degeneracy 满足
$$
\log d(Q_1,Q_5,n)
=2\pi\sqrt{Q_1Q_5n}
+o\!\left(\sqrt{Q_1Q_5n}\right).
$$

**证明.** 把 $c=6Q_1Q_5$ 代入定理 17.8 的 leading term：
$$
2\pi\sqrt{\frac{(6Q_1Q_5)n}{6}}
=2\pi\sqrt{Q_1Q_5n}.
$$
余项随同保留。这个有限代入没有证明 Cardy theorem，也没有证明 index cancellation
不改变 leading exponent。$\square$

**外部输入定理 17.9（D1--D5--P supergravity area calculation）.** 对
D1--D5--P 五维 supersymmetric black hole，在 charges 取大并使 horizon curvature
远低于 string scale、string loops 受控的两导数 supergravity regime 中，charge
normalization 可取为
$$
S_{\mathrm{BH}}
=2\pi\sqrt{Q_1Q_5 n}
+o\!\left(\sqrt{Q_1Q_5 n}\right).
$$
因此其 leading term 与推论 17.8A 的 microscopic growth 匹配。

**使用边界.** 本书不推导相应 supergravity black-hole metric 的 horizon area。
该匹配是受控参数区中的 leading asymptotic，不是有限 charges 上
$S_{\mathrm{BH}}=\log d$ 的精确等式。跨越弱、强耦合还需选取受保护 index、排除
wall crossing，并验证其 leading growth 不被符号 cancellation 改变；这些是
Strominger--Vafa argument 的物理输入边界。

## 17.4 Attractor mechanism 的接口

**定义 17.10（attractor behavior）.** Supersymmetric black holes 的 near-horizon moduli 在许多情形下只由电荷决定，而与无穷远处 moduli 无关。这称为 attractor mechanism。

**外部输入定理 17.11（两导数 BPS attractor 接口）.** 在给定 supersymmetric
supergravity 中，若 charge $Q$ 支持正则、single-center、两导数 BPS attractor，且
attractor equations 有位于物理 moduli space 内的解，则 fixed scalars 的
near-horizon 值由 $Q$ 与 attractor branch 决定，leading area entropy 不依赖其
无穷远边界值。Flat directions、multi-center solutions 与 singular attractors 不在
该陈述内。

**证明路线（外部输入）.** BPS flow equations 把 near-horizon scalar values 固定为
central charge 或相应 black-hole potential 的 extremum，horizon area 由该 extremum
处的 charge data 决定。完整证明依赖具体 supergravity、regularity 与 global flow
存在性；本书不把这条局部路线升级为无条件命题。

## 17.5 高导数修正和指数

**外部输入定理 17.12（Wald Noether-charge entropy）.** 对由局部
diffeomorphism-invariant action
$I=\int d^Dx\sqrt{-g}\,\mathcal L(g,R,\ldots)$ 描述、具有 bifurcate
Killing horizon 的 stationary classical solution，若 $\mathcal L$ 不含 Riemann tensor
的导数，则
$$
S_{\mathrm{Wald}}
=-2\pi\int_H d^{D-2}x\sqrt h\,
\frac{\partial\mathcal L}{\partial R_{\mu\nu\rho\sigma}}
\varepsilon_{\mu\nu}\varepsilon_{\rho\sigma}.
$$
偏导把 $g_{\mu\nu}$ 与 $R_{\mu\nu\rho\sigma}$ 视为独立 algebraic variables，
$\varepsilon_{\mu\nu}\varepsilon^{\mu\nu}=-2$。
对 Einstein--Hilbert Lagrangian，该有限变分计算退化为 $A_H/(4G_N)$。含
$\nabla R$ 时需使用推广公式。

**研究边界 17.12A（量子熵与微观指数）.** 把受保护微观 index 与包含
higher-derivative、one-loop 和 nonperturbative corrections 的宏观 quantum entropy
逐阶相等，不是一般已证定理。$AdS_2$ path integral、supersymmetric localization、
anomaly 与 topological-string 方法在特定 compactification 中给出强检验；每个应用
仍须声明 ensemble、measure、zero modes、contour 和 wall-crossing chamber。

## 本章小结

Black brane entropy 是 string theory 非微扰结构的关键检验。D-brane 计数并不是对所有黑洞的完整解释，但它在 supersymmetric、受保护、大电荷系统中给出可计算且与几何熵一致的微观态增长。

## 练习

**练习 17.1.** 说明 BPS 条件为什么有助于跨耦合常数比较态数。

**练习 17.2.** 用 Cardy formula 推导 D1-D5-P 系统的 leading entropy。

**练习 17.3.** 说明为什么 BPS index 可能在 wall of marginal stability 上跳变。
