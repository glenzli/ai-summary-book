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

**命题 17.2（BPS multiplet shortening）.** BPS state 被部分 supercharges 湮灭，因此位于短表示中。短表示的态数在不穿过 wall of marginal stability 时受连续耦合变形保护。

**证明草图.** Supersymmetry algebra 的正性给出 $M\ge |Z|$。若等号成立，某些 supercharge 组合具有零范数并湮灭该态，从而表示比一般 massive multiplet 短。短表示不能在连续变形下变成长表示，除非与其他短表示在 wall crossing 中重组。$\square$

**定义 17.3（BPS index）.** BPS index 是带符号的态数，例如
$$
\Omega(Q)=\operatorname{Tr}_{\mathcal H_Q}(-1)^F
$$
或带角动量插入的 refined index。Index 比绝对 degeneracy 更稳定。

**注 17.4.** 黑洞熵通常比较的是大电荷极限下的增长率。即使 index 与 degeneracy 不完全相同，其指数增长在许多 supersymmetric 系统中一致。

**定义 17.4A（helicity supertrace）.** 在四维理论中，常用 helicity supertrace
$$
B_{2n}(Q)=\frac1{(2n)!}\operatorname{Tr}_{\mathcal H_Q}
(-1)^{2J_3}(2J_3)^{2n}
$$
作为 BPS index。插入角动量因子是为了吸收 broken supersymmetry zero modes。

**命题 17.4B（wall crossing 的必要性）.** BPS index 只在 moduli space 的 chamber 内保持不变；穿过 marginal stability wall 时，bound state 可衰变，index 可按 wall-crossing formula 跳变。

**证明草图.** BPS bound state 的稳定性依赖 constituent central charges 的相位。当相位对齐时，结合能可变为零，bound state 达到 marginal decay threshold。此时 Hilbert space 的 BPS sector 发生重组，index 允许跳变。$\square$

## 17.2 Black hole entropy

**定义 17.5（Bekenstein-Hawking entropy）.** Classical black hole 的几何熵为
$$
S_{\mathrm{BH}}=\frac{A_H}{4G_N},
$$
其中 $A_H$ 是 horizon area。

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

**命题 17.8（Cardy entropy）.** 对左移动激发数 $n$ 的大电荷态，Cardy formula 给出
$$
S_{\mathrm{micro}}
=2\pi\sqrt{Q_1Q_5 n}.
$$

**证明.** 二维 unitary modular invariant CFT 的 Cardy formula 为
$$
S=2\pi\sqrt{\frac{c\,n}{6}}
$$
在大 $n$ 极限成立。代入 $c=6Q_1Q_5$ 得结论。$\square$

**外部输入定理 17.9（Strominger-Vafa entropy matching）.** 对 D1-D5-P 五维 supersymmetric black hole，在大电荷极限下
$$
S_{\mathrm{BH}}=2\pi\sqrt{Q_1Q_5 n}
$$
并与命题 17.8 的 microscopic entropy 匹配。

**使用边界.** 本书不推导相应 supergravity black hole metric 的 horizon area；只使用其标准结果与 D-brane CFT 计数的匹配。

## 17.4 Attractor mechanism 的接口

**定义 17.10（attractor behavior）.** Supersymmetric black holes 的 near-horizon moduli 在许多情形下只由电荷决定，而与无穷远处 moduli 无关。这称为 attractor mechanism。

**命题 17.11（熵的 moduli independence）.** 若 attractor mechanism 适用，则 BPS black hole 的 leading entropy 是电荷的不变量，可与弱耦合 D-brane 计数比较。

**证明草图.** BPS equations 把 near-horizon scalar values 固定为 central charge 的 extremum。Horizon area 由该 extremum 处的 charge data 决定，不依赖无穷远 moduli。$\square$

## 17.5 高导数修正和指数

**命题 17.12（Wald entropy 与微观指数）.** 当低能有效作用含有高导数项时，几何侧应使用 Wald entropy 或 quantum entropy，而不是单纯面积律；受保护的微观指数应与修正后的宏观熵比较。

**证明草图.** 高导数项改变黑洞 Noether charge entropy。Supersymmetric localization 和 anomaly 方法表明，某些受保护修正可由近地平线 $AdS_2$ path integral 或 topological string amplitudes 捕捉。完整形式依赖具体 compactification。$\square$

## 本章小结

Black brane entropy 是 string theory 非微扰结构的关键检验。D-brane 计数并不是对所有黑洞的完整解释，但它在 supersymmetric、受保护、大电荷系统中给出可计算且与几何熵一致的微观态增长。

## 练习

**练习 17.1.** 说明 BPS 条件为什么有助于跨耦合常数比较态数。

**练习 17.2.** 用 Cardy formula 推导 D1-D5-P 系统的 leading entropy。

**练习 17.3.** 说明为什么 BPS index 可能在 wall of marginal stability 上跳变。

