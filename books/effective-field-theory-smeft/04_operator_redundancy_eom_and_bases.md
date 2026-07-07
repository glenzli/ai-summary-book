# 第四章：算符冗余、EOM 与算符基

## 本章目标

本章解释为什么“所有对称性允许的算符”仍然有冗余，以及如何通过分部积分、运动方程、Bianchi 恒等式和 Fierz 恒等式选取算符基。

## 依赖前置知识

需要局域算符、S-matrix 等价的基本观念和场论变分。

## 4.1 分部积分冗余

**定义 4.1（总导数等价）.** 若两个拉氏量密度相差总导数
$$
\mathcal L_1-\mathcal L_2=\partial_\mu K^\mu,
$$
且边界项不贡献作用量，则称它们在作用量中等价。

**命题 4.2（分部积分删除导数分配冗余）.** 在平直时空、散射边界条件下，总导数项不改变 S-matrix。

**证明（书内推导）.** 作用量差为
$$
\Delta S=\int d^4x\,\partial_\mu K^\mu
=\int_{\partial M} d\Sigma_\mu K^\mu.
$$
若场在边界衰减或采用周期边界条件，则边界积分为零。路径积分相位不变，因此由该差别引起的相关函数变化只含边界项。$\square$

## 4.2 EOM 冗余

**定义 4.3（EOM 冗余算符）.** 若算符可写为
$$
\mathcal O
=
F[\phi]\frac{\delta S_0}{\delta\phi},
$$
则称它相对于领先作用量 $S_0$ 是 EOM 冗余的。

**命题 4.4（局域场重定义删除 EOM 算符）.** 在一阶微扰中，场重定义
$$
\phi\mapsto \phi+\epsilon F[\phi]
$$
使领先作用量变化为
$$
S_0[\phi+\epsilon F]
=
S_0[\phi]
+
\epsilon\int d^4x\,F[\phi]\frac{\delta S_0}{\delta\phi}
+
O(\epsilon^2).
$$
因此 EOM 冗余算符可被场重定义移入其他算符系数。

**证明（书内推导）.** 对泛函 $S_0$ 作一阶 Taylor 展开即得。$\square$

**警告 4.5.** EOM 删除不是说算符在所有场构型上为零，而是说它对 on-shell 物理可观测量可由场重定义吸收。off-shell Green 函数和中间 Wilson 系数会改变。

**例 4.6（标量 EOM 冗余）.** 设领先拉氏量为
$$
{\cal L}_0={1\over2}(\partial\phi)^2-{1\over2}m^2\phi^2-{\lambda\over4!}\phi^4.
$$
其 EOM 为
$$
\Box\phi+m^2\phi+{\lambda\over6}\phi^3=0.
$$
考虑维数六形式的算符
$$
{\cal O}_{\rm EOM}=\phi^3\Box\phi.
$$
利用 EOM 得
$$
\phi^3\Box\phi
=-m^2\phi^4-{\lambda\over6}\phi^6.
$$
因此含 $\phi^3\Box\phi$ 的 EFT 可通过场重定义改写为 $\phi^4$ 和 $\phi^6$ 系数的移动。若只看 off-shell 四点 Green 函数，二者形式不同；若看 on-shell S-matrix，差别被参数重定义吸收。

**例 4.7（分部积分与 EOM 的组合）.**
$$
\partial_\mu(\phi^3\partial^\mu\phi)
=3\phi^2(\partial\phi)^2+\phi^3\Box\phi.
$$
故
$$
\phi^2(\partial\phi)^2
\simeq
-{1\over3}\phi^3\Box\phi,
$$
其中 $\simeq$ 表示相差总导数。再用 EOM 可把它换成势能型算符。这说明算符冗余常需同时使用分部积分和 EOM。

## 4.3 算符基

**定义 4.8（算符基）.** 给定对称性、维数和冗余关系后，若一组算符张成所有等价类且彼此独立，则称其为该阶 EFT 的一个算符基。

**例 4.9（基不是唯一的）.** Warsaw basis、SILH basis、Higgs basis 是 SMEFT 中不同用途的坐标选择。它们描述同一物理空间的不同局部坐标，但截断和非线性变换会使实际拟合中出现方案差异。

## 4.4 其他冗余关系

**Bianchi 恒等式。** 对非阿贝尔场强，
$$
D_{[\mu}X_{\nu\rho]}=0.
$$
含 $D_\mu\widetilde X^{\mu\nu}$ 的算符可因此与其他导数结构相关。

**Fierz 恒等式。** 四费米子算符存在 Lorentz 和内部指标重排恒等式。例如 Weyl spinor 满足
$$
(\chi\psi)(\eta\xi)=-(\chi\eta)(\psi\xi)-(\chi\xi)(\eta\psi),
$$
具体符号依 spinor 约定而变。Warsaw basis 的四费米子独立性依赖这些恒等式。

**原则 4.10（基选择流程）.** 构造算符基时按以下顺序审计：

1.  写出所有满足规范对称性和 Lorentz 对称性的局域结构；
2.  删除总导数等价类；
3.  用 Bianchi 恒等式化简场强导数；
4.  用领先 EOM 删除冗余算符；
5.  用 Fierz 恒等式消去四费米子线性相关；
6.  最后检查 Hermitian conjugation 和 flavor 交换对称。

## 本章小结

算符分类必须先 quotient 掉冗余。算符基是 Wilson 系数空间的坐标，不是物理本身。

## 练习

**练习 4.1.** 用分部积分证明 $\phi^2\Box\phi^2$ 与 $(\partial_\mu\phi^2)(\partial^\mu\phi^2)$ 只差总导数和符号。

**练习 4.2.** 举例说明 off-shell Green 函数会依赖 EOM 冗余算符选择。

**练习 4.3.** 对例 4.6，写出场重定义 $\phi\mapsto \phi+\epsilon\phi^3$ 对 ${\cal L}_0$ 的一阶影响。
