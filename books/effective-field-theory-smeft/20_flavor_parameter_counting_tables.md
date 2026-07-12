# 第二十章：Flavor 参数计数表

## 本章目标

本章把第十四章的 flavor 原则推进为可计算的参数计数规则。目标不是背诵一个总数，而是给出每类 Wilson 张量在不同 flavor 假设下的实参数维数，并说明 Warsaw basis 的完整三代计数为什么需要额外的 flavor/Fierz 恒等式审计。

## 依赖前置知识

需要第十三章的算符分类、第十四章的 Hermiticity 和 CP 纪律，以及附录 A 的场论记号。

## 20.1 计数对象

**定义 20.1（Wilson 参数空间）.** 固定算符基、flavor 假设、CP 假设和 Hermiticity 约束后，Wilson 参数空间是使有效拉氏量 Hermitian 的实向量空间。它的维数称为实参数数。

**警告 20.2.** “一个算符结构”不是“一个实参数”。例如
$$
{\cal O}_{eB}^{pr}
$$
带两个 flavor 指标，三代 full flavor 下对应一个 $3\times3$ 复矩阵。

## 20.2 基本计数引理

令代数数为 $n_g$。

**引理 20.3（Hermitian involution 计数）.** 设复系数 $C_a$ 的指标集合为有限集 $I$，且 Hermiticity 给出
$$
C_a=C_{\iota(a)}^\ast
$$
其中 $\iota:I\to I$ 是 involution。则满足该约束的实维数为 $|I|$。

**证明.** 若 $a$ 是 $\iota$ 的不动点，则 $C_a=C_a^\ast$，贡献一个实参数。若 $\{a,\iota(a)\}$ 是二元轨道，则 $C_a$ 可任取一个复数，$C_{\iota(a)}$ 被其复共轭确定，贡献两个实参数。所有轨道贡献相加正好等于 $I$ 的元素数。$\square$

**推论 20.4（三代常用张量）.** 对 $n_g=3$：

| 系数类型 | 约束 | 实参数数 |
| --- | --- | --- |
| 一个实 bosonic 系数 | Hermitian 算符 | $1$ |
| 一般二指标矩阵 $C^{pr}$ | 复矩阵，拉氏量含 h.c. | $2n_g^2=18$ |
| Hermitian 二指标矩阵 | $C^{pr}=(C^{rp})^\ast$ | $n_g^2=9$ |
| CP-conserving Hermitian 二指标矩阵 | 实对称 | $n_g(n_g+1)/2=6$ |
| diagonal nonuniversal | 只留 $p=r$ | $n_g=3$ |
| flavor universal | 与单位矩阵成比例 | $1$ |
| generic Hermitian 四指标张量 | $C^{prst}=(C^{rpts})^\ast$ | $n_g^4=81$ |
| 同种流交换对称四指标张量 | 还满足 $(pr)\leftrightarrow(st)$ | $(n_g^4+n_g^2)/2=45$ |
| 一般 chiral scalar/tensor 四指标张量 | 复张量，拉氏量含 h.c. | $2n_g^4=162$ |

**证明.** 一般复 $n_g\times n_g$ 矩阵有 $2n_g^2$ 个实参数；其
Hermitian 固定子空间有 $n_g^2$ 个实参数，实对称固定子空间有
$n_g(n_g+1)/2$ 个。四指标张量先把有序指标对 $(p,r)$ 看成一个含
$n_g^2$ 个元素的指标。引理 20.3 给 generic Hermitian 情形 $n_g^4$
个实参数。再要求两个同种流交换，相当于取该 $n_g^2$ 维指标空间的
对称平方；其复维数为 $n_g^2(n_g^2+1)/2$，Hermitian 实结构的固定
空间具有同样的实维数。最后，一般 chiral 张量没有 Hermiticity
约束，故每个复分量贡献两个实参数。代入 $n_g=3$ 得表中数值。$\square$

## 20.3 Warsaw basis 分区计数

下表给出三代 full flavor 下的教材内部计数。这里“generic”表示只使用显式 Hermiticity 和同种流交换对称；Warsaw basis 的全部 flavor/Fierz 线性恒等式需逐项审计。

| Warsaw 分区 | 结构 | generic 实参数数 |
| --- | --- | --- |
| bosonic | 第 13.2 节 15 个结构 | $15$ |
| $\psi^2H^3$ | ${\cal O}_{eH},{\cal O}_{uH},{\cal O}_{dH}$ | $3\times18=54$ |
| dipole $\psi^2XH$ | 8 个 dipole | $8\times18=144$ |
| Hermitian current $\psi^2H^2D$ | 除 ${\cal O}_{Hud}$ 外 7 个 current | $7\times9=63$ |
| non-Hermitian current | ${\cal O}_{Hud}$ | $18$ |
| 同种 current-current | ${\cal O}_{\ell\ell},{\cal O}_{qq}^{(1,3)},{\cal O}_{ee},{\cal O}_{uu},{\cal O}_{dd}$ | $6\times45=270$ |
| 异种 current-current | 14 个 distinct-current 结构 | $14\times81=1134$ |
| chiral scalar/tensor | 第 13.12 节 5 个结构 | $5\times162=810$ |

因此在上述 generic 约束下，
$$
15+54+144+63+18+270+1134+810=2508.
$$

**外部输入 20.5（Warsaw 三代完整计数）.** Warsaw basis 文献中常用的 baryon-number conserving、三代 full flavor、Hermitian 拉氏量实参数总数为 $2499$。与 generic 计数的差异来自 Warsaw-specific 的 flavor/Fierz 线性关系，而不是来自 EFT 定义本身。若教材要把 $2499$ 完全内化，必须逐项证明这些线性关系。

**原则 20.6（计数报告）.** 任何 Wilson 空间维数都必须同时报告：

1.  是否 full flavor；
2.  是否 diagonal；
3.  是否 flavor universal；
4.  是否 MFV；
5.  是否 CP conserving；
6.  是否已经使用所有 Warsaw-specific flavor/Fierz 恒等式。

## 20.4 CP 口径

CP 计数不是简单地把所有复系数实部留下。原因是：

1.  dual field strength 的 bosonic 算符本身是 CP-odd；
2.  dipole 和 Yukawa-like 算符的虚部常对应 CP-odd 方向；
3.  flavor off-diagonal 相可能产生 CP violation；
4.  重新定义 fermion 相位会移动部分相位。

**定义 20.7（教材级 CP-conserving 子空间）.** 本书采用的最低 CP-conserving 口径为：去掉显式 CP-odd bosonic 算符，并把所有允许 h.c. 的 Wilson 矩阵取为实矩阵；Hermitian current 系数取为实对称矩阵。

**警告 20.8.** 这一定义足够用于教材练习，但不是完整 flavor 物理中的弱基不变量分析。完整 CP 分类需要处理 Yukawa 对角化、CKM 相位和 rephasing invariant。

## 20.5 从计数到拟合

设可观测量向量为 $O_a$，Wilson 参数为 $\theta_i$。在线性 SMEFT 截断下
$$
O_a=O_a^{\rm SM}+M_{ai}\theta_i+O(\Lambda^{-4}).
$$
若 $\dim\theta$ 大于独立观测量数，则 Fisher 矩阵
$$
F_{ij}=M_{ai}(\Sigma^{-1})_{ab}M_{bj}
$$
必然有零方向。Flavor 假设的数学作用是把 $\theta$ 限制在较低维子空间，而不是改变 EFT 本身。

## 本章小结

Flavor 完整性首先是线性代数问题。三代 SMEFT 的参数空间很大，且不同 flavor 假设对应不同实向量空间。正式结果必须报告计数口径，否则“多少个 Wilson 系数”不是一个有定义的问题。

## 练习

**练习 20.1.** 证明 Hermitian $n_g\times n_g$ 矩阵的实参数数为 $n_g^2$。

**练习 20.2.** 对 $n_g=2$ 重算推论 20.4 的各项参数数。

**练习 20.3.** 说明为什么 flavor universal 假设不是由标准模型规范群推出的。
