# 第二十章：Flavor 参数计数表

一个三代二指标 Wilson 系数若为一般复矩阵有十八个实参数，若算符在 dagger 下交换两个 flavor 指标，则 Hermiticity 把它减为九个；四指标张量还会因同种流交换和 Fierz 关系继续约化。参数数因此不是由算符表的行数直接相乘得到，而是某个 involution 与置换作用下实固定子空间的维数。这里先证明一个有限指标集上的 Hermitian involution 计数引理，再把它应用到 Warsaw basis 的 bosonic、current、dipole 和四费米子分区。由显式 Hermiticity 与同种流交换得到的 generic 计数为 2508，文献中的完整三代数 2499 还使用 Warsaw 特有的 flavor/Fierz 线性关系；两种数字的假设与证明责任必须分开。

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

下表给出三代 full flavor 下仅使用显式 Hermiticity 和同种流交换对称所得的计数。这里“generic”正是这一有限假设；Warsaw basis 的其余 flavor/Fierz 线性恒等式需另行逐项证明。

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

**外部输入 20.5（Warsaw 三代完整计数）.** Warsaw basis 文献中常用的 baryon-number conserving、三代 full flavor、Hermitian 拉氏量实参数总数为 $2499$。与 generic 计数的差异来自 Warsaw-specific 的 flavor/Fierz 线性关系，而不是来自 EFT 定义本身。要在书内重现 $2499$，必须逐项证明这些线性关系。

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

**定义 20.7（受限的 CP-conserving 子空间）.** 在固定弱基中，去掉显式 CP-odd bosonic 算符，并把所有允许 h.c. 的 Wilson 矩阵取为实矩阵；Hermitian current 系数取为实对称矩阵。

**警告 20.8.** 定义 20.7 是一个便于计数的受限子空间，不是完整 flavor 物理中的弱基不变量分析。完整 CP 分类需要处理 Yukawa 对角化、CKM 相位和 rephasing invariants。

## 20.5 从计数到拟合

设可观测量向量为 $O_a$，并令独立实坐标
$\theta_i=C_i^{(6)}(\mu)/\Lambda_{\rm ref}^2$，故 $[\theta_i]=-2$。在线性 SMEFT 截断下
$$
O_a=O_a^{\rm SM}+M_{ai}\theta_i+R_{a,p\ge4}.
$$
若 $\dim\theta$ 大于 $M$ 的行数，则 $M$ 必有核；在正定协方差下，Fisher 矩阵
$$
F_{ij}=M_{ai}(\Sigma^{-1})_{ab}M_{bj}
$$
具有同一个核。Flavor 假设的数学作用是把 $\theta$ 限制在较低维子空间，而不是改变 EFT 本身；只有当这个子空间避开原有核方向时，参数才可能被现有数据分别识别。

## 20.6 参数维数如何进入拟合

Hermitian involution 把复系数张量变成实参数空间，同种流交换与 Fierz 关系再对它取子空间或商。三代 generic 计数 2508 与完整 Warsaw 计数 2499 的差值正来自后一步的额外关系，而非规范群或 EFT 定义。进入拟合时，flavor universal、diagonal、MFV 与 full flavor 是不同的线性空间；若不声明所用空间，“多少个 Wilson 系数”和 Fisher 矩阵的秩都没有确定含义。

## 练习

**练习 20.1.** 证明 Hermitian $n_g\times n_g$ 矩阵的实参数数为 $n_g^2$。

**练习 20.2.** 对 $n_g=2$ 重算推论 20.4 的各项参数数。

**练习 20.3.** 说明为什么 flavor universal 假设不是由标准模型规范群推出的。
