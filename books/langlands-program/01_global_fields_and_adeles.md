# 第一章：整体域、局部域与 adeles

## 本章目标

本章建立 Langlands 纲领的局部-整体语言：整体域、位置、完备化、adele 环、idele 群和 idele class group。`GL(1)` Langlands、Tate thesis 和自守表示都以这些对象为基础。

## 依赖前置知识

需要基本域论、Dedekind 整环、有限扩张、完备赋值域和局部紧拓扑群的初步知识。Haar 测度只在本章末尾作准备性陈述。附录 F 给出本章所用 Pontryagin 对偶、adeles 自对偶性、$\mathbb A_\mathbb Q/\mathbb Q$ 基本域和 Poisson 求和的分析接口。

收口归一化回指：本章使用的绝对值、乘积公式、adeles、ideles、Haar 测度和 Fourier 测度 convention 见 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 的第 1、3 节。

## 1.1 整体域和局部域

**定义 1.1.** 一个整体域（global field）是下列两类域之一：

1. 数域，即 $\mathbb Q$ 的有限扩张。
2. 函数域，即某个有限域 $\mathbb F_q$ 上一变量函数域，也就是超越次数为 $1$ 且常数域有限的有限生成域扩张 $K/\mathbb F_q$。

**定义 1.2.** 设 $K$ 为域。$K$ 上的绝对值是映射 $|\cdot|:K\to\mathbb R_{\ge 0}$，满足：

1. $|x|=0$ 当且仅当 $x=0$。
2. $|xy|=|x||y|$。
3. $|x+y|\le |x|+|y|$。

若还满足强三角不等式
$$
|x+y|\le \max\{|x|,|y|\},
$$
则称为非 Archimedean 绝对值；否则称为 Archimedean 绝对值。

**定义 1.3.** 两个绝对值 $|\cdot|_1,|\cdot|_2$ 称为等价，若它们诱导 $K$ 上相同的拓扑。整体域 $K$ 的一个位置（place）是 $K$ 上非平凡绝对值的等价类。所有位置的集合记为 $V_K$。

对 $v\in V_K$，选择一个代表绝对值并完成 $K$，所得完备域记为 $K_v$。本书总是使用归一化绝对值 $|\cdot|_v$，使乘积公式成立。

**例 1.4.** 对 $K=\mathbb Q$，位置由通常绝对值 $|\cdot|_\infty$ 和每个素数 $p$ 对应的 $p$-adic 绝对值 $|\cdot|_p$ 给出，其中 $|p|_p=p^{-1}$。相应完备化为 $\mathbb R$ 和 $\mathbb Q_p$。

**定义 1.5.** 一个局部域（local field）是带有非离散局部紧拓扑的完备赋值域。若拓扑来自 Archimedean 绝对值，则局部域同构于 $\mathbb R$ 或 $\mathbb C$。若拓扑来自非 Archimedean 绝对值，则其整数环
$$
\mathcal O_F=\{x\in F:|x|\le 1\}
$$
是紧开子环，极大理想
$$
\mathfrak p_F=\{x\in F:|x|<1\}
$$
的剩余域 $k_F=\mathcal O_F/\mathfrak p_F$ 有限。

**外部输入定理 1.6（整体域的局部化）.** 若 $K$ 是整体域且 $v\in V_K$，则 $K_v$ 是局部域。若 $v$ 非 Archimedean，则 $K_v$ 的剩余域有限。

该定理属于代数数论基础；本书后续直接使用。

## 1.2 乘积公式

**约定 1.7.** 对数域 $K$，归一化 $|\cdot|_v$ 如下。若 $v$ 位于 $\mathbb Q$ 的位置 $w$ 之上，则
$$
|x|_v=|N_{K_v/\mathbb Q_w}(x)|_w,\qquad x\in K_v.
$$
对函数域，归一化由闭点次数给出：若 $v$ 对应离散赋值 $\operatorname{ord}_v$，则
$$
|x|_v=q_v^{-\operatorname{ord}_v(x)}.
$$

**定理 1.8（乘积公式）.** 对任意整体域 $K$ 和任意 $x\in K^\times$，有
$$
\prod_{v\in V_K}|x|_v=1.
$$
并且该乘积只有有限多个因子不等于 $1$。

**证明.** 先设 $K$ 为数域。对 $x\in K^\times$，只有有限多个非 Archimedean 位置满足 $|x|_v\ne 1$，因为 $x$ 在整数环的分式理想分解中只涉及有限多个素理想。按照约定 1.7，
$$
\prod_{v\mid w}|x|_v
=\prod_{v\mid w}|N_{K_v/\mathbb Q_w}(x)|_w
=|N_{K/\mathbb Q}(x)|_w.
$$
对所有 $\mathbb Q$ 的位置 $w$ 相乘，得到
$$
\prod_{v\in V_K}|x|_v
=\prod_w |N_{K/\mathbb Q}(x)|_w.
$$
$N_{K/\mathbb Q}(x)\in\mathbb Q^\times$，而 $\mathbb Q$ 的乘积公式给出右端为 $1$。

若 $K$ 是有限域上的一变量函数域，则 $x\in K^\times$ 定义主除子
$$
\operatorname{div}(x)=\sum_v \operatorname{ord}_v(x)[v].
$$
主除子的次数为 $0$，即
$$
\sum_v \operatorname{ord}_v(x)\deg(v)=0.
$$
因为 $q_v=q^{\deg(v)}$，所以
$$
\prod_v |x|_v
=\prod_v q_v^{-\operatorname{ord}_v(x)}
=q^{-\sum_v\operatorname{ord}_v(x)\deg(v)}
=1.
$$
$\square$

## 1.3 Restricted product 和 adele 环

**定义 1.9.** 设 $I$ 为指标集。对每个 $i\in I$，设 $G_i$ 是拓扑群，并对除有限多个 $i$ 外给定开紧子群 $H_i\subseteq G_i$。restricted product
$$
\prod_i' G_i
$$
是所有元 $(g_i)_i\in\prod_iG_i$ 组成的集合，满足 $g_i\in H_i$ 对除有限多个 $i$ 外成立。其拓扑以集合
$$
\prod_{i\in S}U_i\times\prod_{i\notin S}H_i
$$
为基，其中 $S\subset I$ 有限，$U_i\subseteq G_i$ 为开集，并且包含所有没有指定 $H_i$ 的例外指标。

**定义 1.10.** 整体域 $K$ 的 adele 环是 restricted product
$$
\mathbb A_K=\prod_{v\in V_K}'K_v
$$
相对于非 Archimedean 位置处的开紧子环 $\mathcal O_v\subset K_v$ 取。即
$$
\mathbb A_K
=
\left\{(x_v)_v\in\prod_vK_v:
x_v\in\mathcal O_v\text{ for almost all non-Archimedean }v
\right\}.
$$

**命题 1.11.** $\mathbb A_K$ 在逐坐标加法和乘法下是拓扑环。

**证明.** 先验证集合在加法和乘法下封闭。若 $x=(x_v)_v$ 与 $y=(y_v)_v$ 属于 $\mathbb A_K$，则除有限多个非 Archimedean $v$ 外有 $x_v,y_v\in\mathcal O_v$。因为 $\mathcal O_v$ 是子环，除有限多个 $v$ 外有
$$
x_v+y_v\in\mathcal O_v,\qquad x_vy_v\in\mathcal O_v.
$$
故 $x+y,xy\in\mathbb A_K$。

拓扑连续性按 restricted product 的基开集检验。加法和乘法在每个局部域 $K_v$ 中连续；对几乎所有 $v$，$\mathcal O_v$ 对加法和乘法稳定。因此给定基开邻域，只需在有限多个坐标选择足够小的局部开邻域，其余坐标仍取 $\mathcal O_v$，即可得到加法和乘法的连续性。$\square$

**定义 1.12.** 有限 adele 环定义为
$$
\mathbb A_{K,f}=\prod_{v\in V_K^f}'K_v.
$$
若 $K$ 是数域，则有拓扑环同构
$$
\mathbb A_K\cong K\otimes_\mathbb Q\mathbb R\times \mathbb A_{K,f}.
$$

## 1.4 对角嵌入

**定义 1.13.** 对角嵌入是环同态
$$
\Delta:K\longrightarrow\mathbb A_K,\qquad x\longmapsto (x)_v.
$$
通常把 $K$ 视为 $\mathbb A_K$ 的子环。

**命题 1.14.** 对角嵌入 $\Delta$ 是良定义的。

**证明.** 对 $x\in K$，若 $v$ 是非 Archimedean 位置，则 $|x|_v\le 1$ 等价于 $x\in\mathcal O_v$。固定 $x$ 后，只有有限多个素理想出现在分式理想 $(x)$ 的分母中；因此除有限多个非 Archimedean $v$ 外有 $x\in\mathcal O_v$。故 $(x)_v\in\mathbb A_K$。$\square$

**外部输入定理 1.15（adeles 的基本紧性）.** 对任意整体域 $K$，对角嵌入下的 $K$ 是 $\mathbb A_K$ 中的离散子群，并且商群 $\mathbb A_K/K$ 是紧群。

该定理是 adelic Fourier analysis 的基础。对 $K=\mathbb Q$ 可用 $\mathbb R\times\prod_p\mathbb Z_p$ 的基本域直接证明；一般情形需要 Minkowski 理论或函数域上的 Riemann-Roch。

**注 1.15.1.** 附录 F 的命题 F.18.1 给出 $K=\mathbb Q$ 时的完整证明：$\mathbb Q$ 在 $\mathbb A_\mathbb Q$ 中离散，且 $[0,1]\times\prod_p\mathbb Z_p$ 映到 $\mathbb A_\mathbb Q/\mathbb Q$ 为满。一般整体域情形仍按本定理作为外部输入使用。

## 1.5 Idele 群

**定义 1.16.** 整体域 $K$ 的 idele 群是 restricted product
$$
\mathbb A_K^\times=\prod_{v\in V_K}'K_v^\times
$$
相对于非 Archimedean 位置处的开紧子群 $\mathcal O_v^\times\subset K_v^\times$ 取。即
$$
\mathbb A_K^\times
=
\left\{(x_v)_v\in\prod_vK_v^\times:
x_v\in\mathcal O_v^\times\text{ for almost all non-Archimedean }v
\right\}.
$$

注意 $\mathbb A_K^\times$ 是 $\mathbb A_K$ 的单位群，但其拓扑不是从 $\mathbb A_K$ 继承的子空间拓扑；它使用 restricted product 群拓扑。

**定义 1.17.** Idele norm 是连续同态
$$
|\cdot|_{\mathbb A}:\mathbb A_K^\times\to\mathbb R_{>0},
\qquad
|x|_{\mathbb A}=\prod_v|x_v|_v.
$$
乘积良定义，因为 $x_v\in\mathcal O_v^\times$ 对几乎所有非 Archimedean $v$ 成立，故 $|x_v|_v=1$ 对几乎所有 $v$ 成立。

**定义 1.18.** Idele class group 是商群
$$
C_K=K^\times\backslash\mathbb A_K^\times.
$$

**命题 1.19.** 对任意 $a\in K^\times$，其对角像满足
$$
|a|_{\mathbb A}=1.
$$
因此 idele norm 下降为
$$
|\cdot|_{\mathbb A}:C_K\to\mathbb R_{>0}.
$$

**证明.** 这正是定理 1.8 的乘积公式。若 $a$ 以对角方式嵌入 $\mathbb A_K^\times$，则
$$
|a|_{\mathbb A}=\prod_v|a|_v=1.
$$
所以左乘 $K^\times$ 不改变 idele norm，故该同态下降到商群 $C_K$。$\square$

## 1.6 特征和 Pontryagin 对偶

**定义 1.20.** 局部紧交换群 $G$ 的一个特征（character）是连续同态
$$
\chi:G\to\mathbb C^\times
$$
其像落在单位圆 $S^1$ 时称为酉特征。Pontryagin 对偶 $\widehat G$ 是所有酉特征构成的群，带紧开拓扑。

**外部输入定理 1.21（adeles 的自对偶性）.** 固定非平凡连续加法特征
$$
\psi:\mathbb A_K/K\to\mathbb C^\times.
$$
映射
$$
\mathbb A_K\longrightarrow\widehat{\mathbb A_K},\qquad
a\longmapsto (x\mapsto \psi(ax))
$$
是拓扑群同构。写 $\psi=\prod_v\psi_v$，逐处取 $\psi_v$-self-dual Haar measure $dx_v$；几乎所有有限位置的 additive conductor 为 $\mathcal O_v$，故
$dx=\prod_vdx_v$ 是良定义的 restricted product measure。该测度满足
$$
\operatorname{vol}(K\backslash\mathbb A_K,dx)=1,
$$
且 Fourier inversion 和 Poisson summation 对 $\mathcal S(\mathbb A_K)$ 成立。若某个
$\psi_v$ 的 conductor 不是 $\mathcal O_v$，则不能同时把 $dx_v$ 自对偶并任意规定
$\operatorname{vol}(\mathcal O_v)=1$；精确体积关系见归一化总表第 3 节。

本定理是第二章 Tate thesis 的分析基础。

**注 1.21.1.** 附录 F.4--F.5 把本定理拆成 annihilator、$\widehat{\mathbb A_K/K}\simeq K$、adele Poisson summation 和 idele 缩放公式。第二章的 theta 恒等式具体使用命题 F.21.1。

**收口精修 1.A（adelic analysis 使用边界）.** 后文从本章调用的分析输入只有下表项目：

| 输入 | 本章状态 | 后续作用 |
|---|---|---|
| 乘积公式 | 已证明 | 使 idele norm 下降到 $C_K$ |
| $\mathbb A_K/K$ 紧性 | 外部输入 | 支撑 Fourier analysis 和 Poisson summation |
| adeles 自对偶性 | 外部输入 | 固定 self-dual measure 和 Fourier transform |
| adele Poisson summation | 附录 F 接口 | 进入 Tate thesis 的 theta 恒等式 |
| restricted product 拓扑 | 已定义 | 统一局部对象为全局 adelic 对象 |

## 1.7 本章小结

本章定义了整体域的所有局部化 $K_v$，并用 restricted product 把它们组织为 $\mathbb A_K$ 和 $\mathbb A_K^\times$。乘积公式保证 $K^\times$ 在 idele norm 下平凡，从而 norm 下降到 idele class group $C_K$。下一章将在 $\mathbb A_K$ 上做 Fourier 分析，并把 Hecke 特征的 L 函数写成 zeta 积分。

## 练习

**练习 1.1.** 对 $x=18/25\in\mathbb Q^\times$，直接计算
$$
|x|_\infty\prod_p|x|_p
$$
并验证乘积公式。

**练习 1.2.** 证明 $\prod_p\mathbb Z_p$ 是 $\mathbb A_{\mathbb Q,f}$ 的开紧子环。

**练习 1.3.** 设 $x=(x_v)_v\in\mathbb A_K^\times$。证明 $x^{-1}=(x_v^{-1})_v$ 仍属于 $\mathbb A_K^\times$。

**练习 1.4.** 对 $K=\mathbb Q$，证明 $\mathbb Q$ 在 $\mathbb A_\mathbb Q$ 中是离散的。提示：考虑邻域 $(-1/2,1/2)\times\prod_p\mathbb Z_p$。
