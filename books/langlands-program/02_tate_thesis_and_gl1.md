# 第二章：Tate thesis、Hecke 特征与 `GL(1)` L 函数

第一章把所有局部域装进了 adele 环；现在需要解释 Euler 乘积为何能由一个全局积分产生，以及函数方程为何是 Fourier 对偶的结果。Tate 的方法把 Hecke 特征 $\chi$ 与 Schwartz--Bruhat 函数 $\Phi$ 放进同一个 zeta 积分 $Z(\Phi,\chi,s)$，局部张量分解给出 Euler 因子，Poisson 求和则控制解析延拓与函数方程。由此，`GL(1)` 的 Galois 侧、自守侧和 L 函数相容第一次同时出现，而且可以精确计算。

本章沿用第一章的 adeles、ideles、Haar 测度和 Fourier 约定。Schwartz--Bruhat 空间、adele Poisson summation、idele 缩放公式与 Tate theta 恒等式可在附录 F 查阅；完整解析延拓和 Archimedean gamma 因子作为 Tate thesis 的外部输入。所有积分与 L 函数变量采用 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 第 3、8 节的约定。

## 2.1 Schwartz-Bruhat 函数

**定义 2.1.** 设 $F$ 为局部域。Schwartz-Bruhat 空间 $\mathcal S(F)$ 定义如下：

1. 若 $F$ 为 Archimedean 局部域，则 $\mathcal S(F)$ 是通常的 Schwartz 函数空间。
2. 若 $F$ 为非 Archimedean 局部域，则 $\mathcal S(F)$ 是所有局部常值且紧支撑的复值函数空间。

**定义 2.2.** 整体域 $K$ 的 adele Schwartz-Bruhat 空间是 restricted tensor product
$$
\mathcal S(\mathbb A_K)=\bigotimes_v'\mathcal S(K_v)
$$
相对于标准向量 $\mathbf 1_{\mathcal O_v}$ 取。也就是说，纯张量
$$
\Phi=\otimes_v\Phi_v
$$
属于该 restricted tensor product，当且仅当 $\Phi_v=\mathbf 1_{\mathcal O_v}$ 对几乎所有非 Archimedean $v$ 成立；一般元素为此类纯张量的有限线性组合。

**定义 2.3.** 固定非平凡加法特征 $\psi:\mathbb A_K/K\to\mathbb C^\times$，并写 $\psi_v$ 为其局部分量。对 $\Phi\in\mathcal S(\mathbb A_K)$，Fourier 变换定义为
$$
\widehat\Phi(y)=\int_{\mathbb A_K}\Phi(x)\psi(xy)\,dx,
$$
其中 $dx$ 是与 $\psi$ 相容的自对偶 Haar 测度。

对非 Archimedean $v$，记
$$
\mathfrak c(\psi_v)=\{a\in K_v:\psi_v(a\mathcal O_v)=1\}.
$$
只有当 $\mathfrak c(\psi_v)=\mathcal O_v$ 时，自对偶测度才满足
$\operatorname{vol}(\mathcal O_v,dx_v)=1$。一般体积关系和整体乘积测度见归一化总表第 3 节。

## 2.2 Hecke 特征

**定义 2.4.** 一个 Hecke 特征（Hecke character, Grossencharacter）是连续同态
$$
\chi:C_K=K^\times\backslash\mathbb A_K^\times\to\mathbb C^\times.
$$
等价地，它是连续同态
$$
\chi:\mathbb A_K^\times\to\mathbb C^\times
$$
并满足 $\chi(a)=1$ 对所有 $a\in K^\times$ 成立。

Hecke 特征称为酉的，若其像落在单位圆 $S^1$ 中。

**外部输入引理 2.4.1（quasi-character 的酉化）.** 令
$$
C_K^1=\ker(|\cdot|_{\mathbb A}:C_K\to\mathbb R_{>0}).
$$
$C_K^1$ 是紧群；因而每个 Hecke quasi-character 唯一写成
$$
\chi=\chi_0|\cdot|_{\mathbb A}^{\sigma(\chi)},
\qquad \sigma(\chi)\in\mathbb R,
$$
其中 $\chi_0$ 酉。这里把纯虚 norm 次幂吸收到 $\chi_0$ 中。故
$L(s,\chi)=L(s+\sigma(\chi),\chi_0)$，所有收敛域和极点陈述都应作同一平移。

本引理使用 idele class group 的 norm-one 部分紧性；该紧性属于 idelic 结构定理，本书作为外部输入，不在本章重证。

**命题 2.5.** 每个 Hecke 特征 $\chi$ 可写成局部特征的 restricted product
$$
\chi(x)=\prod_v\chi_v(x_v),\qquad x=(x_v)_v\in\mathbb A_K^\times,
$$
其中 $\chi_v:K_v^\times\to\mathbb C^\times$ 连续，且对几乎所有非 Archimedean $v$，$\chi_v$ 在 $\mathcal O_v^\times$ 上平凡。

**证明.** 对每个位置 $v$，把 $K_v^\times$ 嵌入 $\mathbb A_K^\times$ 为第 $v$ 个坐标，其余坐标取 $1$，并定义 $\chi_v$ 为 $\chi$ 在该子群上的限制。若 $x\in\mathbb A_K^\times$，则 $x_v\in\mathcal O_v^\times$ 对几乎所有 $v$ 成立。

先证明 $\chi_v$ 对几乎所有 $v$ 在 $\mathcal O_v^\times$ 上平凡。取 $\mathbb C^\times$ 中单位元的一个开邻域 $U$，使得 $U$ 不含非平凡子群。由 $\chi$ 在单位元处连续，存在形如
$$
\prod_{v\in S}U_v\times\prod_{v\notin S}\mathcal O_v^\times
$$
的单位元邻域，其像落在 $U$ 中，其中 $S$ 有限。由于该邻域本身含有子群
$$
\prod_{v\notin S}\mathcal O_v^\times
$$
的逐坐标嵌入，而同态的像是 $U$ 中的子群，故该像只能为 $\{1\}$。因此 $v\notin S$ 时 $\chi_v$ 在 $\mathcal O_v^\times$ 上平凡。

于是对任意 $x\in\mathbb A_K^\times$，乘积 $\prod_v\chi_v(x_v)$ 只有有限多个非 $1$ 因子。将 $x$ 写成有限多个单坐标元素与一个落在 $\prod_{v\notin S}\mathcal O_v^\times$ 中的元素之积，即得该乘积等于 $\chi(x)$。$\square$

## 2.3 局部 zeta 积分和局部 L 因子

**定义 2.6.** 设 $F$ 为局部域，$\chi:F^\times\to\mathbb C^\times$ 为连续 quasi-character，$\phi\in\mathcal S(F)$，并固定乘法 Haar 测度 $d^\times x$。局部 zeta 积分定义为
$$
Z(\phi,\chi,s)
=
\int_{F^\times}\phi(x)\chi(x)|x|^s\,d^\times x
$$
在绝对收敛的半平面中成立。其亚纯延拓不是定义的一部分，而由外部输入定理 2.9 给出。

**例 2.7（非分歧非 Archimedean 情形）.** 设 $F$ 是非 Archimedean 局部域，$\chi$ 在 $\mathcal O_F^\times$ 上平凡，取 $\phi=\mathbf 1_{\mathcal O_F}$，并归一化 $d^\times x$ 使 $\operatorname{vol}(\mathcal O_F^\times)=1$。若 $\varpi$ 为一致化元，则
$$
F^\times=\bigsqcup_{n\in\mathbb Z}\varpi^n\mathcal O_F^\times.
$$
因为 $\mathbf 1_{\mathcal O_F}$ 在 $\varpi^n\mathcal O_F^\times$ 上非零当且仅当 $n\ge 0$，所以
$$
Z(\mathbf 1_{\mathcal O_F},\chi,s)
=
\sum_{n\ge 0}\chi(\varpi)^n|\varpi|^{ns}
=
\sum_{n\ge 0}(\chi(\varpi)q^{-s})^n
=
\frac{1}{1-\chi(\varpi)q^{-s}}
$$
在 $|\chi(\varpi)q^{-s}|<1$ 时成立，并由右端给出亚纯延拓。

**定义 2.8.** 在例 2.7 的非分歧情形中，局部 L 因子定义为
$$
L(s,\chi)=\frac{1}{1-\chi(\varpi)q^{-s}}.
$$
在分歧非 Archimedean 情形中，定义 $L(s,\chi)=1$。Archimedean 情形的局部 L 因子由相应 Gamma 因子给出，具体公式依 $\mathbb R$ 或 $\mathbb C$ 以及特征类型而定。

**外部输入定理 2.9（Tate 局部理论）.** 设 $F$ 为局部域，$\psi:F\to\mathbb C^\times$ 为非平凡连续加法特征，$dx$ 为对应自对偶测度，$\chi:F^\times\to\mathbb C^\times$ 为连续 quasi-character。对每个 $\phi\in\mathcal S(F)$，$Z(\phi,\chi,s)$ 从其绝对收敛半平面亚纯延拓；存在局部因子 $L(s,\chi)$ 和非零局部因子 $\varepsilon(s,\chi,\psi)$，使
$$
\frac{Z(\widehat\phi,\chi^{-1},1-s)}{L(1-s,\chi^{-1})}
=
\varepsilon(s,\chi,\psi)
\frac{Z(\phi,\chi,s)}{L(s,\chi)}
$$
对所有 $\phi\in\mathcal S(F)$ 成立。

本定理是 Tate thesis 的局部部分。例 2.7 给出非分歧有限位置处的基本计算。

## 2.4 整体 zeta 积分

**定义 2.10.** 设 $\Phi\in\mathcal S(\mathbb A_K)$，$\chi$ 为 Hecke quasi-character。对每个有限位置取
$\operatorname{vol}(\mathcal O_v^\times,d^\times x_v)=1$，在 Archimedean 位置固定 Haar 测度，并令
$d^\times x=\prod_vd^\times x_v$。整体 zeta 积分定义为
$$
Z(\Phi,\chi,s)
=
\int_{\mathbb A_K^\times}\Phi(x)\chi(x)|x|_{\mathbb A}^s\,d^\times x
$$
在绝对收敛的半平面中成立。

**命题 2.11（Euler 分解）.** 若 $\Phi=\otimes_v\Phi_v$ 且 $\chi=\prod_v\chi_v$，乘法 Haar 测度按定义 2.10 分解，则在绝对收敛半平面中
$$
Z(\Phi,\chi,s)
=
\prod_v Z(\Phi_v,\chi_v,s).
$$

**证明.** 纯张量条件给出 $\Phi_v=\mathbf 1_{\mathcal O_v}$ 且 $\chi_v$ 非分歧对几乎所有非 Archimedean $v$ 成立。对有限集合 $S$，包含所有 Archimedean 位置、$\Phi_v$ 非标准的位置和 $\chi_v$ 分歧的位置，restricted product 的定义给出
$$
\mathbb A_K^\times
=
\left(\prod_{v\in S}K_v^\times\right)
\times
\left(\prod_{v\notin S}'K_v^\times\right).
$$
在绝对收敛半平面中，Fubini 定理适用。对 $v\notin S$，局部积分为例 2.7 的几何级数。于是整体积分等于局部积分的无穷乘积。$\square$

**定义 2.12.** Hecke L 函数首先定义为 Euler 乘积
$$
L(s,\chi)=\prod_vL(s,\chi_v)
$$
若 $\chi=\chi_0|\cdot|_{\mathbb A}^{\sigma(\chi)}$ 且 $\chi_0$ 酉，则该乘积在
$\operatorname{Re}(s)+\sigma(\chi)>1$ 绝对收敛。其余区域中的函数由 Tate thesis 的亚纯延拓给出，而不是由 Euler 乘积定义。

## 2.5 Tate 整体函数方程

**外部输入定理 2.13（Tate thesis，整体形式）.** 设 $K$ 为整体域，$\psi:\mathbb A_K/K\to\mathbb C^\times$ 为非平凡连续加法特征，局部加法测度均取 $\psi_v$-自对偶测度，$\chi$ 为酉 Hecke 特征。完成 L 函数 $\Lambda(s,\chi)$ 有亚纯延拓到整个复平面，并满足函数方程
$$
\Lambda(s,\chi)=\varepsilon(s,\chi)\Lambda(1-s,\chi^{-1})
$$
其中 $\varepsilon(s,\chi)$ 是由上述 $\psi_v$ 和测度归一化下的局部 epsilon 因子乘积给出的非零显式函数。若
$\chi|_{C_K^1}\ne1$，则 $\Lambda(s,\chi)$ 整；若
$\chi=|\cdot|_{\mathbb A}^{it}$，$t\in\mathbb R$，则在数域情形只可能在
$s=-it$ 与 $s=1-it$ 有单极点，且两处确有极点。若 $K$ 是常数域为
$\mathbb F_q$ 的函数域，则 $\Lambda(s,\chi)$ 是 $q^{-s}$ 的有理函数，上述两组极点应按周期
$2\pi i/\log q$ 理解，即
$$
s=-it+\frac{2\pi i n}{\log q},\qquad
s=1-it+\frac{2\pi i n}{\log q},\qquad n\in\mathbb Z.
$$
一般
$\chi=\chi_0|\cdot|_{\mathbb A}^{\sigma}$ 的陈述由
$\Lambda(s,\chi)=\Lambda(s+\sigma,\chi_0)$ 平移得到。特别地，“$\chi$ 非平凡”不足以推出整性，因为非平凡纯虚 norm 特征仍给出 Dedekind zeta 函数的平移。

**证明路线（外部输入）.** 取 $\Phi\in\mathcal S(\mathbb A_K)$，把整体 zeta 积分改写为 $C_K$ 上的积分。附录 F 的命题 F.21.1 给出 theta 恒等式
$$
\Theta_\Phi(t)=|t|_{\mathbb A}^{-1}\Theta_{\widehat\Phi}(t^{-1}),
$$
它来自 $\mathbb A_K/K$ 上的 Poisson summation 和 idele 缩放公式。分离 idele norm 的大于 $1$ 和小于 $1$ 部分，得到 $s$ 与 $1-s$ 的关系；$\chi$ 在 $C_K^1$ 上平凡时，$\Phi(0)$ 与 $\widehat\Phi(0)$ 的常数项产生上述平移后的两个极点。局部函数方程把未归一化 zeta 积分转换为完成 L 函数和 epsilon 因子。完整证明还依赖局部 Tate 理论、Archimedean gamma 因子估计和整体积分截断；本段只记录证明路线，不宣称完成这些解析步骤。

**Tate thesis 的可用结论 2.A.** 后文使用 Tate thesis 时，所需结论精确分为以下几项：

| 输入 | 使用位置 | 状态 |
|---|---|---|
| 非分歧局部 zeta integral 计算 | Euler 因子和 Satake 比较 | 本章例子给出 |
| 局部函数方程与 epsilon 因子 | 局部因子相容和 functional equation | 外部输入 |
| Archimedean gamma 因子 | 完成 L 函数 | 外部输入 |
| Poisson-theta 恒等式 | 整体函数方程骨架 | 附录 F 接口 |
| 类域论解释 Hecke 特征 | `GL(1)` Langlands | 第三章和附录 V 接口 |

## 2.6 `GL(1)` Langlands

**定义 2.14.** `GL(1)` 在 $K$ 上的全局自守表示，在本书中指 Hecke 特征
$$
\chi:K^\times\backslash\mathbb A_K^\times\to\mathbb C^\times.
$$
这是因为
$$
\operatorname{GL}_1(\mathbb A_K)=\mathbb A_K^\times.
$$

**外部输入定理 2.15（全局类域论，表示论形式）.** 存在 Artin reciprocity map
$$
\operatorname{rec}_K:C_K\to G_K^{\operatorname{ab}}
$$
并在所有有限 Abel 商上诱导有限阶 Hecke 特征与有限像一维 Galois 表示之间的对应：
$$
\operatorname{Hom}_{\operatorname{cont}}(G_K,\mathbb C^\times)_{\operatorname{fin}}
\longleftrightarrow
\operatorname{Hom}_{\operatorname{cont}}(C_K,\mathbb C^\times)_{\operatorname{fin}},
\qquad
\rho\longmapsto \rho\circ\operatorname{rec}_K.
$$
更一般的非有限阶 Hecke 特征对应 Weil 群的一维表示。

**解释 2.16.** 定理 2.15 是 `GL(1)` Langlands 的精确模型。数论侧是一维 Galois/Weil 表示；自守侧是 $\operatorname{GL}_1(\mathbb A_K)$ 的自守表示，即 Hecke 特征；二者的 L 函数由 Tate thesis 统一处理。

在这个模型中，Langlands 纲领的三个核心要求都已出现：

1. 局部-整体分解：$\chi=\prod_v\chi_v$。
2. L 函数相容：$L(s,\chi)=\prod_vL(s,\chi_v)$。
3. Galois 与自守对应：类域论给出一维参数和 Hecke 特征的对应。

## 2.7 `GL(1)` 模型的完成

Tate thesis 把 Hecke L 函数从 Dirichlet 级数和 Euler 乘积提升为 adele 上的 zeta 积分。类域论进一步说明，这些 Hecke 特征就是 `GL(1)` 的自守表示，并与一维 Galois/Weil 表示对应。Langlands 纲领可以被看作把这一结构从 `GL(1)` 推广到一般还原群。

## 练习

**练习 2.1.** 在例 2.7 中，若 $\chi$ 分歧，证明存在 $\phi\in\mathcal S(F)$ 使 $Z(\phi,\chi,s)$ 非零，但标准函数 $\mathbf 1_{\mathcal O_F}$ 的积分可能为 $0$。

**练习 2.2.** 设 $\chi$ 为 Dirichlet 特征。说明它如何给出 $\mathbb Q$ 上的有限阶 Hecke 特征。此题要求写出有限素数处的局部分量。

**练习 2.3.** 对平凡 Hecke 特征，解释为什么 Tate thesis 中完成 L 函数允许在 $s=0,1$ 出现极点。

**练习 2.4.** 用自己的话说明 `GL(1)` Langlands 中“Galois 侧”“自守侧”和“L 函数相容”分别是什么。
