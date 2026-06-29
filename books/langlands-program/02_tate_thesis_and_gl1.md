# 第二章：Tate thesis、Hecke 特征与 `GL(1)` L 函数

## 本章目标

本章说明 Tate thesis 如何把 Hecke L 函数写成 adele 上的 zeta 积分，并把 `GL(1)` 的自守表示解释为 idele class group 的特征。这是 Langlands 纲领中最基本且最完整的模型。

## 依赖前置知识

需要第一章的 adeles、ideles、idele class group、Haar 测度和 Fourier 变换约定。局部函数方程和 Poisson summation 在本章作为外部输入。

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

**定义 2.6.** 设 $F$ 为局部域，$\chi:F^\times\to\mathbb C^\times$ 为连续特征，$\phi\in\mathcal S(F)$。局部 zeta 积分定义为
$$
Z(\phi,\chi,s)
=
\int_{F^\times}\phi(x)\chi(x)|x|^s\,d^\times x
$$
在绝对收敛的半平面中成立，并通过解析延拓理解为 $s$ 的亚纯函数。

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

**外部输入定理 2.9（Tate 局部理论）.** 对任意局部域 $F$、非平凡加法特征 $\psi$ 和连续乘法特征 $\chi$，存在局部因子 $L(s,\chi)$、$\varepsilon(s,\chi,\psi)$，使得归一化 zeta 积分满足局部函数方程
$$
\frac{Z(\widehat\phi,\chi^{-1},1-s)}{L(1-s,\chi^{-1})}
=
\varepsilon(s,\chi,\psi)
\frac{Z(\phi,\chi,s)}{L(s,\chi)}
$$
对所有 $\phi\in\mathcal S(F)$ 成立。

本定理是 Tate thesis 的局部部分。例 2.7 给出非分歧有限位置处的基本计算。

## 2.4 整体 zeta 积分

**定义 2.10.** 设 $\Phi\in\mathcal S(\mathbb A_K)$，$\chi$ 为 Hecke 特征。整体 zeta 积分定义为
$$
Z(\Phi,\chi,s)
=
\int_{\mathbb A_K^\times}\Phi(x)\chi(x)|x|_{\mathbb A}^s\,d^\times x
$$
在绝对收敛的半平面中成立。

**命题 2.11（Euler 分解）.** 若 $\Phi=\otimes_v\Phi_v$ 且 $\chi=\prod_v\chi_v$，并且乘法 Haar 测度分解为 $d^\times x=\prod_vd^\times x_v$，则在绝对收敛半平面中
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

**定义 2.12.** Hecke L 函数定义为 Euler 乘积
$$
L(s,\chi)=\prod_vL(s,\chi_v)
$$
在其绝对收敛半平面中成立，并通过 Tate thesis 延拓为亚纯函数。

## 2.5 Tate 整体函数方程

**外部输入定理 2.13（Tate thesis，整体形式）.** 设 $K$ 为整体域，$\chi$ 为酉 Hecke 特征。完成 L 函数 $\Lambda(s,\chi)$ 有亚纯延拓到整个复平面，并满足函数方程
$$
\Lambda(s,\chi)=\varepsilon(s,\chi)\Lambda(1-s,\chi^{-1})
$$
其中 $\varepsilon(s,\chi)$ 是由局部 epsilon 因子乘积给出的显式函数。若 $\chi$ 非平凡，则 $\Lambda(s,\chi)$ 整；若 $\chi$ 平凡，则只可能在 $s=0,1$ 有单极点。非酉 quasi-character 的情形可化为酉特征乘以 idele norm 的复幂，因此相当于把变量 $s$ 平移。

**证明草图.** 取 $\Phi\in\mathcal S(\mathbb A_K)$，把整体 zeta 积分改写为 $C_K$ 上的积分。使用 $\mathbb A_K/K$ 上的 Poisson summation 比较 $\Phi$ 与 $\widehat\Phi$ 的 theta series。分离 idele norm 的大于 $1$ 和小于 $1$ 部分，得到 $s$ 与 $1-s$ 的关系。局部函数方程把未归一化 zeta 积分转换为完成 L 函数和 epsilon 因子。完整证明依赖 Poisson summation、Haar 测度归一化和局部 Tate 理论。$\square$

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

## 2.7 本章小结

Tate thesis 把 Hecke L 函数从 Dirichlet 级数和 Euler 乘积提升为 adele 上的 zeta 积分。类域论进一步说明，这些 Hecke 特征就是 `GL(1)` 的自守表示，并与一维 Galois/Weil 表示对应。Langlands 纲领可以被看作把这一结构从 `GL(1)` 推广到一般还原群。

## 练习

**练习 2.1.** 在例 2.7 中，若 $\chi$ 分歧，证明存在 $\phi\in\mathcal S(F)$ 使 $Z(\phi,\chi,s)$ 非零，但标准函数 $\mathbf 1_{\mathcal O_F}$ 的积分可能为 $0$。

**练习 2.2.** 设 $\chi$ 为 Dirichlet 特征。说明它如何给出 $\mathbb Q$ 上的有限阶 Hecke 特征。此题要求写出有限素数处的局部分量。

**练习 2.3.** 对平凡 Hecke 特征，解释为什么 Tate thesis 中完成 L 函数允许在 $s=0,1$ 出现极点。

**练习 2.4.** 用自己的话说明 `GL(1)` Langlands 中“Galois 侧”“自守侧”和“L 函数相容”分别是什么。
