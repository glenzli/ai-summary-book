# 第三章：类域论作为 `GL(1)` Langlands

## 本章目标

本章把局部和全局类域论改写为 `GL(1)` 的 Langlands 对应。重点不是重证类域论，而是固定 reciprocity map 的归一化、说明 Hecke 特征与一维 Galois/Weil 参数的对应，并检查局部 L 因子相容。

## 依赖前置知识

需要第一章的局部域、整体域、ideles 和 idele class group，第二章的 Hecke 特征和 Tate L 函数。有限 Galois 扩张、分解群、惯性群和 Frobenius 元作为代数数论基础使用。附录 V 给出 class formation、Artin reciprocity、norm subgroup theorem、ray class fields 和 conductor 的接口。

收口归一化回指：本章 reciprocity map、一致化元、几何 Frobenius、算术 Frobenius 和一维局部 L 因子的 convention 固定在 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 第 2、6 节。

## 3.1 局部类域论的陈述

本节设 $F$ 为非 Archimedean 局部域，整数环为 $\mathcal O_F$，极大理想为 $\mathfrak p_F$，剩余域为 $k_F$，$\#k_F=q$。取可分闭包 $\overline F$，记
$$
G_F=\operatorname{Gal}(\overline F/F).
$$

**定义 3.1.** 设 $F^{\operatorname{ur}}\subset\overline F$ 为最大非分歧扩张。惯性群定义为
$$
I_F=\operatorname{Gal}(\overline F/F^{\operatorname{ur}}).
$$
有典范满同态
$$
G_F\longrightarrow \operatorname{Gal}(\overline k_F/k_F)\cong\widehat{\mathbb Z}.
$$
几何 Frobenius 元 $\operatorname{Fr}_F$ 是在剩余域上诱导 $x\mapsto x^{1/q}$ 的元素；算术 Frobenius 则诱导 $x\mapsto x^q$。本书在局部 L 因子中默认使用几何 Frobenius。

**外部输入定理 3.2（局部类域论）.** 存在唯一满足下列条件的拓扑群同构，称为局部 reciprocity map：
$$
\operatorname{rec}_F:F^\times\longrightarrow W_F^{\operatorname{ab}}.
$$

1. 若 $\varpi$ 是 $F$ 的一致化元，则 $\operatorname{rec}_F(\varpi)$ 在非分歧商
   $$
   W_F/I_F\cong\mathbb Z
   $$
   中对应几何 Frobenius。
2. 对任意有限 Abel 扩张 $L/F$，复合映射
   $$
   F^\times\xrightarrow{\operatorname{rec}_F}W_F^{\operatorname{ab}}\longrightarrow\operatorname{Gal}(L/F)
   $$
   诱导同构
   $$
   F^\times/N_{L/F}(L^\times)\cong\operatorname{Gal}(L/F).
   $$
3. 在有限扩张中，reciprocity map 与 norm 和 restriction 相容。

这里 $N_{L/F}:L^\times\to F^\times$ 是域范数。

**注 3.3.** 有些资料采用算术 Frobenius 归一化。若采用该归一化，本书中的 $\operatorname{rec}_F$ 要取逆。后续所有 L 因子相容性都依赖当前选择：一致化元对应几何 Frobenius。

**注 3.3.1.** 附录 V 把定理 3.2 和后续全局 reciprocity 放入 class formation 口径，并解释 norm subgroup theorem 如何分类有限 Abel 扩张。本章只使用其 `GL(1)` Langlands 形式。

## 3.2 非分歧局部对应

**定义 3.4.** 连续特征 $\chi:F^\times\to\mathbb C^\times$ 称为非分歧的，若
$$
\chi|_{\mathcal O_F^\times}=1.
$$
连续一维 Weil 表示 $\rho:W_F\to\mathbb C^\times$ 称为非分歧的，若
$$
\rho|_{I_F}=1.
$$

**命题 3.5.** 通过局部 reciprocity map，非分歧特征 $\chi:F^\times\to\mathbb C^\times$ 与非分歧一维 Weil 表示 $\rho:W_F\to\mathbb C^\times$ 一一对应，且
$$
\rho(\operatorname{Fr}_F)=\chi(\varpi).
$$

**证明.** 因为
$$
F^\times\cong \varpi^{\mathbb Z}\times\mathcal O_F^\times,
$$
非分歧特征 $\chi$ 由 $\chi(\varpi)$ 唯一决定。另一方面，非分歧一维 Weil 表示 $\rho$ 通过商
$$
W_F\longrightarrow W_F/I_F\cong\mathbb Z
$$
分解，因而由 $\rho(\operatorname{Fr}_F)$ 决定。局部 reciprocity map 把 $\varpi$ 送到几何 Frobenius 的像，因此二者对应满足所列等式。$\square$

**命题 3.6（局部 L 因子相容）.** 在命题 3.5 的对应下，Tate 局部因子
$$
L(s,\chi)=\frac{1}{1-\chi(\varpi)q^{-s}}
$$
等于 Weil 侧的局部因子
$$
L(s,\rho)=\det\left(1-q^{-s}\rho(\operatorname{Fr}_F)\mid \mathbb C\right)^{-1}.
$$

**证明.** 由命题 3.5，$\rho(\operatorname{Fr}_F)=\chi(\varpi)$。代入一维行列式公式即得
$$
L(s,\rho)=(1-q^{-s}\chi(\varpi))^{-1}=L(s,\chi).
$$
$\square$

## 3.3 分歧局部特征和导子

**定义 3.7.** 对 $n\ge 1$，设
$$
U_F^n=1+\mathfrak p_F^n,\qquad U_F^0=\mathcal O_F^\times.
$$
连续特征 $\chi:F^\times\to\mathbb C^\times$ 的导子指数 $a(\chi)$ 定义为最小整数 $n\ge 0$，使得
$$
\chi|_{U_F^n}=1.
$$
若 $\chi$ 非分歧，则 $a(\chi)=0$。

**定义 3.8.** 一维 Weil 表示 $\rho:W_F\to\mathbb C^\times$ 的导子指数 $a(\rho)$ 由其在惯性群上的有限像部分的 Artin 导子定义。对一维表示，它测量 $\rho$ 在惯性群及其高阶分歧子群上的非平凡程度。

**外部输入定理 3.9（导子相容）.** 在局部类域论对应下，
$$
a(\chi)=a(\rho).
$$
其中 $\rho$ 是由 $\chi$ 通过 $\operatorname{rec}_F$ 得到的一维表示。

**解释 3.10.** 对 `GL(1)`，分歧程度在自守侧表现为特征 $\chi$ 在单位群 filtration 上的非平凡程度；在 Weil/Galois 侧表现为惯性群和高阶分歧群的作用。定理 3.9 是局部 Langlands 中“导子相容”的一维原型。

## 3.4 全局类域论的陈述

设 $K$ 为整体域，idele class group 为
$$
C_K=K^\times\backslash\mathbb A_K^\times.
$$
取可分闭包 $\overline K$，记 $G_K=\operatorname{Gal}(\overline K/K)$。

**外部输入定理 3.11（全局类域论）.** 存在连续同态
$$
\operatorname{rec}_K:C_K\longrightarrow G_K^{\operatorname{ab}}
$$
称为全局 reciprocity map，满足：

1. 对任意有限 Abel 扩张 $L/K$，复合映射
   $$
   C_K\xrightarrow{\operatorname{rec}_K}G_K^{\operatorname{ab}}\longrightarrow\operatorname{Gal}(L/K)
   $$
   诱导同构
   $$
   C_K/N_{L/K}(C_L)\cong\operatorname{Gal}(L/K).
   $$
2. 对每个位置 $v$，局部嵌入 $K_v^\times\to\mathbb A_K^\times$ 与分解群嵌入相容。换言之，对 $K$ 的有限 Abel 扩张 $L$ 和 $w\mid v$，有交换图
   $$
   \begin{matrix}
   K_v^\times & \xrightarrow{\operatorname{rec}_{K_v}} & \operatorname{Gal}(L_w/K_v) \\
   \downarrow & & \downarrow \\
   C_K & \xrightarrow{\operatorname{rec}_K} & \operatorname{Gal}(L/K).
   \end{matrix}
   $$
3. 若 $v$ 在 $L/K$ 中非分歧且 $\varpi_v$ 是 $K_v$ 的一致化元，则把 $\varpi_v$ 放入第 $v$ 个 idele 坐标所得元素映到 $v$ 处的几何 Frobenius。

**注 3.12.** 映射 $\operatorname{rec}_K$ 到 profinite 群 $G_K^{\operatorname{ab}}$。完整拓扑陈述需要处理 $C_K$ 的连通分量和 profinite 完备化。本书在使用有限阶 Hecke 特征和有限 Abel 扩张时，只需要定理 3.11 的有限商形式。

## 3.5 有限阶 Hecke 特征和一维 Galois 表示

**定义 3.13.** Hecke 特征 $\chi:C_K\to\mathbb C^\times$ 称为有限阶，若其像是有限群。

**命题 3.14.** 全局 reciprocity map 给出有限阶 Hecke 特征与有限像一维复 Galois 表示之间的双射：
$$
\left\{
\begin{array}{c}
\text{有限像连续表示}\\
\rho:G_K\to\mathbb C^\times
\end{array}
\right\}
\longleftrightarrow
\left\{
\begin{array}{c}
\text{有限阶 Hecke 特征}\\
\chi:C_K\to\mathbb C^\times
\end{array}
\right\}.
$$
在本书归一化下，对应由
$$
\chi=\rho\circ\operatorname{rec}_K
$$
给出。

**证明.** 若 $\rho$ 有有限像，则它通过某个有限 Abel 扩张 $L/K$ 的 Galois 群分解。由全局类域论，$\operatorname{Gal}(L/K)$ 是 $C_K/N_{L/K}(C_L)$ 的商，所以 $\rho\circ\operatorname{rec}_K$ 是有限阶 Hecke 特征。

反过来，若 $\chi$ 为有限阶 Hecke 特征，则 $\ker(\chi)$ 是 $C_K$ 的开子群，且商 $C_K/\ker(\chi)$ 有限。全局类域论的存在定理给出有限 Abel 扩张 $L/K$，使得相应 norm 子群落在 $\ker(\chi)$ 中；于是 $\chi$ 通过 $\operatorname{Gal}(L/K)$ 分解，得到一维 Galois 表示。两个构造互逆由 reciprocity map 在所有有限 Abel 商上的同构性给出。$\square$

**命题 3.15（全局 L 函数相容，有限阶情形）.** 设 $\rho:G_K\to\mathbb C^\times$ 为有限像连续表示，$\chi=\rho\circ\operatorname{rec}_K$ 为对应的 Hecke 特征。则除有限多个分歧位置外，局部 L 因子满足
$$
L(s,\rho_v)=L(s,\chi_v).
$$
因此 Euler 乘积的非分歧部分相同。

**证明.** 取非 Archimedean 位置 $v$，使 $\rho$ 和 $\chi_v$ 均非分歧。令 $\varpi_v$ 为一致化元。由全局 reciprocity 与局部 reciprocity 的相容性，$\operatorname{rec}_K(\varpi_v)$ 在分解群中对应几何 Frobenius $\operatorname{Fr}_v$。因此
$$
\chi_v(\varpi_v)=\rho(\operatorname{Fr}_v).
$$
局部因子都等于
$$
\left(1-\rho(\operatorname{Fr}_v)q_v^{-s}\right)^{-1}.
$$
$\square$

## 3.6 `GL(1)` Langlands 的精确形式

`GL(1)` 的自守侧是
$$
\operatorname{GL}_1(K)\backslash\operatorname{GL}_1(\mathbb A_K)
=K^\times\backslash\mathbb A_K^\times=C_K.
$$
因此 `GL(1)` 的自守表示就是 Hecke 特征。

**定理 3.16（`GL(1)` Langlands，有限阶 Galois 形式）.** 有限像一维 Galois 表示
$$
\rho:G_K\to\mathbb C^\times
$$
与有限阶 Hecke 特征
$$
\chi:C_K\to\mathbb C^\times
$$
自然一一对应，并且在几乎所有位置处局部 L 因子相容。

**证明.** 双射由命题 3.14 给出，局部 L 因子相容由命题 3.15 给出。$\square$

**外部输入定理 3.17（`GL(1)` Langlands，Weil 形式）.** 若把 Galois 群替换为 Weil 群，则连续一维 Weil 表示与一般 Hecke quasi-character 对应。该对应保持局部分量、L 因子、epsilon 因子和导子。

该定理是第二章 Tate thesis 与类域论的合并形式。第五章将定义局部 Weil 群和 Weil-Deligne 参数。

## 3.7 本章小结

类域论把 Abel Galois 理论翻译为 $F^\times$ 和 $C_K$ 的特征理论。局部情形中，有限阶特征可通过有限 Galois 商描述，而一般连续特征应通过一维 Weil 参数描述；全局情形中，有限阶 Hecke 特征对应有限像一维 Galois 表示，一般 Hecke quasi-character 对应 Weil 侧的一维表示。`GL(1)` Langlands 不是类比，而是类域论本身的表示论重写。

## 练习

**练习 3.1.** 设 $F$ 为非 Archimedean 局部域。证明非分歧特征 $\chi:F^\times\to\mathbb C^\times$ 由 $\chi(\varpi)$ 唯一决定，并说明该值与一致化元选择无关。

**练习 3.2.** 在本书几何 Frobenius 归一化下，验证命题 3.6 的 L 因子公式。如果改用算术 Frobenius，公式中哪一处需要改变？

**练习 3.3.** 设 $L/K$ 为有限 Abel 扩张。解释为什么全局类域论给出的同构
$$
C_K/N_{L/K}(C_L)\cong\operatorname{Gal}(L/K)
$$
可以被看作 Artin reciprocity 的全局版本。

**练习 3.4.** 对 Dirichlet 特征 $\chi$，说明它作为 $\mathbb Q$ 上有限阶 Hecke 特征时，对应的一维 Galois 表示在哪些素数处分歧。

**练习 3.5.** 设 $\chi:C_K\to\mathbb C^\times$ 为有限阶 Hecke 特征。用各局部分量 $\chi_v$ 的导子指数写出 $\chi$ 的整体 conductor，并说明该 conductor 如何控制 $\chi$ 通过某个 ray class group 分解。
