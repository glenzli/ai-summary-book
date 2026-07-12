# 附录 AE：`GL(2)` 局部 Langlands 的 Principal Series、Steinberg 和 Supercuspidal 例子

本附录补足第五、十二和十四章之间的局部 `GL(2)` 桥梁。目标是让读者在进入一般 `GL(n)` 与 L-packet 之前，先看到 `GL(2)` 的三类基本局部表示及其 Weil-Deligne 参数、L 因子和 conductor 行为。

本附录固定非 Archimedean 局部域 $F$，剩余域大小为 $q$，Borel subgroup $B=TN\subset\operatorname{GL}_2$ 取上三角矩阵，$T\simeq F^\times\times F^\times$。

**收口归一化回指。** 本附录涉及局部互反、归一化诱导、Weil-Deligne 参数、局部 L 因子和 conductor；与第十二章局部 Langlands 及第十四章局部因子比较时使用 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 第 1、2、5、6、7、8 节。

## AE.1 归一化主级数

**定义 AE.1.** 设 $\chi_1,\chi_2:F^\times\to\mathbb C^\times$ 为 smooth characters。归一化主级数表示定义为
$$
I(\chi_1,\chi_2)
:=i_{B(F)}^{\operatorname{GL}_2(F)}
(\chi_1\boxtimes\chi_2)
:=
\operatorname{Ind}_{B(F)}^{\operatorname{GL}_2(F),\operatorname{unn}}
\left(\delta_B^{1/2}(\chi_1\boxtimes\chi_2)\right),
$$
其中最右端的 $\operatorname{Ind}^{\operatorname{unn}}$ 是 smooth unnormalized induction，
$$
\delta_B(\operatorname{diag}(a,d))=|a/d|.
$$
因而 $i_B^G$ 已包含一次且仅一次 $\delta_B^{1/2}$。本附录后文若写
$\operatorname{Ind}_{B(F)}^{\operatorname{GL}_2(F)}$ 而不带上标，均指第四章的 normalized induction，不再额外乘 $\delta_B^{1/2}$。

**外部输入定理 AE.2（`GL(2)` principal series irreducibility）.** 若
$$
\chi_1\chi_2^{-1}\ne|\cdot|^{\pm1},
$$
则 $I(\chi_1,\chi_2)$ 不可约。若 $\chi_1\chi_2^{-1}=|\cdot|^{\pm1}$，则 $I(\chi_1,\chi_2)$ 长度为 $2$，其 Jordan-Hölder factors 包含一维表示和 Steinberg twist。

**命题 AE.3（principal series 的 LLC 参数）.** 在 `GL(2)` 局部 Langlands 下，不可约主级数 $I(\chi_1,\chi_2)$ 对应 Weil-Deligne 参数
$$
\varphi=\varphi_{\chi_1}\oplus\varphi_{\chi_2},\qquad N=0,
$$
其中 $\varphi_{\chi_i}:W_F\to\mathbb C^\times$ 由局部类域论给出。

**证明路线（外部输入）.** `GL(1)` 局部 Langlands 把 $\chi_i$ 送到一维 Weil 参数 $\varphi_{\chi_i}$。局部 Langlands for `GL(2)` 与 parabolic induction 相容：由 Levi subgroup $F^\times\times F^\times$ 的参数通过标准嵌入
$$
\operatorname{GL}_1(\mathbb C)\times\operatorname{GL}_1(\mathbb C)
\hookrightarrow \operatorname{GL}_2(\mathbb C)
$$
直和得到 `GL(2)` 参数。主级数没有 monodromy，故 $N=0$。$\square$

**命题 AE.4.** 若 $\chi_1,\chi_2$ 均非分歧，且 $\alpha_i=\chi_i(\varpi)$，则 spherical principal series 的局部标准 L 因子为
$$
L(s,I(\chi_1,\chi_2))
=
\prod_{i=1}^2(1-\alpha_iq^{-s})^{-1}.
$$

**证明.** 参数在惯性上平凡，几何 Frobenius 的半单特征值为 $\alpha_1,\alpha_2$。按第五章 Weil-Deligne L 因子定义，
$$
L(s,\varphi)=
\det(1-q^{-s}\varphi(\operatorname{Fr})\mid V)^{-1}
=\prod_i(1-\alpha_iq^{-s})^{-1}.
$$
$\square$

## AE.2 Steinberg 表示

**定义 AE.5.** Steinberg representation $\operatorname{St}$ 是归一化诱导
$$
I(|\cdot|^{1/2},|\cdot|^{-1/2})
$$
中的本质平方可积 Jordan-Hölder factor。对 character $\chi:F^\times\to\mathbb C^\times$，简记
$$
\operatorname{St}\otimes\chi
:=\operatorname{St}\otimes(\chi\circ\det),
$$
并称之为 Steinberg twist。

**命题 AE.6（Steinberg 的 Weil-Deligne 参数）.** $\operatorname{St}\otimes\chi$ 对应二维 Weil-Deligne 表示
$$
V=\mathbb C e_1\oplus\mathbb C e_2,
$$
其中
$$
r(w)=\varphi_\chi(w)
\begin{pmatrix}
|w|^{1/2}&0\\
0&|w|^{-1/2}
\end{pmatrix},
\qquad
N(e_2)=e_1,\quad N(e_1)=0.
$$
等价地，它是一个 indecomposable special parameter。
特别地，本书取几何 Frobenius，故
$$
|\operatorname{Fr}_F|=q^{-1},
$$
而 $r(\operatorname{Fr}_F)$ 在 $e_1,e_2$ 上的特征值分别为
$$
\chi(\varpi)q^{-1/2},
\qquad
\chi(\varpi)q^{1/2}.
$$

**证明路线（外部输入）.** Steinberg 表示是 reducible principal series 临界点的 square-integrable factor。LLC 与 Langlands classification 相容：临界点的两个 character 差一个 $|\cdot|$，在参数侧合并为带非零 nilpotent monodromy 的 Jordan block。关系
$$
r(w)Nr(w)^{-1}=|w|N
$$
由两个对角 character 的比值正好为 $|\cdot|$ 给出。$\square$

**命题 AE.7.** 若 $\chi$ 非分歧，则
$$
L(s,\operatorname{St}\otimes\chi)
=\left(1-\chi(\varpi)q^{-s-1/2}\right)^{-1}.
$$

**证明.** 对 Steinberg 参数，局部 L 因子取
$$
(\ker N)^{I_F}.
$$
这里 $\ker N=\mathbb C e_1$，且非分歧的 $\chi$ 使该空间由 $I_F$ 逐点固定。由命题 AE.6 及
$|\operatorname{Fr}_F|=q^{-1}$，$r(\operatorname{Fr}_F)$ 在 $e_1$ 上的特征值是
$\chi(\varpi)q^{-1/2}$。代入第五章的 Weil-Deligne L 因子定义得
$$
\det\left(1-q^{-s}r(\operatorname{Fr}_F)\mid
(\ker N)^{I_F}\right)^{-1}
=\left(1-\chi(\varpi)q^{-s-1/2}\right)^{-1}.
$$
$\square$

**命题 AE.8.** Steinberg twist 的 conductor exponent 满足
$$
a(\operatorname{St}\otimes\chi)=
\begin{cases}
1,&\chi\text{ nonramified},\\
2a(\chi),&\chi\text{ ramified}
\end{cases}
$$
在标准 `GL(2)` conductor convention 下成立。

**证明路线（外部输入）.** 非分歧 Steinberg 的 conductor 来自非零 monodromy $N$，指数为 $1$。若 $\chi$ 分歧，参数为 special representation 再张量分歧 character，惯性在两个维度上都通过 $\chi$ 作用；Artin conductor 的主要贡献为两个 character 的 conductor，monodromy 不再额外改变最大指数，得到 $2a(\chi)$。完整证明属于局部 epsilon 因子和 newvector 理论。$\square$

## AE.3 Supercuspidal 表示

**定义 AE.9.** 不可约光滑表示 $\pi$ of $\operatorname{GL}_2(F)$ 称为 supercuspidal，若它不是任何 proper parabolic subgroup 归一化诱导表示的 subquotient。等价地，它的 matrix coefficients modulo center compactly supported。

**外部输入定理 AE.10（`GL(2)` supercuspidals and admissible pairs）.** 在 tame 情形，特别是 residue characteristic 不为 $2$ 时，许多 supercuspidal representations 由 admissible pairs $(E/F,\theta)$ 构造，其中 $E/F$ 为二次扩张，$\theta:E^\times\to\mathbb C^\times$ 为不经 norm 从 $F^\times$ 降下的 character。相应 Weil 参数为
$$
\varphi_\pi=\operatorname{Ind}_{W_E}^{W_F}\varphi_\theta.
$$
wild 情形，尤其 residue characteristic 为 $2$ 时，需要 Bushnell-Henniart 类型理论；本附录不把该分类化约为 admissible-pair 模型。

**命题 AE.11.** 若 $\pi$ 为 supercuspidal，则其 LLC 参数 $\varphi_\pi$ 是不可约二维 Weil 表示，且 $N=0$。

**证明路线（外部输入）.** `GL(2)` LLC 与 parabolic induction 相容。若参数可约为两个一维参数直和，则表示侧属于 principal series 或其极限情形，不可能 supercuspidal。Steinberg 类型对应可约 Weil 表示加非零 monodromy。故 supercuspidal 对应不可约二维 Weil 表示；不可约 Weil 表示没有非零 $N$ 与其相容，因为 $N$ 的 kernel 会给出非零稳定子空间。$\square$

**命题 AE.12.** 若 $\pi$ 为 $\operatorname{GL}_2(F)$ 的 supercuspidal representation，则其标准局部 L 因子恒为
$$
L(s,\pi)=1.
$$

**证明.** 由命题 AE.11，$V_\pi$ 是不可约二维 Weil 表示且 $N=0$。因 $I_F\triangleleft W_F$，子空间 $V_\pi^{I_F}$ 被 $W_F$ 保持。不可约性迫使
$$
V_\pi^{I_F}=0
\quad\text{or}\quad
V_\pi^{I_F}=V_\pi.
$$
若第二种情形成立，则 $r_\pi$ 经过 Abel 商
$W_F/I_F\simeq\mathbb Z$ 分解。一个 Abel 群的有限维复不可约表示为一维，与 $\dim V_\pi=2$ 矛盾。故
$V_\pi^{I_F}=0$。由 $N=0$ 及第五章的定义，
$$
L(s,\pi)
=\det\left(1-q^{-s}r_\pi(\operatorname{Fr}_F)\mid
(\ker N)^{I_F}\right)^{-1}
=\det(1\mid0)^{-1}=1.
$$
$\square$

## AE.4 三分法与椭圆曲线局部表示

**命题 AE.13.** 对 `GL(2)`，不可约可容许表示按 LLC 参数形状可分为：

1. principal series：$\varphi=\chi_1\oplus\chi_2$，$N=0$；
2. Steinberg twists：$\varphi$ 可约但 $N\ne0$；
3. supercuspidal：$\varphi$ 为不可约二维 Weil 表示，$N=0$。

**证明路线（外部输入）.** 这是 `GL(2)` Langlands classification 和局部 LLC 的相容性。所有不可约可容许表示要么来自 Borel 诱导的 subquotient，要么 supercuspidal；Borel 诱导的一般点给 principal series， reducibility point 给 character 和 Steinberg twist；supercuspidal 对应不可约参数。$\square$

**命题 AE.14.** 椭圆曲线 $E/\mathbb Q_q$ 的局部自守表示类型与约化类型满足如下接口：

1. 好约化对应非分歧 principal series 或非分歧参数；
2. 乘法约化对应 Steinberg twist；
3. 加法 potentially good reduction 通常对应 supercuspidal 或 ramified principal series，取决于潜在约化和惯性作用。

**证明路线（外部输入）.** 好约化时 Tate module 非分歧，参数半单且 $N=0$。乘法约化时 Tate curve 给出非零 monodromy，因此对应 Steinberg twist。加法约化时惯性像有限但非平凡，或经有限扩张后好约化；相应参数可能不可约或分歧可约，分别给 supercuspidal 或 ramified principal series。完整分类依赖局部 `GL(2)` LLC 和椭圆曲线潜在约化理论。$\square$

## 练习

**练习 AE.1.** 设 $\chi_1,\chi_2$ 非分歧，计算 principal series 的 Satake 参数和 L 因子。

**练习 AE.2.** 解释 Steinberg 表示为什么需要非零 monodromy $N$。

**练习 AE.3.** 说明 supercuspidal 参数为什么不能是两个 character 的直和。

**练习 AE.4.** 比较乘法约化椭圆曲线与 Steinberg twist 的共同特征。

**练习 AE.5.** 用 AE.13 解释 `GL(2)` LLC 比 `GL(1)` 多出的现象。
