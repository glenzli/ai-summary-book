# 第五章：Weil 群、Weil-Deligne 数据与局部参数

## 本章目标

本章定义局部 Weil 群、Weil-Deligne 表示和 Langlands 局部参数，并说明 `GL(1)` 局部 Langlands 如何由局部类域论给出。一般 `GL(n)` 和 reductive 群的局部 Langlands 在本章只作精确定式，不作证明。

## 依赖前置知识

需要第三章的局部类域论和第四章的光滑表示。需要知道局部域的绝对 Galois 群、惯性群和 Frobenius 元；附录 A.2 给出分解群、惯性群、非分歧 Frobenius 和高阶分歧群的代数数论接口。附录 AE 给出 `GL(2)` principal series、Steinberg 和 supercuspidal 参数例子，供本章定义后立即计算。

收口归一化回指：本章 Weil 群、Weil-Deligne 数据、局部 reciprocity、Satake 参数和局部 L 因子均采用 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 第 2、4、6 节的 convention。

## 5.1 非 Archimedean 局部 Weil 群

本节设 $F$ 为非 Archimedean 局部域，剩余域大小为 $q$，绝对 Galois 群为 $G_F$，惯性群为 $I_F$。

**定义 5.1.** 几何 Frobenius 归一化下的局部 Weil 群定义为
$$
W_F=\{g\in G_F:\text{其在 }G_F/I_F\cong\widehat{\mathbb Z}\text{ 中的像属于 }\mathbb Z\cdot\operatorname{Fr}_F\}.
$$
它带有如下拓扑：$I_F$ 作为开子群带从 $G_F$ 继承的 profinite 拓扑，而商
$$
W_F/I_F\cong\mathbb Z
$$
带离散拓扑。

**定义 5.2.** Weil 群上的 norm character 是同态
$$
|\cdot|:W_F\to\mathbb R_{>0}
$$
定义为：若 $w$ 在 $W_F/I_F$ 中的像为 $\operatorname{Fr}_F^n$，则
$$
|w|=q^{-n}.
$$

该定义与第三章的归一化相容：局部 reciprocity map 满足 $\operatorname{rec}_F(\varpi)=\operatorname{Fr}_F$，而 $|\varpi|_F=q^{-1}$。

**命题 5.3.** 有短正合列
$$
1\longrightarrow I_F\longrightarrow W_F\longrightarrow\mathbb Z\longrightarrow 1.
$$
并且 $W_F$ 在 $G_F$ 中稠密。

**证明.** 第一条正合列由定义直接给出：$W_F$ 是 $G_F\to\widehat{\mathbb Z}$ 下 $\mathbb Z\subset\widehat{\mathbb Z}$ 的原像，核为 $I_F$。由于 $\mathbb Z$ 在 $\widehat{\mathbb Z}$ 中稠密，且 $G_F\to\widehat{\mathbb Z}$ 连续满射，原像 $W_F$ 在 $G_F$ 中稠密。$\square$

## 5.2 Archimedean Weil 群

**定义 5.4.** Archimedean 局部域的 Weil 群定义如下：

1. 对 $F=\mathbb C$，
   $$
   W_\mathbb C=\mathbb C^\times.
   $$
2. 对 $F=\mathbb R$，
   $$
   W_\mathbb R=\mathbb C^\times\sqcup j\mathbb C^\times
   $$
   其中乘法由
   $$
   jzj^{-1}=\overline z,\qquad j^2=-1
   $$
   决定。

有短正合列
$$
1\longrightarrow\mathbb C^\times\longrightarrow W_\mathbb R\longrightarrow\operatorname{Gal}(\mathbb C/\mathbb R)\longrightarrow 1.
$$

**注 5.5.** Archimedean 局部 Langlands 使用 $W_\mathbb R$ 和 $W_\mathbb C$ 的有限维半单复表示描述实群表示的无穷小参数。完整理论需要 Harish-Chandra 模，本书后续单独处理。

## 5.3 Weil-Deligne 表示

**定义 5.6.** 设 $F$ 为非 Archimedean 局部域。一个复 Weil-Deligne 表示是三元组 $(V,r,N)$，其中：

1. $V$ 是有限维复向量空间。
2. $r:W_F\to\operatorname{GL}(V)$ 是连续表示，且 $r(I_F)$ 的像有限。
3. $N:V\to V$ 是 nilpotent 线性算子。
4. 对所有 $w\in W_F$，有
   $$
   r(w)Nr(w)^{-1}=|w|N.
   $$

若 $N=0$，则 Weil-Deligne 表示退化为具有有限惯性像的 Weil 表示。

**定义 5.7.** Weil-Deligne 表示 $(V,r,N)$ 称为 Frobenius-semisimple，若 $r(\operatorname{Fr}_F)$ 的半单部分已被选定，或者等价地，在同构类中以 Frobenius 半单化代替 $r$。本书局部 L 因子默认对 Frobenius-semisimple 对象定义。

**定义 5.8.** 对 Weil-Deligne 表示 $(V,r,N)$，其局部 L 因子定义为
$$
L(s,V,r,N)
=
\det\left(1-q^{-s}r(\operatorname{Fr}_F)\mid (\ker N)^{I_F}\right)^{-1}.
$$

**例 5.9.** 若 $V=\mathbb C$，$N=0$，且 $r$ 非分歧，则
$$
L(s,V,r,0)=\left(1-q^{-s}r(\operatorname{Fr}_F)\right)^{-1}.
$$
这与第三章的 `GL(1)` 局部因子相同。

## 5.4 局部 Langlands 参数

设 $G$ 为局部域 $F$ 上的 connected reductive group。其复对偶群记为 $\widehat G$。若 $G$ 非 split，则 L 群为半直积
$$
{}^LG=\widehat G\rtimes W_F
$$
或在有限 Galois 作用下的相应形式；本书后续在还原群章节精确定义。

**定义 5.10.** 若 $F$ 为非 Archimedean 局部域且 $G$ 为 split connected reductive group，$G$ 的 Langlands 局部参数是连续同态
$$
\varphi:W_F\times\operatorname{SL}_2(\mathbb C)\longrightarrow\widehat G
$$
满足：

1. $\varphi|_{\operatorname{SL}_2(\mathbb C)}$ 是代数群同态。
2. 对每个 $w\in W_F$，$\varphi(w,1)$ 是半单元素。

参数的 $\widehat G$-共轭类才是局部 Langlands 中的不变量。

**注 5.11.** 也常把 $W_F\times\operatorname{SL}_2(\mathbb C)$ 记作 $W_F'$。对 $\operatorname{GL}_n$，给出这样的参数等价于给出 $n$ 维 Frobenius-semisimple Weil-Deligne 表示。非 split 群和一般 L 群的完整定义将在还原群章节给出；此处只固定后续非分歧例子所需的 split 口径。

**定义 5.12.** 非分歧参数是指 $\varphi$ 在惯性群 $I_F$ 上平凡，且在 $\operatorname{SL}_2(\mathbb C)$ 上平凡。此时参数由半单共轭类
$$
s_\varphi=\varphi(\operatorname{Fr}_F)\in\widehat G
$$
决定。

**命题 5.13（Satake 参数与非分歧局部参数）.** 在第四章 Satake 同构的归一化下，非分歧不可约球表示的 Hecke 本征值给出 $\widehat G$ 中的半单共轭类；该共轭类可解释为非分歧局部 Langlands 参数在几何 Frobenius 上的值。

**证明草图.** 第四章的 Satake 同构把球 Hecke 代数识别为 $\widehat G$ 的 Weyl 不变量函数代数。不可约球表示 $\pi$ 的一维空间 $\pi^{G(\mathcal O_F)}$ 给出该交换代数的特征，即 $\widehat G$ 中半单共轭类。定义非分歧参数 $\varphi_\pi$ 使 $\varphi_\pi(\operatorname{Fr}_F)$ 等于该共轭类，并令 $\varphi_\pi$ 在 $I_F$ 和 $\operatorname{SL}_2(\mathbb C)$ 上平凡，即得到所述解释。完整证明依赖 Satake 同构。$\square$

## 5.5 `GL(1)` 局部 Langlands

**定理 5.14（`GL(1)` 局部 Langlands）.** 局部 reciprocity map 给出连续特征
$$
\chi:F^\times\to\mathbb C^\times
$$
与一维 Weil 参数
$$
\varphi_\chi:W_F\to\mathbb C^\times
$$
之间的双射。具体地，
$$
\varphi_\chi(w)=\chi\left(\operatorname{rec}_F^{-1}(w^{\operatorname{ab}})\right).
$$
在该对应下，局部 L 因子满足
$$
L(s,\chi)=L(s,\varphi_\chi).
$$

**证明.** 局部类域论给出拓扑同构
$$
F^\times\cong W_F^{\operatorname{ab}}
$$
其中一致化元对应几何 Frobenius。于是 $F^\times$ 的连续特征与 $W_F^{\operatorname{ab}}$ 的连续特征互相拉回，等价于 $W_F$ 的一维连续表示。

若 $\chi$ 非分歧，则 $\varphi_\chi$ 在 $I_F$ 上平凡，且
$$
\varphi_\chi(\operatorname{Fr}_F)=\chi(\varpi).
$$
于是 L 因子相容由例 5.9 和命题 3.6 得到。若 $\chi$ 分歧，则两侧局部 L 因子均按惯性不变量定义；一维情形下该定义与 Tate 局部因子一致，这是 Tate 局部理论的一部分。$\square$

## 5.6 `GL(n)` 局部 Langlands 的定式

**外部输入定理 5.15（局部 Langlands for `GL(n)`）.** 设 $F$ 为非 Archimedean 局部域。存在不可约可容许光滑表示同构类与 Frobenius-semisimple Weil-Deligne 表示同构类之间的自然双射
$$
\left\{
\begin{array}{c}
\text{不可约可容许光滑表示}\\
\pi\text{ of }\operatorname{GL}_n(F)
\end{array}
\right\}
\longleftrightarrow
\left\{
\begin{array}{c}
n\text{ 维 Frobenius-semisimple}\\
\text{Weil-Deligne 表示}
\end{array}
\right\}
$$
满足以下相容性：

1. 中心特征相容：
   $$
   \omega_\pi(a)=\det\varphi_\pi(\operatorname{rec}_F(a)),
   \qquad a\in F^\times,
   $$
   其中 $\operatorname{rec}_F$ 采用第三章的一致化元到几何 Frobenius 的归一化。
2. 对 $n=1$，该对应等于定理 5.14。
3. 局部 L 因子和 epsilon 因子相容。
4. 非分歧表示对应非分歧参数，Satake 参数等于 $\varphi_\pi(\operatorname{Fr}_F)$。

本定理的完整证明超出本书当前章节范围。`GL(2)` 的特殊情形可由 Bushnell-Henniart 等理论处理；一般 `GL(n)` 由 Harris-Taylor、Henniart 等工作完成。

**注 5.15.1.** 附录 AE 展开 `GL(2)` 的三个基本模型：主级数对应两个一维参数的直和，Steinberg twist 对应非零 monodromy 的 special parameter，supercuspidal 对应不可约二维 Weil 表示。这是理解一般 `GL(n)` Langlands 分类前最小的可计算例子。

**收口精修 5.A（局部参数使用表）.** 本章之后的局部参数只按以下层次使用：

| 层次 | 参数对象 | 状态 |
|---|---|---|
| `GL(1)` | 一维 Weil 参数 | 由局部类域论证明 |
| 非分歧球表示 | $\widehat G$ 中半单 Satake 共轭类 | 由 Satake 同构给出接口 |
| `GL(n)` | $n$ 维 Frobenius-semisimple Weil-Deligne 表示 | 外部输入定理 |
| 一般 reductive group | ${}^LG$-值参数和 L-packet | 猜想或已知特殊情形 |
| 几何局部 LLC | Fargues-Scholze semisimple 参数框架 | 附录 AC 接口，不替代 enhanced LLC |

## 5.7 一般 reductive 群的局部 Langlands 猜想

**猜想 5.16（局部 Langlands，L-packet 形式）.** 设 $G/F$ 为 connected reductive group。满足定义 5.15 中连续性、半单性、代数性和 bounded/admissible 条件的 Langlands 参数
$$
\varphi:W_F'\to{}^LG
$$
的 $\widehat G$-共轭类应对应 $G(F)$ 的不可约可容许表示的有限集合
$$
\Pi_\varphi(G),
$$
称为 L-packet。该对应应满足：

1. packet 内表示具有相同的稳定分布特征数据。
2. 对每个有限维表示 $r:{}^LG\to\operatorname{GL}(V)$，局部因子
   $$
   L(s,\pi,r),\quad \varepsilon(s,\pi,r,\psi)
   $$
   只依赖于参数 $\varphi$ 和 $r$。
3. 对 endoscopy 和内形式有相容的 transfer 规则。
4. 对 $G=\operatorname{GL}_n$，每个 packet 只有一个元素，并退化为定理 5.15。

**注 5.17.** 一般 reductive 群的局部 Langlands 在许多情形已知，但完整陈述需要精确处理 pure inner forms、增强参数、component groups 和 endoscopy。本书后续章节会逐步引入这些修正。

## 5.8 本章小结

Weil 群把 Galois 群的 profinite 拓扑改造成适合复表示和 L 函数的局部对象。Weil-Deligne 表示进一步记录单值 monodromy 算子 $N$，从而描述分歧表示的局部 L 因子。`GL(1)` 局部 Langlands 正是局部类域论；`GL(n)` 局部 Langlands 把不可约可容许表示与 $n$ 维 Weil-Deligne 表示对应；一般 reductive 群则需要 L 群和 L-packet。

## 练习

**练习 5.1.** 证明 $W_F$ 在 $G_F$ 中稠密，并说明为什么 $W_F/I_F$ 是 $\mathbb Z$ 而不是 $\widehat{\mathbb Z}$。

**练习 5.2.** 设 $\chi:F^\times\to\mathbb C^\times$ 为非分歧特征。用定理 5.14 计算对应 Weil 参数在 $\operatorname{Fr}_F$ 上的值。

**练习 5.3.** 设 $(V,r,N)$ 为 Weil-Deligne 表示。证明 $(\ker N)^{I_F}$ 被 $r(\operatorname{Fr}_F)$ 保持。

**练习 5.4.** 对 $G=\operatorname{GL}_2$，写出非分歧主级数表示对应的二维非分歧 Weil 参数。
