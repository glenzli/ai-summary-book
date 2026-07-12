# 附录 C：六函子、perverse sheaves 与 IC 技术细节

## 本章目标

本附录登记第三章和后续章节使用的六函子、perverse t-structure、IC sheaf、decomposition theorem 和 semismall 特化的技术条件。它不替代 BBD 或 Kashiwara--Schapira 的完整证明，但每次调用必须能在这里判断模型、输入和结论类型。

## C.1 Sheaf theory 模型表

**约定 C.1.** 本书允许三种主要 sheaf theory 模型：

| 模型 | 空间 | 系数 | 主要用途 |
| --- | --- | --- | --- |
| Betti constructible sheaves | finite-type complex algebraic varieties 的 analytification | characteristic-zero field $E$ | 第三至十三章的默认模型、Riemann--Hilbert |
| $\ell$-adic sheaves | finite-type $k$-schemes，$\ell\ne\operatorname{char}k$ | $\mathbb Q_\ell$ 的有限扩张 | weights、purity、finite-field trace；需写 Tate twist |
| mixed Hodge modules | complex algebraic varieties | $\mathbb Q$ 或适当扩域 | decomposition、purity、Hodge-theoretic refinements |

同一证明中若要跨模型，必须给出 comparison functor 或说明只是在类比层面。Betti category 中的 $f_!,f_\ast$ 与 $\ell$-adic category 中同名函子类型不同；相同符号不构成 comparison。

## C.2 Perverse normalization

**约定 C.2.** 若 $X$ 光滑纯维数 $d$，局部系统 $\mathcal L$ 的 perverse normalization 为
$$
\mathcal L[d].
$$
Schubert cell $X_w\simeq\mathbb A^{\ell(w)}$ 上的常值局部系统对应 perverse object
$$
E_{X_w}[\ell(w)].
$$

若复代数群 $H$ 作用于 $X$，本书的 $\operatorname{Perv}_H(X,E)$ 由 forgetful functor 到 $X$ 上的 perverse category 定义，不采用 quotient stack 的 intrinsic dimension shift。若改用使 atlas $X\to[X/H]$ 的 $u^\ast[\dim H]$ t-exact 的 stack convention，必须把所有单位对象、smooth pullback 和 convolution shift 一并转换。

**定义 C.3.** 对 locally closed embedding $j:U\hookrightarrow X$ 和 $\mathcal F\in\operatorname{Perv}(U,E)$，middle extension
$$
j_{!*}:\operatorname{Perv}(U)\to\operatorname{Perv}(X)
$$
定义为 $j_!$ 到 $j_\ast$ 的自然 morphism 的 image：
$$
j_{!*}\mathcal F=\operatorname{im}_{\operatorname{Perv}(X,E)}
\big({}^pH^0(j_!\mathcal F)\longrightarrow{}^pH^0(j_\ast\mathcal F)\big).
$$

这里的 image 取在 perverse heart 中，不是 derived category 中一个未定义的 kernel/image。若 $j$ 是 open immersion，上式中的 morphism 由 $j_!\to j_\ast$ 的自然变换诱导；一般 locally closed 情形通过 open embedding 与 closed embedding 的分解定义，所得对象独立于该分解是外部 formalism 的一部分。

**外部输入定理 C.4.** $j_{!*}$ 的 image 定义给出无 subobject 或 quotient 支撑在边界 $X\setminus U$ 上的唯一延拓。  
来源：BBD。

## C.3 Decomposition theorem 使用规则

**规则 C.5.** 使用 decomposition theorem 时必须记录：

1. morphism $f:X\to Y$ 是否 algebraic、proper；若调用 relative hard Lefschetz，是否 projective；
2. 输入是否为 $\operatorname{IC}_X$，以及 $X$ 光滑时是否已验证 $\operatorname{IC}_X=E_X[\dim X]$；
3. 若输入不是 $\operatorname{IC}_X$，是否有 geometric-origin、purity 或 Hodge-module 假设；
4. 使用 Betti、$\ell$-adic、mixed Hodge module 还是其他版本及其系数；
5. 使用的是非 canonical derived splitting、canonical perverse cohomology，还是 semisimplicity；
6. 是否需要 relative hard Lefschetz、purity、weight 或 trace argument。

**外部输入定理 C.6（decomposition theorem）.** 在第三章约定的 Betti 模型中，令 $f:X\to Y$ proper，$X$ 不可约，$E$ 的特征为 $0$。则
$$
Rf_\ast\operatorname{IC}_X
\simeq
\bigoplus_i{}^pH^i(Rf_\ast\operatorname{IC}_X)[-i],
$$
每个 perverse cohomology object semisimple。右侧 splitting 一般非 canonical。若 $f$ projective 且给定 $f$-ample class $\eta$，才另有 relative hard Lefschetz
$$
\eta^i:{}^pH^{-i}\xrightarrow{\sim}{}^pH^i\qquad(i\ge0).
$$
来源定位：`BBD-2`，BBD 6.2.5；本书中的 Springer sheaf 和低秩 affine-Grassmannian convolution 会调用它。Geometric Satake 的 convolution t-exactness 则使用更直接的 stratified-semismall dimension estimate，不能只写“由 properness”。

**外部输入定理 C.7（semismall 特化）.** 令 $f:Z\to Y$ 为 proper surjective morphism，$Z$ 光滑、连通、纯维数 $n$。设 $Y=\coprod S$ 是使 $f$ stratified locally trivial 的有限 Whitney stratification，并对每个 $s\in S$ 有
$$
2\dim f^{-1}(s)\le n-\dim S.
$$
称等号成立的 $S$ 为 relevant stratum。则 $Rf_\ast E_Z[n]$ 是 semisimple perverse sheaf，且
$$
Rf_\ast E_Z[n]
\simeq
\bigoplus_{S\ \mathrm{relevant}}
\operatorname{IC}(\overline S,\mathcal L_S),
$$
其中 $\mathcal L_S$ 的 fiber 由 $f^{-1}(s)$ 的最大维不可约分支的 top Borel--Moore homology 给出，monodromy 来自这些分支沿 $S$ 的延拓。该分解调用 decomposition theorem 与 relevant-stratum intersection forms 的非退化性，不是本书内部证明。来源定位：`BBD-SS-1`；de Cataldo--Migliorini, *The hard Lefschetz theorem and the topology of semismall maps*。

**反例边界 C.8.** 若系数域有正特征，即使 $f$ proper，$Rf_\ast\operatorname{IC}_X$ 也不保证 semisimple 或分解；若 $f$ 只 proper 而未给 projective structure 和 $f$-ample class，也不能写出上面的 relative hard Lefschetz morphism。故“proper $\Rightarrow$ decomposition + hard Lefschetz”不是本书允许的缩写。

## 本章小结

本附录固定了 sheaf 模型、equivariant perverse shift、middle extension 的 heart-level image、decomposition theorem 的非 canonical splitting，以及 semismall 特化的精确维数条件。后续每个使用 IC、semisimplicity、semismall pushforward 或 relative hard Lefschetz 的章节都必须指回本附录和附录 D 的 locator。
