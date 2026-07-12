# 附录 U：p-adic Hodge、Shimura Varieties 和 Cohomological Automorphic Galois 表示

收口归一化回指：本附录涉及 RAECSDC Galois 表示、局部-整体相容、Tate twists、Hodge-Tate weights 和 automorphic normalization；统一约定见 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 第 5、6、7、8 节。

## U.1 数域高维 Galois 表示构造的形状

第十四章陈述数域上 regular algebraic automorphic representations 的 Galois 表示构造。本附录固定其技术接口。

**定义 U.1.** 设 $K$ 为数域，$\pi$ 为 $\operatorname{GL}_n(\mathbb A_K)$ 的 cuspidal automorphic representation。称 $\pi$ regular algebraic，若对每个 Archimedean embedding，其 infinitesimal character 与某个 irreducible algebraic representation 的 infinitesimal character 相同，且相应 highest-weight integers 互异。Hodge-Tate weights 是由此预测并在 Galois 表示构造后验证的输出，不能放进定义造成循环。

**定义 U.2.** 若 $K$ 为 CM field，称 $\pi$ essentially conjugate self-dual，若存在 algebraic Hecke character $\chi$ 使
$$
\pi^\vee\simeq\pi^c\otimes\chi
$$
其中 $c$ 为 CM complex conjugation。若 $K$ totally real，则相应条件另写为
$\pi^\vee\simeq\pi\otimes\chi$。两种条件连同 $\chi$ 的 parity/polarization sign 构成来源定理的 polarization hypotheses。

**外部输入定理 U.3（polarizable regular algebraic Galois representations）.** 设 $K$ 为 CM 或 totally real field，$\pi$ 为 regular algebraic cuspidal automorphic representation of $\operatorname{GL}_n(\mathbb A_K)$，并满足所引用构造定理的 conjugate-self-dual/self-dual polarization、parity 和 ramification hypotheses。则对每个 $\ell$ 和同构 $\iota:\overline{\mathbb Q}_\ell\simeq\mathbb C$，存在连续半单表示
$$
r_{\ell,\iota}(\pi):G_K\to\operatorname{GL}_n(\overline{\mathbb Q}_\ell)
$$
设 $w$ 为该 compatible system 的 purity weight，并令
$\pi^{\mathrm{alg}}=\pi\otimes|\det|^{-w/2}$。对每个 $v\nmid\ell$，若
$\pi_v$ 与 $K_v$ 均非分歧，则
$$
\det(1-r_{\ell,\iota}(\pi)(\operatorname{Frob}^{\operatorname{arith}}_v)X)
$$
等于 $\pi_v^{\mathrm{alg}}$ 的 Satake polynomial 经 $\iota^{-1}$ 转换后的多项式。这里 arithmetic Frobenius、algebraic twist 和 coefficient isomorphism 已固定，不再用“可能取逆”代替公式。

**注 U.4.** 定理 U.3 是一个定理族的共同接口，不宣称单篇文献在同一假设下覆盖所有 $K,n,\pi$。每次实际调用必须选定 Harris-Taylor、Shin、Scholze、Caraiani 或 Harris-Lan-Taylor-Thorne 等具体版本，并核对 ramification、polarization、cohomological weight 和 local conditions。本书不由该接口推出非 polarizable 或非 regular 情形。

## U.2 Shimura varieties 的上同调接口

**定义 U.5.** Shimura datum 是一对 $(G,X)$，其中 $G/\mathbb Q$ 为 reductive group，$X$ 为 Deligne torus 到 $G_\mathbb R$ 的共轭类，满足 Shimura axioms。对紧开子群 $K\subset G(\mathbb A_f)$，Shimura variety 记为
$$
\operatorname{Sh}_K(G,X).
$$

**外部输入定理 U.6（Shimura varieties and Hecke-Galois actions）.** Shimura varieties 有 canonical models，其 $\ell$-adic cohomology
$$
H^i(\operatorname{Sh}_K(G,X)_{\overline E},\mathcal L_\xi)
$$
同时带有 Hecke algebra 作用和 $\operatorname{Gal}(\overline E/E)$ 作用。这里 $E$ 为 reflex field，$\mathcal L_\xi$ 为代数表示 $\xi$ 给出的 local system。

**命题 U.7.** 若 automorphic representation $\Pi$ 出现在 Shimura variety 的 cohomology 中，则 Hecke eigensystem 与 Galois representation 的 Frobenius trace 可在同一 cohomology 空间中比较。

**证明.** Hecke algebra 与 Galois group 的作用都定义在同一个 $\ell$-adic cohomology 上，且二者交换。对某个 Hecke maximal ideal 或 eigensystem 局部化后，Galois 作用保留该局部化子空间。于是 Frobenius 在该子空间上的 characteristic polynomial 可与 Hecke polynomial 比较。具体相等来自 Shimura variety 的局部模型、nearby cycles 或 trace formula 计算，作为外部输入。$\square$

## U.3 从一般 `GL(n)` 到 Shimura varieties

许多 $\operatorname{GL}_n$ 表示不直接来自 $\operatorname{GL}_n$ 的 Shimura variety，因为 $\operatorname{GL}_n$ 本身通常没有可产生所需 Hermitian symmetric domain 的 Shimura datum。常用策略是把它们转移到 unitary 或 similitude groups。

**外部输入定理 U.8（Unitary group realization）.** 在 conjugate self-duality 或 polarization、CM/totally real base field、base change descent 和 cohomological weight 条件均固定的情形，regular algebraic conjugate self-dual $\operatorname{GL}_n$ 表示可与 unitary group 的 cohomological automorphic representations 相关，并出现在相应 Shimura varieties 的 cohomology 中。

**命题 U.9.** 定理 U.8 解释了为什么 polarizable/self-dual 假设常出现在数域 Galois 表示构造中。

**证明.** Shimura varieties 的 reductive group 通常是 unitary、symplectic 或 similitude 型。要把 $\operatorname{GL}_n$ 的 automorphic representation 放入这些群的 cohomology，需要它来自这些群的 base change 或 endoscopic transfer。Unitary group 的标准表示对偶性强制 $\operatorname{GL}_n$ 侧满足 conjugate self-dual 或 polarizable 条件。因此该假设不是形式装饰，而是进入 Shimura variety cohomology 的结构条件。$\square$

## U.4 p-adic Hodge 条件

设 $F/\mathbb Q_p$ 为有限扩张。

**定义 U.10.** 令 $V$ 为 $n$ 维 $\mathbb Q_p$-向量空间，带连续 $G_F$-作用。定义
$$
D_{\operatorname{dR}}(V)
=
(V\otimes_{\mathbb Q_p}B_{\operatorname{dR}})^{G_F},
$$
它是带 filtration 的 $F$-向量空间。称 $V$ de Rham，若
$$
\dim_FD_{\operatorname{dR}}(V)=\dim_{\mathbb Q_p}V=n.
$$
若表示以有限扩张 $E/\mathbb Q_p$ 为系数，则在
$F\otimes_{\mathbb Q_p}E$ 的各 embedding 分量上采用同一 rank 条件；$\overline{\mathbb Q}_p$-值表示必须先下降到某个有限 $E$，不能把
$\overline{\mathbb Q}_p$ 本身当作有限维系数域。按本书
$\operatorname{HT}(\chi_p)=\{1\}$ 的 convention，整数 $i$ 的重数为
$$
\dim_F\operatorname{gr}^{-i}D_{\operatorname{dR}}(V)
$$
（有系数时逐 embedding 取相应维数）。

**外部输入定理 U.11（p-adic comparison，按约化类型分层）.** 对 characteristic-$0$ local field 上 proper smooth variety，$p$-adic etale cohomology 是 de Rham；若有 suitable good reduction，则 crystalline comparison 给出 crystalline 表示；若有 suitable semistable model，则 semistable comparison 给出 semistable 表示。非 proper Shimura varieties 必须使用 compact support、boundary compactification 或 intersection cohomology 的相应版本。因而“来自 cohomology”本身只保证在满足具体几何假设时具有相应 p-adic Hodge 性质，不能无条件写成 crystalline。

**外部输入定理 U.12（Hodge-Tate weights）.** 在 U.3 所选具体构造定理中，$r_{\ell,\iota}(\pi)$ 在 $v\mid\ell$ 处 de Rham，其按
$\operatorname{HT}(\chi_\ell)=\{1\}$ 编号的 Hodge-Tate multiset 由 $\pi$ 在对应 embedding 的 algebraic highest weight 明确决定，并因 regularity 而无重数。

**证明路线（外部输入）.** Archimedean infinitesimal character 确定 algebraic local system 的 highest weight；p-adic comparison 把其 Hodge filtration 送到 Galois representation。具体整数公式随 $C$-algebraic/$L$-algebraic 和 Hodge-Tate sign conventions 改变，故本书把计算保留在所引用构造定理中，不以本段代替证明。

## U.5 局部-整体相容

**外部输入定理 U.13（局部-整体相容，$v\nmid\ell$）.** 在 U.3 所选具体构造定理覆盖的局部类型中，若 $v\nmid\ell$，则
$$
\operatorname{rec}_{K_v,n}(\pi_v)
\cong
\iota\,\operatorname{WD}
(r_{\ell,\iota}(\pi)^\vee|_{G_{K_v}})^{\mathrm{F\text{-}ss}}
\otimes|\cdot|^{w/2}.
$$
这里 $\pi$ 是 unitary normalization，$w$ 是 U.3 的 purity weight，$\operatorname{WD}$ 使用几何 Frobenius。许多来源只先证明 Frobenius-semisimplification；保留 monodromy rank 的更强相容需逐个局部类型另核对。

**研究边界 U.13.1（$v\mid\ell$）.** 当 $v\mid\ell$ 时，必须先用
$D_{\operatorname{pst}}$ 或 crystalline/de Rham comparison 从 p-adic representation 产生 filtered
$(\varphi,N)$ data，再与 $\pi_v$ 比较。U.13 的 $v\nmid\ell$ Weil-Deligne functor 不能直接套用；一般
p-adic local-global compatibility、trianguline refinements 和 Banach representation correspondences 也不由 U.13 陈述。

**命题 U.14.** 几乎所有非分歧位置的 Satake-Frobenius 相容是局部-整体相容的弱形式。

**证明.** 若 $v\nmid\ell$ 且 $\pi_v$ 非分歧，则 local parameter 惯性平凡且 $N=0$，由 Satake class 决定。U.3 给出 arithmetic Frobenius 与 $\pi_v^{\mathrm{alg}}$ 的 characteristic polynomial；取 Galois dual、改用 geometric Frobenius，再张量 $|\cdot|^{w/2}$，恰把 algebraic roots 变成 unitary Satake roots。因此 U.13 在该处退化为 U.3 经归一化转换后的共轭类相等。$\square$

## U.6 与模性提升和费马应用的接口

**外部输入定理族 U.15（Automorphy lifting interface）.** 设 $\overline\rho:G_K\to\operatorname{GL}_n(\overline{\mathbb F}_\ell)$ 是 residual representation。选定一条具体 automorphy lifting theorem，并逐项满足其 residual automorphy、adequacy/big-image、polarization、multiplier parity、regular Hodge-Tate weights、ramified local deformation conditions、$v\mid\ell$ potentially diagonalizable/ordinary conditions及 patching hypotheses 后，该定理可把指定 lift 的 automorphy 推出。不存在把这些占位条件全部省略后仍成立的统一 U.15；正文调用时必须指向具体来源版本。

**命题 U.16.** Taylor-Wiles patching 是定理 U.15 的代数核心，而 p-adic Hodge theory 提供局部条件。

**证明.** Patching 比较 deformation ring 和 Hecke algebra，需要为 $v\mid\ell$ 指定局部变形环。De Rham、crystalline、ordinary 或 potentially diagonalizable 条件由 p-adic Hodge theory 定义；具体 lifting theorem 还须证明相应局部环具有所需维数、分量和光滑性性质。满足 U.15 已选版本的这些外部局部结论后，全局 patching 才能导出 $R=T$ 或 automorphy lifting。$\square$

## U.7 边界

**命题 U.17.** 定理 U.3 不等于数域完整 Langlands 对应。

**证明.** 定理 U.3 只覆盖 regular algebraic、polarizable 或可进入 Shimura variety/cohomology 方法的 automorphic representations。一般 Maass forms、非 cohomological representations、非 self-dual `GL(n)` 表示和一般 reductive groups 的完整参数化不由该定理覆盖。因此它是数域 Langlands 的巨大已知部分，而不是完整纲领。$\square$

## 练习

**练习 U.1.** 解释 regular algebraic 条件为什么与 Hodge-Tate weights 相关。

**练习 U.2.** 说明 Shimura variety cohomology 为什么同时携带 Hecke 和 Galois 作用。

**练习 U.3.** 解释 conjugate self-dual 假设为什么常用于 unitary group 方法。

**练习 U.4.** 说明局部-整体相容在非分歧位置退化为 Satake-Frobenius 相容。

**练习 U.5.** 解释 p-adic Hodge 条件在 automorphy lifting 中的角色。
