# 附录 U：p-adic Hodge、Shimura Varieties 和 Cohomological Automorphic Galois 表示

收口归一化回指：本附录涉及 RAECSDC Galois 表示、局部-整体相容、Tate twists、Hodge-Tate weights 和 automorphic normalization；统一约定见 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 第 5、6、7、8 节。

## U.1 数域高维 Galois 表示构造的形状

第十四章陈述数域上 regular algebraic automorphic representations 的 Galois 表示构造。本附录固定其技术接口。

**定义 U.1.** 设 $\pi$ 为 $\operatorname{GL}_n(\mathbb A_K)$ 的 cuspidal automorphic representation。称 $\pi$ regular algebraic，若其 Archimedean infinitesimal character 与某个代数表示的 infinitesimal character 相同，且 Hodge-Tate weights 在每个嵌入处互异。

**定义 U.2.** 称 $\pi$ essentially conjugate self-dual，若存在 Hecke character $\chi$ 使
$$
\pi^\vee\simeq\pi^c\otimes\chi
$$
其中 $c$ 为 CM extension 的复共轭。Totally real 情形中相应条件为 essentially self-dual。

**外部输入定理 U.3（RAECSDC Galois representations）.** 设 $K$ 为 CM 或 totally real field，$\pi$ 为 regular algebraic essentially conjugate self-dual cuspidal automorphic representation of $\operatorname{GL}_n(\mathbb A_K)$。则对每个 $\ell$ 和同构 $\iota:\overline{\mathbb Q}_\ell\simeq\mathbb C$，存在连续半单表示
$$
r_{\ell,\iota}(\pi):G_K\to\operatorname{GL}_n(\overline{\mathbb Q}_\ell)
$$
使得对几乎所有 $v\nmid\ell$，
$$
\det(1-r_{\ell,\iota}(\pi)(\operatorname{Frob}^{\operatorname{arith}}_v)X)
$$
等于 $\pi_v$ 的 Satake polynomial 经 $\iota^{-1}$ 转换后的多项式，按本书 Frobenius convention 可能需取对偶或逆。

**注 U.4.** 定理 U.3 汇总 Harris-Taylor、Clozel、Taylor、Shin、Chenevier-Harris、Scholze、Caraiani、Harris-Lan-Taylor-Thorne 等路线的接口。不同定理的假设在 ramification、polarization、cohomological weight 和 local conditions 上有所不同。

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

**定义 U.10.** $\ell=p$ 时，表示
$$
\rho:G_F\to\operatorname{GL}_n(\overline{\mathbb Q}_p)
$$
称为 de Rham，若 Fontaine 的
$$
D_{\operatorname{dR}}(\rho)=(\rho\otimes_{\mathbb Q_p}B_{\operatorname{dR}})^{G_F}
$$
维数等于 $n$。Hodge-Tate weights 由 graded pieces of $D_{\operatorname{dR}}$ 给出。

**外部输入定理 U.11（p-adic Hodge comparison）.** Shimura varieties 或相关代数簇的 $\ell=p$ étale cohomology 与 de Rham/crystalline cohomology 通过 p-adic comparison isomorphisms 相连。因此从 cohomology 构造的 Galois representations 在 $v\mid p$ 处满足 de Rham、crystalline 或 semistable 条件，并有可计算的 Hodge-Tate weights。

**命题 U.12.** Regular algebraic weight 预期决定 $r_{\ell,\iota}(\pi)$ 的 Hodge-Tate weights。

**证明草图.** Automorphic representation 的 Archimedean infinitesimal character 对应代数 local system $\mathcal L_\xi$ 的 highest weight。Shimura variety cohomology 中出现的 Galois representation 的 de Rham realization 与代数 de Rham cohomology 比较。Hodge filtration 由 $\mathcal L_\xi$ 和 Shimura datum 的 Hodge cocharacter 计算。因此 Hodge-Tate weights 由 $\xi$ 决定。完整公式依赖归一化和嵌入 convention。$\square$

## U.5 局部-整体相容

**外部输入定理 U.13（局部-整体相容，接口形式）.** 对定理 U.3 中的 $r_{\ell,\iota}(\pi)$，若 $v\nmid\ell$，则 Weil-Deligne representation
$$
\operatorname{WD}(r_{\ell,\iota}(\pi)|_{G_{K_v}})
$$
与 $\pi_v$ 的 local Langlands parameter 相容，至少在 Frobenius-semisimplification 后成立；在许多情形中有更强的 monodromy 相容。

**命题 U.14.** 几乎所有非分歧位置的 Satake-Frobenius 相容是局部-整体相容的弱形式。

**证明.** 若 $v\nmid\ell$ 且 $\pi_v$ 非分歧，则 local Langlands parameter 惯性平凡，monodromy $N=0$，由 Satake parameter 决定。定理 U.13 在该处化为 Frobenius 半单共轭类相等，即定理 U.3 中 characteristic polynomial 相等的陈述。$\square$

## U.6 与模性提升和费马应用的接口

**外部输入定理 U.15（Automorphy lifting interface）.** 设 $\overline\rho:G_K\to\operatorname{GL}_n(\overline{\mathbb F}_\ell)$ 是 residual representation。若某个 lift 已知 automorphic，且另一个 lift 满足已固定的 ramified 与 $v\mid\ell$ 局部 deformation conditions、adequacy、polarization、regular Hodge-Tate weights 和 Taylor-Wiles patching 所需全局假设，则可推出该 lift automorphic。

**命题 U.16.** Taylor-Wiles patching 是定理 U.15 的代数核心，而 p-adic Hodge theory 提供局部条件。

**证明.** Patching 比较 deformation ring 和 Hecke algebra，需要为 $v\mid\ell$ 指定局部变形环。De Rham、crystalline、ordinary 或 potentially diagonalizable 等条件由 p-adic Hodge theory 定义，并保证局部变形环有可控维数和几何性质。全局 patching 把这些局部环与 Hecke module 拼合，得到 $R=T$ 或 automorphy lifting。$\square$

## U.7 边界

**命题 U.17.** 定理 U.3 不等于数域完整 Langlands 对应。

**证明.** 定理 U.3 只覆盖 regular algebraic、polarizable 或可进入 Shimura variety/cohomology 方法的 automorphic representations。一般 Maass forms、非 cohomological representations、非 self-dual `GL(n)` 表示和一般 reductive groups 的完整参数化不由该定理覆盖。因此它是数域 Langlands 的巨大已知部分，而不是完整纲领。$\square$

## 练习

**练习 U.1.** 解释 regular algebraic 条件为什么与 Hodge-Tate weights 相关。

**练习 U.2.** 说明 Shimura variety cohomology 为什么同时携带 Hecke 和 Galois 作用。

**练习 U.3.** 解释 conjugate self-dual 假设为什么常用于 unitary group 方法。

**练习 U.4.** 说明局部-整体相容在非分歧位置退化为 Satake-Frobenius 相容。

**练习 U.5.** 解释 p-adic Hodge 条件在 automorphy lifting 中的角色。
