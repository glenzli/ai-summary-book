# 第十四章：`GL(n)` 的 Langlands 对应与已知定理

## 本章目标

本章把前面的一般语言专门用于 $G=\operatorname{GL}_n$。这是 Langlands 纲领中最完整、最可计算的一族群：局部 L-packet 退化为单元素，局部 Langlands 对应是定理；标准 L 函数、Rankin-Selberg L 函数和强重数一都有成熟理论；在函数域上，`GL(n)` 的全局 Langlands 对应由 Lafforgue 证明。数域情形的全局 Galois-自守对应仍只在若干重要情形中已知，本章会把定理、猜想和接口分开陈述。

## 依赖前置知识

需要第五章的 Weil-Deligne 表示，第十二章的局部 Langlands 猜想，第十三章的全局自守表示和 L 函数。需要知道 `GL(n)` 的 parabolic subgroups、Levi subgroups、归一化抛物诱导和标准表示。本章把 Bernstein-Zelevinsky 分类、局部 Langlands for `GL(n)`、Godement-Jacquet、Rankin-Selberg、强重数一、converse theorem、Lafforgue 函数域全局定理和数域中 regular algebraic automorphic representations 的 Galois 表示构造作为外部输入。附录 AE 给出 `GL(2)` 的 principal series、Steinberg 和 supercuspidal 低维模型，附录 Q 给出一般 `GL(n)` 的 Bernstein-Zelevinsky 接口。

收口归一化回指：本章集中比较 `GL(n)` LLC、Rankin-Selberg 因子、函数域 Galois 表示和数域 regular algebraic 表示；Frobenius、Satake、Tate twist 和 L 函数变量 convention 见 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 第 2、4、6、7、8 节。

## 14.1 `GL(n)` 的特殊性

设 $F$ 为局部域或整体域的完备化。对 $G=\operatorname{GL}_{n,F}$，第十一章给出
$$
\widehat G=\operatorname{GL}_n(\mathbb C),
\qquad
{}^LG=\operatorname{GL}_n(\mathbb C)\times W_F
$$
在 split 情形成立。标准表示为
$$
\operatorname{Std}:\operatorname{GL}_n(\mathbb C)\to\operatorname{GL}_n(\mathbb C).
$$

**命题 14.1.** `GL(n)` 的 L-packet 均为单元素。

**证明.** 第十二章命题 12.15 已说明：`GL(n)` 参数的 centralizer component group 平凡。因此 enhanced parameter 没有非平凡 packet 内部标签。局部 Langlands 对应中每个参数对应一个不可约可容许表示。$\square$

**注 14.2.** 这是 `GL(n)` 与 classical groups 的关键区别。对 $\operatorname{SL}_n$、symplectic groups、orthogonal groups，参数的 centralizer 可能非连通，L-packet 通常含多个表示。

## 14.2 局部 `GL(n)` Langlands 对应

设 $F$ 为非 Archimedean 局部域。记
$$
\operatorname{WDRep}_n(F)
$$
为 $n$ 维 Frobenius-semisimple Weil-Deligne 表示的同构类集合。

**外部输入定理 14.3（局部 Langlands for `GL(n)`）.** 存在唯一自然双射
$$
\operatorname{rec}_{F,n}:
\operatorname{Irr}(\operatorname{GL}_n(F))
\longrightarrow
\operatorname{WDRep}_n(F),
$$
满足：

1. $n=1$ 时，$\operatorname{rec}_{F,1}$ 等于第五章的局部类域论。
2. 中心特征相容：
   $$
   \omega_\pi=\det(\operatorname{rec}_{F,n}(\pi))
   $$
   其中右侧通过局部类域论看作 $F^\times$ 的 character。
3. 非分歧相容：若 $\pi$ 为 spherical 且 Satake 参数为 semisimple conjugacy class $s_\pi\subset\operatorname{GL}_n(\mathbb C)$，则
   $$
   \operatorname{rec}_{F,n}(\pi)(\operatorname{Fr}_F)=s_\pi
   $$
   在几何 Frobenius 归一化下成立。
4. 对任意 character $\chi:F^\times\to\mathbb C^\times$，
   $$
   \operatorname{rec}_{F,n}(\pi\otimes(\chi\circ\det))
   =
   \operatorname{rec}_{F,n}(\pi)\otimes\operatorname{rec}_{F,1}(\chi).
   $$
5. 反表示相容：
   $$
   \operatorname{rec}_{F,n}(\pi^\vee)
   =
   \operatorname{rec}_{F,n}(\pi)^\vee.
   $$
6. Rankin-Selberg 局部因子相容：对 $\pi_i\in\operatorname{Irr}(\operatorname{GL}_{n_i}(F))$，
   $$
   L(s,\pi_1\times\pi_2)
   =
   L(s,\operatorname{rec}_{F,n_1}(\pi_1)\otimes\operatorname{rec}_{F,n_2}(\pi_2))
   $$
   且 $\varepsilon$ 因子、$\gamma$ 因子相容。
7. Tempered 表示对应 bounded Weil-Deligne 参数；essentially square-integrable 表示对应 indecomposable Weil-Deligne 参数。

**注 14.4.** 该定理是局部 Langlands 对应中最完整的核心定理之一。`GL(2)` 可由 Bushnell-Henniart 理论处理；一般 `GL(n)` 由 Harris-Taylor、Henniart 等工作建立，Scholze 给出几何证明路线。

**命题 14.5.** 若 $\pi$ 为 $\operatorname{GL}_n(F)$ 的非分歧主级数，其 Satake 参数为
$$
\operatorname{diag}(\alpha_1,\ldots,\alpha_n),
$$
则标准局部因子为
$$
L(s,\pi,\operatorname{Std})
=
\prod_{i=1}^n(1-\alpha_iq^{-s})^{-1}.
$$

**证明.** 由定理 14.3 的非分歧相容性，$\operatorname{rec}_{F,n}(\pi)(\operatorname{Fr}_F)$ 的半单共轭类为 $\operatorname{diag}(\alpha_1,\ldots,\alpha_n)$。标准局部因子按第五章 Weil-Deligne 因子定义为
$$
\det(1-q^{-s}\operatorname{rec}_{F,n}(\pi)(\operatorname{Fr}_F))^{-1}.
$$
代入该 diagonal matrix 得到公式。$\square$

## 14.3 Langlands 分类与局部参数的加法性

`GL(n)` 的局部表示可以由 essentially square-integrable 表示经抛物诱导组织起来。

**外部输入定理 14.6（Bernstein-Zelevinsky 与 Langlands 分类，接口形式）.** 设 $F$ 为非 Archimedean 局部域。每个 $\pi\in\operatorname{Irr}(\operatorname{GL}_n(F))$ 可唯一表示为归一化抛物诱导
$$
\operatorname{Ind}_{P(F)}^{\operatorname{GL}_n(F)}
(\delta_1|\det|^{t_1}\otimes\cdots\otimes\delta_r|\det|^{t_r})
$$
的 Langlands quotient，其中 $\delta_i$ 为 essentially square-integrable representations of $\operatorname{GL}_{n_i}(F)$，$t_1>\cdots>t_r$ 为实数，且 $\sum_i n_i=n$。

在局部 Langlands 对应下，该 quotient 的参数为 direct sum
$$
\bigoplus_{i=1}^r
\operatorname{rec}_{F,n_i}(\delta_i)\otimes|\cdot|^{t_i}.
$$

**注 14.6.1.** 附录 Q 把定理 14.6 拆成 cuspidal support、segments、multisegments、Langlands quotient theorem、tempered/generic 分类和局部因子相容。本章只使用这些结果组织 `GL(n)` 对应的已知定理边界。

**注 14.7.** 定理 14.6 说明 `GL(n)` 的局部 LLC 与抛物诱导相容：表示侧的 parabolic induction 对应参数侧的 direct sum。一般 reductive 群中，这一语句会被 L-packet、R-groups 和 reducibility 现象复杂化。

## 14.4 全局 `GL(n)` 自守表示

设 $K$ 为整体域，$\pi$ 为 cuspidal automorphic representation of $\operatorname{GL}_n(\mathbb A_K)$。由第十三章，
$$
\pi=\bigotimes_v'\pi_v.
$$
对每个位置 $v$，局部 Langlands 给出参数
$$
\varphi_{\pi_v}=\operatorname{rec}_{K_v,n}(\pi_v).
$$

**定义 14.8.** $\pi$ 的标准 L 函数定义为
$$
L(s,\pi)=L(s,\pi,\operatorname{Std})
=\prod_vL(s,\pi_v,\operatorname{Std})
$$
在局部因子均已定义的意义下。若 $S$ 包含所有 ramified 和 Archimedean 位置，则
$$
L^S(s,\pi)=\prod_{v\notin S}L(s,\pi_v,\operatorname{Std}).
$$

**定义 14.9.** 若 $\pi$ 为 $\operatorname{GL}_n(\mathbb A_K)$ 的自守表示，$\pi'$ 为 $\operatorname{GL}_m(\mathbb A_K)$ 的自守表示，则 Rankin-Selberg L 函数的非分歧局部因子定义为
$$
L(s,\pi_v\times\pi_v')
=
L(s,\varphi_{\pi_v}\otimes\varphi_{\pi_v'}).
$$
全局部分 L 函数为
$$
L^S(s,\pi\times\pi')
=
\prod_{v\notin S}L(s,\pi_v\times\pi_v').
$$

**外部输入定理 14.10（Godement-Jacquet 标准 L 函数）.** 若 $\pi$ 为 cuspidal automorphic representation of $\operatorname{GL}_n(\mathbb A_K)$，则完全标准 L 函数 $L(s,\pi)$ 有 meromorphic continuation 和函数方程。若 $n>1$，该 L 函数为 entire；若 $n=1$，平凡 character 的 L 函数有 Tate thesis 中的标准极点。

**外部输入定理 14.11（Rankin-Selberg L 函数）.** 若 $\pi$ 和 $\pi'$ 分别为 $\operatorname{GL}_n(\mathbb A_K)$ 与 $\operatorname{GL}_m(\mathbb A_K)$ 的 cuspidal automorphic representations，则 $L(s,\pi\times\pi')$ 有 meromorphic continuation 和函数方程。其极点由 $\pi'$ 与 $\pi^\vee$ 的 twist 关系控制；精确陈述依赖中心特征和归一化。

**注 14.12.** 定理 14.10 和 14.11 是 `GL(n)` 理论比一般还原群更强的地方。它们不是形式 Euler 乘积的直接结果，而来自全局 zeta integrals、unfolding 和局部函数方程。

**注 14.12.1.** 附录 I 给出这些全局 zeta integrals 的统一接口：Godement-Jacquet 积分处理标准 L 函数，Rankin-Selberg 积分处理 `GL(n)\times GL(m)` L 函数，converse theorem 把足够多 twist 的解析性质转化为 automorphy。

## 14.5 强重数一与全局确定性

**外部输入定理 14.13（强重数一）.** 设 $\pi,\pi'$ 为 cuspidal automorphic representations of $\operatorname{GL}_n(\mathbb A_K)$。若存在有限集合 $S$，使得对所有 $v\notin S$ 有
$$
\pi_v\cong\pi_v',
$$
则
$$
\pi\cong\pi'.
$$

**推论 14.14.** Cuspidal automorphic representation of $\operatorname{GL}_n(\mathbb A_K)$ 由几乎所有非分歧 Satake 参数唯一决定。

**证明.** 几乎所有非分歧 Satake 参数相同意味着几乎所有位置的 spherical representations 同构。由定理 14.13，两个全局表示同构。$\square$

**推论 14.15.** 若 $\pi,\pi'$ 为 cuspidal automorphic representations of $\operatorname{GL}_n(\mathbb A_K)$，且对几乎所有 $v$ 有
$$
L(s,\pi_v,\operatorname{Std})=L(s,\pi_v',\operatorname{Std})
$$
并且这些局部标准因子的 Satake 多项式相同，则 $\pi\cong\pi'$。

**证明.** 局部标准因子的倒数给出 Satake 参数在标准表示下的 characteristic polynomial。对 $\operatorname{GL}_n(\mathbb C)$，semisimple conjugacy class 由 characteristic polynomial 决定。因此几乎所有 Satake 参数相同。应用推论 14.14。$\square$

## 14.6 Converse theorem 与函子性检测

`GL(n)` 的 converse theorem 是全局函子性证明中的核心工具：若一个候选表示有足够多扭曲 L 函数的解析性质，则它是自守的。

**外部输入定理 14.16（Cogdell-Piatetski-Shapiro converse theorem，接口形式）.** 设 $\Pi=\otimes_v'\Pi_v$ 为 $\operatorname{GL}_n(\mathbb A_K)$ 的 irreducible admissible representation，满足适当中心特征、局部有限性和单位性条件。若对所有足够多的 cuspidal automorphic representations $\tau$ of $\operatorname{GL}_m(\mathbb A_K)$，$1\le m\le n-1$，扭曲 L 函数
$$
L(s,\Pi\times\tau)
$$
具有期望的解析延拓、函数方程和有界性条件，则 $\Pi$ 是 automorphic。

**注 14.17.** Converse theorem 常用于证明从某个 L 群同态
$$
{}^LH\to{}^L\operatorname{GL}_N
$$
得到的候选局部参数确实来自一个 $\operatorname{GL}_N$ 的全局自守表示。它把函子性问题转化为 L 函数解析性质问题。

## 14.7 函数域上的全局 `GL(n)` Langlands

设 $K$ 为有限域 $\mathbb F_q$ 上光滑射影几何连通曲线的函数域。设 $\ell$ 为不等于 $\operatorname{char}K$ 的素数。

**定义 14.18.** 一个 $\ell$-adic Galois 表示
$$
\rho:G_K\to\operatorname{GL}_n(\overline{\mathbb Q}_\ell)
$$
称为几乎处处非分歧，若除有限多个位置 $v$ 外，$\rho$ 在 inertia subgroup $I_v$ 上平凡。

**外部输入定理 14.19（Lafforgue，全局 Langlands for `GL(n)` over function fields）.** 在有限阶中心特征和有限阶 determinant 的对应归一化下，函数域 $K$ 上 cuspidal automorphic representations of $\operatorname{GL}_n(\mathbb A_K)$ 与 $n$ 维不可约 $\ell$-adic Galois representations
$$
\rho:G_K\to\operatorname{GL}_n(\overline{\mathbb Q}_\ell)
$$
之间存在对应。该对应满足：对几乎所有非分歧位置 $v$，$\pi_v$ 的 Satake 参数与 $\rho(\operatorname{Frob}_v^{\operatorname{arith}})$ 的 characteristic polynomial 相同，按固定 Frobenius convention 需要取逆或对偶调整。

**注 14.20.** 函数域情形的 Frobenius convention 必须谨慎。本书局部 L 因子默认几何 Frobenius，而 $\ell$-adic Galois 表示文献常用算术 Frobenius。比较 Euler 因子时必须明确采用哪一个。

**注 14.20.1.** 附录 S 给出 Drinfeld、Laurent Lafforgue 和 V. Lafforgue 的函数域接口，并解释 shtukas 与 excursion operators 的角色。

**推论 14.21.** 在函数域 `GL(n)` 情形，几乎所有局部 Euler 因子同时决定自守表示和对应 Galois 表示的半单化。

**证明.** 自守侧由强重数一确定。Galois 侧由 Chebotarev density theorem 确定半单连续表示的特征多项式数据。因此几乎所有非分歧 Euler 因子确定两侧半单对象。$\square$

## 14.8 数域上的已知 Galois 表示接口

数域上的完整全局 `GL(n)` Langlands 对应仍是猜想；但在 regular algebraic 条件下，许多自守表示已经构造出 Galois 表示。

**定义 14.22.** 设 $K$ 为数域。Cuspidal automorphic representation $\pi$ of $\operatorname{GL}_n(\mathbb A_K)$ 称为 algebraic，若其 Archimedean infinitesimal character 与某个代数表示的最高权相容。称为 regular algebraic，若对应权没有重复。若 $\pi^\vee$ 与 $\pi$ 的 Galois 共轭或 character twist 相容，则称为 self-dual 或 conjugate self-dual，具体形式依 $K$ 是否全实、CM 或一般数域而定。

**外部输入定理 14.23（数域中的 Galois 表示构造，接口形式）.** 对许多 regular algebraic cuspidal automorphic representations $\pi$ of $\operatorname{GL}_n(\mathbb A_K)$，特别是满足 conjugate self-dual 或极化条件的情形，存在连续半单 $\ell$-adic Galois 表示
$$
\rho_{\pi,\ell}:G_K\to\operatorname{GL}_n(\overline{\mathbb Q}_\ell)
$$
使得对几乎所有 $v\nmid\ell$ 且 $\pi_v$ 非分歧的位置，有 Euler 因子相容：
$$
\det\left(1-X\rho_{\pi,\ell}(\operatorname{Frob}_v^{\operatorname{arith}})\right)
$$
与 $\pi_v$ 的 Satake polynomial 相同，至多差由归一化 convention 决定的 Tate twist、对偶或 $|\det|$ 平移。

**注 14.24.** 定理 14.23 是接口表述，不是单一来源的完整定理。其证明来自 Shimura varieties、cohomology of locally symmetric spaces、Taylor-Wiles-Kisin patching、Harris-Taylor、Clozel、Taylor、Scholze、Caraiani、Harris-Lan-Taylor-Thorne 等一系列工作。一般数域、一般 regular algebraic $\pi$ 和非极化情形仍需额外假设或属于开放问题。

**注 14.24.1.** 附录 U 将定理 14.23 的常用假设拆成 regular algebraic、polarizable、Shimura variety realization、p-adic Hodge condition 和 local-global compatibility。该附录用于防止把“数域 `GL(n)` 已知构造”误读为完整数域 Langlands 对应。

**注 14.25.** 第九、十章讨论的模形式和椭圆曲线 Galois 表示是 $n=2$、$K=\mathbb Q$ 的特殊来源。费马大定理使用的是非常具体的 `GL(2)/\mathbb Q` 模性与降层，而不是数域上完整 `GL(n)` 全局 Langlands。

## 14.9 全局 `GL(n)` Langlands 猜想

**猜想 14.26（数域 `GL(n)` 全局 Langlands，粗略形式）.** 设 $K$ 为数域。适当的 $n$ 维 Galois、Weil 或 conjectural Langlands group 表示应与 cuspidal automorphic representations of $\operatorname{GL}_n(\mathbb A_K)$ 对应，并满足：

1. 几乎所有位置的 Frobenius characteristic polynomials 与 Satake polynomials 相同。
2. 局部化与局部 Langlands 对应相容。
3. 标准、Rankin-Selberg、对称幂、外方幂和 adjoint L 函数相容。
4. 中心特征对应 determinant。
5. 函子性对应于有限维表示或 L 群同态。

**注 14.27.** 猜想 14.26 不能直接写成 $G_K$ 的复表示与自守表示的双射。数域中应使用 $\ell$-adic Galois 表示、motivic Galois group 或 conjectural Langlands group 的合适版本；Archimedean 参数、Hodge-Tate weights、纯性和代数性条件都必须纳入。

## 14.10 本章小结

`GL(n)` 是 Langlands 纲领中结构最清晰的主线。局部上，`GL(n)` 的 L-packet 是单点，局部 Langlands 是不可约可容许表示与 $n$ 维 Weil-Deligne 表示之间的双射。全局自守侧，cuspidal automorphic representations of `GL(n)` 有标准 L 函数、Rankin-Selberg L 函数、强重数一和 converse theorem。函数域上，Lafforgue 定理给出 `GL(n)` 的全局 Galois-自守对应。数域上，完整全局对应仍是猜想，但 regular algebraic 和极化条件下已有深刻的 Galois 表示构造。

## 练习

**练习 14.1.** 证明 `GL(n)` 的 semisimple conjugacy class 由标准表示的 characteristic polynomial 决定。

**练习 14.2.** 对非分歧 $\pi$，由 Satake 参数推导标准局部 L 因子。

**练习 14.3.** 说明局部 Langlands 中中心特征对应 determinant 如何退化为 $n=1$ 的局部类域论。

**练习 14.4.** 解释 Bernstein-Zelevinsky 分类中 parabolic induction 与参数 direct sum 的对应。

**练习 14.5.** 用强重数一证明几乎所有局部 Satake 参数决定 cuspidal automorphic representation of `GL(n)`。

**练习 14.6.** 说明 converse theorem 如何服务于函子性证明。

**练习 14.7.** 比较函数域 Lafforgue 定理与数域 regular algebraic Galois 表示构造的差异。
