# 附录 AA：Bruhat-Tits 建筑、Parahoric、Hyperspecial 和非分歧群

**收口归一化回指。** 本附录涉及 hyperspecial subgroup、parahoric subgroup、unramified reductive group 和 Satake 参数；与非分歧局部对应及 spherical Hecke algebra 比较时使用 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 第 1、4、8 节。

## AA.1 建筑的基本对象

设 $F$ 为非 Archimedean 局部域，$G/F$ 为 connected reductive group。

**外部输入定理 AA.1（Bruhat-Tits building）.** 与 $G(F)$ 关联一个 Euclidean building
$$
\mathcal B(G,F)
$$
使 $G(F)$ 作用其上。Apartments 由 maximal $F$-split tori 参数化，facets 对应 parahoric subgroups。

**定义 AA.2.** 若 $x\in\mathcal B(G,F)$，其 stabilizer 的合适紧开子群记为
$$
G(F)_{x,0},
$$
称为 parahoric subgroup。若 facet 为 alcove，对应 parahoric 称为 Iwahori subgroup。

**外部输入定理 AA.3（parahoric group schemes）.** 对每个 facet $\mathfrak f\subset\mathcal B(G,F)$，存在 smooth affine group scheme
$$
\mathcal G_{\mathfrak f}/\mathcal O_F
$$
使得
$$
\mathcal G_{\mathfrak f}(\mathcal O_F)=G(F)_{\mathfrak f,0}.
$$
其特殊纤维的 reductive quotient 控制 parahoric Hecke algebra 的有限群部分。

## AA.2 Hyperspecial subgroups 和非分歧群

**定义 AA.4.** 群 $G/F$ 称为 unramified，若 $G$ 在某个非分歧扩张上 split，且存在 reductive group scheme
$$
\mathcal G/\mathcal O_F
$$
以 $G$ 为 generic fiber。此时
$$
K=\mathcal G(\mathcal O_F)
$$
称为 hyperspecial maximal compact subgroup。

**外部输入定理 AA.5（hyperspecial vertices）.** Hyperspecial maximal compact subgroups 与 building 中的 hyperspecial vertices 对应。它们存在当且仅当 $G$ 为 unramified。

**命题 AA.6.** 对 split $G=\operatorname{GL}_n$，$K=\operatorname{GL}_n(\mathcal O_F)$ 是 hyperspecial。

**证明.** 取 reductive group scheme
$$
\mathcal G=\operatorname{GL}_{n,\mathcal O_F}.
$$
其 generic fiber 为 $\operatorname{GL}_{n,F}$，特殊纤维为 $\operatorname{GL}_{n,k_F}$，仍 reductive。于是
$$
\mathcal G(\mathcal O_F)=\operatorname{GL}_n(\mathcal O_F)
$$
为 hyperspecial maximal compact subgroup。$\square$

## AA.3 Iwasawa、Cartan 和 Bruhat 分解

设 $G/F$ split，$B=TN$ 为 Borel subgroup，$K$ hyperspecial。

**外部输入定理 AA.7（Iwasawa decomposition）.** 有分解
$$
G(F)=B(F)K.
$$

**外部输入定理 AA.8（Cartan decomposition）.** 有分解
$$
G(F)=\bigsqcup_{\lambda\in X_*(T)^+}K\lambda(\varpi)K.
$$

**外部输入定理 AA.9（Iwahori-Bruhat decomposition）.** 若 $I$ 为 Iwahori subgroup，则
$$
G(F)=\bigsqcup_{w\in\widetilde W}IwI,
$$
其中 $\widetilde W$ 为 extended affine Weyl group。

**命题 AA.10.** Cartan 分解给出球 Hecke 代数的标准基。

**证明.** 球 Hecke 代数由 compactly supported bi-$K$-invariant functions 组成。Cartan 分解把双陪集空间 $K\backslash G(F)/K$ 识别为 $X_*(T)^+$。因此每个函数都是有限个双陪集特征函数
$$
\mathbf 1_{K\lambda(\varpi)K}
$$
的线性组合，这些特征函数构成标准基。$\square$

## AA.4 Moy-Prasad filtrations

**外部输入定理 AA.11（Moy-Prasad filtrations）.** 对 $x\in\mathcal B(G,F)$ 和 $r\ge0$，存在 filtration subgroups
$$
G(F)_{x,r}\supset G(F)_{x,r+}
$$
以及 Lie algebra filtrations。它们用于定义 depth of representations 和 refined local character expansions。

**定义 AA.12.** 不可约 smooth representation $\pi$ 的 depth 是最小 $r$，使得存在 $x$ 满足
$$
\pi^{G(F)_{x,r+}}\ne0.
$$

**命题 AA.13.** Spherical representations 的 depth 为 $0$。

**证明.** 若 $\pi$ spherical，则存在 hyperspecial $K=G(F)_{x,0}$ 使 $\pi^K\ne0$。由于 $G(F)_{x,0+}\subset K$，同一向量也被 $G(F)_{x,0+}$ 固定。因此 depth 至多为 $0$。Depth 按定义非负，故为 $0$。$\square$

## AA.5 Parahoric Hecke algebras

**定义 AA.14.** 对 parahoric subgroup $P\subset G(F)$，定义 parahoric Hecke algebra
$$
\mathcal H(G,P)=C_c(P\backslash G(F)/P).
$$
当 $P=I$ 为 Iwahori 时称 Iwahori-Hecke algebra。

**外部输入定理 AA.15（Iwahori-Hecke algebra presentation）.** Iwahori-Hecke algebra 由 affine Weyl group 的 generators 给出，满足 braid relations 和 quadratic relations
$$
(T_s+1)(T_s-q_s)=0
$$
以及 length-additive multiplication rules。

**命题 AA.16.** 球 Hecke 代数可由 Iwahori-Hecke 代数的 $K$-biinvariant corner 获得。

**证明草图.** 若 $I\subset K$，则 $e_K$ 可在 Iwahori-Hecke algebra 的合适 completion 或 finite sum setting 中作为 $K$-平均 idempotent 表示。Bi-$K$-invariant convolution algebra 等同于
$$
e_K\mathcal H(G,I)e_K.
$$
该 corner 的 Bernstein presentation 化为 Satake 同构。完整证明依赖 Iwahori decomposition 和 affine Hecke algebra theory。$\square$

## AA.6 与 L 群和非分歧 LLC 的接口

**命题 AA.17.** Hyperspecial subgroup 的选择是非分歧 Satake 参数的组成部分。

**证明.** 非分歧表示定义为存在 $K$-fixed vector，其中 $K$ 为 hyperspecial maximal compact subgroup。球 Hecke 代数 $\mathcal H(G,K)$ 依赖 $K$。Satake 同构把其 characters 识别为 $\widehat G\rtimes\operatorname{Fr}$ 中的半单共轭类。因此没有 hyperspecial subgroup，就没有该归一化下的 spherical Hecke eigencharacter，也不能定义相同的非分歧 Satake parameter。$\square$

**命题 AA.18.** 非分歧局部 Langlands 的基本模型依赖 Bruhat-Tits theory。

**证明.** 非分歧 LLC 的表示侧使用 hyperspecial $K$、Cartan decomposition 和 Satake isomorphism。定理 AA.5 给出 $K$ 的存在判准，定理 AA.8 给出球 Hecke 代数基，Satake 同构再把 Hecke characters 转为对偶群半单共轭类。这些均来自 Bruhat-Tits 建筑和相应 integral models。$\square$

## 练习

**练习 AA.1.** 对 $\operatorname{GL}_n$，说明 hyperspecial subgroup 的 integral model 来源。

**练习 AA.2.** 解释 parahoric 与 facet 的关系。

**练习 AA.3.** 用 Cartan decomposition 说明球 Hecke 代数有双陪集基。

**练习 AA.4.** 说明 spherical representation 的 depth 为 $0$。

**练习 AA.5.** 解释 hyperspecial 选择为何影响 Satake 参数归一化。
