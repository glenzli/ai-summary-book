# 附录 AC：Fargues-Fontaine 曲线、Diamonds、Local Shtukas 和几何局部 Langlands

**收口归一化回指。** 本附录涉及 Weil group、Frobenius、局部 shtukas、cohomological shifts 和局部 Langlands 参数；与 Galois/Satake/几何归一化比较时使用 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 第 1、4、5、7、9 节。

## AC.1 Perfectoid fields and diamonds

设 $F$ 为 $p$-adic local field。

**外部输入定理 AC.1（perfectoid spaces and diamonds）.** Perfectoid spaces 构成一类适合 $p$-adic 几何的空间。Diamonds 是 perfectoid spaces 的 pro-etale sheaves quotient；在 Scholze 的 diamond/v-stack finiteness hypotheses 下，它们支持 etale cohomology、six functors 和 moduli of local shtukas。

**定义 AC.2.** 若 $X$ 为 adic space，其 diamond 记为
$$
X^\diamond.
$$
Diamonds 允许把非离散 valuation 和 untilts 的 moduli 纳入几何对象。

**注 AC.3.** 本书不发展 perfectoid geometry。附录 AC 只记录它在局部 Langlands 几何化中的接口。

## AC.2 Fargues-Fontaine 曲线

**外部输入定理 AC.4（Fargues-Fontaine curve）.** 对 perfectoid field $C$ of characteristic $p$ 及 untilt 数据，可构造一条 regular noetherian-like curve
$$
X_{FF}
$$
其 vector bundles 分类与 isocrystals/Newton slopes 等价。

**外部输入定理 AC.5（classification of vector bundles）.** Fargues-Fontaine 曲线上的 vector bundles 按 slope decomposition 分类。对每个 rational slope $\lambda$ 有 semistable bundle $\mathcal O(\lambda)$，任意 vector bundle 分解为这些基本块的直和。

**命题 AC.6.** Fargues-Fontaine 曲线把 isocrystal 的 Newton slope 几何化为 vector bundle slope。

**证明.** 定理 AC.5 给出 vector bundles 的 slope decomposition；isocrystals 也有 Dieudonne-Manin slope decomposition。Fargues-Fontaine construction 建立二者之间的对应，使 isocrystal 中 Frobenius slope $\lambda$ 对应 $\mathcal O(\lambda)$ 型 vector bundle。因此 Newton polygon 成为曲线上向量丛的 Harder-Narasimhan polygon。$\square$

## AC.3 $G$-bundles on Fargues-Fontaine curve

设 $G/F$ 为 connected reductive group。

**外部输入定理 AC.7（classification of $G$-bundles）.** $G$-bundles on the Fargues-Fontaine curve are classified by Kottwitz set
$$
B(G),
$$
with Newton point and Kottwitz invariant. Semistable $G$-bundles correspond to basic elements.

**定义 AC.8.** 对 $b\in B(G)$，记相应 $G$-bundle 为
$$
\mathcal E_b.
$$
其 automorphism group 为 inner form
$$
J_b(F).
$$

**命题 AC.9.** Basic elements 给出内形式。

**证明.** 若 $b$ basic，则其 Newton cocharacter central modulo center。对应 $G$-bundle $\mathcal E_b$ semistable，automorphism group $J_b$ 是 $G$ 的 inner form。该 construction 与 isocrystal centralizer 的定义相同，故 $J_b(F)$ 是局部 Shimura varieties 中出现的 group of self-quasi-isogenies。$\square$

## AC.4 Local shtukas and local Shimura varieties

**定义 AC.10.** Local Shimura datum 是三元组
$$
(G,b,\mu)
$$
其中 $b\in B(G)$，$\mu$ 为 conjugacy class of cocharacters，满足 acceptability conditions。

**外部输入定理 AC.11（local Shimura varieties as diamonds）.** 对 local Shimura datum $(G,b,\mu)$，存在 diamond
$$
\operatorname{Sht}_{G,b,\mu}
$$
参数化 Fargues-Fontaine 曲线上从 $\mathcal E_1$ 到 $\mathcal E_b$、相对位置由 $\mu$ 控制的 modifications。它带有 $G(F)$ 与 $J_b(F)$ 的作用，并有 Weil group action。

**命题 AC.12.** Rapoport-Zink spaces 是 local Shimura varieties 的特殊情形。

**证明草图.** EL/PEL 型 Rapoport-Zink space 参数化 $p$-divisible groups with additional structure up to quasi-isogeny。通过 Scholze-Weinstein theory，$p$-divisible groups 对应 Fargues-Fontaine 曲线上的 modifications of vector bundles with tensors。固定 framing object 给出 $b$，Hodge filtration 给出 $\mu$，于是相应 moduli diamond 是 $\operatorname{Sht}_{G,b,\mu}$ 的实例。$\square$

## AC.5 Cohomology and local Langlands

**外部输入定理 AC.13（cohomology of local Shimura varieties）.** Compactly supported etale cohomology
$$
R\Gamma_c(\operatorname{Sht}_{G,b,\mu},\mathcal L_\xi)
$$
携带 $G(F)\times J_b(F)\times W_F$ 的作用，并预期实现 local Langlands 和 Jacquet-Langlands correspondences 的几何结构。

**命题 AC.14.** Local Shimura cohomology 同时看见表示侧和 Galois 侧。

**证明.** 定理 AC.11 给出 $G(F)$ 和 $J_b(F)$ 的几何作用，Weil descent datum 给出 $W_F$ 作用。对 cohomology 取表示分解，$G(F)$ 与 $J_b(F)$ 分量给出局部表示，$W_F$ 分量给出 Galois/Weil 参数信息。因此同一 cohomology object 同时包含 LLC 的两侧。$\square$

## AC.6 Fargues-Scholze geometrization

**外部输入定理 AC.15（Fargues-Scholze spectral action and semisimple LLC）.** 对 $p$-adic group $G(F)$，Fargues-Scholze 构造 stack of Langlands parameters
$$
\operatorname{LocSys}_{{}^LG}
$$
及其对 sheaves on $\operatorname{Bun}_G$ over Fargues-Fontaine curve 的 spectral action，并从中得到 semisimple local Langlands parameterization for irreducible smooth representations in broad generality.

**注 AC.16.** 该结果给出 semisimple 参数化和几何框架，但不自动给出所有增强 packet、endoscopic character identities 或完整 expected LLC 的每个 refined statement。

**命题 AC.17.** Fargues-Scholze 框架是局部 Langlands 的几何化，而不是全局几何 Langlands 的直接替代。

**证明.** 全局几何 Langlands 的曲线是代数曲线 $X$，自动侧是 $\operatorname{Bun}_G(X)$ 上的 sheaves，谱侧是 $\widehat G$-local systems on $X$。Fargues-Scholze 的曲线是 Fargues-Fontaine curve，自动侧是其上 $G$-bundles 的 stack，谱侧是局部 Weil group 参数。二者结构相似，都有 spectral action，但输入域、曲线和参数群不同。因此它是局部 Langlands 的几何化。$\square$

## AC.7 与本书其他章节的接口

**命题 AC.18.** Fargues-Fontaine $G$-bundles 解释了局部 LLC 中 inner forms 和 $B(G)$ 的统一来源。

**证明.** 附录 N/X 中内形式通过 Kottwitz 或 rigid inner twist 数据出现。定理 AC.7 把 $B(G)$ 的元素解释为 Fargues-Fontaine 曲线上的 $G$-bundles，basic 元素的 automorphism group 为 inner form $J_b(F)$。因此局部表示、内形式和 Newton strata 可在同一几何对象 $\operatorname{Bun}_G(X_{FF})$ 中组织。$\square$

**命题 AC.19.** 非分歧 Satake 参数是几何局部 Langlands 的最简单影子。

**证明草图.** 当 $G$ unramified 且表示 spherical 时，局部 LLC 参数由 hyperspecial Hecke eigencharacter 给出，即半单 Frobenius conjugacy class。Fargues-Scholze 的 spectral action 在这类最简单对象上应恢复同一 semisimple parameter。一般 local shtuka cohomology 和 sheaves on $\operatorname{Bun}_G$ 则把 ramified、inner form 和 higher-depth 情形纳入。$\square$

## 练习

**练习 AC.1.** 说明 Fargues-Fontaine 曲线如何把 Newton slope 几何化。

**练习 AC.2.** 解释 $B(G)$ 与 $G$-bundles 的关系。

**练习 AC.3.** 说明 local Shimura variety cohomology 为什么同时有 $G(F)$、$J_b(F)$ 和 $W_F$ 作用。

**练习 AC.4.** 解释 Fargues-Scholze 结果为什么主要给出 semisimple 参数化。

**练习 AC.5.** 比较全局几何 Langlands 与 Fargues-Fontaine 几何局部 Langlands 的曲线和谱侧。
