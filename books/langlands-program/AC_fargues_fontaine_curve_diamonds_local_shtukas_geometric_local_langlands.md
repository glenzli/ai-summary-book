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

**外部输入定理 AC.4（Fargues-Fontaine curve；本附录的几何点假设）.** 设 $F$ 为本附录固定的剩余特征 $p$ 的非 Archimedean 局部域，取完备代数闭非 Archimedean 扩张 $C/F$。则 $C$ 为 perfectoid field，其 tilt $C^\flat$ 是含 $k_F$ 的完备代数闭 characteristic-$p$ perfectoid field。由 pair $(C^\flat,F)$ 的 period rings 可构造 regular noetherian one-dimensional scheme
$$
X_{C^\flat,F},
$$
称为 Fargues-Fontaine curve。选定的 untilt $C/F$ 还给出 local-shtuka 修改所用的 distinguished untilt divisor，但曲线本身只依赖 $(C^\flat,F)$。以下 AC.5 的 vector-bundle 分类、AC.7 的 $G$-bundle 分类及其后续调用均固定这个完备代数闭几何点，并把该曲线缩写为 $X_{FF}$；它不是由一个裸 $p$-adic field 唯一给出的无参数曲线。

**外部输入定理 AC.5（classification of vector bundles；完备代数闭情形）.** 在 AC.4 的假设下，对每个 $\lambda\in\mathbb Q$ 有 stable vector bundle $\mathcal O_{X_{FF}}(\lambda)$；$X_{FF}$ 上每个 vector bundle 都同构于唯一的有限直和
$$
\bigoplus_{\lambda\in\mathbb Q}
\mathcal O_{X_{FF}}(\lambda)^{\oplus m_\lambda},
$$
其中仅有限个 $m_\lambda$ 非零。特别地，slope 为 $\lambda$ 的 semistable vector bundle 同构于 $\mathcal O_{X_{FF}}(\lambda)^{\oplus m}$。这里分类的是 vector bundles 的同构类，不是说 isocrystals 与 vector bundles 两个范畴等价。

**注 AC.5.1（一般基底上的 descent 边界）.** 对一般 characteristic-$p$ perfectoid space $S$，可构造相对曲线 $X_{S,F}$，且 vector bundles 对 v-topology 满足 descent；然而 AC.5 只是在 $S=\operatorname{Spa}(C^\flat)$、$C^\flat$ 完备代数闭时的几何纤维分类。若起始 perfectoid field 未代数闭，只能先拉回到完备代数闭扩张，再连同 descent datum 下降；slope 多重集本身不分类原场上的 bundles。

**命题 AC.6.** 在 AC.4 的完备代数闭设定下，Fargues-Fontaine 曲线把 isocrystal 的 Newton slope 几何化为 vector bundle slope。

**证明.** 定理 AC.5 给出 vector bundles 的 slope decomposition；isocrystals 也有 Dieudonne-Manin slope decomposition。Fargues-Fontaine construction 建立二者之间的对应，使 isocrystal 中 Frobenius slope $\lambda$ 对应 $\mathcal O(\lambda)$ 型 vector bundle。因此 Newton polygon 成为曲线上向量丛的 Harder-Narasimhan polygon。$\square$

## AC.3 $G$-bundles on Fargues-Fontaine curve

设 $G/F$ 为 connected reductive group。

**外部输入定理 AC.7（classification of $G$-bundles；完备代数闭情形）.** 在 AC.4 的假设下，映射 $b\mapsto\mathcal E_b$ 给出双射
$$
B(G)\xrightarrow{\sim}
\left\{\text{$G$-bundles on $X_{FF}$ 的同构类}\right\},
$$
并以 Newton point 与 Kottwitz invariant 记录相应 Harder-Narasimhan 数据；semistable $G$-bundles 恰对应 basic elements。对一般 perfectoid 基底 $S$，$b\in B(G)$ 仍给出相对 bundle $\mathcal E_b$，但 $\operatorname{Bun}_G(S)$ 还包含 families 与 v-descent 数据，不能把整个 groupoid 直接等同于集合 $B(G)$。

**定义 AC.8.** 对 $b\in B(G)$，记相应 $G$-bundle 为
$$
\mathcal E_b.
$$
记 $J_b$ 为 $b$ 的 $\sigma$-centralizer。若 $b$ basic，则 $J_b$ 是 $G$ 的 inner form，且相应 semistable 几何纤维的 automorphism group 为
$$
J_b(F).
$$
若 $b$ 非 basic，则 $J_b$ 只与 Newton centralizer 的 inner form 对应，完整 automorphism object 也不应简写成一个 $G$ 的 inner form。

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
参数化相对 Fargues-Fontaine 曲线的 untilt divisor 处从 $\mathcal E_1$ 到 $\mathcal E_b$、相对位置由 $\mu$ 控制的 modifications；拉回任一 AC.4 型完备代数闭几何点后，$\mathcal E_b$ 由 AC.7 的 $b$ 唯一确定到同构。该 diamond 带有 $G(F)$ 与 $J_b(F)$ 的作用，并有 Weil group action。

**命题 AC.12.** Rapoport-Zink spaces 是 local Shimura varieties 的特殊情形。

**证明路线（外部输入）.** EL/PEL 型 Rapoport-Zink space 参数化 $p$-divisible groups with additional structure up to quasi-isogeny。通过 Scholze-Weinstein theory，$p$-divisible groups 对应 Fargues-Fontaine 曲线上的 modifications of vector bundles with tensors。固定 framing object 给出 $b$，Hodge filtration 给出 $\mu$，于是相应 moduli diamond 是 $\operatorname{Sht}_{G,b,\mu}$ 的实例。这里从模问题到 diamond 的可表性和两种模问题的同构均属外部输入，本段不构成证明。

## AC.5 Cohomology and local Langlands

**外部输入定理 AC.13（cohomology actions on local Shimura varieties）.** 取辅助素数 $\ell\ne p$。Compactly supported $\ell$-adic etale cohomology
$$
R\Gamma_c(\operatorname{Sht}_{G,b,\mu},\mathcal L_\xi)
$$
携带 commuting $G(F)\times J_b(F)\times W_F$ actions。在 EL/PEL 与若干 local Shimura data 中，其表示分解与已知 local Langlands/Jacquet-Langlands correspondences 有定理性相容；对一般 $G,b,\mu$ 的完整 packet realization 仍是研究问题。

**命题 AC.14.** 定理 AC.13 的 cohomology object 同时携带表示侧和 Weil 侧的作用；这只推出可定义 joint isotypic data，不单独推出完整 LLC。

**证明.** 定理 AC.11 给出 $G(F)$ 和 $J_b(F)$ 的几何作用，Weil descent datum 给出 $W_F$ 作用；函子性使三者在 cohomology 上交换。因此可对一个群的 isotypic component 保留另两个群的作用。把这些作用进一步识别为 expected packets 或完整参数需要 AC.13 所列的额外外部定理，不能由“同时作用”这一形式事实推出。$\square$

## AC.6 Fargues-Scholze geometrization

**外部输入定理 AC.15（Fargues-Scholze spectral action and semisimple parameter map）.** 设 $F$ 为剩余特征 $p$、剩余域 $\mathbb F_q$ 的非 Archimedean 局部域，$G/F$ 为 reductive group，$\ell\ne p$，并取含 $\sqrt q$ 的 algebraically closed $\ell$-adic coefficient field $L$。Fargues-Scholze 构造 derived stack of L-parameters
$$
\left[Z^1(W_F,\widehat G)_L/\widehat G\right]
$$
上的 perfect complexes 对 $D_{\mathrm{lis}}(\operatorname{Bun}_G,L)$ 的 spectral action。对每个 irreducible smooth $L$-representation
$\pi$ of $G(F)$，该作用给出唯一的 $\widehat G(L)$-共轭类
$$
\varphi_\pi^{\mathrm{ss}}:W_F\to\widehat G(L)\rtimes W_F
$$
of continuous semisimple L-parameters；这里到 $W_F$ 的投影是恒等映射，因此 $\varphi_\pi^{\mathrm{ss}}$ 是 L-parameter section，而不是任意同态。对 $G=\operatorname{GL}_n$，它等于经典 LLC 参数的 semisimplification；特别地，它不保留一般 Weil-Deligne monodromy $N$。

**注 AC.16.** 定理 AC.15 构造的是从表示到 semisimple 参数的映射，不是 fibres 的双射分类。它不自动给出 monodromy、component-group enhancements、Whittaker normalization、inner-form relevance 或 endoscopic character identities。其 sheaf coefficients 满足 $\ell\ne p$；它也不是 $p$-adic Banach representations 意义下的 $p$-adic Langlands correspondence。

**命题 AC.17.** Fargues-Scholze 框架是局部 Langlands 的几何化，而不是全局几何 Langlands 的直接替代。

**证明.** 全局几何 Langlands 的曲线是代数曲线 $X$，自动侧是 $\operatorname{Bun}_G(X)$ 上的 sheaves，谱侧是 $\widehat G$-local systems on $X$。Fargues-Scholze 的曲线是 Fargues-Fontaine curve，自动侧是其上 $G$-bundles 的 stack，谱侧是局部 Weil group 参数。二者结构相似，都有 spectral action，但输入域、曲线和参数群不同。因此它是局部 Langlands 的几何化。$\square$

## AC.7 与本书其他章节的接口

**命题 AC.18.** 在 AC.4 的完备代数闭几何点上，Fargues-Fontaine $G$-bundles 解释了局部 LLC 中 inner forms 和 $B(G)$ 的统一来源；相对版本通过 $\operatorname{Bun}_G$ 的 v-descent 组织这些几何纤维。

**证明.** 附录 N/X 中内形式通过 Kottwitz 或 rigid inner twist 数据出现。定理 AC.7 把 $B(G)$ 的元素解释为 Fargues-Fontaine 曲线上的 $G$-bundles，basic 元素的 automorphism group 为 inner form $J_b(F)$。因此局部表示、内形式和 Newton strata 可在同一几何对象 $\operatorname{Bun}_G(X_{FF})$ 中组织。$\square$

**外部输入推论 AC.19.** 在 AC.15 的设定下，若 $G$ unramified、$K$ hyperspecial 且 $\pi^K\ne0$，则
$\varphi_\pi^{\mathrm{ss}}$ 的非分歧 Frobenius 类与归一化 spherical Satake parameter 相容。

**证明路线（外部输入）.** Fargues-Scholze 参数与 excursion operators、parabolic induction 和 unramified Hecke action 的相容性质把球向量上的 spectral character 识别为 Satake character。该识别是 AC.15 来源定理性质的一部分，本书不从抽象 spectral action 重证。

## 练习

**练习 AC.1.** 在 AC.4 的完备代数闭设定下，说明 Fargues-Fontaine 曲线如何把 Newton slope 几何化。

**练习 AC.2.** 在 AC.4 的完备代数闭设定下，解释 $B(G)$ 与 $G$-bundles 同构类的关系，并说明一般 perfectoid 基底为何还需要 descent 数据。

**练习 AC.3.** 说明 local Shimura variety cohomology 为什么同时有 $G(F)$、$J_b(F)$ 和 $W_F$ 作用。

**练习 AC.4.** 解释 Fargues-Scholze 结果为什么主要给出 semisimple 参数化。

**练习 AC.5.** 比较全局几何 Langlands 与 Fargues-Fontaine 几何局部 Langlands 的曲线和谱侧。
