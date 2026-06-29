# 附录 I：Koszul、bar-cobar 与 twisting 的严格约定

## 本附录目标

第八章和第九章已经给出二次 operad、twisting morphism、bar 构造和 cobar 构造。本附录把其中容易混淆的技术约定集中列出：

1. reduced、augmented、coaugmented 和 conilpotent 的边界；
2. free operad 上的 derivation 与 cofree cooperad 上的 coderivation；
3. twisting morphism 的 Maurer-Cartan 方程；
4. twisted composite product 和 Koszul complex；
5. bar-cobar resolution 使用的权重滤过。

本附录仍不证明 operadic Koszul duality 的主判别定理；该定理保留为外部输入。

## I.1 Reduced 与增广约定

**约定 I.1.** 本附录固定域 $k$，链复形采用同调分次。所有 dg 对称序列默认取值于 $\mathbf{Ch}_k$。除非特别说明，operad 与 cooperad 均为 reduced：
$$
\mathcal P(0)=0,\qquad \mathcal P(1)\cong k\cdot\mathbf 1\oplus\overline{\mathcal P}(1),
$$
并且 cooperad 有 coaugmentation
$$
I_k\to\mathcal C
$$
及 coaugmentation coideal $\overline{\mathcal C}$。

**定义 I.2.** Augmented dg-operad 是 dg-operad $\mathcal P$ 连同 operad morphism
$$
\epsilon:\mathcal P\to I_k.
$$
Coaugmented dg-cooperad 是 dg-cooperad $\mathcal C$ 连同 cooperad morphism
$$
\eta:I_k\to\mathcal C.
$$
Reduced 语境中，twisting morphism 总是指在单位与余单位上为零的映射
$$
\overline{\mathcal C}\to\overline{\mathcal P}.
$$

**警告 I.3.** 第六章含单位的 $\operatorname{Ass}$、$\operatorname{Com}$ 与第八章 reduced 二次理论中的 $\operatorname{Ass}$、$\operatorname{Com}$ 不是同一个对象。Bar-cobar 和 Koszul 对偶默认处理 augmented/reduced 部分；含 arity $0$ 单位的非齐次关系需要额外 curved 或 unital 理论。

## I.2 Derivations 与 coderivations

**定义 I.4.** 设 $\mathcal P$ 是 dg-operad。一个次数 $r$ 的 derivation 是对称序列映射
$$
D:\mathcal P\to\mathcal P
$$
满足对每个分块复合
$$
\gamma_\pi:\mathcal P(\operatorname{Bl}(\pi))\otimes\bigotimes_B\mathcal P(B)\to\mathcal P(S)
$$
有 Leibniz 公式：
$$
D\gamma_\pi(x;(y_B)_B)
=
\gamma_\pi(Dx;(y_B)_B)
+\sum_B(-1)^{r(|x|+\sum_{B'<B}|y_{B'}|)}
\gamma_\pi(x;(y_{B'}')_{B'}),
$$
其中 $y_B'=Dy_B$，其他 $y_{B'}'=y_{B'}$。这里为了写符号选取了块集合的一个顺序；无坐标定义由对称幺半范畴中的 Leibniz rule 给出，任何两个顺序由 Koszul braiding 相容。

**命题 I.5.** 自由 operad $\mathbb F(M)$ 上的 derivation 由其在生成对称序列 $M$ 上的限制唯一决定。

**证明.** 自由 operad 的元素由 $M$-装饰树生成。若已知 $D$ 在每个顶点装饰上的值，则 Leibniz 公式强制
$$
D(T)=\sum_{v\in V(T)}\pm T\text{ with }D\text{ applied at }v.
$$
符号由 $D$ 穿过该顶点左侧张量因子的 Koszul rule 给出。因此最多有一个延拓。反过来，上式定义的映射与树 grafting 相容，因为对 grafting 后的顶点集合求和等于先在外树或内树中求和再 graft；故得到 derivation。$\square$

**定义 I.6.** 设 $\mathcal C$ 是 dg-cooperad。一个次数 $r$ 的 coderivation 是对称序列映射
$$
D:\mathcal C\to\mathcal C
$$
使得 cooperad 分解满足余 Leibniz 公式：
$$
\Delta D=(D\circ\operatorname{id}+\operatorname{id}\circ D)\Delta,
$$
其中第二项包含把 $D$ 穿过张量因子的 Koszul 符号。

**命题 I.7.** Cofree conilpotent cooperad $\mathbb T^c(M)$ 上的 coderivation 由复合
$$
\mathbb T^c(M)\xrightarrow{D}\mathbb T^c(M)\to M
$$
唯一决定，其中最后映射投影到单顶点树部分。

**证明.** Coderivation 的余 Leibniz 公式说明 $D$ 在多顶点树上的值由对每个连通子树收缩到单顶点部分的值决定。Conilpotence 保证递归过程有限。反向构造：给定 $f:\mathbb T^c(M)\to M$，对每棵树 $T$ 和每个连通子树 $U\subseteq T$，用 $f$ 替换 $U$ 为一个新顶点，其他顶点装饰保持不变，并对所有 $U$ 求和。该公式满足余 Leibniz 公式。唯一性由投影到单顶点部分后按顶点数归纳得到。$\square$

## I.3 Quasi-free 与 quasi-cofree 对象

**定义 I.8.** 一个 quasi-free dg-operad 是形如
$$
(\mathbb F(M),d=d_{\mathrm{int}}+\partial)
$$
的 dg-operad，其中 $d_{\mathrm{int}}$ 由 $M$ 的内部微分诱导，$\partial$ 是次数 $-1$ 的 derivation，并且 $\partial(M)$ 落在至少两个顶点的树部分。

**定义 I.9.** 一个 quasi-cofree dg-cooperad 是形如
$$
(\mathbb T^c(M),d=d_{\mathrm{int}}+\partial)
$$
的 dg-cooperad，其中 $\partial$ 是次数 $-1$ 的 coderivation，并且其投影到单顶点部分为零，或者等价地，$\partial$ 降低适当的顶点数滤过。

**命题 I.10.** Cobar 构造 $\Omega\mathcal C$ 是 quasi-free dg-operad；bar 构造 $B\mathcal P$ 是 quasi-cofree dg-cooperad。

**证明.** $\Omega\mathcal C$ 的底层 graded operad 是 $\mathbb F(s^{-1}\overline{\mathcal C})$。其二次微分由 infinitesimal decomposition 定义，并作为 derivation 延拓到自由 operad，所以是 quasi-free。$B\mathcal P$ 的底层 graded cooperad 是 $\mathbb T^c(s\overline{\mathcal P})$。其二次微分由收缩一条内部边给出，是由投影到单顶点部分决定的 coderivation，所以是 quasi-cofree。$\square$

## I.4 Convolution Lie algebra 与 Maurer-Cartan 方程

**定义 I.11.** 对 coaugmented dg-cooperad $\mathcal C$ 与 augmented dg-operad $\mathcal P$，定义 convolution complex
$$
\operatorname{Conv}(\mathcal C,\mathcal P)
=
\operatorname{Hom}_{\mathbb S}(\overline{\mathcal C},\overline{\mathcal P}).
$$
其微分为
$$
\partial f=d_{\mathcal P}f-(-1)^{|f|}fd_{\mathcal C}.
$$
其 pre-Lie product 为
$$
f\star g=\gamma_{(1)}(f\circ_{(1)}g)\Delta_{(1)}.
$$
Bracket 定义为
$$
[f,g]=f\star g-(-1)^{|f||g|}g\star f.
$$

**命题 I.12.** $\operatorname{Conv}(\mathcal C,\mathcal P)$ 是 dg Lie algebra。

**证明.** 微分平方为零来自 Hom complex 的标准计算。$\partial$ 对 $\star$ 满足 Leibniz rule，因为 $\gamma_{(1)}$ 和 $\Delta_{(1)}$ 是链映射。Pre-Lie 恒等式来自两次 infinitesimal 代入的嵌套项与分离项分类；嵌套项由 operad 结合律和 cooperad 余结合律匹配，分离项在交换两次代入后只差 Koszul 符号。Pre-Lie bracket 满足 graded Jacobi 恒等式。$\square$

**定义 I.13.** Twisting morphism 是次数 $-1$ 元素
$$
\alpha\in\operatorname{Conv}(\mathcal C,\mathcal P)_{-1}
$$
满足 Maurer-Cartan 方程
$$
\partial\alpha+\alpha\star\alpha=0.
$$

**命题 I.14.** 对次数 $-1$ 的 $\alpha$，Maurer-Cartan 方程等价于
$$
\partial\alpha+\frac12[\alpha,\alpha]=0
$$
在 $\operatorname{char}k\ne2$ 时的 dg Lie Maurer-Cartan 方程。

**证明.** 因为 $|\alpha|=-1$，
$$
[\alpha,\alpha]=\alpha\star\alpha-(-1)^{1}\alpha\star\alpha=2\alpha\star\alpha.
$$
若 $2$ 可逆，则两式等价。为避免特征限制，本书用 $\partial\alpha+\alpha\star\alpha=0$ 作为定义。$\square$

## I.5 Twisted composite product

**定义 I.15.** 设 $\alpha:\mathcal C\to\mathcal P$ 是 twisting morphism。右 twisted composite product
$$
\mathcal C\circ_\alpha\mathcal P
$$
的底层对称序列为 $\mathcal C\circ\mathcal P$，微分为
$$
d=d_{\mathcal C\circ\mathcal P}+d_\alpha,
$$
其中 $d_\alpha$ 先用 $\Delta_{(1)}$ 把 $\mathcal C$ infinitesimally 分解，再用 $\alpha$ 把内层 cooperad 因子送入 $\mathcal P$，最后用 $\mathcal P$ 的 operad 乘法并入右侧 $\mathcal P$ 因子。

左 twisted composite product
$$
\mathcal P\circ_\alpha\mathcal C
$$
类似定义。

**命题 I.16.** 若 $\alpha$ 满足 Maurer-Cartan 方程，则 $\mathcal C\circ_\alpha\mathcal P$ 和 $\mathcal P\circ_\alpha\mathcal C$ 的微分平方为零。

**证明.** 展开 $(d_{\mathrm{int}}+d_\alpha)^2$。内部微分平方为零。交叉项为 $\partial\alpha$ 诱导的算子。$d_\alpha^2$ 为 $\alpha\star\alpha$ 诱导的算子。Maurer-Cartan 方程说明二者相加为零。左版本同理。$\square$

## I.6 Koszul complex

**定义 I.17.** 设 $\mathcal P$ 是二次 operad，$\mathcal P^¡$ 是其二次对偶 cooperad，$\kappa:\mathcal P^¡\to\mathcal P$ 是 Koszul twisting morphism。右 Koszul complex 定义为
$$
K_r(\mathcal P)=\mathcal P^¡\circ_\kappa\mathcal P.
$$
左 Koszul complex 定义为
$$
K_l(\mathcal P)=\mathcal P\circ_\kappa\mathcal P^¡.
$$
若需要双侧版本，则写作
$$
\mathcal P\circ_\kappa\mathcal P^¡\circ_\kappa\mathcal P.
$$

**定义 I.18.** $\mathcal P$ 称为 Koszul，若 $K_r(\mathcal P)\to I$ 是 quasi-isomorphism；在常用有限型 reduced 假设下，这与左 Koszul complex 的相应条件等价。

**外部输入定理 I.19.** 对有限型 reduced 二次 operad，以下条件等价：

1. $\mathcal P$ 是 Koszul；
2. $\Omega\mathcal P^¡\to\mathcal P$ 是 quasi-isomorphism；
3. $\mathcal P^¡\to B\mathcal P$ 是 quasi-isomorphism；
4. 相应 bar-cobar weight spectral sequence 在期望页退化并给出单位同调。

该定理的 Ginzburg--Kapranov classical core 已定位为 GK-1--GK-7：Definition 4.1.3、Proposition 4.1.4、Theorem 4.1.13、Theorem 4.2.5、Corollary 4.2.7、Theorem 3.2.16 和 Section 4.2.12。Fresse modern cobar/twisted-composite/cofibrant replacement 已定位为 FRE-1--FRE-6。现代 $\Omega\mathcal P^¡\to\mathcal P$、$\Omega B\mathcal P\to\mathcal P$ 的书本 convention 写法已由附录 D 和 `FINAL_OPERAD_THEORY_CLOSURE.md` 关闭为 convention/bibliography production work。

## I.7 权重滤过与谱序列边界

**定义 I.20.** 对由生成对称序列 $M$ 构成的自由 operad $\mathbb F(M)$，权重为树的顶点数。定义递增滤过
$$
F_r\mathbb F(M)=\bigoplus_{0\le q\le r}\mathbb F^{(q)}(M).
$$
对 cofree cooperad $\mathbb T^c(M)$ 也按顶点数定义滤过。

**命题 I.21.** Cobar differential $d=d_1+d_2$ 中，$d_1$ 保持权重，$d_2$ 增加权重 $1$。Bar differential 中，$d_1$ 保持权重，$d_2$ 降低权重 $1$。

**证明.** Cobar 中 $d_1$ 逐生成元应用内部微分，不改变树顶点数；$d_2$ 把一个 cooperad 生成元替换为两个生成元的树形复合，所以顶点数增加 $1$。Bar 中 $d_1$ 逐顶点作用，不改变顶点数；$d_2$ 收缩一条内部边，把两个顶点复合为一个顶点，所以顶点数降低 $1$。$\square$

**说明 I.22.** Koszul 判别的谱序列证明正是利用这种权重行为。第零页通常由内部微分控制，下一页由二次关系或 Koszul differential 控制。由于不同文献对 homological/cohomological grading、suspension 和权重滤过方向的约定不同，正文引用谱序列定理时必须说明采用哪一种 convention。

## I.8 与同伦 operad 的关系

**定义 I.23.** 若 $\mathcal P$ 是 Koszul operad，本书写
$$
\mathcal P_\infty=\Omega\mathcal P^¡.
$$
该定义依赖 $\mathcal P^¡$ 的 cooperad 结构和本附录的 cobar differential。

**命题 I.24.** 若 $\mathcal P$ Koszul，则存在 quasi-isomorphism
$$
\mathcal P_\infty=\Omega\mathcal P^¡\longrightarrow\mathcal P.
$$

**证明.** 这是外部输入定理 I.19 的第 2 条。$\square$

**警告 I.25.** 记号 $\mathcal P_\infty$ 不等于“任意 weakly equivalent operad”。它在本书中特指 Koszul dual cooperad 的 cobar resolution。若使用 Boardman-Vogt resolution 或 cofibrant replacement $Q\mathcal P$，必须另行说明。

## I.9 本附录小结

Bar-cobar 理论的核心可压缩为一条链：
$$
\text{conilpotent cooperad }\mathcal C
\xrightarrow{\alpha}
\text{augmented operad }\mathcal P
\quad\Longleftrightarrow\quad
\partial\alpha+\alpha\star\alpha=0.
$$
该方程保证 twisted composite products 有微分。Koszul 性则断言由典范 twisting morphism $\kappa:\mathcal P^¡\to\mathcal P$ 构造的 Koszul complex 解析单位，并等价于 $\Omega\mathcal P^¡\to\mathcal P$ 是 quasi-isomorphism。所有这些陈述都依赖 reduced、conilpotent、suspension 和权重滤过约定；这些约定已在本附录固定。
