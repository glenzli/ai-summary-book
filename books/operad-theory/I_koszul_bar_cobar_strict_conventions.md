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

**约定 I.1.** 本附录固定域 $k$，链复形采用同调分次。所有 dg 对称序列默认取值于 $\mathbf{Ch}_k$。除非特别说明，operad 均 augmented 且无 nullary operations：
$$
\mathcal P(0)=0,\qquad \mathcal P(1)\cong k\cdot\mathbf 1\oplus\overline{\mathcal P}(1),
$$
并且 cooperad 有 coaugmentation
$$
I_k\to\mathcal C
$$
及 coaugmentation coideal $\overline{\mathcal C}$，同时 $\mathcal C(0)=0$。

本附录称 $\mathcal P$ **connected**，若
$$
\overline{\mathcal P}(0)=\overline{\mathcal P}(1)=0,
$$
并对 cooperad 作同样约定。第八章的二次 operads 是 connected。允许 $\overline{\mathcal P}(1)\ne0$ 的一般 bar-cobar 伴随与 connected Koszul 判别必须分开；后者的逐 arity 顶点滤过有限性会用到 connectedness。

**定义 I.1.1（conilpotence）.** 对 coaugmented cooperad $\mathcal C$，把分解投影到 coaugmentation coideal 得到 reduced decomposition。若对每个 $c\in\overline{\mathcal C}$，存在 $N(c)$，使所有具有多于 $N(c)$ 个顶点的迭代 reduced decompositions 在 $c$ 上为零，则称 $\mathcal C$ conilpotent。该条件是逐元素有限性，不表示 $\mathcal C$ 在每个 arity 有统一 nilpotence 次数。

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
单位/增广与 coaugmentation/余单位分别给出分裂
$$
\mathcal P\cong I_k\oplus\overline{\mathcal P},
\qquad
\mathcal C\cong I_k\oplus\overline{\mathcal C}.
$$
本附录使用定义 9.11 由这些分裂得到的 $\Delta_{(1)}$ 和 $\overline\Delta_{(1)}$。

**警告 I.3.** 第六章含单位的 $\operatorname{Ass}$、$\operatorname{Com}$ 与第八章 reduced 二次理论中的 $\operatorname{Ass}$、$\operatorname{Com}$ 不是同一个对象。Bar-cobar 和 Koszul 对偶默认处理 augmented/reduced 部分；含 arity $0$ 单位的非齐次关系需要额外 curved 或 unital 理论。

## I.2 Derivations 与 coderivations

**定义 I.4.** 设 $\mathcal P$ 是 dg-operad。一个次数 $r$ 的 derivation 是对称序列映射
$$
D:\mathcal P\to\mathcal P
$$
满足对每个有限集映射 $f:S\to T$ 的复合
$$
\gamma_f:\mathcal P(T)\otimes\bigotimes_{t\in T}\mathcal P(f^{-1}(t))\to\mathcal P(S)
$$
有 Leibniz 公式：
$$
D\gamma_f(x;(y_t)_t)
=
\gamma_f(Dx;(y_t)_t)
+\sum_t(-1)^{r(|x|+\sum_{t'<t}|y_{t'}|)}
\gamma_f(x;(y_{t'}')_{t'}),
$$
其中第 $t$ 项取 $y_t'=Dy_t$，其他 $y_{t'}'=y_{t'}$。这里为了写符号选取了 $T$ 的一个顺序；无坐标定义由对称幺半范畴中的 Leibniz rule 给出，任何两个顺序由 Koszul braiding 相容。空纤维没有例外，其因子就是 $\mathcal P(0)$。

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
的 dg-cooperad，其中 $\partial$ 是次数 $-1$ 的 coderivation，并且它在 cogenerators 上没有线性部分：
$$
M\hookrightarrow\mathbb T^c(M)
\xrightarrow{\partial}\mathbb T^c(M)
\twoheadrightarrow M
=0.
$$
注意整个复合 $\mathbb T^c(M)\xrightarrow{\partial}\mathbb T^c(M)\twoheadrightarrow M$ 一般不为零；bar differential 在二顶点树上的 corestriction 正是 operad composition。对 bar 构造，$\partial$ 每次收缩一条内边，故将顶点数恰降低 $1$。

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
f\star g
=
\pi_{\mathcal P}\gamma_{(1)}
(\widetilde f\circ_{(1)}\widetilde g)
\Delta_{(1)}\iota_{\mathcal C},
$$
其中 $\widetilde f=\iota_{\mathcal P}f\pi_{\mathcal C}$、$\widetilde g=\iota_{\mathcal P}g\pi_{\mathcal C}$ 是在 coaugmentation 因子上取零的延拓。等价地，可把中间的 $\Delta_{(1)}$ 换成
$$
\overline\Delta_{(1)}:
\overline{\mathcal C}\to
\overline{\mathcal C}\circ_{(1)}\overline{\mathcal C}
$$
并省略显式包含与投影。这样公式的定义域和值域均为 $\operatorname{Hom}_{\mathbb S}(\overline{\mathcal C},\overline{\mathcal P})$。
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

**定义 I.18.** 设 $\mathcal P$ 是第八章意义下的 connected、weight-graded 二次 operad。称 $\mathcal P$ Koszul，若增广
$$
K_r(\mathcal P)=\mathcal P^¡\circ_\kappa\mathcal P\longrightarrow I
$$
是 quasi-isomorphism。左 Koszul complex、bar inclusion 和 cobar resolution 的等价性不是本定义的一部分，而是下一条外部输入。

**外部输入定理 I.19（quadratic Koszul criterion；LV-1、LV-2）.** 采用 Loday--Vallette *Algebraic Operads* 的 characteristic-$0$ symmetric-operad 语境。设 $\mathcal P=\mathcal P(E,R)$ 是 connected、weight-graded 二次 dg-operad，微分保持权重；令 $\mathcal P^¡=\mathcal C(sE,s^2R)$，并令 $\kappa:\mathcal P^¡\to\mathcal P$ 为保持权重的典范 twisting morphism。则以下条件等价：

1. $K_r(\mathcal P)\to I$ 是 quasi-isomorphism；
2. $K_l(\mathcal P)\to I$ 是 quasi-isomorphism；
3. 典范 twisting morphism 诱导的 $\Omega\mathcal P^¡\to\mathcal P$ 是 quasi-isomorphism；
4. $\mathcal P^¡\to B\mathcal P$ 是 quasi-isomorphism。

**证明路线（外部输入）.** LV-1，即 Loday--Vallette Theorem 6.6.2，对任意 connected weight-graded dg cooperad $\mathcal C$、operad $\mathcal P$ 和保持权重的 twisting morphism $\alpha:\mathcal C\to\mathcal P$，把左右 twisted composites 的 acyclicity、$\mathcal C\to B\mathcal P$ 与 $\Omega\mathcal C\to\mathcal P$ 的 quasi-isomorphism 证明为四个等价条件。LV-2，即 Theorem 7.4.6，把该定理特化到 $\mathcal C=\mathcal P^¡$ 和 $\alpha=\kappa$，正得到上列四项。GK-3/GK-7 是 classical cross-check；FRE-2--FRE-4 是带 $C$-cofibrancy 等模型假设的另一来源包。谱序列是证明工具，不是无条件的第五个等价命题。

## I.7 权重滤过与谱序列边界

**定义 I.20.** 对由生成对称序列 $M$ 构成的自由 operad 和 cofree conilpotent cooperad，记恰有 $q$ 个顶点的部分为
$$
\mathbb F^{(q)}(M),\qquad \mathbb T^{c,(q)}(M).
$$
Bar 构造使用递增滤过
$$
F_rB\mathcal P
=
\bigoplus_{0\le q\le r}\mathbb T^{c,(q)}(s\overline{\mathcal P}).
$$
Cobar 构造使用递减滤过
$$
F^r\Omega\mathcal C
=
\bigoplus_{q\ge r}\mathbb F^{(q)}(s^{-1}\overline{\mathcal C}).
$$
这两个方向不同；不能用同一个递增顶点滤过同时处理 bar 与 cobar。

**命题 I.21.** Cobar differential $d=d_1+d_2$ 中，$d_1$ 保持权重，$d_2$ 增加权重 $1$，因而
$$
d(F^r\Omega\mathcal C)\subseteq F^r\Omega\mathcal C.
$$
Bar differential 中，$d_1$ 保持权重，$d_2$ 降低权重 $1$，因而
$$
d(F_rB\mathcal P)\subseteq F_rB\mathcal P.
$$
若 $\overline{\mathcal C}(0)=\overline{\mathcal C}(1)=0$，则 $F^\bullet\Omega\mathcal C(n)$ 对每个固定 $n$ 是有限递减滤过；bar 的 connected 情形同理。

**证明.** Cobar 中 $d_1$ 逐生成元应用内部微分，不改变树顶点数；$d_2$ 把一个 cooperad 生成元替换为两个生成元的树形复合，所以顶点数增加 $1$。故 $d_2(F^r)\subseteq F^{r+1}\subseteq F^r$。Bar 中 $d_1$ 逐顶点作用，不改变顶点数；$d_2$ 收缩一条内部边，把两个顶点复合为一个顶点，所以顶点数降低 $1$，故保持 $F_r$。

若所有顶点的输入数至少为 $2$，具有 $n$ 个叶、$q$ 个顶点的有根树满足
$$
n-1=\sum_{v\in V(T)}(\operatorname{in}(v)-1)\ge q.
$$
因此固定 arity $n$ 时只有 $0\le q\le n-1$。Connectedness 排除了 nullary 与非单位 unary 顶点，故两个滤过逐 arity 有限。$\square$

**说明 I.22（收敛口径）.** 在定义 I.20 的两个滤过上，associated graded differential 先由保持顶点数的 $d_1$ 给出，改变顶点数的 $d_2$ 出现在后续 differential。若对象 connected，则滤过逐 arity 有限，故相应谱序列在每个 arity 强收敛到该 filtered complex 的同调。若允许非平凡 unary 部分，逐 arity 有限性消失；此时必须另外验证 exhaustive、separated、complete 和 convergence 条件，或改用完成化构造。

**反例 I.22.1（direct sum 不等于完成化）.** 令 $M(1)=k$ 集中在次数 $0$，其他 arity 为 $0$。Arity $1$ 的自由 operad包含每个长度 $q\ge0$ 的 unary 线性树，因此
$$
\mathbb F(M)(1)\cong\bigoplus_{q\ge0}k.
$$
对递减顶点滤过完成化得到
$$
\widehat{\mathbb F(M)(1)}
=
\varprojlim_r\mathbb F(M)(1)/F^r
\cong
\prod_{q\ge0}k.
$$
序列 $(1,1,1,\ldots)$ 属于右侧而不属于左侧。故在有 unary generators 时，把形式无穷树和直接和 bar-cobar 对象混用会改变对象；conilpotence 本身不能把这个乘积自动缩回直和。

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
