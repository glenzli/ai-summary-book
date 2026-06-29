# 附录 H：树约定、叶标号与自由 operad 的群胚商

## 本附录目标

正文中出现三类树：

1. 第三章使用的平面有根树，用于非对称 operad 和偏复合。
2. 第四章使用的装饰树，用于自由对称 operad。
3. 第十六章使用的 Moerdijk-Weiss rooted trees，用于 dendroidal sets。

这三类树不能混用。本附录固定它们之间的转换，并给出自由对称 operad 的群胚商公式。

## H.1 平面有根树

**定义 H.1.** 平面有根树是有限有根树 $T$，并且每个内部顶点 $v$ 的输入边集合 $\operatorname{In}(v)$ 配有一个全序。叶边集合记为 $\operatorname{Leaf}(T)$，根边记为 $r_T$，内部顶点集合记为 $V(T)$。

**说明 H.2.** 平面结构不是叶标号。平面结构只规定每个顶点的输入槽顺序；叶标号是额外指定一个双射
$$
\ell:S\to\operatorname{Leaf}(T)
$$
或 $\operatorname{Leaf}(T)\to S$。非对称 operad 通常只需要平面结构，不需要对称群重标号。

**命题 H.3.** 平面树代入严格满足结合律。

**证明.** 这就是命题 B.13。代入的两种顺序给出同一个顶点集合、同一个 incidence relation 和同一个逐层继承的平面顺序。$\square$

## H.2 非平面叶标号树

**定义 H.4.** 设 $S$ 是有限集。一个 $S$-叶标号有根树是有根非平面树 $T$ 连同双射
$$
\ell:S\to\operatorname{Leaf}(T).
$$
其同构是保持根边并与叶标号相容的树同构。

记由 $S$-叶标号有根树构成的群胚为
$$
\mathbf{Tree}_S.
$$

**定义 H.5.** 若 $T\in\mathbf{Tree}_S$，则一个 $E$-装饰是对每个顶点 $v\in V(T)$ 选取元素
$$
e_v\in E(\operatorname{In}(v)),
$$
其中 $E$ 是集合值对称序列。所有装饰组成集合
$$
\operatorname{Dec}_E(T)=\prod_{v\in V(T)}E(\operatorname{In}(v)).
$$

树同构 $\phi:T\to T'$ 诱导输入边双射
$$
\operatorname{In}(v)\to\operatorname{In}(\phi v)
$$
并由 $E$ 的函子性给出
$$
\operatorname{Dec}_E(T)\to\operatorname{Dec}_E(T').
$$
因此 $\operatorname{Dec}_E$ 是 $\mathbf{Tree}_S$ 上的函子。

**定义 H.6.** 自由对称 operad 的树群胚公式定义为
$$
\mathbb F(E)(S)
=
\int^{T\in\mathbf{Tree}_S}\operatorname{Dec}_E(T).
$$
等价地，
$$
\mathbb F(E)(S)
\cong
\coprod_{[T]\in\pi_0\mathbf{Tree}_S}
\operatorname{Dec}_E(T)_{\operatorname{Aut}_{\mathbf{Tree}_S}(T)}.
$$

**命题 H.7.** 公式 H.6 与第四章的装饰树自由 operad 公式一致。

**证明.** 第四章的公式把同构的叶标号装饰树相识别。群胚 coend 正是执行这件事：先对每个 $S$-叶标号树取装饰集合，再对树同构生成的关系取商。选择每个同构类代表 $T$ 后，同一同构类内部的商就是自同构群的 coinvariants
$$
\operatorname{Dec}_E(T)_{\operatorname{Aut}(T)}.
$$
因此 H.6 是第四章装饰树商公式的无坐标表达。$\square$

## H.3 自由 operad 复合

**定义 H.8.** 设 $T\in\mathbf{Tree}_S$，并对每个叶 $s\in S$ 给定 $T_s\in\mathbf{Tree}_{S_s}$。把 $T_s$ 的根边粘到 $T$ 中标号为 $s$ 的叶边处，得到
$$
T\circ_s(T_s)_{s\in S}\in\mathbf{Tree}_{\coprod_s S_s}.
$$
顶点集合是
$$
V(T)\coprod\coprod_s V(T_s).
$$
叶标号由所有 $T_s$ 的叶标号给出。

**命题 H.9.** 树 grafting 诱导 operad 复合
$$
\mathbb F(E)(S)\times\prod_{s\in S}\mathbb F(E)(S_s)
\to
\mathbb F(E)\left(\coprod_{s\in S}S_s\right).
$$

**证明.** 在代表元层面，一个元素由 $E$-装饰树 $T$ 和 $E$-装饰树族 $T_s$ 给出。Grafting 后的树顶点集合是不交并，因此装饰也按原样合并。若改变任一代表元，即沿树同构替换装饰树，则 grafting 后得到的总树也由诱导同构相连。故该构造通过 coend 商。$\square$

**命题 H.10.** 命题 H.9 的复合满足 operad 结合律和单位律。

**证明.** 结合律来自 grafting 的结合律：若先把 $T_{s,t}$ graft 到 $T_s$，再 graft 到 $T$，或先把 $T_s$ graft 到 $T$，再把 $T_{s,t}$ graft 到对应叶上，最终顶点集合、边 incidence relation、根边和叶标号都相同。装饰也只是同一组顶点装饰的不交并。单位树是只有一个叶边且无内部顶点的树；grafting 它不改变原树。因此 operad 公理成立。$\square$

## H.4 自由性的证明

**定理 H.11.** $\mathbb F(E)$ 是由对称序列 $E$ 生成的自由 operad。即对任意 operad $\mathcal O$，有自然双射
$$
\operatorname{Hom}_{\operatorname{Op}}(\mathbb F(E),\mathcal O)
\cong
\operatorname{Hom}_{\operatorname{SymSeq}}(E,U\mathcal O).
$$

**证明.** 设 $\theta:E\to U\mathcal O$ 是对称序列 morphism。对 $E$-装饰树 $T$，按树的根方向自底向上复合：每个顶点装饰 $e_v\in E(\operatorname{In}(v))$ 先由 $\theta$ 送到 $\mathcal O(\operatorname{In}(v))$，再沿树的内部边依次使用 $\mathcal O$ 的 operad 复合。Operad 结合律说明该值不依赖于先收缩哪条内部边；等变性说明它对树同构不变。因此得到函数
$$
\mathbb F(E)(S)\to\mathcal O(S)
$$
并且这些函数组成 operad morphism $\widehat\theta:\mathbb F(E)\to\mathcal O$。

反过来，任意 operad morphism $\Phi:\mathbb F(E)\to\mathcal O$ 限制到 corolla 装饰树，给出对称序列 morphism
$$
E\to U\mathcal O.
$$
由于 $\mathbb F(E)$ 的每个元素由 corolla 装饰经树 grafting 生成，operad morphism 必须把它送到定理 H.11 证明中构造的逐顶点复合值，因此 $\Phi=\widehat\theta$。两种构造互逆，并且关于 $E$ 与 $\mathcal O$ 自然。$\square$

## H.5 平面公式与对称公式的关系

**命题 H.12.** 选择每个顶点输入边的全序，可以把非平面叶标号树公式改写为平面树公式再除以对称群重标号。

**证明.** 给定非平面树 $T$，对每个顶点 $v$ 选择 $\operatorname{In}(v)$ 的全序，得到平面树。不同选择组成集合
$$
\prod_{v\in V(T)}\operatorname{Lin}(\operatorname{In}(v)),
$$
其上 $\prod_v\Sigma_{\operatorname{In}(v)}$ 自由传递作用。对称序列 $E$ 的函子性正好记录改变这些局部输入顺序时装饰如何重标号。故“先选择所有平面顺序再取局部对称群商”与直接使用非平面输入边集合 $E(\operatorname{In}(v))$ 给出同一个 coend。$\square$

**警告 H.13.** 平面树公式适合非对称 operad；自由对称 operad 必须保留叶标号和顶点输入边的对称群作用。若把平面树公式直接用于对称 operad 而不取重标号商，会得到过大的对象。

## H.6 与 Moerdijk-Weiss 树范畴的关系

**定义 H.14.** 对有根非平面树 $T$，Moerdijk-Weiss 构造的 colored operad $\Omega(T)$ 以边集 $E(T)$ 为颜色；每个顶点 $v$ 给出一个生成运算，其输入颜色为 $\operatorname{In}(v)$，输出颜色为该顶点的输出边。

**说明 H.15.** $\Omega(T)$ 不是自由 operad $\mathbb F(E)$ 的一个值。它是由单棵树 $T$ 自由生成的 colored operad，用于定义树范畴 $\Omega$。而 $\mathbb F(E)(S)$ 是对所有 $S$-叶标号树的 $E$-装饰取群胚商后得到的单色或多色 operad 值。

**命题 H.16.** Dendroidal nerve
$$
N_d(\mathcal P)_T=\operatorname{Hom}_{\operatorname{Operad}}(\Omega(T),\mathcal P)
$$
记录的是把树 $T$ 的每个顶点标到 $\mathcal P$ 中相容运算的方式。

**证明.** 因为 $\Omega(T)$ 由边颜色和顶点生成运算自由生成，一个 colored operad morphism $\Omega(T)\to\mathcal P$ 等价于为每条边选择 $\mathcal P$ 的颜色，并为每个顶点选择一个输入颜色和输出颜色匹配的运算。自由性的关系正是沿树内部边的颜色匹配关系。$\square$

## H.7 本附录小结

平面树控制非对称代入；叶标号非平面树控制自由对称 operad；Moerdijk-Weiss 树控制 dendroidal nerve。自由 operad 的严格公式应写成树群胚 coend
$$
\mathbb F(E)(S)=\int^{T\in\mathbf{Tree}_S}\prod_{v\in V(T)}E(\operatorname{In}(v)),
$$
而不是未说明自同构群和叶标号的“树的集合”。
