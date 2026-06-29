# 附录 B：树、分块、代入乘积和 coinvariants 公式

本附录整理全书反复使用的三个计算语言：

1. 有限集分块；
2. 树代入；
3. arity 公式中的对称群 coinvariants。

正文优先使用有限集和树的无坐标语言；本附录给出与传统 arity 公式的互译。

## B.1 有限集分块

**定义 B.1.** 设 $S$ 是有限集。$S$ 的分块 $\pi$ 是有限个非空两两不交子集组成的集合
$$
\operatorname{Bl}(\pi)=\{B_1,\ldots,B_r\}
$$
满足
$$
S=\coprod_{B\in\operatorname{Bl}(\pi)}B.
$$
块集合 $\operatorname{Bl}(\pi)$ 本身是有限集。

**定义 B.2.** 若 $\pi$ 和 $\rho$ 是 $S$ 的分块，称 $\rho$ refining $\pi$，若每个 $\rho$-块都包含于唯一的 $\pi$-块中。此时对每个 $B\in\operatorname{Bl}(\pi)$，得到 $B$ 的诱导分块
$$
\rho|_B.
$$

**命题 B.3（分块拉平）.** 给定 $S$ 的分块 $\pi$，并对每个 $B\in\operatorname{Bl}(\pi)$ 给定 $B$ 的分块 $\rho_B$，则所有 $\rho_B$ 的块组成 $S$ 的分块
$$
\rho=\coprod_{B\in\operatorname{Bl}(\pi)}\rho_B.
$$
并且 $\rho$ refining $\pi$。

**证明.** 每个 $\rho_B$ 的块两两不交且并为 $B$。不同 $B$ 之间两两不交，因为 $\pi$ 是分块。因此所有 $\rho_B$ 的块两两不交。它们的并为
$$
\coprod_{B\in\operatorname{Bl}(\pi)}B=S.
$$
每个 $\rho_B$-块包含于对应的 $B$，故 $\rho$ refining $\pi$。$\square$

**命题 B.4（分块拉平的结合律）.** 若分块被分三层给出，则先拉平内两层再拉平外层，或先拉平外两层再拉平内层，得到同一个 $S$ 的分块。

**证明.** 三层数据可写为：$S$ 的分块 $\pi$；每个 $B\in\operatorname{Bl}(\pi)$ 的分块 $\rho_B$；每个 $C\in\operatorname{Bl}(\rho_B)$ 的分块 $\tau_C$。无论按哪种顺序拉平，最终块集合都是所有 $\tau_C$ 的块。块作为 $S$ 的子集相同，因此得到同一个分块。$\square$

## B.2 对称序列的代入乘积

设 $X,Y:\mathbf B_{\mathcal U}\to\mathbf{Set}_{\mathcal U}$ 是集合值对称序列。

**定义 B.5.** 代入乘积定义为
$$
(X\circ Y)(S)
=
\coprod_{\pi\in\operatorname{Part}(S)}
X(\operatorname{Bl}(\pi))\times
\prod_{B\in\operatorname{Bl}(\pi)}Y(B).
$$

一个元素写作
$$
(\pi;x;(y_B)_{B\in\operatorname{Bl}(\pi)}).
$$

**命题 B.6.** 若 $\sigma:S\to S'$ 是双射，则它诱导函数
$$
(X\circ Y)(S)\to(X\circ Y)(S').
$$

**证明.** 双射 $\sigma$ 把 $S$ 的分块 $\pi$ 送到 $S'$ 的分块 $\sigma\pi$，其块为 $\sigma(B)$。它还诱导块集合双射
$$
\operatorname{Bl}(\pi)\to\operatorname{Bl}(\sigma\pi).
$$
因此 $X$ 的函子性给出
$$
X(\operatorname{Bl}(\pi))\to X(\operatorname{Bl}(\sigma\pi)),
$$
而每个限制双射 $B\to\sigma(B)$ 给出
$$
Y(B)\to Y(\sigma(B)).
$$
把这些映射相乘并送入对应的 coproduct summand，即得所需函数。$\square$

**定理 B.7.** 代入乘积 $\circ$ 在对称序列范畴上满足结合律 up to natural isomorphism：
$$
(X\circ Y)\circ Z\cong X\circ(Y\circ Z).
$$

**证明.** 两边都等价于同一个“三层分块”数据。

首先计算 $X\circ(Y\circ Z)$ 在 $S$ 上的元素。它由如下数据组成：

1. $S$ 的外层分块 $\pi$；
2. 元素 $x\in X(\operatorname{Bl}(\pi))$；
3. 对每个 $B\in\operatorname{Bl}(\pi)$，一个元素 of $(Y\circ Z)(B)$。

第 3 项等价于为每个 $B$ 选择 $B$ 的分块 $\rho_B$，元素
$$
y_B\in Y(\operatorname{Bl}(\rho_B)),
$$
以及每个 $C\in\operatorname{Bl}(\rho_B)$ 上的元素 $z_C\in Z(C)$。因此右端数据就是 $S$ 的外层分块 $\pi$、每个外块内部的分块 $\rho_B$，以及 $x,y_B,z_C$。

再计算 $(X\circ Y)\circ Z$。它由 $S$ 的分块 $\rho$、每个 $C\in\operatorname{Bl}(\rho)$ 上的元素 $z_C\in Z(C)$，以及
$$
(X\circ Y)(\operatorname{Bl}(\rho))
$$
中的一个元素组成。后一元素等价于块集合 $\operatorname{Bl}(\rho)$ 的分块 $\bar\pi$，元素
$$
x\in X(\operatorname{Bl}(\bar\pi)),
$$
以及对每个 $D\in\operatorname{Bl}(\bar\pi)$ 的元素
$$
y_D\in Y(D).
$$
由于 $D$ 是若干 $\rho$-块组成的集合，它对应 $S$ 的子集
$$
B_D=\coprod_{C\in D}C.
$$
这些 $B_D$ 构成 $S$ 的分块 $\pi$，而 $\rho$ 在每个 $B_D$ 上的限制给出分块 $\rho_{B_D}$。于是得到与上一段相同的三层数据。

两个构造互逆：从 $\pi$ 与各 $\rho_B$ 出发，用命题 B.3 拉平得到 $\rho$，并用每个 $\rho_B$ 的块集合形成 $\operatorname{Bl}(\rho)$ 的分块 $\bar\pi$；从 $\rho$ 与 $\bar\pi$ 出发，把 $\bar\pi$ 的每个块对应回 $S$ 的子集得到 $\pi$ 和 $\rho_B$。三层及更多层的相干性由命题 B.4 的分块拉平结合律给出。$\square$

## B.3 单位对称序列

**定义 B.8.** 单位对称序列 $I$ 定义为
$$
I(S)=
\begin{cases}
\{*\},& |S|=1,\\
\varnothing,& |S|\ne1.
\end{cases}
$$

**命题 B.9.** 对任意对称序列 $X$，有自然同构
$$
I\circ X\cong X,\qquad X\circ I\cong X.
$$

**证明.** 对 $I\circ X$，一个 summand 非空当且仅当外层 $I(\operatorname{Bl}(\pi))$ 非空，即 $\operatorname{Bl}(\pi)$ 只有一个元素。这等价于 $\pi=\{S\}$。此时 summand 为
$$
I(\{S\})\times X(S)\cong X(S).
$$

对 $X\circ I$，一个 summand 非空当且仅当每个块 $B$ 满足 $|B|=1$。这等价于 $\pi$ 是离散分块。此时 $\operatorname{Bl}(\pi)$ 由 singleton blocks 组成，并与 $S$ 有 canonical bijection；由 $X$ 的函子性得到 summand $X(S)$。两构造与双射 $S\to S'$ 相容，故自然。$\square$

## B.4 Arity 公式

设使用骨架 $[n]$，并把对称序列写成右 $\Sigma_n$-对象 $X(n),Y(n)$。

**命题 B.10.** 代入乘积的 arity 公式为
$$
(X\circ Y)(n)
\cong
\coprod_{k\ge0}
X(k)\times_{\Sigma_k}
\left(
\coprod_{n_1+\cdots+n_k=n}
\operatorname{Bij}\big([n],[n_1]\sqcup\cdots\sqcup[n_k]\big)
\times_{\Sigma_{n_1}\times\cdots\times\Sigma_{n_k}}
\prod_{i=1}^kY(n_i)
\right).
$$

**证明.** 一个 $[n]$ 的分块 $\pi$ 有 $k$ 个块。选择块集合与 $[k]$ 的双射会把块编号为 $1,\ldots,k$；选择每个块与 $[n_i]$ 的双射会把该分块编码为一个双射
$$
[n]\cong[n_1]\sqcup\cdots\sqcup[n_k].
$$
改变块编号由 $\Sigma_k$ 作用；改变每个块内部编号由 $\Sigma_{n_i}$ 作用。有限集公式中的
$$
X(\operatorname{Bl}(\pi))\times\prod_BY(B)
$$
经这些选择变成
$$
X(k)\times\prod_iY(n_i),
$$
而不同选择正由上述对称群作用取 coinvariants 识别。对所有 $k$ 和 $n_1+\cdots+n_k=n$ 求 coproduct，得到公式。$\square$

**警告 B.11.** 公式 B.10 的左右作用取决于附录 A 的转换约定。若采用不同文献的右作用约定，$\Sigma_k$ 与 $\Sigma_{n_i}$ 的作用方向可能需要整体取逆。

## B.5 树代入

**定义 B.12.** 平面有根树 $T$ 的顶点集合记为 $V(T)$，叶集合记为 $\operatorname{Leaf}(T)$。若对每个顶点 $v$ 指定一个平面有根树 $T_v$，且 $T_v$ 的叶数等于 $v$ 的输入数，则可把每个 $v$ 替换为 $T_v$，得到树
$$
T\{T_v\}_{v\in V(T)}.
$$

**命题 B.13.** 树代入满足结合律：若还对每个 $T_v$ 的顶点 $w$ 指定树 $T_{v,w}$，则
$$
T\{T_v\{T_{v,w}\}_w\}_v
=
\big(T\{T_v\}_v\big)\{T_{v,w}\}_{v,w}
$$
作为平面有根树相等。

**证明.** 两边的顶点集合都是所有 $T_{v,w}$ 的顶点的不交并。边的 incidence relation 由三类关系生成：每个 $T_{v,w}$ 内部的 incidence；同一 $T_v$ 中不同 $w$ 之间原有边对应的 grafting；原树 $T$ 中不同 $v$ 之间原有边对应的 grafting。两种代入顺序使用同一组三类 incidence 关系，因此得到同一棵树。平面顺序也由原有平面顺序逐层继承，两边相同。$\square$

**说明 B.14.** 非对称自由 operad 的结合律来自命题 B.13；对称自由 operad 还需加入叶标号和对称群重标号。Dendroidal sets 中的树范畴 $\Omega$ 使用非平面 rooted trees，并通过 free colored operad $\Omega(T)$ 编码复合。

## B.6 本附录小结

有限集分块给出对称序列代入乘积的无坐标定义；arity 公式是选择骨架和块编号后的 coinvariants 表达；树代入给出自由 operad 和非对称 operad 的组合模型。使用公式时应先判断当前语境是有限集、arity 右作用，还是树代入。
