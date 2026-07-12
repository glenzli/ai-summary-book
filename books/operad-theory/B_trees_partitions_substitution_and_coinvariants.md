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
\operatorname*{colim}_{(f:S\to T)\in\operatorname{Fib}(S)}
X(T)\times\prod_{t\in T}Y(f^{-1}(t)),
$$
其中 $\operatorname{Fib}(S)$ 是定义 1.4.1 的纤维分解群胚，并且 $f$ 不要求满射。

一个元素写作
$$
(T,f;x;(y_t)_{t\in T}),
$$
并对目标双射 $u:T\to T'$ 施加
$$
(T,f;x;(y_t))
\sim
(T',uf;X(u)x;(y'_{u(t)}=y_t)).
$$

若 $Y(\varnothing)=\varnothing$，只有满射项有贡献，定义 B.5 才退化为 B.1 的非空分块公式。

**命题 B.6.** 若 $\sigma:S\to S'$ 是双射，则它诱导函数
$$
(X\circ Y)(S)\to(X\circ Y)(S').
$$

**证明.** 把代表元 $(T,f;x;(y_t))$ 送到
$$
(T,f\sigma^{-1};x;
(Y(\sigma|_{f^{-1}(t)})(y_t))_{t\in T}).
$$
目标双射关系在重标号前后使用同一个 $u:T\to T'$，故该规则通过 colimit 商。恒等双射与复合双射的结论分别来自恒等限制和
$(\tau\sigma)|_{f^{-1}(t)}=\tau|_{\sigma(f^{-1}(t))}\sigma|_{f^{-1}(t)}$。$\square$

**定理 B.7.** 代入乘积 $\circ$ 在对称序列范畴上满足结合律 up to natural isomorphism：
$$
(X\circ Y)\circ Z\cong X\circ(Y\circ Z).
$$

**证明.** 两边都等价于同一个二层映射数据。具体地，该数据是
$$
S\xrightarrow{g}U\xrightarrow{p}T,
$$
连同
$$
x\in X(T),\qquad
y_t\in Y(p^{-1}(t)),\qquad
z_u\in Z(g^{-1}(u)),
$$
并对与 $g,p$ 交换的 $U,T$ 双射取商。

对 $(X\circ Y)\circ Z$，$g$ 与 $z_u$ 来自最外一次定义 B.5，而 $p,x,y_t$ 表示 $(X\circ Y)(U)$。对 $X\circ(Y\circ Z)$，外层映射是 $pg:S\to T$；在其 $t$-纤维上，限制
$$
g_t:(pg)^{-1}(t)\to p^{-1}(t)
$$
与 $y_t,(z_u)_{u\in p^{-1}(t)}$ 表示一个 $(Y\circ Z)$-元素。反向把各 $g_t$ 的目标作不交并即可恢复 $U$ 与 $p$。两构造互逆且尊重全部目标双射关系。

四重代入的任一加括号方式同样展开成三层映射 $S\to U_1\to U_2\to U_3$。五边形两条路径在该共同数据上都是恒等映射，故结合约束满足 Mac Lane 五边形。$\square$

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

**证明.** 对 $I\circ X$，外层项 $I(T)$ 非空当且仅当 $T$ 是单点集。函数 $S\to T$ 唯一，其唯一纤维是 $S$，故 colimit 自然同构于 $X(S)$。

对 $X\circ I$，乘积 $\prod_{t\in T}I(f^{-1}(t))$ 非空当且仅当每个纤维恰有一个元素，即 $f:S\to T$ 是双射。对应类由 $x\in X(T)$ 给出，并通过 $X(f^{-1})(x)$ 自然识别为 $X(S)$。当 $S=\varnothing$ 时，第一段使用 $\varnothing\to\{*\}$，第二段使用 $\varnothing\to\varnothing$；故两个单位约束都覆盖 arity $0$。三角形相干性由二层映射表示中插入、删除上述唯一单位层直接得到。$\square$

## B.4 Arity 公式

设使用骨架 $[n]$，并把 $X(n)$ 的函子性左作用按命题 A.9 转为右 $\Sigma_n$-作用。定义
$$
Y^{\langle k\rangle}(n)
=
\coprod_{f:[n]\to[k]}\prod_{i=1}^kY(f^{-1}(i)).
$$
$\Sigma_k$ 的左作用把 $(f,(y_i))$ 送到
$(\sigma f,(y'_{\sigma(i)}=y_i))$。

**命题 B.10.** 代入乘积的 arity 公式为
$$
(X\circ Y)(n)
\cong
\coprod_{k\ge0}
X(k)\times_{\Sigma_k}Y^{\langle k\rangle}(n).
$$
按纤维基数分组后，$Y^{\langle k\rangle}(n)$ 也可写成
$$
\coprod_{\substack{n_1+\cdots+n_k=n\\ n_i\ge0}}
\operatorname{Bij}\left(\coprod_{i=1}^k[n_i],[n]\right)
\times_{\Sigma_{n_1}\times\cdots\times\Sigma_{n_k}}
\prod_{i=1}^kY(n_i),
$$
其中 $\prod_iY(n_i)$ 的左作用由本书的右作用取逆得到。外层 $\Sigma_k$ 同时置换 $n_i$、双射的各源分量和 $Y(n_i)$ 因子。

**证明.** 在定义 B.5 中选择目标 $T=[k]$，每个目标双射正是一个 $\Sigma_k$-元素；colimit 关系因此恰给出第一式的 balanced product。对固定 $f:[n]\to[k]$，令 $n_i=|f^{-1}(i)|$，并选择双射 $[n_i]\to f^{-1}(i)$。这些选择合成
$\coprod_i[n_i]\to[n]$；改变第 $i$ 个选择由 $\Sigma_{n_i}$ 作用，反向由显示的双射恢复 $f$ 及各纤维坐标。故得到第二式。这里允许 $n_i=0$，正对应 $f$ 的空纤维。$\square$

**反例 B.10.1（非空分块会丢失 nullary substitution）.** 取 $X=I$，并取满足 $Y(\varnothing)=\{a\}$ 的对称序列。正确的左单位给出
$$
(I\circ Y)(\varnothing)\cong Y(\varnothing)=\{a\}.
$$
若误用非空分块公式，$\varnothing$ 的唯一分块具有空块集合，外层因子变成 $I(\varnothing)=\varnothing$，从而错误地得到空集。失败的假设正是 $Y(\varnothing)=\varnothing$。

**警告 B.11.** 公式 B.10 的左右作用取决于命题 A.9 和约定 A.10 的转换约定。若采用不同文献的右作用约定，$\Sigma_k$ 与 $\Sigma_{n_i}$ 的作用方向可能需要整体取逆。非空分块公式还额外要求内层序列在 arity $0$ 为初对象。

## B.5 树代入

**定义 B.12.** 平面有根树 $T$ 的顶点集合记为 $V(T)$，叶集合记为 $\operatorname{Leaf}(T)$。若对每个顶点 $v$ 指定一个平面有根树 $T_v$，且 $T_v$ 的叶数等于 $v$ 的输入数，则可把每个 $v$ 替换为 $T_v$，得到树
$$
T\{T_v\}_{v\in V(T)}.
$$
若 $v$ 是 nullary 顶点，则 $T_v$ 允许为零叶树；下述结合律仍使用同一个顶点与 incidence-relation 论证。

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

任意有限集映射的纤维给出允许 arity $0$ 的代入乘积；非空分块只是内层 arity $0$ 消失时的简化。Arity 公式是选择目标骨架和纤维坐标后的 coinvariants 表达；树代入给出自由 operad 和非对称 operad 的组合模型。使用公式时应先判断当前语境是有限集左作用、arity 右作用，还是树代入。
