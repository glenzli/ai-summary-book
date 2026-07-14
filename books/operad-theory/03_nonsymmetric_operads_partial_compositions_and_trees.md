# 第三章：非对称 operad、偏复合与树

整体代入一次把许多内层运算送入外层运算，公式紧凑，却遮住了最小的组合步骤。若固定 $p\in\mathcal P(m)$ 和 $q\in\mathcal P(n)$，只把 $q$ 放进 $p$ 的第 $i$ 个槽，就得到偏复合 $p\circ_iq$；三个运算的不同代入次序正对应一棵三顶点平面树的不同收缩次序。去掉对称群作用以后，输入槽已有固定次序，这一对应尤其清楚。本章由第一章的 arity 写法出发，证明偏复合与整体代入包含相同数据，并用有限平面有根树把结合律变成可逐边检查的组合事实。

## 3.1 非对称序列与代入乘积

**定义 3.1.** 一个非对称序列是集合族
$$
X=\{X(n)\}_{n\ge0}.
$$
非对称序列的态射 $f:X\to Y$ 是函数族 $f_n:X(n)\to Y(n)$。

**定义 3.2.** 对非对称序列 $X,Y$，定义非对称代入乘积
$$
(X\circ_{\mathrm{ns}}Y)(n)
=
\coprod_{k\ge0}
\coprod_{\substack{n_1+\cdots+n_k=n\\ n_i\ge0}}
X(k)\times Y(n_1)\times\cdots\times Y(n_k).
$$
允许 $n_i=0$，对应在第 $i$ 个槽代入 nullary operation。当 $k=0$ 时，内层和只在 $n=0$ 时有一项。

单位非对称序列 $I_{\mathrm{ns}}$ 定义为
$$
I_{\mathrm{ns}}(1)=\{*\},\qquad
I_{\mathrm{ns}}(n)=\varnothing\quad(n\ne1).
$$

**命题 3.3.** 非对称序列范畴连同 $\circ_{\mathrm{ns}}$ 与 $I_{\mathrm{ns}}$ 构成幺半范畴。

**证明.** 一个元素
$$
((x;y_1,\ldots,y_k);z_{11},\ldots,z_{k n_k})
\in ((X\circ_{\mathrm{ns}}Y)\circ_{\mathrm{ns}}Z)(n)
$$
等价于给出整数分解
$$
n=\sum_{i=1}^k\sum_{j=1}^{n_i}m_{ij},
$$
元素 $x\in X(k)$、$y_i\in Y(n_i)$、$z_{ij}\in Z(m_{ij})$。这与元素
$$
(x;(y_1;z_{11},\ldots,z_{1n_1}),\ldots,
(y_k;z_{k1},\ldots,z_{kn_k}))
\in (X\circ_{\mathrm{ns}}(Y\circ_{\mathrm{ns}}Z))(n)
$$
是同一数据。该对应给出结合约束。单位约束来自唯一分解 $n=1+\cdots+1$ 和 $n=n$。五边形和三角形相干性化为有限整数分块的拉平次序无关性。$\square$

**定义 3.4.** 非对称 operad 是幺半范畴
$$
(\operatorname{Seq}_{\mathrm{ns}},\circ_{\mathrm{ns}},I_{\mathrm{ns}})
$$
中的幺半对象。也就是说，它由集合族 $\mathcal P(n)$、单位 $\mathbf 1\in\mathcal P(1)$ 和代入映射
$$
\gamma:
\mathcal P(k)\times\mathcal P(n_1)\times\cdots\times\mathcal P(n_k)
\to
\mathcal P(n_1+\cdots+n_k)
$$
组成，满足单位和结合公理。

这里以及下文整体代入中始终允许 $n_i=0$。

## 3.2 偏复合公理

**定义 3.5.** 设 $\mathcal P$ 是非对称 operad。对 $p\in\mathcal P(m)$、$q\in\mathcal P(n)$ 和 $1\le i\le m$，定义第 $i$ 个偏复合
$$
p\circ_i q\in\mathcal P(m+n-1)
$$
为
$$
\gamma(p;\mathbf 1,\ldots,\mathbf 1,q,\mathbf 1,\ldots,\mathbf 1),
$$
其中 $q$ 位于第 $i$ 个输入槽。

**命题 3.6.** 非对称 operad 的偏复合满足以下恒等式。设 $p\in\mathcal P(m)$、$q\in\mathcal P(n)$、$r\in\mathcal P(\ell)$。

1. 单位：
   $$
   \mathbf 1\circ_1 p=p,\qquad p\circ_i\mathbf 1=p.
   $$
2. 嵌套代入：若 $1\le j\le n$，则
   $$
   (p\circ_i q)\circ_{i+j-1}r
   =
   p\circ_i(q\circ_j r).
   $$
3. 分离代入：若 $k<i$，则
   $$
   (p\circ_i q)\circ_k r
   =
   (p\circ_k r)\circ_{i+\ell-1}q.
   $$
   若 $k\ge i+n$，则
   $$
   (p\circ_i q)\circ_k r
   =
   (p\circ_{k-n+1}r)\circ_i q.
   $$

**证明.** 所有等式都是非对称 operad 结合律的特例。以嵌套代入为例，左侧先把 $q$ 代入 $p$ 的第 $i$ 槽，再把 $r$ 代入所得运算中原来属于 $q$ 的第 $j$ 槽；右侧先在 $q$ 内做同一代入，再把结果代入 $p$ 的第 $i$ 槽。两边对应同一个三层整数分块。

若 $k<i$，则 $r$ 被代入 $p$ 的第 $k$ 槽，位于 $q$ 所在槽之前；先插入 $r$ 会使原第 $i$ 槽后移 $\ell-1$ 位，所以右侧第二次代入位置为 $i+\ell-1$。若 $k\ge i+n$，则 $r$ 被代入 $p$ 中位于 $q$ 后方的槽；在先插入 $q$ 的表达式中位置为 $k$，对应 $p$ 原来的第 $k-n+1$ 槽。单位等式由 operad 单位律得到。$\square$

**定义 3.7.** 一个偏复合型非对称 operad 由集合族 $\mathcal P(n)$、元素 $\mathbf 1\in\mathcal P(1)$ 和函数
$$
\circ_i:\mathcal P(m)\times\mathcal P(n)\to\mathcal P(m+n-1)
$$
组成，并满足命题 3.6 中的单位、嵌套代入和分离代入恒等式。

**定理 3.8.** 非对称 operad 与偏复合型非对称 operad 是等价的数据。

**证明.** 从非对称 operad 到偏复合型结构由定义 3.5 和命题 3.6 给出。

反过来，给定偏复合型结构，定义整体代入
$$
\gamma(p;q_1,\ldots,q_m)
$$
为迭代偏复合
$$
(((p\circ_m q_m)\circ_{m-1}q_{m-1})\cdots)\circ_1 q_1,
$$
其中在第 $i$ 步使用当前运算中由原第 $i$ 个输入槽对应的位置。更明确地说，从右向左代入避免后方槽被前方代入改变位置。

需要证明该定义满足整体代入结合律。任意两种迭代顺序都可由相邻交换连接：相邻交换要么发生在一个运算内部和其子运算内部，对应嵌套代入恒等式；要么发生在两个互不包含的输入槽，对应分离代入恒等式。因此迭代结果只依赖于由 $p$ 和 $q_i$ 形成的两层树，而不依赖计算顺序。三层整体结合律同理化为三层树的收缩顺序无关性。单位律由定义 3.7 的单位公理给出。两种构造互逆。$\square$

## 3.3 从对称 operad 到非对称 operad

**定义 3.9.** 设 $\mathcal O$ 是对称 operad。其底层非对称 operad $U_{\mathrm{ns}}\mathcal O$ 定义为
$$
(U_{\mathrm{ns}}\mathcal O)(n)=\mathcal O([n]).
$$
对 $n_1,\ldots,n_k\ge0$，整体代入使用唯一的非降映射
$$
[n_1+\cdots+n_k]\longrightarrow[k]
$$
其第 $i$ 个纤维是连续的 $n_i$-元素区间；当 $n_i=0$ 时该纤维为空。各非空纤维按自然顺序排列。

**定义 3.10.** 对 $p\in\mathcal O(m)$ 和 $q\in\mathcal O(n)$，对称 operad 的偏复合
$$
p\circ_i q\in\mathcal O(m+n-1)
$$
是沿非降映射 $\rho_i:[m+n-1]\to[m]$ 的代入，其中第 $i$ 个纤维有 $n$ 个元素，其余纤维各有一个元素。第 $i$ 个纤维上放入 $q$，其余单点纤维上放入单位 $\mathbf 1$。当 $n=0$ 时，$\rho_i$ 的第 $i$ 个纤维为空；该公式仍定义把 nullary $q$ 代入第 $i$ 个槽。

**命题 3.11.** 定义 3.10 的偏复合满足命题 3.6 的三类恒等式。

**证明.** 这些恒等式是对称 operad 对上述非降有限集映射的结合律。所有非空纤维都是区间，空纤维保留其目标槽，且没有额外目标置换；因此证明与非对称情形相同。$\square$

## 3.4 平面有根树

**定义 3.12.** 一个平面有根树（planar rooted tree）由有限有向无环图 $T$ 组成，满足：

- 每条边方向从叶到根；
- 有唯一输出外边，称为根边；
- 每个内部顶点有恰好一条输出边和有限多条输入边；
- 每个内部顶点的输入边集合带有线性顺序；
- 没有输入顶点的外边称为叶。

树的 arity 是叶的个数。只含一条边且没有内部顶点的树称为单位树，arity 为 $1$。

**定义 3.13.** 设 $\mathcal P$ 是非对称 operad。一个 $\mathcal P$-装饰平面树是平面有根树 $T$ 连同对每个内部顶点 $v$ 的元素
$$
p_v\in\mathcal P(\operatorname{in}(v)),
$$
其中 $\operatorname{in}(v)$ 是 $v$ 的输入边数。

**定义 3.14.** 若 $e$ 是连接顶点 $v$ 的输出与顶点 $w$ 的第 $i$ 个输入的内部边，则收缩 $e$ 得到新顶点，其装饰为
$$
p_w\circ_i p_v.
$$
其他顶点装饰保持不变。单位树的值定义为 $\mathbf 1$。

**命题 3.15.** 对一个 $\mathcal P$-装饰平面树，反复收缩内部边直到得到单顶点树，所得元素
$$
\operatorname{ev}_{\mathcal P}(T)\in\mathcal P(n)
$$
与内部边收缩顺序无关。

**证明.** 任意两个内部边收缩序列可由交换相邻收缩次序连接。相邻两次收缩若发生在同一嵌套链上，则对应命题 3.6 的嵌套代入恒等式；若发生在两个互不包含的分支上，则对应分离代入恒等式。单位顶点的插入或删除由单位律控制。因此所有收缩序列给出同一元素。$\square$

**推论 3.16.** 非对称 operad 可以等价地看作给每个装饰平面有根树指定一个复合运算，并且该复合只依赖于树本身而不依赖于把树分解为二元 grafting 的方式。

**证明.** 由命题 3.15 得到从偏复合到树复合的构造。反过来，取只有两个内部顶点且由一条内部边连接的树，即恢复偏复合。树复合的收缩次序无关性限制到三顶点树时正是命题 3.6 的公理。$\square$

## 3.5 收缩次序与结合律

偏复合把代入拆成单条内边的收缩：嵌套情形处理同一枝上的两条边，交换情形处理不同枝上的两条边。命题 3.6 的恒等式保证任意装饰平面树的所有收缩序列都得到同一个元素；反过来，这个收缩无关性恢复整体代入。树因而不只是示意图，而是 operad 表达式的语法。下一章将不再从已有 operad 中计算树值，而是把所有装饰树本身组成自由 operad，再精确说明“施加关系”意味着取哪一种商。

## 练习

**练习 3.1.** 对非对称序列写出 $(X\circ_{\mathrm{ns}}Y)(0)$、$(X\circ_{\mathrm{ns}}Y)(1)$、$(X\circ_{\mathrm{ns}}Y)(2)$。

**练习 3.2.** 用偏复合公理证明
$$
((p\circ_2 q)\circ_1 r)=((p\circ_1 r)\circ_{1+\ell}q)
$$
在 arity 匹配时成立。

**练习 3.3.** 画出有三个内部顶点的一条链形平面树，并把命题 3.6 的嵌套代入恒等式标在图上。

**练习 3.4.** 证明一个对称 operad 的底层非对称 operad 忘掉了哪些信息。

**练习 3.5.** 对 $\operatorname{Ass}$ 的底层非对称 operad，计算二元乘法 $m$ 的 $m\circ_1m$ 和 $m\circ_2m$，并解释它们在 $\operatorname{Ass}(3)$ 中为何不同、在代数上为何由结合律联系。
