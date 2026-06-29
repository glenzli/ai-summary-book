# 附录 Y：合成上同调证明核与高级输入

本附录把第十二章的合成上同调入口改写为可引用的严格骨架。目标不是在本书中重建完整稳定同伦论，而是给出 HoTT 中 Eilenberg-Mac Lane 型、上同调群、函子性、悬挂同构、球面计算和 cup product 的精确定理形态，并标明哪些部分依赖外部高阶归纳构造或经典同伦论输入。

固定阿贝尔群 $A$。若讨论 cup product，则固定交换环 $R$。

## Y.1 Eilenberg-Mac Lane 型输入

**输入 Y.1（Eilenberg-Mac Lane 型塔）.** 对每个 $n:\mathbb N$，给定带基点类型
$$
K(A,n)
$$
并给定以下结构：

1.  $K(A,0)$ 是表示集合 $A$ 的 0-型；
2.  对 $n\ge 0$，有基点等价
    $$
    \Omega K(A,n+1)\simeq K(A,n);
    $$
3.  $K(A,n)$ 是 $n$-型；
4.  $K(A,n)$ 带有由 $A$ 的阿贝尔群结构诱导的交换 $H$-space 结构，使得映射类型的集合截断继承阿贝尔群结构。

**使用边界。** 该输入可由高阶归纳类型、谱或 Eilenberg-Mac Lane 对象构造给出。本书不把 EM 型塔作为基础规则；它是第十二章的高级构造输入。

**命题 Y.2（EM 型的同伦群特征）.** 在输入 Y.1 下，
$$
\pi_k(K(A,n))\cong
\begin{cases}
A, & k=n,\\
0, & k\ne n,
\end{cases}
$$
其中 $k,n\ge 1$ 时同构为群同构。

**证明（证明核）。** 对 $n$ 作归纳。由
$$
\pi_k(K(A,n+1))=\pi_{k-1}(\Omega K(A,n+1))
$$
和输入 Y.1 的等价
$$
\Omega K(A,n+1)\simeq K(A,n)
$$
把命题降到低一阶。基步 $K(A,0)$ 是表示集合 $A$ 的 0-型：其 $0$-截断点集为 $A$，高阶 loop 截断平凡。群结构相容性由 EM 塔的 $H$-space 结构和等价保持群结构给出。$\square$

## Y.2 上同调群

**定义 Y.3（非约化上同调）.** 对类型 $X$，定义
$$
H^n(X;A)\coloneqq \|X\to K(A,n)\|_0.
$$

**定义 Y.4（约化上同调）.** 对带基点类型 $(X,x_0)$，定义
$$
\widetilde H^n(X;A)
\coloneqq
\|X\to_\ast K(A,n)\|_0.
$$

**命题 Y.5（上同调的阿贝尔群结构）.** 对 $n\ge1$，$H^n(X;A)$ 和 $\widetilde H^n(X;A)$ 继承阿贝尔群结构。

**证明（证明核）。** 输入 Y.1 给出 $K(A,n)$ 的交换 $H$-space 运算
$$
\mu:K(A,n)\times K(A,n)\to K(A,n),
$$
单位和逆映射。对代表元 $f,g:X\to K(A,n)$，定义
$$
f+g\coloneqq \lambda x.\,\mu(f(x),g(x)).
$$
单位和逆逐点定义。结合律、单位律、交换律和逆元律在映射类型中由函数外延性逐点归约为 $K(A,n)$ 的 $H$-space 群律；再由集合截断递归下降到 $\|X\to K(A,n)\|_0$。带基点版本相同，需额外检查运算保持基点；这由 $K(A,n)$ 的基点是 $H$-space 单位给出。$\square$

## Y.3 函子性

**定义 Y.6（反变函子性）.** 若 $f:X\to Y$，定义
$$
f^\ast:H^n(Y;A)\to H^n(X;A)
$$
为在代表元上预合成：
$$
f^\ast([u])\coloneqq[u\circ f].
$$

**命题 Y.7（函子律）.** 有
$$
(\mathsf{id}_X)^\ast=\mathsf{id}_{H^n(X;A)}
$$
和
$$
(g\circ f)^\ast=f^\ast\circ g^\ast.
$$

**证明.** 对集合截断代表元归纳，两个等式分别化为函数复合的 judgmental equality 或由函数外延性给出的路径。目标是集合，因此截断归纳合法。$\square$

**命题 Y.8（群同态性）.** 对 $n\ge1$，$f^\ast$ 是阿贝尔群同态。

**证明.** 对代表元 $u,v:Y\to K(A,n)$，
$$
(u+v)\circ f
=
\lambda x.\,\mu(u(fx),v(fx))
=
(u\circ f)+(v\circ f).
$$
单位和逆同理，下降到集合截断。$\square$

## Y.4 悬挂同构

**输入 Y.9（pointed suspension-loop adjunction）.** 对带基点类型 $X$ 和带基点类型 $Y$，有自然等价
$$
(\Sigma X\to_\ast Y)\simeq (X\to_\ast \Omega Y).
$$

这是悬挂 HIT 的标准消去/递归性质；本书第十章只给出基础入口，完整依赖形式见附录 L。

**定理 Y.10（约化上同调悬挂同构）.** 对 $n\ge0$，
$$
\widetilde H^{n+1}(\Sigma X;A)\cong \widetilde H^n(X;A).
$$

**证明（证明核）。** 展开定义：
$$
\widetilde H^{n+1}(\Sigma X;A)
=
\|\Sigma X\to_\ast K(A,n+1)\|_0.
$$
由输入 Y.9，
$$
(\Sigma X\to_\ast K(A,n+1))
\simeq
(X\to_\ast \Omega K(A,n+1)).
$$
由 Y.1 的 loop 等价
$$
\Omega K(A,n+1)\simeq K(A,n),
$$
得到
$$
(X\to_\ast \Omega K(A,n+1))
\simeq
(X\to_\ast K(A,n)).
$$
对等价取集合截断，并用等价保持阿贝尔群结构，得到所需同构。$\square$

## Y.5 球面上同调计算

令 $\mathbb S^n$ 为 $n$-球，取标准基点。

**输入 Y.11（维数公理）.** 对阿贝尔群 $A$，
$$
\widetilde H^0(\mathbb S^0;A)\cong A,
\qquad
\widetilde H^k(\mathbb S^0;A)\cong 0\quad(k>0).
$$
并且对 $m\ge1$，球面 $\mathbb S^m$ 连通，从而
$$
\widetilde H^0(\mathbb S^m;A)\cong0.
$$

该输入属于 EM 上同调的维数性质，可由经典同伦论或合成上同调构造提供。

**定理 Y.12（约化球面上同调）.** 对 $n\ge0$，
$$
\widetilde H^k(\mathbb S^n;A)\cong
\begin{cases}
A,& k=n,\\
0,& k\ne n.
\end{cases}
$$

**证明（证明核）。** 使用球面的悬挂表示
$$
\mathbb S^{n+1}\simeq\Sigma\mathbb S^n
$$
和悬挂同构 Y.10。归纳地把
$$
\widetilde H^k(\mathbb S^n;A)
$$
在 $k\ge n$ 时降为
$$
\widetilde H^{k-n}(\mathbb S^0;A)
$$
的相应陈述；最后用输入 Y.11。若 $k<n$，则反复使用 Y.10 共 $k$ 次，化为
$$
\widetilde H^0(\mathbb S^{n-k};A),
$$
而 $n-k\ge1$，故由 Y.11 的连通性部分该群为零。$\square$

**推论 Y.13（非约化球面上同调，$n\ge1$）.** 对 $n\ge1$，
$$
H^k(\mathbb S^n;A)\cong
\begin{cases}
A,& k=0,\\
A,& k=n,\\
0,& k\ne 0,n.
\end{cases}
$$

**证明.** 非约化上同调分解为基点分量与约化分量：
$$
H^k(X;A)\cong H^k(\mathbf 1;A)\oplus \widetilde H^k(X;A)
$$
在 $X$ 连通且带基点时成立。对 $\mathbb S^n$（$n\ge1$）使用连通性和 Y.12；$H^0(\mathbf 1;A)\cong A$，$H^k(\mathbf 1;A)=0$ 对 $k>0$。$\square$

## Y.6 Cup product

**输入 Y.14（EM 乘法）.** 对交换环 $R$，存在自然的乘法映射
$$
\smile_{p,q}:K(R,p)\times K(R,q)\to K(R,p+q),
$$
并满足单位、结合、双线性和 graded commutativity 的相干律。

**定义 Y.15（cup product）.** 对代表元
$$
u:X\to K(R,p),\qquad v:X\to K(R,q),
$$
定义
$$
u\smile v
\coloneqq
\smile_{p,q}\circ (u,v)\circ\Delta_X,
$$
即
$$
(u\smile v)(x)\coloneqq \smile_{p,q}(u(x),v(x)).
$$
下降到集合截断得到
$$
\smile:H^p(X;R)\times H^q(X;R)\to H^{p+q}(X;R).
$$

**定理 Y.16（上同调环律）.** 直和
$$
H^\ast(X;R)\coloneqq\bigoplus_{n\ge0}H^n(X;R)
$$
在 cup product 下构成 graded-commutative ring：
$$
a\smile b=(-1)^{pq}b\smile a
$$
其中 $a\in H^p(X;R)$，$b\in H^q(X;R)$。

**证明（外部输入 / 证明核）。** 运算由 Y.15 定义。结合律、单位律、双线性和 graded commutativity 分别由输入 Y.14 的相干律逐点推出，再用函数外延性和集合截断归纳下降到上同调群。本书不在正文重写其全部高阶相干。$\square$

## Y.7 Eilenberg-Steenrod 性质

**输入 Y.17（合成 Eilenberg-Steenrod 性质）.** EM 上同调满足同伦不变性、长正合列、切除、悬挂同构和维数公理的合适 HoTT 形式。

**使用边界。** 这些性质依赖更系统的 cofibration、pushout、exactness 和群代数开发。本书可引用 Y.17 作为高级外部输入；任何具体计算若不在 Y.10-Y.16 覆盖范围内，应记录所用论文定理或经典来源。

## Y.8 来源与边界

本附录供第十二章引用为教材层定义与证明核。其外部输入分为三类：

1.  EM 型塔与其 loop 等价；
2.  上同调乘法的高阶相干；
3.  Eilenberg-Steenrod 性质和球面维数计算。

这三类输入不回流为前十一章的基础规则。若后续扩写为完整稳定同伦论章节，应先补谱、cofiber sequence、长正合列和具体球面计算，再把 Y.17 拆成书内证明。
