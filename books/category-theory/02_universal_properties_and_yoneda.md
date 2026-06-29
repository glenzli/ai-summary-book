# 第二章：泛性质与 Yoneda 引理

## 本章目标

本章把“由映射进入或映射出来的方式唯一刻画对象”写成严格语言。核心内容是终对象、始对象、泛元素、可表函子、Yoneda 引理和 Yoneda 嵌入的完全忠实性。

## 依赖前置知识

需要第一章的范畴、函子、自然变换、反范畴和完全忠实函子。

## 2.1 终对象与始对象

**定义 2.1.** 设 $\mathcal C$ 为范畴。对象 $1\in\mathcal C$ 称为终对象（terminal object），若对任意 $X\in\mathcal C$，集合 $\mathcal C(X,1)$ 恰有一个元素。

对象 $0\in\mathcal C$ 称为始对象（initial object），若对任意 $X\in\mathcal C$，集合 $\mathcal C(0,X)$ 恰有一个元素。

**命题 2.2.** 若 $1$ 与 $1'$ 都是 $\mathcal C$ 的终对象，则存在唯一同构 $1\cong 1'$。

**证明.** 由 $1'$ 终，存在唯一态射 $u:1\to 1'$；由 $1$ 终，存在唯一态射 $v:1'\to 1$。复合 $v\circ u:1\to 1$ 必须等于唯一的态射 $1\to 1$，即 $\operatorname{id}_1$。同理 $u\circ v=\operatorname{id}_{1'}$。故 $u$ 是同构。若 $w:1\to1'$ 也是同构，则它首先是态射 $1\to1'$，由 $1'$ 的终性等于 $u$。$\square$

**例子 2.3.** 在 $\mathbf{Set}_{\mathcal U}$ 中，任意单点集合是终对象，空集是始对象。在 $\mathbf{Grp}$ 中，平凡群既是终对象又是始对象。

## 2.2 泛元素与可表函子

**定义 2.4.** 设 $F:\mathcal C\to\mathbf{Set}_{\mathcal U}$ 为函子。$F$ 的一个表示（representation）是一个对象 $A\in\mathcal C$ 和一个自然同构

$$
\theta:\mathcal C(A,-)\xrightarrow{\cong}F.
$$

若这样的表示存在，称 $F$ 可表（representable）。

反变函子 $P:\mathcal C^{\operatorname{op}}\to\mathbf{Set}_{\mathcal U}$ 的表示是对象 $A$ 与自然同构

$$
\mathcal C(-,A)\xrightarrow{\cong}P.
$$

**定义 2.5.** 设 $F:\mathcal C\to\mathbf{Set}$。一个泛元素（universal element）是二元组 $(A,u)$，其中 $A\in\mathcal C$ 且 $u\in F(A)$，使得对任意 $X\in\mathcal C$ 和任意 $x\in F(X)$，存在唯一态射 $f:A\to X$ 满足

$$
F(f)(u)=x.
$$

**命题 2.6.** 函子 $F:\mathcal C\to\mathbf{Set}$ 可表，当且仅当 $F$ 有泛元素。

**证明.** 若 $\theta:\mathcal C(A,-)\cong F$，取

$$
u=\theta_A(\operatorname{id}_A)\in F(A).
$$

给定 $x\in F(X)$，由于 $\theta_X:\mathcal C(A,X)\to F(X)$ 是双射，存在唯一 $f:A\to X$ 使 $\theta_X(f)=x$。自然性给出

$$
\theta_X(f)=F(f)(\theta_A(\operatorname{id}_A))=F(f)(u),
$$

所以 $(A,u)$ 泛。

反过来，若 $(A,u)$ 泛，定义

$$
\theta_X:\mathcal C(A,X)\to F(X),\qquad f\mapsto F(f)(u).
$$

泛性说明每个 $\theta_X$ 是双射。对 $g:X\to Y$，

$$
F(g)(\theta_X(f))=F(g)(F(f)(u))=F(g\circ f)(u)=\theta_Y(g\circ f),
$$

所以 $\theta$ 自然。$\square$

## 2.3 Yoneda 引理

**定理 2.7（Yoneda 引理）.** 设 $\mathcal C$ 为小范畴，$F:\mathcal C^{\operatorname{op}}\to\mathbf{Set}_{\mathcal U}$ 为预层，$A\in\mathcal C$。存在自然双射

$$
\operatorname{Nat}(\mathcal C(-,A),F)\cong F(A).
$$

该双射把自然变换 $\alpha:\mathcal C(-,A)\Rightarrow F$ 送到

$$
\alpha_A(\operatorname{id}_A)\in F(A).
$$

**证明.** 给定 $a\in F(A)$，定义自然变换 $\alpha^a:\mathcal C(-,A)\Rightarrow F$。对对象 $X$ 和态射 $f:X\to A$，令

$$
\alpha^a_X(f)=F(f)(a)\in F(X).
$$

若 $u:Y\to X$，需验证自然性：

$$
F(u)(\alpha^a_X(f))=F(u)(F(f)(a))=F(f\circ u)(a)=\alpha^a_Y(f\circ u).
$$

这正是反变函子 $F$ 的函子性。

另一方面，给定自然变换 $\alpha$，令 $a=\alpha_A(\operatorname{id}_A)$。对任意 $f:X\to A$，自然性方块给出

$$
\alpha_X(f)=F(f)(\alpha_A(\operatorname{id}_A))=F(f)(a).
$$

所以 $\alpha=\alpha^a$。两个构造互逆。对 $A$ 和 $F$ 的自然性由同一公式直接检验。$\square$

**推论 2.8.** Yoneda 嵌入

$$
y:\mathcal C\to\widehat{\mathcal C},\qquad A\mapsto\mathcal C(-,A)
$$

完全忠实。

**证明.** 对 $A,B\in\mathcal C$，由 Yoneda 引理应用于 $F=\mathcal C(-,B)$ 得

$$
\operatorname{Nat}(\mathcal C(-,A),\mathcal C(-,B))\cong \mathcal C(A,B).
$$

该双射正是函子 $y$ 在 Hom 集上的映射。因此 $y$ 完全忠实。$\square$

## 2.4 Yoneda 的使用原则

**命题 2.9.** 若 $A,B\in\mathcal C$ 且存在自然同构

$$
\mathcal C(-,A)\cong \mathcal C(-,B),
$$

则 $A\cong B$。

**证明.** 由推论 2.8，自然同构对应于 $\mathcal C(A,B)$ 中的态射 $f:A\to B$。其逆自然变换对应于 $g:B\to A$。自然变换复合对应态射复合，故 $g\circ f=\operatorname{id}_A$ 且 $f\circ g=\operatorname{id}_B$。$\square$

**例子 2.10.** 若一个集合 $S$ 表示函子

$$
\mathbf{Set}(A,-):\mathbf{Set}\to\mathbf{Set},
$$

则 $S$ 与 $A$ 同构。这里“表示对象唯一”不是集合元素层面的猜测，而是命题 2.9 的特例。

## 2.5 可表性的边界与计算

**命题 2.11.** 若 $F:\mathcal C\to\mathbf{Set}$ 被 $(A,u)$ 表示，且 $F$ 也被 $(B,v)$ 表示，则存在唯一同构 $\phi:A\to B$ 使得

$$
F(\phi)(u)=v.
$$

**证明.** 由 $(A,u)$ 的泛性，应用于 $v\in F(B)$，存在唯一 $\phi:A\to B$ 使 $F(\phi)(u)=v$。由 $(B,v)$ 的泛性，应用于 $u\in F(A)$，存在唯一 $\psi:B\to A$ 使 $F(\psi)(v)=u$。于是

$$
F(\psi\phi)(u)=F(\psi)(F(\phi)(u))=F(\psi)(v)=u.
$$

而 $\operatorname{id}_A$ 也满足 $F(\operatorname{id}_A)(u)=u$。由 $(A,u)$ 的唯一性，$\psi\phi=\operatorname{id}_A$。同理 $\phi\psi=\operatorname{id}_B$。故 $\phi$ 是同构。若 $\phi'$ 也满足 $F(\phi')(u)=v$，由 $(A,u)$ 的唯一性 $\phi'=\phi$。$\square$

**例子 2.12（不可表函子）.** 并非每个集合值函子都可表。考虑幂集函子

$$
\mathcal P:\mathbf{Set}\to\mathbf{Set},\qquad X\mapsto\mathcal P(X)
$$

其中态射 $f:X\to Y$ 被送到直接像函数 $\mathcal P(f):\mathcal P(X)\to\mathcal P(Y)$。若 $\mathcal P$ 可由集合 $A$ 表示，则存在自然同构

$$
\mathbf{Set}(A,X)\cong\mathcal P(X)
$$

对所有集合 $X$ 成立。取 $X=\varnothing$。若 $A\ne\varnothing$，则 $\mathbf{Set}(A,\varnothing)=\varnothing$，但 $\mathcal P(\varnothing)=\{\varnothing\}$，矛盾。故必须 $A=\varnothing$。但取 $X=\{0,1\}$ 时，

$$
\mathbf{Set}(\varnothing,X)
$$

为单点集，而 $\mathcal P(X)$ 有四个元素，仍矛盾。因此 $\mathcal P$ 不可表。

**例子 2.13（自然性是必要条件）.** 对有限维向量空间，$V\cong V^*$ 虽然可由选择基给出，但这种同构不自然。相反，自然映射

$$
V\to V^{**}
$$

由评价定义，不依赖基。Yoneda 引理关心的是自然变换，而不是逐对象可任意选择的双射。

**命题 2.14（Yoneda 计算原则）.** 设 $P,Q:\mathcal C^{op}\to\mathbf{Set}$ 为预层。若 $P$ 可写成可表预层的余极限

$$
P\cong\operatorname{colim}_{i\in\mathcal I}y(A_i),
$$

则对任意预层 $Q$ 有自然同构

$$
\operatorname{Nat}(P,Q)\cong
\lim_{i\in\mathcal I^{op}}Q(A_i).
$$

**证明.** 在预层范畴中，固定目标 $Q$ 的 Hom 函子把余极限变成 $\mathbf{Set}$ 中的极限。因此

$$
\operatorname{Nat}(P,Q)
\cong
\operatorname{Nat}\left(\operatorname{colim}_{i}y(A_i),Q\right)
\cong
\lim_{i\in\mathcal I^{op}}\operatorname{Nat}(y(A_i),Q).
$$

由 Yoneda 引理，

$$
\operatorname{Nat}(y(A_i),Q)\cong Q(A_i).
$$

合并即得所需同构。自然性来自 Hom-余极限伴随和 Yoneda 同构的自然性。$\square$

## 2.6 本章小结

泛性质不是非正式描述，而是可表性。Yoneda 引理说明：预层 $F$ 在对象 $A$ 上的元素，等价于从可表预层 $\mathcal C(-,A)$ 到 $F$ 的自然变换。由此得到 Yoneda 嵌入完全忠实，并且对象可以由其全部映入方式精确恢复。可表性是强条件；逐对象同构或逐对象选择通常不足以给出自然表示。

## 练习

**练习 2.1.** 证明始对象若存在，则在唯一同构意义下唯一。

**练习 2.2.** 写出 $\mathbf{Set}$ 中二元积 $A\times B$ 的泛性质，并把它改写为某个函子的表示。

**练习 2.3.** 设 $M$ 为幺半群，视为单对象范畴。描述该范畴上的一个集合值预层等价于什么代数结构。

**练习 2.4.** 对协变函子版本证明 Yoneda 引理：
$$
\operatorname{Nat}(\mathcal C(A,-),F)\cong F(A).
$$

**练习 2.5.** 证明 Yoneda 嵌入反映同构：若 $y(f)$ 是预层同构，则 $f$ 是 $\mathcal C$ 中的同构。

**练习 2.6.** 证明协变可表函子 $\mathcal C(A,-)$ 的表示对象唯一到唯一同构。

**练习 2.7.** 证明恒等函子 $\operatorname{id}_{\mathbf{Set}}:\mathbf{Set}\to\mathbf{Set}$ 可表，并找出泛元素。

**练习 2.8.** 证明常值双点函子 $\Delta 2:\mathbf{Set}\to\mathbf{Set}$ 不可表。

**练习 2.9.** 设 $P=y(A)\sqcup y(B)$。计算 $\operatorname{Nat}(P,Q)$。

**练习 2.10.** 解释为什么“每个有限维向量空间与其对偶同构”不推出函子 $(-)^*$ 与恒等函子自然同构。
