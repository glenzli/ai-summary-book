# 第二章：泛性质与 Yoneda 引理

许多数学对象不是由一份坐标或元素清单定义，而是由“所有映入它或从它映出的态射”唯一刻画。终对象、自由对象和张量积都体现这种泛性质；Yoneda 引理则把这一直觉精确化为自然变换集合与对象取值之间的双射，并说明一个对象在可表预层中的像保留全部态射信息。本章从始终对象和泛元素逐步走到 Yoneda 嵌入，特别强调唯一性总是相对于指定自然性变量而言。

这里沿用第一章的范畴、反范畴、函子与自然变换。$\mathcal C$ 默认局部 $\mathcal U$-小；当使用 $\widehat{\mathcal C}$ 或要求自然变换全体为 $\mathcal U$-小集合时，再明确假设 $\mathcal C$ 为 $\mathcal U$-小。这样 Yoneda 公式的两端始终位于已声明的 universe 中。

## 2.1 终对象与始对象

**定义 2.1.** 设 $\mathcal C$ 为局部 $\mathcal U$-小范畴。对象 $1\in\mathcal C$ 称为终对象（terminal object），若对任意 $X\in\mathcal C$，集合 $\mathcal C(X,1)$ 恰有一个元素。

对象 $0\in\mathcal C$ 称为始对象（initial object），若对任意 $X\in\mathcal C$，集合 $\mathcal C(0,X)$ 恰有一个元素。

**命题 2.2.** 若 $1$ 与 $1'$ 都是 $\mathcal C$ 的终对象，则存在唯一同构 $1\cong 1'$。

**证明.** 由 $1'$ 终，存在唯一态射 $u:1\to 1'$；由 $1$ 终，存在唯一态射 $v:1'\to 1$。复合 $v\circ u:1\to 1$ 必须等于唯一的态射 $1\to 1$，即 $\operatorname{id}_1$。同理 $u\circ v=\operatorname{id}_{1'}$。故 $u$ 是同构。若 $w:1\to1'$ 也是同构，则它首先是态射 $1\to1'$，由 $1'$ 的终性等于 $u$。$\square$

**例子 2.3.** 在 $\mathbf{Set}_{\mathcal U}$ 中，任意单点集合是终对象，空集是始对象。在 $\mathbf{Grp}$ 中，平凡群既是终对象又是始对象。

## 2.2 泛元素与可表函子

**定义 2.4.** 设 $\mathcal C$ 局部 $\mathcal U$-小，且
$F:\mathcal C\to\mathbf{Set}_{\mathcal U}$。$F$ 的一个表示（representation）是一个对象 $A\in\mathcal C$ 和一个自然同构

$$
\theta:\mathcal C(A,-)\xrightarrow{\cong}F.
$$

若这样的表示存在，称 $F$ 可表（representable）。

这里 $\theta$ 对 $X\in\mathcal C$ 自然；其分量类型为

$$
\theta_X:\mathcal C(A,X)\longrightarrow F(X).
$$

反变函子 $P:\mathcal C^{\operatorname{op}}\to\mathbf{Set}_{\mathcal U}$ 的表示是对象 $A$ 与自然同构

$$
\mathcal C(-,A)\xrightarrow{\cong}P.
$$

此时分量

$$
\theta_X:\mathcal C(X,A)\longrightarrow P(X)
$$

对 $X\in\mathcal C^{\operatorname{op}}$ 自然。两种表示的 Hom 方向相反，不能只凭同一个“可表”记号省略。

**定义 2.5.** 设 $F:\mathcal C\to\mathbf{Set}_{\mathcal U}$。一个协变泛元素（covariant universal element）是二元组 $(A,u)$，其中 $A\in\mathcal C$ 且 $u\in F(A)$，使得对任意 $X\in\mathcal C$ 和任意 $x\in F(X)$，存在唯一态射 $f:A\to X$ 满足

$$
F(f)(u)=x.
$$

若 $P:\mathcal C^{\operatorname{op}}\to\mathbf{Set}_{\mathcal U}$，则一个反变泛元素是二元组 $(A,u)$，其中 $u\in P(A)$，且对任意 $X\in\mathcal C$ 和 $x\in P(X)$，存在唯一态射 $f:X\to A$ 满足

$$
P(f)(u)=x.
$$

协变情形的泛箭头从 $A$ 指向测试对象；反变情形的泛箭头从测试对象指向 $A$。

**命题 2.6.** 函子 $F:\mathcal C\to\mathbf{Set}_{\mathcal U}$ 可表，当且仅当 $F$ 有协变泛元素。

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

为看清唯一性的类型，定义协变元素范畴
$\operatorname{El}(F)$：对象是 $(X,x)$，态射

$$
f:(X,x)\to(Y,y)
$$

是满足 $F(f)(x)=y$ 的态射 $f:X\to Y$。定义 2.5 恰好说
$(A,u)$ 是 $\operatorname{El}(F)$ 的始对象。对反变
$P$，取态射 $f:(X,x)\to(Y,y)$ 满足 $P(f)(y)=x$，则反变泛元素是
$\operatorname{El}(P)$ 的终对象。因此“表示对象唯一”精确地表示：
两个带泛元素的表示之间存在唯一保持泛元素的同构；它不声称两个底层对象之间只有一个同构。

## 2.3 Yoneda 引理

**定理 2.7（Yoneda 引理）.** 设 $\mathcal C$ 为 $\mathcal U$-小范畴，$F:\mathcal C^{\operatorname{op}}\to\mathbf{Set}_{\mathcal U}$ 为预层，$A\in\mathcal C$。存在 $\mathbf{Set}_{\mathcal U}$ 中的双射

$$
\operatorname{Nat}(\mathcal C(-,A),F)\cong F(A).
$$

该双射把自然变换 $\alpha:\mathcal C(-,A)\Rightarrow F$ 送到

$$
\alpha_A(\operatorname{id}_A)\in F(A).
$$

记该映射为 $Y_{A,F}$。它同时对 $A$ 和 $F$ 自然，具体是：

1. 若 $h:A\to B$ 且 $\alpha:yB\Rightarrow F$，则
   $$
   Y_{A,F}\bigl(\alpha\circ y(h)\bigr)
   =F(h)\bigl(Y_{B,F}(\alpha)\bigr).
   $$
   两边都是 $F(A)$ 的元素；这里 $A$ 变量是反变的。
2. 若 $\sigma:F\Rightarrow G$ 且 $\alpha:yA\Rightarrow F$，则
   $$
   Y_{A,G}(\sigma\circ\alpha)
   =\sigma_A\bigl(Y_{A,F}(\alpha)\bigr).
   $$
   这里预层变量是协变的。

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

所以 $\alpha=\alpha^a$。反向复合也满足

$$
Y_{A,F}(\alpha^a)=\alpha^a_A(\operatorname{id}_A)
=F(\operatorname{id}_A)(a)=a,
$$

故两个构造互逆。

还需验证两个自然性变量。若 $h:A\to B$，则
$y(h):yA\Rightarrow yB$ 的 $X$ 分量把
$f:X\to A$ 送到 $h f:X\to B$。因此

$$
\begin{aligned}
Y_{A,F}(\alpha\circ y(h))
&=(\alpha\circ y(h))_A(\operatorname{id}_A)\\
&=\alpha_A(h)\\
&=F(h)\bigl(\alpha_B(\operatorname{id}_B)\bigr)\\
&=F(h)\bigl(Y_{B,F}(\alpha)\bigr),
\end{aligned}
$$

其中第三个等号是反变自然变换 $\alpha:yB\Rightarrow F$ 对
$h:A\to B$ 的自然性。若 $\sigma:F\Rightarrow G$，则

$$
Y_{A,G}(\sigma\alpha)
=(\sigma\alpha)_A(\operatorname{id}_A)
=\sigma_A(\alpha_A(\operatorname{id}_A))
=\sigma_A(Y_{A,F}(\alpha)).
$$

所以双射对两个变量都自然。$\square$

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

**证明.** 由推论 2.8，给定的自然同构唯一对应于
$\mathcal C(A,B)$ 中的态射 $f:A\to B$。其逆自然变换对应于
$g:B\to A$。自然变换复合对应态射复合，故
$g\circ f=\operatorname{id}_A$ 且
$f\circ g=\operatorname{id}_B$。所以由给定自然同构诱导的
$f$ 是同构且唯一；若改变自然同构，可能得到不同的底层同构。$\square$

**例子 2.10.** 若一个 $\mathcal U$-小集合 $S$ 表示函子

$$
\mathbf{Set}_{\mathcal U}(A,-):
\mathbf{Set}_{\mathcal U}\to\mathbf{Set}_{\mathcal U},
$$

则 $S$ 与 $A$ 同构。这里“表示对象唯一”不是集合元素层面的猜测，而是命题 2.9 的特例。

## 2.5 可表性、方差与自然性边界

**命题 2.11.** 若
$F:\mathcal C\to\mathbf{Set}_{\mathcal U}$ 被 $(A,u)$ 表示，且
$F$ 也被 $(B,v)$ 表示，则存在唯一同构 $\phi:A\to B$ 使得

$$
F(\phi)(u)=v.
$$

**证明.** 由 $(A,u)$ 的泛性，应用于 $v\in F(B)$，存在唯一 $\phi:A\to B$ 使 $F(\phi)(u)=v$。由 $(B,v)$ 的泛性，应用于 $u\in F(A)$，存在唯一 $\psi:B\to A$ 使 $F(\psi)(v)=u$。于是

$$
F(\psi\phi)(u)=F(\psi)(F(\phi)(u))=F(\psi)(v)=u.
$$

而 $\operatorname{id}_A$ 也满足 $F(\operatorname{id}_A)(u)=u$。由 $(A,u)$ 的唯一性，$\psi\phi=\operatorname{id}_A$。同理 $\phi\psi=\operatorname{id}_B$。故 $\phi$ 是同构。若 $\phi'$ 也满足 $F(\phi')(u)=v$，由 $(A,u)$ 的唯一性 $\phi'=\phi$。$\square$

**例子 2.12（唯一的是保持泛元素的同构）.** 令

$$
F:\mathbf{Set}_{\mathcal U}\to\mathbf{Set}_{\mathcal U},
\qquad F(X)=X\times X.
$$

双点集 $2=\{0,1\}$ 与 $u=(0,1)\in F(2)$ 给出泛元素：对
$(x_0,x_1)\in X\times X$，唯一的 $f:2\to X$ 由
$f(0)=x_0,f(1)=x_1$ 决定。底层集合 $2$ 有非平凡自同构
$s$，它交换 $0,1$；但

$$
F(s)(u)=(1,0)\ne(0,1)=u.
$$

所以 $s$ 不是表示 $(2,u)$ 的自同构。命题 2.11 的唯一性发生在
“对象连同泛元素”的范畴中，而不是在忘掉泛元素后的
$\mathbf{Set}_{\mathcal U}$ 中。

**例子 2.13（不可表函子）.** 并非每个集合值函子都可表。考虑幂集函子

$$
\mathcal P:\mathbf{Set}_{\mathcal U}\to\mathbf{Set}_{\mathcal U},
\qquad X\mapsto\mathcal P(X)
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

**例子 2.14（先检查方差）.** 在有限维向量空间范畴
$\mathbf{Vect}^{\mathrm{fd}}_k$ 上，对偶构造的类型是

$$
D=(-)^*:
(\mathbf{Vect}^{\mathrm{fd}}_k)^{\operatorname{op}}
\longrightarrow\mathbf{Vect}^{\mathrm{fd}}_k,
$$

而恒等函子的类型是

$$
\operatorname{id}:\mathbf{Vect}^{\mathrm{fd}}_k
\longrightarrow\mathbf{Vect}^{\mathrm{fd}}_k.
$$

二者源范畴不同，所以“$\operatorname{id}\Rightarrow D$”不是一个
已定义的自然变换类型；这比“不自然”更早失败。可以合法比较的是
$\operatorname{id}$ 与协变双对偶函子 $D^2=(-)^{**}$。评价映射

$$
\iota_V:V\to V^{**},\qquad
\iota_V(v)(\lambda)=\lambda(v)
$$

组成自然变换；在有限维子范畴上每个 $\iota_V$ 都是同构。

**例子 2.15（逐对象同构不推出自然同构）.** 设
$k$ 为特征不等于 $2$ 的 $\mathcal U$-小域，令 $BC_2$ 为循环群
$C_2=\{1,s\}$ 对应的 $\mathcal U$-小单对象范畴。定义两个函子

$$
R_{\mathrm{triv}},R_{\mathrm{sgn}}:BC_2\to\mathbf{Vect}_k
$$

都把唯一对象送到一维空间 $k$，但令

$$
R_{\mathrm{triv}}(s)=\operatorname{id}_k,
\qquad
R_{\mathrm{sgn}}(s)=-\operatorname{id}_k.
$$

两个对象值当然同构。若
$\alpha:R_{\mathrm{triv}}\Rightarrow R_{\mathrm{sgn}}$ 是自然变换，则唯一分量
$\alpha_*:k\to k$ 对态射 $s$ 的自然性要求

$$
(-\operatorname{id}_k)\alpha_*
=\alpha_*\operatorname{id}_k,
$$

即 $2\alpha_*=0$。由特征假设，$\alpha_*=0$，不可能可逆。因此
$R_{\mathrm{triv}}$ 与 $R_{\mathrm{sgn}}$ 逐对象同构，却不自然同构。失败点正是群作用态射上的自然性方块。

## 2.6 由全部映射重建对象

泛性质不是非正式描述，而是可表性或某个结构范畴中的始、终对象。Yoneda 引理说明：预层 $F$ 在对象 $A$ 上的元素，等价于从可表预层 $\mathcal C(-,A)$ 到 $F$ 的自然变换；该双射在对象变量上反变、在预层变量上协变。表示对象的唯一性只针对保持泛元素的同构。逐对象同构既不能修复方差不匹配，也不能替代自然性方块。

## 练习

**练习 2.1.** 证明始对象若存在，则在唯一同构意义下唯一。

**练习 2.2.** 写出 $\mathbf{Set}_{\mathcal U}$ 中二元积 $A\times B$ 的泛性质，并把它改写为某个函子的表示。

**练习 2.3.** 设 $M$ 为幺半群，视为单对象范畴。描述该范畴上的一个集合值预层等价于什么代数结构。

**练习 2.4.** 对协变函子版本证明 Yoneda 引理：
$$
\operatorname{Nat}(\mathcal C(A,-),F)\cong F(A).
$$

**练习 2.5.** 证明 Yoneda 嵌入反映同构：若 $y(f)$ 是预层同构，则 $f$ 是 $\mathcal C$ 中的同构。

**练习 2.6.** 证明协变可表函子 $\mathcal C(A,-)$ 的表示对象唯一到唯一同构。

**练习 2.7.** 证明恒等函子
$\operatorname{id}_{\mathbf{Set}_{\mathcal U}}$ 可表，并找出泛元素。

**练习 2.8.** 证明常值双点函子
$\Delta 2:\mathbf{Set}_{\mathcal U}\to\mathbf{Set}_{\mathcal U}$ 不可表。

**练习 2.9.** 设 $\mathcal C$ 为 $\mathcal U$-小范畴，$Q\in\widehat{\mathcal C}$。直接定义预层
$$
P(X)=\mathcal C(X,A)\sqcup\mathcal C(X,B),
$$
其在态射上的作用在两个余并分支中分别由预复合给出。只用集合余并的泛性质和 Yoneda 引理计算 $\operatorname{Nat}(P,Q)$。

**练习 2.10.** 写出 $(-)^*$、$(-)^{**}$ 和恒等函子的定义域和值域。说明为什么第一章的评价映射给出
$\operatorname{id}\Rightarrow(-)^{**}$，而表达式
$\operatorname{id}\Rightarrow(-)^*$ 在全体线性映射组成的范畴上没有自然变换的类型。
