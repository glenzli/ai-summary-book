# 第一章：范畴、函子与自然变换

## 本章目标

本章建立普通范畴论的第一层语言：范畴、态射、函子、自然变换、同构、完全忠实函子和范畴等价。后续所有泛性质、极限、伴随和高阶推广都依赖这些定义。

## 依赖前置知识

需要熟悉集合、函数、二元运算、群、偏序集和基本代数结构。不预设读者已经学过范畴论。

## 1.1 范畴的定义

**定义 1.1.** 一个范畴（category）$\mathcal C$ 由以下数据组成：

1. 一个对象集合 $\operatorname{Ob}(\mathcal C)$。
2. 对任意对象 $X,Y\in\operatorname{Ob}(\mathcal C)$，一个集合
   $$
   \mathcal C(X,Y),
   $$
   其元素称为从 $X$ 到 $Y$ 的态射。
3. 对任意 $X,Y,Z\in\operatorname{Ob}(\mathcal C)$，一个复合映射
   $$
   \mathcal C(Y,Z)\times\mathcal C(X,Y)\longrightarrow\mathcal C(X,Z),
   \qquad (g,f)\longmapsto g\circ f.
   $$
4. 对任意对象 $X$，一个恒等态射
   $$
   \operatorname{id}_X\in\mathcal C(X,X).
   $$

这些数据满足以下公理：

- 结合律：若 $f:X\to Y$，$g:Y\to Z$，$h:Z\to W$，则
  $$
  h\circ(g\circ f)=(h\circ g)\circ f.
  $$
- 单位律：若 $f:X\to Y$，则
  $$
  f\circ\operatorname{id}_X=f,\qquad
  \operatorname{id}_Y\circ f=f.
  $$

**约定 1.2.** 若 $f\in\mathcal C(X,Y)$，写作 $f:X\to Y$。若不特别说明，本章所有范畴均为相对于固定 universe 的局部小范畴，即 $\mathcal C(X,Y)$ 是集合。

**例子 1.3.** 集合范畴 $\mathbf{Set}_{\mathcal U}$ 的对象是 $\mathcal U$-小集合，态射是函数，复合是函数复合，恒等态射是恒等函数。函数复合满足结合律，恒等函数满足单位律，因此这确实是范畴。

**例子 1.4.** 群范畴 $\mathbf{Grp}$ 的对象是群，态射是群同态。若 $f:G\to H$ 与 $g:H\to K$ 是群同态，则 $g\circ f:G\to K$ 仍是群同态，因为

$$
(g\circ f)(xy)=g(f(xy))=g(f(x)f(y))=g(f(x))g(f(y)).
$$

恒等函数是群同态，所以 $\mathbf{Grp}$ 是范畴。

**例子 1.5.** 任意偏序集 $(P,\leq)$ 给出一个范畴 $\mathcal P$：对象是 $P$ 的元素；若 $x\leq y$，则 $\mathcal P(x,y)$ 有唯一元素；若 $x\nleq y$，则 $\mathcal P(x,y)=\varnothing$。复合由传递性 $x\leq y\leq z\Rightarrow x\leq z$ 给出，恒等态射由自反性 $x\leq x$ 给出。由偏序集得到的范畴称为薄范畴（thin category）。

**例子 1.6.** 任意幺半群 $(M,\cdot,e)$ 给出一个只有一个对象 $*$ 的范畴 $\mathcal C_M$，令

$$
\mathcal C_M(*,*)=M,
$$

复合由 $M$ 的乘法给出，恒等态射为 $e$。结合律和单位律正是幺半群公理。反过来，任意只有一个对象的范畴给出一个幺半群。

## 1.2 反范畴

**定义 1.7.** 给定范畴 $\mathcal C$，其反范畴（opposite category）$\mathcal C^{\operatorname{op}}$ 定义如下：

- $\operatorname{Ob}(\mathcal C^{\operatorname{op}})=\operatorname{Ob}(\mathcal C)$。
- 对任意对象 $X,Y$，
  $$
  \mathcal C^{\operatorname{op}}(X,Y)=\mathcal C(Y,X).
  $$
- 若 $f:X\to Y$ 与 $g:Y\to Z$ 是 $\mathcal C^{\operatorname{op}}$ 中的态射，则它们在 $\mathcal C$ 中分别是 $f:Y\to X$ 与 $g:Z\to Y$，定义
  $$
  g\circ_{\operatorname{op}} f=f\circ_{\mathcal C}g.
  $$
- 恒等态射与 $\mathcal C$ 中相同。

**命题 1.8.** $\mathcal C^{\operatorname{op}}$ 是范畴，且

$$
(\mathcal C^{\operatorname{op}})^{\operatorname{op}}=\mathcal C
$$

作为范畴严格相等，若我们按上述数据定义反范畴。

**证明.** 只需验证结合律和单位律。设在 $\mathcal C^{\operatorname{op}}$ 中有

$$
X\xrightarrow{f}Y\xrightarrow{g}Z\xrightarrow{h}W.
$$

这些态射在 $\mathcal C$ 中分别是

$$
Y\xrightarrow{f}X,\qquad Z\xrightarrow{g}Y,\qquad W\xrightarrow{h}Z.
$$

于是

$$
h\circ_{\operatorname{op}}(g\circ_{\operatorname{op}}f)
=(g\circ_{\operatorname{op}}f)\circ_{\mathcal C}h
=(f\circ_{\mathcal C}g)\circ_{\mathcal C}h,
$$

而

$$
(h\circ_{\operatorname{op}}g)\circ_{\operatorname{op}}f
=f\circ_{\mathcal C}(g\circ_{\mathcal C}h).
$$

二者由 $\mathcal C$ 中复合的结合律相等。单位律同理由 $\mathcal C$ 中的单位律得到。再次取反会把 Hom 的方向改回，复合次序也改回，因此得到原范畴。$\square$

## 1.3 函子

**定义 1.9.** 设 $\mathcal C,\mathcal D$ 为范畴。一个函子（functor）

$$
F:\mathcal C\to\mathcal D
$$

由以下数据组成：

1. 对每个对象 $X\in\mathcal C$，给出对象 $F(X)\in\mathcal D$。
2. 对每个态射 $f:X\to Y$，给出态射
   $$
   F(f):F(X)\to F(Y).
   $$

这些数据满足：

- $F(\operatorname{id}_X)=\operatorname{id}_{F(X)}$。
- 若 $X\xrightarrow{f}Y\xrightarrow{g}Z$，则
  $$
  F(g\circ f)=F(g)\circ F(f).
  $$

**定义 1.10.** 从 $\mathcal C$ 到 $\mathcal D$ 的反变函子（contravariant functor）是函子

$$
\mathcal C^{\operatorname{op}}\to\mathcal D.
$$

等价地，它把态射 $f:X\to Y$ 送到 $F(f):F(Y)\to F(X)$，并满足反向的复合公式

$$
F(g\circ f)=F(f)\circ F(g).
$$

**例子 1.11.** 忘却函子

$$
U:\mathbf{Grp}\to\mathbf{Set}_{\mathcal U}
$$

把群送到底层集合，把群同态送到底层函数。它保持恒等态射和复合，因此是函子。

**例子 1.12.** 对固定范畴 $\mathcal C$ 和对象 $A\in\mathcal C$，若 $\mathcal C$ 局部小，则

$$
\mathcal C(A,-):\mathcal C\to\mathbf{Set}_{\mathcal U}
$$

是协变函子：对象 $X$ 被送到集合 $\mathcal C(A,X)$；态射 $f:X\to Y$ 给出函数

$$
\mathcal C(A,f):\mathcal C(A,X)\to\mathcal C(A,Y),
\qquad u\mapsto f\circ u.
$$

恒等态射和复合保持性分别来自范畴的单位律和结合律。

**例子 1.13.** 同样，对固定对象 $A$，

$$
\mathcal C(-,A):\mathcal C^{\operatorname{op}}\to\mathbf{Set}_{\mathcal U}
$$

是反变函子。态射 $f:X\to Y$ 在原范畴中给出函数

$$
\mathcal C(Y,A)\to\mathcal C(X,A),
\qquad v\mapsto v\circ f.
$$

## 1.4 自然变换

**定义 1.14.** 设 $F,G:\mathcal C\to\mathcal D$ 是函子。一个自然变换（natural transformation）

$$
\alpha:F\Rightarrow G
$$

是对每个对象 $X\in\mathcal C$ 给出态射

$$
\alpha_X:F(X)\to G(X)
$$

使得对每个态射 $f:X\to Y$，下列等式在 $\mathcal D$ 中成立：

$$
G(f)\circ\alpha_X=\alpha_Y\circ F(f).
$$

这称为自然性条件。等价地，下图交换：

$$
\begin{matrix}
F(X)&\xrightarrow{F(f)}&F(Y)\\
\downarrow{\alpha_X}&&\downarrow{\alpha_Y}\\
G(X)&\xrightarrow{G(f)}&G(Y).
\end{matrix}
$$

**例子 1.15.** 设 $\mathbf{Vect}_k$ 为域 $k$ 上向量空间范畴。双对偶给出函子

$$
(-)^{**}:\mathbf{Vect}_k\to\mathbf{Vect}_k.
$$

对每个向量空间 $V$，定义

$$
\eta_V:V\to V^{**},\qquad
\eta_V(v)(\lambda)=\lambda(v)
$$

其中 $\lambda\in V^*$。若 $f:V\to W$ 是线性映射，则自然性要求

$$
f^{**}\circ\eta_V=\eta_W\circ f.
$$

对 $v\in V$ 和 $\mu\in W^*$ 计算：

$$
\bigl(f^{**}(\eta_V(v))\bigr)(\mu)=\eta_V(v)(\mu\circ f)=\mu(f(v)),
$$

而

$$
\eta_W(f(v))(\mu)=\mu(f(v)).
$$

两者相等，因此 $\eta:\operatorname{id}\Rightarrow(-)^{**}$ 是自然变换。

**定义 1.16.** 若 $\alpha:F\Rightarrow G$ 与 $\beta:G\Rightarrow H$ 是自然变换，则其纵向复合（vertical composition）$\beta\alpha:F\Rightarrow H$ 定义为

$$
(\beta\alpha)_X=\beta_X\circ\alpha_X.
$$

**命题 1.17.** 纵向复合后的族 $\beta\alpha$ 是自然变换。

**证明.** 对任意 $f:X\to Y$，由 $\alpha$ 的自然性有

$$
G(f)\circ\alpha_X=\alpha_Y\circ F(f),
$$

由 $\beta$ 的自然性有

$$
H(f)\circ\beta_X=\beta_Y\circ G(f).
$$

于是

$$
H(f)\circ(\beta_X\circ\alpha_X)
=(H(f)\circ\beta_X)\circ\alpha_X
=(\beta_Y\circ G(f))\circ\alpha_X
=\beta_Y\circ(G(f)\circ\alpha_X)
=\beta_Y\circ(\alpha_Y\circ F(f))
=(\beta_Y\circ\alpha_Y)\circ F(f).
$$

这正是 $\beta\alpha$ 的自然性。$\square$

**定义 1.18.** 若 $F,G:\mathcal C\to\mathcal D$ 是函子，所有自然变换 $F\Rightarrow G$ 的集合记为

$$
\operatorname{Nat}(F,G).
$$

若 $\mathcal C$ 小且 $\mathcal D$ 局部小，则函子和自然变换组成函子范畴

$$
\operatorname{Fun}(\mathcal C,\mathcal D).
$$

其对象是函子 $\mathcal C\to\mathcal D$，态射是自然变换，复合为纵向复合。

## 1.5 同构、忠实、完全与本质满

**定义 1.19.** 范畴 $\mathcal C$ 中的态射 $f:X\to Y$ 称为同构，若存在态射 $g:Y\to X$ 使得

$$
g\circ f=\operatorname{id}_X,\qquad
f\circ g=\operatorname{id}_Y.
$$

此时 $g$ 称为 $f$ 的逆，记作 $f^{-1}$。

**命题 1.20.** 同构的逆唯一。

**证明.** 设 $g,h:Y\to X$ 都是 $f:X\to Y$ 的逆。则

$$
g=g\circ\operatorname{id}_Y
=g\circ(f\circ h)
=(g\circ f)\circ h
=\operatorname{id}_X\circ h
=h.
$$

所以逆唯一。$\square$

**定义 1.21.** 设 $F:\mathcal C\to\mathcal D$ 为函子。

- $F$ 称为忠实（faithful），若对任意 $X,Y\in\mathcal C$，函数
  $$
  F_{X,Y}:\mathcal C(X,Y)\to\mathcal D(FX,FY)
  $$
  是单射。
- $F$ 称为完全（full），若每个 $F_{X,Y}$ 是满射。
- $F$ 称为完全忠实（fully faithful），若每个 $F_{X,Y}$ 是双射。
- $F$ 称为本质满（essentially surjective），若对任意 $D\in\mathcal D$，存在 $C\in\mathcal C$ 和同构 $F(C)\cong D$。

**例子 1.22.** 包含函子

$$
\mathbf{Ab}\hookrightarrow\mathbf{Grp}
$$

是忠实且完全的：阿贝尔群之间的群同态正是阿贝尔群同态。但它不是本质满的，因为非阿贝尔群不与任何阿贝尔群同构。

## 1.6 范畴等价

**定义 1.23.** 函子 $F:\mathcal C\to\mathcal D$ 称为范畴等价（equivalence of categories），若 $F$ 完全忠实且本质满。两个范畴 $\mathcal C,\mathcal D$ 称为等价，若存在一个范畴等价 $F:\mathcal C\to\mathcal D$。

此定义把“同一个数学结构的不同呈现”拆成两个可检查条件：完全忠实说明 $F$ 不改变任意两个对象之间的态射集合；本质满说明 $\mathcal D$ 中每个对象都与某个 $F(C)$ 同构。

**定理 1.24.** 假设允许对每个同构类选择代表和同构。若 $F:\mathcal C\to\mathcal D$ 完全忠实且本质满，则存在函子

$$
G:\mathcal D\to\mathcal C
$$

以及自然同构

$$
FG\cong\operatorname{id}_{\mathcal D},\qquad
\operatorname{id}_{\mathcal C}\cong GF.
$$

因此，本章的定义推出许多教材中使用的“存在拟逆函子”的表述。

**证明.** 设 $F$ 完全忠实且本质满。对每个 $D\in\mathcal D$，选择对象 $G(D)\in\mathcal C$ 和同构

$$
\varepsilon_D:F(GD)\xrightarrow{\cong}D.
$$

对态射 $u:D\to D'$，考虑复合

$$
F(GD)\xrightarrow{\varepsilon_D}D\xrightarrow{u}D'\xrightarrow{\varepsilon_{D'}^{-1}}F(GD').
$$

由于 $F$ 完全忠实，存在唯一态射

$$
G(u):G(D)\to G(D')
$$

使得

$$
F(G(u))=\varepsilon_{D'}^{-1}\circ u\circ\varepsilon_D.
$$

唯一性直接推出 $G(\operatorname{id}_D)=\operatorname{id}_{G(D)}$ 和 $G(v\circ u)=G(v)\circ G(u)$，所以 $G:\mathcal D\to\mathcal C$ 是函子。族 $\varepsilon_D$ 按构造满足自然性，因此给出自然同构

$$
\varepsilon:FG\Rightarrow\operatorname{id}_{\mathcal D}.
$$

还需构造 $\eta:\operatorname{id}_{\mathcal C}\Rightarrow GF$。对 $C\in\mathcal C$，对象 $F(GF C)$ 与 $F(C)$ 之间有同构

$$
\varepsilon_{FC}:F(GF C)\to F(C).
$$

由完全性，存在态射

$$
\eta_C:C\to GF C
$$

使得

$$
F(\eta_C)=\varepsilon_{FC}^{-1}.
$$

再由完全性，存在态射 $\xi_C:GF C\to C$ 使得

$$
F(\xi_C)=\varepsilon_{FC}.
$$

于是

$$
F(\xi_C\circ\eta_C)=\varepsilon_{FC}\circ\varepsilon_{FC}^{-1}
=\operatorname{id}_{F C}
=F(\operatorname{id}_C),
$$

且

$$
F(\eta_C\circ\xi_C)=\varepsilon_{FC}^{-1}\circ\varepsilon_{FC}
=\operatorname{id}_{F G F C}
=F(\operatorname{id}_{G F C}).
$$

由于 $F$ 忠实，得到

$$
\xi_C\circ\eta_C=\operatorname{id}_C,\qquad
\eta_C\circ\xi_C=\operatorname{id}_{G F C}.
$$

所以 $\eta_C$ 是同构。

用完全忠实性再次检验自然性：对 $f:C\to C'$，自然性等式

$$
GF(f)\circ\eta_C=\eta_{C'}\circ f
$$

在施加 $F$ 后化为由 $\varepsilon$ 自然性给出的同一等式；因 $F$ 忠实，原等式成立。因此 $\eta$ 是自然同构，$F$ 与 $G$ 给出范畴等价。$\square$

**注 1.25.** 反方向，即存在拟逆函子和自然同构推出完全忠实且本质满，也是标准事实。本书将在伴随函子章节用伴随等价的三角恒等式给出更结构化的证明，避免在第一章提前引入尚未定义的伴随语言。

## 1.7 骨架与边界例子

**定义 1.26.** 范畴 $\mathcal C$ 的一个骨架（skeleton）是全子范畴 $\mathcal S\subseteq\mathcal C$，使得：

1. $\mathcal C$ 中每个对象都同构于 $\mathcal S$ 中某个对象；
2. $\mathcal S$ 中任意两个不同对象在 $\mathcal C$ 中不同构。

**命题 1.27.** 若 $\mathcal S$ 是 $\mathcal C$ 的骨架，则包含函子

$$
I:\mathcal S\hookrightarrow\mathcal C
$$

是范畴等价。

**证明.** 因为 $\mathcal S$ 是全子范畴，任意 $A,B\in\mathcal S$ 上的 Hom 映射

$$
\mathcal S(A,B)\to\mathcal C(IA,IB)
$$

是恒等双射，所以 $I$ 完全忠实。又由骨架定义，任意 $C\in\mathcal C$ 都同构于某个 $I(S)$，故 $I$ 本质满。因此 $I$ 是范畴等价。$\square$

**例子 1.28.** 有限集范畴 $\mathbf{FinSet}$ 的一个骨架由集合

$$
\{0,\dots,n-1\},\qquad n\ge 0
$$

组成。每个有限集同构于唯一基数的标准有限集。因此该骨架等价于 $\mathbf{FinSet}$，但通常不严格等于 $\mathbf{FinSet}$。

**例子 1.29（边界条件）.** 忠实、完全和本质满相互独立，不能随意替换。

1. 包含 $\mathbf{Ab}\hookrightarrow\mathbf{Grp}$ 完全忠实但非本质满。
2. 忘却函子 $\mathbf{Grp}\to\mathbf{Set}$ 忠实但不完全，因为任意集合函数通常不是群同态。
3. 从离散单对象范畴 $*$ 到单对象群范畴 $BG$ 的唯一函子本质满且忠实，但当 $G$ 非平凡时不完全，因为
   $$
   *(*,*)=\{\operatorname{id}\}\to BG(*,*)=G
   $$
   不是满射。

这些例子说明范畴等价必须同时控制对象和 Hom 集：只控制对象不够，只控制态射也不够。

**命题 1.30.** 若 $F:\mathcal C\to\mathcal D$ 是范畴等价，则 $F$ 反映同构：若 $F(f)$ 是 $\mathcal D$ 中同构，则 $f$ 是 $\mathcal C$ 中同构。

**证明.** 设 $f:X\to Y$ 且 $F(f)$ 有逆 $u:FY\to FX$。由于 $F$ 完全，存在 $g:Y\to X$ 使 $F(g)=u$。于是

$$
F(gf)=F(g)F(f)=uF(f)=\operatorname{id}_{FX}=F(\operatorname{id}_X),
$$

由 $F$ 忠实得 $gf=\operatorname{id}_X$。同理

$$
F(fg)=F(f)u=\operatorname{id}_{FY}=F(\operatorname{id}_Y),
$$

故 $fg=\operatorname{id}_Y$。所以 $f$ 是同构。$\square$

## 1.8 本章小结

范畴由对象、态射、复合和恒等态射组成；函子保持这些结构；自然变换比较两个函子并要求对所有态射满足同一个交换方块。范畴等价不是范畴同构，而是完全忠实且本质满的函子所表达的“相同数学内容”。后续 Yoneda 引理、极限和伴随都将把这些定义作为基本语言。

## 练习

**练习 1.1.** 验证环和环同态构成范畴。明确说明你采用的环是否要求有单位，以及环同态是否保持单位。

**练习 1.2.** 设 $M$ 为幺半群。证明由 $M$ 构成的单对象范畴 $\mathcal C_M$ 中的同构态射正好是 $M$ 中的可逆元素。

**练习 1.3.** 设 $(P,\leq)$ 是偏序集。证明其对应薄范畴中的两个对象同构当且仅当它们在偏序集中相等。

**练习 1.4.** 对固定对象 $A\in\mathcal C$，详细验证 $\mathcal C(A,-)$ 是函子。

**练习 1.5.** 设 $F,G,H:\mathcal C\to\mathcal D$，$\alpha:F\Rightarrow G$，$\beta:G\Rightarrow H$。验证纵向复合满足结合律，并说明恒等自然变换是什么。

**练习 1.6.** 给出一个完全忠实但非本质满的函子。

**练习 1.7.** 证明若 $F:\mathcal C\to\mathcal D$ 与 $G:\mathcal D\to\mathcal E$ 都完全忠实，则 $GF$ 完全忠实。

**练习 1.8.** 构造一个范畴等价但不是范畴同构的例子，并指出它为什么不可能是严格同构。

**练习 1.9.** 证明任意范畴若存在骨架，则该骨架在等价意义下唯一。

**练习 1.10.** 证明完全忠实函子反映同构。

**练习 1.11.** 给出一个本质满但不忠实的函子。

**练习 1.12.** 设 $G$ 为非平凡群。把单对象群范畴 $BG$ 与终范畴 $*$ 比较，说明唯一函子 $BG\to *$ 是否完全、忠实、本质满。
