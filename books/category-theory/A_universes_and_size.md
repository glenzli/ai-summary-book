# 附录 A：集合论宇宙与大小问题

## 本章目标

本附录固定本书使用的 universe 口径，并逐项确定集合范畴、函子范畴、自然变换集合、预层范畴、锥范畴和逗号范畴所在的层级。大小是对象类型的一部分；“所有小图形”中的“小”必须相对于已经声明的 universe 理解。

## A.1 Grothendieck universe 与元理论地位

**定义 A.1.** Grothendieck universe $\mathcal U$ 是一个非空集合，满足：

1. 若 $x\in\mathcal U$ 且 $y\in x$，则 $y\in\mathcal U$；
2. 若 $x,y\in\mathcal U$，则 $\{x,y\}\in\mathcal U$；
3. 若 $x\in\mathcal U$，则幂集 $\mathcal P(x)\in\mathcal U$；
4. 若 $I\in\mathcal U$ 且 $(x_i)_{i\in I}$ 是一族 $\mathcal U$ 的元素，则
   $$
   \bigcup_{i\in I}x_i\in\mathcal U.
   $$

一个集合称为 $\mathcal U$-小，若它与 $\mathcal U$ 的某个元素双射。选定具体代表时，本书通常直接取该集合属于 $\mathcal U$。

这些公理推出 $\mathcal U$ 对有限积、$\mathcal U$-小余并、子集、商集和函数集闭合。例如若 $A,B\in\mathcal U$，则

$$
B^A\subseteq\mathcal P(A\times B),
$$

故函数集 $B^A$ 属于 $\mathcal U$。

**元理论假设 A.2.** 本书固定三个嵌套的 Grothendieck universes

$$
\mathcal U\in\mathcal V\in\mathcal W.
$$

由传递性有 $\mathcal U\subseteq\mathcal V\subseteq\mathcal W$。三个 universe 的存在是本书采用的元理论假设，不是书内定理；在通常的 ZFC 实现中，它需要相应的大基数假设。正文只使用闭包性质，不尝试证明这些 universe 存在。

## A.2 范畴的大小轮廓

**定义 A.3.** 设 $\mathcal C$ 为普通范畴。

1. $\mathcal C$ 称为 $\mathcal U$-小，若对象集 $\operatorname{Ob}(\mathcal C)$ 与总态射集
   $$
   \operatorname{Mor}(\mathcal C)
   =\coprod_{X,Y\in\operatorname{Ob}(\mathcal C)}\mathcal C(X,Y)
   $$
   都是 $\mathcal U$-小集合。
2. $\mathcal C$ 称为局部 $\mathcal U$-小，若对任意对象 $X,Y$，Hom 类 $\mathcal C(X,Y)$ 是 $\mathcal U$-小集合。
3. $\mathcal C$ 称为本质 $\mathcal U$-小，若它等价于一个 $\mathcal U$-小范畴。

“大范畴”是相对说法。例如 $\mathbf{Set}_{\mathcal U}$ 相对于 $\mathcal U$ 是大范畴，但相对于 $\mathcal V$ 是小范畴。因此在需要推导大小结论时，本书不把“大的”单独当作类型声明。

**全书约定 A.4.** 在普通范畴论章节中，未另行声明的范畴具有以下默认轮廓：

$$
\operatorname{Ob}(\mathcal C),\operatorname{Mor}(\mathcal C)\text{ 是 }\mathcal V\text{-小的},
\qquad
\mathcal C(X,Y)\text{ 是 }\mathcal U\text{-小的}.
$$

也就是说，默认范畴是 $\mathcal V$-小且局部 $\mathcal U$-小的。“小范畴”单独出现时指 $\mathcal U$-小范畴；“所有小极限或余极限”指由 $\mathcal U$-小范畴索引的极限或余极限。若索引范畴只在 $\mathcal V$ 层小，正文必须写成“$\mathcal V$-小极限”而不能沿用默认说法。

## A.3 $\mathbf{Set}$、$\mathbf{Cat}$ 与函子范畴

**命题 A.5.** 集合范畴 $\mathbf{Set}_{\mathcal U}$ 是 $\mathcal V$-小且局部 $\mathcal U$-小的，但不是 $\mathcal U$-小的。所有 $\mathcal U$-小范畴和函子组成的范畴 $\mathbf{Cat}_{\mathcal U}$ 也具有同一大小轮廓。

**证明.** $\mathbf{Set}_{\mathcal U}$ 的对象集可取为 $\mathcal U$ 本身；由于 $\mathcal U\in\mathcal V$，它是 $\mathcal V$-小的。对 $A,B\in\mathcal U$，函数集 $B^A\in\mathcal U$，故 Hom 是 $\mathcal U$-小的。集合 $\mathcal U$ 不可能与任何 $x\in\mathcal U$ 双射：否则 $\mathcal P(x)\in\mathcal U$ 及传递性给出注入 $\mathcal P(x)\hookrightarrow\mathcal U\cong x$，与 Cantor 定理矛盾。因此 $\mathbf{Set}_{\mathcal U}$ 不是 $\mathcal U$-小的。

一个 $\mathcal U$-小范畴可由 $\mathcal U$ 中的对象集、态射集、源靶映射、恒等映射和复合映射编码，所以所有此类编码组成 $\mathcal V$-小集合。若 $\mathcal C,\mathcal D$ 都是 $\mathcal U$-小范畴，则函子 $\mathcal C\to\mathcal D$ 的全体是若干 $\mathcal U$-小函数集乘积中由函子性等式切出的子集，因而仍为 $\mathcal U$-小。故 $\mathbf{Cat}_{\mathcal U}$ 局部 $\mathcal U$-小。它不是 $\mathcal U$-小：离散范畴构造把对象集
$\mathcal U$ 注入 $\mathbf{Cat}_{\mathcal U}$ 的对象集；若后者与某个
$x\in\mathcal U$ 双射，就会有注入 $\mathcal U\hookrightarrow x$。而
$x\subseteq\mathcal U$，Cantor--Bernstein 会给出
$x\cong\mathcal U$，与上一段的 Cantor 论证矛盾。$\square$

**命题 A.6（自然变换的大小）.** 设 $\mathcal C$ 是 $\mathcal U$-小范畴，$\mathcal D$ 是 $\mathcal V$-小且局部 $\mathcal U$-小的范畴。若 $F,G:\mathcal C\to\mathcal D$，则

$$
\operatorname{Nat}(F,G)\in\mathbf{Set}_{\mathcal U}.
$$

因此 $\operatorname{Fun}(\mathcal C,\mathcal D)$ 是 $\mathcal V$-小且局部 $\mathcal U$-小的。若 $\mathcal C$ 仅为 $\mathcal V$-小，则同一论证通常只能推出 $\operatorname{Nat}(F,G)$ 为 $\mathcal V$-小，不能推出它是 $\mathcal U$-小的。

**证明.** 自然变换是乘积

$$
\prod_{C\in\operatorname{Ob}(\mathcal C)}
\mathcal D(F(C),G(C))
$$

中满足所有自然性等式的族。索引集和每个因子均为 $\mathcal U$-小，故该乘积及其子集为 $\mathcal U$-小。函子全体由从 $\mathcal U$-小数据到 $\mathcal V$-小数据的映射编码，组成 $\mathcal V$-小集合。若对象索引集只在 $\mathcal V$ 中小，相应乘积一般也只在 $\mathcal V$ 中小。$\square$

## A.4 预层、锥与逗号范畴

**命题 A.7（预层范畴的大小）.** 若 $\mathcal C$ 是 $\mathcal U$-小范畴，则

$$
\widehat{\mathcal C}
=\operatorname{Fun}(\mathcal C^{\operatorname{op}},\mathbf{Set}_{\mathcal U})
$$

是 $\mathcal V$-小且局部 $\mathcal U$-小的范畴，通常不是 $\mathcal U$-小的。

**证明.** 预层的对象值、态射值与函子性等式由 $\mathcal U$-小份数据组成，而可取值的全部 $\mathcal U$-小集合组成 $\mathcal U\in\mathcal V$，故全部预层的编码构成 $\mathcal V$-小集合。

若 $F,G$ 是预层，则 $\operatorname{Nat}(F,G)$ 是

$$
\prod_{C\in\operatorname{Ob}(\mathcal C)}
\mathbf{Set}_{\mathcal U}(F(C),G(C))
$$

中由自然性等式切出的子集，由命题 A.6 它是 $\mathcal U$-小的。当 $\mathcal C=*$ 时，$\widehat{\mathcal C}\simeq\mathbf{Set}_{\mathcal U}$；命题 A.5 表明后者不是 $\mathcal U$-小的，所以不能一般断言 $\widehat{\mathcal C}$ 为 $\mathcal U$-小。$\square$

**命题 A.8（常用索引范畴的大小）.** 设 $\mathcal J$ 与 $\mathcal C$ 为 $\mathcal U$-小范畴，$\mathcal D$ 局部 $\mathcal U$-小。

1. 对图形 $D:\mathcal J\to\mathcal D$，锥之间保持各腿的态射构成 $\mathcal U$-小 Hom 集；若允许锥顶点遍历一个 $\mathcal V$-小范畴，则锥范畴是 $\mathcal V$-小且局部 $\mathcal U$-小的。
2. 对函子 $K:\mathcal C\to\mathcal D$ 和对象 $d\in\mathcal D$，逗号范畴 $K/d$ 与 $d/K$ 都是 $\mathcal U$-小的。

**证明.** 两个固定锥之间的态射是 $\mathcal D(X,Y)$ 中满足一族等式的子集，因此为 $\mathcal U$-小。锥的全部数据由一个 $\mathcal V$-小顶点选择和 $\mathcal U$-小族 Hom 元素编码，故锥范畴为 $\mathcal V$-小。

$K/d$ 的对象集是

$$
\coprod_{c\in\operatorname{Ob}(\mathcal C)}\mathcal D(Kc,d),
$$

这是 $\mathcal U$-小集合；其态射可编码为两个对象与一个 $\operatorname{Mor}(\mathcal C)$ 中态射组成的三元组，再由交换等式切出，故态射集也为 $\mathcal U$-小。$d/K$ 的证明把 Hom 方向反转即可。$\square$

## A.5 换 universe 与完备性边界

包含映射给出完全忠实函子

$$
i_{\mathcal U,\mathcal V}:\mathbf{Set}_{\mathcal U}\hookrightarrow
\mathbf{Set}_{\mathcal V}.
$$

它允许把 $\mathcal U$-小对象提升到 $\mathcal V$ 层讨论，但不会把“$\mathcal U$-小完备”自动升级成“$\mathcal V$-小完备”。

**例子 A.9（索引大小不能省略）.** $\mathbf{Set}_{\mathcal U}$ 有所有 $\mathcal U$-小极限和余极限，因为相应的积、余并、子集和商集仍属于 $\mathcal U$。但是它没有所有 $\mathcal V$-小积。

具体地，以 $\mathcal U$ 本身作为 $\mathcal V$-小离散索引集，考虑常值为双点集 $2$ 的图形。若其积 $Q$ 存在于 $\mathbf{Set}_{\mathcal U}$，把积的 Hom 泛性质取在单点集 $1$ 上会给出双射

$$
Q\cong\mathbf{Set}_{\mathcal U}(1,Q)\cong
\prod_{u\in\mathcal U}\mathbf{Set}_{\mathcal U}(1,2)\cong 2^{\mathcal U}.
$$

由 Cantor 定理，$2^{\mathcal U}$ 的基数严格大于 $\mathcal U$，因而不可能与 $\mathcal U$ 的元素双射；所以它不是 $\mathcal U$-小集合。而 $Q$ 按定义必须是 $\mathcal U$-小的，矛盾。故“有所有小积”必须保留所指 universe。

## A.6 选择原则与选择独立性

**元理论假设 A.10.** 本书在 $\mathcal V$-小族上使用选择公理。例如，若 $F:\mathcal C\to\mathcal D$ 本质满且 $\operatorname{Ob}(\mathcal D)$ 为 $\mathcal V$-小，则可为每个 $D\in\mathcal D$ 选择对象 $G(D)$ 和同构 $F(GD)\cong D$。第一章定理 1.24 正是在这一步使用选择。

这不是全局选择原则：若对象真类超出已固定 universe，本书不据此断言可以同时选择代表。另一方面，选择得到的拟逆不严格唯一；其与原伴随或等价结构相容的自然同构唯一性必须另行由泛性质证明。

**资料定位 A.11.** Universe 约定采用 `SOURCES.md` 所列 SGA 4 的 Grothendieck-universe 口径；普通范畴的大小管理、函子范畴和预层范畴口径对照 Mac Lane 与 Riehl。本附录中的命题 A.5--A.9 均由定义在书内证明，不作为外部输入定理。

## A.7 本章小结

本书默认范畴在 $\mathcal V$ 层小、在 $\mathcal U$ 层局部小；单说“小范畴”或“小极限”时，小性相对于 $\mathcal U$。$\operatorname{Nat}(F,G)$ 是否仍为 $\mathcal U$-小取决于源范畴是否 $\mathcal U$-小。预层范畴通常在 $\mathcal V$ 层小而局部 $\mathcal U$-小。换 universe 可以容纳更多索引图形，却不能省略完备性假设中的大小上界。

## 练习

**练习 A.1.** 解释为什么 $\mathbf{Set}_{\mathcal U}$ 不可能是 $\mathcal U$-小范畴。

**练习 A.2.** 证明两个 $\mathcal U$-小范畴之间的函子全体是 $\mathcal U$-小集合。

**练习 A.3.** 找出第一章定理 1.24 中使用选择原则的位置，并说明为什么默认大小约定只需要 $\mathcal V$-小选择而非全局选择。
