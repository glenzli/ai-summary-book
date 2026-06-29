# 第九章：闭范畴、张量-Hom 伴随与 Day 卷积

## 本章目标

本章定义闭幺半范畴，并说明张量-Hom 伴随如何推广集合中的函数对象。最后给出 Day 卷积的定义公式，作为预层范畴上的幺半结构来源。

## 依赖前置知识

需要幺半范畴、伴随函子、预层和 coend 的基本动机；coend 的严格定义将在第十一章给出。

## 9.1 闭幺半范畴

**定义 9.1.** 幺半范畴 $(\mathcal C,\otimes,\mathbb 1)$ 称为右闭（right closed），若对每个 $X\in\mathcal C$，函子

$$
-\otimes X:\mathcal C\to\mathcal C
$$

有右伴随，记作

$$
[X,-]:\mathcal C\to\mathcal C.
$$

也就是说，有自然同构

$$
\mathcal C(Y\otimes X,Z)\cong\mathcal C(Y,[X,Z]).
$$

若对每个 $X$，$X\otimes-$ 也有右伴随，则称为双闭。对称幺半范畴中左右闭等价。

**例子 9.2.** $\mathbf{Set}$ 对笛卡尔积闭，内部 Hom 为函数集：

$$
[X,Z]=Z^X.
$$

伴随双射为 currying：

$$
\mathbf{Set}(Y\times X,Z)\cong\mathbf{Set}(Y,Z^X).
$$

**例子 9.3.** $\mathbf{Vect}_k$ 对张量积闭，内部 Hom 为线性映射空间 $\operatorname{Hom}_k(X,Z)$，伴随为

$$
\mathbf{Vect}_k(Y\otimes_k X,Z)\cong
\mathbf{Vect}_k(Y,\operatorname{Hom}_k(X,Z)).
$$

## 9.2 评价与余评价

**定义 9.4.** 闭结构的评价态射是伴随双射下 $\operatorname{id}_{[X,Z]}$ 的转置：

$$
\operatorname{ev}_{X,Z}:[X,Z]\otimes X\to Z.
$$

对任意 $f:Y\otimes X\to Z$，其 curry 化 $\bar f:Y\to[X,Z]$ 满足

$$
\operatorname{ev}_{X,Z}\circ(\bar f\otimes\operatorname{id}_X)=f
$$

在相干约束下成立。

**命题 9.5.** 在闭幺半范畴中，函子 $-\otimes X$ 保持所有存在的余极限。

**证明.** $-\otimes X$ 是左伴随，应用定理 4.7。$\square$

## 9.3 笛卡尔闭范畴

**定义 9.6.** 有有限积的范畴 $\mathcal C$ 若对每个 $X$，积函子 $-\times X$ 有右伴随 $(-)^X$，则称 $\mathcal C$ 为笛卡尔闭范畴（cartesian closed category）。

**例子 9.7.** $\mathbf{Set}$ 是笛卡尔闭范畴。许多 sheaf topos 也是笛卡尔闭范畴；这将在第十四章作为 Grothendieck topos 的性质出现。

**命题 9.8.** 在笛卡尔闭范畴中，指数对象 $Z^X$ 若存在，则在唯一同构意义下唯一。

**证明.** $Z^X$ 表示函子

$$
Y\longmapsto\mathcal C(Y\times X,Z).
$$

表示对象由命题 2.9 唯一。$\square$

## 9.4 Day 卷积

**定义 9.9.** 设 $\mathcal C$ 为小幺半范畴，$\widehat{\mathcal C}=\operatorname{Fun}(\mathcal C^{\operatorname{op}},\mathbf{Set})$。预层 $P,Q$ 的 Day 卷积由公式

$$
(P\star Q)(c)=
\int^{a,b\in\mathcal C}
P(a)\times Q(b)\times\mathcal C(c,a\otimes b)
$$

定义，其中 $\int^{a,b}$ 是 coend。

**外部输入定理 9.10.** Day 卷积使 $\widehat{\mathcal C}$ 成为幺半范畴，且 Yoneda 嵌入

$$
y:\mathcal C\to\widehat{\mathcal C}
$$

是强幺半函子：

$$
y(a)\star y(b)\cong y(a\otimes b).
$$

本书第十一章会证明 coend 的基本计算公式；完整的 Day 卷积相干性证明作为外部输入，来源见 Day、Kelly 和 Riehl。

## 9.5 闭结构的基本计算

**命题 9.11（单位的内部 Hom）.** 在右闭幺半范畴中，对任意对象 $Z$ 有自然同构

$$
[\mathbb1,Z]\cong Z.
$$

**证明.** 对任意 $Y$，由闭结构和右单位约束有自然双射

$$
\mathcal C(Y,[\mathbb1,Z])
\cong
\mathcal C(Y\otimes\mathbb1,Z)
\cong
\mathcal C(Y,Z).
$$

因此 $[\mathbb1,Z]$ 与 $Z$ 表示同一 Hom 函子，由表示对象唯一性得到自然同构。$\square$

**命题 9.12（指数律）.** 在右闭幺半范畴中，若只使用右闭结构，则存在自然同构

$$
[X\otimes Y,Z]\cong [X,[Y,Z]].
$$

**证明.** 对任意 $T$，连续使用闭结构与结合约束得到

$$
\begin{aligned}
\mathcal C(T,[X\otimes Y,Z])
&\cong \mathcal C(T\otimes(X\otimes Y),Z)\\
&\cong \mathcal C((T\otimes X)\otimes Y,Z)\\
&\cong \mathcal C(T\otimes X,[Y,Z])\\
&\cong \mathcal C(T,[X,[Y,Z]]).
\end{aligned}
$$

该同构对 $T$ 自然，故由 Yoneda 引理得到所需同构。$\square$

**例子 9.13（偏序中的闭结构）.** 把有有限交的偏序集 $P$ 视为范畴。笛卡尔闭结构等价于：对每个 $x,z\in P$，存在元素 $z^x$ 使得

$$
y\wedge x\le z
\quad\Longleftrightarrow\quad
y\le z^x.
$$

这正是 Heyting 蕴含的范畴论形式。这里不需要证明 Heyting 代数的外部逻辑性质；只需注意闭结构就是上述伴随。

**例子 9.14（非闭的幺半结构）.** $(\mathbf{Set},\sqcup,\varnothing)$ 是对称幺半范畴，但不是闭幺半范畴。若对某个非空集合 $X$，函子 $-\sqcup X$ 有右伴随，则它作为左伴随必须保持初对象。然而

$$
\varnothing\sqcup X\cong X\ne\varnothing,
$$

矛盾。因此幺半结构本身不保证闭结构存在。

**命题 9.15（Day 卷积的单位计算）.** 设 $\mathcal C$ 为小幺半范畴。在 Day 卷积下，预层 $y(\mathbb1)=\mathcal C(-,\mathbb1)$ 是单位对象；即对任意预层 $P$ 有自然同构

$$
y(\mathbb1)\star P\cong P,\qquad
P\star y(\mathbb1)\cong P.
$$

**证明.** 只证明左单位，右单位同理。在对象 $c$ 处，

$$
(y\mathbb1\star P)(c)
=
\int^{a,b}
\mathcal C(a,\mathbb1)\times P(b)\times\mathcal C(c,a\otimes b).
$$

先对 $a$ 使用 co-Yoneda，得到

$$
\int^b P(b)\times\mathcal C(c,\mathbb1\otimes b).
$$

再用左单位约束 $\mathbb1\otimes b\cong b$，并对 $b$ 使用 co-Yoneda，得到 $P(c)$。这些同构对 $c$ 和 $P$ 自然。完整的结合相干性仍属定理 9.10 的外部输入。$\square$

## 9.6 本章小结

闭幺半范畴把“函数对象”内化到范畴中，核心公式是张量-Hom 伴随。由伴随可推出单位内部 Hom、指数律和张量保持余极限等计算规则。闭结构是额外条件，并非所有幺半范畴都有。Day 卷积说明小幺半范畴的结构可延拓到其预层范畴；本书证明可表对象和单位的 co-Yoneda 计算，把完整相干性作为外部输入。

## 练习

**练习 9.1.** 写出 $\mathbf{Set}$ 中评价映射 $Z^X\times X\to Z$，并验证 curry 化公式。

**练习 9.2.** 证明若 $\mathcal C$ 是闭幺半范畴，则 $-\otimes X$ 保持初对象。

**练习 9.3.** 对有限维向量空间，比较内部 Hom 与对偶张量 $X^*\otimes Z$。

**练习 9.4.** 使用 Yoneda 引理验证 $y(a)\star y(b)$ 的值应同构于 $\mathcal C(-,a\otimes b)$。

**练习 9.5.** 查阅并写出 coend 作为余等化子的公式，为第十一章做准备。

**练习 9.6.** 证明命题 9.11 中的同构对 $Z$ 自然。

**练习 9.7.** 在笛卡尔闭范畴中，把命题 9.12 写成通常的指数律 $(Z^Y)^X\cong Z^{X\times Y}$。

**练习 9.8.** 证明若 $X\ne\varnothing$，则 $-\sqcup X:\mathbf{Set}\to\mathbf{Set}$ 不可能有右伴随。

**练习 9.9.** 用 co-Yoneda 逐步计算 $P\star y(\mathbb1)$。

**练习 9.10.** 设 $P$ 为有有限交的偏序集。证明 $P$ 作为范畴是笛卡尔闭范畴，当且仅当对每个 $x$，映射 $(-)\wedge x$ 在偏序意义下有右伴随。
