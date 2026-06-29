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

## 9.5 本章小结

闭幺半范畴把“函数对象”内化到范畴中，核心公式是张量-Hom 伴随。笛卡尔闭范畴是逻辑、类型论和 topos 理论中的基本环境。Day 卷积说明小幺半范畴的结构可延拓到其预层范畴。

## 练习

**练习 9.1.** 写出 $\mathbf{Set}$ 中评价映射 $Z^X\times X\to Z$，并验证 curry 化公式。

**练习 9.2.** 证明若 $\mathcal C$ 是闭幺半范畴，则 $-\otimes X$ 保持初对象。

**练习 9.3.** 对有限维向量空间，比较内部 Hom 与对偶张量 $X^*\otimes Z$。

**练习 9.4.** 使用 Yoneda 引理验证 $y(a)\star y(b)$ 的值应同构于 $\mathcal C(-,a\otimes b)$。

**练习 9.5.** 查阅并写出 coend 作为余等化子的公式，为第十一章做准备。
