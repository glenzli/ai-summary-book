# 附录 A：集合论宇宙与大小问题

## 本章目标

本附录固定本书使用的 universe 口径，并解释“小范畴”“局部小范畴”“大范畴”和“范畴的范畴”的层级。

## A.1 Grothendieck universe

**定义 A.1.** Grothendieck universe $\mathcal U$ 是一个集合，满足足够强的闭包性质，使得 $\mathcal U$ 内部可进行通常集合论构造：若 $x\in\mathcal U$ 且 $y\in x$，则 $y\in\mathcal U$；若 $x,y\in\mathcal U$，则 $\{x,y\}\in\mathcal U$；若 $x\in\mathcal U$，则 $\mathcal P(x)\in\mathcal U$；若 $I\in\mathcal U$ 且 $x_i\in\mathcal U$，则 $\bigcup_{i\in I}x_i\in\mathcal U$。

**约定 A.2.** 本书固定

$$
\mathcal U\in\mathcal V\in\mathcal W.
$$

“集合”默认指 $\mathcal U$-小集合。若对象集合和态射集合属于 $\mathcal U$，范畴称为 $\mathcal U$-小。所有 $\mathcal U$-小范畴组成的范畴 $\mathbf{Cat}_{\mathcal U}$ 是 $\mathcal V$-层级中的对象。

## A.2 局部小范畴

**定义 A.3.** 范畴 $\mathcal C$ 称为局部 $\mathcal U$-小，若对任意 $X,Y\in\mathcal C$，Hom 类 $\mathcal C(X,Y)$ 是 $\mathcal U$-小集合。

**例子 A.4.** $\mathbf{Set}_{\mathcal U}$ 不是 $\mathcal U$-小范畴，因为其对象集合不属于 $\mathcal U$；但它是局部 $\mathcal U$-小的。

## A.3 预层范畴的大小

**命题 A.5.** 若 $\mathcal C$ 是 $\mathcal U$-小范畴，则

$$
\widehat{\mathcal C}=\operatorname{Fun}(\mathcal C^{\operatorname{op}},\mathbf{Set}_{\mathcal U})
$$

是局部 $\mathcal U$-小但通常不是 $\mathcal U$-小的范畴；它可作为 $\mathcal V$-层级中的范畴处理。

**证明.** 一个预层由对象值 $F(C)\in\mathbf{Set}_{\mathcal U}$、态射值 $F(f)$ 以及函子性等式组成。由于 $\mathcal C$ 的对象和态射均为 $\mathcal U$-小，这些数据整体属于下一层 universe $\mathcal V$。因此全部预层可作为 $\mathcal V$-层级中的对象类处理。

若 $F,G$ 是两个预层，则自然变换 $\alpha:F\Rightarrow G$ 是族

$$
\alpha_C:F(C)\to G(C)
$$

满足对每个 $f:C\to D$ 的自然性等式。所有族组成集合

$$
\prod_{C\in\operatorname{Ob}\mathcal C}\mathbf{Set}_{\mathcal U}(F(C),G(C)),
$$

自然变换集合是其中由等式切出的子集。因此 Hom 是集合，故 $\widehat{\mathcal C}$ 局部小。对象类通常不属于 $\mathcal U$，例如当 $\mathcal C=*$ 时，$\widehat{\mathcal C}\simeq\mathbf{Set}_{\mathcal U}$，所以通常不是 $\mathcal U$-小。$\square$

## A.4 选择原则

**约定 A.6.** 本书在普通范畴论中使用足够的选择原则，例如从每个同构类选择代表，或为本质满函子选择拟逆对象。若结果依赖选择，本书会在命题中标明。

**例子 A.7.** 第一章中“完全忠实且本质满推出存在拟逆”需要为每个 $D\in\mathcal D$ 选择 $G(D)$ 和同构 $F(GD)\cong D$。

## A.5 本章小结

Universe 约定不是装饰，而是使 $\mathbf{Cat}$、预层范畴、函子范畴和 $\mathcal{Cat}_\infty$ 等对象可被合法讨论的基础。本书采用固定多层 universe 的方式，在正文中只在必要时重新提醒。

## 练习

**练习 A.1.** 解释为什么 $\mathbf{Set}_{\mathcal U}$ 不可能是 $\mathcal U$-小范畴。

**练习 A.2.** 证明小范畴之间的函子集合是一个集合。

**练习 A.3.** 找出第一章定理 1.24 中使用选择原则的位置。
