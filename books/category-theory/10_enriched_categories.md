# 第十章：富范畴、加权极限与 enriched Yoneda

普通范畴的 Hom 是集合，但在许多情形中，态射之间还带有线性、拓扑、链复形或同伦结构。若把 Hom 集替换为闭对称幺半范畴 $\mathcal V$ 中的对象，复合与单位就必须写成 $\mathcal V$ 中的态射，并满足带结合子的相干图；这产生 $\mathcal V$-富范畴。富函子、富自然变换和富 Yoneda 引理由同一原则得到，却不能简单地把普通公式中的 Set 替换为 $\mathcal V$ 而忽略内部 Hom。

本章依赖幺半闭结构、普通 Yoneda 与 end 公式。我们会先定义富 Hom 与复合，再构造富函子范畴，并逐步证明 enriched Yoneda；每个自然性陈述都发生在 $\mathcal V$ 中，而不是只在底层集合上验证。

## 10.1 富范畴

**定义 10.1.** 设 $(\mathcal V,\otimes,\mathbb 1)$ 为幺半范畴。一个 $\mathcal V$-富范畴（$\mathcal V$-enriched category）$\mathcal A$ 由以下数据组成：

1. 一个对象类 $\operatorname{Ob}(\mathcal A)$。
2. 对任意 $A,B$，一个 Hom 对象
   $$
   \mathcal A(A,B)\in\mathcal V.
   $$
3. 对任意 $A,B,C$，复合态射
   $$
   \mathcal A(B,C)\otimes\mathcal A(A,B)\to\mathcal A(A,C).
   $$
4. 对任意 $A$，单位态射
   $$
   \mathbb 1\to\mathcal A(A,A).
   $$

这些数据满足由 $\mathcal V$ 中图交换表达的结合律和单位律。

**例子 10.2.** 当 $\mathcal V=\mathbf{Set}$ 且 $\otimes=\times$ 时，$\mathcal V$-富范畴就是普通范畴。

**例子 10.3.** 当 $\mathcal V=\mathbf{Ab}$ 且 $\otimes=\otimes_{\mathbb Z}$ 时，$\mathcal V$-富范畴是预加性范畴：Hom 是阿贝尔群，复合双线性。

**例子 10.4.** 当 $\mathcal V$ 是偏序集 $([0,\infty],\geq,+,0)$ 视为幺半范畴时，$\mathcal V$-富范畴给出广义度量空间。复合公理成为三角不等式。

## 10.2 富函子

**定义 10.5.** $\mathcal V$-富函子 $F:\mathcal A\to\mathcal B$ 由对象函数 $A\mapsto F A$ 和 Hom 态射

$$
F_{A,B}:\mathcal A(A,B)\to\mathcal B(F A,F B)
$$

组成，并与单位和复合相容。

若 $\mathcal V$ 有适当的基础集合函子 $\mathcal V(\mathbb 1,-)$，则每个富范畴有底层普通范畴，其 Hom 集为

$$
\mathcal V(\mathbb 1,\mathcal A(A,B)).
$$

## 10.3 富自然变换与富函子范畴

**定义 10.6.** 设 $\mathcal V$ 为闭对称幺半范畴，并设相关 end 存在。若 $F,G:\mathcal A\to\mathcal B$ 是 $\mathcal V$-富函子，则富自然变换对象定义为 end

$$
\operatorname{Nat}_{\mathcal V}(F,G)=
\int_{A\in\mathcal A}\mathcal B(F A,G A).
$$

当这些 end 对所有 $F,G$ 存在时，富函子和上述 Hom 对象构成 $\mathcal V$-富函子范畴，记作

$$
\operatorname{Fun}_{\mathcal V}(\mathcal A,\mathcal B).
$$

**命题 10.7.** 当 $\mathcal V=\mathbf{Set}$ 时，定义 10.6 恢复普通自然变换集合。

**证明.** 此时 end 公式变为

$$
\int_A\mathbf{Set}(F A,G A),
$$

它按命题 11.6 正是所有满足自然性条件的分量族，即 $\operatorname{Nat}(F,G)$。$\square$

**命题 10.8.** 若 $\mathcal B=\mathcal V$，则对富函子 $F,G:\mathcal A^{\operatorname{op}}\to\mathcal V$，Hom 对象可写为

$$
\operatorname{Fun}_{\mathcal V}(\mathcal A^{\operatorname{op}},\mathcal V)(F,G)
=
\int_{B\in\mathcal A}[F(B),G(B)].
$$

**证明.** $\mathcal V$ 作为自身富范畴时，Hom 对象为内部 Hom $[X,Y]$。代入定义 10.6 即得。$\square$

## 10.4 富预层与 enriched Yoneda

**定义 10.9.** 设 $\mathcal V$ 闭对称幺半且完备余完备。$\mathcal A$ 上的富预层是富函子

$$
\mathcal A^{\operatorname{op}}\to\mathcal V.
$$

富 Yoneda 嵌入把 $A$ 送到富可表预层

$$
\mathcal A(-,A):\mathcal A^{\operatorname{op}}\to\mathcal V.
$$

**定理 10.10（enriched Yoneda）.** 对富预层 $F:\mathcal A^{\operatorname{op}}\to\mathcal V$，存在 $\mathcal V$ 中自然同构

$$
\operatorname{Fun}_{\mathcal V}(\mathcal A^{\operatorname{op}},\mathcal V)(\mathcal A(-,A),F)\cong F(A),
$$

其中左边是富函子范畴中的 Hom 对象。

**证明.** 由命题 10.8，左边是 end

$$
\int_{B\in\mathcal A}[\mathcal A(B,A),F(B)].
$$

按 end 的泛性质，要给出从对象 $X\in\mathcal V$ 到该 end 的态射，等价于给出一族态射

$$
\theta_B:X\to[\mathcal A(B,A),F(B)]
$$

满足 dinatural 条件。由闭结构，这等价于给出一族

$$
\widetilde\theta_B:X\otimes\mathcal A(B,A)\to F(B).
$$

另一方面，态射 $u:X\to F(A)$ 通过富预层 $F$ 的作用给出

$$
X\otimes\mathcal A(B,A)
\xrightarrow{u\otimes\operatorname{id}}
F(A)\otimes\mathcal A(B,A)
\longrightarrow F(B),
$$

其中最后一箭头是反变富函子 $F:\mathcal A^{\operatorname{op}}\to\mathcal V$ 对 Hom 的作用，即

$$
\mathcal A(B,A)\to [F(A),F(B)]
$$

的转置。这个族满足 dinatural 条件，因为 $F$ 与 $\mathcal A$ 的复合相容。

反过来，给定 dinatural 族 $\widetilde\theta_B$，取 $B=A$ 并与单位 $\mathbb1\to\mathcal A(A,A)$ 复合，得到

$$
X\cong X\otimes\mathbb1
\to X\otimes\mathcal A(A,A)
\xrightarrow{\widetilde\theta_A}F(A).
$$

两个构造互逆：从 $u:X\to F(A)$ 构造出的族在 $B=A$ 处与单位复合正是 $u$，由富函子单位律给出；从 dinatural 族取出 $u$ 后再构造族，dinatural 条件应用于态射对象 $\mathcal A(B,A)$ 和单位 $\mathbb1\to\mathcal A(A,A)$，给出对每个 $B$ 原来的 $\widetilde\theta_B$。故对所有 $X$ 有自然双射

$$
\mathcal V\left(X,\int_B[\mathcal A(B,A),F(B)]\right)
\cong
\mathcal V(X,F(A)).
$$

由 Yoneda 引理在 $\mathcal V$ 中的形式，得到所需同构。$\square$

## 10.5 加权极限

**定义 10.11.** 设 $D:\mathcal J\to\mathcal A$ 是 $\mathcal V$-富图形，$W:\mathcal J\to\mathcal V$ 是权重（weight）。$D$ 的 $W$-加权极限是对象 $\{W,D\}\in\mathcal A$，满足对任意 $A\in\mathcal A$ 有自然同构

$$
\mathcal A(A,\{W,D\})
\cong
\operatorname{Fun}_{\mathcal V}(\mathcal J,\mathcal V)(W,\mathcal A(A,D-)).
$$

加权余极限 $W\star D$ 由对偶公式定义：

$$
\mathcal A(W\star D,A)
\cong
\operatorname{Fun}_{\mathcal V}(\mathcal J,\mathcal V)(W,\mathcal A(D-,A)).
$$

**注 10.12.** 普通极限是加权极限的特例。选择常值权重 $\mathbb 1$ 可恢复 conical limit。

## 10.6 富满忠实、张量与边界条件

**例子 10.13（一对象富范畴）.** 一对象 $\mathcal V$-富范畴等价于 $\mathcal V$ 中的代数对象。若唯一对象为 $*$，则 Hom 对象 $M=\mathcal A(*,*)$ 带有复合

$$
M\otimes M\to M
$$

和单位 $\mathbb1\to M$，富范畴的结合律和单位律正是代数对象公理。

**例子 10.14（度量空间的富函子）.** 对 $\mathcal V=([0,\infty],\ge,+,0)$，富函子 $F:X\to Y$ 是非扩张映射。确切地说，Hom 态射

$$
d_X(x,x')\to d_Y(Fx,Fx')
$$

在偏序 $([0,\infty],\ge)$ 中存在，当且仅当

$$
d_Y(Fx,Fx')\le d_X(x,x').
$$

因此富函子不增加距离。

**定义 10.15.** 富函子 $F:\mathcal A\to\mathcal B$ 称为富满忠实（fully faithful enriched functor），若对任意 $A,A'\in\mathcal A$，Hom 态射

$$
\mathcal A(A,A')\to\mathcal B(FA,FA')
$$

是 $\mathcal V$ 中的同构。若底层普通范畴上的函子本质满，则称 $F$ 在通常意义下本质满。

**命题 10.16（富 Yoneda 嵌入全忠实）.** 设定理 10.10 的假设成立。富 Yoneda 嵌入

$$
y:\mathcal A\to\operatorname{Fun}_{\mathcal V}(\mathcal A^{\operatorname{op}},\mathcal V),
\qquad
A\mapsto\mathcal A(-,A)
$$

是富满忠实的。

**证明.** 对任意 $A,B$，由定理 10.10 取 $F=\mathcal A(-,B)$，得到

$$
\operatorname{Fun}_{\mathcal V}(\mathcal A^{\operatorname{op}},\mathcal V)(\mathcal A(-,A),\mathcal A(-,B))
\cong
\mathcal A(A,B).
$$

这正是 Yoneda 嵌入在 Hom 对象上的比较态射为同构。$\square$

**定义 10.17.** 设 $\mathcal A$ 为 $\mathcal V$-富范畴。若 $V\in\mathcal V$ 且 $A\in\mathcal A$，对象 $V\odot A\in\mathcal A$ 称为 $A$ 被 $V$ 张量（tensor），若对任意 $B$ 有自然同构

$$
\mathcal A(V\odot A,B)\cong [V,\mathcal A(A,B)].
$$

对象 $A^V\in\mathcal A$ 称为 $A$ 被 $V$ 余张量（cotensor），若对任意 $B$ 有自然同构

$$
\mathcal A(B,A^V)\cong [V,\mathcal A(B,A)].
$$

**命题 10.18（一对象权重）.** 设 $\mathcal J$ 为一对象单位富范畴，权重由对象 $W\in\mathcal V$ 给出，图形 $D:\mathcal J\to\mathcal A$ 选出对象 $A\in\mathcal A$。若相应张量和余张量存在，则

$$
W\star D\cong W\odot A,\qquad
\{W,D\}\cong A^W.
$$

**证明.** 加权余极限的定义给出

$$
\mathcal A(W\star D,B)
\cong
\operatorname{Fun}_{\mathcal V}(\mathcal J,\mathcal V)(W,\mathcal A(A,B)).
$$

一对象富函子范畴的 Hom 对象在此处就是内部 Hom $[W,\mathcal A(A,B)]$。因此右边表示张量 $W\odot A$ 的泛性质。加权极限同理：

$$
\mathcal A(B,\{W,D\})
\cong
[W,\mathcal A(B,A)],
$$

这正是余张量 $A^W$ 的泛性质。$\square$

**例子 10.19（存在性边界）.** 定义 10.6 中的富函子范畴 Hom 对象由 end 给出。若 $\mathcal V$ 缺少相应的积和等化子，该 end 未必存在，因此富函子范畴不是仅由对象级数据自动产生的。类似地，加权极限要求表示对象存在；定义 10.11 是泛性质，不是存在性定理。

## 10.7 让态射本身携带结构

富范畴把 Hom 集替换为 Hom 对象，因此能统一线性范畴、度量空间、拓扑富范畴和谱富范畴。富自然变换由 end 给出，enriched Yoneda 由 end 公式和闭结构证明，并推出富 Yoneda 嵌入全忠实。加权极限是富环境中正确的极限概念；张量和余张量是一对象权重的基本特例。所有这些存在性都依赖基范畴中的 end、闭结构和表示对象。

## 练习

**练习 10.1.** 证明 $\mathbf{Ab}$-富范畴的复合双线性。

**练习 10.2.** 将一个预序集写成 $\mathbf{2}$-富范畴，其中 $\mathbf{2}=\{0<1\}$。

**练习 10.3.** 描述 $\mathcal V=\mathbf{Cat}$ 时 $\mathcal V$-富范畴与 2-范畴的关系。

**练习 10.4.** 在普通范畴情形下，把定义 10.8 化为通常的极限泛性质。

**练习 10.5.** 查阅 Kelly，写出 enriched natural transformation 的 end 公式。

**练习 10.6.** 在 $\mathcal V=\mathbf{Ab}$ 的情形下，把定理 10.10 写成预加性范畴上的加性 Yoneda 引理。

**练习 10.7.** 解释定理 10.10 的证明中为什么需要闭结构。

**练习 10.8.** 证明一对象 $\mathcal V$-富范畴与 $\mathcal V$ 中的代数对象等价。

**练习 10.9.** 用定理 10.10 证明富 Yoneda 嵌入富满忠实。

**练习 10.10.** 对 $\mathcal V=([0,\infty],\ge,+,0)$，验证富范畴的结合律等价于三角不等式，富函子等价于非扩张映射。

**练习 10.11.** 展开命题 10.18 中加权极限部分的证明。

**练习 10.12.** 给出一个理由说明：若 $\mathcal V$ 缺少无限积，则小富范畴之间的富函子范畴 Hom 对象可能不存在。
