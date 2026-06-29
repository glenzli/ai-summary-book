# 第十章：富范畴、加权极限与 enriched Yoneda

## 本章目标

本章在闭对称幺半范畴 $\mathcal V$ 中定义 $\mathcal V$-富范畴、富函子、富自然变换、富函子范畴的 Hom 对象，并证明 enriched Yoneda 引理。

## 依赖前置知识

需要幺半范畴、闭结构、end 公式和普通 Yoneda 引理。

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

其中最后一箭头是反变富函子 $F:\mathcal A^{op}\to\mathcal V$ 对 Hom 的作用，即

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

## 10.6 本章小结

富范畴把 Hom 集替换为 Hom 对象，因此能统一线性范畴、度量空间、拓扑富范畴和谱富范畴。富自然变换由 end 给出，enriched Yoneda 由 end 公式和闭结构证明。加权极限是富环境中正确的极限概念。

## 练习

**练习 10.1.** 证明 $\mathbf{Ab}$-富范畴的复合双线性。

**练习 10.2.** 将一个预序集写成 $\mathbf{2}$-富范畴，其中 $\mathbf{2}=\{0<1\}$。

**练习 10.3.** 描述 $\mathcal V=\mathbf{Cat}$ 时 $\mathcal V$-富范畴与 2-范畴的关系。

**练习 10.4.** 在普通范畴情形下，把定义 10.8 化为通常的极限泛性质。

**练习 10.5.** 查阅 Kelly，写出 enriched natural transformation 的 end 公式。

**练习 10.6.** 在 $\mathcal V=\mathbf{Ab}$ 的情形下，把定理 10.10 写成预加性范畴上的加性 Yoneda 引理。

**练习 10.7.** 解释定理 10.10 的证明中为什么需要闭结构。
