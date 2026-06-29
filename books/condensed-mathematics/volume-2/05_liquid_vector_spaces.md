# 第五章：liquid 向量空间入口

## 本章目标

solid 结构适合非阿基米德和代数完备性，但实分析方向需要更细的测度理论。liquid 向量空间就是为此引入的结构。本章只建立入口和基本定义，不重写完整 liquid tensor experiment。

## 依赖

需要第三、四章的解析环和解析化。

## 5.1 为什么 solid 不够

对 profinite $S$，solid 理论使用整值测度

$$
\mathbb Z^\square[S].
$$

这适合控制乘积型对象

$$
\prod_I\mathbb Z.
$$

但实分析中的对象，例如 Banach 空间、Fréchet 空间和分布空间，不能只由整值测度的 solidification 处理。实数上的测度有大小、收敛、可求和性等额外条件。

## 5.2 $p$-型测度理论

令 $0<p\le1$。Scholze 的 analytic geometry 讲义构造一族实数值测度理论

$$
S\mapsto \mathcal M_p[S],
$$

它们应理解为满足 $p$-型可求和条件的测度对象。

更常用的是

$$
\mathcal M_{<p}[S]=\varinjlim_{q<p}\mathcal M_q[S].
$$

**输入定理 5.1（Scholze）.** 对第二卷输入定理 D.5 指定范围内的 $p$，$(\mathbb R,\mathcal M_{<p})$ 是解析环。

这一定理是 liquid 理论的核心输入。它不是 Banach 空间常识，而是凝聚数学中的解析环定理。

## 5.3 $p$-liquid 模

**定义 5.2.** $(\mathbb R,\mathcal M_{<p})$-模称为 $p$-liquid 实向量空间。

也就是说，一个凝聚 $\mathbb R$-模 $V$ 是 $p$-liquid，如果对所有极不连通 $S$，

$$
\operatorname{Hom}_{\mathbb R}(\mathcal M_{<p}[S],V)
\cong
V(S).
$$

派生版本使用

$$
R\operatorname{Hom}_{\mathbb R}(\mathcal M_{<p}[S],C)
\simeq
R\operatorname{Hom}_{\mathbb R}(\mathbb R[\underline S],C).
$$

## 5.4 liquid 的用法

术语“liquid vector space”在不同文本中会按 $p$ 的范围略有不同。本书采用如下约定：

- $p$-liquid：明确指 $(\mathbb R,\mathcal M_{<p})$-模。
- liquid：泛指由这些 $p$-liquid 结构组织起来的实分析凝聚向量空间理论。

在需要精确命题时，本书总写明所用的 $p$ 或 $<p$。

## 5.5 与 Banach 空间的关系

许多 Banach 空间可嵌入 liquid 框架，但 liquid 范畴不是 Banach 空间范畴的改名。

区别在于：

1. Banach 空间以范数和 Cauchy 完备性为基本结构。
2. liquid 空间以极不连通测试对象上的测度 Hom 判别为基本结构。
3. Banach 空间范畴不适合一般同调代数；liquid 范畴被设计为适合导出范畴和张量运算。

## 5.6 输入定理：范畴结构

**输入定理 5.3（Scholze）.** $p$-liquid 实向量空间所在的 analytic 模范畴具有 kernel、cokernel、扩张、派生范畴和与解析化相容的张量结构。

本卷后续只使用该定理的三个后果：

1. 可以谈论 liquid 对象的 kernel、cokernel 和扩张。
2. 可以构造派生张量积。
3. 可以把复几何中的函数空间放入同调代数框架。

## 5.7 本章小结

liquid 理论是 analytic rings 在实分析方向的主要例子。它用测度理论替代单纯的范数完备化，使实向量空间可以进入凝聚同调代数。

附录 J 补充说明 Banach/Fréchet 空间与 liquid 对象之间的边界，以及为什么“拓扑向量空间”本身不足以给出适合导出范畴的同调代数环境。

## 练习

**练习 5.1.** 说明 solidification 与 Banach completion 的定义差异。

**练习 5.2.** 写出 $p$-liquid 模的 Hom 判别式。

**练习 5.3.** 解释为什么本书在精确命题中必须写明 $p$ 或 $<p$。

**练习 5.4.** 给出一个 Banach 空间范畴不够适合做同调代数的现象。
