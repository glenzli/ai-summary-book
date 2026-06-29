# 第三章：Dolbeault 复形与 liquid 模

## 本章目标

Dolbeault 复形是复几何中把解析问题转化为微分形式复形的工具。本章说明它在 condensed/liquid 语言中的位置。

## 3.1 Dolbeault 复形

设 $X$ 是复流形，$\mathcal F$ 是全纯向量丛对应的相干层。Dolbeault 复形形如

$$
\mathcal A^{0,0}(X,\mathcal F)
\xrightarrow{\bar\partial}
\mathcal A^{0,1}(X,\mathcal F)
\xrightarrow{\bar\partial}
\cdots.
$$

经典 Dolbeault 定理给出

$$
H^i(X,\mathcal F)
\cong
H^i(\mathcal A^{0,\bullet}(X,\mathcal F),\bar\partial).
$$

## 3.2 拓扑向量空间问题

每个 $\mathcal A^{0,q}(X,\mathcal F)$ 是拓扑向量空间，通常是 Fréchet 或 LF 型空间。普通向量空间导出范畴无法记录其拓扑性质。

liquid 模用于把这些函数空间放入适合做同调代数的范畴中。

## 3.3 凝聚 Dolbeault 复形

**输入定理 3.1（Clausen-Scholze，形式）.** Dolbeault 复形可提升为 liquid/analytic 派生范畴中的复形，并计算相干层的导出全局截面。

也就是说，存在同构

$$
R\Gamma(X,\mathcal F)
\simeq
\left(\mathcal A^{0,\bullet}(X,\mathcal F),\bar\partial\right)
$$

在第二卷 D.6 和第三卷 AR.2 所指定的 condensed/analytic/liquid 派生范畴中成立。

## 3.4 证明路线

证明分为三步：

1. 局部 Poincare lemma 或 Dolbeault lemma。
2. 用 partition of unity 或软层性质处理全局解析。
3. 验证所有函数空间和算子在 liquid 范畴中连续并保持所需正合性。

第三步是 condensed 方法的核心，因为它把分析拓扑放入范畴结构，而不是事后处理。

前两步的 sheaf-theoretic 形式证明见附录 N：fine sheaf 的 acyclicity、acyclic resolution 定理，以及 Dolbeault lemma 推出 sheaf cohomology 计算的过程都在那里展开。Dolbeault lemma 的局部解析骨架见附录 R。

## 3.5 本章小结

Dolbeault 复形是从经典复几何进入 liquid 范畴的桥梁。后续有限性和对偶性都依赖它提供的计算模型。

## 练习

**练习 3.1.** 写出 Dolbeault complex 的微分。

**练习 3.2.** 解释为什么 $\bar\partial^2=0$。

**练习 3.3.** 说明 Fréchet 空间结构为什么不能在普通向量空间范畴中忽略。
