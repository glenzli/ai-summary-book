# 第一章：复解析空间的凝聚语言

## 本章目标

本章解释复解析空间如何进入凝聚数学。核心思想是：全纯函数、光滑函数、分布和上同调对象都应放入 analytic/liquid 范畴，而不只看作普通拓扑向量空间。

## 1.1 经典复解析空间

经典复解析空间局部由

$$
V(f_1,\ldots,f_r)\subset U\subset\mathbb C^n
$$

给出，其中 $f_i$ 是全纯函数。结构层为

$$
\mathcal O_X.
$$

相干解析层是局部有限表示的 $\mathcal O_X$-模。

## 1.2 凝聚视角

在凝聚语言中，一个函数空间不只是一组函数，而是一个对紧 Hausdorff 测试对象取值的 sheaf。对于拓扑向量空间 $V$，其凝聚化为

$$
S\mapsto \operatorname{Cont}(S,V).
$$

若 $V$ 是 Fréchet 或 nuclear 空间，还需要 liquid 结构来保留分析性质。

## 1.3 analytic structure sheaf

在复几何中，$\mathcal O_X$ 的截面通常形成 Fréchet 空间。condensed/analytic 方法将其视为 analytic 或 liquid 模对象。

**输入定理 1.1（Clausen-Scholze，形式）.** 对适当的复解析空间 $X$，存在 analytic 派生范畴

$$
D_{\operatorname{an}}(X)
$$

以及结构对象 $\mathcal O_X$，使相干解析层嵌入该范畴。

## 1.4 局部模型

开多圆盘 $U\subset\mathbb C^n$ 上的全纯函数空间

$$
\mathcal O(U)
$$

可看作带有自然 Fréchet 拓扑的向量空间。condensed 处理不是丢掉拓扑，而是把它转化为 liquid/analytic 对象。

## 1.5 本章小结

复解析空间进入凝聚数学的关键，是把函数空间和层的截面放入 analytic/liquid 模范畴。这样之后，导出函子和对偶性可以在同一个范畴系统中表达。

## 练习

**练习 1.1.** 写出复解析空间的局部定义。

**练习 1.2.** 解释为什么 $\mathcal O(U)$ 自然是拓扑向量空间。

**练习 1.3.** 比较普通凝聚化和 liquid 结构的差异。
