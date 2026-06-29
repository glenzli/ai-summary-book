# 第六章：GAGA

## 本章目标

本章讨论 algebraic coherent sheaves 与 analytic coherent sheaves 的比较。

## 6.1 经典 GAGA

设 $X$ 是 $\mathbb C$ 上 proper scheme，$X^{an}$ 是其复解析空间。经典 GAGA 给出：

1. algebraic coherent sheaves 与 analytic coherent sheaves 的范畴等价。
2. 上同调比较同构
   $$
   H^i(X,\mathcal F)\cong H^i(X^{an},\mathcal F^{an}).
   $$

## 6.2 导出表述

导出版本可写成

$$
D_{\operatorname{coh}}(X)
\simeq
D_{\operatorname{coh}}(X^{an})
$$

并与导出全局截面相容。

## 6.3 凝聚表述

**输入定理 6.1（Clausen-Scholze）.** 在 condensed/analytic 框架中，proper algebraic varieties over $\mathbb C$ 的 algebraic coherent theory 与 analytic coherent theory 可通过同一套 analytic 派生范畴比较，并恢复 GAGA。

## 6.4 证明路线

1. 建立 algebraic 与 analytic 两侧的解析环模型。
2. 对仿射或局部模型证明比较。
3. 用 properness 保证上同调有限性和下降。
4. 用 Cech 下降或 derived descent 粘合局部比较。

接受 classical 或 Clausen-Scholze 的 GAGA 输入后，附录 K 证明 exact coherent equivalence 如何诱导

$$
D^b_{\operatorname{coh}}(X)
\simeq
D^b_{\operatorname{coh}}(X^{an})
$$

以及 $R\Gamma$ 比较、Euler characteristic 比较等形式推论。

附录 Q 补充 properness 不可省略的 $\mathbb A^1$ 反例，并展开 exact equivalence 到 bounded derived equivalence、上同调比较到 $R\Gamma$ 比较的证明细节。

Projective GAGA 的 classical 证明结构，包括 Serre twisting、cohomology comparison、full faithfulness 和 essential surjectivity 的模块化推导，见附录 Y。

## 6.5 本章小结

GAGA 说明 algebraic 和 analytic 两种几何在 proper 条件下给出同一套相干理论。condensed/analytic 语言提供了统一的比较范畴。

## 练习

**练习 6.1.** 写出经典 GAGA 的两个主要结论。

**练习 6.2.** 解释 properness 在 GAGA 中的作用。

**练习 6.3.** 说明为什么导出版本比逐个上同调比较更自然。
