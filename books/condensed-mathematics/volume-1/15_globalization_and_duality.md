# 第十五章：全局化与相干对偶纲要

## 本章目标

本章给出凝聚数学后半部分的结构性路线：如何从仿射解析环走向几何空间上的范畴，以及 solid modules 如何进入相干对偶。这里不完整证明六函子形式，只建立概念地图和基本定义。

## 依赖前置知识

需要解析环、固体模、导出范畴以及代数几何中的仿射概形、Huber pair 和 valuation。

## 15.1 离散 Huber pair

**定义 15.1.** 离散 Huber pair 是一对

$$
(A,A^+)
$$

其中 $A$ 是离散环，$A^+\subset A$ 是整闭子环；在 Huber pair 语境中还要求 $A^+$ 由幂有界元素控制。对本章所需的有限生成离散例子，可把它看作指定“积分元素”的子环。

Scholze 讲义把离散 Huber pair 送到解析环：

$$
(A,A^+)\mapsto (A,A^+)^\square.
$$

对有限生成情形，该构造可由前面 $A^\square$ 型测度理论给出；一般情形通过有限生成子对的滤过余极限定义。

## 15.2 Spa 与 rational subsets

**定义 15.2.** 对离散 Huber pair $(A,A^+)$，定义

$$
\operatorname{Spa}(A,A^+)
$$

为 $A$ 上满足 $|A^+|\le1$ 的 valuation 等价类集合。

典型开集为 rational subset：

$$
U\left(\frac{g_1,\dots,g_n}{f}\right)
=
\{x\mid |g_i(x)|\le |f(x)|\ne0,\;1\le i\le n\}.
$$

这些集合构成拓扑基，并使 $\operatorname{Spa}(A,A^+)$ 成为 spectral space。

## 15.3 解析环的局部化

全局化的关键问题是：若

$$
U\subset \operatorname{Spa}(A,A^+)
$$

是 rational subset，是否能给出相应的解析环

$$
(B,B^+)^\square
$$

并在导出范畴上得到限制、推前和扩展函子？

Scholze 讲义中的答案是肯定的：对离散 Huber pair 的 rational subset 所给出的 rational localization，第二卷输入定理 D.7 登记了所需构造，即解析环之间的映射、解析模范畴的限制函子以及 rational Čech 下降。

第一卷不证明这些局部化定理，只记录它们是后续几何化的入口。第二卷需要把 rational localization 的解析环构造、限制函子和 Cech 下降逐一补齐。

## 15.4 仿射有限型情形的 $f_!$

设 $A$ 是有限生成 $\mathbb Z$-代数，令

$$
f:\operatorname{Spec}A\to\operatorname{Spec}\mathbb Z.
$$

Scholze 讲义构造函子

$$
f_!:D(A^\square)\to D(\mathbb Z^\square)
$$

可理解为“带紧支撑的推前”。

**定理 15.3（Scholze，纲要）.** 对仿射有限型情形，$f_!$ 满足：

1. 与直接和相容。
2. 保持紧对象。
3. 满足投影公式：
   $$
   f_!((M\otimes_{\mathbb Z^\square}^L A^\square)\otimes_{A^\square}^L N)
   \simeq
   M\otimes_{\mathbb Z^\square}^L f_!N.
   $$
4. 有右伴随 $f^!$。

**证明说明.** 该定理在 Scholze 讲义第八讲中证明。证明通过比较 $A^\square$ 与相对解析环 $(A,\mathbb Z)^\square$，并引入边界项控制非 proper 行为。

## 15.5 相干对偶

在经典代数几何中，Grothendieck duality 研究 $f_*$、$f^!$、trace map、projection formula 和 proper base change。凝聚数学的贡献之一是：通过 solid modules，可以把非 proper 情形中的“边界贡献”纳入同一个范畴形式。

Scholze 讲义后半部分把一个 scheme $X$ 送到闭对称幺半三角范畴

$$
D(\mathcal O_{X,\square}),
$$

并讨论 $f^*,f_*,f_!,f^!$ 的相互关系。

第一卷只给纲要；完整六函子形式放入第二卷和后续几何部分。

## 15.6 本章小结

本章把前面建立的 algebraic machinery 放入几何图景：

1. 离散 Huber pair 提供解析环的全局化入口。
2. $\operatorname{Spa}(A,A^+)$ 和 rational subsets 组织局部化。
3. 对仿射有限型映射，可构造 $f_!$ 并满足投影公式。
4. solid modules 是非 proper 相干对偶的技术语言。

## 练习

**练习 15.1.** 写出 valuation 的三条基本性质，并解释 $\operatorname{Spa}(A,A^+)$ 中条件 $|A^+|\le1$ 的意义。

**练习 15.2.** 对 $A=\mathbb Z[T]$，解释为什么“无穷远边界”会出现在非 proper 推前中。

**练习 15.3.** 查阅 Scholze 讲义第八讲，记录 $A=\mathbb Z[T]$ 时 $A_\infty$ 的定义。

**练习 15.4.** 比较经典 proper pushforward 与本章中的 $f_!$。
