# 序章：第二卷的主题和边界

## 本章目标

第一卷建立了凝聚数学的基础语言。本卷开始处理凝聚数学真正发挥威力的部分：solid、analytic、liquid 和几何化。

## 0.1 从第一卷到第二卷

第一卷完成了如下基础：

1. 凝聚集合是紧 Hausdorff 测试站点上的 sheaf。
2. 凝聚阿贝尔群构成阿贝尔范畴。
3. 极不连通空间提供投射测试对象。
4. 自由对象 $\mathbb Z[\underline S]$ 和 solid 自由对象 $\mathbb Z^\square[S]$ 可用于计算。
5. 基本 $\operatorname{Ext}$、$\operatorname{Tor}$ 和派生张量已经可用。

这些内容足以定义 solid 对象，但还不足以解释 solid 理论为什么适合做代数、分析和几何。第二卷处理这个问题。

## 0.2 第二卷的核心问题

本卷围绕五个问题展开：

1. solidification 为什么是一个派生 localization？
2. solid 张量积为什么能控制完备代数对象？
3. analytic rings 如何把“允许的测度”纳入范畴定义？
4. liquid 向量空间如何修正实分析方向的缺口？
5. 这些结构如何进入 $f_!$、$f^!$ 和相干对偶？

## 0.3 第一卷输入定理

本卷默认使用第一卷结果，尤其是：

$$
\mathbb Z^\square[S]\cong\prod_I\underline{\mathbb Z}
$$

以及 solid 阿贝尔群的判别式

$$
\operatorname{Hom}(\mathbb Z^\square[S],A)\cong A(S).
$$

这些公式本身依赖 Nöbeling 定理。第二卷会继续使用它们，不再每次重证。

## 0.4 第二卷证明策略

本卷把证明分成三类：

1. 可以由第一卷直接推出的命题，给出完整证明。
2. Scholze 讲义中的核心结构定理，标为输入定理并解释如何使用。
3. 涉及 liquid、复几何或六函子的长证明，先写定义和路线图，再逐步补齐。

这个策略的目的不是回避证明，而是保持卷册边界：第二卷要发展 solid/analytic 结构，而不是重新写一本一般拓扑、一般 sheaf 理论或一般同调代数教材。

## 0.5 本章小结

第二卷从 solid 派生范畴开始。第一卷的对象仍在场，但重点从“对象是什么”转向“这些对象如何形成可计算的派生代数和几何工具”。

## 练习

**练习 0.1.** 回顾第一卷第十二章，写出 solid 阿贝尔群定义中的 Hom 判别式。

**练习 0.2.** 解释为什么 $\mathbb Z^\square[S]\cong\prod_I\underline{\mathbb Z}$ 不应被理解为典范同构。

**练习 0.3.** 比较普通张量积、solid 张量积和派生 solid 张量积的定义。
