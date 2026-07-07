# 第十六章：Nadler-Zaslow、microlocal sheaves 与 cotangent bundles

## 本章目标

本章介绍 cotangent bundle 的 Fukaya category 与 constructible sheaves 之间的关系。它是 HMS、geometric representation theory、microlocal geometry 和 wrapped categories 的交汇点。

## 依赖前置知识

需要第六章 wrapped categories、第二章 derived categories，以及 sheaf 和 microsupport 的基本语言。

## 16.1 Constructible sheaves

**定义 16.1.** 设 $Q$ 是实解析流形，$\mathcal S$ 是有限 Whitney stratification。一个复形 $\mathcal F$ 称为 $\mathcal S$-constructible，若对每个 stratum $S\in\mathcal S$，限制 $\mathcal F|_S$ 的 cohomology sheaves 为局部常值且有限维。constructible sheaves 的 dg category 记为
$$
\operatorname{Sh}_c(Q).
$$

**定义 16.2.** sheaf $\mathcal F$ 的 microsupport $SS(\mathcal F)\subset T^\ast Q$ 是测量 $\mathcal F$ 在各方向传播失败的闭 conic subset。严格定义使用局部函数的 cohomology vanishing 条件。

**警告 16.3.** Microsupport 不是普通 support。普通 support 位于 $Q$，microsupport 位于 $T^\ast Q$，记录方向性信息。

## 16.2 Cotangent Fukaya category

**定义 16.4.** $T^\ast Q$ 带 canonical Liouville form。其 conic Lagrangians 与 constructible sheaves 的 microsupport 条件相匹配。对 conic Lagrangian $\Lambda\subset T^\ast Q$，记
$$
\operatorname{Sh}_\Lambda(Q)
$$
为 microsupport 包含于 $\Lambda$ 的 constructible sheaves category。

**外部输入定理 16.5（Nadler-Zaslow）.** 对 compact real analytic manifold $Q$，constructible sheaves category 与 $T^\ast Q$ 的适当 Fukaya $A_\infty$ category 之间存在 fully faithful embedding 或等价版本；对象字典把标准/余标准 sheaves 与 conormal Lagrangians 对应。  
来源：Nadler-Zaslow, *Constructible Sheaves and the Fukaya Category*。

**外部输入定理 16.6（wrapped/microlocal 扩展）.** 对带 stop 的 cotangent bundles 和更一般 Weinstein sectors，partially wrapped Fukaya categories 与 microsupport 受限的 sheaf categories 存在等价版本。  
来源：Ganatra-Pardon-Shende, *Microlocal Morse theory of wrapped Fukaya categories*，以及 Nadler 后续 microlocal branes 工作。

## 16.3 标准对象

**定义 16.7.** 若 $S\subset Q$ 是 stratum，标准 sheaf 是 $j_{S!}k_S$，余标准 sheaf 是 $j_{S*}k_S$，其中 $j_S:S\hookrightarrow Q$。

**解释 16.8.** 在 Fukaya 侧，标准和余标准 sheaves 对应于正/负 conormal branes。Morphism spaces 的计算对应 sheaf Ext groups 与 Floer cochains 的比较。

**命题 16.9.** 若 $Q$ 有有限 stratification 且 constructible category 由标准 sheaves 生成，则对应 Fukaya category 由相应 conormal branes split-generate。

**证明.** 由 Nadler-Zaslow 型等价，标准 sheaves 的生成性传递到对应 conormal branes。增强等价保持 thick closure 和 split-closure。证毕。

## 16.4 HMS 中的作用

Microlocal sheaf 模型在 HMS 中有三种作用：

1. 把 Fukaya category 的计算转化为 sheaf-theoretic Ext 计算；
2. 为 stopped/partially wrapped categories 提供 cosheaf/sheaf 模型；
3. 在 skeleton 上给出 combinatorial category，从而连接 tropical geometry 和 mirror symmetry。

**模板 16.10.** 若 $M$ 是 Weinstein sector，$\mathfrak L$ 是 skeleton，且有 sheaf model
$$
\mathcal W(M)\simeq \operatorname{Sh}_{\mathfrak L}(Q),
$$
则 HMS 可通过证明
$$
\operatorname{Sh}_{\mathfrak L}(Q)\simeq \mathcal B
$$
来完成。

## 本章小结

Nadler-Zaslow correspondence 把 cotangent Fukaya categories 与 constructible sheaves 联系起来。Microlocal sheaf theory 使 wrapped 和 stopped Fukaya categories 可被 sheaf categories 描述，是现代 HMS 中局部计算和 functoriality 的核心工具之一。

## 练习

**练习 16.1.** 区分 support 与 microsupport，并给出常值 sheaf 的 microsupport。

**练习 16.2.** 对区间分层，写出标准 sheaves 和余标准 sheaves 的例子。

**练习 16.3.** 解释 conormal bundle 为什么是 Lagrangian。

**练习 16.4.** 用模板 16.10 写出一个 sheaf 模型证明 HMS 的形式路线。
