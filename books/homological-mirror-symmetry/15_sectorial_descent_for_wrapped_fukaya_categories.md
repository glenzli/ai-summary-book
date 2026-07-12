# 第十五章：wrapped Fukaya categories 的 sectorial descent

## 本章目标

本章把第六章的 Liouville sectors 推进到 sectorial descent：wrapped Fukaya category 对 sectorial covers 呈 cosheaf 行为。这是高维 HMS 的局部到整体机制。

## 依赖前置知识

需要第六章 Liouville sectors、第七章 stops、第十四章生成性工具。

## 15.1 Sectorial cover

**定义 15.1（Weinstein sectorial cover）.** 设 $X$ 是 Liouville sector。
有限族 Liouville subsectors
$$
\{X_i\}_{i\in\mathcal I}
$$
称为 sectorial cover，若它们覆盖 $X$，并且由 $\partial X$ 与各 $X_i$ 的
internal cylindrical boundary faces 构成的 hypersurface family 是 GPS 意义下
sectorial：这些 hypersurfaces 的 characteristic foliations 在交处
$\omega$-orthogonal，并可在各 face $H_a$ 的邻域选择无穷远线性的函数
$I_a$，使
$$
dI_a|_{\operatorname{char}(H_a)}\ne0,\qquad
dI_a|_{\operatorname{char}(H_b)}=0\ (a\ne b),\qquad
\{I_a,I_b\}=0.
\tag{15.1}
$$
于是适当 smoothing corners 后，每个非空有限交
$X_J=\bigcap_{j\in J}X_j$ 都继承 Liouville sector 结构。

对有序分拆 $P\sqcup Q\sqcup R=\mathcal I$，考虑 strata
$$
X_{P,Q,R}
=\left(\bigcap_{p\in P}X_p\right)
 \cap\left(\bigcap_{q\in Q}\partial X_q\right)
 \setminus\left(\bigcup_{r\in R}X_r\right).
\tag{15.2}
$$
若每个非空 stratum closure 的 characteristic foliation symplectic reduction
在 convexification 后均可经 Liouville deformation 变为 Weinstein sector，
则称该 cover 为 **Weinstein sectorial cover**。仅要求有限交是 Liouville
sectors，并不足以满足这里的 Weinstein 条件。

**定义 15.2.** 对定义 15.1 的 cover，写
$$
\mathcal W(X_\bullet)
$$
为由所有有限交的 wrapped Fukaya categories 组成的 Cech diagram。

## 15.2 Descent 陈述

**外部输入定理 15.3（Weinstein sectorial descent）.** 对定义 15.1 的
Weinstein sectorial cover $\{X_i\}_{i\in\mathcal I}$，由定理 6.13 的 proper
cylindrical inclusions 诱导的自然 functor
$$
\operatorname*{hocolim}_{\varnothing\ne J\subset\mathcal I}\mathcal W(X_J)
\longrightarrow\mathcal W(X)
$$
在 pretriangulated envelopes 上是 equivalence，因而特别是 Morita
equivalence。该结论不在任意 sectorial cover 上无条件陈述。
来源：Ganatra-Pardon-Shende, *Sectorial descent for wrapped Fukaya categories*。

**解释 15.4.** “cosheaf”意味着 Weinstein sectorial pieces 的 categories
通过 homotopy colimit 胶合成全局 category。方向是协变的：定理 6.13
允许的 inclusion of sectors 诱导 $\mathcal W(X_i)\to\mathcal W(X)$。

**命题 15.5.** 假设定理 15.3。若每个 $X_J$ 的 wrapped category 由对象集合 $\mathcal G_J$ split-generate，则 $\mathcal W(X)$ 由所有 $\mathcal G_J$ 在 inclusion functors 下的像 split-generate。

**证明.** homotopy colimit 的对象由局部 diagram 中对象的像生成，morphisms 由局部 morphisms 和 gluing relations 生成。若每个局部 category 由 $\mathcal G_J$ 生成，则整个 diagram 的 homotopy colimit 由这些生成对象的像生成。定理 15.3 把该 homotopy colimit 与 $\mathcal W(X)$ Morita 等价，故得到全局生成。证毕。

## 15.3 Kunneth 与 product sectors

**外部输入定理 15.6（wrapped Kunneth 公式）.** 在适当 Liouville sector 假设下，存在 wrapped Fukaya categories 的 Kunneth 型关系
$$
\mathcal W(X\times Y)\simeq \mathcal W(X)\otimes\mathcal W(Y)
$$
的 Morita 版本。  
来源：GPS sectorial descent 体系。

**解释 15.7.** Kunneth 公式允许把局部模型拆成基本 pieces 的乘积，是 pair-of-pants 和 microlocal 模型计算中的重要工具。

## 15.4 HMS 的 descent 比较

**模板 15.8.** 要用 sectorial descent 证明 HMS，需构造两个 diagrams：
$$
J\mapsto \mathcal W(X_J),\qquad
J\mapsto \mathcal B_J
$$
并证明：

1. 每个 $J$ 上有局部 HMS $\mathcal W(X_J)\simeq\mathcal B_J$；
2. inclusion functors 与 B-side restriction/pushforward/localization functors 相容；
3. A-side 和 B-side 都满足 descent；
4. homotopy colimits 给出全局 categories。

**命题 15.9.** 若模板 15.8 的四项成立，则得到全局 HMS Morita equivalence。

**证明.** 局部 HMS 给出两个 Cech diagrams 的逐点 Morita equivalence；相容性给出 diagrams 等价。homotopy colimit 保持逐点等价。由 A/B 两边 descent，把 colimits 识别为全局 categories，得证。证毕。

## 本章小结

Sectorial descent 是 wrapped Fukaya categories 的局部到整体定理。这里使用的
精确外部输入以 Weinstein sectorial cover 为假设；在该范围内，它把全局
wrapped category 表成局部 categories 的 homotopy colimit，从而把 HMS 证明
分解为局部 HMS 与 gluing 相容性。

## 练习

**练习 15.1.** 写出两个开集覆盖时 Cech diagram 的三个对象和两个交叠箭头。

**练习 15.2.** 说明为什么 sectorial descent 是 cosheaf 而不是 sheaf 形式。

**练习 15.3.** 证明命题 15.5 中生成对象在 homotopy colimit 下的形式稳定性。

**练习 15.4.** 按模板 15.8 写出 pair-of-pants decomposition 的 HMS 证明框架。
