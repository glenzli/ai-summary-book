# 第十五章：wrapped Fukaya categories 的 sectorial descent

把 Liouville 空间切成若干易算部分并不自动得到全局 Fukaya 范畴：普通开覆盖没有控制伪全纯曲线越过边界的方式，交叠处的 wrapped 函子也未必存在。Weinstein sectorial cover 增加了特征叶正交性、可交换边界函数和 reduction 条件，恰好使各有限交仍处在 wrapped theory 的可控范围。Ganatra--Pardon--Shende 的 descent 定理随后把全局范畴识别为这些局部范畴的同伦余极限。本章从第六章的 sector functoriality 出发，写清 cover 假设、Cech 图方向、局部生成元的传递以及两侧 descent 图如何产生全局 HMS。

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

**例 15.4A（两个 sectors 的 derived pushout）.** 若
$X=X_1\cup X_2$ 是 Weinstein sectorial cover，且
$X_{12}=X_1\cap X_2$，则定理 15.3 专门化为
$$
\mathcal W(X_1)\mathop{\sqcup}\limits^{h}_{\mathcal W(X_{12})}
\mathcal W(X_2)\longrightarrow\mathcal W(X)
\tag{15.3}
$$
在 pretriangulated/Morita 口径下的等价。左边不是 morphism 集合的普通并集：
它还加入由交叠范畴识别两侧对象和态射所需的 homotopy-coherent relations。
这正是普通 pushout 不能代替 homotopy pushout 的原因。

**命题 15.5.** 假设定理 15.3。若每个 $X_J$ 的 wrapped category 由对象集合 $\mathcal G_J$ split-generate，则 $\mathcal W(X)$ 由所有 $\mathcal G_J$ 在 inclusion functors 下的像 split-generate。

**证明.** 在小、幂等完备稳定范畴的 Morita 局部化中取 Cech 图的 homotopy
colimit，记结构函子为 $i_J$。令 $\mathcal T$ 是该 colimit 中由全部
$i_J(G)$（$G\in\mathcal G_J$）厚生成的子范畴。对每个 $J$，
$\mathcal G_J$ 厚生成 $\operatorname{Perf}\mathcal W(X_J)$，而 exact functor
$i_J$ 保持有限余极限、shifts 与 retracts，故 $i_J$ 的整个像包含于
$\mathcal T$。Homotopy colimit 由各结构函子的像生成，因此
$\mathcal T$ 等于整个 colimit。定理 15.3 再以 Morita 等价把该结论传到
$\mathcal W(X)$。证毕。

## 15.3 Kunneth 与 product sectors

**外部输入定理 15.6（wrapped Kunneth 公式）.** 对满足 GPS product-sector、
brane 与无穷远 admissibility 假设的 Liouville sectors $X,Y$，外积函子在
perfect/Morita 口径给出 Kunneth 型关系
$$
\mathcal W(X\times Y)\simeq \mathcal W(X)\otimes\mathcal W(Y)
$$
；若带 stops，必须同时采用来源规定的 product stop。
来源：GPS sectorial descent 体系。

**解释 15.7.** Kunneth 公式允许把局部模型拆成基本 pieces 的乘积，是 pair-of-pants 和 microlocal 模型计算中的重要工具。

## 15.4 两侧 Cech 图的比较

**定义 15.8（descent-compatible local HMS datum）.** 对同一有限索引范畴，
构造两个 diagrams
$$
J\mapsto \mathcal W(X_J),\qquad
J\mapsto \mathcal B_J
$$
。称逐点 Morita 等价 $E_J:\mathcal W(X_J)\simeq\mathcal B_J$ 为
descent-compatible，若满足：

1. 每个 $J$ 上有局部 HMS $\mathcal W(X_J)\simeq\mathcal B_J$；
2. inclusion functors 与 B-side restriction/pushforward/localization functors 相容；
3. A-side 和 B-side 都满足 descent；
4. homotopy colimits 给出全局 categories。

**命题 15.9.** 若定义 15.8 的四项成立，则得到全局 HMS Morita equivalence。

**证明.** 局部 HMS 给出两个 Cech diagrams 的逐点 Morita equivalence；相容性给出 diagrams 等价。homotopy colimit 保持逐点等价。由 A/B 两边 descent，把 colimits 识别为全局 categories，得证。证毕。

Weinstein sectorial 条件是从几何覆盖通向范畴余极限的非形式部分；一旦外部输入定理 15.3 可用，局部生成元和逐点 Morita 等价都能沿 homotopy colimit 传到全局。因而局部 HMS 本身仍不够，交叠函子的自然相容性与 B-side descent 同样是全局等价的组成部分。Cotangent bundle 的 sheaf 模型会给这种局部到整体结构一个更具体的计算语言。

## 练习

**练习 15.1.** 写出两个开集覆盖时 Cech diagram 的三个对象和两个交叠箭头。

**练习 15.2.** 说明为什么 sectorial descent 是 cosheaf 而不是 sheaf 形式。

**练习 15.3.** 证明命题 15.5 中生成对象在 homotopy colimit 下的形式稳定性。

**练习 15.4.** 对例 15.4A 写出 B-side derived pushout，并给出使两个
pushout Morita 等价所需的自然变换数据。
