# 附录 AR：Clausen-Scholze 复几何核心定理图谱

## AR.0 目标

本附录把第三卷中使用的 Clausen-Scholze 复几何定理整理为核心图谱。第三卷不是把复几何当作凝聚数学的可选展示，而是把它作为 analytic/liquid 结构的主要检验场：

1. classical complex geometry 给出对象、上同调、对偶和 Riemann-Roch；
2. condensed/analytic/liquid 语言给出新的范畴表达；
3. Clausen-Scholze 的定理说明这两套语言相容。

本附录不新增外部输入；它把这些输入按教材主线排列。

## AR.1 建模定理

**核心定理 AR.1（复解析对象的 condensed/analytic 建模）。** 复流形、复解析空间及其结构层可以进入 condensed/analytic 框架；相干解析层和相关函数空间具有与 classical 结构相容的 analytic/liquid 表示。

**书内部分。** 第一章给出复解析空间的语言；第二章说明相干层和导出范畴；第二卷提供 analytic/liquid 范畴和 realization 接口。

**外部部分。** Clausen-Scholze 建模定理本身，包括对象进入 analytic/liquid 范畴时保持正确 Hom、tensor 和 cohomology 的断言。

**依赖位置。** 第三卷第一、二章；第二卷附录 S、Z；第三卷附录 B、AQ。

## AR.2 Dolbeault-liquid 定理

**核心定理 AR.2（Dolbeault 复形的严格 liquid 比较）。** 固定
\(0<p\le1\)。对 compact complex manifold \(X\) 和 holomorphic vector bundle
\(E\)，令 Dolbeault Fréchet 复形为

$$
\Gamma(X,\mathcal A^{0,\bullet}(E)),\bar\partial
$$

则每一项的凝聚化都是 \(p\)-liquid。若该复形具有连续 Hodge/Green splitting，
则对每个 \(q\) 有

$$
H^q\!\left(\underline{\Gamma(X,\mathcal A^{0,\bullet}(E))}\right)
\cong
\underline{H^q_{\mathrm{top}}
\left(\Gamma(X,\mathcal A^{0,\bullet}(E))\right)}
\cong
\underline{H^q(X,E)}.
$$

第一个同构是 liquid/condensed 严格性比较，第二个是 classical Dolbeault theorem。

**书内部分。** 第三章、附录 N、R 证明 Dolbeault resolution 的 sheaf cohomology
形式层；第二卷附录 P、Z 证明连续 splitting 使 boundary 与 cycle quotient 对
profinite 参数族局部可提升，并由此推出第一个同构。

**外部部分。** CS26 Theorem 2.14、Lemma 2.16 及其逆极限推论给 Fréchet 项的
liquid membership；classical Dolbeault lemma 和 Fredholm-Hodge 定理给连续
Green/Hodge splitting。不存在单独的笼统“liquid realization exactness”输入。

**边界.** 只知道 \(\bar\partial\) 有闭像，只能保证拓扑 cohomology 为 Hausdorff
Fréchet 空间；要推出第一个同构仍需局部提升，连续 Hodge splitting 是本书采用的
可核查充分条件。

**形式结论。** 若 cohomology 有限维，则 realization 后 cohomology 是 perfect liquid object。

## AR.3 相干有限性

**核心定理 AR.3（coherent cohomology finite-dimensionality）。** 对紧复空间 \(X\) 和相干解析层 \(\mathcal F\)，

$$
\dim_\mathbb C H^q(X,\mathcal F)<\infty.
$$

**书内部分。** 第四章、附录 L、M、X、AC、AN、AQ 证明接受 Grauert 或 Hodge-Fredholm 输入后有限性如何传播，并说明 Stein-Čech 计算本身不推出有限维性。

**外部部分。** Grauert direct image theorem、Hodge-Fredholm 定理或 Clausen-Scholze 对 compactness 的核心输入。

**凝聚意义。** 有限性使 \(R\Gamma(X,\mathcal F)\) 在 analytic/liquid 范畴中对应 finite/perfect 对象，并为 duality 和 Euler characteristic 提供有限性基础。

## AR.4 Serre 与 Grothendieck duality

**核心定理 AR.4（Serre duality in condensed/analytic language）。** 对 \(n\)-维紧复流形 \(X\) 和向量丛 \(E\)，有 perfect pairing

$$
H^i(X,E)\times H^{n-i}(X,E^\vee\otimes\omega_X)\to\mathbb C.
$$

一般相干层情形由 dualizing complex \(\omega_X^\bullet\) 表达为

$$
R\Gamma(X,\mathcal F)^\vee
\simeq
R\Gamma(X,R\mathcal Hom(\mathcal F,\omega_X^\bullet)).
$$

**书内部分。** 第五章、附录 J、O、AA、AD、AQ 证明链级配对、Hodge star 情形、有限 resolution 推广和 dualizing complex 形式后果。

**外部部分。** Trace theorem、dualizing complex 存在性、一般 perfectness 和 Clausen-Scholze 对 \(f^!\) 语言的建模相容。

**凝聚意义。** 对偶不仅是 vector space duality，还应与 \(f_!\dashv f^!\)、closed monoidal structure 和 analytic/liquid realization 相容。

## AR.5 GAGA

**核心定理 AR.5（GAGA theorem package）。** 对 proper complex algebraic variety \(Y\)，解析化 \(X=Y^{an}\) 给出 coherent sheaves、上同调和 bounded coherent derived categories 的比较：

$$
\operatorname{Coh}(Y)\simeq\operatorname{Coh}(X),
$$

$$
H^q(Y,\mathcal F)\cong H^q(X,\mathcal F^{an}),
$$

并诱导

$$
D^b_{\operatorname{coh}}(Y)\simeq D^b_{\operatorname{coh}}(X).
$$

**书内部分。** 第六章、附录 K、Q、Y、AI、AO、AQ 证明 properness 边界、exact equivalence 到 derived equivalence、projective GAGA 骨架和形式函数路线。

**外部部分。** Serre analytic finite generation、Grothendieck existence、形式函数定理和解析形式代数化。

**凝聚意义。** GAGA 允许 algebraic coherent geometry 与 analytic/liquid realization 在同一 \(R\Gamma\)、trace 和 \(K\)-theory 语言中比较。

## AR.6 HRR 与 GRR

**核心定理 AR.6（HRR/GRR theorem package）。** 对 smooth proper 情形，有

$$
\chi(X,E)=\int_X\operatorname{ch}(E)\operatorname{td}(T_X),
$$

一般 proper morphism \(f:X\to Y\) 下有 Grothendieck-Riemann-Roch：

$$
\operatorname{ch}(Rf_\ast E)\operatorname{td}(T_Y)
=
f_\ast(\operatorname{ch}(E)\operatorname{td}(T_X)).
$$

**书内部分。** 第七章、附录 P、U、AE、AK、AP、AQ 证明 Chern/Todd 形式代数、\(\mathbb P^n\) 线丛 HRR、一般 GRR 形式后果和 deformation-to-normal-cone 证明模块。

**外部部分。** Chern 类的完整几何构造、localized Chern character、excess intersection、deformation specialization 和 GRR 基本因子定理。

**凝聚意义。** 左侧 Euler characteristic 依赖 AR.3 的有限性；右侧 characteristic class 公式需要和 GAGA、pushforward、trace 相容。

## AR.7 Six functor 接口

**核心定理 AR.7（复几何中的 \(f_!,f^!,\otimes,R\mathcal Hom\) 接口）。** 对满足相应 six-functor 输入假设的 morphism \(f\)，复几何中的 pushforward、proper/compact-support pushforward、pullback、exceptional pullback、tensor 和 internal Hom 组成与 condensed/analytic language 相容的六函子结构。

**书内部分。** 第八章和第二卷附录 F、L 证明接受 \(f_!\)、projection formula 和 adjunction 后，\(f^!\) 与 internal Hom 公式如何推出。

**外部部分。** 完整 six functor formalism、proper base change、trace/counit 的构造和与 classical duality 的相容。

**边界。** 本卷只把 six functors 作为后续接口，不声称完成完整理论。

## AR.8 闭包表

| 定理块 | classical 输入 | Clausen-Scholze/analytic 输入 | 书内形式后果 |
| --- | --- | --- | --- |
| 建模 | 复解析空间、相干层 | condensed/analytic 表示 | 术语对照、范畴接口 |
| Dolbeault | Dolbeault lemma、连续 Hodge-Fredholm splitting | Fréchet 项的 liquid membership | 局部提升、cohomology 比较、perfect 性 |
| 有限性 | Grauert 或 Hodge theorem | compactness/perfectness 语言 | \(R\Gamma\) finite/perfect |
| Serre duality | trace、dualizing complex | \(f^!\)、closed monoidal 相容 | perfect pairing、Ext-Serre |
| GAGA | Serre/Grothendieck GAGA | analytic/coherent 比较 | derived equivalence、cohomology comparison |
| HRR/GRR | Chern/Todd/GRR | pushforward 与 trace 相容 | Euler characteristic 公式 |
| six functors | Verdier/Grothendieck duality | analytic six-functor package | 投影公式和 internal Hom 后果 |

## AR.9 核心闭包定理

**定理 AR.8（Clausen-Scholze 复几何输入闭包）。** 接受 AR.1-AR.7 中列出的外部输入后，第三卷的复几何主线闭合：Dolbeault、有限性、Serre/Grothendieck duality、GAGA 和 HRR/GRR 均能在 condensed/analytic/liquid 语言中表达，并且它们之间的形式依赖在书内闭合。

**证明.** AR.1 提供对象和范畴语言。AR.2 给 \(R\Gamma\) 的 Dolbeault/liquid 模型。AR.3 给有限性，使 derived dual、Euler characteristic 和 perfectness 有意义。AR.4 给 trace 和 duality。AR.5 把代数与解析相干几何比较。AR.6 把 Euler characteristic 与 characteristic classes 联系。AR.7 说明这些结构进入 six functor 接口。附录 AQ 已证明这些输入的组合没有逻辑缺环。证毕。

## 练习

1. 解释为什么 AR.3 是 AR.4 和 AR.6 的共同前提。
2. 说明 GAGA 的 properness 假设在 AR.5 中为何不可删。
3. 把 AR.2 写成“classical Hodge/Dolbeault 输入 + liquid membership 输入 + 书内局部提升
   推论”的三段式。
4. 说明 AR.7 为什么在本卷只能作为接口，而不能作为已完整证明的定理。
