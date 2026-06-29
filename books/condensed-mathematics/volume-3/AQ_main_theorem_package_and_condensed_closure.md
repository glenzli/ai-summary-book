# 附录 AQ：复几何主定理包与凝聚闭包

## AQ.0 目标

第三卷已经分别处理 Dolbeault、有限性、Serre duality、GAGA 和 Riemann-Roch。本附录把它们组织成一个主定理包，说明在接受输入定理后，凝聚/analytic 语言中的结论如何同时闭合。

本附录不新增深层输入；它证明输入定理之间的形式组合。

## AQ.1 输入包

固定一个紧复流形 \(X\)。设 \(\operatorname{Coh}(X)\) 为相干解析层范畴，\(D^b_{\operatorname{coh}}(X)\) 为有界相干导出范畴。

本附录使用以下输入：

1. Dolbeault resolution 和 liquid realization；
2. Grauert 或 Hodge-Fredholm 有限性；
3. Serre duality 或 Grothendieck-Serre duality；
4. 若 \(X=Y^{an}\) 来自 proper algebraic variety，则使用 GAGA；
5. 若 \(X\) smooth proper algebraic 或处在 HRR 适用范围，则使用 GRR/HRR；
6. Clausen-Scholze 的 condensed/analytic complex geometry 建模。

这些输入的外部部分见总登记表 [../INPUT_THEOREM_REGISTER.md](../INPUT_THEOREM_REGISTER.md)。

## AQ.2 有限性闭包

**定理 AQ.1（相干上同调有限性闭包）。** 接受 Grauert direct image theorem 或 Hodge-Fredholm 输入后，对任意

$$
\mathcal F\in\operatorname{Coh}(X)
$$

有

$$
\dim_\mathbb C H^q(X,\mathcal F)<\infty
$$

对所有 \(q\) 成立。因而

$$
R\Gamma(X,\mathcal F)
$$

在 \(D(\mathbb C)\) 中是有限维有界复形。

**证明.** Grauert 路线：取 \(f:X\to *\)。Grauert 定理给

$$
R^qf_\ast\mathcal F
$$

为点上的相干解析层，即有限维复向量空间。这等于 \(H^q(X,\mathcal F)\)。

Hodge-Fredholm 路线：若 \(\mathcal F\) 有有限局部自由 resolution，则附录 X 把向量丛情形有限性传播到 \(\mathcal F\)。一般相干层使用 Grauert 或 dualizing/Grauert 输入。有限维上同调只在有限多个 \(q\) 中非零，因此 \(R\Gamma\) 是有限维有界复形。证毕。

**推论 AQ.2（analytic/liquid 紧性口径）。** 若 liquid realization 与有限维复向量空间相容，则 \(R\Gamma(X,\mathcal F)\) 在 analytic/liquid 派生范畴中对应 compact/perfect 对象。

**证明.** 有限维复向量空间是单位对象的有限直和。有限有界复形由有限次 shift、cone 和有限直和构成。compact/perfect 性对这些操作封闭。证毕。

## AQ.3 Serre 对偶闭包

设 \(n=\dim_\mathbb C X\)，\(\omega_X\) 为 canonical bundle；奇异情形用 dualizing complex \(\omega_X^\bullet\)。

**定理 AQ.3（Ext-Serre duality 形式）。** 接受 Grothendieck-Serre duality 输入后，对 \(\mathcal F\in D^b_{\operatorname{coh}}(X)\) 有自然 perfect pairing

$$
H^i(X,\mathcal F)
\times
\operatorname{Ext}^{-i}_X(\mathcal F,\omega_X^\bullet)
\to
\mathbb C.
$$

在 \(X\) smooth 且 \(\mathcal F\) 为向量丛 \(E\) 时，它化为

$$
H^i(X,E)^\vee
\cong
H^{n-i}(X,E^\vee\otimes\omega_X).
$$

**证明.** Grothendieck-Serre duality 给导出同构

$$
R\Gamma(X,\mathcal F)^\vee
\simeq
R\Gamma(X,R\mathcal Hom(\mathcal F,\omega_X^\bullet)).
$$

取 cohomology 得第一式。有限性由 AQ.1 保证，使 derived dual 的 cohomology 与 ordinary linear dual 相容。smooth 向量丛情形中

$$
R\mathcal Hom(E,\omega_X[n])
\simeq
E^\vee\otimes\omega_X[n],
$$

得到第二式。证毕。

**推论 AQ.4（trace/counit 解释）。** 若 \(f:X\to *\)，则 AQ.3 的配对由

$$
f_! \dashv f^!
$$

的 counit/trace 诱导。

**证明.** 对 \(f:X\to *\)，\(f^!\mathbb C\simeq\omega_X^\bullet\)。闭幺半伴随给

$$
\operatorname{Map}(\mathbb C,f_\ast R\mathcal Hom(\mathcal F,\omega_X^\bullet))
\simeq
\operatorname{Map}(f_!\mathcal F,\mathbb C).
$$

右侧就是 \(R\Gamma(X,\mathcal F)^\vee\)。证毕。

## AQ.4 GAGA 闭包

设 \(Y\) 是 proper \(\mathbb C\)-scheme，\(X=Y^{an}\)。

**定理 AQ.5（GAGA theorem package）。** 接受 Serre GAGA 或形式 GAGA 输入后，解析化函子给等价

$$
\operatorname{Coh}(Y)
\simeq
\operatorname{Coh}(X),
$$

并对 \(\mathcal F\in\operatorname{Coh}(Y)\) 给上同调比较

$$
H^q(Y,\mathcal F)
\cong
H^q(X,\mathcal F^{an}).
$$

该等价提升为

$$
D^b_{\operatorname{coh}}(Y)
\simeq
D^b_{\operatorname{coh}}(X).
$$

**证明.** coherent sheaf 范畴等价和上同调比较是 GAGA 输入。由于等价是 exact，逐项作用于 bounded complex，并保持 quasi-isomorphism、cone 和 cohomology sheaf。因此诱导 bounded coherent derived categories 的等价。对 \(R\Gamma\)，用 stupid filtration 和上同调比较得到 hypercohomology 比较。证毕。

**推论 AQ.6（代数-解析有限性与对偶相容）。** 若代数侧 Serre duality 或 HRR 成立，则经 AQ.5 转移到解析侧；反向亦然。

**证明.** GAGA 等价保持 coherent sheaves、Ext groups、bounded derived Hom 和 cohomology。若 trace map 在等价下相容，则 pairing 和 Euler characteristic 被保持。HRR 两侧分别由 \(K_0\)、Chern character、Todd class 和 pushforward 构成；这些对象在 GAGA 比较下相容时，公式转移。证毕。

## AQ.5 HRR/GRR 闭包

设 \(X\) 为 smooth proper complex variety 或紧复流形中 HRR 已建立的情形。

**定理 AQ.7（HRR theorem package）。** 对 \(E\in K^0(X)\)，接受 HRR 输入后有

$$
\chi(X,E)=
\int_X\operatorname{ch}(E)\operatorname{td}(T_X).
$$

若 \(f:X\to Y\) proper，接受 GRR 输入后有

$$
\operatorname{ch}(Rf_\ast E)\operatorname{td}(T_Y)
=
f_\ast(\operatorname{ch}(E)\operatorname{td}(T_X)).
$$

**证明.** 第一式是第二式在 \(Y=*\) 时的特例。书内已证明：

1. Euler characteristic 对 distinguished triangle 可加；
2. Chern character 对 \(K^0\) 加法和乘法相容；
3. Todd class 对 exact sequence 乘法相容；
4. proper pushforward 复合相容；
5. projective space line bundle 情形可直接计算。

一般式的深层部分是 GRR 输入；接受后，以上形式性质保证公式可在 \(K\)-群上稳定传播。证毕。

**推论 AQ.8（与有限性和 GAGA 的相容）。** 在 AQ.5 的情形中，HRR 的左侧可用解析上同调计算，右侧可用代数或解析特征类计算。

**证明.** 左侧

$$
\chi(X,E)=\sum_i(-1)^i\dim H^i(X,E)
$$

由 AQ.1 有限。GAGA 给代数与解析 cohomology 比较。特征类比较是 GAGA/Betti-de Rham 比较输入；接受后右侧一致。证毕。

## AQ.6 凝聚/analytic 总闭包

**定理 AQ.9（凝聚复几何主闭包）。** 接受 AQ.1-AQ.8 的输入定理和 Clausen-Scholze 建模后，紧复几何中的以下对象可在 condensed/analytic 语言中同时表达，并保持经典结论：

1. \(R\Gamma(X,\mathcal F)\) 的有限性；
2. Serre/Grothendieck duality 的 trace pairing；
3. GAGA 的范畴等价与上同调比较；
4. HRR/GRR 的 Euler characteristic 与 characteristic class 公式；
5. liquid Dolbeault 模型与经典 Dolbeault cohomology 的比较。

**证明.** Clausen-Scholze 建模把复解析对象、相干层和相关函数空间送入 condensed/analytic 框架。第二卷附录 P 给 Fréchet/liquid realization 的类型检查，第三卷附录 N、R 给 Dolbeault resolution 的 sheaf cohomology 计算形式层。AQ.1-AQ.8 分别证明有限性、对偶、GAGA 和 RR 在接受输入后的闭包。各定理使用同一 \(R\Gamma\)、trace、derived Hom、pushforward 和 \(K\)-理论类，因此在依赖图中相容。证毕。

## AQ.7 基本完本意义

本附录说明：第三卷在“输入定理型严格教材”标准下可以闭合为一个 theorem package。它没有重证外部深定理，但已经证明外部输入一旦接受，主结论之间的逻辑关系不再缺环。

## 练习

1. 用 AQ.1 证明 \(R\Gamma(X,\mathcal F)\) 是 \(D(\mathbb C)\) 中 perfect object。
2. 从 AQ.3 推出 Riemann surface 上线丛的 Serre duality 公式。
3. 证明 exact equivalence of abelian categories 诱导 bounded derived categories 的等价。
4. 说明 HRR 为什么需要有限性才能把左侧解释为 Euler characteristic。
