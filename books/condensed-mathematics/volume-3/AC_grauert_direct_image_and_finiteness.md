# 附录 AC：Grauert 直接像定理与有限性

## AC.0 目标

相干上同调有限性可由 Grauert direct image theorem 推出。本附录给出该深层定理的精确形式，并证明它如何推出卷三使用的有限性结论。

Grauert 定理本身作为经典复几何输入；其证明依赖 semi-continuity、Banach analytic methods 和局部有限 presentation。

## AC.1 Grauert direct image theorem

**输入定理 AC.1（Grauert direct image theorem）.** 设

$$
f:X\to Y
$$

是 proper holomorphic map，$\mathcal F$ 是 $X$ 上相干解析层。则对每个 $q\ge0$，

$$
R^qf_\ast\mathcal F
$$

是 $Y$ 上相干解析层。

此外，若 $f$ 的 fibers 维数有界，则 $R^qf_\ast\mathcal F=0$ 对足够大的 $q$ 成立。

**输入定理 AC.2（Grauert semi-continuity/base change 形式）.** 在 AC.1 的假设下，函数

$$
y\mapsto \dim_{\mathbb C}H^q(X_y,\mathcal F|_{X_y})
$$

上半连续。在满足局部自由性或维数常数条件的点，base change 映射

$$
(R^qf_\ast\mathcal F)_y\otimes_{\mathcal O_{Y,y}}\mathbb C(y)
\to
H^q(X_y,\mathcal F|_{X_y})
$$

为同构。

## AC.2 紧空间有限性

**定理 AC.3（coherent cohomology finite-dimensionality）.** 若 $X$ 是紧复空间，$\mathcal F$ 是相干解析层，则

$$
\dim_{\mathbb C}H^q(X,\mathcal F)<\infty
$$

对所有 $q$ 成立。

**证明.** 令

$$
f:X\to *
$$

为到一点的 proper holomorphic map。由 Grauert direct image theorem，

$$
R^qf_\ast\mathcal F
$$

是点上的相干解析层。点上的相干解析层等价于有限维复向量空间。又

$$
R^qf_\ast\mathcal F
=
H^q(X,\mathcal F).
$$

故 $H^q(X,\mathcal F)$ 有限维。证毕。

## AC.3 与向量丛 Hodge 方法的关系

附录 Z-AA 证明了向量丛情形的 Hodge/Fredholm 路线。附录 X 证明若 $\mathcal F$ 有全局有限局部自由 resolution，则向量丛有限性传播到 $\mathcal F$。

Grauert 定理更强，因为它不要求 $\mathcal F$ 有全局有限局部自由 resolution，也允许奇异紧复空间。

## AC.4 Derived direct image 形式

在导出范畴中，AC.1 可写为：

**推论 AC.4.** 若 $f:X\to Y$ proper，且 $\mathcal F\in D^b_{\operatorname{coh}}(X)$，则

$$
Rf_\ast\mathcal F\in D^b_{\operatorname{coh}}(Y).
$$

**证明.** 对有界复形取 hypercohomology spectral sequence

$$
E_2^{p,q}=R^pf_\ast H^q(\mathcal F)
\Rightarrow
H^{p+q}(Rf_\ast\mathcal F).
$$

每个 $H^q(\mathcal F)$ 相干，AC.1 给 $R^pf_\ast H^q(\mathcal F)$ 相干。有界性保证每个总次数只涉及有限项，相干 sheaf 范畴对 kernel、cokernel 和 extension 封闭，故 abutment 相干。证毕。

## AC.5 condensed/analytic 接口

Clausen-Scholze 语境中，Grauert finite-dimensionality 需要被增强为 analytic/liquid 范畴中的 finiteness statement：

1. $R\Gamma(X,\mathcal F)$ 的底层 cohomology 有限维；
2. 该对象在 analytic/liquid 派生范畴中是 perfect 或 compact；
3. proper pushforward 与 classical $Rf_\ast$ 比较；
4. trace 和 duality 与该比较相容。

AC.3 只处理第一项。

## 练习

1. 证明点上的相干解析层等价于有限维复向量空间。
2. 用 AC.1 推出 compact Riemann surface 上相干层上同调有限维。
3. 解释 AC.4 中为什么需要有界复形。
4. 说明 Grauert finiteness 比附录 X 的有限性传播强在哪里。
