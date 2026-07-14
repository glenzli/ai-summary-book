# 附录 AA：由 Hodge 理论推出向量丛 Serre 对偶

## AA.0 目标

一般 Serre duality 的深层部分是配对完美性。附录 J 已给出链级配对与导出范畴形式，附录 T 证明了 $\mathbb P^n$ 线丛模型。本附录补充向量丛情形的经典 Hodge 理论证明结构。

设 $X$ 是配备 Hermitian 度量的紧复流形，$\dim_{\mathbb C}X=n$，$E$ 是配备
Hermitian 度量的全纯向量丛。目标是得到完美配对

$$
H^q(X,\mathcal O(E))
\times
H^{n-q}(X,\mathcal O(E^\vee\otimes\omega_X))
\to
\mathbb C.
$$

## AA.1 Dolbeault 配对

Dolbeault 复形给

$$
H^q(X,\mathcal O(E))
\cong
H^q(\Gamma\mathcal A^{0,\bullet}(E),\bar\partial).
$$

同时

$$
H^{n-q}(X,\mathcal O(E^\vee\otimes\omega_X))
\cong
H^{n-q}(\Gamma\mathcal A^{0,\bullet}(E^\vee\otimes\omega_X),\bar\partial).
$$

若

$$
\alpha\in\mathcal A^{0,q}(E),
\qquad
\beta\in\mathcal A^{0,n-q}(E^\vee\otimes\omega_X),
$$

则 contraction $E\otimes E^\vee\to\mathcal O_X$ 和 wedge product 给出 $(n,n)$-形式

$$
\langle\alpha,\beta\rangle.
$$

定义

$$
P(\alpha,\beta)=\int_X\langle\alpha,\beta\rangle.
$$

**命题 AA.1.** $P$ 下降到 Dolbeault cohomology。

**证明.** Leibniz 规则给

$$
\bar\partial\langle\alpha,\beta\rangle
=
\langle\bar\partial\alpha,\beta\rangle
 +(-1)^q\langle\alpha,\bar\partial\beta\rangle.
$$

紧无边界流形上 Stokes 定理给 $\int_X\bar\partial(\cdot)=0$。因此改变任一变量一个 $\bar\partial$-边界不改变积分，闭形式之间的配对只依赖 cohomology 类。证毕。

## AA.2 Hodge star 输入

**输入定理 AA.2（Hodge star 与 Serre duality）.** Hermitian 度量给出共轭线性 Hodge star 型同构

$$
\star_E:
\mathcal A^{0,q}(E)
\to
\mathcal A^{0,n-q}(E^\vee\otimes\omega_X)
$$

满足：

1. $\star_E$ 把 $\bar\partial$-harmonic $E$-值 $(0,q)$-形式同构到 $E^\vee\otimes\omega_X$-值 $(0,n-q)$-harmonic 形式；
2. 在固定体积形式、复共轭和楔积次序后，可给 $\star_E$ 乘以只依赖
   $(n,q)$ 的单位复数作归一化，使对 harmonic $\alpha$ 有
   $$
   \int_X\langle\alpha,\star_E\alpha\rangle
   =
   \|\alpha\|^2.
   $$

该定理依赖 Hermitian 线性代数和 Dolbeault Laplacian 与 Hodge star 的相容性。

## AA.3 完美性

**定理 AA.3（向量丛 Serre 对偶）.** 在 AA.2 的输入下，配对

$$
H^q(X,\mathcal O(E))
\times
H^{n-q}(X,\mathcal O(E^\vee\otimes\omega_X))
\to
\mathbb C
$$

是完美配对。

**证明.** 由附录 Z，每个 cohomology 类有唯一 harmonic 代表。因此配对可限制到有限维 harmonic 空间

$$
\mathcal H^{0,q}(E)
\times
\mathcal H^{0,n-q}(E^\vee\otimes\omega_X).
$$

若 $0\ne[\alpha]$，取 harmonic 代表 $\alpha\ne0$。由 AA.2，$\star_E\alpha$ 是第二个 harmonic 空间中的元素，并且

$$
P(\alpha,\star_E\alpha)=\|\alpha\|^2\ne0.
$$

因此左核为零。有限维性给左侧到右侧对偶空间的线性映射为单射。两边维数相同，因为 $\star_E$ 给 harmonic 空间的共轭线性同构，所以该单射为同构。右核同理为零。证毕。

## AA.4 从向量丛到相干层

若相干层 $\mathcal F$ 有有限局部自由 resolution，则附录 O 的同调代数把向量丛 Serre 对偶推广到

$$
R\Gamma(X,\mathcal F)^\vee
\simeq
R\Gamma(X,R\mathcal Hom(\mathcal F,\omega_X[n])).
$$

若没有全局有限局部自由 resolution，则需要 dualizing complex 和 Grothendieck-Serre duality 的一般理论。本附录不证明该一般理论。

## AA.5 condensed/analytic 接口

在 condensed/analytic 版本中，AA.3 的每个对象还带有拓扑或 analytic 结构。要得到范畴内的 Serre duality，必须证明：

1. harmonic projection 是 analytic/liquid 范畴中的 morphism；
2. finite-dimensional harmonic spaces 取通常欧氏拓扑后，对应有限自由
   $\underline{\mathbb C}$-模；它们一般不是离散凝聚向量空间；
3. integration trace 与 $f_!\dashv f^!$ 的 counit 相同；
4. Hodge star 与 analytic realization 相容。

这些是 Clausen-Scholze 版本相干对偶需要处理的结构性问题。

## 练习

1. 证明命题 AA.1 中边界项积分为零。
2. 在有限维 Hermitian 向量空间中证明非零向量与其 Hodge dual 配对非零。
3. 解释 AA.3 中为什么有限维性可把单射提升为同构。
4. 说明从向量丛 Serre duality 到相干层 Serre duality 需要哪些额外假设。
