# 附录 X：从向量丛情形传播相干上同调有限性

## X.0 目标

卷三有限性定理的深层输入是 Grauert finiteness 或 Hodge-Fredholm 理论。本附录证明一个重要的形式化约：

> 若相干层有有限局部自由 resolution，并且每个向量丛的上同调有限维，则该相干层的上同调有限维。

这不能替代 Grauert finiteness，因为并非所有紧复空间上的相干层都给定全局有限向量丛 resolution。但它说明向量丛 Hodge 理论如何通过同调代数传播到相干层。

## X.1 有限复形的超上同调

设 $X$ 是紧复空间，$E^\bullet$ 是有界复形

$$
E^{-m}\to\cdots\to E^0
$$

其中每个 $E^{-p}$ 是全纯向量丛。

**假设 X.1（向量丛有限性）.** 对每个全纯向量丛 $E$ 和所有 $q$，

$$
\dim_{\mathbb C}H^q(X,E)<\infty.
$$

**命题 X.2.** 在 X.1 下，超上同调

$$
\mathbb H^k(X,E^\bullet)
$$

对所有 $k$ 有限维。

**证明.** 使用 hypercohomology spectral sequence

$$
E_1^{p,q}=H^q(X,E^p)
\Rightarrow
\mathbb H^{p+q}(X,E^\bullet).
$$

因为 $E^\bullet$ 有界，且 $X$ 有有限维数，固定总次数 $k$ 只涉及有限多个 $E_1^{p,q}$。每个 $E_1^{p,q}$ 由 X.1 有限维。谱序列每一页由前一页的 kernel 和 cokernel 构成，有限维性逐页保持。收敛过滤有限，故 abutment $\mathbb H^k$ 有限维。证毕。

## X.2 Resolution 推出有限性

**定理 X.3.** 设 $\mathcal F$ 是相干解析层，并有有限局部自由 resolution

$$
E^\bullet\to\mathcal F.
$$

若 X.1 成立，则

$$
\dim_{\mathbb C}H^k(X,\mathcal F)<\infty
$$

对所有 $k$ 成立。

**证明.** resolution $E^\bullet\to\mathcal F$ 是 quasi-isomorphism，因此

$$
R\Gamma(X,\mathcal F)
\simeq
R\Gamma(X,E^\bullet).
$$

右侧的 cohomology 是 $\mathbb H^k(X,E^\bullet)$，由命题 X.2 有限维。证毕。

## X.3 两项 resolution 的长正合证明

为说明 X.3 的机制，考虑短正合列

$$
0\to E_1\to E_0\to\mathcal F\to0
$$

其中 $E_i$ 是向量丛。长正合列给

$$
\cdots\to H^q(X,E_1)\to H^q(X,E_0)
\to H^q(X,\mathcal F)
\to H^{q+1}(X,E_1)\to\cdots.
$$

于是 $H^q(X,\mathcal F)$ 是有限维向量空间之间某个 cokernel 到 kernel 的扩张；因此有限维。这是谱序列证明在长度一情形下的展开。

## X.4 与 Grauert finiteness 的关系

**输入定理 X.4（Grauert finiteness）.** 若 $X$ 是紧复空间，$\mathcal F$ 是相干解析层，则 $H^q(X,\mathcal F)$ 有限维。

X.3 只证明以下条件性命题：

1. $X$ 上每个向量丛上同调有限维。
2. $\mathcal F$ 有全局有限局部自由 resolution。

Grauert finiteness 不要求第二条，因此严格强于 X.3 的形式化约。

## X.5 condensed/analytic 版本的类型检查

进入 condensed/analytic 语言时，X.3 的每一步需要替换为相应范畴中的语句：

1. $E^\bullet\to\mathcal F$ 是 analytic 派生范畴中的 quasi-isomorphism。
2. $R\Gamma$ 是目标 analytic/liquid 派生范畴中的导出全局截面。
3. 谱序列来自有界过滤或 t-structure。
4. “有限维”应解释为带通常欧氏拓扑的有限维复向量空间；其凝聚化是有限自由
   $\underline{\mathbb C}$-模，不是对底层集合赋离散拓扑。

如果只在普通向量空间层面证明有限性，不能自动推出所有拓扑增强结构相容。

## 练习

1. 写出三项 resolution 情形下证明有限性的两个短正合列步骤。
2. 证明命题 X.2 中有限过滤的 subquotient 有限维推出整体有限维。
3. 给出 X.3 不适用于没有全局有限局部自由 resolution 的相干层的原因。
4. 解释 Grauert finiteness 比 X.3 多解决了哪个问题。
