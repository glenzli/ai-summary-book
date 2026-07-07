# 附录 F：三角范畴和稳定 infinity-范畴翻译表

## 本附录目标

Motivic homotopy 的早期文献大量使用模型范畴和三角范畴，而本书主体使用 stable presentable infinity-categories。本附录给出翻译规则，防止把同伦范畴层面的同构误当作 infinity-范畴中的相干等价。

## 依赖前置知识

需要 stable infinity-categories、homotopy categories、triangulated categories、exact functors、cofiber sequences、adjunctions 和 mapping spectra。

## F.1 从 stable infinity-category 到 triangulated category

**定理 F.1.** 若 `\mathcal C` 是 stable infinity-category，则其 homotopy category `h\mathcal C` 有自然 triangulated category 结构。

**定义 F.2.** `\mathcal C` 中的 cofiber sequence

$$
X\to Y\to Z
$$

在 `h\mathcal C` 中给出 distinguished triangle

$$
X\to Y\to Z\to \Sigma X.
$$

**命题 F.3.** Stable infinity-category 中的 fiber sequence 与 cofiber sequence 等价。

**证明.** Stable infinity-category 定义要求 pointed、有限极限和有限余极限存在，且 pullback square 等价于 pushout square。由此任意 map 的 fiber 和 cofiber 通过同一 bicartesian square 相关，故 fiber sequence 和 cofiber sequence 是同一结构的两个方向。`\square`

## F.2 Exact functors

**定义 F.4.** Stable infinity-categories 之间的 functor 称为 exact，若保持有限极限；等价地保持有限余极限。

**命题 F.5.** Exact functor 诱导 triangulated functor。

**证明.** Exact functor 保持零对象和 cofiber sequences，因此在 homotopy categories 上保持 shift 和 distinguished triangles。`\square`

## F.3 Mapping spectra and Hom groups

**定义 F.6.** Stable infinity-category 中对象 `X,Y` 的 mapping spectrum 记为

$$
\operatorname{Map}_{\mathcal C}^{sp}(X,Y).
$$

Homotopy category 中的 Hom 群为

$$
\operatorname{Hom}_{h\mathcal C}(X,Y)=\pi_0\operatorname{Map}_{\mathcal C}(X,Y).
$$

**命题 F.7.** 对整数 `n`，

$$
\pi_n\operatorname{Map}_{\mathcal C}^{sp}(X,Y)
\simeq
\operatorname{Hom}_{h\mathcal C}(\Sigma^nX,Y).
$$

**证明.** Mapping spectrum 的第 `n` 个同伦群按定义由 suspension 坐标计算。Stable adjunction 给出 `\pi_n Map(X,Y)\simeq\pi_0 Map(\Sigma^nX,Y)`，右侧即 homotopy category Hom。`\square`

## F.4 Adjunctions and enhancements

**命题 F.8.** Infinity-categorical adjunction 诱导 homotopy category adjunction。

**证明.** 若 `L\dashv R`，则有 mapping space 等价

$$
\operatorname{Map}(LX,Y)\simeq\operatorname{Map}(X,RY).
$$

取 `\pi_0` 得 Hom 集自然同构，故在 homotopy categories 上为伴随。`\square`

**注 F.9.** 反向不成立：homotopy category 上的伴随不一定提升为 infinity-categorical adjunction，因为缺少 mapping spaces 的相干信息。

## F.5 Six operations translation

**定义 F.10.** 文献中的 triangulated six functors

$$
f^*,f_*,f_!,f^!,\otimes,\underline{\operatorname{Hom}}
$$

若来自 stable infinity-categorical six functor formalism，则它们是取 homotopy category 后的影子。

**命题 F.11.** 若 base-change map 在 infinity-category 中为等价，则在 homotopy category 中为同构。

**证明.** Infinity-category 中的等价在 homotopy category 中变为同构。`\square`

**注 F.12.** 若文献只证明 triangulated category 中的同构，不能自动推出 infinity-categorical coherence。正式比较需要 enhancement theorem。

## F.6 本附录小结

Stable infinity-category 提供相干层级，triangulated category 提供一阶同伦影子。早期 motivic 文献中的许多定理以 triangulated form 出现；本书使用时必须说明是否需要 enhancement 和 higher coherence。

## 练习

**练习 F.1.** 从 cofiber sequence 写出 distinguished triangle。

**练习 F.2.** 证明 exact functor 保持 distinguished triangles。

**练习 F.3.** 推导公式 `\pi_n Map(X,Y)\simeq Hom(\Sigma^nX,Y)`。

**练习 F.4.** 解释为什么 homotopy category 同构不足以保证 higher coherence。

**练习 F.5.** 找出一个六操作定理，说明其 triangulated 版本和 infinity-categorical 版本的差别。
