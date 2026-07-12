# 第十三章：pairs of pants、tropical degeneration 与 hypersurfaces in $(\mathbb C^\ast)^n$

## 本章目标

本章说明 pair-of-pants decomposition 和 tropical degeneration 如何把高维 HMS 的计算局部化。重点是 hypersurfaces in algebraic tori 与 toric Landau-Ginzburg mirror 的范畴形态。

## 依赖前置知识

需要第六章 wrapped categories、第七章 stops、第十二章 Calabi-Yau hypersurfaces 的证明策略。

## 13.1 Pair of pants

**定义 13.1.** $n$-dimensional pair of pants 是
$$
P^n=\{(z_1,\ldots,z_{n+1})\in(\mathbb C^\ast)^{n+1}\mid z_1+\cdots+z_{n+1}=1\}
$$
或其等价的 toric/open hypersurface 模型。

**解释 13.2.** 高维 hypersurface 的 large complex structure degeneration 可局部分解为 pairs of pants。A-side 上对应 wrapped 或 partially wrapped pieces；B-side 上对应 toric 或 singular pieces。

## 13.2 Tropical degeneration

**定义 13.3.** 一个 Laurent polynomial family
$$
f_t(z)=\sum_{m\in A} c_m(t)z^m
$$
的 tropicalization 是函数
$$
\operatorname{Trop}(f)(u)=\min_{m\in A}\{v(c_m)+\langle m,u\rangle\}.
$$
其非光滑 locus 给出 tropical hypersurface。

**解释 13.4.** Tropical hypersurface 编码 degeneration 的组合骨架。Pair-of-pants decomposition 可理解为 tropical hypersurface 的局部顶点模型。

**命题 13.5.** 若一个 degeneration 的 tropical hypersurface 可由标准局部
模型覆盖，该覆盖在 A-side 提升为定义 15.1 的 Weinstein sectorial cover，
且局部 Fukaya models 与这些 inclusions 相容，则全局 category 可由局部
pair-of-pants categories 胶合。

**证明.** 这是定理 15.3 的形式结论。Weinstein sectorial descent 把全局
wrapped Fukaya category 识别为局部 categories 沿交叠的 homotopy colimit。
每个局部 piece 为 pair-of-pants 模型，故全局由它们胶合。证毕。

**警告 13.6.** 命题 13.5 的关键假设是 Weinstein sectorial descent，这是
外部输入；只有有限交存在或 tropical cover 与 Liouville sector 结构兼容，
都还不足以替代定义 15.1 的 Weinstein hypotheses。

## 13.3 Hypersurfaces in $(\mathbb C^\ast)^n$

**外部输入定理 13.7（Abouzaid-Auroux hypersurface HMS）.** 对 $(\mathbb C^\ast)^n$ 中 maximally degenerating hypersurface families 及其 mirror toric Landau-Ginzburg A-models，Abouzaid-Auroux 证明了 HMS 型结果；核心技术包括 fiberwise wrapped Fukaya category 和 admissible Lagrangian 的 Floer cohomology 与 hypersurface regular functions 的比较。  
来源：Abouzaid-Auroux, *Homological mirror symmetry for hypersurfaces in $(\mathbb C^\ast)^n$*。

**解释 13.8.** 该结果不是简单的 compact Fukaya category 等价，而使用适合 fibration 的 fiberwise wrapped 版本。写入本书主线时必须保留这个 category 名称和假设。

## 13.4 Categorical resolution

**外部输入定理 13.9（higher-dimensional pants）.** 高维 pair-of-pants 的 wrapped/partially wrapped Fukaya category 可与某些 singular affine varieties 的 derived categories 或 categorical resolutions 联系起来。  
来源：Lekili-Polishchuk 关于 higher-dimensional pairs of pants 的工作。

**解释 13.10.** Pair-of-pants 模型把 singular B-side 和 stopped/wrapped A-side 直接联系起来。Stop removal 对应 categorical resolution 到 singular category 的 localization。

## 13.5 HMS 中的局部到整体格式

**模板 13.11.**

1. 选取 hypersurface degeneration 和 tropical skeleton。
2. 把 A-side Liouville space 切成 sectorial pieces。
3. 对每个 piece 识别其 wrapped category。
4. 在 B-side 构造相应 affine/toric cover 或 categorical resolution。
5. 比较两边 descent diagrams。
6. 取 homotopy colimit 或 Morita colimit 得到全局 HMS。

**命题 13.12.** 若模板 13.11 中每个局部比较为 Morita equivalence，且两边 descent diagrams 在 Morita homotopy category 中相容，则全局 categories Morita equivalent。

**证明.** Morita equivalences 在 homotopy colimit 下保持：若两个 diagrams 由逐点等价和相容自然变换连接，则它们的 homotopy colimits 等价。全局 categories 由 descent 识别为这些 colimits，因此得到结论。证毕。

## 本章小结

Pair-of-pants 和 tropical degeneration 提供高维 HMS 的局部计算语言。它们把复杂 hypersurface 分解为可控局部 pieces，再通过 wrapped Fukaya category 的 descent 和 B-side categorical gluing 重建全局等价。

## 练习

**练习 13.1.** 写出 $P^1$ 与三点去除球面的同构关系。

**练习 13.2.** 计算二项式和三项式 Laurent polynomial 的 tropical hypersurface。

**练习 13.3.** 解释为什么 sectorial descent 是命题 13.5 的非形式输入。

**练习 13.4.** 按模板 13.11 写出一个 pair-of-pants gluing 的形式证明框架。
