# 第十六章：Cubical Type Theory、计算单值性与模型

## 本章目标

本章讨论 HoTT 的元理论背景：simplicial model、cubical type theory、计算单值性、canonicity、directed/simplicial type theory、two-level type theory、cohesive HoTT 和模型比较。这里的内容属于外部输入，不作为前面内部证明的替代。对象语言、元语言和实现语言的边界见附录 Z。

## 依赖前置知识

本章依赖单值性、HIT 和模型比较。读者应区分对象语言中的类型论证明与元语言中的模型论证明。

## 16.1 Simplicial model

**定理 16.1（单值基础的 simplicial model，外部输入）.** Kapulkin、Lumsdaine、Voevodsky 构造的 simplicial set 模型证明了单值性与相应类型论规则的一致性背景。

**使用边界。** 模型论证明说明规则不导致矛盾，并解释同伦语义。它不在对象语言中给出任意具体定理的项。

## 16.2 Cubical type theory

**事实 16.2.** Cubical type theory 引入区间对象、面格、路径类型和 Glue 类型等结构，给出单值性的构造性解释。

**定理 16.3（计算单值性，外部输入）.** Cohen、Coquand、Huber、Mörtberg 的 cubical type theory 给出了 univalence 的计算性解释，使单值性不只是外部公理。

**验证状态。** 见附录 Z.3。公理化 HoTT 中这仍是模型/元理论输入；cubical 口径中则体现为区间、Glue 和 transport 计算规则。

**警告 16.4.** Cubical type theory 不是“HoTT Book 加上一个实现细节”。它改变了底层类型论的规则和计算行为。把 cubical 口径下的证明翻译到公理化 HoTT 需要逐项检查所用规则。

## 16.3 Canonicity 与 normalization

**定义 16.5.** Canonicity 粗略说：若闭项 $n:\mathbb N$ 可类型检查，则它计算到某个标准自然数。

**事实 16.6.** 对不同 cubical 系统，canonicity、normalization 和 decidability of type checking 是独立的元理论课题。

**验证状态：研究边界。** 见附录 Z.4。本书只记录概念和来源方向，不证明这些元定理，也不把 canonicity 当作对象语言中的消去原则。

## 16.4 模型比较

**事实 16.7.** Cubical 模型有多种变体，包括 de Morgan cubical sets、cartesian cubical sets 和其他 presheaf/cubical categories。不同模型支持的连接、退化、填充和 Glue 结构不同。Orton-Pitts 对 univalence 的分解和 cubical set 语义提供了重要参考。

**来源边界。** Coherence 和 strictification 可参考 Lumsdaine-Warren 的 local universes model；universe hierarchy 和 Grothendieck topos 语义可参考 Gratzer-Shulman-Sterling 的 strict universes 工作；cubical 计算规则应回到 cubical type theory 的原始论文与模型论文献。

**事实 16.8（weak/categorical univalence 的分离）.** 某些弱形式的单值性，例如 universe wild category 的 categorical univalence，并不推出函数外延性。Cavallo-Höfer 2026 给出模型分离结果；见附录 AO.1。

**教材后果。** 第六章使用的仍是 universe univalence，不得用 categorical univalence 替代；附录 T 中单值性推出函数外延性的链条只适用于相应强口径。

**事实 16.9（interval reversal）.** Cubical interval 是否带 reversal、connections、composition 和 Glue 结构会改变语法与模型。Cavallo-Sattler 2026 证明了若干 self-dual interval theories 中加入 reversal 的保守性，并给出 strict cubical type theory 模型；见附录 AO.3。

**事实 16.10（strict Rezk completion 与 homotopy canonicity）.** Strict Rezk completion 为 HoTT 的 homotopy canonicity 提供新的元理论路线。该结果与第十四章范畴 Rezk completion 有类比，但层级不同：前者是模型/语法元理论，后者是对象语言中的单值范畴构造；见附录 AO.2。

## 16.5 Directed / Simplicial type theory

**事实 16.11.** Simplicial type theory 和 directed type theory 引入 directed hom 类型
$$
\mathsf{hom}_A(a,b),
$$
它不是 identity type，不能自动反向，也不能直接当作路径。

**事实 16.12（directed univalence）.** 在相应的 simplicial HoTT 口径中，离散类型宇宙的 directed hom 可与函数类型等价：
$$
\mathsf{hom}_{\mathcal S}(A,B)\simeq(A\to B).
$$
这是一种 directed univalence，不是第六章的路径-等价单值性；研究接口见附录 AN，规则核见附录 AS。

**事实 16.12.1（Rezk/Segal 高阶范畴接口）.** Synthetic $\infty$-category type theory 可把 Segal condition、Rezk completeness、functor Rezk object 和 dependent Yoneda 作为对象语言结构处理。附录 BB 给出本书使用的 Rezk/Segal 接口；附录 AX 给出 simplicial 语义接口。

## 16.6 HIIT、QIIT 与计算 HIT 语义

**事实 16.12.2（QIIT 元理论边界）.** Higher inductive-inductive types 和 quotient inductive-inductive types 可表达语法商、Cauchy 实数和复杂代数对象，但其一般存在性、初始代数语义、strict positivity 和 canonicity 需要单独元理论。接口见附录 BC。

**事实 16.12.3（cubical HIT 计算语义）.** Cubical 系统可为许多 HIT 提供更强的计算行为，但不能把“有 HIT 构造子”自动升级为所有路径计算 judgmental。每个签名必须分别核查 composition/filling、universe 和 canonicity。

**事实 16.12.3.1（逻辑公理与计算性）。** LEM、choice、resizing 等原则不由 cubical univalence 的计算解释自动提供。若加入这些原则，应按附录 BL 标注 classical mode，并重新解释 canonicity 或数据计算。

## 16.7 Cohesive 与几何对象语言

**事实 16.12.4（cohesive HoTT）。** Cohesive HoTT 加入 shape、discrete、global sections、codiscrete 等模态和 exactness 条件，用于表达几何、拓扑和微分对象。附录 AT 给出模态接口，附录 BD 给出 SDG、de Rham 和 Zariski 使用边界。

**边界 16.12.5.** Cohesive HoTT、synthetic differential geometry 和 synthetic algebraic geometry 不是附录 BA 构造性分析的推论。它们需要额外对象语言和模型。

## 16.8 Two-level type theory

**事实 16.12.6（strict equality layer）。** Two-level type theory 同时具有外部 strict equality 层和内部 fibrant HoTT 层，可用于表达 semisimplicial types、Reedy fibrant diagrams 和元理论构造。接口见附录 BG。

**边界 16.12.7.** Strict equality 不是 HoTT identity path。用 strict equality 关闭 semisimplicial 相干不能自动给出内部路径等式；所有比较都需要明确桥接原则。

## 16.9 对本书的影响

**原则 16.13.** 本书正文分三种阅读方式：

1.  公理化 HoTT 阅读：函数外延性、单值性、HIT 和截断作为规则或公理加入；
2.  Cubical 阅读：单值性和部分 HIT 具有计算性实现，但需要接受 cubical primitives。
3.  Directed/simplicial 阅读：额外引入 directed hom、Segal 条件或 simplicial primitives；规则核见附录 AS，且这些规则不回流到前文 identity type 证明。
4.  Cohesive/geometry 阅读：额外引入 shape、cohesive modalities、infinitesimal objects 或 Zariski gluing；接口见附录 AT 和 BD。
5.  Two-level 阅读：额外区分 strict 外部层和 fibrant HoTT 层；接口见附录 BG。

后续若把某证明称为“计算性”，必须说明采用第二种阅读方式。

**边界原则。** 附录 Z.7 给出本书各章与 cubical/HIT 元理论的接口。后续新增 HIT 或 cubical 计算规则时，必须同步记录形成、构造、消去、计算规则和元理论假设。

## 本章小结

HoTT 的严谨性来自内部类型论与外部模型论的双重控制。Simplicial model 支持一致性和同伦解释；cubical type theory 支持计算单值性；strict Rezk completion、weak univalence 分离和 interval reversal 结果约束我们如何理解模型；directed/simplicial type theory 提供高阶范畴的新对象语言；two-level type theory 提供 strict 外部层；cohesive HoTT 和 QIIT 则分别扩展几何对象与归纳语法的表达能力。

## 练习

**练习 16.1.** 解释对象语言证明和模型论证明的区别。

**练习 16.2.** 查找 cubical type theory 文献中 Glue types 的定义，并说明它与 univalence 的关系。

**练习 16.3.** 说明 canonicity 对自然数计算为什么重要。

**练习 16.4.** 比较公理化 univalence 与 cubical univalence 的计算行为。
