# 第十六章：Cubical Type Theory、计算单值性与模型

## 本章目标

本章讨论 HoTT 的元理论背景：simplicial model、cubical type theory、计算单值性、canonicity 和模型比较。这里的内容属于外部输入，不作为前面内部证明的替代。对象语言、元语言和实现语言的边界见附录 Z。

## 依赖前置知识

本章依赖单值性、HIT 和形式化库比较。读者应区分对象语言中的类型论证明与元语言中的模型论证明。

## 16.1 Simplicial model

**定理 16.1（单值基础的 simplicial model，外部输入）.** Kapulkin、Lumsdaine、Voevodsky 构造的 simplicial set 模型证明了单值性与相应类型论规则的一致性背景。

**使用边界。** 模型论证明说明规则不导致矛盾，并解释同伦语义。它不在对象语言中给出任意具体定理的项。

## 16.2 Cubical type theory

**事实 16.2.** Cubical type theory 引入区间对象、面格、路径类型和 Glue 类型等结构，给出单值性的构造性解释。

**定理 16.3（计算单值性，外部输入）.** Cohen、Coquand、Huber、Mörtberg 的 cubical type theory 给出了 univalence 的计算性解释，使单值性不只是外部公理。

**验证状态。** 见附录 Z.3。公理化 HoTT 中这仍是模型/元理论输入；Cubical Agda 中则体现为实现语言提供的 cubical primitives、`ua` 和 transport 计算。

**警告 16.4.** Cubical type theory 不是“HoTT Book 加上一个实现细节”。它改变了底层类型论的规则和计算行为。把 Cubical Agda 证明翻译到公理化 HoTT 需要逐项检查。

## 16.3 Canonicity 与 normalization

**定义 16.5.** Canonicity 粗略说：若闭项 $n:\mathbb N$ 可类型检查，则它计算到某个标准自然数。

**事实 16.6.** 对不同 cubical 系统，canonicity、normalization 和 decidability of type checking 是独立的元理论课题。

**验证状态：研究边界。** 见附录 Z.4。本书只记录概念和来源方向，不证明这些元定理，也不把 canonicity 当作对象语言中的消去原则。

## 16.4 模型比较

**事实 16.7.** Cubical 模型有多种变体，包括 de Morgan cubical sets、cartesian cubical sets 和其他 presheaf/cubical categories。不同模型支持的连接、退化、填充和 Glue 结构不同。Orton-Pitts 对 univalence 的分解和 cubical set 语义提供了重要参考。

**来源边界。** Coherence 和 strictification 可参考 Lumsdaine-Warren 的 local universes model；universe hierarchy 和 Grothendieck topos 语义可参考 Gratzer-Shulman-Sterling 的 strict universes 工作。具体 cubical 实现仍以 Cubical Agda 文档和 cubical type theory 论文为准。

## 16.5 对本书的影响

**原则 16.8.** 本书正文分两种阅读方式：

1.  公理化 HoTT 阅读：函数外延性、单值性、HIT 和截断作为规则或公理加入；
2.  Cubical 阅读：单值性和部分 HIT 具有计算性实现，但需要接受 cubical primitives。

后续若把某证明称为“计算性”，必须说明采用第二种阅读方式。

**边界原则。** 附录 Z.7 给出本书各章与 cubical/HIT 元理论的接口。后续新增 HIT 或 cubical 计算规则时，必须同步记录形成、构造、消去、计算规则和形式化入口。

## 本章小结

HoTT 的严谨性来自内部类型论与外部模型论的双重控制。Simplicial model 支持一致性和同伦解释；cubical type theory 支持计算单值性；形式化库把这些思想落实为可检查代码。

## 练习

**练习 16.1.** 解释对象语言证明和模型论证明的区别。

**练习 16.2.** 查找 Agda Cubical 文档中 Glue types 的位置，并说明它与 univalence 的关系。

**练习 16.3.** 说明 canonicity 对自然数计算为什么重要。

**练习 16.4.** 比较公理化 univalence 与 cubical univalence 的计算行为。
