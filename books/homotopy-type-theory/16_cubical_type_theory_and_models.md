# 第十六章：Cubical Type Theory、计算单值性与模型

## 本章目标

本章讨论 HoTT 的元理论背景：simplicial model、cubical type theory、计算单值性、canonicity、directed/simplicial type theory、two-level type theory、cohesive HoTT 和模型比较。这里的内容属于外部输入，不作为前面内部证明的替代。对象语言、元语言和实现语言的边界见附录 Z。

## 依赖前置知识

本章依赖单值性、HIT 和模型比较。读者应区分对象语言中的类型论证明与元语言中的模型论证明。

## 16.1 Simplicial model

**外部输入定理 16.1（一个单值宇宙的 simplicial set 模型）.** Kapulkin 与 Lumsdaine 给出 Voevodsky 构造的严格化表述：在经典集合论元理论中，以 Kan simplicial sets 构造 contextual category 模型，解释相应的 Martin-Löf 类型论，并使一个 universe 满足单值性。其相对一致性推论使用带两个不可达基数的 ZFC 作为元理论假设。

**来源与未重证边界。** 精确来源为 Kapulkin--Lumsdaine, *The Simplicial Model of Univalent Foundations (after Voevodsky)*, arXiv:1211.2851。论文构造模型并验证 univalence；本书不重证 weakly universal Kan fibration、coherence 或模型解释的可靠性。结论是条件性的相对一致性：若所用集合论元理论一致，则被解释的对象理论一致。它不把模型中的对象变成对象理论的项，也不推出 normalization、canonicity、HIT 规则或 judgmental computation。

## 16.2 Cubical type theory

**定义 16.2（本节所指的 CCHM 对象语言）.** Cohen--Coquand--Huber--Mörtberg 的 cubical type theory 不是只向第一章添加一个常量。其语法包含区间 $\mathbb I$ 及端点、面格 $\mathbb F$、以区间项表示的 $\mathsf{Path}$、composition/filling 操作、Glue 类型和 Russell-style universe。$\mathsf{Path}_A(a,b)$ 的项是带端点约束的区间方向项；它不是第二章归纳生成的 $\mathsf{Id}_A(a,b)$ 的同一个语法构造子。

**外部输入定理 16.3（CCHM 单值性）.** 在上述特定 calculus 中，函数外延性可内部证明；Glue 构造给出从等价到 universe path 的映射，并且规范映射

$$
\mathsf{pathToEq}_{A,B}:
\mathsf{Path}_{\mathcal U}(A,B)\to(A\simeq B)
$$

是等价。该 calculus 还在构造性元理论中具有 cubical set 语义。

**来源与未重证边界。** 精确来源为 Cohen--Coquand--Huber--Mörtberg, *Cubical Type Theory: A Constructive Interpretation of the Univalence Axiom*, LIPIcs TYPES 2015, DOI `10.4230/LIPIcs.TYPES.2015.5`：第 4 节给 composition 与 transport，第 6 节给 Glue，第 7.2 节的 Corollary 11 给 univalence，第 8 节给语义。本书不重证其语法保存、composition 算法或模型可靠性。

**对象语言边界。** 第六章的 $\mathsf{ua}_i$ 是公理 6.8 所选逆，命题 6.9.1 只得到路径等式。CCHM 的 Glue 与归约规则说明另一对象语言中的计算内容；除非先给出两种语法的解释或保守性定理，不能据此把第六章的 $\beta_e$、$\eta_p$ 或 transport 等式升级为 judgmental equality。

**警告 16.4.** Cubical type theory 不是“HoTT Book 加上一个实现细节”。它改变了底层类型论的规则和计算行为。把 cubical 口径下的证明翻译到公理化 HoTT 需要逐项检查所用规则。

## 16.3 Canonicity 与 normalization

**定义 16.5（自然数 canonicity）.** 对一个带指定 judgmental equality $\equiv$ 的类型论，closed canonicity 断言：若

$$
\cdot\vdash n:\mathbb N,
$$

则存在元语言自然数 $k$，使

$$
\cdot\vdash n\equiv\overline{k}:\mathbb N.
$$

这里“存在 $k$”是元语言量词，结论使用 judgmental equality。较弱的 homotopy canonicity 只要求对象语言中存在路径 $n=_{\mathbb N}\overline{k}$；两者不可互换。

**外部输入定理 16.6（CCHM canonicity 的精确范围）.** Huber 对 CCHM cubical type theory 证明：若语境只含区间 name variables，且 $I\vdash n:\mathbb N$，则存在唯一 $k$ 使 $I\vdash n\equiv\overline{k}:\mathbb N$，并可有效计算该 $k$。

**来源与边界。** 精确来源为 Simon Huber, *Canonicity for Cubical Type Theory*, Journal of Automated Reasoning 63 (2019), DOI `10.1007/s10817-018-9469-1`。该定理不是第二章的 J，也不自动覆盖任意 cubical 变体、resizing 或任意 HIT 扩展。

**外部输入定理 16.6.1（Cartesian cubical normalization）.** Sterling 与 Angiuli 对 univalent Cartesian cubical type theory 证明 normalization，并推出 judgmental equality 的可判定性和类型构造子的单射性。

**来源与边界。** 精确来源为 *Normalization for Cubical Type Theory*, arXiv:2101.11479。它针对 Cartesian cubical syntax；不能仅凭同属“cubical”就迁移到 CCHM de Morgan calculus 或第 9 章的公理化 HIT 包。每个新增构造都需要重新验证 normalization/canonicity 保存性。附录 Z.4 记录这些元定理与对象语言的隔离原则。

## 16.4 模型比较

**事实 16.7.** Cubical 模型有多种变体，包括 de Morgan cubical sets、cartesian cubical sets 和其他 presheaf/cubical categories。不同模型支持的连接、退化、填充和 Glue 结构不同。Orton-Pitts 对 univalence 的分解和 cubical set 语义提供了重要参考。

**来源边界。** Coherence 和 strictification 可参考 Lumsdaine-Warren 的 local universes model；universe hierarchy 和 Grothendieck topos 语义可参考 Gratzer-Shulman-Sterling 的 strict universes 工作；cubical 计算规则应回到 cubical type theory 的原始论文与模型论文献。

**事实 16.8（weak/categorical univalence 的分离）.** 某些弱形式的单值性，例如 universe wild category 的 categorical univalence，并不推出函数外延性。Cavallo-Höfer 2026 给出模型分离结果；见附录 AO.1。

**教材后果。** 第六章使用的仍是 universe univalence，不得用 categorical univalence 替代；附录 T 中单值性推出函数外延性的链条只适用于相应强口径，且本书非累积宇宙下只采用基底与 fibers 同属该单值 universe 的实例。

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

**外部输入定理 16.12.3（CHM 的列举型 HIT 计算语义）.** 第 9 章定理 9.11 所引的 CHM 系统为 spheres、torus、suspensions、truncations 和 pushouts 给出语法与构造性语义；在该系统中，这些列出的签名对所有构造子有 judgmental computation，严格保持替换，并保持参数的 universe level。

**边界。** CHM 2018 没有给出无条件的一般 HIT schema。因而不能把“有某个 HIT 构造子”自动升级为“其所有路径计算 judgmental”，也不能推断 arbitrary HIIT/QIIT 的 canonicity。每个签名仍须分别核查 strict positivity、composition/filling、universe closure 和元理论。

**事实 16.12.3.1（逻辑公理与计算性）。** LEM、choice、resizing 等原则不由 cubical univalence 的计算解释自动提供。若加入这些原则，应按附录 BL 标注 classical mode，并重新解释 canonicity 或数据计算。

## 16.7 Cohesive 与几何对象语言

**事实 16.12.4（cohesive HoTT）。** Cohesive HoTT 加入 shape、discrete、global sections、codiscrete 等模态和 exactness 条件，用于表达几何、拓扑和微分对象。附录 AT 给出模态接口，附录 BD 给出 SDG、de Rham 和 Zariski 使用边界。

**边界 16.12.5.** Cohesive HoTT、synthetic differential geometry 和 synthetic algebraic geometry 不是附录 BA 构造性分析的推论。它们需要额外对象语言和模型。

## 16.8 Two-level type theory

**事实 16.12.6（strict equality layer）。** Two-level type theory 同时具有外部 strict equality 层和内部 fibrant HoTT 层，可用于表达 semisimplicial types、Reedy fibrant diagrams 和元理论构造。接口见附录 BG。

**边界 16.12.7.** Strict equality 不是 HoTT identity path。用 strict equality 关闭 semisimplicial 相干不能自动给出内部路径等式；所有比较都需要明确桥接原则。

## 16.9 对本书的影响

**原则 16.13.** 本书正文区分五种阅读方式：

1.  公理化 HoTT 阅读：函数外延性、单值性、HIT 和截断作为规则或公理加入；
2.  Cubical 阅读：单值性和列举出的部分 HIT 具有计算性实现，但需要接受指定 calculus 的 cubical primitives 与归约规则。
3.  Directed/simplicial 阅读：额外引入 directed hom、Segal 条件或 simplicial primitives；规则核见附录 AS，且这些规则不回流到前文 identity type 证明。
4.  Cohesive/geometry 阅读：额外引入 shape、cohesive modalities、infinitesimal objects 或 Zariski gluing；接口见附录 AT 和 BD。
5.  Two-level 阅读：额外区分 strict 外部层和 fibrant HoTT 层；接口见附录 BG。

后续若把某证明称为“计算性”，必须说明采用第二种阅读方式中的哪一个具体 cubical calculus；只写“cubical”不足以确定 judgmental equality。

**边界原则。** 附录 Z.7 给出本书各章与 cubical/HIT 元理论的接口。后续新增 HIT 或 cubical 计算规则时，必须同步记录形成、构造、消去、计算规则和元理论假设。

## 本章小结

HoTT 的严谨性要求同时控制内部推导与外部元理论。Simplicial model 给出带明确集合论假设的相对一致性解释；CCHM 通过不同的 $\mathsf{Path}$、composition 和 Glue 语法证明单值性；canonicity 与 normalization 只对其精确 calculus 成立；CHM 的强 HIT 计算只覆盖列出的签名。其余 directed、two-level、cohesive 和 QIIT 接口同样不得回流为基础 HoTT 的隐式规则。

## 练习

**练习 16.1.** 解释对象语言证明和模型论证明的区别。

**练习 16.2.** 查找 cubical type theory 文献中 Glue types 的定义，并说明它与 univalence 的关系。

**练习 16.3.** 说明 canonicity 对自然数计算为什么重要。

**练习 16.4.** 比较公理化 univalence 与 cubical univalence 的计算行为。
