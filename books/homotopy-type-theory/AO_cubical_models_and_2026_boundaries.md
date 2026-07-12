# 附录 AO：Cubical 模型、弱单值性与 2026 边界

本附录更新第十六章的模型论边界。重点是：某些看似接近 univalence 的原则并不等价；cubical interval 的操作选择有可证明的保守性结果；严格 Rezk 完备化与 homotopy canonicity 形成新的元理论路线。

## AO.1 Categorical univalence 不推出函数外延性

**定义 AO.1（wild category of a universe）.** 给定 universe $\mathcal U$，其 wild category 以类型为对象，以函数为态射，但不要求 Hom 为集合，也不截断高阶相干。

**定义 AO.2（categorical univalence）.** 对 $A,B:\mathcal U$，记 $A\cong_{\mathrm w}B$ 为 wild category 中的同构：其底层是函数 $f:A\to B$，并有左右逆函数，逆律是函数类型中的路径。$\mathcal U$ 称为 categorically univalent，若 canonical map
$$
(A=_{\mathcal U}B)\longrightarrow(A\cong_{\mathrm w}B)
$$
对所有 $A,B$ 都是等价。这弱于把右端换成 homotopy equivalence $A\simeq B$ 的普通 universe univalence。

**外部输入定理 AO.3（categorical univalence 不推出函数外延性）.** 存在 Martin-Löf type theory 模型，其 universe categorically univalent，但函数外延性失败。

**精确来源、采用与未采用假设.** Cavallo--Höfer, *Univalence without function extensionality*, arXiv:2605.00812v1（2026-05-01）。Definitions 1.1--1.4 区分 $\mathsf{FE}_{\mathcal U}$、$\mathsf{UA}_{\mathcal U}$、categorical 与 familial categorical univalence；Theorem 1.5（正文 Theorem 4.17）在基模型具有 extensive finite coproducts、相关 strict $\eta$ 规则及 familial categorical univalence 时证明 polynomial model 保持后者；Proposition 5.3 证明 polynomial model 否定 $\mathsf{FE}_{\mathcal U}$；Theorem 1.6（正文 Theorem 5.6）给出最终不蕴含结论。本书只采用 Theorem 5.6 的模型分离结论。Extensive finite coproducts、strict $\eta$、familial categorical univalence 和 polynomial construction 都只是构造反模型的语义假设，不加入本书基础对象语言，也不把 categorical univalence 当作本书公理。

**教材后果 AO.4.** 本书第六章使用的是普通 universe univalence
$$
(A=B)\simeq(A\simeq B),
$$
并由附录 T 记录其推出函数外延性的数学链条。不得把 AO.2 的弱原则误当成第六章单值性。

## AO.2 Strict Rezk completion 与 homotopy canonicity

**输入 AO.5（strict Rezk completion of a model）.** 在 Bocquet 的 cartesian cubical 内部语义中，设全局 HoTT 模型 $\mathcal M$ 的各组成是 fibrant。其 strict Rezk completion 是一个 complete model $\widehat{\mathcal M}$ 和模型态射 $i:\mathcal M\to\widehat{\mathcal M}$：外部化后的 $i$ 是 split weak equivalence，并且对每个 $x:\mathsf{Tm}(\Gamma,A)$，
$$
\sum_{y:\mathsf{Tm}(\Gamma,A)}
\mathsf{Tm}(\Gamma,\mathsf{Id}_A(x,y))
$$
作为 fibrant set 可收缩。该完备性给出 identity terms 与 ambient cubical paths 的等价，而不是对象语言中新添的 equality reflection。

**外部输入定理 AO.6（homotopy canonicity 的 Rezk 路线）.** 在该文固定的 HoTT 语法中，初始模型 $\mathcal S$ 的任意闭布尔项 $b:\mathbf 2$ 满足
$$
\mathsf{Id}_{\mathbf 2}(b,\mathsf{true})
+
\mathsf{Id}_{\mathbf 2}(b,\mathsf{false}).
$$
证明在 cartesian cubical sets 的 topos 内构造 $\mathcal S$ 的 strict Rezk completion，再作 gluing/sconing。

**精确来源、采用与未采用假设.** Bocquet, *Strict Rezk completions of models of HoTT and homotopy canonicity*, arXiv:2311.05849v2（2025-10-08）。Definitions 5.1--5.2 给出 complete model 与 strict Rezk completion；Theorem 5.18 假设模型是 global、algebraically cofibrant 且 components fibrant；Remark 5.19 把初始语法置于该范围；Theorem 6.1 与 §6.2 给出上述 Boolean homotopy canonicity。本文只采用 Theorems 5.18、6.1 的元理论路线，不采用来源 §4 的 cumulative universe hierarchy、函数外延性、W-types 等作为本书基础规则，也不把 Theorem 5.18 扩大为“任意 HoTT 模型都有 strict Rezk completion”。它泛化一范畴的 Rezk completion，但不等同于本书第十四章的对象语言构造。

## AO.3 Cubical interval 的 reversal

**定义 AO.7（interval reversal）.** 在 cubical type theory 中，reversal 是区间上的 involution
$$
r:\mathbb I\to\mathbb I
$$
满足 $r(0)=1$、$r(1)=0$。

**外部输入定理 AO.8（reversal 的保守性，opaque 情形）.** 对每个 self-dual interval theory $(\Phi,\varphi)$，从相应 opaque cubical type theory 到加入 internalizes duality 的 reversal 后理论的包含，在忘却到带 $\Sigma$ 与 identity types 的 MLTT 模型后诱导 weak equivalence。例子包括只有两个端点的 cartesian interval theory 和有界分配格 interval theory。

**精确来源、采用与未采用假设.** Cavallo--Sattler, *Eliminating reversals from cubical type theories*, arXiv:2605.15080v1（2026-05-14）。§3.3 的 opaque theory 省去 filling 在具体 type former 上的严格归约，并把 HIT eliminator 在 path constructor 上的计算弱化为路径；Definition 23 定义 self-dual interval theory，Definitions 25、27 与 Theorem 42 给出 reversal extension 和 twist interpretation；Theorem 65 是上述 weak-equivalence/conservativity 定理。本书采用 Theorem 65，且保留 opaque 与 self-dual 两项假设；不采用“所有 strict cubical theories 上也保守”。§7 的 Theorem 71 是另一条模型存在性结论：在特定 ABCHFL setup 中得到带 reversal 的 strict cubical model，其同伦理论在 classical logic 下呈现 $\infty$-groupoids；该模型没有 connection，论文也明确未证明带 connections 的对应结论。本书不把 Theorem 71 当作 Theorem 65 的 strict 版本。

**教材后果 AO.9.** 第十六章讨论 de Morgan/cartesian cubical 差异时，不能只说“是否有 reversal 是实现细节”。它影响语法操作和模型构造，但在特定 opaque 条件下有保守性定理。

## AO.4 非标准 HoTT 模型

**输入 AO.10（filter quotient construction）.** 设 $\mathcal M$ 是模型范畴，$\mathcal F$ 是其 subterminal objects 的 model filter：$\mathcal F$ 中对象 fibrant，且 cofibrations 与 weak equivalences 对同 $\mathcal F$ 中对象作乘积稳定。要保持 simplicial 结构、typal initial algebras 和严格单值 universe 塔，还要求 $\mathcal F$ 是 simplicial model filter。只有在这些条件下，本附录使用 filter quotient $\mathcal M/\mathcal F$。

**事实 AO.11（非标准模型）.** 在 AO.10 的假设下，filter quotient 保持相应 HoTT 构造所需的模型范畴性质；但具体 filter product 可以没有无限极限与余极限，并且不是 locally presentable 或 cofibrantly generated，同时仍解释该来源列出的 HoTT 构造与单值 universes。

**精确来源、采用与未采用假设.** Rasekh, *Non-Standard Models of Homotopy Type Theory*, arXiv:2508.07736v2（2025-08-12）。Definition 2.7 给出 model/simplicial model filter；Theorems 2.13--2.14 分别列出保持的模型范畴性质与由此保留的类型构造；Example 2.18 对 simplicial sets 的 Kan model structure 和 $\mathbb N$ 上非主 filter 给出仍建模 HoTT、却失去 infinite (co)limits、local presentability 与 cofibrant generation 的实例；Corollaries 2.21--2.22 记录无限 (co)limits 的独立性和非标准自然数对象。本书只采用带 Definition 2.7 假设的 Theorems 2.13--2.14 及 Example 2.18 的边界，不声称任意 filter、任意模型范畴或任意 HoTT 语法都被保持，也不把这些外部范畴性质写成对象语言定理。

**教材后果 AO.12.** 模型论章节应把“模型满足 HoTT 规则”与“模型具有熟悉的外部范畴性质”分开。非标准模型说明后者不能从前者自动推出。

## AO.5 与第十六章的接口

1.  Simplicial model 给一致性；cubical model 给计算性解释；strict Rezk completion 给 homotopy canonicity 的新证明路线。
2.  Weak/categorical univalence 是单独原则，不能替代 universe univalence。
3.  Interval theory 的操作集合是模型和实现选择，需记录是否有 reversal、connections、composition、Glue 和 HIT path constructor 计算。
4.  非标准模型提醒本书：任何“所有 HoTT 模型都……”的断言都必须写明所需外部范畴假设。
