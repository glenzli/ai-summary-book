# 附录 AO：Cubical 模型、弱单值性与 2026 边界

本附录更新第十六章的模型论边界。重点是：某些看似接近 univalence 的原则并不等价；cubical interval 的操作选择有可证明的保守性结果；严格 Rezk 完备化与 homotopy canonicity 形成新的元理论路线。

## AO.1 Categorical univalence 不推出函数外延性

**定义 AO.1（wild category of a universe）.** 给定 universe $\mathcal U$，其 wild category 以类型为对象，以函数为态射，但不要求 Hom 为集合，也不截断高阶相干。

**定义 AO.2（categorical univalence）.** $\mathcal U$ 的 wild category 称为 categorically univalent，若对象路径到同构/等价对象的 canonical map 满足对应的单值性条件。

**定理 AO.3（categorical univalence 不推出函数外延性）.** 存在 Martin-Löf type theory 模型，其 universe categorically univalent，但函数外延性失败。

**来源与证明状态.** Cavallo-Höfer 2026 通过分析 von Glehn 的 polynomial model construction 得到该分离结果：若基模型有 univalent universe，其 polynomial model 可有 categorically univalent universe 而无 function extensionality。

**教材后果 AO.4.** 本书第六章使用的是普通 universe univalence
$$
(A=B)\simeq(A\simeq B),
$$
并由附录 T 记录其推出函数外延性的数学链条。不得把 AO.2 的弱原则误当成第六章单值性。

## AO.2 Strict Rezk completion 与 homotopy canonicity

**输入 AO.5（strict Rezk completion of a model）.** 对 HoTT 模型 $\mathcal M$，strict Rezk completion 给出等价模型 $\widehat{\mathcal M}$，并使其满足一种饱和/完备条件：identity type 的项与 cubical path 之间有等价。

**定理 AO.6（homotopy canonicity 的 Rezk 路线）.** 通过在 cartesian cubical sets 的 topos 中构造语法模型的 strict Rezk completion，可给出 HoTT homotopy canonicity 的构造性证明。

**来源与边界.** Bocquet 2023/2025 给出该路线。该结果是元理论定理，不是对象语言中的归纳原则。它泛化一范畴的 Rezk completion，但不等同于本书第十四章的范畴 Rezk 完备化构造。

## AO.3 Cubical interval 的 reversal

**定义 AO.7（interval reversal）.** 在 cubical type theory 中，reversal 是区间上的 involution
$$
r:\mathbb I\to\mathbb I
$$
满足 $r(0)=1$、$r(1)=0$。

**定理 AO.8（reversal 的保守性，opaque 情形）.** 对 self-dual interval theories，例如只有两个端点的最小理论或有界分配格理论，加入 internalizes duality 的 reversal 是保守扩张，至少在 opaque cubical type theories 中成立。

**证明状态.** Cavallo-Sattler 2026 使用 twist construction：区间与其对偶的乘积仍为带 reversal 的区间。该保守性不覆盖所有带严格计算规则的 cubical 实现；论文还构造了带 reversal 的 strict cubical type theory 模型。

**教材后果 AO.9.** 第十六章讨论 de Morgan/cartesian cubical 差异时，不能只说“是否有 reversal 是实现细节”。它影响语法操作和模型构造，但在特定 opaque 条件下有保守性定理。

## AO.4 非标准 HoTT 模型

**输入 AO.10（filter quotient construction）.** 对满足 HoTT 构造所需性质的模型范畴，可通过 filter quotient 构造新的模型。

**事实 AO.11（非标准模型）.** Rasekh 2025 提出 filter quotient 方法，并证明在适当假设下保持实现类型论构造与公理的模型范畴性质，但可能不保持局部 presentability、cocompleteness 或 cofibrant generation 等外部集合论性质。

**教材后果 AO.12.** 模型论章节应把“模型满足 HoTT 规则”与“模型具有熟悉的外部范畴性质”分开。非标准模型说明后者不能从前者自动推出。

## AO.5 与第十六章的接口

1.  Simplicial model 给一致性；cubical model 给计算性解释；strict Rezk completion 给 homotopy canonicity 的新证明路线。
2.  Weak/categorical univalence 是单独原则，不能替代 universe univalence。
3.  Interval theory 的操作集合是模型和实现选择，需记录是否有 reversal、connections、composition、Glue 和 HIT path constructor 计算。
4.  非标准模型提醒本书：任何“所有 HoTT 模型都……”的断言都必须写明所需外部范畴假设。
