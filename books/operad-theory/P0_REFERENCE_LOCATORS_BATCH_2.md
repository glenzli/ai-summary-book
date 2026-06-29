# P0 引用定位批次 2：Straightening/Unstraightening

本文件记录第二批已精确定位的 P0 外部输入：Lurie 的 straightening/unstraightening theorem。它只覆盖 ordinary coCartesian fibrations over an infinity-category 的 straightening/unstraightening，不覆盖 operadic straightening、monoidal localization 或 algebra localization comparison。

## 1. Lurie HTT straightening/unstraightening

**主来源.** Jacob Lurie, *Higher Topos Theory*, author PDF, version available from the author site.

**本书对应位置.** 第十九章、附录 M、附录 D.6、REFERENCE_LOCATOR_LEDGER 中 P0 “Straightening/unstraightening”。

### 1.1 主定理定位

**定位 HTT-1.** Lurie, *Higher Topos Theory*, Theorem 3.2.0.1.

**本书使用.** 第十九章中
$$
\operatorname{coCartFib}(S)\simeq \operatorname{Fun}(S,\operatorname{Cat}_\infty)
$$
型的外部输入，即 coCartesian fibrations over an infinity-category 与从 base 到 $\operatorname{Cat}_\infty$ 的 functors 之间的 straightening/unstraightening equivalence。

**需要同时记录的假设.**

1. Base $S$ 是 simplicial set / infinity-category 语境中的对象。
2. Fibrations 需带 coCartesian structure。
3. 等价发生在 Lurie 的 marked simplicial set / Cartesian model structure 语境中。
4. 若要把结论用于 ordinary relative categories 或 model categories，必须先经 coherent nerve、Dwyer--Kan localization 或 underlying infinity-category 过渡。

**允许用法.** 可用于第十九章 ordinary straightening/unstraightening 叙述。可支撑“families of infinity-categories over $S$ 由 coCartesian fibrations 建模”的外部输入。

**禁止用法.**

- 不得把 HTT-1 直接用作 operadic straightening；operadic straightening 使用 P0 引用定位批次 10 中 PRA-1--PRA-5 的 infinity-operadic 版本。
- 不得把 HTT-1 直接推出 monoidal localization 或 algebra localization comparison。
- 不得把 ordinary Grothendieck construction 的 1-categorical statement 与 HTT-1 混为同一定理。

## 2. 与本书第十九章的替换规则

| 旧表述 | 替换为 |
| --- | --- |
| Straightening/unstraightening 外部输入 | HTT-1 |
| coCartesian fibrations classify functors | HTT-1，需说明 Lurie model context |
| operadic straightening | 不由 HTT-1 单独推出；后续由 [P0_REFERENCE_LOCATORS_BATCH_10.md](P0_REFERENCE_LOCATORS_BATCH_10.md) 中 PRA-1--PRA-5 作为 spaces-valued/preprint locator 分层处理 |
| monoidal localization | 不由 HTT-1 单独推出；后续由 [P0_REFERENCE_LOCATORS_BATCH_8.md](P0_REFERENCE_LOCATORS_BATCH_8.md)、[P0_REFERENCE_LOCATORS_BATCH_9.md](P0_REFERENCE_LOCATORS_BATCH_9.md) 和 [P0_REFERENCE_LOCATORS_BATCH_10.md](P0_REFERENCE_LOCATORS_BATCH_10.md) 分别处理 preservation、algebra comparison 和 localization comparison |

## 3. 本批次未解决

本批次自身不解决下列项目；后续批次已经分层处理：

1. Dwyer--Kan localization 的 theorem locator，见 DKR-1--DKR-7；
2. simplicial model category coherent nerve comparison，见 DKR-3--DKR-6；
3. Quillen equivalence induces equivalence of underlying infinity-categories，见 DKR-7；
4. monoidal localization preservation，见 WHT-1--WHT-4；
5. algebra localization/comparison，见 WY-1--WY-3、PSAR-5--PSAR-6 和 HA-ALG-1--HA-ALG-3；
6. operadic straightening，见 PRA-1--PRA-5。

这些结论不得倒填为 HTT-1 的内容；引用时必须使用对应后续 locator。
