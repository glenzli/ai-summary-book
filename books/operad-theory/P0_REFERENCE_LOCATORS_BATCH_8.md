# P0 引用定位批次 8：Monoidal Bousfield localization 与 operad algebras

本文件记录第八批已精确定位的 P0 外部输入：模型范畴意义下的 monoidal Bousfield localization 保持 operad algebra structures。它服务于第十九章、附录 D、附录 M 和附录 R 中“localization 与 operad 代数结构的相容性”这一证明边界。

本批次只覆盖模型范畴中的 Bousfield localization preservation 问题；它不覆盖 Lurie-style infinity-operad 中
$$
\operatorname{Alg}_{\mathcal O}(\mathcal M)_\infty
\simeq
\operatorname{Alg}_{\mathcal O^{loc}}(\mathcal M_\infty)
$$
的完整 infinity-categorical algebra localization comparison，也不覆盖 operadic straightening、dendroidal-Lurie comparison 或 Pavlov--Scholbach 的全部 symmetric flatness/rectification 体系。

## 1. David White: 单色 operad 版本

**来源.** David White, "Monoidal Bousfield Localizations and Algebras over Operads," arXiv:1404.5197。

**WHT-1.** Definition 3.1 defines what it means for a localization functor $L_C$ to preserve $P$-algebras. 该定义要求：对每个 $P$-algebra $E$，存在 $P$-algebra $\widetilde E$，其底层对象 weak equivalent to $L_CE$，且若 $E$ 本身已经 cofibrant，则可把 localization map $E\to L_CE$ 提升到 $P$-algebra map $E\to\widetilde E$。

**WHT-2.** Theorem 3.2 gives a criterion for preservation of $P$-algebras by $L_C$: 若每个 $P$-algebra 的 localization 可在 $P$-algebra category 中由底层 $C$-local object 实现，则 $L_C$ preserves $P$-algebras in the sense of WHT-1。

**WHT-3.** Corollary 3.4 states the main monoid-axiom criterion: 若 $C$ 是 monoidal model category $\mathcal M$ 中一组 maps，且 left Bousfield localization $L_C(\mathcal M)$ exists and satisfies the monoid axiom，则 $L_C$ preserves algebras over any operad $P$ in $\mathcal M$。当 $P$ cofibrant 时，所需 monoid axiom 条件可只在 $P$-algebras 上使用。

**WHT-4.** Section 4, especially the main criteria around Theorems 4.5--4.6, studies when a left Bousfield localization of a monoidal model category is itself monoidal. 本书可用它支撑“monoidal localization 的模型范畴版本需要 pushout-product/unit/monoid-axiom 类假设”这一边界说明；不得把它直接替代 Lurie *Higher Algebra* 的 symmetric monoidal infinity-categorical localization theorem。

## 2. White--Yau: colored operad 版本

**来源.** David White and Donald Yau, "Bousfield localization and algebras over colored operads," arXiv:1503.06720。

**WY-1.** Definition 7.2.1 defines preservation of algebras over a colored operad $\mathcal O$ by a Bousfield localization. 它是 WHT-1 的 colored analogue，要求 localization 后的底层 colored objects 可由 $\mathcal O$-algebra object 表示，并在 cofibrant 情形中提升 localization map。

**WY-2.** Theorem 7.2.3 gives the colored-operad preservation criterion: 若每个 $\mathcal O$-algebra $X$ 有 $\mathcal O$-algebra $\widetilde X$ 与 map $X\to\widetilde X$，且各颜色分量给出 $L_C$-local replacement，则 $L_C$ preserves $\mathcal O$-algebras。

**WY-3.** Theorems 7.4.1--7.4.3 give classes of colored operads and localizations for which preservation holds under additional hypotheses, including entry points through colored operad admissibility and localization compatibility. 本书只把这些定理作为 colored algebra preservation 的 theorem locator，不把它们扩大为任意 enriched/infinity-operadic comparison。

## 3. 本书使用边界

**可用于以下位置.**

1. 第十九章定理 19.21 附近：说明 symmetric monoidal Bousfield localization 的模型范畴版本需要 monoidal compatibility 假设。
2. 第十九章定理 19.25 附近：说明“localization preserves $P$-algebra structures”有模型范畴版本，且可通过 WHT-1--WHT-4、WY-1--WY-3 引用。
3. 附录 D.12.6、附录 M.15--M.18 和附录 R 中：把 colored operad algebra preservation 与 infinity-categorical algebra localization comparison 分开。

**禁止用法.**

1. 不得由 WHT-1--WHT-4 或 WY-1--WY-3 直接推出
   $$
   \operatorname{Alg}_{\mathcal O}(\mathcal M)_\infty
   \simeq
   \operatorname{Alg}_{\mathcal O^{loc}}(\mathcal M_\infty).
   $$
   该比较不由本批次推出；应另引 [P0_REFERENCE_LOCATORS_BATCH_9.md](P0_REFERENCE_LOCATORS_BATCH_9.md) 中 PSAR-5--PSAR-6、HA-ALG-1--HA-ALG-3，或 [P0_REFERENCE_LOCATORS_BATCH_10.md](P0_REFERENCE_LOCATORS_BATCH_10.md) 中 DKR-7 等相应 comparison locator。
2. 不得把单色 WHT-1--WHT-4 用作 colored operad 结论；colored 版本必须引用 WY-1--WY-3 或其他 colored source。
3. 不得把“$L_C$ preserves $\mathcal O$-algebras”解释为“任意 operad weak equivalence 在 localization 后给出 Quillen equivalence”；后者属于 rectification/admissibility theorem locator。

## 4. 假设转换清单

把本批次用于正文时，必须逐项检查：

1. 底范畴是否是 White 或 White--Yau 假设中的 monoidal model category；
2. left Bousfield localization $L_C(\mathcal M)$ 是否存在；
3. localized model structure 是否满足 monoid axiom 或 colored analogue；
4. 使用的是单色 operad 还是 colored operad；
5. operad cofibrancy/admissibility 是否足以支持 lifted algebra structure；
6. 结论只声明 preservation of algebra structures，还是更强的 infinity-categorical comparison。

若第 6 项需要更强结论，本批次只能作为前置参考，不能单独闭合证明链。
