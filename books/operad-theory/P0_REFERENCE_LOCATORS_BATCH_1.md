# P0 引用定位批次 1：模型范畴 operad 与 dendroidal model structure

本文件记录已经精确到 theorem/proposition/corollary 编号的第一批 P0 外部输入。它只覆盖两个高影响包：

1. Berger--Moerdijk 的模型范畴中 operad/admissibility/rectification；
2. Cisinski--Moerdijk 的 dendroidal operadic model structure。

未列入本批次的 P0 包由后续 locator 批次、[REFERENCE_LOCATOR_LEDGER.md](REFERENCE_LOCATOR_LEDGER.md) 或 [FINAL_OPERAD_THEORY_CLOSURE.md](FINAL_OPERAD_THEORY_CLOSURE.md) 分类处理。

## 1. Berger--Moerdijk：模型范畴中的 operad

**主来源.** Clemens Berger and Ieke Moerdijk, “Axiomatic homotopy theory for operads,” arXiv:math/0206094v3.

**本书对应位置.** 第十四章、附录 G、附录 R、附录 D.4。

### 1.1 Operads 的 transferred model structure

**定位 BM-1.** arXiv:math/0206094v3, Theorem 3.1.

**本书使用.** 第十四章中“operads in a symmetric monoidal model category 有 transferred model structure”的外部输入。

**需要同时记录的假设.**

- 底范畴为 cofibrantly generated symmetric monoidal model category；
- 需要 interval 或 path-object 型假设；
- weak equivalences 和 fibrations 在底层 collections 中检测；
- 小性和 cofibrant generation 条件不能省略。

**允许用法.** 只允许推出 $\operatorname{Op}(\mathcal M)$ 上的模型结构存在性和逐 arity 检测性质；不得推出固定 operad 的 algebra category admissibility。

### 1.2 固定 operad 的 algebra category

**定位 BM-2.** arXiv:math/0206094v3, Theorem 3.2.

**本书使用.** 第十四章和附录 G/R 中“$\mathcal P$-algebras 的 transferred model structure”的外部输入。

**需要同时记录的假设.**

- $\mathcal P$ 是 non-symmetric operad，或 symmetric operad 且满足文中规定的 cofibrancy/ambient hypotheses；
- free-forgetful adjunction 创建 fibrations 和 weak equivalences；
- pushout 和 transfinite composition 的同伦控制来自原文假设。

**允许用法.** 只在满足 BM 假设的底范畴和 operad 类型中使用。Colored operads 或现代 all-small-colored admissibility 使用 P0 引用定位批次 9 中 PSAR/PSP 条目。

### 1.3 $\Sigma$-cofibrancy 与代数范畴 rectification

**定位 BM-3.** arXiv:math/0206094v3, Proposition 4.1.

**本书使用.** 附录 G/R 中“cofibrant operad 或 $\Sigma$-cofibrant operad 带来更好 homotopical behavior”的来源之一。

**需要同时记录的假设.**

- operad 的 cofibrancy 类型必须按原文区分；
- 不得把 cofibrant as operad、entrywise cofibrant 和 $\Sigma$-cofibrant 混为一谈。

### 1.4 Weak equivalence of operads induces equivalence on homotopy categories

**定位 BM-4.** arXiv:math/0206094v3, Theorem 4.4.

**本书使用.** 第十四章 rectification criterion 的早期版本来源。

**需要同时记录的假设.**

- $\varphi:\mathcal P\to\mathcal Q$ 是 operads 的 weak equivalence；
- $\mathcal P,\mathcal Q$ 与底范畴满足原文 cofibrancy 和 model-category hypotheses；
- 原文结论是 homotopy categories 或 Quillen adjunction 层面的比较，使用时必须检查本书是否需要 Quillen equivalence 版本。

**允许用法.** 可作为 rectification 的早期来源；若正文需要现代 colored/symmetric flatness 版本，改用 P0 引用定位批次 9 中 PSAR/PSP 与 HA-ALG 条目。

### 1.5 Cofibrant replacement of operads and algebras

**定位 BM-5.** arXiv:math/0206094v3, Corollary 4.5.

**本书使用.** 第十四章中 $W\mathcal O$ 或 cofibrant replacement 与 algebra homotopy theory 的比较边界。

**注意.** Boardman--Vogt $W$-construction 的具体 functorial resolution 还需要 Boardman--Vogt 原书或 Berger--Moerdijk 后续 resolution 文献；BM-5 不替代完整 $W$-construction 构造。

## 2. Cisinski--Moerdijk：dendroidal operadic model structure

**主来源.** Denis-Charles Cisinski and Ieke Moerdijk, “Dendroidal sets as models for homotopy operads,” arXiv:0902.1954v2.

**本书对应位置.** 第十六、十七章，附录 M/T，附录 D.5。

### 2.1 Normal monomorphisms

**定位 CM-1.** arXiv:0902.1954v2, Proposition 1.4.

**本书使用.** 第十七章和附录 T 中 normal monomorphism 的基本稳定性来源。

**允许用法.** 用于 normal monomorphism 的生成和闭包性质；不得从该命题单独推出 operadic model structure。

### 2.2 Inner anodyne extensions

**定位 CM-2.** arXiv:0902.1954v2, Proposition 1.5 and Corollaries 1.6--1.8.

**本书使用.** 第十七章 inner anodynes 对 inner Kan dendroidal sets 的 lifting 性质和 horn calculus。

**需要同时记录的假设.**

- inner horn inclusions 的 weakly saturated closure；
- normality 与 lifting 性质需分开引用；
- strict operad nerve 的唯一 fillers 仍依赖 strict operad nerve 的内部证明或 Moerdijk--Weiss nerve 结果。

### 2.3 Operadic model structure

**定位 CM-3.** arXiv:0902.1954v2, Theorem 2.4.

**本书使用.** 第十七章“$\mathbf{dSet}$ 上存在 operadic model structure”的主要 P0 来源。

**原文结论在本书中的转写.**

- cofibrations 为 normal monomorphisms；
- fibrant objects 为 infinity-operads / inner Kan dendroidal sets；
- weak equivalences 由映入 fibrant objects 后的 homotopy category 条件刻画。

**允许用法.** 可以支撑第十七章 operadic model structure 的存在性。不得把它替换为 Lurie-style infinity-operad 定义；dendroidal-Lurie comparison 后续已由 [P0_REFERENCE_LOCATORS_BATCH_10.md](P0_REFERENCE_LOCATORS_BATCH_10.md) 中 HHM-1--HHM-5 定位。

### 2.4 Weak equivalences between fibrant objects

**定位 CM-4.** arXiv:0902.1954v2, Proposition 2.6.

**本书使用.** 第十七章中 fibrant dendroidal sets 之间弱等价的判别来源。

**注意.** 该论文有 erratum。出版或最终版引用时必须核查 erratum 对相关命题的影响，尤其不能不加检查地使用原文关于 monoidal model structure 的附加说法。

## 3. 对本书现有文件的替换规则

本批次完成后，下列旧表述应被理解为更精确的 locator：

| 旧表述 | 替换为 |
| --- | --- |
| Berger--Moerdijk transferred model structure | BM-1 |
| 固定 operad algebra transferred structure | BM-2 |
| 早期 rectification criterion | BM-4 + BM-5；现代版本使用批次 9 PSAR/PSP 与 HA-ALG |
| normal monomorphism facts | CM-1 |
| inner anodynes / horn calculus | CM-2 |
| Cisinski--Moerdijk operadic model structure | CM-3 |
| fibrant dendroidal weak equivalence criterion | CM-4，带 erratum 检查 |

## 4. 本批次未解决

本批次不解决：

1. Koszul 判别和 bar-cobar resolution 的 theorem locator；
2. homological perturbation 和 homotopy transfer 的 theorem locator；
3. Lurie straightening/unstraightening、monoidal localization 和 factorization homology 的 theorem locator；
4. colored operad admissibility 的 Pavlov--Scholbach 精确 locator；
5. dendroidal-Lurie comparison 的 Heuts--Hinich--Moerdijk 精确 locator；
6. Fukaya category 构造和 descent 的几何定理 locator。

这些仍按 [REFERENCE_LOCATOR_LEDGER.md](REFERENCE_LOCATOR_LEDGER.md) 的 P0/P1 列表推进。
