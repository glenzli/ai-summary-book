# P0 引用定位批次 7：Moerdijk--Weiss dendroidal nerve

本文件记录第七批已精确定位的 P0 外部输入：Moerdijk--Weiss 的 dendroidal tree category、dendroidal nerve fully faithfulness、strict operad nerve 的 inner Kan/唯一填充性质，以及 homotopy coherent dendroidal nerve 的基本 inner Kan 来源。

本批次不覆盖 Cisinski--Moerdijk operadic model structure 的全部内容；该部分已由 [P0_REFERENCE_LOCATORS_BATCH_1.md](P0_REFERENCE_LOCATORS_BATCH_1.md) 中 CM-1--CM-4 覆盖，并且 erratum 影响属于 production-level 核查。本批次也不覆盖 Heuts--Hinich--Moerdijk 的 dendroidal-Lurie model comparison；该比较后续已由 [P0_REFERENCE_LOCATORS_BATCH_10.md](P0_REFERENCE_LOCATORS_BATCH_10.md) 中 HHM-1--HHM-5 定位。

## 1. Moerdijk--Weiss：tree category and dendroidal nerve

**主来源.** Ieke Moerdijk and Ittay Weiss, “Dendroidal Sets,” arXiv:math/0701293v2.

**本书对应位置.** 第十六、十七章，附录 M/T，附录 D.4、D.12.4，REFERENCE_LOCATOR_LEDGER 中 P0 “Dendroidal nerve fully faithful” 和 P1 “Dendroidal-Lurie comparison”的 strict dendroidal 前置部分。

### 1.1 线性树嵌入 $\Delta\subset\Omega$

**定位 MW-1.** Moerdijk--Weiss, arXiv:math/0701293v2, Section 3, paragraph after the construction of the linear trees.

**本书使用.** 第十六章外部输入定理 16.27 中 $i:\Delta\to\Omega$ fully faithful，以及 $i^\*:\mathbf{dSet}\to\mathbf{sSet}$ 的线性限制。

**需要同时记录的假设.**

1. $\Delta$ 嵌入 $\Omega$ 时把 $[n]$ 送到带 $n$ 个顶点的线性树。
2. 来源还说明 $\Delta$ 是 $\Omega$ 中的 sieve/ideal：若 $S\to T$ 且 $T$ 线性，则 $S$ 也线性。
3. 本书使用时必须保持根方向和 face/degeneracy convention 与第十六章一致。

### 1.2 Dendroidal nerve fully faithful

**定位 MW-2.** Moerdijk--Weiss, arXiv:math/0701293v2, Example 4.2.

**本书使用.** 第十六章外部输入定理 16.19：dendroidal nerve
$$
N_d:\operatorname{Operad}\longrightarrow \mathbf{dSet}
$$
是 fully faithful。

**需要同时记录的假设.**

1. 来源中的 operad 是 colored symmetric operad。
2. 来源定义
$$
N_d(P)_T=\operatorname{Hom}_{\operatorname{Operad}}(\Omega(T),P).
$$
3. 本书的 small colored operad 口径与来源的 universe convention 需要按附录 A 固定。

### 1.3 Simplicial sets 到 dendroidal sets 的 extension by zero

**定位 MW-3.** Moerdijk--Weiss, arXiv:math/0701293v2, Section 4, paragraph following Example 4.1.

**本书使用.** 第十六章和附录 M 中 $i_!:\mathbf{sSet}\to\mathbf{dSet}$ full faithful，以及 simplicial nerve 是 dendroidal nerve 的线性特例。

**需要同时记录的假设.** $i_!$ 是沿 $\Delta\subset\Omega$ 的 left Kan extension；因为 $\Delta$ 是 sieve，它表现为 extension by zero。

### 1.4 Strict operad nerve 的 inner Kan 与唯一填充

**定位 MW-4.** Moerdijk--Weiss, arXiv:math/0701293v2, Example 7.1.

**本书使用.** 第十七章定理 17.5 和附录 M 定义 M.6 中“strict operad 的 dendroidal nerve 是 inner Kan，且 inner horn fillers 唯一”的外部来源。

**需要同时记录的假设.**

1. 结论针对 ordinary strict operads 的 dendroidal nerve。
2. 唯一填充是 strict composition 的反映；一般 dendroidal inner Kan object 只要求填充存在。
3. 来源还说明唯一扩张性质刻画 nerves of operads；本书只把它作为 strict nerve 边界使用。

### 1.5 Homotopy coherent dendroidal nerve

**定位 MW-5.** Moerdijk--Weiss, arXiv:math/0701293v2, Proposition 7.2.

**本书使用.** 第十七、十八章和附录 M 中 homotopy coherent dendroidal nerve 的 inner Kan 入口：若 $\mathcal E$ 是带 interval 的 monoidal model category 且 $P\in\operatorname{Operad}(\mathcal E)$ aritywise fibrant，则 $hcN_d(P)$ 满足 inner Kan condition。

**需要同时记录的假设.**

1. 底范畴需要 monoidal model category 和 chosen interval。
2. Operad $P$ 需 aritywise fibrant。
3. 该命题不是 dendroidal-Lurie comparison theorem；它只给 homotopy coherent dendroidal nerve 的 fibrancy 条件。

### 1.6 Internal Hom 保持 inner Kan 的条件

**定位 MW-6.** Moerdijk--Weiss, arXiv:math/0701293v2, Theorem 7.5.

**本书使用.** 附录 M/T 中 internal Hom 与 weak maps 的 inner Kan 条件：若 $K$ 是 inner Kan dendroidal set，$X$ 是 normal dendroidal set，则
$$
\operatorname{Hom}_{\mathbf{dSet}}(X,K)
$$
也是 inner Kan。

**需要同时记录的假设.**

1. $X$ 必须 normal。
2. $K$ 必须满足 inner Kan condition。
3. 该结果与后续 operadic model structure 有关，但不等于 Cisinski--Moerdijk model structure 的存在定理。

## 2. 与本书现有文件的替换规则

| 旧表述 | 替换为 |
| --- | --- |
| $\Delta\hookrightarrow\Omega$ fully faithful | MW-1 |
| Dendroidal nerve of operads is fully faithful | MW-2 |
| $i_!:\mathbf{sSet}\to\mathbf{dSet}$ extension by zero is full faithful | MW-3 |
| Strict operad nerve has unique inner horn fillers | MW-4 |
| Homotopy coherent dendroidal nerve is inner Kan under fibrancy hypotheses | MW-5 |
| Internal Hom preserves inner Kan under normal source hypothesis | MW-6 |

## 3. 本批次未解决

本批次不解决：

1. Cisinski--Moerdijk operadic model structure 的 erratum 影响核查；
2. Heuts--Hinich--Moerdijk dendroidal-Lurie model comparison，见 HHM-1--HHM-5，并保留其 open/no-constants 限制；
3. Moerdijk--Weiss/Cisinski--Moerdijk 树范畴 generalized Reedy 分解的完整 bibliography/page 核查；
4. Operadic weak equivalence 的现代等价刻画；
5. Lurie-style infinity-operad、category-of-operators nerve 与 dendroidal nerve 的比较，见 HA-OP-1--HA-OP-3 与 HHM-1--HHM-5；前一入口允许 constants，并不意味着后一 open zig-zag 也允许 constants。

这些不由 MW-1--MW-6 单独推出；后续使用时必须引用对应 model-structure 或 model-comparison locator。
