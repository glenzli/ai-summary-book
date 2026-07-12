# P0/P1 外部输入 locator：第二批源码级核查

初次核查日期：2026-07-08。PDF 定理号复核：2026-07-11。

本文件延续 [P0_REFERENCE_LOCATORS_BATCH_1.md](P0_REFERENCE_LOCATORS_BATCH_1.md)，补齐 BMS2/THH-Breuil-Kisin 方向和 Bhatt-Lurie prismatization 方向的 source-label locator。它不新增数学定理。2026-07-11 已复核的 BMS2 numbered statements 标为 `L3`；其余 rows 仍须在出版前转换为稳定编号或保持真实的 research-interface 等级。

## P0-4：Bhatt-Morrow-Scholze, Topological Hochschild homology and integral p-adic Hodge theory

来源版本：Bhatt-Morrow-Scholze, arXiv:1802.03261, v2 final, 2019-04-09。
核查介质：上述 arXiv 版本的 PDF 与 TeX source snapshot；下表记录的
source labels/line neighborhoods 不依赖临时解压路径。

| 本书 ID | 用途 | label | 源码位置 | 当前等级 |
| --- | --- | --- | --- | --- |
| BMS2-BK | Breuil-Kisin cohomology 主定理 | `thm:main1`; Theorem 1.2 | line 147 附近及 PDF | L3 |
| BMS2-TC | perfectoid rings 上 $\pi_0TC^-$ 与 $A_{\inf}$ | `thm:TCperfectoid` | line 225 附近 | L2S |
| BMS2-AOMEGA | $A\Omega$ via $TC^-$ descent | `thm:main2`, `main_theorem` | lines 246, 2598 附近 | L2S |
| BMS2-CRYS | equal characteristic crystalline comparison | `thm:main3`, `thm:tccharp` | lines 258, 2290 附近 | L2S |
| BMS2-FILT | THH/TC/TP motivic filtration and Nygaard package | `thm:main5`, `MotivicFiltTHH`, `NygaardGradedFilt`, `NygaardSmooth` | lines 284, 1720, 1741, 1763 附近 | L2S |
| BMS2-SYN | syntomic fibre、乘法及 nearby cycles | `eq:TateTwist`, `thm:main6`, `thm:nearbycycles`; Theorem 1.12 (5), Theorem 1.15, Theorem 10.1 | lines 1929, 352, 2785 附近及 PDF | L3 |
| BMS2-BKTWIST | Breuil-Kisin twist compatibility | `subsec:BKtwist`, `prop:breuilkisintwist` | lines 1462, 1495 附近 | L2S |
| BMS2-BKLOCAL | local Breuil-Kisin construction from relative THH | `thm:BKlocal`; Theorem 11.2（其余配套结果仍见源码 labels） | line 2938 附近及 PDF | L3 |

### 约定核查结果

BMS2 在 quasisyntomic site 上把 syntomic sheaf 写成 THH/TC filtration 的 graded piece；其模 $p$ 入口为
$$
\mathbf Z/p\mathbf Z(i)(A)
=
\operatorname{hofib}\left(
\varphi_i-1:
\mathcal N^{\ge i}\widehat{\Prism}_A\{i\}/p
\to
\widehat{\Prism}_A\{i\}/p
\right),
$$
见 `eq:TateTwist`。这说明源码中 $\varphi_i-1$ 的简写是可追溯的；本书附录 F 在类型敏感处写作 $\varphi_i-\operatorname{can}_i$，把这个 “$1$” 解释为进入同一 Tate-twisted target 的规范映射。最终章节公式仍必须决定是否使用 $p$-complete、mod $p^r$、truncate 或 nearby cycle 版本。

`thm:main6` 给出两个重要出口：在 characteristic $p$ smooth 情形与 logarithmic de Rham-Witt sheaves 对应；在 mixed characteristic smooth formal $\mathcal O_C$ 情形与截断 nearby cycles 对应。第十一章和第十四章若使用 syntomic-etale comparison，应引用 `BMS2-SYN`，而不是只引用 BMS1。

`thm:BKlocal` 是本书第五章和第十二章使用 Breuil-Kisin cohomology 时的关键源码入口：它给出相对于 $\mathfrak S$ 的 $(p,z)$-complete $E_\infty$-algebra、Frobenius、$A\Omega$ comparison、de Rham comparison 和 crystalline comparison。

## P1-1：Bhatt-Lurie, The prismatization of p-adic formal schemes

来源版本：Bhatt-Lurie, arXiv:2201.06124, v1 preliminary, 2022-01-16。
核查介质：上述 arXiv 版本的 TeX source snapshot；临时 source 路径不作为
可复现 locator，且该 preliminary 文献继续保持 `L2S`。

| 本书 ID | 用途 | label | 源码位置 | 当前等级 |
| --- | --- | --- | --- | --- |
| BL-ANIM | animated prisms | `DefAnimPrism` | line 319 附近 | L2S |
| BL-WCART | perfectoid prismatization example | `PerfectoidPrismatize` | line 602 附近 | L2S |
| BL-WCART | relative prismatization definition | `RelativePrismatize` | line 692 附近 | L2S |
| BL-HT | relative Hodge-Tate gerbe | `RelativeHT`, `HTDerivedPrismatize` | lines 813, 1139 附近 | L2S |
| BL-WCART-COH | classical relative cohomology comparison | `RelativePrismaticCohPrismatizeNonDer` | line 960 附近 | L2S |
| BL-WCART-CRYS | crystals via Cartier-Witt stack | `CrystalsCartWitt` | line 978 附近 | L2S |
| BL-WCART-DER | derived prismatic cohomology via prismatization | `PrismaticCohPrismatization`, `QSynWCartClassical` | lines 1354, 1392 附近 | L2S |
| BL-WCART-ABS | absolute derived Cartier-Witt stack | `CartWittAnimDef`, `DerCartWittAbs`, `QSynAbsCrys` | lines 1418, 1451, 1507 附近 | L2S |
| BL-PUSH | pushforward compatibility | `AbsPushWCart`, `AbsPushWCartZ` | lines 1517, 1530 附近 | L2S |

### 约定核查结果

Bhatt-Lurie 的结果在本书中只作为 prismatization 和 Cartier-Witt stack 的前沿接口使用。它提供两个可进入正文的受控用途：

1. 相对 prismatization $\WCart_{X/A}$ 与 relative prismatic site 的比较：在 $p$-completely lci 或更一般 $p$-quasisyntomic 条件下，$\RGamma(\WCart_{X/A},\mathcal O)$ 与 prismatic cohomology 比较。
2. quasi-coherent sheaves on $\WCart$ 与 prismatic crystals 的解释：在相应假设下，$\mathcal D_{qc}(\WCart_{X/A})$ 或 $\mathcal D_{qc}(\WCart_X)$ 与 prismatic crystals 的完备导出范畴等价。

该文献 arXiv 页面标注为 preliminary version。因此本书不得把 BL-WCART 条目提升为基础定义链；它只可作为第八章、第十三章和第十四章的研究边界或解释性接口，直到完成正式版本核查。

## 本批次结论

本批次及 2026-07-11 PDF 复核完成了以下升级：

- BMS2-BK：Breuil-Kisin cohomology 主定理升级到 `L3`。
- BMS2-SYN：syntomic fibre、乘法和 nearby cycles 出口升级到 `L3`。
- BMS2-BKLOCAL：局部 Breuil-Kisin 构造主定理升级到 `L3`；其配套命题仍为 `L2S`。
- BL-WCART：Cartier-Witt/prismatization、crystals-as-QCoh 和 pushforward compatibility。

仍未完成：

- Classical Fontaine/Faltings/Tsuji comparison 的最终 source selection。
- BMS2 中 TC/TP filtration、$A\Omega$ descent 等未进入本轮正文主链的条目，以及 Bhatt-Lurie 条目的出版级 `L3` 转换。
- 前沿 2025-2026 预印本的二次 locator。
