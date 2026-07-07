# P0/P1 外部输入 locator：第二批源码级核查

核查日期：2026-07-08。

本文件延续 [P0_REFERENCE_LOCATORS_BATCH_1.md](P0_REFERENCE_LOCATORS_BATCH_1.md)，补齐 BMS2/THH-Breuil-Kisin 方向和 Bhatt-Lurie prismatization 方向的 source-label locator。它不新增数学定理；所有条目仍需在最终出版前转换为正式页码、定理号和期刊版本引用。

## P0-4：Bhatt-Morrow-Scholze, Topological Hochschild homology and integral p-adic Hodge theory

来源版本：Bhatt-Morrow-Scholze, arXiv:1802.03261, v2 final, 2019-04-09。
本地核查源：`/private/tmp/prismatic_locator/1802/bms2.tex`。

| 本书 ID | 用途 | label | 源码位置 | 当前等级 |
| --- | --- | --- | --- | --- |
| BMS2-BK | Breuil-Kisin cohomology 主定理 | `thm:main1` | line 147 附近 | L2S |
| BMS2-TC | perfectoid rings 上 $\pi_0TC^-$ 与 $A_{\inf}$ | `thm:TCperfectoid` | line 225 附近 | L2S |
| BMS2-AOMEGA | $A\Omega$ via $TC^-$ descent | `thm:main2`, `main_theorem` | lines 246, 2598 附近 | L2S |
| BMS2-CRYS | equal characteristic crystalline comparison | `thm:main3`, `thm:tccharp` | lines 258, 2290 附近 | L2S |
| BMS2-FILT | THH/TC/TP motivic filtration and Nygaard package | `thm:main5`, `MotivicFiltTHH`, `NygaardGradedFilt`, `NygaardSmooth` | lines 284, 1720, 1741, 1763 附近 | L2S |
| BMS2-SYN | syntomic sheaves and nearby cycles | `subsection_syntomic`, `cor:CocontinuousTateTwist`, `eq:TateTwist`, `thm:main6`, `thm:nearbycycles` | lines 1862, 1916, 1929, 352, 2785 附近 | L2S |
| BMS2-BKTWIST | Breuil-Kisin twist compatibility | `subsec:BKtwist`, `prop:breuilkisintwist` | lines 1462, 1495 附近 | L2S |
| BMS2-BKLOCAL | local Breuil-Kisin construction from relative THH | `thm:BKlocal`, `cor:BKtwisted`, `rmk:NygaardBK`, `prop:BKnc` | lines 2938, 3121, 3174, 3186 附近 | L2S |

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
见 `eq:TateTwist`。这说明本书附录 F 的 syntomic convention form 使用 $\varphi_i-1$ 是可追溯的；但最终章节公式仍必须决定是否使用 $p$-complete、mod $p^r$、truncate 或 nearby cycle 版本。

`thm:main6` 给出两个重要出口：在 characteristic $p$ smooth 情形与 logarithmic de Rham-Witt sheaves 对应；在 mixed characteristic smooth formal $\mathcal O_C$ 情形与截断 nearby cycles 对应。第十一章和第十四章若使用 syntomic-etale comparison，应引用 `BMS2-SYN`，而不是只引用 BMS1。

`thm:BKlocal` 是本书第五章和第十二章使用 Breuil-Kisin cohomology 时的关键源码入口：它给出相对于 $\mathfrak S$ 的 $(p,z)$-complete $E_\infty$-algebra、Frobenius、$A\Omega$ comparison、de Rham comparison 和 crystalline comparison。

## P1-1：Bhatt-Lurie, The prismatization of p-adic formal schemes

来源版本：Bhatt-Lurie, arXiv:2201.06124, v1 preliminary, 2022-01-16。
本地核查源：`/private/tmp/prismatic_locator/2201/prismatization-for-arxiv.tex`。

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

本批次把以下缺口从 `L1` 升级为 `L2S`：

- BMS2-BK：Breuil-Kisin cohomology via relative THH。
- BMS2-SYN：syntomic sheaves、Tate twist convention 和 nearby cycles 出口。
- BL-WCART：Cartier-Witt/prismatization、crystals-as-QCoh 和 pushforward compatibility。

仍未完成：

- Classical Fontaine/Faltings/Tsuji comparison 的最终 source selection。
- BMS2 与 Bhatt-Lurie 的出版级 `L3` 页码/定理号转换。
- 前沿 2025-2026 预印本的二次 locator。
