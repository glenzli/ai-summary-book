# 附录 D：定理定位索引

## 本附录目标

本附录记录外部输入定理的定位状态。当前版本以 source-level locator 为主；最终出版前必须升级为 section/theorem/page locator。

## D.1 Locator 等级

- `L0`：只有主题来源，不可用于最终出版。
- `L1`：有论文/书名、版本和 arXiv/出版信息，可用于草稿。
- `L2`：有章节或 section，可用于正式校对。
- `L2S`：有 arXiv 版本、TeX 文件、label 和源码行附近位置，可用于源码级复查。
- `L3`：有 theorem/proposition/definition/page，可用于最终出版。

## D.2 核心 locator

| ID | 本书用途 | 来源 | 当前等级 | 后续动作 |
| --- | --- | --- | --- | --- |
| BS-PRISM-DEF | prism, bounded prism, prismatic site | Bhatt-Scholze, arXiv:1905.08229 v4, 2022-01-12；见 [P0_REFERENCE_LOCATORS_BATCH_1.md](P0_REFERENCE_LOCATORS_BATCH_1.md) | L2S | 转换为期刊/正式 PDF 定义号和页码 |
| BS-COMP | Hodge-Tate, de Rham, crystalline, etale, base change comparison | Bhatt-Scholze, arXiv:1905.08229 v4；labels `thm:A`, `CrysComp`, `HTCompPrismatic`, `dRComp1`, `EtaleCompThm`, `BaseChangePrismCoh`, `generaldeRham`, `ImageofPhi` | L2S | 转换为期刊/正式 PDF 定理号和页码 |
| BS-PERF | perfect prisms = perfectoid rings | Bhatt-Scholze, arXiv:1905.08229 v4；label `PerfdPrism` | L2S | 转换为期刊/正式 PDF 定理号和页码 |
| BS-NYG | relative Nygaard theorem and Frobenius factorization | Bhatt-Scholze, arXiv:1905.08229 v4；label `thmCagain` | L2S | 转换为期刊/正式 PDF 定理号和页码 |
| BMS1-AINF | $A_{\inf}$-cohomology and integral comparison | BMS, arXiv:1602.03148 v3 final, 2019-01-15；见 [P0_REFERENCE_LOCATORS_BATCH_1.md](P0_REFERENCE_LOCATORS_BATCH_1.md) | L2S | 转换为期刊/正式 PDF 定理号和页码 |
| BMS1-LETA | $L\eta$, Bockstein and completion compatibility | BMS, arXiv:1602.03148 v3 final；labels `cor:LetaExists`, `prop:Letalaxsymmmon`, `prop:LetaBock`, `lem:Letapreservecompleteness`, `lem:Letacommutecompletion` | L2S | 转换为期刊/正式 PDF 定理号和页码 |
| BMS2-BK | Breuil-Kisin cohomology via THH | BMS2, arXiv:1802.03261 v2 final, 2019-04-09；labels `thm:main1`, `thm:BKlocal`, `cor:BKtwisted`；见 [P0_REFERENCE_LOCATORS_BATCH_2.md](P0_REFERENCE_LOCATORS_BATCH_2.md) | L2S | 转换为期刊/正式 PDF 定理号和页码 |
| BMS2-SYN | syntomic sheaves, Tate twists and nearby cycles | BMS2, arXiv:1802.03261 v2 final；labels `eq:TateTwist`, `thm:main6`, `thm:nearbycycles` | L2S | 转换为期刊/正式 PDF 定理号和页码 |
| BS-FCRYS | prismatic $F$-crystals = crystalline lattices | Bhatt-Scholze, arXiv:2106.14735 v2, 2023-09-12；labels `PrismaticFCrysDef`, `MainThm` | L2S | 转换为期刊/正式 PDF 定义、定理号和页码 |
| BL-WCART | prismatization and Cartier-Witt stack | Bhatt-Lurie, arXiv:2201.06124 v1, preliminary；labels `RelativePrismatize`, `CrystalsCartWitt`, `PrismaticCohPrismatization`, `QSynAbsCrys`；见 [P0_REFERENCE_LOCATORS_BATCH_2.md](P0_REFERENCE_LOCATORS_BATCH_2.md) | L2S | preliminary 版本，需等待/核查正式版本 |
| FON-PER | Fontaine period rings and admissibility | Fontaine sources; Brinon-Conrad notes | L0 | 选定最终教材源 |
| FAL-TSUJI | de Rham/crystalline/semistable comparison | Faltings, Tsuji, Brinon-Conrad | L0 | 拆分 classical theorem locator |

## D.3 前沿 locator

| ID | 本书用途 | 来源 | 当前等级 | 处理 |
| --- | --- | --- | --- | --- |
| TSUJI-COEFF | prismatic cohomology with coefficients | Tsuji, arXiv:2509.04954, 2025-09-05 | L1 | 研究边界 |
| QY-HT | rational Hodge-Tate prismatic crystals | Qu-Yu, arXiv:2511.03458 v3, 2026-01-13 | L1 | 研究边界 |
| CF-SYNOP | syntomic Steenrod algebra | Carmeli-Feng, arXiv:2507.13471, 2025-07-17 | L1 | 研究边界 |
| ANP-BRAUER | wild Brauer classes | Ambrosi-Newton-Pagano, arXiv:2509.22025, 2025-09-26 | L1 | 应用边界 |
| MO-FGAUGE | height 1 group schemes and prismatic $F$-gauges | Mondal-Olsson, arXiv:2604.16066 v1, 2026-04-17 | L1 | 研究边界 |
| KP-STACK | $p$-adic Hodge theory for Artin stacks | Kubrak-Prikhodko, arXiv:2105.05319, 2021-05-11 | L1 | 应用边界 |
| IKY-SHIMURA | prismatic realization for Shimura varieties | Imai-Kato-Youcis, arXiv:2310.08472, 2023-10-12 | L1 | 应用边界 |

## D.4 使用规则

1. L1 locator 可支撑草稿中的“外部输入定理”标记，但不能支撑最终出版的精确引用。
2. L2S locator 可支撑源码级数学复查，但 TeX 行号不是出版稳定编号。
3. 研究边界条目不得用于证明基础章节命题。
4. 若一个外部输入在多个章节使用，应只在本索引登记一次，并在章节中引用 ID。
5. 最终版必须把所有 P0 条目升级到 L3。

## 本附录小结

当前 locator 已经足以进行核心 prismatic 定理链、BMS2/THH-BK 链和 Bhatt-Lurie prismatization 接口的源码级复查，但不足以达到最终出版标准。下一轮工作应优先把 BS-COMP、BMS1-AINF、BMS2-BK、BMS2-SYN、BS-FCRYS 的 L2S 转换为 L3，并补齐 FON-PER、FAL-TSUJI。
