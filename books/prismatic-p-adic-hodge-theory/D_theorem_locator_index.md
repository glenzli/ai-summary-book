# 附录 D：定理定位索引

## 本附录目标

本附录记录外部输入定理的定位状态。核心主线中已经完成 PDF 定理号复核的条目标为 `L3`；仍只有源码 label 的技术输入继续标为 `L2S`。

## D.1 Locator 等级

- `L0`：只有主题来源，不可用于最终出版。
- `L1`：有论文/书名、版本和 arXiv/出版信息，可用于草稿。
- `L2`：有章节或 section，可用于正式校对。
- `L2S`：有 arXiv 版本、TeX 文件、label 和源码行附近位置，可用于源码级复查。
- `L3`：有 theorem/proposition/definition/page，可用于最终出版。

## D.2 核心 locator

| ID | 本书用途 | 来源 | 当前等级 | 后续动作 |
| --- | --- | --- | --- | --- |
| BS-DCOMP | Koszul derived completion、complete flatness、complete Tor-amplitude | Bhatt-Scholze, arXiv:1905.08229 v4, §1.2, pp. 10--11 | L3 | 已核查；正文另证明所需的初等特例 |
| BS-PRISM-DEF | prism、bounded prism、prism map ideal rigidity | Bhatt-Scholze, Definition 3.2, Proposition 3.5, Lemma 3.7 | L3 | 已核查 |
| BS-PRISM-SITE | relative prismatic site、topology、structure sheaves | Bhatt-Scholze, Corollary 3.12, Definition 4.1 | L3 | 已核查 |
| BS-COMP-HT | Hodge--Tate comparison | Bhatt-Scholze, Theorems 4.11, 6.3 | L3 | 已核查 twists 与 cohomological grading |
| BS-COMP-CRYS | crystalline comparison | Bhatt-Scholze, Theorem 5.2 | L3 | 已核查 derived $p$-completion |
| BS-COMP-DR | general de Rham comparison | Bhatt-Scholze, Corollary 15.4 | L3 | 已核查 Frobenius-twisted、completed tensor product |
| BS-COMP-ETALE | finite-level etale comparison：任意 $X$ 为 sheaf-level，global formula 仅作 affine corollary | Bhatt-Scholze, Theorem 9.1 | L3 | 已核查 mod $p^n$、$I^{-1}$ 与 sheaf/global 层级 |
| BS-COMP-BC | bounded-prism base change | Bhatt-Scholze, Corollary 4.12 | L3 | 已核查 $(p,IB)$-completed base change |
| BS-COMP-PHI | Frobenius image/isogeny | Bhatt-Scholze, Corollary 15.5 | L3 | 已核查只在 $I$ 反演后同构 |
| BS-COMP-AINF | $A_{\inf}$-cohomology 与 prismatic cohomology | Bhatt-Scholze, Theorem 17.2 and global descent | L3 | 已核查 Frobenius pullback |
| BS-COMP-BK | Breuil--Kisin/prismatic comparison | Bhatt--Scholze, Example 1.9 (3), Proposition 15.7 and concluding paragraph of §15.2, p. 105 | L3 | 已核查 coefficient maps 与 specialization 边界 |
| BS-PERF | perfect prisms = perfectoid rings | Bhatt--Scholze, Theorem 3.10 | L3 | 已核查 |
| BS-NYG | relative Nygaard theorem、graded pieces、Frobenius factorization | Bhatt--Scholze, Theorem 1.16 / Theorem 15.3 | L3 | 已核查 $C^{(1)}$、$\tau^{\le i}$、$\{i\}$ 与 $L\eta_I$ |
| BMS1-AINF | $A_{\inf}$-cohomology and integral comparison | BMS, arXiv:1602.03148 v3 final, Theorems 1.8, 14.3 | L3 | 已核查四个 derived comparisons |
| BMS1-BKF | BKF definition、finite-free classification、cohomological torsion thresholds | BMS, Definition 1.5, Theorem 4.28, Theorem 14.5 (iii) | L3 | 已核查有限表示与有限自由边界 |
| BMS1-LETA | $L\eta$ existence、monoidal/Bockstein package、completion boundary | BMS, Corollary 6.5, Proposition 6.7, Proposition 6.12, Lemmas 6.19--6.20 | L3 | 已核查；different-ideal completion 不自动交换 |
| BMS2-BK | Breuil--Kisin cohomology via THH | BMS2, arXiv:1802.03261 v2 final, Theorem 1.2 | L3 | 已核查 perfect complex、localized Frobenius 与 comparisons |
| BMS2-SYN | syntomic fibre、products、nearby cycles | BMS2, Theorem 1.12 (5), Theorem 1.15, Theorem 10.1 | L3 | 已核查 truncation 与 characteristic hypotheses |
| BMS2-BKLOCAL | local Breuil--Kisin construction | BMS2, Theorem 11.2 | L3 | 主定理已核查；配套命题仍见第二批 `L2S` locators |
| BS-FCRYS | prismatic $F$-crystal definition and crystalline lattice equivalence | Bhatt-Scholze, arXiv:2106.14735 v2, Definition 4.1, Theorem 5.6 | L3 | 已核查 localized Frobenius 与 effective 子范畴 |
| BS-FCRYS-BK | Breuil--Kisin evaluation and Kisin full faithfulness | Bhatt-Scholze, Examples 2.6, 4.3, Theorem 7.9 | L3 | 已核查 evaluation/descent boundary |
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
5. 最终版必须把正文实际依赖的所有 P0 条目升级到 L3；未进入证明链的研究接口可保留其真实等级。

## 本附录小结

核心 prismatic comparison/Nygaard、BMS1 $A_{\inf}$/BKF/$L\eta$、BMS2 Breuil--Kisin/syntomic 以及 prismatic $F$-crystal 主链均已达到 PDF numbered-statement level。尚未闭合的出版风险集中在 Bhatt--Lurie preliminary 接口、未进入主线的配套结果，以及 classical Fontaine/Faltings/Tsuji source selection；这些条目不得被提升为已闭合的主线定理。
