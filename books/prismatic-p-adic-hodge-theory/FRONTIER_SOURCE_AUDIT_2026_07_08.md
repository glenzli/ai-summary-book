# 前沿资料核查记录：2026-07-08

本文档记录本书扩展稿写作前后完成的联网核查。它不新增数学定理。

## 核心资料

- Bhatt-Scholze, “Prisms and Prismatic Cohomology,” arXiv:1905.08229：arXiv 页面显示 v1 提交于 2019-05-20，v4 修订于 2022-01-12。用途：prism、prismatic site 和 comparison theorem 主源。
- Bhatt-Morrow-Scholze, “Integral $p$-adic Hodge theory,” arXiv:1602.03148：arXiv 页面显示 v1 提交于 2016-02-09，v3 final version 修订于 2019-01-15。用途：$A_{\inf}$-cohomology 与 integral comparison 主源。
- Bhatt-Morrow-Scholze, “Topological Hochschild homology and integral $p$-adic Hodge theory,” arXiv:1802.03261：arXiv 页面显示 v1 提交于 2018-02-09，v2 final version 修订于 2019-04-09。用途：THH/TC filtration、Breuil-Kisin cohomology、syntomic sheaves and nearby cycles 主源。
- Bhatt-Scholze, “Prismatic $F$-crystals and crystalline Galois representations,” arXiv:2106.14735：arXiv 页面显示 v1 提交于 2021-06-28，v2 修订于 2023-09-12。用途：prismatic $F$-crystals 与 crystalline lattices 的范畴等价。
- Bhatt-Lurie, “The prismatization of $p$-adic formal schemes,” arXiv:2201.06124：arXiv 页面显示 v1 提交于 2022-01-16，标注为 preliminary version。用途：Cartier-Witt stack 和 prismatization 前沿接口。

## 核心资料源码 locator 批次

2026-07-08 已完成两批核心源码级 locator。第一批详见 [P0_REFERENCE_LOCATORS_BATCH_1.md](P0_REFERENCE_LOCATORS_BATCH_1.md)，覆盖：

- Bhatt-Scholze v4 的 prism/prismatic site、perfect prism、主比较定理、Nygaard theorem。
- BMS1 v3 final 的 $A\Omega$、$L\eta$、Bockstein、completion compatibility、integral comparison 入口。
- Bhatt-Scholze prismatic $F$-crystals v2 的对象定义、effective 条件、main theorem 和 Breuil-Kisin evaluation。
第二批详见 [P0_REFERENCE_LOCATORS_BATCH_2.md](P0_REFERENCE_LOCATORS_BATCH_2.md)，覆盖 BMS2/THH-Breuil-Kisin、BMS2 syntomic/Tate twists 和 Bhatt-Lurie prismatization。

这两批把核心来源从书目级 `L1` 升级到源码 label 级 `L2S`，但仍未替代出版级 `L3` 页码/定理号 locator。

## 2025-2026 边界

- Carmeli-Feng, arXiv:2507.13471，2025-07-17：syntomic Steenrod algebra、spectral syntomic cohomology、spectral prismatic $F$-gauges。
- Tsuji, arXiv:2509.04954，2025-09-05：带系数 prismatic cohomology 与 $A_{\inf}$-cohomology。
- Qu-Yu, arXiv:2511.03458，2025-11-05：rational Hodge-Tate prismatic crystals 与 non-abelian $p$-adic Hodge theory。
- Ambrosi-Newton-Pagano, arXiv:2509.22025，2025-09-26：wild Brauer classes via prismatic cohomology。
- Mondal-Olsson, arXiv:2604.16066，2026-04-17：height 1 group schemes and prismatic $F$-gauges。
- Qu-Yu, arXiv:2511.03458：v1 2025-11-05，v3 2026-01-13；rational Hodge-Tate prismatic crystals and non-abelian $p$-adic Hodge theory。
- Kubrak-Prikhodko, arXiv:2105.05319，2021-05-11：Artin stacks 中的 integral $p$-adic Hodge theory；作为应用边界和 stacky extension 入口。

## 写作限制

上述 2025-2026 条目只可写入研究边界或应用边界章节。没有完成二次 locator 和独立 proof audit 前，不得把这些预印本结论提升为正文基础定理。
