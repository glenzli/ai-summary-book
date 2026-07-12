# 正式教材扩展审计

初次审计日期：2026-07-08。严格化复核：2026-07-11。

前半部分保留 2026-07-08 扩展动作的历史记录；文末复核段给出当前状态。

## 本轮扩展动作

- 更新 [README.md](README.md) 状态为正式教材扩展稿。
- 新增第九至第十五章：
  - Hodge-Tate/de Rham specialization；
  - crystalline/de Rham-Witt/$q$-de Rham specialization；
  - etale comparison/Frobenius fixed/syntomic tower；
  - Breuil-Kisin/BKF modules and lattices；
  - coefficients and non-abelian boundary；
  - Artin stacks/Shimura/Brauer/finite flat group applications；
  - closure/failure modes/open problems。
- 新增附录 C-F：
  - comparison hypotheses；
  - theorem locator；
  - label ledger；
  - Nygaard/Tate twist crosswalk。
- 新增 [PUBLICATION_CLOSURE_MATRIX.md](PUBLICATION_CLOSURE_MATRIX.md)。

## 严格性处理

- 所有大型 comparison theorem 继续标为外部输入。
- 所有 2025-2026 预印本结果继续标为研究边界或应用边界。
- 没有伪造 theorem/page locator；当前 locator 诚实标为 L1。
- Nygaard/syntomic 公式标为 convention form，并登记最终核对队列。

## 覆盖度判定

| 主题 | 状态 |
| --- | --- |
| $\delta$-rings and prisms | 已成章 |
| prismatic site and cohomology | 已成章 |
| Hodge-Tate/de Rham/crystalline/etale comparison | 已成体系 |
| BMS and Breuil-Kisin | 已成体系 |
| prismatic $F$-crystals | 已成章 |
| Nygaard/syntomic/Tate twists | 已成章，需 locator |
| prismatization/$F$-gauges | 研究边界 |
| coefficients/non-abelian | 研究边界 |
| applications | 应用边界 |
| closure/failure modes | 已成章 |

## 2026-07-08 时点的剩余风险

- Fontaine period rings 的 construction 仍未重建，只作为 classical interface。
- Derived complete flatness 当时尚未按 source definition 收口。
- Nygaard/Tate twist indexing 当时尚未完成 numbered-statement 核对。
- 章节内交叉引用尚未全部替换为稳定 label。

## 审计结论

本轮已把目录从“严格教材草稿”推进到“正式教材扩展稿”。它已经具备正式教材的章节、附录、账本和边界结构；下一轮应进入 locator 精确化和编号交叉引用重写。

## 后续内部完整性补强

补强日期：2026-07-08。

- 新增附录 G-K，覆盖 formal schemes/sites、$\delta$-prism 证明、crystals/descent、period linear algebra 和 worked examples。
- 新增 `SOLUTIONS.md`、`TERM_INDEX.md`、`INTERNAL_COMPLETENESS_AUDIT.md`。
- 闭包矩阵状态升级为 `Textbook-Internal`。

补强后，本书不再只是章节和账本成体系，而是具备内部技术基础和章末练习解答；剩余瓶颈转为外部 locator 与出版校对。

## 2026-07-11 严格化复核

- 附录 A 已按 Bhatt--Scholze §1.2 给出 Koszul derived completion、complete
  flatness 与 complete Tor-amplitude 的精确定义，并完成 principal
  bounded-torsion 与 flat reduction 的初等证明。
- 第二、三、五至七、九、十一、十二章已分别收口 prism/site、completed
  Frobenius twist、BMS comparisons、relative Nygaard、syntomic truncation
  与 BK/BKF lattice boundaries。
- 正文使用的 Bhatt--Scholze/BMS numbered statements 已升级为 `L3`；当前
  数学来源风险集中在 classical Fontaine/Faltings/Tsuji source selection
  与未进入主线的配套结果。
