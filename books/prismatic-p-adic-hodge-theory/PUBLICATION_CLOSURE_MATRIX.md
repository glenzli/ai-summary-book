# 正式教材闭包矩阵

初次核查日期：2026-07-08。主线复核：2026-07-11。

## 状态等级

- `Draft-1`：第一批严格草稿。
- `Textbook-Expanded`：正式教材扩展稿，章节和账本成体系。
- `Textbook-Internal`：正式教材内部完整稿，定义链、技术基础、代表性解答和局部模型成体系。
- `Chapter-Closed-Draft`：逐章教材收口草稿，每章具备正文细节、例子/证明/结构表和练习。
- `Math-Closed`：数学定义链、外部输入链和边界链完成最终审查。
- `Camera-Ready`：完成出版校对、locator、编号、排版和参考格式。

## 当前判定

当前状态：`Chapter-Closed-Draft`。

理由：

- 正文从 8 章扩展到 15 章，覆盖基础、比较、积分理论、表示论、系数、应用边界和错误模式。
- 附录从 2 个扩展到 11 个，包含比较假设表、locator、编号账本、Nygaard/Tate twist 交叉表、formal schemes/sites、$\delta$-prism 详细证明、crystals/descent、period linear algebra 和 worked examples。
- 已新增并闭合 [SOLUTIONS.md](SOLUTIONS.md)、[TERM_INDEX.md](TERM_INDEX.md) 和 [INTERNAL_COMPLETENESS_AUDIT.md](INTERNAL_COMPLETENESS_AUDIT.md)。
- 已新增 [CHAPTER_CLOSURE_AUDIT.md](CHAPTER_CLOSURE_AUDIT.md)，并对第 0-15 章补充正文型细节。
- 已建立 `THEOREM_LEDGER.md`、`DEPENDENCY_GRAPH.md`、`D_theorem_locator_index.md` 和本闭包矩阵。
- [P0_REFERENCE_LOCATORS_BATCH_1.md](P0_REFERENCE_LOCATORS_BATCH_1.md) 已把 derived completion、prism/site、基础 comparisons、BMS1/BKF 与 prismatic $F$-crystal 主链升级到 PDF numbered-statement `L3`。
- [P0_REFERENCE_LOCATORS_BATCH_2.md](P0_REFERENCE_LOCATORS_BATCH_2.md) 已把 BMS2 Breuil--Kisin、syntomic/products/nearby cycles 与 local construction 主定理升级到 `L3`；Bhatt--Lurie preliminary 接口仍为 `L2S`。
- 2025-2026 前沿资料仍被限制在研究边界，不进入基础定理链。

## 闭包矩阵

| 维度 | 当前状态 | 阻塞项 | 下一步 |
| --- | --- | --- | --- |
| 定义链 | 内部闭合 | 无主线定义阻塞；更一般 animated/stack variants 不在本轮范围 | 保持 bare prism/bounded prism、complete/ordinary Tor 边界 |
| 比较定理链 | 正文核心 prismatic/Nygaard/BMS1/BMS2/$L\eta$/$F$-crystal 主链已到 L3 | 非主线配套结果仍有 L2S；classical 仍低于 L2S | 补配套 locator 与 classical source selection |
| 符号链 | completion ideal、Frobenius twist、Tate twist、truncation 与 inversion 层级已核对 | 跨文献 Tate-twist normalization 尚需出版复核 | 完成 normalization crosswalk |
| 前沿边界 | 已分层 | 2026 后续版本可能更新 | 定期联网核查 |
| 练习体系 | 章末练习解答已闭合 | 出版版可扩展多解法和课堂提示 | 按教学需要继续增补 |
| 技术基础 | 已补附录 G-K | 仍非完整 EGA/SGA 替代 | 保持为本书最小技术词典 |
| 逐章正文 | 收口草稿 | 仍需最终 copy-editing | 按 `CHAPTER_CLOSURE_AUDIT.md` 做出版校对 |
| 出版校对 | 未完成 | 术语、断行、交叉引用 | 最终 copy-editing |

## 结论

本书当前可作为逐章教材收口草稿继续迭代；derived completion、核心 prismatic/Nygaard comparisons、BMS1/BMS2/$L\eta$、BK/BKF 与 $F$-crystal 主线已经有稳定定义边界和 PDF-numbered locators。它不能标为 camera-ready。数学上最大的剩余风险已经收缩为 classical comparison 源选择、非主线配套 locators，以及出版校对。
