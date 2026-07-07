# 正式教材闭包矩阵

核查日期：2026-07-08。

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
- 已新增 [P0_REFERENCE_LOCATORS_BATCH_1.md](P0_REFERENCE_LOCATORS_BATCH_1.md)，把 Bhatt-Scholze prismatic 主源、BMS1 和 prismatic $F$-crystal 主源的 P0 条目升级到源码 label 级 `L2S`。
- 已新增 [P0_REFERENCE_LOCATORS_BATCH_2.md](P0_REFERENCE_LOCATORS_BATCH_2.md)，把 BMS2/THH-BK、BMS2 syntomic/Tate twists 和 Bhatt-Lurie prismatization 升级到源码 label 级 `L2S`。
- 2025-2026 前沿资料仍被限制在研究边界，不进入基础定理链。

## 闭包矩阵

| 维度 | 当前状态 | 阻塞项 | 下一步 |
| --- | --- | --- | --- |
| 定义链 | 内部闭合 | completed flatness 的最终文献口径仍需 locator | 用 Bhatt-Scholze 精确定义替换附录 A.9 |
| 比较定理链 | 系统化，核心 prismatic/BMS2/BL 已到 L2S | P0 locator 未到 L3；classical 仍低于 L2S | 升级附录 D 到出版 locator |
| 符号链 | Bhatt-Scholze Hodge-Tate/Nygaard 和 BMS2 syntomic 基础公式已源码核对并吸收到正文 | mod $p^r$、truncation、nearby cycles 变体未到 L3 | 完成所有变体的出版 locator |
| 前沿边界 | 已分层 | 2026 后续版本可能更新 | 定期联网核查 |
| 练习体系 | 章末练习解答已闭合 | 出版版可扩展多解法和课堂提示 | 按教学需要继续增补 |
| 技术基础 | 已补附录 G-K | 仍非完整 EGA/SGA 替代 | 保持为本书最小技术词典 |
| 逐章正文 | 收口草稿 | 仍需最终 copy-editing | 按 `CHAPTER_CLOSURE_AUDIT.md` 做出版校对 |
| 出版校对 | 未完成 | 术语、断行、交叉引用 | 最终 copy-editing |

## 结论

本书当前可作为逐章教材收口草稿继续迭代；核心 prismatic、BMS2 和 prismatization 接口已经具备源码级复查入口，BMS2 syntomic 基础公式也已吸收到第七章和第十一章。它不能标为 camera-ready。数学上最大的剩余风险已经从“核心来源不可定位”收缩为“L2S 到 L3 的出版 locator 转换、syntomic 变体细分、classical comparison 源选择”。
