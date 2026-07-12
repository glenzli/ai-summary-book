# 教材内容收口审定

本文档只判定“教材内容本身是否收口”，不判定排版、出版编辑、图表美化或最终例题密度。

## 1. 总判定

当前版本达到教材内容层面的基本收口：

1. 内容范围已闭合：第 0 至 20 章覆盖 worldsheet、CFT、量子化、BRST、振幅、超弦、D-branes、紧化、对偶性、topological strings、black branes、AdS/CFT、flux compactification 和外部接口边界。
2. 内部依赖已闭合：主线依赖由 [DEPENDENCY_GRAPH.md](DEPENDENCY_GRAPH.md) 和 [MAINLINE_PROOF_CHAINS.md](MAINLINE_PROOF_CHAINS.md) 给出；后置章节不再依赖未定义的核心对象。
3. 证明状态已闭合：非平凡陈述已登记在 [THEOREM_INDEX.md](THEOREM_INDEX.md)，并标为 `P`、`S`、`E` 或 `C`。
4. 外部输入已闭合：大型外部定理和物理猜想未被写成无条件已证定理，使用边界在正文或本文件中说明。
5. 引用范围已闭合：核心教材、CFT、几何、对偶性和 AdS/CFT 的资料源列于 [SOURCES.md](SOURCES.md)，逐章映射列于 [REFERENCE_MAP.md](REFERENCE_MAP.md)。

因此，本书可以视为“内容收口版”。后续工作应是出版化细化，而不是继续扩张主线。

## 2. 证明闭合标准

本书不要求把所有外部数学和物理理论都在正文中重证。证明闭合采用四类状态：

- `P`：正文给出证明。
- `S`：正文给出足以支撑主线使用的标准物理推导说明。
- `E`：外部输入定理，正文给出使用边界和资料源。
- `C`：物理猜想或对偶性原则，正文不得把它当作数学定理使用。

按此标准，证明义务已经闭合。剩余的“可继续补强”项，例如完整 Kac determinant、no-ghost theorem 证明、Yau theorem、DUY theorem、supermoduli theory、AdS/CFT 非微扰证明，不属于本书主线内部证明义务。

## 3. 章节内容闭合表

| 范围 | 内容状态 | 证明状态 | 引用状态 |
|---|---|---|---|
| 0-2 | 作用量、变分、经典弦 | 闭合 | 闭合 |
| 3-5 | CFT、Virasoro、ghost、BRST | 闭合，表示论保留接口 | 闭合 |
| 6-8 | 顶点、振幅、T-duality、RNS | 闭合，loop/supermoduli 保留接口 | 闭合 |
| 9-12 | type II、heterotic、低能作用、D-branes | 闭合，kappa/anomaly/inflow 标为外部输入 | 闭合 |
| 13-16 | CY、duality、Riemann surfaces、topological strings | 闭合，几何大定理标为外部输入 | 闭合 |
| 17-20 | BPS、AdS/CFT、flux、接口边界 | 闭合，非微扰对偶标为猜想 | 闭合 |
| A-E | 附录接口 | 足以支撑正文 | 闭合 |

## 4. 仍可改进但不阻塞收口的项目

以下项目属于出版化或高级专题扩展，不影响内容收口：

1. 增加图表、书末排印索引和历史注。
2. 增加更多例题，例如 explicit residue、orientifold tadpole、quintic Yukawa coupling。
3. 把部分标准物理推导说明扩成完整证明，例如 light-cone Lorentz algebra closure。
4. 为每章增加历史注和进一步阅读。
5. 将附录 B-E 扩成独立公式手册。

## 5. 后续准入规则

后续新增内容必须满足至少一项：

1. 修正数学或物理错误。
2. 补足现有章节的例题、习题、证明细节。
3. 改善引用映射、索引、符号和排版一致性。
4. 加入不改变主线范围的附录公式表。

不应新增主线章，也不应把外部学科扩展成另一部教材。
