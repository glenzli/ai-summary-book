# 附录 G：前沿预印本进入正文的验证协议

## G.1 分级

**定义 G.1.** 本书把资料分为四级：

| 级别 | 含义 | 可用于证明链 |
| --- | --- | --- |
| Core | 经典教材、专著或稳定发表定理 | 可以 |
| External theorem | 论文定理，已完成 locator 和假设翻译 | 可以 |
| Frontier | 近期预印本或新结果，已核查摘要和版本 | 不可以，除非升级 |
| Lead | 线索、讲义、博客或百科 | 不可以 |

**规则 G.2.** Frontier 升级为 External theorem 需要完成：

1. arXiv/DOI/发表版本定位；
2. theorem/proposition/corollary 编号；
3. 假设逐条翻译到本书 notation；
4. 结论逐条翻译到本书 notation；
5. 与既有定理链的依赖关系；
6. 失败模式检查；
7. 在 `THEOREM_LEDGER.md` 和 `D_source_theorem_index.md` 同步登记。

## G.2 版本固定

**规则 G.3.** 引用 arXiv 预印本时必须记录版本号和日期。若正文使用的是 v3 结论，不得只写 arXiv 编号而不写版本。

**规则 G.4.** 若预印本已有发表版本，以发表版本为主；若发表版本与 arXiv 版本不同，需说明采用哪一个。

## G.3 假设翻译

**检查表 G.5.** 对每个前沿定理，必须回答：

- 固定素数范围是什么？
- 高度范围是什么？
- 对象是 spectrum、ring spectrum、$\mathbb E_1$-ring、$\mathbb E_\infty$-ring 还是 stable infinity-category？
- 使用 $K(n)$、$T(n)$、$E(n)$ 还是 $L_n^f$？
- 是否要求 finite、compact、dualizable、bounded below 或 $p$-complete？
- 结论是在 homotopy category、stable infinity-category、module category 还是 algebraic K-theory spectrum 中？

## G.4 前沿定理的降级处理

**规则 G.6.** 若无法完成假设翻译，正文只能使用以下措辞：

- “前沿记录”；
- “研究边界”；
- “该方向提供证据/路线”；
- “进入正文前需 theorem locator”。

不得使用：

- “已经证明一般地”；
- “因此基础理论中可认为”；
- “自动推出”；
- “无需区分”。

## G.5 当前 Frontier 项

| 项目 | 当前状态 | 升级前任务 |
| --- | --- | --- |
| BHLŠ telescope counterexample | 已核查，接近 P0 | 主定理 locator 和记号换算 |
| Hahn-Wilson redshift | 已核查，接近 P0 | theorem A/B/Corollary locator |
| Chromatic Nullstellensatz | 已核查，接近 P0 | nilpotence/redshift corollary locator |
| BCSY cyclotomic redshift | 已核查，P0 候选 | JAMS 版本和 theorem locator |
| BSSW rational $K(n)$-local sphere | 已核查，P1 | 主计算定理和 rational/integral 边界 |
| Ben-Moshe semiadditivity 2024/2025 | 已核查，P1 | 与 Hopkins-Lurie/CSY 的依赖比较 |
| Angelini-Knoll 2026 BP<n> syntomic | 已核查，Frontier | v3 theorem locator 和术语边界 |
| Behrens-Carlisle equivariant periodicity | 已核查，P1 | equivariant theorem locator |
| Allen-Piessevaux synthetic equivariant motivic | 已核查，Frontier | completion/cellular/base-field 假设 |

## 本附录小结

正式教材可以覆盖前沿，但不能让前沿结果破坏证明链。所有近期结果必须先通过版本、定理编号和假设翻译，再从 Frontier 升级为 External theorem。
