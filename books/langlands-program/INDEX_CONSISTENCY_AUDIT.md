# 索引一致性审计

本文档记录 [MAINLINE_PROOF_CHAINS.md](MAINLINE_PROOF_CHAINS.md)、[THEOREM_INDEX.md](THEOREM_INDEX.md) 与 [E_external_input_theorem_index.md](E_external_input_theorem_index.md) 的收口审计结果。

## 审计标准

1. 证明链中标为 `P` 的步骤，应能在 [THEOREM_INDEX.md](THEOREM_INDEX.md) 中找到同一编号或对应命题，并且状态为 `P` 或由 `P/S` 组合推出。
2. 证明链中标为 `S` 的步骤，应能在 [THEOREM_INDEX.md](THEOREM_INDEX.md) 中找到证明草图状态，或者在正文中明确依赖外部输入。
3. 证明链中标为 `E` 的步骤，应能在 [THEOREM_INDEX.md](THEOREM_INDEX.md) 中标为 `E`，并能在 [E_external_input_theorem_index.md](E_external_input_theorem_index.md) 或 [SOURCES.md](SOURCES.md) 中追溯来源。
4. 证明链中标为 `C` 的步骤，不能作为应用章的无条件输入。

## 总体结论

四条主线的关键节点均有索引落点。当前没有发现会阻断基本收口的状态冲突。

仍需在后续精校中继续做两类工作：

1. 附录层外部输入的逐条来源细化，例如把若干 “参考标准文献” 进一步拆成具体定理名。
2. 附录中同一理论的多处接口保持同一状态，例如 Satake、BZ 分类、Arthur 分类、几何 Satake、Fargues-Scholze。

## `GL(1)` 链

| 证明链节点 | 状态审计 | 来源审计 | 结论 |
|---|---|---|---|
| $1.15$、$1.21$ | `E` | 附录 E.1 记录 Pontryagin duality、Fourier/Poisson、Tate-Weil Fourier 分析 | 一致 |
| $2.9$、$2.13$ | `E` | 附录 E.1 记录 Tate thesis | 一致 |
| $3.2$、$3.11$ | `E` | 附录 E.1 记录局部/全局类域论和 class formations | 一致 |
| $3.5$、$3.6$、$3.14$、$3.15$、$3.16$ | `P` | 定理索引第一部分标为 `P` | 一致 |
| $3.17$ | `E` | 附录 E.1 与 SOURCES 中 class field theory/Tate 来源覆盖 | 一致 |

判断：`GL(1)` 链已满足基本收口。完整 class formation 证明后置或另卷。

## 费马应用链

| 证明链节点 | 状态审计 | 来源审计 | 结论 |
|---|---|---|---|
| $90.2$ | `P` | 定理索引应用章标为 `P` | 一致 |
| $90.5$ | `E` | 附录 E.1/E.3 记录 Frey 曲线导子、Tate algorithm、Ribet/Frey 来源 | 一致 |
| $90.7$ | `E` | 附录 E.3 记录 Wiles、Taylor-Wiles、BCDT | 一致 |
| $90.8$ | `E` | 附录 E.3 记录 Ribet 降层 | 一致 |
| $90.9$ | `P` | 定理索引标为 `P`，依赖附录 D/W 的 genus 和 newspace 接口 | 一致 |
| $90.10$、$90.1$ | `P` | 定理索引应用章标为 `P` | 一致 |

判断：费马链的逻辑闭环成立。它不是 Wiles-Taylor-Wiles 或 Ribet 证明本身。

## 一般算术 Langlands 链

| 证明链节点 | 状态审计 | 来源审计 | 结论 |
|---|---|---|---|
| Haar、Hecke、Satake | `P/E/S` | 定理索引第 4、5 章和附录 B/P/Z/AA；附录 E.2 覆盖 Satake、Bruhat-Tits、Harish-Chandra | 一致 |
| 还原群、根资料、L 群 | `P/E/S` | 第 11 章和附录 G；附录 E.2 覆盖 reductive groups/root datum | 一致 |
| LLC 和 packets | `C/E/P` | 第 12、14 章和附录 N/Q/AE；附录 E.2 覆盖 `GL(n)` LLC、tori、一般 reductive LLC 接口 | 一致 |
| 全局自守表示和 L 函数 | `P/E/C` | 第 13、14 章和附录 I/L/M/U；附录 E.4 覆盖 Godement-Jacquet、Rankin-Selberg、Shahidi、RAECSDC Galois 表示 | 一致 |
| 函子性 | `P/E/C` | 第 15 章；附录 E.5 覆盖 base change、low lifts、Arthur-Mok 接口 | 一致 |
| trace formula/endoscopy | `P/E/S` | 第 16 章和附录 N/R；附录 E.5 覆盖 Arthur trace formula、fundamental lemma、endoscopic transfer | 一致 |
| Arthur 参数 | `P/E/S` | 第 17 章和附录 X；附录 E.5 覆盖 Arthur classification、multiplicity formula、standard transfer | 一致 |

判断：一般算术 Langlands 链达到对象链收口；证明层仍依赖外部输入和猜想，状态标记没有混用。

## 几何 Langlands 链

| 证明链节点 | 状态审计 | 来源审计 | 结论 |
|---|---|---|---|
| $\operatorname{Bun}_G$、Hecke stack | `P/E` | 第 18 章；附录 E.6 覆盖代数栈和 uniformization 来源 | 一致 |
| 几何 Satake | `E/P/S` | 第 19 章和附录 Y；附录 E.6 覆盖 geometric Satake 文献 | 一致 |
| Hecke eigensheaf | `P/C/E` | 第 20 章；附录 E.6 覆盖几何类域论和 Hecke eigensheaves | 一致 |
| 范畴化几何 Langlands | `C/E/S` | 第 21 章和附录 O/AB；附录 E.6 覆盖 D-modules、IndCoh、singular support、categorical GL | 一致 |
| sheaf-function dictionary | `E/S` | $20.11$、$22.3$、$22.5$；附录 E.6 覆盖 Grothendieck-Lefschetz 和 sheaf-function | 一致 |
| Drinfeld-Lafforgue/shtukas | `E` | $22.6$、$22.7$、$22.11$、附录 S；附录 E.6 和 SOURCES 记录 Drinfeld、Lafforgue、V. Lafforgue | 一致 |
| Fargues-Fontaine/Fargues-Scholze | `E/P/S` | 附录 AC；附录 E.6 和 SOURCES 记录 Fargues、Fontaine、Scholze、Fargues-Scholze | 一致 |

判断：几何链达到接口收口；完整范畴技术层不作为本书基本收口目标。

## 后续精校项

| 项目 | 优先级 | 处理方式 |
|---|---|---|
| 附录 E 对 `90.5` Frey 曲线性质的来源可拆得更细 | 已完成 | 已把 Frey-Hellegouarch、Tate algorithm、Serre-Ribet local computation 分成独立条目 |
| `Satake` 相关外部输入分布在 4、5、11、12、13、19、P、Y | 已完成 | 已在附录 E.2/E.6 拆分 classical Satake、非分歧参数和 geometric Satake；继续保持 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 第 4、9 节为总 convention |
| 几何 Langlands 的 `C` 与 `E` 边界 | 中 | 范畴化对应本身保持 `C`，技术定理保持 `E` |
| 数域完整全局 Langlands | 高 | 继续保持 `C`，不能被 `GL(n)` 已知情形或 RAECSDC 构造替代 |

## 收口判定

索引一致性达到第一轮收口要求：主线证明链没有把猜想当作定理使用，也没有发现外部输入缺少资料源大类的问题。附录层归一化回指已在第二轮收口中补齐，逐章风险已在第三轮收口中改写为 [CHAPTER_CLOSURE_AUDIT.md](CHAPTER_CLOSURE_AUDIT.md)，重点外部输入来源已在第四轮收口中拆细；后续剩余工作是编号和交叉引用审稿。
