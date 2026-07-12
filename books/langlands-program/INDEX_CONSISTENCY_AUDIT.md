# 索引一致性审计

本文档记录 [MAINLINE_PROOF_CHAINS.md](MAINLINE_PROOF_CHAINS.md)、[THEOREM_INDEX.md](THEOREM_INDEX.md) 与 [E_external_input_theorem_index.md](E_external_input_theorem_index.md) 的收口审计结果。

## 审计标准

1. 证明链中标为 `P` 的步骤，应能在 [THEOREM_INDEX.md](THEOREM_INDEX.md) 中找到同一编号或对应命题，并且状态为 `P` 或由 `P/S` 组合推出。
2. 证明链中标为 `S` 的步骤，应能在 [THEOREM_INDEX.md](THEOREM_INDEX.md) 中找到外部输入的证明路线状态，或者在正文中明确依赖外部输入。
3. 证明链中标为 `E` 的步骤，应能在 [THEOREM_INDEX.md](THEOREM_INDEX.md) 中标为 `E`，并能在 [E_external_input_theorem_index.md](E_external_input_theorem_index.md) 或 [SOURCES.md](SOURCES.md) 中追溯来源。
4. 证明链中标为 `C` 的步骤，不能作为应用章的无条件输入。

## 总体结论

四条主线的关键节点均有索引落点。当前没有发现会阻断审定前闭合版维护的状态冲突。

本轮更新后 [THEOREM_INDEX.md](THEOREM_INDEX.md) 含 558 条编号记录；新增的 2.4.1、5.8.1、5.8.2、10.2.1、13.8.1、21.8.1、O.16.1 均已逐项核对正文落点。当前运行
`python3 books/audit_oet_rigor.py langlands-program` 的结果为 `errors=0 warnings=0`，Markdown/LaTeX 结构与本地链接检查通过。

审定前维护中继续做两类工作：

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

判断：`GL(1)` 链已满足审定前闭合版的索引一致性要求。完整 class formation 证明后置或另卷。

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
| Haar、Hecke、Satake | `P/E/S` | 定理索引第 4、5 章和附录 B/P/Z/AA；$e_J=\operatorname{vol}(J)^{-1}1_J$，split 与仅 unramified Satake 参数已分开；附录 E.2 覆盖来源 | 一致 |
| 还原群、根资料、L 群 | `P/E/S` | 第 11 章和附录 G；附录 E.2 覆盖 reductive groups/root datum | 一致 |
| LLC 和 packets | `C/E/P` | 第 12、14 章和附录 N/Q/AE；附录 E.2 覆盖 `GL(n)` LLC、tori、一般 reductive LLC 接口 | 一致 |
| 全局自守表示和 L 函数 | `P/E/C` | 第 7、10、13、14 章和附录 I/L/M/U；unitary/classical 变量平移、Galois dual/twist 和函数方程单次对偶已统一；附录 E.3/E.4 覆盖来源 | 一致 |
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
| 范畴化几何 Langlands | `E/C/P` | 21.7/O.15 为特征零外部 preprint theorem；21.8.1/O.16.1 为正特征某些连通分支上的部分外部 preprint theorem；unrestricted、ramified、integral、quantum 与 local variants 保持边界 | 一致 |
| sheaf-function dictionary | `E/S` | $20.11$、$22.3$、$22.5$；附录 E.6 覆盖 Grothendieck-Lefschetz 和 sheaf-function | 一致 |
| Drinfeld-Lafforgue/shtukas | `E` | $22.6$、$22.7$、$22.11$、附录 S；附录 E.6 和 SOURCES 记录 Drinfeld、Lafforgue、V. Lafforgue | 一致 |
| Fargues-Fontaine/Fargues-Scholze | `E/P/S` | 附录 AC；附录 E.6 和 SOURCES 记录 Fargues、Fontaine、Scholze、Fargues-Scholze | 一致 |

判断：几何链达到接口收口；完整范畴技术层不作为本书审定前闭合版目标。

## 出版前审定维护项

| 项目 | 优先级 | 处理方式 |
|---|---|---|
| 附录 E 对 `90.5` Frey 曲线性质的来源可拆得更细 | 已完成 | 已把 Frey-Hellegouarch、Tate algorithm、Serre-Ribet local computation 分成独立条目 |
| `Satake` 相关外部输入分布在 4、5、11、12、13、19、P、Y | 已完成 | 已在附录 E.2/E.6 拆分 classical Satake、非分歧参数和 geometric Satake；继续保持 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 第 4、9 节为总 convention |
| 几何 Langlands 的 `C` 与 `E` 边界 | 已完成 | 特征零 proof series 标 `E`；正特征部分分支结果标 `E`；未覆盖版本继续作为研究边界，不以旧的总括 `C` 或总括 `E` 混写 |
| 数域完整全局 Langlands | 高 | 继续保持 `C`，不能被 `GL(n)` 已知情形或 RAECSDC 构造替代 |

## 收口判定

索引一致性达到本轮闭合要求：主线证明链没有把猜想当作定理使用，新增外部输入均在附录 E 与 [SOURCES.md](SOURCES.md) 有来源落点。Class field theory、newforms、Satake、trace formula、几何与 p-adic Langlands 的外部边界已经按实际定理范围重写；正文中的 elementary/formal consequences 仍由 `P` 标记并给出证明。后续维护若改变任何 Frobenius、Haar、Tate twist 或 automorphic normalization，必须同步修改正文、归一化总表和三个索引文件。
