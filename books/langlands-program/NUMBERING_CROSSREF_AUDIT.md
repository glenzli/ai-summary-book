# 编号与交叉引用审计

本文档记录本书在主线收口阶段的编号一致性和交叉引用审计结果。审计目标不是把所有定义、注和练习都加入 [THEOREM_INDEX.md](THEOREM_INDEX.md)，而是确认已进入主线证明链、定理索引、外部输入索引和习题解答的编号没有漂移。

## 审计范围

本轮审计覆盖以下对象。

1. [THEOREM_INDEX.md](THEOREM_INDEX.md) 中所有表格编号。
2. 正文第 0-22 章、第 90 章和附录 A-AE 中实际出现的定理、命题、引理、推论、猜想、条件、定义、例、注和练习编号。
3. [SOLUTIONS.md](SOLUTIONS.md) 中所有已写解答的练习编号。
4. `books/langlands-program` 内部 Markdown 相对链接。
5. 第 3、7、10、14、16、19、22、90 章的归一化回指。

## 自动检查结果

| 检查项 | 数量 | 结果 |
|---|---:|---|
| 定理索引编号 | 551 | 全部能在对应正文或附录文件中找到落点 |
| 定理索引错误文件落点 | 0 | 未发现编号落在错误章节或附录 |
| 正文/附录实际编号 | 1139 | 其中 590 个未入定理索引，属于定义、注、例、练习和非主线局部结果 |
| 解答文件中的练习编号 | 134 | 全部能找到对应原题 |
| 全书实际练习编号 | 260 | 解答覆盖核心练习，未要求全量覆盖 |
| Markdown `.md` 相对链接 | 278 | 未发现断链 |

## 结论

当前没有发现阻断收口的编号漂移。

[THEOREM_INDEX.md](THEOREM_INDEX.md) 的定位是“主要定理、命题、外部输入和猜想的状态索引”，不是全量编号索引。因此正文和附录中存在大量未入索引的定义、注、例和练习是预期状态。只有以下内容必须继续进入或维护索引：

1. 主线证明链中使用的结果。
2. 应用章使用的外部输入。
3. 跨章节反复调用的归一化、局部因子、导子、Satake、Galois 表示和几何 Satake 接口。
4. 状态可能混淆为 `P/S/E/C` 的结果。

## 重点章节回指

第 3、7、10、14、16、19、22、90 章均已含 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 回指。其作用如下。

| 章节 | 回指作用 |
|---|---|
| 第 3 章 | 固定 reciprocity、Frobenius 和一维局部 L 因子 convention |
| 第 7 章 | 固定 classical normalization 与 automorphic normalization 的转换 |
| 第 10 章 | 固定 residual conductor、Weil-Deligne 参数和局部因子转换 |
| 第 14 章 | 固定 `GL(n)` LLC、Rankin-Selberg、函数域和数域 Galois 表示比较 |
| 第 16 章 | 固定 trace formula、orbital integral、transfer factor 的测度口径 |
| 第 19 章 | 固定几何 Satake 与 classical Satake 的 $q$-因子和 Tate twist |
| 第 22 章 | 固定函数域 Frobenius、trace function 和 sheaf-function convention |
| 第 90 章 | 固定 Frey 曲线、残余表示、导子和模形式 L 函数比较 |

## 后续维护规则

1. 新增主线命题、外部输入或猜想时，同步更新 [THEOREM_INDEX.md](THEOREM_INDEX.md)。
2. 新增外部输入时，先登记 [E_external_input_theorem_index.md](E_external_input_theorem_index.md) 或 [SOURCES.md](SOURCES.md)。
3. 新增已解习题时，确认 [SOLUTIONS.md](SOLUTIONS.md) 中编号与原题一致。
4. 新增跨文件链接时，使用相对 Markdown 链接，并确认目标文件存在。
5. 不把所有定义和注强行加入定理索引，避免索引失去状态审计功能。
