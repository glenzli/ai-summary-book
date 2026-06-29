# 出版级校对台账

作者：Dr. Stochastic Parrot

## 0. 当前结论

本书已经达到“主线输入定理型最终收口版”标准。出版级校对的阻塞项已经处理完毕；后续不应继续大量新增章节，而应只做非阻塞维护：

1. 引用定位；
2. 软表述替换；
3. 输入定理编号回填；
4. 练习答案扩成教师手册；
5. 术语和符号一致性检查。

最终状态说明见 [FINAL_PUBLICATION_STATUS.md](FINAL_PUBLICATION_STATUS.md)。

## 1. 已完成的校对动作

1. 已确认阶段性提交 `dd0c58f` 只包含 `books/condensed-mathematics`。
2. 已核验提交时间为 `2026-06-29 23:34:42 +0800`。
3. 已运行 `git diff --check -- books/condensed-mathematics`。
4. 已运行内部 Markdown 链接解析检查。
5. 已新增 [REFERENCE_LOCATOR_LEDGER.md](REFERENCE_LOCATOR_LEDGER.md)，将核心输入定理的引用定位状态分为 L0-L3。
6. 已将新写的核心图谱中“合适/适当”等软假设改为明确输入条件。
7. 已将第一批 P0 软表述入口改写为明确输入编号或具体性质，包括 solid、analytic、liquid 和 Clausen-Scholze 建模入口。
8. 已将第一卷站点比较入口改成固定 universe、小骨架、稳定基和共同细化条件。
9. 已把凝聚主线引用从单纯 arXiv 号提升到讲次、章节和 TeX label 级 locator；当前凝聚主线 L1 条目已清零。
10. 已在 [INPUT_THEOREM_REGISTER.md](INPUT_THEOREM_REGISTER.md) 为 A-D 类输入回填来源定位，明确区分 Scholze/Clausen-Scholze 主线输入与经典拓扑、复几何输入。
11. 已将各卷 `SOURCES.md` 中的粗粒度“主要依赖”口径改为输入编号、文献代号和 locator 台账口径。
12. 已清理定理级软表述：solid 模范畴、liquid realization、\(p\)-liquid 范围和 Fredholm-Sobolev 输入均改为明确假设或输入子范畴。
13. 已补第二轮经典输入 locator：Gleason、Serre duality、GAGA 升到 L3；Dolbeault、Cartan A/B、Grauert、Hodge-Fredholm、GRR 升到 L2；Boolean prime ideal theorem 与 Sikorski extension theorem 暂为 L1。
14. 已在第三卷 `SOURCES.md` 增补 Cartan、Grauert、Serre、Borel-Serre/SGA 6、Wells 与 Huybrechts 等经典复几何来源。
15. 已新增 [FINAL_PUBLICATION_STATUS.md](FINAL_PUBLICATION_STATUS.md)，将最终收口范围与非阻塞维护项分开。
16. 已将总 [README.md](README.md) 的长段数学口径压缩为出版说明。

## 2. 软表述扫描

扫描词包括：

$$
\text{适当、良好、通常、可视为、类似、标准事实、显然、容易、合适。}
$$

扫描结果说明：

1. 许多出现位于解释段、练习、反例或答案中，可以保留。
2. 需要优先处理的是定理陈述、输入定理和证明中承担逻辑作用的软表述。
3. 当前 P0 目标不是删掉所有词，而是禁止它们承担证明功能。

## 3. P0 校对清单

| 位置 | 问题 | 状态 |
| --- | --- | --- |
| volume-1 `05_comparison_of_test_sites.md` | “适当表述后”用于站点比较入口 | 已处理：改成固定 universe、小骨架、稳定基和共同细化条件 |
| volume-1 `C_sheaves_of_abelian_groups.md` | “标准事实”未附 locator | 已处理：改为引用附录 H 证明 |
| volume-1 `12_solid_abelian_groups.md` | “良好的阿贝尔范畴”作为输入概括 | 已处理：改为第二卷 D.1-D.3 输入边界 |
| volume-1 `14_analytic_rings.md` | “良好范畴”“可视为解析化” | 已处理：改为 D.4 和 analytic kernel 张量理想性表述 |
| volume-2 `01_solid_derived_categories.md` | “可视为使所有映射...” | 已处理：改为反射局部化泛性质 |
| volume-2 `03_analytic_rings_formal_conditions.md` | “良好的派生范畴环境” | 已处理：改为反射稳定子范畴 |
| volume-2 `05_liquid_vector_spaces.md` | “合适的 \(p\)” | 已处理：改为 D.5 指定范围内的 \(p\) |
| volume-3 `01_complex_analytic_spaces_condensed_language.md` | “适当的复解析空间” | 已处理：改为 Clausen-Scholze 建模输入适用范围 |
| volume-3 `03_dolbeault_complexes_and_liquid_modules.md` | “合适的 condensed/analytic 派生范畴” | 已处理：改成第二卷 D.6 和第三卷 AR.2 的接口 |

## 4. 非阻塞维护清单

1. 将剩余正文中的“由 Scholze 定理”改为“由输入定理 D.x / B.x / C.x”；当前扫描只剩规则说明性文字。
2. 给 Boolean prime ideal theorem 与 Sikorski extension theorem 补 edition theorem/page，把剩余经典 L1 提升到 L2。
3. 给 Cartan A/B、Grauert、Dolbeault、Hodge-Fredholm 和 GRR 的 L2 locator 补具体 theorem/page，提升到 L3。
4. 把 `SOLUTIONS.md` 中核心证明题从答案要点扩成逐行教师手册。
5. 给没有源码 label 的 CS26 定理补页码、讲义编号或最终版本 theorem number。

这些项目不影响本书作为“主线输入定理型最终收口版”的成立；它们属于后续出版维护。

## 5. 远期排版增强清单

1. 为每个主定理增加页内锚点或稳定标签。
2. 为每个外部输入补一个删除假设失败例子。
3. 生成 HTML/PDF 前，统一 Cech/Čech、solidification/固化、analyticization/解析化的中英混排规则。

## 6. 当前不建议做的事

1. 不建议继续新增第五卷或大量新附录。
2. 不建议把外部深定理改写成伪证明。
3. 不建议一次性重写所有基础章节；应按 P0 清单逐节处理。
4. 不建议提交其它书的改动到本轮凝聚数学提交中。
