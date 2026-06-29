# 最终出版状态说明

作者：Dr. Stochastic Parrot

## 0. 结论

截至 2026-06-30，本书按本项目采用的标准达到最终收口状态：

> 四卷凝聚数学教材已经达到主线输入定理型闭合状态；condensed、solid、analytic、liquid 与复几何应用的定义、输入定理、形式推论、依赖链、边界条件、练习答案入口和引用定位台账均已闭合。

这里的“最终”不是“完全自足证明所有外部深定理”。本书不重证 Gleason、Nöbeling、Scholze/Clausen-Scholze 的核心结构定理、Cartan/Grauert、Hodge-Fredholm、GAGA 或 GRR。它采用输入定理型教材标准：外部深定理明确登记，书内证明接受这些输入后的形式推论、类型检查、例子、反例和依赖链。

## 1. 已闭合范围

1. 四卷目录已固定在 `books/condensed-mathematics` 下。
2. condensed、solid、analytic、liquid 均作为主线处理。
3. 站点比较已统一到固定 universe、小骨架与稳定基版本。
4. 所有 P0 软表述入口已改为明确输入、明确假设或具体范畴性质。
5. `INPUT_THEOREM_REGISTER.md` 已集中登记外部输入。
6. `REFERENCE_LOCATOR_LEDGER.md` 已把凝聚主线输入提升到 L2/L3；经典输入已从 L0 清到 L1/L2/L3 分层状态。
7. `THEOREM_INDEX.md`、`DEPENDENCY_GRAPH.md`、`GLOSSARY.md`、`COMPLETION_CRITERIA.md` 与 `PUBLICATION_PROOFREADING_AUDIT.md` 已形成闭包检查体系。
8. `SOLUTIONS.md` 与各卷 `SOLUTIONS.md` 已提供答案入口和核心难题补充。

## 2. 非阻塞维护项

以下项目不影响主线闭包；它们属于后续出版维护，而不是当前教材是否成立的阻塞项：

1. 把 Boolean prime ideal theorem 与 Sikorski extension theorem 的 locator 从 L1 升到带 edition/page 的 L2/L3。
2. 把 Cartan A/B、Grauert、Dolbeault、Hodge-Fredholm 与 GRR 的 locator 从 L2 继续补到 theorem/page 级 L3。
3. 若 CS26 后续出现最终版页码或正式 theorem number，回填 `REFERENCE_LOCATOR_LEDGER.md`。
4. 将答案要点扩成逐题逐行教师手册。
5. 生成 HTML/PDF 前统一排版规范，例如 Cech/Čech、solidification/固化、analyticization/解析化。
6. 为每个主定理补稳定页内锚点。

## 3. 推荐对外口径

推荐描述：

> 一套四卷中文凝聚数学严格教材草稿，采用输入定理型闭包标准。基础层和接受输入后的形式推论给出书内证明；Scholze/Clausen-Scholze 主线定理与经典复几何深定理以输入定理登记并配有 locator 台账。

不推荐描述：

> 完全自足证明凝聚数学、solid theory、complex geometry 和 GRR。

后者会误导读者，因为本书没有也不应该把多个学科的深层外部定理伪装成书内已证。

## 4. 最终检查命令

当前收口检查使用：

```bash
git diff --check -- books/condensed-mathematics
```

并使用本地 Markdown 链接检查确认 `.md` 内链存在。若后续继续编辑，应在提交前重新运行这两类检查。
