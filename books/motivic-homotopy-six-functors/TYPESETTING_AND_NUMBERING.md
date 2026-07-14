# 编号、排版与交叉引用规范

核查日期：2026-07-15

本规范用于统一本书的章节编号、命题环境、练习编号、引用方式和 Markdown 排版。

## 1. 文件与章节结构

正文文件按两位数字排序：

```text
00_preface_and_scope.md
01_base_schemes_smooth_sites_and_nisnevich_descent.md
...
24_research_frontier_2026_open_problems_and_source_boundaries.md
```

附录文件按大写字母排序：

```text
A_universes_presentability_and_localization.md
...
H_worked_examples_and_basic_computations.md
```

每个正文章采用如下骨架：

```text
# 第n章：标题

以内容特定的自然段引出问题、约定与必要前置知识。

## n.1 第一节
...
## n.m 内容特定的收束标题
## 练习
```

自然导言与收束段必须说明本章实际数学内容，不设置固定的“本章目标”“依赖前置知识”
或“本章小结”栏目。必要依赖应在首次使用处精确回指。

附录采用：

```text
# 附录 A：标题

## 本附录目标
## 依赖前置知识
## A.1 第一节
...
## A.m 本附录小结
## 练习
```

## 2. 命题环境编号

正文命题环境按章内编号：

```text
**定义 5.1.**
**外部输入定理 5.2.**
**命题 5.3.**
**推论 5.4.**
**注 5.5.**
**例子 5.6.**
```

附录命题环境按附录字母编号：

```text
**定义 A.1.**
**定理 A.2.**
**命题 A.3.**
```

规则：

- 编号局部于每章或每个附录。
- 同一章内不为定义、命题、定理分别开独立计数；它们共用章内序列。
- “外部输入定理”必须保留此标签，不改成普通“定理”。
- 研究边界结果不得编号为无条件定理；应使用“研究边界”“问题”“注”或明确假设的外部输入。

## 3. 练习与解答编号

练习编号只按章内练习顺序：

```text
**练习 8.1.**
**练习 8.2.**
```

附录练习写作：

```text
**练习 D.1.**
```

解答文件 [EXERCISE_SOLUTIONS.md](EXERCISE_SOLUTIONS.md) 使用同一编号：

```text
**解答 8.1.**
```

若后续加入更完整详解，不改变编号，只在对应解答下扩展证明。

## 4. 数学排版

本书 Markdown 约定如下：

- 行内数学表达式使用反引号包裹，例如 `\mathbf{SH}(S)`、`f_!`、`\mathbb A^1`。
- 多行公式使用 `$$ ... $$`。
- 不在公式前额外写 “Math:” 或 “公式：” 标识。
- 同一对象第一次出现时说明所在范畴；后文可用固定符号。
- 英文术语保留在必要位置，例如 presentable、localization、six operations、framed correspondences；术语首次出现时尽量配中文说明。

## 5. 交叉引用

正文引用采用中文环境名加编号：

```text
由命题 5.14 ...
见定义 6.1 ...
由外部输入定理 15.4 ...
```

资料源引用采用账本式追踪：

- 一般资料源见 [SOURCES.md](SOURCES.md)。
- 本书使用的定理见 [THEOREM_LEDGER.md](THEOREM_LEDGER.md)。
- 精确 locator 见 [REFERENCE_LOCATOR_LEDGER.md](REFERENCE_LOCATOR_LEDGER.md)。
- 已完成的 P0 定位批次见 [P0_REFERENCE_LOCATORS_BATCH_1.md](P0_REFERENCE_LOCATORS_BATCH_1.md)。

正文不强制为每个“见命题 n.m”做 Markdown anchor 链接；出版排版阶段可由脚本生成锚点或 LaTeX labels。

## 6. 证明格式

证明段落统一写作：

```text
**证明.** 证明正文。`\square`
```

规则：

- 不用“显然”“容易看出”跳过关键步骤。
- 若结论是外部深定理，不写伪证明；写“外部输入定理”并登记资料源。
- 若命题只是外部输入的形式后果，应先陈述外部输入，再给出形式推导证明。

## 7. 索引格式

总索引见 [INDEX.md](INDEX.md)。索引条目采用“术语：章节或附录”的形式，并按主题而非字母严格排序。原因是本书混合中文、英文和数学符号，主题索引比字母索引更适合当前 Markdown 草稿。

## 8. 当前编号状态

当前正文和附录已经基本满足本规范：

- 正文使用章内编号。
- 附录使用字母编号。
- 练习编号与解答编号可一一对应。
- 尚未做的是自动化 anchor、全书 LaTeX label 和最终索引页码；这些属于出版排版层。
