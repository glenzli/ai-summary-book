# 编号与交叉引用审定

本文档记录编号、链接和索引的机械审定。每次结构性扩写后必须更新。

## 审定项目

1. Markdown 链接必须指向存在的本地文件或明确外部资料源。
2. [THEOREM_INDEX.md](THEOREM_INDEX.md) 中列出的编号必须能在正文中找到。
3. 归一化相关公式必须与 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 一致。
4. 新增章节若包含非平凡命题，必须同步写入定理索引。
5. 新增习题若属主线核心题，必须同步写入 [EXERCISE_COVERAGE.md](EXERCISE_COVERAGE.md) 或 [SOLUTIONS.md](SOLUTIONS.md)。

## 本轮审定记录

最终内容收口检查采用允许 `5.11A`、`A.3` 等插入式编号的规则。机械检查结果：

- Markdown 本地链接经 OET 严格审计，缺失 0 个。
- 定理索引编号：178 个，缺失 0 个。
- 正文定义、命题、例子等编号标签：370 个，重复 0 个。
- 正文习题：70 题；核心习题解答：70 题；题号重复 0 题。
- 正文章节叙事检查：第 0 至 20 章均在 H1 后有自然导言、在练习前有问题导向的
  自然收束，并已移除“本章目标”“依赖前置知识”“主线”“本章小结”等固定标题。
- `git diff --check -- books/string-theory`：通过。
