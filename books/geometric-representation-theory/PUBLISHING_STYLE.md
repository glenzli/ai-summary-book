# 出版排版与阅读样式

本文件规定《Geometric Representation Theory》进入阅读优化阶段后的统一排版。它不新增数学内容，只约束呈现方式。

## 1. 章节结构

主体章节统一采用以下顺序：

1. 章标题；
2. 本章目标；
3. 依赖前置知识；
4. 编号小节；
5. 本章小结；
6. 练习。

附录可省略“依赖前置知识”，但必须保留“本章目标”和“本章小结”。证明核、审查文件和索引文件可按功能组织。

## 2. 数学块格式

定义、命题、外部输入、警告、边界说明、检查表和练习统一使用：

```text
**定义 3.1.** ...
**命题 3.2.** ...
**证明.** ... $\square$
```

外部输入统一写为：

```text
**外部输入定理 8.9.** ...
```

不得把大型结果写成无来源的“定理”。若只有研究方向说明，使用“边界说明”而不是“定理”。

## 3. 公式和文本

1. 行内数学只用于短符号，如 $G/B$、$\mathcal O$、$\operatorname{IC}_w$。
2. 长公式使用独立 display math。
3. 一个 display math 前后保留空行。
4. 不使用“显然”“易知”跳过数学理由。
5. 证明末尾统一用 `$\square$`。

## 4. 术语压缩

正文保留少量稳定英文术语，避免在中文里反复切换多个译名：

| 统一术语 | 不再混用 |
| --- | --- |
| perverse sheaf | 反常层、perverse 层 |
| D-module | D 模、微分模 |
| sheaf | 层、sheaf 混写时只在普通中文语境用“层” |
| affine Grassmannian | 仿射 Grassmannian、仿射格拉斯曼 |
| affine flag variety | 仿射旗簇、affine flag |
| category $\mathcal O$ | O 范畴、BGG O |
| convolution | 卷积乘法、convolution product |
| external input | 外部输入 |
| locator | theorem locator、引用定位 |

首次出现时可写中文解释；后文使用统一术语。

## 5. 习题答案

正文保留练习题干，不在每章末尾插入答案，以免干扰主线。答案集中放在 [EXERCISE_SOLUTIONS.md](EXERCISE_SOLUTIONS.md)，采用“答案提示”而不是完整展开证明。需要长证明的习题只给关键步骤和引用位置。

## 6. 索引

术语索引集中放在 [INDEX.md](INDEX.md)。符号索引集中放在 [SYMBOL_INDEX.md](SYMBOL_INDEX.md)，例子与计算索引集中放在 [EXAMPLE_INDEX.md](EXAMPLE_INDEX.md)。索引条目指向最先定义处、核心使用处和技术附录。正文不维护手写反向索引，避免多处漂移。

## 7. 引用

正文只写必要来源提示。外部输入的完整来源统一放在：

1. [SOURCES.md](SOURCES.md)；
2. [D_source_theorem_index.md](D_source_theorem_index.md)；
3. [P0_REFERENCE_LOCATORS_BATCH_1.md](P0_REFERENCE_LOCATORS_BATCH_1.md)。

出版终稿再补页码级 locator。
