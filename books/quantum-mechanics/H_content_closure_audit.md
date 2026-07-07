# 附录 H：教材内容收口审查

## 收口口径

本附录按“教材内容本身”审查本书，而不是按出版排版审查。本书达到内容收口，指以下事项已经固定：

1. 内容范围完整：非相对论量子力学主线、核心模型、现代测量语言和必要边界章均已覆盖。
2. 证明责任完整：书内可证明的核心命题已给出证明；不在书内证明的大型定理均标为外部输入或边界。
3. 引用追踪完整：外部输入在 `THEOREM_DEPENDENCIES.md`、`D_external_theorem_index.md` 和 `CHAPTER_SOURCE_NOTES.md` 中可追溯。
4. 练习闭包完整：章末练习和综合题均有提示手册与答案手册条目。
5. 阅读入口完整：统一索引、压缩术语表、提示手册、答案手册入口和外部输入入口已经接入主目录。

## 内容范围审查

- 第 0--5 章：固定 Hilbert 空间、态、谱、公设和时间演化。
- 第 6--11 章：覆盖方势阱、势垒、谐振子、复合系统、角动量、对称性和 CCR。
- 第 12--16 章：覆盖扰动、变分、WKB、散射、相同粒子和绝热/Berry 相位。
- 第 17--19 章：覆盖密度算子、开放系统、POVM、Kraus 表示、信道、熵和态距离。
- 第 20--23 章：覆盖路径积分、相对论单粒子边界、中心势、氢原子、电磁耦合和规范结构。
- 第 24--27 章：覆盖不确定性、概率流、Ehrenfest、相互作用图像、Dyson 展开、角动量耦合、选择定则和标准精算例题。

## 证明闭包审查

内部完成证明的内容包括：

- Cauchy-Schwarz、射线投影表示、有限维谱定理、正交投影结构。
- Born 规则相位不变性、Luders 更新归一化、有限维共同对角化。
- 有界 Hamiltonian 演化、Heisenberg 方程、有限维谱相位演化。
- 一维匹配条件、流守恒、谐振子升降算符、相干态、Schmidt 分解、偏迹。
- Robertson 不确定性、连续性方程、Ehrenfest 定理。
- 相互作用图像、Dyson 级数和一阶跃迁振幅的有界有限维证明。
- 中心势径向方程、规范协变性、Landau 能级代数、Rabi 振荡等标准计算。

外部输入闭包包括：

- 谱定理、Stone、Kato-Rellich、Friedrichs、Stone-von Neumann、Wigner。
- Sturm-Liouville/Fourier-Hermite 完备性、Coulomb 谱理论、WKB 连接公式。
- 散射渐近完备性、光学定理、partial wave 展开、Trotter-Kato。
- Stinespring/Naimark 一般形式、Lindblad、Uhlmann、Kato 解析扰动、Wigner-Eckart。

## 引用完整性审查

逐章引用入口见 [CHAPTER_SOURCE_NOTES.md](CHAPTER_SOURCE_NOTES.md)。外部输入按标签 `QM-EXT-1` 至 `QM-EXT-20` 在 [THEOREM_DEPENDENCIES.md](THEOREM_DEPENDENCIES.md) 中列出，并在 [D_external_theorem_index.md](D_external_theorem_index.md) 中给出定理陈述。

正文中未证明而作为数学前提使用的结果不得只以普通说明出现；若后续校订发现此类结果，应升级为外部输入定理并同步三个索引文件。

## 阅读排版审查

阅读入口见 [BOOK_INDEX.md](BOOK_INDEX.md)。该索引集中列出正文部分、附录、练习提示、练习答案、综合题答案、术语表、外部输入和资料源。术语索引已压缩为主题化核心项，避免把同一章节的相邻概念拆成过长清单。

## 练习与答案审查

当前校验脚本要求：

- 每个结构化章节和附录含“本章目标”“依赖前置知识”“本章小结”“练习”。
- 所有章末练习在 [HINTS.md](HINTS.md) 中有提示。
- 所有章末练习在 [SOLUTIONS.md](SOLUTIONS.md) 中有答案。
- 所有综合题在 [HINTS.md](HINTS.md) 中有提示。
- 所有综合题在 [COMPREHENSIVE_SOLUTIONS.md](COMPREHENSIVE_SOLUTIONS.md) 中有答案。
- 术语索引无中英文重复项。
- 外部输入编号连续、已定义且在结构化章节中被使用。
- Markdown 文件无行尾空白。
- 禁止遗留验证脚本定义的任何未完成占位标记。

截至本收口稿，校验结果为：

```text
markdown_files=49
structured_chapters=31
term_index_rows=30
external_inputs=20
chapter_and_appendix_exercises=68
chapter_and_appendix_hints=68
comprehensive_exercises=18
comprehensive_hints=18
validation=ok
```

## 收口结论

按“教材内容本身”并加上基础阅读排版标准，本书已经达到内容收口状态。后续工作若继续，应属于以下三类之一：

- 增广型：增加更多例题、习题或专题章，但这会改变范围。
- 形式化型：把更多外部输入定理内化证明，但这会显著扩大泛函分析和数学物理背景。
- 深度出版型：统一 LaTeX、页码索引、正式书目格式和版式。
