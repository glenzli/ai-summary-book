# 第二十四章：2025-2026 研究边界、开放问题与资料源定位

## 本章目标

本章给出截至 2026-07-08 的研究边界图谱。目标不是宣传最新结果，而是为本书后续扩写建立严格边界：哪些结果可以作为基础外部输入，哪些仍应列为研究边界，哪些开放问题只应作为问题陈述。

## 依赖前置知识

需要前二十三章的全部口径，特别是六操作、stacky motivic homotopy、realization、norms、framed transfers、fundamental classes 和 universal formalisms。

## 24.1 已纳入 P0 的基础外部输入

**定义 24.1.** P0 外部输入是本书正文主链依赖的定理，出版前必须补充精确 theorem locator。

**P0 包括但不限于：**

- Morel-Voevodsky `A1`-homotopy theory 和 homotopy purity。
- Ayoub/Cisinski-Deglise motivic 六操作。
- Drew-Gallauer universal six-functor formalism。
- Voevodsky/Spitzweck `H\mathbb Z`。
- Röndigs-Ostvær/Cisinski-Deglise `DM` 与 `H\mathbb Z`-modules 比较。
- Röndigs-Spitzweck-Ostvær `KGL` strict ring models。
- Panin-Pimenov-Röndigs `MGL` universality。
- Hoyois Hopkins-Morel 型比较。
- Voevodsky zero slice。
- Elmanto-Hoyois-Khan-Sosnilo-Yakerson framed recognition。
- Bachmann-Hoyois norm functors。
- Deglise-Jin-Khan fundamental classes。
- Morel `GW(k)`/Milnor-Witt computations。

**命题 24.2.** P0 输入不等于无假设定理。

**证明.** P0 表示正文依赖强度，而不是定理适用范围。每个 P0 输入仍有基、
系数、有限性、正则性或特征假设。前三批 locator 已把基础、六操作和第
09-18 章的这些假设写入定理账本；后续扩展条目仍须遵守同一规则。
`\square`

## 24.2 研究边界

**研究边界 24.3（Perfect schemes）.** 2025 年 perfect schemes 上的 motivic homotopy theory 提供 positive characteristic 的新模型和六操作。当前仅作为研究边界。

**研究边界 24.4（Pullback formalisms）.** 2025 年 pullback formalism 的几何判据增强了 universal property 的表达，尤其关注 morphisms 与六操作的相容。

**研究边界 24.5（Complex analytic stacks）.** 2026 年 complex analytic stacks 的 localization theorem 支持 analytic motivic homotopy 和 six-operation-compatible realization。

**研究边界 24.6（Derived/log/cobordism refinements）.** Derived algebraic cobordism、log motivic homotopy 和非 `\mathbb A^1`-不变理论提供边界扩展，但不属于本书基础链。

## 24.3 开放问题目录

**问题 24.7.** 在尽可能广的 algebraic stacks 类别上，`\mathbf{SH}` 的六操作形式主义能否以统一的 universal property 描述，并与所有标准 realization 相容？

**问题 24.8.** Norms、framed transfers、Milnor-Witt transfers 和 finite correspondences 之间是否存在统一的 higher categorical transfer calculus？

**问题 24.9.** Slice filtration 在一般基和 stacky/equivariant 情形下的收敛性、slices 描述和计算工具能否系统化？

**问题 24.10.** Motivic stable homotopy groups 的计算能否在更多非代数闭域、实闭域和 arithmetic bases 上形成可操作表格？

**问题 24.11.** Quadratic enumerative invariants 与 derived/log intersection theory 是否能在一个统一 bivariant motivic formalism 中闭合？

## 24.4 标准教材收口标准

**定义 24.12.** 本书称为达到标准教材收口状态，若满足以下条件：

1. 主体概念覆盖闭合；
2. 每章有定义、核心命题、证明或外部输入、例子、失败模式或边界说明、练习；
3. 基础工具在附录中可追溯；
4. 外部输入分 P0/P1/R 并有 locator；
5. 术语、符号和交叉引用统一；
6. 关键计算不只停留在口号层。

**命题 24.13.** 截至当前版本，本书满足 1、2、3、6 的草稿级要求；条件 4
在基础、六操作及第 09-18 章 P0 主线已经达到，条件 5 和第 19-23 章的全书
出版级定位仍未完成。

**证明.** 主体 24 章与附录 A-H 已覆盖核心概念、形式工具和低阶计算；每章
均含定义、命题/定理和练习。`P0_REFERENCE_LOCATORS_BATCH_1.md` 至 batch 3
已为所述主线给出定理/章节号、假设与稳定链接。另一方面，equivariant、
stacky 和 realization 队列仍有 P0 条目未精确定位，交叉引用也未统一为稳定
label。因此只能得到所述分层结论。`\square`

## 24.5 资料源定位计划

**定义 24.14.** Theorem locator 是外部输入在资料源中的精确定位，包括版本、章节、定理号、页码、假设和本书使用方式。

**计划 24.15.** 前三批已经完成 Morel--Voevodsky/Drew--Gallauer、基础
六操作，以及 `H\mathbb Z`、`DM`、`KGL`、`MGL`、slice、finite/framed
transfers、fundamental classes、norms、Milnor--Witt 主线。后续出版级校订
只按以下顺序继续：

1. Equivariant 与 stacky P0 扩展；log/perfect/analytic 保留既定 P1/R 边界。
2. Realization functors 及其 six-operation compatibility、conservativity 和
   comparison theorems。
3. 全书 page locator 与自动化 labels 终校。

**命题 24.16.** 没有 theorem locator 的外部输入不得进入最终出版态的无条件正文。

**证明.** 外部输入的严格性来自可追溯性。若没有 locator，读者无法核查定理假设、版本、证明依赖和结论强度。把无 locator 的结果写成正文定理会破坏本书“证明或外部输入可追溯”的标准。因此出版态必须补 locator。`\square`

## 24.6 当前闭合判断

**命题 24.17.** 截至本章，本书已达到“正式教材范围的内部闭合草稿”，但未达到“出版级数学闭合”。

**证明.** 第一至第二十四章已经覆盖 motivic spaces、stable `\mathbf{SH}`、
六操作、purity、duality、`H\mathbb Z`、`DM`、`KGL`、`MGL`、slices、
transfers、framed recognition、fundamental classes、norms、Milnor--Witt、
equivariant/stacky/log/perfect/realization/universal formalisms 和研究边界。
基础至第十八章的 P0 主线已有精确 locator；但第 19-23 章仍有扩展/realization
P0 队列，且自动交叉引用和最终排版尚未闭合。因此概念与主线教学闭合，
全书出版闭合仍未达到。`\square`

## 24.7 本章小结

Motivic homotopy and six functors 的现代边界集中在
stacky/analytic/perfect/log 扩展、transfer calculus、slice computations 和
quadratic refinements。基础至第十八章 P0 主线已完成定位；下一阶段应集中于
extensions/realization locator、交叉引用和出版终校。

## 练习

**练习 24.1.** 解释 P0 外部输入和研究边界的区别。

**练习 24.2.** 为任一外部输入写出 theorem locator 应包含的信息。

**练习 24.3.** 说明为什么 first-version conceptual closure 不等于 publication closure。

**练习 24.4.** 从问题 24.7-24.11 中选一个，列出它依赖的前三章基础。

**练习 24.5.** 给出一个 realization functor 相关的保守性问题。

**练习 24.6.** 用定义 24.12 检查任意一章是否仍像大纲。
