# 内部闭合矩阵

核查日期：2026-07-11

本矩阵判断本书作为正式教材的内部完整性。结论分三层：

- `概念闭合`：主题已覆盖，读者能看到理论版图。
- `内部闭合`：定义、符号、基础范畴论和形式推导在书内可追溯。
- `教学闭合`：章节密度、证明/外部输入分界、例子和练习足以支撑完整阅读与教学。
- `出版闭合`：外部输入 locator、交叉引用、编号、习题、证明细节和排版全部完成。

## 总体判断

| 层级 | 当前状态 | 说明 |
| --- | --- | --- |
| 概念闭合 | 达到 | 24 章覆盖 motivic homotopy and six functors 主体方向 |
| 内部闭合 | 初步达到 | 附录 A-H 补齐大小、站点、稳定化、mate calculus、代数几何背景、三角/∞ 翻译、源定理索引和低阶计算 |
| 教学闭合 | 达到 | 主体章不再是大纲态；定义、证明、外部输入、边界和练习已经形成可读教材链条 |
| 出版闭合 | 部分达到 | 核心范畴论、稳定化、六操作、purity 及第 09-18 章教学主线 P0 locator 已由三批账本闭合；扩展/realization、自动化交叉引用、最终排版和长篇详解仍待完成 |

## 主体章节闭合

| 模块 | 文件 | 内部状态 | 剩余工作 |
| --- | --- | --- | --- |
| Motivic spaces | 00-02 | 内部闭合 | 补 Nisnevich cd-structure locator |
| Stabilization | 03, C | 内部闭合；反演/谱模型/稳定性已分层 | Robalo/Hoyois locator 已补；模型范畴历史比较可继续扩写 |
| Abstract six functors | 04, D | 内部闭合；方差和 ordinary/exceptional maps 已类型检查 | 增加相干图编号 |
| Motivic six operations | 05-08 | Hoyois trivial-group package 已定位；形式推导闭合 | 若并用 Ayoub/Cisinski-Deglise 版本，逐项补不同假设 locator |
| Cohomology and motives | 09-13 | P0 教学主线及比较定理 locator 已闭合；三角/稳定 infinity 层级已分开 | P1 仅含 etale/Bloch-Kato、rational Chern character、一般基 `DM` 与 Adams 型推广 |
| Transfers and refinements | 14-18 | P0 finite transfers、framed recognition、fundamental classes、norms、Morel locator 已闭合 | P1 仅含 Hilbert/framed 全相容、Tambara refinements、Chow-Witt motives 与 quadratic enumerative 公式 |
| Extensions and realization | 19-24 | 概念闭合，研究边界分层完成 | stack/log/perfect/analytic/realization locator |

## 附录闭合

| 附录 | 内容 | 当前状态 |
| --- | --- | --- |
| A | 宇宙、小骨架、presentability、accessible localization | OET 闭合；深基础输入已改为 HTT located 外部定理 |
| B | Grothendieck topology、Nisnevich squares、cd-structures | 内部闭合，cd-structure 生成定理外部输入 |
| C | Pointed categories、smash product、T-stabilization | 对象反演、3-symmetry、谱模型与稳定性已分层；外部输入 located |
| D | Mate calculus、Beck-Chevalley、projection formula | 内部闭合 |
| E | smooth/etale/proper/open/closed/lci 背景 | 内部闭合 |
| F | stable infinity 与 triangulated 翻译 | OET 闭合；triangulated shadow 外部基础定理 located |
| G | 资料源定理索引 | 第 09-18 章 P0 已闭合；扩展/realization 队列继续保留 |
| H | `T`、localization、Thom、transfer/norm 基本计算 | 内部闭合 |

## 出版增强项

1. 为第 19-23 章 extensions/realization 的剩余 P0 条目补精确 theorem locator；第 09-18 章不再列入该队列。
2. 把正文中的“外部输入定理”逐条链接到 `THEOREM_LEDGER.md` 标签。
3. 将 `TYPESETTING_AND_NUMBERING.md` 的规范转成自动化 anchors 或 LaTeX labels。
4. 继续增加高级计算例子：Gysin excess、slice spectral sequence、Chern character、quadratic Euler characteristic。
5. 将 `EXERCISE_SOLUTIONS.md` 的解答要点扩展为长篇详解。
6. 做术语终校：`\mathbb A^1` vs `A1`，`infinity-category` vs `∞-范畴`，`T` vs `\mathbb P^1`-stabilization。

## 当前可接受结论

本书现在可以称为“完整教材可读版”和“学术教学闭合草稿”。第 09-18 章
主线外部输入已有定理/章节级 locator 和稳定链接；其高级研究边界已明确降为
P1。全书仍不能称为 camera-ready 或出版终稿，因为 extensions/realization、
自动化交叉引用、长篇解答和最终排版尚未闭合。
