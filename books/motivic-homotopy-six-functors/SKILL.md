---
name: motivic-homotopy-six-functors-textbook
description: Use when writing, revising, or checking the rigorous Chinese textbook on motivic homotopy theory and six functor formalisms in books/motivic-homotopy-six-functors. Requires primary-source grounding, explicit base-scheme hypotheses, infinity-categorical and model-categorical distinctions, theorem-proof structure, external-input tracking, and careful separation between stable motivic homotopy over schemes, equivariant/stacky variants, motivic sheaves, and current research frontiers.
---

# Motivic Homotopy and Six Functors 教材写作技能

本技能约束 `books/motivic-homotopy-six-functors/` 中《Motivic Homotopy and Six Functors》教材的写作、扩写、校订和审稿。

## 基本原则

- 使用中文叙述；标准英文术语第一次出现时括注，例如“动机同伦论（motivic homotopy theory）”。
- 定义先于直觉；非平凡命题必须给出证明、证明草图，或明确标注为“外部输入定理”。
- 任何基概形、站点、拓扑、局部化、稳定化、谱对象、六操作、纯性、定向、转移、范畴等价和保守性断言都必须写明假设。
- 不把同伦范畴、模型范畴、稳定 presentable infinity-范畴和三角范畴中的等式混用；“等价”“同构”“弱等价”“Quillen 等价”“伴随等价”必须按所在语境使用。
- 不使用“显然”“容易看出”等词跳过关键步骤；短证明也要列明使用的定义、泛性质、伴随、Yoneda、局部化或 Beck-Chevalley 构造。
- 所有外部深定理必须在 `SOURCES.md` 和 `THEOREM_LEDGER.md` 中可追溯，并注明用途。
- 不复制资料原文；所有正文使用本书自己的中文重写。

## 本书口径

- 默认基对象为固定宇宙中的有限维 Noetherian 概形；若采用 qcqs 概形、代数栈、解析栈、log 概形或 perfect schemes，必须显式标注为扩展口径。
- 默认光滑站点为 `Sm_S` 的小骨架，拓扑为 Nisnevich topology；其他拓扑如 etale、cdh、rh、v、h、Zariski 只能在明确章节引入。
- 非稳定 motivic spaces 写作 `\mathbf H(S)`，定义为 Nisnevich space-valued sheaves 的 `\mathbb A^1`-局部化。
- 稳定 motivic homotopy category 写作 `\mathbf{SH}(S)`，优先用 presentable symmetric monoidal infinity-category 口径；模型范畴构造作为历史和比较模型处理。
- `\mathbf{SH}(S)` 中的六操作形式主义作为外部输入定理处理；本书内部证明其形式后果，例如基变换态射的构造、投影公式的形式推论、局部化三角的 recollement 后果。
- 研究前沿只收录已联网核查的一手资料；除非完成独立验证，不把 2025-2026 预印本中的新结果写成无条件正文定理。

## 资料源规则

- 基础定义优先使用 Morel-Voevodsky、Voevodsky、Morel、Jardine、Ayoub、Cisinski-Deglise、Hoyois、Drew-Gallauer、Dugger-Isaksen、Robalo 等一手资料。
- 六操作、纯性、基本类、bivariant theory、规范/范数、framed transfers、stacky/equivariant/log/perfect/analytic 扩展必须使用论文、专著或作者发布版本核查。
- Wikipedia、nLab、讲义和博客只能用于发现线索，不得作为核心定义或核心定理最终依据。
- `SOURCES.md` 必须标明资料用途：基础定义、核心外部输入、形式主义、计算样例、研究边界或历史说明。

## 写作格式

- 文件名使用两位编号，例如 `01_sites_nisnevich_and_a1_localization.md`。
- 每章开头列出“本章目标”和“依赖前置知识”。
- 定义、约定、例子、命题、引理、定理、推论、练习使用“**定义 1.2.**”格式。
- 每章末尾必须包含“本章小结”和“练习”。
- 全书性符号必须登记到 `NOTATION.md`。
- 外部输入定理必须在正文中保留标签，并在 `THEOREM_LEDGER.md` 中登记依赖。

## 严谨性检查

扩写或修改章节后逐项检查：

- 是否声明基概形类别、大小宇宙、小骨架和有限性假设。
- 是否区分 presheaves、Nisnevich sheaves、hypercomplete sheaves、`\mathbb A^1`-local objects 和稳定谱。
- `\mathbb A^1`-局部化是否写成 accessible localization，而不是未经证明的商范畴。
- Tate sphere、`\mathbb G_m`、`\mathbb P^1`、`S^{p,q}` 的基点和 suspension 坐标是否一致。
- `\mathbf{SH}(-)` 的 pullback 是否为强对称幺半，pushforward 是否来自伴随函子定理，`f_!` 与 `f^!` 是否作为额外几何定理处理。
- 基变换、投影公式、proper compatibility、局部化、纯性、绝对纯性和 ambidexterity 是否区分为形式后果或外部输入。
- realization、comparison、conservativity、slice filtration、framed recognition、normed spectra 等是否写明额外假设。
- 近期研究是否在 `FRONTIER_SOURCE_AUDIT_2026_07_08.md` 中记录版本和边界。
