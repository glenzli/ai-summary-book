---
name: operad-theory-textbook
description: Use when writing, revising, or checking the rigorous Chinese textbook on operad theory in books/operad-theory. Requires primary-source grounding, formal definitions before intuition, theorem-proof structure, explicit universe and size conventions, notation consistency, and careful separation between classical operad theory, homotopical operads, infinity-operads, and current research frontiers.
---

# Operad Theory 教材写作技能

本技能约束 `books/operad-theory/` 中《Operad Theory》教材的写作、扩写、校订和审稿。

## 基本原则

- 使用中文叙述；标准英文术语第一次出现时括注，例如“对称 operad（symmetric operad）”。
- 定义必须先于直觉；例子必须逐条验证定义中的结构和公理。
- 非平凡命题必须给出完整书内证明，或明确标注为“外部输入定理”并在 `SOURCES.md` 中记录来源；证明路线只能解释外部输入，不计作书内证明。
- 不使用“显然”“容易看出”等词跳过关键步骤；短证明也要写出使用的定义、泛性质、结合律或等变性。
- 所有对象、态射、张量积、商、coinvariants、链复形、模型结构、同伦范畴和 infinity-对象必须标明所在范畴或模型。
- 区分 set-operad、topological operad、dg-operad、colored operad、PROP/properad、dendroidal set、Lurie-style infinity-operad；不得把不同模型中的等式、同构、弱等价和等价混用。
- 不复制资料原文；所有正文使用本书自己的中文重写。

## 本书口径

- 第一部分从有限集上的对称序列与代入乘积开始，优先使用可检查的集合论定义。
- operad 默认表示含零元 arity 的单色对称 operad；若排除零元 arity、去掉对称群作用或允许多色，必须显式说明。
- 基础章节采用固定 Grothendieck universes 处理小性问题；“有限集”指指定宇宙中的有限集。
- 代数结构章节先处理 `Set`、`Mod_R`、`Ch_R` 中的 operad，再进入 Koszul 对偶、bar-cobar 和同伦代数。
- 同伦章节区分严格 operad、模型范畴中的 operad、colored operad、dendroidal model、quasi-category/operadic fibration 模型。
- 研究前沿只收录已联网核查的一手资料；除非完成独立验证，不把 2025-2026 预印本中的新结果写成正文定理。

## 资料源规则

- 优先使用正式教材、专著、作者主页、arXiv 论文和出版商页面：May、Boardman-Vogt、Markl-Shnider-Stasheff、Loday-Vallette、Fresse、Berger-Moerdijk、Cisinski-Moerdijk、Moerdijk-Weiss、Lurie、Hoffbeck-Moerdijk 等。
- 涉及 infinity-operad、dendroidal set、operadic localization、higher algebra 或近期预印本时，必须联网核查具体版本和发布日期。
- Wikipedia、百科页面和博客只能用于发现线索，不得作为核心定义或定理的最终依据。
- `SOURCES.md` 必须标明资料用途：基础定义、核心定理、外部输入、研究边界或历史说明。

## 写作格式

- 文件名使用两位编号，例如 `01_symmetric_sequences_and_operads.md`。
- 每章开头列出“本章目标”和“依赖前置知识”。
- 定义、约定、例子、命题、引理、定理、推论、练习使用“**定义 1.2.**”格式。
- 每章末尾必须包含“本章小结”和“练习”。
- 公式使用 Markdown/LaTeX；树形组合可用明确的有限集分块、树代入或 tikzcd 风格代码块描述。
- 全书性符号必须先登记到 `NOTATION.md`。

## 严谨性检查

扩写或修改章节后逐项检查：

- 是否已经声明集合论宇宙和小性层级。
- arity 是否允许 $0$，单位是否在 arity $1$，以及相关约定是否与 `NOTATION.md` 一致。
- 对称群作用采用左作用还是右作用，是否与代入公理相容。
- 代入乘积中的分块、coinvariants、张量积或余积是否存在于当前范畴。
- operad 代数是否写成到 endomorphism operad 的 morphism，或等价的动作映射，并证明两者等价。
- examples 是否满足单位、结合律和等变性，而不是只说明直觉。
- 模型范畴章节是否区分 cofibrancy、fibrancy、weak equivalence、Quillen equivalence 和派生映射空间。
- infinity-operad 章节是否说明采用 dendroidal set、simplicial operad nerve、Lurie operadic fibration 或其他模型。
- 外部输入定理是否在 `SOURCES.md` 中可追溯。
