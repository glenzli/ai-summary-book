---
name: homotopy-type-theory-textbook
description: Use when writing, revising, or checking the Chinese rigorous textbook on Homotopy Type Theory and Univalent Foundations in this repository. Requires primary-source grounding, explicit type-theoretic rules, theorem-proof structure, source traceability, universe bookkeeping, and strict separation between book-proved results, external metatheorems, and research boundaries.
---

# HoTT 教材写作技能

本技能约束 `books/homotopy-type-theory/` 中《同伦类型论与单值基础》教材的写作、扩写、校订和审稿。

## 基本原则

- 以中文叙述，但第一次出现标准术语时保留英文括注，例如“单值性（univalence）”。
- 每章必须先列出“本章目标”和“依赖前置知识”，再进入定义、命题、定理、证明、例子和练习。
- 正文默认读者熟悉基础抽象代数、范畴语言和普通数学证明，但不默认读者熟悉依赖类型论。
- 不用拓扑直觉替代类型论规则；所有同伦解释必须落回构造、归纳原理或已声明的外部模型定理。
- 不写“显然”“容易看出”来跳过关键步骤；若证明只用一次归纳原理，也要说明归纳对象和归纳后的目标。
- 对 judgmental equality、propositional equality、path equality 使用不同符号并持续区分。
- 不把尚未引入的函数外延性、单值性、高阶归纳类型或截断性当作基础规则使用。
- 不复制来源原文；所有内容用本书自己的中文重写，并在 `SOURCES.md` 中记录资料源。

## 资料源规则

- 优先使用一手或正式数学资料：HoTT Book、Rijke 教材、Voevodsky 资料、arXiv 论文、正式出版论文和经典教材。
- 涉及近期预印本、模型论结果或对象语言状态时必须核查来源，并在 `SOURCES.md` 中写明核查日期。
- 不以 Wikipedia、博客或二手科普作为核心定义来源；可用于发现线索，但不得作为主依据。
- 正文若使用模型论一致性、计算解释或 cubical metatheory，必须标注为外部输入定理，除非本书已给出完整元理论证明。
- 正文若使用 directed/simplicial type theory、cohesive/modal HoTT 或其他对象语言扩展，必须明确其新增判断和规则，不得把扩展语言中的 directed hom 或模态规则混入基础 HoTT 的 identity type 证明。

## 证明与验证标签

每个非平凡命题、定理或构造至少带有以下一种状态：

- **书内证明。** 本书已从前文定义和引理推出。
- **证明说明。** 本书给出严格证明路线，但压缩了长篇标准细节；必须说明未逐行展开内容属于哪一类标准理论。
- **外部输入。** 本书不证明，只引用一手来源；必须说明后续依赖风险。
- **研究边界。** 近期研究、预印本或仍在发展的方向；不得作为基础定理无条件使用。

## 写作格式

- 文件名使用两位编号，例如 `01_dependent_type_theory_and_judgments.md`。
- 定义使用“**定义 1.2.**”格式；命题、定理、引理、例子、练习同理。
- 公式使用 Markdown/LaTeX；上下文、宇宙层级和依赖变量必须在公式附近可见。
- 每章末尾必须有“本章小结”和“练习”。
- 若某章引入新符号，必须同步检查 `NOTATION.md`。
- 若某章引用新核心来源，必须同步检查 `SOURCES.md`。

## 数学严谨性检查

扩写或修改章节后，逐项检查：

- 当前语境 $\Gamma$、类型所在宇宙、自由变量和替换是否明确。
- 规则是 judgmental 规则、归纳原理、命题等价还是外部公理，是否已经说明。
- 每次使用 path induction、dependent path induction 或 higher inductive induction principle 时，归纳族是否写清。
- 每次使用 $\Pi$、$\Sigma$、identity type、univalence、HIT 或 truncation 时，形成规则、引入规则、消去规则和计算规则是否已在前文建立或引用。
- 函数外延性、命题外延性、唯一选择、排中律、选择公理等原则是否被显式声明，而不是偷用。
- universe polymorphism、cumulativity、resizing、impredicativity 是否被回避得当。
- 同伦层级命题是否说明是 proposition、set、groupoid 还是更高类型。
- 例子是否满足前面定义，而不是只满足拓扑直觉。

## 本书口径

- 基础卷采用强度适中的 intensional Martin-Lof type theory：依赖函数、依赖对、恒等类型、自然数、空类型、单位类型、和类型与宇宙。
- 默认不把函数外延性、单值性、高阶归纳类型、截断和商类型作为原始规则；它们在专章引入，并标注公理化、cubical 计算解释或模型论来源。
- 默认采用层级宇宙 $\mathcal U_0,\mathcal U_1,\ldots$；不默认 resizing。
- 本书按文本出版口径收口，不以书稿外材料作为封稿条件。

## 收口模式规则

本书已进入收口模式。后续修改默认不再新增横向方向，而应服务于以下目标：

- 关闭 `K_remaining_obligations.md` 中已有证明义务。
- 把“证明说明”降为“书内证明”，或明确“外部输入”“研究边界”。
- 按 `DEPENDENCY_LAYERS.md` 检查低层证明是否误用高层规则。
- 按 `CLOSURE_SCOPE.md` 的封稿门槛检查链接、符号、来源、证明状态和公理使用。
- 新增任何章节、附录或研究方向前，必须说明它关闭了哪条已登记义务；否则不得新增。
- 高级接口保留为接口即可，不要求为出版候选继续横向扩张。
