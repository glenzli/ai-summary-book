---
name: prismatic-p-adic-hodge-theory-textbook
description: Use when writing, revising, or checking the rigorous Chinese textbook on prismatic cohomology and p-adic Hodge theory in books/prismatic-p-adic-hodge-theory. Requires primary-source grounding, formal definitions before intuition, explicit derived-completion and torsion hypotheses, theorem-proof structure, and clear separation between classical p-adic Hodge theory, BMS integral theory, prismatic cohomology, prismatic F-crystals, and current research frontiers.
---

# Prismatic / p-adic Hodge Theory 教材写作技能

本技能约束 `books/prismatic-p-adic-hodge-theory/` 中《Prismatic / p-adic Hodge Theory》教材的写作、扩写、校订和审稿。

## 基本原则

- 使用中文叙述；标准英文术语第一次出现时括注，例如“棱柱（prism）”“有界棱柱（bounded prism）”。
- 定义必须先于直觉；例子必须验证定义中的结构映射、拓扑完备性、Cartier divisor 条件和 Frobenius 条件。
- 非平凡命题必须给出证明、证明草图，或明确标注为“外部输入定理”，并在 `SOURCES.md` 中记录来源。
- 不使用“显然”“容易看出”等词跳过关键步骤；短证明也要写出使用的定义、完备性、正合性、基变换或滤过论证。
- 所有环、拓扑环、导出完备化、张量积、site、sheaf、derived category、filtered object 和 Frobenius-semilinear map 必须标明所在范畴或模型。
- 区分经典 Fontaine 周期环理论、Faltings/Tsuji 比较定理、BMS 的 $A_{\inf}$-上同调、Bhatt-Scholze 的 prismatic cohomology、prismatic $F$-crystals 和 2025-2026 研究边界；不得把不同层级的等式、同构、拟同构、filtered isomorphism 和范畴等价混用。
- 不复制资料原文；所有正文使用本书自己的中文重写。

## 本书口径

- 固定素数 $p$。除非显式说明，所有含 $p$-完备性的环均为交换环，所有完备化均按导出 $p$-完备或导出 $(p,I)$-完备解释。
- 基础章节先处理 $\delta$-环、Frobenius lift、Witt vectors、完备化与 perfectoid 背景，再定义 prism 和 prismatic site。
- prism 默认指 Bhatt-Scholze 意义下的有界 prism，若使用 derived prism、absolute prism、perfect prism、oriented prism 或 $q$-crystalline prism，必须显式说明。
- prismatic cohomology 默认记作 $R\Gamma_\Delta(X/A)$，其中 $(A,I)$ 是基 prism，$X$ 是 $p$-adic formal scheme over $A/I$。无基版本记作 $R\Gamma_\Delta(X)$ 并另行说明。
- 比较定理必须列出光滑性、properness、有界性、完备性、torsion 和 base prism 假设。没有这些假设不得把比较态射写成同构。
- 对 classical $p$-adic Hodge theory 的内容只在明确的 Fontaine period ring 语境中使用；不要把 $B_{\mathrm{dR}}$、$B_{\mathrm{cris}}$、$A_{\inf}$、Breuil-Kisin prism 和 absolute prismatic cohomology 的对象混成一个未标注对象。
- 研究前沿只收录已联网核查的一手资料；除非完成独立验证，不把 2025-2026 预印本中的新结果写成正文定理。

## 资料源规则

- 基础定义和主定理优先使用 Fontaine、Illusie、Berthelot-Ogus、Faltings、Tsuji、Brinon-Conrad、Berger、Kedlaya-Liu、Scholze、Bhatt-Morrow-Scholze、Bhatt-Scholze、Bhatt-Lurie 等一手资料。
- 涉及 prismatic cohomology、prismatization、prismatic $F$-crystals、$F$-gauges、syntomic operations 或 2025-2026 预印本时，必须联网核查具体版本和发布日期。
- Wikipedia、百科页面和博客只能用于发现线索，不得作为核心定义或定理的最终依据。
- `SOURCES.md` 必须标明资料用途：基础定义、核心定理、外部输入、研究边界或历史说明。

## 写作格式

- 文件名使用两位编号，例如 `01_delta_rings_witt_vectors_and_perfectoid_background.md`。
- 每章开头列出“本章目标”和“依赖前置知识”。
- 定义、约定、例子、命题、引理、定理、推论、警告、练习使用“**定义 1.2.**”格式。
- 每章末尾必须包含“本章小结”和“练习”。
- 公式使用 Markdown/LaTeX；长交换图优先用明确的态射链或 `tikzcd` 风格代码块描述。
- 全书性符号必须先登记到 `NOTATION.md`。

## 严谨性检查

扩写或修改章节后逐项检查：

- 是否固定了素数 $p$、基域、完备性和小性口径。
- $\delta$-环定义是否给出两条恒等式，且 $\phi(x)=x^p+p\delta(x)$ 是否确实是环同态。
- prism 定义是否检查 $I$ 是 Cartier divisor、$A$ 是导出 $(p,I)$-完备、且 $p\in I+\phi(I)A$。
- boundedness 是否明确写成 $A/I$ 的 $p^\infty$-torsion 有界。
- prismatic site 的对象、态射、覆盖和结构层是否都在固定 base prism 上说明。
- Hodge-Tate、de Rham、crystalline、etale、syntomic 比较是否列出全部假设和 twist/filtration convention。
- $A_{\inf}$、$\theta$、$\xi$、$\mu$、Nygaard filtration 和 Breuil-Kisin prism 的符号是否与 `NOTATION.md` 一致。
- Galois 表示章节是否区分 lattice、$\mathbf Q_p$-representation、crystalline representation、Breuil-Kisin module、prismatic $F$-crystal 和 filtered $\varphi$-module。
- 前沿章节是否把 2025-2026 结果标为“研究边界/外部输入”，并给出版本日期。

