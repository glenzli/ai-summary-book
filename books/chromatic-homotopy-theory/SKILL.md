---
name: chromatic-homotopy-theory-textbook
description: Use when writing, revising, or checking the rigorous Chinese textbook on chromatic homotopy theory in books/chromatic-homotopy-theory. Requires fixed prime and universe conventions, precise separation between spectra, finite spectra, Bousfield localization, Morava K/E theories, chromatic and telescopic localizations, theorem ledgers for Hopkins-Smith/Ravenel/Goerss-Hopkins-Miller inputs, and conservative handling of 2023-2026 redshift, telescope, semiadditivity, and K(n)-local frontier results.
---

# Chromatic Homotopy Theory 教材写作技能

本技能约束 `books/chromatic-homotopy-theory/` 中《Chromatic Homotopy Theory》教材的写作、扩写、校订和审稿。

## 基本原则

- 使用中文叙述；标准英文术语第一次出现时括注，例如“Bousfield 局部化（Bousfield localization）”。
- 定义必须先于直觉；例子必须逐项检查对象、结构映射、同伦不变量和局部化条件。
- 非平凡命题必须给出完整书内证明，或明确标注为“外部输入定理”；证明路线只解释外部输入，不计作书内证明，且外部输入必须能在 `SOURCES.md` 和 `THEOREM_LEDGER.md` 中定位。
- 不使用“显然”“容易看出”等词跳过关键步骤；短证明也要写出使用的泛性质、cofiber/fiber 序列、局部化定义或谱序列收敛条件。
- 所有谱、环谱、模谱、同伦群、局部化函子、厚子范畴、形式群、Hopf algebroid、Morava stabilizer group 和连续群上同调必须标明所在范畴和固定素数。
- 严格区分 $K(n)$-局部、$E(n)$-局部、finite/telescopic 局部、$T(n)$-局部、$p$-局部、$p$-完备和有理化；不得把等价、Bousfield 等价、同伦等价、Quillen 等价和 infinity-范畴等价混用。
- 不复制资料原文；所有正文使用本书自己的中文重写。

## 本书口径

- 全书固定一个素数 $p$，除非明确声明全素数同时讨论。
- 默认工作在稳定 infinity-范畴 $\mathbf{Sp}$ 及其 $p$-局部子范畴 $\mathbf{Sp}_{(p)}$。模型范畴表述只作为计算模型。
- “有限谱”默认指 $\mathbf{Sp}_{(p)}$ 中由球谱经过有限次 cofiber、悬挂、脱悬挂和 retract 生成的 compact 对象。
- $K(0)$ 记为有理同调理论 $H\mathbb Q$；$K(n)$ 对 $n\ge 1$ 表示第 $n$ 个 Morava K-theory，系数为 $\mathbb F_p[v_n^{\pm 1}]$，$|v_n|=2(p^n-1)$。
- $E(n)$ 表示 Johnson-Wilson theory，$L_n=L_{E(n)}$；$M_nX=\operatorname{fib}(L_nX\to L_{n-1}X)$。
- $E_n$ 表示高度 $n$ 的 Morava E-theory/Lubin-Tate theory。其 $\mathbb E_\infty$ 结构、Morava stabilizer group 作用和 descent spectral sequence 作为 Goerss-Hopkins-Miller/Devinatz-Hopkins 型外部输入处理。
- telescope conjecture 已按 2023 之后的状态处理：高度至少 $2$ 时 telescopic 和 chromatic 局部化不能再默认相同。早期教材中的等同写法必须改写为历史陈述或假设性陈述。
- 2023-2026 前沿结果只在联网核查后进入研究边界；除非完成 theorem locator 和假设翻译，不进入基础证明链。

## 资料源规则

- 基础定义优先使用 Adams、Ravenel、Hovey-Strickland、Hovey-Palmieri-Strickland、Barthel-Beaudry survey、Lurie 讲义等。
- nilpotence、periodicity、thick subcategory 和 chromatic convergence 作为 Hopkins-Smith、Devinatz-Hopkins-Smith、Hopkins-Ravenel/Ravenel 体系的外部输入，不在正文重证。
- Morava E-theory 的高度、Lubin-Tate 变形和 $\mathbb E_\infty$ 精化优先使用 Goerss-Hopkins-Miller、Devinatz-Hopkins、Rognes 和 Lurie/SAG 体系。
- redshift、telescope 反例、higher semiadditivity、rational $K(n)$-local sphere、syntomic/K-theory of BP<n> 等近年结果必须记录 arXiv 编号、版本日期、用途和是否可作为定理输入。
- Wikipedia、百科页面和博客只能用于发现线索，不得作为核心定义或定理的最终依据。

## 写作格式

- 文件名使用两位编号，例如 `01_stable_spectra_localization_and_bousfield_classes.md`。
- 每章开头列出“本章目标”和“依赖前置知识”。
- 定义、约定、例子、命题、引理、定理、推论、练习使用“**定义 1.2.**”格式。
- 每章末尾必须包含“本章小结”和“练习”。
- 全书性符号必须先登记到 `NOTATION.md`。
- 外部输入必须同步登记到 `THEOREM_LEDGER.md`；近期研究必须同步登记到 `FRONTIER_SOURCE_AUDIT_2026_07_08.md`。

## 严谨性检查

扩写或修改章节后逐项检查：

- 是否固定素数 $p$，以及对象是在 $\mathbf{Sp}$、$\mathbf{Sp}_{(p)}$、$\mathbf{Sp}^{\wedge}_p$、$\mathbf{Sp}_{K(n)}$ 还是 $\mathbf{Sp}_{T(n)}$ 中。
- 是否区分 spectrum、homology theory、ring spectrum、$\mathbb E_1$-ring、$\mathbb E_\infty$-ring 和 module category。
- Bousfield 等价、局部对象、无挠对象、acyclic 对象是否按定义使用。
- finite spectrum、compact spectrum、dualizable spectrum 是否被错误混用。
- type、$v_n$ self-map、telescope、$T(n)$、$K(n)$ 的关系是否依赖 Hopkins-Smith 外部输入。
- chromatic tower、monochromatic layer、fracture square 的 fiber/pullback 方向是否正确。
- Adams-Novikov、Morava change-of-rings、homotopy fixed point spectral sequence 是否说明收敛和连续群上同调条件。
- telescope conjecture、redshift 和 higher semiadditivity 是否按最新状态分层，不把前沿预印本当成基础定理。
