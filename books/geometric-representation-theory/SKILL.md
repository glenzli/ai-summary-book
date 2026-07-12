---
name: geometric-representation-theory-textbook
description: Use when writing, revising, or checking the rigorous Chinese textbook on geometric representation theory in books/geometric-representation-theory. Requires primary-source grounding, formal definitions before intuition, explicit base field and sheaf-theoretic conventions, theorem-proof structure, strict separation between internal arguments, external input theorems, and current research frontiers.
---

# Geometric Representation Theory 教材写作技能

本技能约束 `books/geometric-representation-theory/` 中《Geometric Representation Theory》教材的写作、扩写、校订和审稿。

## 基本原则

- 使用中文叙述；标准英文术语第一次出现时可给中文解释，例如“geometric Satake equivalence（geometric Satake 等价）”。
- 定义先于直觉。所有对象必须说明所在范畴、底域、拓扑或 Grothendieck site、系数环、群作用和商的含义。
- 非平凡命题必须给出完整书内证明，或标注为“外部输入定理”。证明路线只解释外部输入，不计作证明；外部输入必须在 `SOURCES.md` 和 `THEOREM_LEDGER.md` 中可追踪。
- 不使用“显然”“易知”等词跳过关键步骤。短证明也要写明使用的定义、泛性质、伴随性、基变换、滤过或三角恒等式。
- 区分代数表示、Lie 代数模、Harish-Chandra 模、constructible sheaf、perverse sheaf、D-module、l-adic sheaf、ind-coherent sheaf 和 dg/infinity category。不得把等式、同构、准同构、等价、Morita 等价和 t-exact 等价混用。
- 不复制资料原文；所有正文使用本书自己的中文重写。

## 本书口径

- 默认底域为代数闭域 $k$。除非另行声明，经典复几何、D-module 和 category $\mathcal O$ 章节取 $k=\mathbb C$，且系数域为特征 $0$ 域 $E$。
- 默认 $G$ 为连通 reductive algebraic group，$B\subset G$ 为 Borel subgroup，$T\subset B$ 为 maximal torus，$W=N_G(T)/T$ 为 Weyl group。
- flag variety 默认指完全旗簇 $\mathcal B=G/B$；partial flag variety 写作 $G/P$。
- “几何表示论”在本书中不是单一理论，而是用几何范畴、层、D-module、卷积和导出范畴构造或控制表示范畴的技术体系。
- 基础章节先建立 $G/B$、Schubert 分解、category $\mathcal O$、equivariant sheaves、perverse sheaves 和 Hecke category。高级章节再进入 Springer 理论、Beilinson-Bernstein localization、geometric Satake、affine Grassmannian、character sheaves、quiver varieties、categorification、Coulomb branches 和 geometric Langlands。
- 近期研究只收录已联网核查的一手资料或作者页面。除非本书完成独立验证，不把 2024-2026 预印本中的新结论写成基础定理链。

## 资料源规则

- 优先使用正式论文、专著、作者主页、arXiv、出版商页面和课程讲义。百科页面只能用于发现线索，不作为核心定理最终依据。
- 基础来源包括 Borel、Springer、Jantzen、Humphreys、Chriss-Ginzburg、Kashiwara-Schapira、Hotta-Takeuchi-Tanisaki、Beilinson-Bernstein-Deligne、Beilinson-Bernstein、Brylinski-Kashiwara、Kazhdan-Lusztig、Ginzburg、Mirkovic-Vilonen、Lusztig。
- 涉及 geometric Langlands、Coulomb branches、symplectic duality、categorical representation theory、KLR/Rouquier、Hodge/Soergel/Koszul duality 等近期或活跃方向时，必须记录版本日期和使用范围。
- `SOURCES.md` 必须标明资料用途：基础定义、核心定理、外部输入、研究边界或历史说明。

## 写作格式

- 文件名使用两位编号，例如 `01_reductive_groups_flag_varieties_and_weights.md`。
- 每章开头列出“本章目标”和“依赖前置知识”。
- 定义、约定、例子、命题、引理、定理、推论、练习使用“**定义 1.2.**”格式。
- 每章末尾必须包含“本章小结”和“练习”。
- 公式使用 Markdown/LaTeX。卷积图、六函子图和基变换图可用 tikzcd 风格代码块表示。
- 全书性符号必须先登记到 `NOTATION.md`。

## 严谨性检查

扩写或修改章节后逐项检查：

- 是否声明底域、系数域、拓扑/site 和 sheaf theory 模型。
- 群作用是左作用还是右作用；商 $G/H$、$[X/H]$、$H\backslash X/K$ 是否类型正确。
- equivariant category 是否解释为 quotient stack 上的层，还是使用 descent datum。
- functor 是普通函子、导出函子、三角函子、dg functor 还是 infinity functor。
- perverse t-structure 的维数函数、support/cosupport 条件和 shift 约定是否一致。
- convolution 是否写出 correspondence、拉回/张量/推前、properness 或 compact support 条件。
- Schubert cell、orbit closure、Bruhat order 和 Weyl group convention 是否一致。
- category $\mathcal O$ 是否固定 triangular decomposition、中心 character、integral block 和 dot action 约定。
- D-module 章节是否区分 left/right D-module、twisted D-module、regular holonomic 条件和 Riemann-Hilbert 对应。
- 大型结果如 decomposition theorem、Beilinson-Bernstein localization、Kazhdan-Lusztig conjecture、geometric Satake、Springer correspondence、geometric Langlands 必须列为外部输入，除非正文给出完整证明。
