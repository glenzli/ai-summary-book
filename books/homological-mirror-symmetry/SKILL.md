---
name: homological-mirror-symmetry-textbook
description: Use when writing, revising, or checking the rigorous Chinese textbook on Homological Mirror Symmetry in books/homological-mirror-symmetry. Requires source-grounded theorem-proof exposition, explicit universe/field/category conventions, strict separation between A-model, B-model, dg/A-infinity enhancements, triangulated shadows, Fukaya-category analytic inputs, and current research frontiers.
---

# Homological Mirror Symmetry 教材写作技能

本技能约束 `books/homological-mirror-symmetry/` 中《Homological Mirror Symmetry》教材的写作、扩写、校订和审稿。

## 基本原则

- 使用中文叙述；标准英文术语第一次出现时括注，例如“同调镜像对称（homological mirror symmetry）”。
- 定义必须先于直觉。若某个对象来自物理或几何直觉，正文仍必须先给出可检查的数学数据。
- 非平凡命题必须给出完整书内证明，或明确标注为“外部输入定理”并在 `SOURCES.md` 中记录来源；证明路线只能解释外部输入，不计作书内证明。
- 不使用“显然”“容易看出”等词跳过关键步骤；短证明也要写出使用的定义、交换图、同伦方程、泛性质或紧致化边界分解。
- 严格区分 dg category、$A_\infty$-category、pretriangulated envelope、triangulated homotopy category、stable infinity-category、Morita equivalence、quasi-equivalence 和 ordinary equivalence。
- 严格区分 A-side 的 Fukaya category、wrapped Fukaya category、partially wrapped Fukaya category、Fukaya-Seidel category、Rabinowitz Fukaya category，以及 B-side 的 $\operatorname{Perf}$、$\mathrm D^b\operatorname{Coh}$、matrix factorizations 和 singularity categories。
- 不把 “$A$ 与 $B$ 相等”写成数学结论；HMS 断言必须写成带增强的范畴等价、Morita 等价或稳定 $\infty$-范畴等价，并说明采用的模型。
- 不复制资料原文；所有正文使用本书自己的中文重写。

## 本书口径

- 全书固定 Grothendieck universes 与基域 $k$；除非另有说明，线性范畴为 $k$-线性。
- $A_\infty$ 约定采用 cohomological grading；高阶复合 $\mu^d$ 的次数为 $2-d$。为避免符号歧义，基础定义优先用 suspension coalgebra 约定表述，低阶公式再展开。
- B-side 默认以 dg 或 stable $\infty$ enhancement 为主体；三角范畴只作为 $H^0$ 或同伦范畴的影子。
- A-side 正文先处理 exact/compact 或 Liouville 语境，在 brane data、grading、orientation、Novikov 系数、bounding cochains、transversality 和 compactness 已经说明后再进入一般 Fukaya 范畴。
- 涉及 general compact Fukaya category、virtual fundamental chains、Kuranishi/polyfold/implicit-atlas 技术时，除非本书已建立完整分析基础，否则标为外部输入。
- 研究前沿只收录已联网核查的一手资料；除非完成独立验证，不把 2025-2026 预印本中的新结果写成正文定理。

## 资料源规则

- 优先使用正式教材、专著、作者主页讲义、arXiv 论文和出版商页面：Kontsevich、Seidel、Fukaya-Oh-Ohta-Ono、Auroux、Keller、Lefevre-Hasegawa、Huybrechts、Bondal-Orlov、Ganatra-Pardon-Shende、Nadler、Abouzaid、Auroux、Sheridan、Lekili-Ueda 等。
- 涉及近期研究、预印本、版本日期、定理归属或开放问题状态时必须联网核查。
- Wikipedia、百科页面和博客只能用于发现线索，不得作为核心定义或定理的最终依据。
- `SOURCES.md` 必须标明资料用途：基础定义、核心定理、外部输入、研究边界或历史说明。

## 写作格式

- 文件名使用两位编号，例如 `01_dg_and_a_infinity_categories.md`。
- 每章在标题后以自然导言引出镜像对称问题，并把必要依赖融入叙述或精确回指；不使用固定“本章目标”“依赖前置知识”栏目。
- 正文不使用项目收尾、审查流程、任务队列或未完成占位等编辑术语；应把它们改写为定理适用范围、所需外部输入、失败模式或明确开放问题。账本与校对文件不受此限制。
- 定义、约定、例子、命题、引理、定理、推论、练习使用“**定义 1.2.**”格式。
- 每章以内容特定的段落收束论证并保留“练习”；不使用固定“本章小结”栏目。
- 公式使用 Markdown/LaTeX；交换图可用 tikzcd 风格代码块、矩阵或明确的等式条件描述。
- 全书性符号必须先登记到 `NOTATION.md`。

## 严谨性检查

扩写或修改章节后逐项检查：

- 是否声明集合论宇宙、基域、分次和符号约定。
- 是否说明当前使用的是 dg、$A_\infty$、triangulated、Morita 或 stable $\infty$ 语境。
- 是否区分 quasi-isomorphism、quasi-equivalence、fully faithful、split-generation、Morita equivalence 和 equivalence of triangulated categories。
- B-side 的 $\operatorname{Perf}(X)$、$\mathrm D^b\operatorname{Coh}(X)$、matrix factorization 是否有明确几何假设。
- A-side 的 Fukaya 类别是否说明 exact/monotone/Novikov/obstructed/curved 的口径。
- holomorphic curve 计数是否说明 compactness、orientation、transversality 和 gluing 是内部证明还是外部输入。
- HMS 例子是否列出镜像对数据、两边类别、系数、生成元、等价函子和可验证不变量。
- 外部输入定理是否在 `SOURCES.md` 中可追溯。
