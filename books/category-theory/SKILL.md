---
name: category-theory
description: Use when writing, revising, or checking the rigorous Chinese category theory textbook in books/category-theory. Requires formal theorem-proof exposition, explicit universe and size conventions, source traceability, notation consistency, and coverage from ordinary categories through adjunctions, Kan extensions, monoidal/enriched category theory, to higher and infinity category theory.
---

# 范畴论教材写作技能

本技能约束 `books/category-theory/` 中《范畴论》教材的写作、扩写、校订和审稿。

## 基本原则

- 使用中文叙述；标准英文术语第一次出现时括注，例如“伴随函子（adjoint functor）”。
- 每个概念先给形式定义，再给例子、反例或边界条件；不得用类比替代定义。
- 非平凡命题必须给出完整证明，或明确标注为“外部输入定理”并在 `SOURCES.md` 记录来源。
- 不使用“显然”“容易看出”等词跳过关键步骤；若步骤短，应写出使用的定义、交换图或泛性质。
- 所有对象、态射、函子、自然变换、极限、余极限、同伦对象必须标明所在范畴或模型。
- 不把范畴论写成哲学随笔或科普史；正文默认读者愿意接受集合论宇宙、泛性质和交换图证明。

## 范围口径

- 本书以一阶范畴论为基础：范畴、函子、自然变换、Yoneda、极限、伴随、Kan 延拓、单子。
- 中段进入结构性主题：幺半范畴、闭范畴、富范畴、end/coend、可表现范畴、Grothendieck 范畴、topos。
- 后段进入同伦与高阶理论：2-范畴、双范畴、模型范畴、单纯集、quasi-category、Cartesian fibration、稳定 $\infty$-范畴、高阶 topos 和高阶代数。
- $\infty$-范畴默认采用 quasi-category 口径；与 simplicial categories、complete Segal spaces、relative categories 的比较作为后续章节专题处理。
- 固定 Grothendieck universes 处理大小问题；所有“范畴的范畴”必须说明小性层级。
- 后续扩写以范畴论本体为边界；外部数学理论只作为范畴论接口或例子出现。
- 若使用外部领域深定理，只标为“外部输入定理”并记录来源；不在本教材内部补其证明。
- 具体边界见 `F_scope_boundary_and_external_input_policy.md`。

## 资料源规则

- 优先使用正式教材、作者主页讲义、论文或专著：Mac Lane、Borceux、Kelly、Adamek-Rosicky、Riehl、Leinster、Joyal、Lurie、Cisinski、Hinich 等。
- 涉及 $\infty$-范畴定义、模型结构、straightening/unstraightening、higher topos 或 higher algebra 时，必须优先对照 Lurie、Riehl-Verity、Kerodon 或其他一手资料。
- 涉及版本、在线讲义链接、定理归属或近期修订时必须联网核查。
- Wikipedia、博客和百科型页面只能用于发现线索，不得作为核心定义或定理的最终依据。
- 不复制资料原文；所有正文用本书自己的中文重写。

## 写作格式

- 文件名使用两位编号，例如 `01_categories_functors_natural_transformations.md`。
- 每章在标题后以自然导言引出核心问题，并把真正需要的前置知识融入叙述或具体回指；不使用固定“本章目标”“依赖前置知识”栏目。
- 定义、约定、例子、命题、引理、定理、推论、练习使用“**定义 1.2.**”格式。
- 每章以内容特定的段落收束论证并保留“练习”；不使用固定“本章小结”栏目。
- 每个练习必须能在 `SOLUTIONS.md` 中找到对应答案或解题要点。
- 公式使用 Markdown/LaTeX；交换图可用矩阵、tikzcd 风格代码块或明确的等式条件描述。
- 术语和符号必须与 `NOTATION.md` 一致；新增全书性符号必须先更新 `NOTATION.md`。

## 严谨性检查

扩写或修改章节后逐项检查：

- 是否已经声明集合论宇宙和小性层级。
- 每个 Hom、Nat、Fun、Map、极限、余极限是否有明确环境。
- 泛性质是否写成自然双射、终对象/始对象，或等价的表示性陈述。
- 伴随是否说明单位、余单位和三角恒等式。
- Kan 延拓是否说明沿哪个函子延拓，在哪个函子范畴中具有泛性质。
- 幺半、闭、富范畴是否给出相干性条件，而不只给出张量记号。
- $\infty$-范畴章节是否区分严格等式、同构、等价、同伦和可缩选择空间。
- 外部输入定理是否在 `SOURCES.md` 中可追溯。
- 新增或修改练习后是否同步更新 `SOLUTIONS.md`。

## 本书口径

- 第一部分先建立普通范畴论，不预设读者已经熟悉同调代数或拓扑。
- 第二部分将抽象工具应用到代数、拓扑、逻辑和几何中的标准例子。
- 第三部分把一阶范畴论升级到同伦和高阶范畴论；进入 $\infty$-范畴时不再把“等号”和“同伦等价”混用。
- 附录承担集合论宇宙、单纯形范畴、常用交换图和证明模板，不把技术约定散落到正文各处。
