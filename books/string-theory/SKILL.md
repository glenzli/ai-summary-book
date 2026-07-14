---
name: string-theory
description: Use when writing, extending, or reviewing the rigorous Chinese textbook on string theory in books/string-theory. Requires theorem-definition-proof style, explicit worldsheet/spacetime conventions, alpha-prime normalization, CFT and BRST rigor, source traceability, and separation between derived results, standard external input, and physical conjectures.
---

# String Theory 教材写作技能

本技能约束 `books/string-theory/` 中《String Theory：从世界面量子场论到对偶性》教材的写作、扩写、校订和审稿。

## 基本原则

- 使用中文叙述；标准英文术语第一次出现时括注，例如“世界面（worldsheet）”。
- 每个概念先由具体问题、失败的旧语言或可计算例子引出，再给形式定义并立即说明
  良定义性与物理含义；不得用图像或比喻替代定义。
- 非平凡数学命题必须给出完整书内证明或明确标注为“外部输入定理”。标准物理形式主义中的压缩计算标为“推导说明（标准物理口径）”，并说明正规化、路径积分或微扰输入；它不计作严格数学证明。
- 不使用“显然”“容易看出”等词跳过关键步骤；若步骤短，应写出变分、Poisson bracket、OPE 或 Ward identity 计算。
- 每个公式必须说明所处环境：Minkowski 或 Euclidean worldsheet、target-space signature、closed/open string、tree/genus、operator formalism 或 path integral formalism。
- 不把 string theory 写成历史散文；正文默认读者愿意接受研究生层级的微分几何、量子场论、复分析和表示论。

## 范围口径

- 本书以 perturbative string theory 为主线：worldsheet action、CFT、BRST、critical dimension、scattering amplitudes、D-branes、superstrings、heterotic strings 和 compactification。
- 非微扰结构作为后半部分专题处理：duality、M-theory、black branes、AdS/CFT 和 topological strings。
- 数学语言用于表达 sigma models、CFT、moduli spaces、fiber bundles、index/anomaly、Calabi-Yau geometry 和 derived/stack-like 接口；不得把尚未证明的物理对偶性写成数学定理。
- 物理假设、标准但未完全严格的路径积分步骤和 conjectural dualities 必须与已证明的 classical/quantum field-theoretic statements 分开标注。

## 外部输入分级

- 核心结构：直接服务 string spectrum、worldsheet CFT、BRST cohomology、modular invariance、D-brane boundary condition、low-energy effective action 的材料，应在正文展开。
- 支撑接口：完整 CFT 构造、unitarity/no-ghost theorem、moduli of Riemann surfaces、index theorem、supergravity 分类、Calabi-Yau Hodge theory，可给精确定理陈述和使用位置。
- 卫星理论：完整代数几何、derived geometry、完整 supersymmetric QFT 构造、full nonperturbative string theory，不并入主体。

## 归一化规则

- 默认 target-space metric 为 mostly plus：$\eta_{\mu\nu}=\operatorname{diag}(-,+,\ldots,+)$。
- 默认 worldsheet coordinates 为 $(\tau,\sigma)$，闭弦取 $\sigma\sim\sigma+2\pi$，开弦取 $\sigma\in[0,\pi]$。
- 默认 Polyakov 作用量为
  $$
  S_P=-\frac{1}{4\pi\alpha'}\int d^2\sigma\,\sqrt{-h}\,h^{ab}\partial_aX^\mu\partial_bX_\mu.
  $$
- Euclidean worldsheet 下作用量改写为正定号，并显式声明 Wick rotation。
- OPE 默认采用
  $$
  X^\mu(z,\bar z)X^\nu(w,\bar w)\sim-\frac{\alpha'}{2}\eta^{\mu\nu}\log|z-w|^2.
  $$
- Virasoro generators、normal ordering constant、mass formula 和 ghost number 必须与 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 一致。

## 写作格式

- 文件名使用两位编号，例如 `03_worldsheet_cft.md`。
- 每章 H1 后、第一个 H2 前写自然导言，原则上不少于约 120 个有效汉字；导言从
  现象、计算或前章缺口进入，并自然吸收真正需要的前置知识、convention 与推进方向。
- 不使用“本章目标”“依赖前置知识”“主线”“本章小结”等固定正文标题，也不以
  同义标题恢复统一模板。
- 定义、约定、例子、命题、引理、定理、推论、练习使用“**定义 1.2.**”格式。
- 证明结束使用 `$\square$`。
- 每章在练习前以自然段回应章首问题并说明所得结论的适用边界；收束不设统一标题。
  每章保留“练习”分节，题目必须使用正文实际建立的定义、计算或证明机制。
- 新增全书性符号必须先更新 [NOTATION.md](NOTATION.md)。
- 新增或改名术语必须同步更新 [GLOSSARY.md](GLOSSARY.md)。
- 新增习题必须同步更新 [EXERCISE_INDEX.md](EXERCISE_INDEX.md) 和 [SOLUTIONS.md](SOLUTIONS.md)。

## 严谨性检查

扩写或修改章节后逐项检查：

- 是否声明 worldsheet/target-space signature。
- 是否固定 $\alpha'$、string tension、mode expansion 和 oscillator commutator 的归一化。
- 是否区分 classical constraint、quantum constraint 和 BRST cohomology。
- 是否把 gauge fixing、Faddeev-Popov determinant 和 ghost action 的来源说清楚。
- 是否区分 theorem、external input、physical conjecture 和 perturbative calculation。
- 是否说明 anomaly cancellation、critical dimension、modular invariance 和 GSO projection 的使用层级。
- 是否在 [SOURCES.md](SOURCES.md) 中记录使用的主要资料源。
