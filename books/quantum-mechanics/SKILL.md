---
name: quantum-mechanics
description: Use when writing, revising, or checking the rigorous Chinese quantum mechanics textbook in books/quantum-mechanics. Requires Hilbert-space precision, theorem-proof exposition, explicit domain conventions for unbounded operators, source traceability, and complete exercise-solution coverage.
---

# 量子力学教材写作技能

本技能约束 `books/quantum-mechanics/` 中《量子力学》教材的写作、扩写、校订和审稿。

## 基本原则

- 使用中文叙述；标准英文术语第一次出现时括注，例如“自伴算子（self-adjoint operator）”。
- 定义先于物理直觉；非平凡命题给出证明，或明确标注为“外部输入定理”。
- 不使用“显然”“容易看出”等词跳过数学步骤；若计算短，应写出内积、交换子、谱投影或概率公式。
- 凡涉及无界算子，必须说明定义域、闭性、对称性、自伴性或外部输入边界。
- 凡涉及测量，必须区分投影测量、POVM、仪器和测后态。
- 凡涉及复合系统，必须使用张量积、偏迹和完全正映射的明确公式。
- 本书写作对象是严格教材，不写成量子神秘主义、历史随笔或只给计算菜谱的题解集。

## 范围口径

- 本书主线是非相对论量子力学：Hilbert 空间、态、可观测量、谱定理、Schrodinger 演化、角动量、扰动、散射、相同粒子、密度算子和测量。
- 数学口径以可分复 Hilbert 空间为默认背景；有限维情形作为严格模型和计算训练，无限维情形引入自伴无界算子。
- 谱定理、Stone 定理、Kato-Rellich 定理、Mourre 理论、完整 scattering completeness 等大型泛函分析定理只作为外部输入定理。
- 本书包含相对论一粒子方程的边界章，但不展开完整量子场论；Fock 空间只作为多体量子力学与二次量子化的接口。
- 单位约定默认 $\hbar=1$；需要物理量纲时恢复 $\hbar$，并在公式附近说明。

## 写作格式

- 文件名使用两位编号，例如 `01_hilbert_spaces_states_and_rays.md`。
- 每章开头列出“本章目标”和“依赖前置知识”。
- 定义、约定、例子、命题、引理、定理、推论、练习使用“**定义 1.2.**”格式。
- 每章末尾必须包含“本章小结”和“练习”。
- 每个练习必须能在 `SOLUTIONS.md` 中找到对应答案或解题要点。
- 公式使用 Markdown/LaTeX；矩阵计算、交换子计算和谱分解必须写成可检查等式。
- 全书性符号必须写入 `NOTATION.md`。

## 严谨性检查

扩写或修改章节后逐项检查：

- Hilbert 空间内积约定是否一致。
- 态矢量、射线、密度算子和概率测度是否区分。
- 自伴算子与厄米矩阵、对称算子是否区分。
- 无界算子的乘积、交换子和指数是否有共同定义域或外部输入说明。
- 归一化、单位制和 Fourier 变换规范是否一致。
- 近似方法是否写明误差阶或适用条件。
- 外部输入定理是否在 `SOURCES.md` 和 `THEOREM_DEPENDENCIES.md` 中可追溯。
- 新增或修改练习后是否同步更新 `SOLUTIONS.md`。

