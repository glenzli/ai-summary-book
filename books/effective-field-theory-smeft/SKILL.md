---
name: eft-smeft-textbook
description: Use when writing, revising, or checking the Chinese rigorous textbook on Effective Field Theory and Standard Model Effective Field Theory in this repository. Requires scale separation, symmetry-first construction, explicit power counting, operator-basis bookkeeping, matching/RG discipline, source traceability, and strict separation between book-derived statements, external calculations, phenomenological conventions, and research boundaries.
---

# EFT/SMEFT 教材写作约束

本文件约束 `books/effective-field-theory-smeft/` 中《有效场论与标准模型有效场论》教材的写作、扩写和审查。

## 基本原则

- 以中文叙述；第一次出现标准术语时保留英文括注，例如“Wilson 系数（Wilson coefficient）”。
- 每章必须包含“本章目标”“依赖前置知识”“本章小结”“练习”。
- 定义先于直觉；凡涉及非平凡陈述，必须标注为“书内推导”“推导说明”“外部输入”或“研究边界”。
- 不用“自然性”“量纲分析”“对称性允许”替代推导；必须说明采用的尺度、场归一化、对称群和幂计数。
- 不写“显然”“容易看出”跳过关键步骤；若省略标准计算，必须说明省略的是张量代数、环积分、群论恒等式、还是全局拟合细节。
- 不把 UV 模型、LEFT、HEFT、SMEFT、SME 或具体拟合工具混用；每次改变有效理论必须写清自由度、对称性和匹配尺度。
- 不复制来源原文；所有内容用本书自己的中文重写，来源集中记录在 `SOURCES.md`。

## 证明与推导标签

- **书内推导。** 本书从前文定义、代数恒等式、路径积分操作或 Feynman 规则推出。
- **推导说明。** 本书给出完整路线，但压缩了标准长计算；必须说明压缩的具体位置。
- **外部输入。** 本书不推导，只引用论文、综述或教材；必须说明依赖风险。
- **研究边界。** 仍在发展的方向，例如维数八完整重整化、全局拟合方案选择、非线性电弱 EFT 的适用范围。

## 技术纪律

- 每个 EFT 必须写出有效自由度、对称群、截断阶数、展开参数和适用能区。
- 每个 Wilson 系数必须带有算符维数、归一化约定和 flavor 假设。
- 使用 EOM、分部积分、Bianchi 恒等式或 Fierz 恒等式删除算符时，必须说明这是基变换或 S-matrix 等价，不是逐点拉氏量相等。
- 匹配必须说明是树级、单圈、on-shell、off-shell、Green 函数匹配还是振幅匹配。
- RG 方程必须说明重整化方案；默认使用维数正规化和 $\overline{\mathrm{MS}}$。
- SMEFT 默认采用线性电弱实现，Higgs 为 $SU(2)_L$ 双重态；若改用 HEFT，必须另开章节或明确边界。
- 当前书稿默认只把 baryon-number conserving dimension-six Warsaw basis 作为内部主线；flavor 展开、CP 分类和 dimension-eight 作为高级边界。

## 来源纪律

- 核心定义优先来自 Weinberg/Georgi/Burgess 等 EFT 教材或综述、Warsaw basis 原始论文、Jenkins-Manohar-Trott RGE 系列、现代 SMEFT 综述和 Snowmass 报告。
- 涉及 2024 年以后文献时必须记录访问日期或版本。
- Wikipedia、博客、讲义可用于发现线索，但不得作为核心定义来源。

## 当前范围

本书第一版目标是建立 EFT 到 SMEFT 的严格主线：

1. 尺度分离、局域性和 Wilsonian 观点；
2. 路径积分匹配和 Wilson 系数；
3. 幂计数、重整化和 RG；
4. 算符冗余与基；
5. 标准模型场内容和规范对称性；
6. SMEFT 的定义、维数展开和维数五/六结构；
7. Warsaw basis 的组织方式；
8. 匹配、运行和可观测量的工作流。

后续扩写应优先关闭 `K_remaining_obligations.md` 中的义务，而不是横向加入无关物理方向。
