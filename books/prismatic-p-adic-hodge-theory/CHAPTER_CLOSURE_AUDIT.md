# 逐章教材收口审计

审计日期：2026-07-08。

## 审计标准

一章达到“教材收口草稿”需满足：

1. 有目标和前置知识；
2. 核心概念在章内或前置章节中已定义；
3. 非平凡断言有证明、证明草图或外部输入标记；
4. 不只是目录式列点，必须含至少一种正文细节：计算、例子、结构表、错误边界、形式推论或长正合列；
5. 有小结和练习；
6. 外部输入可追溯到 `SOURCES.md` 或 `D_theorem_locator_index.md`。

## 逐章判定

| 章 | 主题 | 细节类型 | 判定 |
| --- | --- | --- | --- |
| 0 | 范围与严格性 | 阅读路线、内部学习闭包命题 | 收口草稿 |
| 1 | $\delta$-环 | Frobenius lift 证明、低阶 $\delta$ 公式、BK 例子 | 收口草稿 |
| 2 | Prism 与 site | prism 条件证明、affine probe 读法、site 方向警告 | 收口草稿 |
| 3 | 比较定理接口 | specialization 类型、base change、fixed-point comparison 判别 | 收口草稿 |
| 4 | Fontaine 理论 | admissibility、filtered $\varphi$-module、lattice 不可见性 | 收口草稿 |
| 5 | BMS | $L\eta$ 边界、四个出口、torsion spectral sequence | 收口草稿 |
| 6 | $F$-crystals | crystal 条件、tensor/dual、classification 外部输入边界 | 收口草稿 |
| 7 | Nygaard/syntomic | naive filtration、乘法相容、syntomic fibre convention | 收口草稿 |
| 8 | Prismatization | $F$-gauge 区分、site-to-stack 信息保真条件 | 收口草稿 |
| 9 | Hodge-Tate/de Rham | conjugate filtration、perfectness 推论、低维谱序列 | 收口草稿 |
| 10 | Crystalline/q-de Rham | crystalline boundedness、$q$-difference 局部计算 | 收口草稿 |
| 11 | Etale/syntomic | derived fixed points、长正合列、cup product 要求 | 收口草稿 |
| 12 | BK/BKF modules | rank-one 模型、height 条件、lattice interface | 收口草稿 |
| 13 | Coefficients/non-abelian | coefficient package、free crystal 例子、边界判别 | 收口草稿 |
| 14 | Applications | Artin/Shimura/Brauer/group scheme 假设表 | 收口草稿 |
| 15 | Closure | 错误模式、逐章收口判据、开放问题 | 收口草稿 |

## 不再只是大纲的证据

- 每章至少有一个编号命题、定义、例子、警告或结构表。
- 第 1、2、7、9、10、11、12、13 章含内部证明或计算。
- 第 3、4、5、6、8、14、15 章含结构边界或错误模式，避免把外部大定理写成无证陈述。
- 附录 G-K 提供跨章节技术细节，防止正文依赖未解释背景。
- `SOLUTIONS.md` 覆盖正文与技术附录的章末练习，支持教学使用。

## 剩余非收口项

这些项不影响“教材收口草稿”，但影响最终出版：

- 外部输入 locator 尚未全部达到 L3；
- Bhatt-Scholze Hodge-Tate/Nygaard 与 BMS2 syntomic/Tate twist 基础公式已完成源码级核对；仍需出版级 L3 和变体细分；
- 章内散文交叉引用尚未全部替换为稳定 label；
- 习题解答已闭合；出版版仍可扩展为多解法或更长讲义式提示；
- 中文术语和公式断行仍需出版社级校对。

## 判定

按正式教材内容范围与章节细节标准，本书当前达到“逐章教材收口草稿”。核心定义链、正文密度、内部证明与 prismatic/BMS2 外部输入链已经收口；它不是 camera-ready 出版稿，且 classical comparison locator 仍需补强。
