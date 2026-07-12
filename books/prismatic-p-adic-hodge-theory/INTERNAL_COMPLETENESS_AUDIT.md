# 内部完整性审计

初次审计日期：2026-07-08。主线复核：2026-07-11。

## 审计目标

本审计回答一个具体问题：本书是否已按正式教材内容范围，具有内部完整的定义链、证明链、技术基础和边界说明。

## 内部完整性状态

当前状态：`Chapter-Closed-Draft`。

含义：

- 正文范围已经覆盖正式教材应包含的基础对象、比较定理、积分理论、表示论接口、系数方向和应用边界。
- 内部定义链已闭合；读者不需要先去外部资料才能理解 prism、site、cohomology、crystal、Nygaard、syntomic 和 lattice 等对象的基本定义。
- 大型比较定理仍作为外部输入，不伪装为内部证明。
- 技术基础已由附录 G-K 和 `SOLUTIONS.md` 补强，并由 `CHAPTER_CLOSURE_AUDIT.md` 逐章核对正文密度。

## 定义链审计

| 对象 | 定义位置 | 是否内部闭合 |
| --- | --- | --- |
| $\delta$-ring | 定义 1.1, 附录 H | 是 |
| Frobenius lift | 命题 1.3, H.2 | 是 |
| Koszul derived completion | 定义 A.1--A.2 | 是；一般 quotient-power formula 明确标出额外假设 |
| prism | 定义 2.3 | 是 |
| prismatic site | 定义 2.13--2.16, 附录 G | 是；probe ideal 与 complete-flat topology 已固定 |
| $R\Gamma_\Delta$ | 定义 2.18, 附录 G | 是 |
| Hodge-Tate/de Rham specialization | 第九章 | 是 |
| Nygaard/syntomic | 第七、十一章，附录 F | convention 内闭合；relative Nygaard 与 BMS2 fibre/products/nearby cycles 均已达 L3 |
| prismatic crystals | 第六章，附录 I | 是；localized/effective Frobenius 已分开 |
| BK/BKF modules | 第十二章，附录 J | 精确定义内闭合；finite projective/free/height 边界已分开 |

## 证明链审计

| 类型 | 状态 |
| --- | --- |
| $\delta$-环基本代数 | 已内部证明 |
| distinguished element 推 prism 条件 | 已内部证明 |
| boundedness 基本检查 | 已内部证明 |
| principal derived completion 与 bounded torsion | 已内部证明；多生成元一般比较为外部输入 |
| complete flatness reduction 与 ordinary Tor-dimension-one base change | 已内部证明 |
| derived global sections 基础性质 | 已内部说明 |
| crystals/descent 基础 | 已内部说明，faithfully flat descent 为外部输入 |
| comparison theorem | 全部外部输入 |
| prismatic $F$-crystal classification | 外部输入 |
| period ring admissibility | 外部输入 |

## 细节完整性审计

已补齐：

- 技术基础：formal schemes、sites、derived global sections。
- 代数细节：$\delta$-恒等式、Frobenius lift、distinguished elements。
- 线性对象：crystals、descent、vector bundles、semilinear maps、lattices。
- 完备性与 torsion：Koszul tower、$p/I/(p,I)$ 区分、complete flatness、complete/ordinary Tor-amplitude。
- 局部模型：crystalline prism、Hodge-Tate graded pieces、derived fixed points、$q$-difference、rank-one BK modules。
- 教学辅助：术语索引和完整习题解答。

仍需外部完成：

- 未进入正文主线的 THH/$A\Omega$/de Rham--Witt 配套输入之 L3 locator。
- Classical Fontaine/Faltings/Tsuji comparison 的精确 locator。
- 习题解答可继续扩展为多解法、更多反例和课堂提示。
- 出版级交叉引用重写和排版。

## 判定

按“正式教材内容范围、内部完整、细节完整”的标准，本书现在达到逐章教材收口草稿：内容范围、内部定义/证明细节和每章正文密度已经成体系。正文实际依赖的核心 prismatic/Nygaard/BMS1/BMS2/$F$-crystal statements 已完成 PDF 定理号核查；尚未达到最终引用闭包，因为 classical comparison、未进入主线的配套结果和出版级书目仍需补强。
