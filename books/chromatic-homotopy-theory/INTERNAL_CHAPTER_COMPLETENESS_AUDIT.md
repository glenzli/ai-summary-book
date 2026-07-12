# 逐章完整性审计

核查日期：2026-07-12

本文件判断每章是否仍停留在大纲态。判定标准：

- **正文态**：有定义、命题/证明、例子或使用规则，能支撑后续章节引用。
- **接口正文态**：有足够定义和边界，但深层定理依赖外部领域，作为本书接口可接受。
- **未收口**：缺少核心证明链、例子或 locator。

## 主体章节

| 章节 | 状态 | 理由 | 剩余缺口 |
| --- | --- | --- | --- |
| 00 | 正文态 | 固定范围、严格性、外部输入边界，并证明复定向给形式群律 | 无主要内容缺口 |
| 01 | 正文态 | Bousfield acyclic/local/localization、smashing、生成 acyclics 和 local object 检查均有证明 | localization 存在性 locator |
| 02 | 正文态 | 复定向、形式群、$BP/E(n)/K(n)$、Landweber exactness、$p$-typical convention 已展开 | CHT-P0-01/02/03 已定位到 Quillen Theorems 2/4、Ravenel 4.1.12(c)/4.1.18(a)、Landweber 2.6/Corollary 2.7 |
| 03 | 正文态 | $K(n)$ 检测、$E_n$、residue field、Kunneth、Nakayama 风险均有细节 | CHT-P0-12 已定位到 Goerss--Hopkins Section 7；未调用的 Hovey--Strickland 全书级结构归非主线 P1 |
| 04 | 正文态 | type 改为首次检测高度；thick/nilpotence/periodicity 状态分离；$v_0$ 与 $n\ge1$ 分开；$M(p)$ type $1$ 已内证；finite detection 定位到 HS II Theorem 14 | 无主链缺口 |
| 05 | 正文态 | 嵌套局部化、smash product、任意谱 fracture、finite-type specialization、convergence 的 $\lim/\lim^1$ 与截断误差边界已展开 | 主链 locator 已闭合 |
| 06 | 正文态 | $K(n)$-local category、stabilizer、continuous HFP、descent strategy 已展开 | CHT-P0-13/14 已定位到 Devinatz--Hopkins Definition 1.5、Theorems 1/2、Proposition 6.7；GHM 使用 CHT-P0-12 |
| 07 | 正文态 | telescope 反例、redshift 模板、cyclotomic redshift、frontier 分层已展开 | BHLŠ/HW/BSY 保持版本化 Frontier，不进入 P0 证明链 |
| 08 | 接口正文态 | elliptic/tmf 三层对象、ordinary/supersingular、$K(2)$ 使用规则已展开 | CHT-P0-19/20 已定位到 Silverman IV.7.4/7.5、AHS 与 Goerss Theorem 1.2/Definition 1.3；更多 level examples 属 P1 扩展 |
| 09 | 正文态 | semiadditivity、norm/Tate、cardinality、transchromatic character 边界已展开 | HKR 固定高度接口已精确定位；一般 higher-semiadditive height 结论保持非主线 P1 |
| 10 | 接口正文态 | splitting、GH duality、Picard、algebraic approximation 和 convention 模板已展开 | Strickland、Mor 与 GHMR locator 已登记；更多低高度实例属 P1 扩展 |
| 11 | 接口正文态 | genuine/motivic/synthetic 的定义、固定点、realization 风险已展开 | 若作为核心部分需多章扩写 |
| 12 | 正文态 | ANSS、chromatic SS、change of rings、HFPSS、Tate、计算流程、hidden extensions 和低 stem 表已展开 | Ravenel Theorem 4.4.1、Definition 5.1.7、Proposition 5.1.8 已定位；完整 ANSS 表属出版扩展 |

## 附录

| 附录 | 状态 | 作用 |
| --- | --- | --- |
| A | 正文态 | 形式群律逐项验算 |
| B | 正文态 | 谱序列约定和收敛检查 |
| C | 正文态 | Hopf algebroid/comodule/change-of-rings 口径 |
| D | 账本态 | 外部输入索引 |
| E | 正文态 | Bousfield/localization 失败模式 |
| F | 正文态 | 低高度初步例子 |
| G | 正文态 | 前沿预印本准入协议 |
| H | 正文态 | 稳定局部化和 $K(n)$-module field-like 细节 |
| I | 正文态 | $v_n$/telescope 约定 |
| J | 正文态 | Morava modules 和 descent 细节 |
| K | 接口正文态 | tmf/level/power operation 约定 |
| L | 接口正文态 | GH/Picard convention |
| M | 正文态 | ANSS hidden extension 和计算协议 |
| N | 正文态 | 低高度/fracture worked examples |
| O | 正文态 | 综合习题和解题提示 |
| Q | 正文态 | 低阶 stable stems 和 ANSS 校验表 |

## 当前结论

当前草稿已经不属于大纲态。它达到“教材内容基本收口稿”：每个主体章节都有
可阅读的正文模块、定义链、证明或使用规则；本轮审计的主线 P0 外部输入均有
theorem/section/page 与稳定 URL，非主线 P1 接口的证明角色也已显式降级；
计算链有低 stem 校验表。以下工作属于出版扩展而非当前严格完整性阻断项：

1. 扩充完整 ANSS 表格；
2. 加厚 tmf/Gross--Hopkins/Picard 的低高度案例；
3. 把 equivariant/motivic 拆成与普通 chromatic 同等深度的独立扩展部分。
