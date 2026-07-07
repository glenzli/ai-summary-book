# 数学审查记录

核查日期：2026-07-08
状态：正式教材扩展初稿

## 1. 已完成检查

- 已固定素数 $p$、稳定 infinity-范畴和 $p$-局部谱口径。
- 已建立 `K(0)`、`K(n)`、`E(n)`、`E_n`、`L_n`、`M_n`、`T(n)`、`L_n^f` 的符号表。
- 已把 Bousfield 偏序方向风险改成显式 acyclic implication 规则。
- 已把 finite spectrum、compact spectrum 和一般谱区分。
- 已把 Quillen、DHS、Hopkins-Smith、chromatic convergence、Goerss-Hopkins-Miller、Devinatz-Hopkins 等大型定理登记为外部输入。
- 已按 2023 后状态将 telescope conjecture 改写为历史命题/失败模式，而非默认假设。
- 已把 2026 年 syntomic/K-theory of `BP<n>` 结果登记为 frontier，不进入证明链。
- 已补齐第 9-12 章，覆盖 higher semiadditivity、transchromatic character、chromatic splitting、Gross-Hopkins duality、Picard groups、equivariant/motivic chromatic theory 和计算工具。
- 已补齐附录 C/E/F/G，覆盖 Hopf algebroid、局部化失败模式、低高度样例和前沿预印本准入协议。
- 已补齐附录 H-M，覆盖稳定局部化细节、$K(n)$-module 场性边界、$v_n$/telescope 约定、Morava module/descent、tmf/level/power operations、Gross-Hopkins/Picard convention 和 Adams-Novikov hidden extension 协议。
- 已新增 `PUBLICATION_CLOSURE_MATRIX.md`，区分正式教材范围覆盖、内部完整性和细节完整性。
- 已新增 `P0_REFERENCE_LOCATORS_BATCH_1.md`，为 Quillen、BP、Landweber、DHS、Hopkins-Smith、chromatic convergence 和 Hovey-Strickland 建立第一批 bibliographic locator。
- 已新增 `P0_REFERENCE_LOCATORS_BATCH_2.md`，为 GHM、Devinatz-Hopkins、HKR、tmf、Gross-Hopkins、Picard 和 ANSS 建立第二批 bibliographic locator。
- 已新增附录 N，补低高度和 fracture worked examples。
- 已新增附录 O，补综合习题与解题提示。
- 已新增逐章完整性审计，明确所有主体章节已脱离大纲态，但 08、10、11 仍属接口正文态。
- 已新增 `P1_REFERENCE_LOCATORS_FRONTIER.md`，使前沿、半加性、equivariant/motivic 的 content-level 引用闭合。
- 已新增附录 Q，补低阶 stable stems、primary decomposition、hidden extension 最小模型和 ANSS 记录模板。
- 已扩写第 1-12 章正文：每章新增至少一个教材性细节模块，包括定义、命题、证明、例子、使用规则或计算流程，降低“大纲化”风险。

## 2. 当前数学风险

| 风险 | 状态 | 处理 |
| --- | --- | --- |
| $E(n)$ Bousfield 类与 Morava K wedge 的等式 | 外部输入 | 已有 bibliographic locator，需精确 theorem locator |
| $K(n)$ module category 的 field-like 性质 | 已补证明草图 | 仍需外部 theorem locator |
| $v_n$ self-map 定义中的 nilpotence 条件 | 已精化 | 仍需 Hopkins-Smith locator |
| chromatic fracture square 适用范围 | 外部输入 | 需要按对象类别和 localization 版本定位 |
| Morava descent 谱序列收敛 | 外部输入 | 附录 B 已标检查表 |
| tmf 构造 | 外部输入 | 当前只作接口 |
| redshift 定理版本差异 | 前沿风险 | 需 theorem locator 和假设翻译 |
| higher semiadditivity 定理版本 | 外部输入 | 需区分 Hopkins-Lurie、CSY、Ben-Moshe |
| Gross-Hopkins duality 公式 | convention 风险 | 暂不写未定位悬挂公式 |
| Picard group exotic part | 计算风险 | 需 descent spectral sequence locator |
| equivariant/motivic 迁移 | 模型风险 | 已加入 naive/genuine、topological/motivic 警告 |
| 每章正文厚度 | 已改善 | 已补综合习题和低 stem 表；完整 ANSS 表属扩展 |

## 3. 下一轮严格化任务

1. 补 `D_source_theorem_index.md`，为 P0 定理加精确 locator。
2. 将 P0 bibliographic locator 升级为精确 theorem/section/page locator。
3. 为第九章补 semiadditive height 的正式定义和 HKR/transchromatic character locator。
4. 为第十章补 Gross-Hopkins duality 具体 locator 和 Picard group 低高度案例。
5. 为第十二章补 Adams-Novikov 具体低 stem 样例。
6. 为第十一章补 equivariant Balmer spectrum、motivic $MGL$ 和 synthetic reconstruction 的 locator。

## 4. 当前收口判定

本书达到“教材内容基本收口稿”：主体方向覆盖较完整，每章已具备正文细节模块，内部定义链和主要 convention 已经闭合，核心外部输入已有 content-level locator，前沿边界已纳入，低阶计算链已有最小可核查表。剩余工作主要是 camera-ready 前的 theorem/section/page locator、完整 ANSS 表和若干低高度案例加厚。
