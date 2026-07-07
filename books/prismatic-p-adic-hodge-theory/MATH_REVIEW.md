# 数学审查记录

核查日期：2026-07-08。
状态：逐章教材收口草稿；基础定义链、比较定理接口和核心 P0 源码 locator 已系统化，尚未达到最终出版收口。

## 当前已固定的严格性边界

- 已固定素数 $p$、derived completion 口径、$\delta$-环和 Frobenius lift 符号。
- 已把 prism 默认解释为 Bhatt-Scholze 有界 prism。
- 已区分 internal proof、external input theorem 和 frontier-only result。
- 已把 2025-2026 资料列入研究边界，不纳入基础定理链。

## 当前高风险点

- Hodge-Tate comparison 的 twist 符号已经在附录 F 中与 Bhatt-Scholze v4 `thm:A` 源码核对；出版前仍需转换为 L3 locator。
- Nygaard filtration 已与 Bhatt-Scholze v4 `thmCagain` 源码核对；syntomic complex 的基础 fibre convention 已定位到 BMS2 `eq:TateTwist` 和 `thm:main6`，并已吸收到第七章和第十一章；mod $p^r$、truncation、nearby cycles 变体仍需最终 L3。
- Classical Fontaine period rings 的 construction 本身很长，本书正式教材稿给出严谨定义口径、admissibility interface 和 comparison theorem interface，但仍不声称完整重建 Fontaine 理论。
- Prismatic $F$-crystals 与 crystalline lattice 的等价是外部输入定理；本书只证明形式推论与结构保真要求。
- Prismatization 与 $F$-gauges 当前处于研究边界；不得与基础 prismatic cohomology 混写。

## 第二轮严格化路线

1. 把 [D_theorem_locator_index.md](D_theorem_locator_index.md) 中所有 `L2S` locator 升级为精确 section/theorem/page locator，并补齐仍处于 `L0/L1` 的 classical 条目。
2. 把 [E_label_ledger.md](E_label_ledger.md) 中的稳定编号扩展到每个新命题和练习。
3. 把 [F_nygaard_tate_twist_crosswalk.md](F_nygaard_tate_twist_crosswalk.md) 中 BMS2 syntomic/Tate twist 的 mod $p^r$、truncation、nearby cycles 变体补成 L3 公式。
4. 扩充 [B_examples_and_local_calculations.md](B_examples_and_local_calculations.md) 中 $q$-de Rham 与 Breuil-Kisin 例子的逐项计算。
5. 进行全书散文交叉引用到编号引用的替换。
6. 做最终出版校对：术语一致、中文句式统一、公式断行、链接和 locator 校验。

## 当前判定

本书当前达到“逐章教材收口草稿”状态：基础对象、主比较定理、积分理论、表示论接口、前沿边界和失败模式均已成体系，核心 prismatic/BMS2/prismatization 外部输入已具备源码级 locator。数学上还不能标为最终出版收口；主要缺口不是定义链，而是 `L2S` 到 `L3` 的 locator 转换、syntomic 变体细分和 classical comparison 源选择。
