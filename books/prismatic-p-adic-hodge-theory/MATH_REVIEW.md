# 数学审查记录

初次核查日期：2026-07-08。主线复核：2026-07-11。技术状态校准：2026-07-15。
状态：逐章教材收口草稿；可作为在线教材使用；derived completion、比较定理接口和正文核心 P0 numbered locators 已系统化，但尚未达到 `Math-Closed` 或 `Camera-Ready`。

本次 2026-07-15 校准未联网，未重新核验外部文献最新版本；这里只整理库内文本当前应如何诚实对外表述。

## 当前已固定的严格性边界

- 已固定素数 $p$、derived completion 口径、$\delta$-环和 Frobenius lift 符号。
- 已区分 bare prism 与 bounded prism；只有本书 site/comparison 主线按全局约定使用 bounded prisms。
- 已用 Koszul tower 定义 derived completion，并区分 $p$-、$I$-、$(p,I)$-completion、complete flatness 与 ordinary Tor-amplitude。
- 已把 Frobenius-semilinear maps 全部改写为类型明确的 linearizations，并区分 integral、after-$I$、after-$p$ 与 rational statements。
- 已区分 internal proof、external input theorem 和 frontier-only result。
- 已把 2025-2026 资料列入研究边界，不纳入基础定理链。

## 当前高风险点

- Hodge--Tate、de Rham、crystalline、finite-level etale、base change 与 Frobenius-isogeny statements 已分别绑定 Bhatt--Scholze 的 numbered theorems/corollaries；不得再合并成无类型的“comparison theorem”。
- Nygaard filtration 已绑定 Bhatt--Scholze Theorem 1.16 / Theorem 15.3；BMS2 syntomic fibre/products 及 characteristic-$p$/mixed-characteristic nearby cycles 已绑定 Theorems 1.12 (5), 1.15, 10.1。
- BMS1/BMS2 cohomology outputs 只给一般 BKF/BK modules；finite projective/free、effectivity 与 height 需要正文列出的 torsion 或 amplitude 假设。
- Classical Fontaine period rings 的 construction 本身很长，本书正式教材稿给出严谨定义口径、admissibility interface 和 comparison theorem interface，但仍不声称完整重建 Fontaine 理论。
- Classical comparison 的最终源选择仍未闭合；在完成 Fontaine/Faltings/Tsuji/Brinon-Conrad/Berger/Kedlaya-Liu 等来源之间的出版级选择前，不得把 classical comparison 章标为数学闭合。
- Prismatic $F$-crystals 与 crystalline lattice 的等价是外部输入定理；本书只证明形式推论与结构保真要求。
- Bhatt--Lurie preliminary material 只作为 preliminary/frontier 接口；当前 locator 不足以支撑出版级基础定理链。
- Nygaard/Tate twist normalization 已有交叉表，但跨文献 convention 仍需最终出版复核。
- Prismatization 与 $F$-gauges 当前处于研究边界；不得与基础 prismatic cohomology 混写。

## 第二轮严格化路线

1. 把 [D_theorem_locator_index.md](D_theorem_locator_index.md) 中正文仍依赖的 `L2S` 技术 locators 升级为精确 section/theorem/page locators，并补齐仍处于 `L0/L1` 的 classical 条目。
2. 按 [E_label_ledger.md](E_label_ledger.md) 维护后续新增命题和练习；本轮
   正式声明已登记。
3. 完成 [F_nygaard_tate_twist_crosswalk.md](F_nygaard_tate_twist_crosswalk.md) 中跨文献 Tate-twist normalization 的出版复核；relative Nygaard 与 BMS2 syntomic variants 已达 L3。
4. 扩充 [B_examples_and_local_calculations.md](B_examples_and_local_calculations.md) 中 $q$-de Rham 与 Breuil-Kisin 例子的逐项计算。
5. 进行全书散文交叉引用到编号引用的替换。
6. 做最终出版校对：术语一致、中文句式统一、公式断行、链接和 locator 校验。

## 当前判定

本书当前达到“逐章教材收口草稿”状态：基础对象、主比较定理、积分理论、表示论接口、前沿边界和失败模式均已成体系，正文核心 prismatic/Nygaard/BMS1/BMS2/$L\eta$/$F$-crystal 外部输入已具备 numbered locators。数学上还不能标为 `Math-Closed`，出版上不能标为 `Camera-Ready`；主要缺口不是定义链，而是 classical comparison 源选择、Bhatt--Lurie preliminary 接口、Nygaard/Tate twist normalization、非主线配套 locators 和出版 copy-editing。
