# 第十五章：正式教材收口、错误模式与开放问题

## 本章目标

本章把全书作为 Prismatic / p-adic Hodge Theory 自身进行收口：列出已闭合定义链、不可混用边界、常见错误模式、外部输入定位需求和开放问题。它不新增基础定理，而是保证前十四章的数学对象和证明依赖可以被审查。

## 15.1 已闭合的定义链

**命题 15.1.** 本书的基础定义链在内部闭合：$\delta$-环、Frobenius lift、Cartier divisor ideal、bounded prism、prismatic site、structure sheaf、prismatic cohomology、Hodge-Tate/de Rham specialization、prismatic crystal 和 prismatic $F$-crystal 均已在进入使用前定义。

**证明.** 第一章定义 $\delta$-环和 Frobenius lift；第二章定义 prism、bounded prism、prismatic site 和 $R\Gamma_\Delta$；第三章定义 specialization；第六章定义 prismatic crystals 和 $F$-crystals；第七章定义 Nygaard/syntomic 的工作框架。后续章节使用这些对象时均引用前置章节。证毕。

## 15.2 外部输入链

**约定 15.2.** 以下内容在本书中不可作为内部证明结论使用：

- perfectoid rings 与 perfect prisms 的等价；
- prismatic comparison theorem；
- BMS integral comparison；
- Fontaine-Faltings-Tsuji classical comparison；
- prismatic $F$-crystals 与 crystalline lattices 的范畴等价；
- Bhatt-Lurie prismatization 主解释；
- 2025-2026 研究边界预印本中的新定理。

这些内容必须引用 [D_theorem_locator_index.md](D_theorem_locator_index.md) 或后续精确 locator。

## 15.3 常见错误模式

**错误模式 15.3.** 把 $R\Gamma_\Delta(X/A)$ 直接等同于 etale cohomology。

**修正 15.4.** Etale comparison 需要 perfect prism、invert $I$、Frobenius fixed construction 和 modulo $p^n$ 或 inverse limit。

**错误模式 15.5.** 把 Hodge-Tate specialization 的 conjugate filtration 当成 de Rham Hodge filtration。

**修正 15.6.** 前者位于 $\overline\Delta_{X/A}$，后者位于 $\phi_A^\ast R\Gamma_\Delta(X/A)\otimes_A^L A/I\simeq R\Gamma_{\mathrm{dR}}$。

**错误模式 15.7.** 把 Breuil-Kisin module、Breuil-Kisin-Fargues module、filtered $\varphi$-module 和 prismatic $F$-crystal 混用。

**修正 15.8.** 这些对象处于不同底环和范畴中，由 comparison 或 classification theorem 连接。

**错误模式 15.9.** 在 Nygaard/syntomic 公式中省略 twist convention。

**修正 15.10.** 必须说明 $\{i\}$ 是 $(I/I^2)^{\otimes i}$、其 dual，还是经 orientation 后的 $d^i$ 表示。

## 15.4 正式教材剩余工作

**说明 15.11.** 本书达到正式教材扩展稿后，剩余工作分为四类：

1. locator：给所有外部输入定理补 section/theorem/page；
2. numbering：把所有正式陈述登记到稳定 label ledger；
3. convention：核对 Nygaard、Tate twist、Frobenius pullback 和 filtration indexing；
4. production：统一术语、语气、公式断行、图表和参考格式。

## 15.5 开放问题目录

**研究边界 15.12.** 以下问题代表本书后续可扩展方向：

- prismatic cohomology with coefficients 的教材化定义和 comparison theorem；
- prismatic non-abelian Hodge theory 的对象范畴；
- spectral syntomic operations 的基础教材化；
- prismatic $F$-gauges 与 displays、Dieudonne theory 的统一口径；
- Artin stacks 和 derived stacks 上的 prismatic theory；
- Shimura varieties 的 prismatic realization 与 integral Langlands-type structures。

## 15.6 逐章收口判据

**定义 15.13.** 称一章达到教材收口，如果它满足：

1. 有明确目标和前置知识；
2. 所有核心术语在使用前定义；
3. 非平凡命题有证明、证明草图或外部输入标记；
4. 至少有一个例子、计算、结构表或错误边界；
5. 章末有小结和练习；
6. 外部输入可在 locator 索引中追踪。

**命题 15.14.** 按定义 15.13，本书第 0-15 章当前达到教材收口草稿标准，但未达到 camera-ready 标准。

**证明.** 每章均有目标、前置知识、小结和练习。核心术语由正文或附录 G-K 定义。大型定理均标为外部输入并登记在附录 D 或资料源中。第 0-15 章均包含命题、证明、例子、结构表或错误边界。尚未达到 camera-ready 的原因是 locator 多数仍为 L1，且交叉引用和排版未做最终出版处理。证毕。

**说明 15.15.** “教材收口草稿”不同于“最终数学闭包”：前者要求内部教学链完整，后者还要求外部输入精确定位、符号 convention 全部逐源核对。

## 本章小结

本书当前已经具有正式教材的主要结构：定义链、正文链、外部输入链、边界链和审查链。数学上可继续向最终收口推进，但最终出版仍需要 locator、编号、符号和 production 四类工作。

## 练习

**练习 15.1.** 从错误模式 15.3、15.5、15.7、15.9 中任选一个，写出一个错误证明并修正。

**练习 15.2.** 给定一个新的 prismatic 预印本，按约定 14.10 和说明 15.11 判断它能否进入正文基础定理链。

**练习 15.3.** 为定理 11.8 设计一个 locator 表项，至少包括 source、版本、定理号、hypotheses 和本书用途。
