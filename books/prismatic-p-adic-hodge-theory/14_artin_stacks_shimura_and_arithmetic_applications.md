# 第十四章：Artin stacks、Shimura varieties 与算术应用边界

## 本章目标

本章把 prismatic theory 的几何应用边界组织为四类：Artin stacks、Shimura varieties、Brauer groups 和 finite flat group schemes。这些方向展示 prismatic methods 的扩展能力，但本书只把它们作为应用边界或外部输入，不纳入基础定理链。

## 依赖前置知识

依赖第六章 prismatic $F$-crystals、第七章 syntomic structures、第八章 prismatization and $F$-gauges、第十二章 lattices。

## 14.1 Artin stacks

**研究边界 14.1.** Kubrak-Prikhodko 研究 Artin stacks 情形中的 integral $p$-adic Hodge theory。对 Hodge-proper stack，他们使用 prismatic cohomology 建立与 Raynaud generic fibre etale cohomology 相关的 $p$-adic Hodge theory，并得到 crystalline Galois representation 和 Breuil-Kisin module 描述。

**说明 14.2.** Stacky 情形的重要变化是 properness、cohomological finiteness 和 generic fibre comparison 的假设与 schemes 情形不同。正式引入该理论需要先建立 stacks 上的 prismatic cohomology 和 cohomological descent。

## 14.2 Shimura varieties

**研究边界 14.3.** Imai-Kato-Youcis 构造 abelian type Shimura varieties 的 prismatic realization functor，并发展 integral analogue of $D_{\mathrm{crys}}$。该方向把 prismatic theory 与 integral canonical models、Serre-Tate theory 和 local Shimura varieties 相连。

**警告 14.4.** Shimura varieties 的 prismatic realization 依赖 Shimura datum、level structure、integral model 和 group-theoretic tensors。不能把它简化为普通 smooth proper formal scheme 的 comparison theorem。

## 14.3 Brauer groups and operations

**研究边界 14.5.** Carmeli-Feng 构造 syntomic Steenrod algebra 并用于 Brauer groups 上的 arithmetic duality。他们引入 spectral syntomic cohomology 和 spectral prismatic $F$-gauges，说明 prismatic theory 可承载 cohomology operations。

**研究边界 14.6.** Ambrosi-Newton-Pagano 使用 prismatic cohomology 研究 wild Brauer classes，并把 Newton-above-Hodge 型结果用于 weak approximation 障碍。

**说明 14.7.** Brauer applications 同时涉及 syntomic operations、Hodge numbers、Newton polygons、evaluation maps 和 arithmetic fields。它们应放在应用边界，而不是基础 comparison theorem 部分。

## 14.4 Finite flat group schemes

**研究边界 14.8.** Mondal-Olsson 2026 年工作描述 smooth positive characteristic variety 上 finite flat height one group scheme 对应的 prismatic $F$-gauge，并恢复 crystalline Dieudonne module 和 flat cohomology 的相关结果。

**命题 14.9（Dieudonne 接口的最低要求）.** 若一个 prismatic statement 声称恢复 Dieudonne module 描述，则必须说明 group scheme 的高度、基底 characteristic、crystalline site 或 prismatic site、以及 Frobenius/Verschiebung 数据的对应。

**证明.** Dieudonne theory 的对象由 group scheme 类型和 Frobenius/Verschiebung 结构决定。Prismatic statement 只有在同时给出 site、height condition 和结构映射对应时，才足以恢复 classical Dieudonne module。证毕。

## 14.5 应用边界原则

**约定 14.10.** 本书处理应用边界时遵守三条规则：

1. 只使用一手文献和版本已核查预印本；
2. 不把应用定理倒用为 prismatic cohomology 的基础定义；
3. 对每个应用记录其额外 hypotheses，而不是只记录结论名称。

## 14.6 应用定理的假设表

**说明 14.11.** 应用章节不得只写“某某方向使用 prismatic cohomology”。一个可审查的应用条目至少要有如下表格。

| 方向 | 几何对象 | 额外结构 | 输出 | 不能省略的假设 |
| --- | --- | --- | --- | --- |
| Artin stacks | Hodge-proper stack | stacky descent | integral $p$-adic Hodge object | cohomological finiteness |
| Shimura varieties | integral model | tensors, level, group datum | prismatic realization | good reduction/local model hypotheses |
| Brauer groups | arithmetic variety | syntomic operations | duality/evaluation obstruction | field and cohomological range |
| finite flat groups | height one group scheme | Frobenius/Verschiebung | prismatic $F$-gauge | characteristic and height |

**命题 14.12.** 若一个应用结果没有说明几何对象和额外结构，则不能纳入本书的应用边界表。

**证明.** Prismatic methods 的输出依赖输入几何对象和额外结构。Artin stack、Shimura variety、Brauer class 和 finite flat group scheme 的比较对象不同；未说明输入类型时，目标 cohomology 或 representation 也无法确定。证毕。

## 本章小结

Artin stacks、Shimura varieties、Brauer groups 和 finite flat group schemes 说明 prismatic theory 已经成为 arithmetic geometry 的通用工具。但正式教材的基础链仍然是 $\delta$-环、prism、site、comparison theorem、integral modules 和 $F$-crystals；应用边界必须分层记录。

## 练习

**练习 14.1.** 说明 Artin stacks 情形相比 schemes 情形多出的至少两个技术问题。

**练习 14.2.** 解释为什么 Shimura varieties 的 prismatic realization 不能只由 smooth proper comparison theorem 推出。

**练习 14.3.** 按命题 14.9，列出 finite flat height one group scheme 与 prismatic $F$-gauge 比较所需的数据。
