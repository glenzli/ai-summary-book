# 第八章：ACM、NASEM 与 VIM 的不同词典

一支独立团队下载作者归档的代码和数据，在另一座城市重算论文表 2，数值落入预定
容差。按 ACM 当前制品政策，这项工作可以属于 Reproducibility；按 NASEM 2019，
它也是 computational reproducibility；若把过程当作计量学测量，仅凭“另一城市”却
还不足以计算 VIM 所说的 measurement reproducibility，因为测量程序、操作者、系统
和变化条件尚未完整声明。

同一个英文词在前两套词典中碰巧出现，不表示定义相同；换成新数据或独立实现后，
分类又会分离。本章不制造一个统一中文译法，而是用同一份案例记录逐项填写三套词典
需要的坐标。机构定义都是外部输入，版本和一手链接见 [SOURCES.md](SOURCES.md)。

## 8.1 为什么一份报告会有三个名称

三个来源关注的坐标不同：

- ACM 主要按团队、实验设置与是否使用作者制品区分；
- NASEM 区分同一数据/代码的计算重算与新数据研究；
- VIM 定义测量精密度所处的条件集合。

中文“重复、复现、复制”无法单独保存这些坐标。严谨写法应保留英文术语和来源，并另写对象、变化条件与一致判据。

## 8.2 ACM 当前术语

**外部输入 8.A（ACM Artifact Review and Badging 当前口径）.** ACM 当前政策给出：

| ACM 术语 | 团队 | 实验设置 | 计算实验中的典型含义 |
|---|---|---|---|
| Repeatability | 同一团队 | 同一设置 | 研究者以声明精度可靠地重复自己的计算 |
| Reproducibility | 不同团队 | 同一设置 | 独立团队使用作者制品，在声明精度内得到一致结果 |
| Replicability | 不同团队 | 不同设置 | 独立团队使用完全独立开发的制品，在声明精度内得到一致结果 |

ACM 的简写“same experimental setup”不能只理解为同一物理地点。其详细文本对 Reproducibility 保持同一测量程序、测量系统和运行条件，同时允许在相同或不同地点由不同团队执行；Replicability 则改变测量系统并使用独立制品。

ACM 页面还明确说明，因与 NISO 协调，ACM 曾交换 reproducibility 与 replication 的既有标签，并更新先前徽章。因此引用旧论文中的 ACM 术语时必须核对当时版本，不能把当前词典追溯套用。

## 8.3 ACM 徽章不是同义词

ACM 当前 Results Validated 下区分：

- **Results Reproduced**：独立团队在后续研究中至少部分使用作者提供的制品，获得论文主要结果；
- **Results Replicated**：独立团队不使用作者提供的制品，独立获得主要结果。

政策明确表示不要求数值完全相同，而要求落在该类实验可接受的容差内，且差异不改变论文主要主张。这个“可接受容差”仍需按第一章和第九章具体化；徽章定义本身不给出跨领域统一数值阈值。

Artifacts Available 只说明作者制品置于可长期检索的公开归档；Artifacts Evaluated--Functional/Reusable 评价文档、完整性、可执行性或复用质量。它们都不自动等于 Results Reproduced/Replicated。

## 8.4 NASEM 2019 口径

**外部输入 8.B（NASEM 2019）.** 报告采用：

- **Reproducibility**：使用相同输入数据、计算步骤、方法、代码和分析条件，获得一致的计算结果；
- **Replicability**：多个研究各自取得新数据，针对同一科学问题获得一致结果。

因此，在 NASEM 口径下，用原数据和代码重算是 computational reproducibility；重新收集数据研究同一问题是 replicability。后者不要求实现或数据逐位相同，并且报告明确指出复制程度未必适合压成二元 pass/fail。

NASEM 的“consistent results”仍不是自解释对象。它可以指效应方向、估计区间、预测区间、预定等效界或模型比较结论，必须由研究合同另行定义。

## 8.5 VIM 第三版口径

**外部输入 8.C（VIM 3）.** VIM 先定义条件，再定义这些条件下的测量精密度：

- **repeatability condition of measurement（2.20）**：同一测量程序、同一操作者、同一测量系统、同一运行条件和地点，在短时间内对同一或相似对象重复测量；
- **measurement repeatability（2.21）**：在 repeatability conditions 下的 measurement precision；
- **reproducibility condition of measurement（2.24）**：不同地点、操作者和测量系统，对同一或相似对象重复测量；不同系统可以使用不同测量程序，并应尽可能声明哪些条件改变、哪些不变；
- **measurement reproducibility（2.25）**：在 reproducibility conditions 下的 measurement precision。

VIM 的 repeatability/reproducibility 是测量精密度概念，不是软件结果字节身份的通用别名。报告“reproducibility standard deviation”时，应说明测量模型、变化条件和离散量，而不是套用 ACM 团队/制品维度。

## 8.6 把同一份重算记录放进三套词典

考虑“另一团队用作者代码和数据在另一城市重算表 2”：

- ACM：若测量程序、系统和运行条件仍属于同一设置，则是 Reproducibility；地点可以不同；
- NASEM：因使用相同数据、代码和分析条件，是 computational reproducibility；
- VIM：只有当该计算被明确建模为测量，且规定了操作者、系统、地点等测量条件时，才讨论 repeatability/reproducibility precision。

这个例子恰好在 ACM 和 NASEM 中都出现 reproducibility 一词，但两者判定理由不同。换一个“同团队、全新数据”的场景，两套分类就会分离。

把案例写成记录后，分类依据更清楚：

| 字段 | 已知值 |
|---|---|
| 团队 | 与原作者不同的团队 B |
| 制品 | 作者归档的代码、容器和数据 |
| 数据 | 原论文数据，不是新收集数据 |
| 计算条件 | 同一测量程序、系统与声明的运行条件 |
| 地点 | 与原团队不同 |
| 判据 | 表 2 各目标量落入预先给定容差 |

ACM 的分类使用“团队、设置、作者制品”这些字段；NASEM 使用“数据、代码和计算步骤
是否相同”；VIM 若要介入，还需把对象建模为测量并报告变化条件下的精密度。若把
`制品` 改成团队 B 独立开发、把 `数据` 改成新收集数据，那么 ACM 的 Results
Replicated 与 NASEM 的 replicability 才可能同时成为候选；二者仍需各自的一致判据，
不能仅凭字段变化自动授予结论。

## 8.7 本书的规范写法

首次出现时采用以下格式：

> 按 NASEM（2019）的 computational reproducibility 口径，团队 B 使用论文归档的数据、代码和分析条件重算表 2；数值一致规则为合同 C-17。

或：

> 按 ACM 当前 Artifact Review and Badging 口径，本工作属于 Results Replicated 候选：独立团队未使用作者制品；主要主张的一致判据为预注册的等效界。

或：

> 按 VIM 3 的 reproducibility conditions，本测量改变地点、操作者和测量系统，保持被测量、校准链与分析规则不变；报告的是这些条件下的标准差。

后文可以简称，但简称只在同一已声明词典内有效。

## 8.8 术语不能替代统计与科学判断

无论采用哪套词典，“一致结果”都必须指出目标量与判决规则。下列句子均不充分：

- “两个研究都显著”；
- “置信区间有重叠”；
- “准确率只差一点”；
- “同一个 seed 跑通了”；
- “获得了 ACM 制品徽章”。

第九章将把差异检验、等效检验、置信区间和多重比较写成可审计规则。机构词典告诉我们在比较什么条件，不替代统计证据。

这份跨团队重算记录在 ACM 与 NASEM 中使用了相同英文标签，却依赖不同字段；VIM
还要求另一组测量条件。因而后文出现任何中文简称，都要同时保留来源和变化条件。
词典只告诉我们比较发生在哪些条件下，不能决定数值差异是否足够小；这个任务属于
下一章的统计模型、等效界与错误率。

## 练习

**练习 8.1.** 构造一个在 ACM 口径下为 Repeatability、在 NASEM 口径下却不能仅凭现有信息分类的场景，并说明缺少什么。

**练习 8.2.** 准确区分 ACM 的 Results Reproduced、Results Replicated、Artifacts Available 和 Artifacts Evaluated--Functional。

**练习 8.3.** 为一个跨实验室测量写出 VIM reproducibility conditions，列出改变条件与保持条件。

**练习 8.4.** 解释为什么“两个研究都显著”和“两个置信区间重叠”都不是充分的一致性定义。

**练习 8.5.** 为一篇计算论文写一句同时无 ACM/NASEM 术语歧义、且包含数值判据的复现声明。
