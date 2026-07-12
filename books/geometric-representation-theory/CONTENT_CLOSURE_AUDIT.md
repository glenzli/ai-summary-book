# 教材内容收口审查

审查日期：2026-07-08  
审查对象：《Geometric Representation Theory：几何、层与表示》  
审查口径：本文件只判断教材内容是否自身收口，不判断排版、页码级 locator、索引和最终出版格式。

## 1. 总判定

当前书稿达到“教材内容收口”：

1. 主体章节覆盖 geometric representation theory 的基础主线、经典主线、仿射/Satake/Langlands 接口、范畴化、辛几何、Coulomb branches、Hall/CoHA 和 quantum/canonical bases。
2. 每个主体章节都有定义链、核心构造、内部命题证明或外部输入标记、例子和练习。
3. 本书自行证明的命题均带证明块；大型定理以“外部输入定理”隔离。
4. 外部输入已经达到源级引用覆盖：每个主要外部输入簇能追溯到 `SOURCES.md`、附录 D 或 P0 locator 批次。
5. 前沿材料不进入基础证明链，而由第二十三章和附录 J 的验证流程管理。

因此，若目标是“作为一本研究生级教材的内容闭合稿”，当前版本可以收口。若目标是“出版终稿”，仍需页码级 locator、稳定 label、索引、术语统一和模型分拆。

## 2. 内容完备性

| 部分 | 覆盖内容 | 判定 |
| --- | --- | --- |
| 基础层 | reductive groups、root data、flag varieties、category $\mathcal O$、six functors、perverse sheaves | 收口 |
| 经典几何表示论 | Schubert/Hecke/KL、Springer、nilpotent orbits、D-modules、BB localization、BWB、Harish-Chandra bimodules | 收口 |
| 仿射与 Satake | affine Grassmannian、convolution、geometric Satake、affine flag、Iwahori-Hecke、Kac-Moody、GLanglands 接口 | 收口 |
| 范畴化与辛几何 | Soergel、KLR/Rouquier、quiver varieties、symplectic resolutions、category $\mathcal O$、symplectic duality | 收口 |
| Coulomb/Hall/Quantum | BFN Coulomb branches、Hall/CoHA/DT 接口、quantum groups、crystals、canonical bases | 收口 |
| 前沿管理 | 2024-2026 geometric Langlands、parity/torsion、categorical/spectral representation theory | 收口为边界章 |

“收口”在此处表示：章节本身不再只是目录或接口说明，而包含可读的数学对象、公式、证明或外部输入、例子和练习。

## 3. 证明完备性

内部证明采用两层标准：

1. 本书能在当前背景下证明的命题，正文给出证明。
2. 需要大型理论的结论，标为“外部输入定理”，不伪装为内部证明。

静态检查结果：

- `**命题 ...**` 条目均有后续 `**证明.**` 块。
- `外部输入定理` 均被定理账本或附录 D 管理。
- `边界说明` 和 `警告` 不作为后续证明的输入。

这满足教材内容层面的证明完备性。出版层面仍需把部分证明压缩、统一术语，并把外部输入的精确定理编号加入脚注或 locator。

## 4. 引用完整性

引用分三级：

1. **源级引用覆盖**：知道应查哪一篇论文、哪本书或哪组资料。
2. **定理包级 locator**：知道对应的 theorem package 和章节用途。
3. **出版级 locator**：给出定理编号、章节号、页码、版本或 arXiv 版本差异。

当前状态：

| 层级 | 状态 | 文件 |
| --- | --- | --- |
| 源级引用覆盖 | 已完成 | `SOURCES.md` |
| 定理包级 locator | 已完成 P0/P1/P2 覆盖 | `D_source_theorem_index.md`, `P0_REFERENCE_LOCATORS_BATCH_1.md` |
| 出版级 locator | 待出版校对 | 后续页码级 locator 批次 |

因此，内容层面的引用完整已经达到；出版层面的精确页码和定理编号尚未完成。

## 5. 不纳入“未收口”的事项

以下事项不影响当前内容收口判定：

1. 页码级 locator 的逐页复核列入独立的出版校对流程；
2. 部分外部定理尚未写出完整证明，因为它们本来作为外部输入使用；
3. geometric Langlands proof series、symplectic duality、critical CoHA 等前沿方向只作边界章或接口章；
4. 正特征、parity sheaves、real groups、microlocal character formulas 尚未扩成独立正文。

这些属于扩展卷、专题章或出版校对任务，而不是当前教材内容闭合的阻塞项。
