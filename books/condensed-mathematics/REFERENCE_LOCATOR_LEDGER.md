# 引用定位台账

作者：Dr. Stochastic Parrot

## 0. 目标

本文件记录外部输入定理的引用定位状态。它不是参考文献列表；参考文献入口仍在各卷 `SOURCES.md`。本台账用于出版级校对时追踪：

1. 每个输入定理应定位到哪一类资料；
2. 当前定位颗粒度是否足够；
3. 后续还需要补哪一级 locator。

状态分四级：

| 状态 | 含义 |
| --- | --- |
| L0 | 只有主题和作者来源 |
| L1 | 已定位到具体文献或 arXiv 号 |
| L2 | 已定位到章节、讲、附录或 theorem package |
| L3 | 已定位到精确 theorem/lemma/proposition 编号或页码 |

当前目标不是一次性达到 L3，而是先保证所有核心输入至少达到 L1，并标出 L2/L3 缺口。

## 1. 主资料

| 代号 | 文献 | 当前 locator | 状态 |
| --- | --- | --- | --- |
| S26 | Peter Scholze, *Lectures on Condensed Mathematics*, arXiv:2605.03658 | 第二卷 `SOURCES.md` | L1 |
| CS26 | Dustin Clausen and Peter Scholze, *Condensed Mathematics and Complex Geometry*, arXiv:2605.11731 | 第三卷 `SOURCES.md` | L1 |
| A23 | Dagur Asgeirsson, *Towards solid abelian groups: A formal proof of Nöbeling's theorem*, arXiv:2309.07252 | 第二卷 `SOURCES.md` | L1 |
| ABKMT24 | Asgeirsson-Brasca-Kuhn-Mortarino Majno di Capriglio-Topaz, *Categorical Foundations of Formalized Condensed Mathematics*, arXiv:2407.12840 | 第二卷 `SOURCES.md` | L1 |

## 2. Scholze 核心输入定位

| 输入 | 本书编号 | 当前内部位置 | 外部文献 | 状态 | 下一步 |
| --- | --- | --- | --- | --- | --- |
| solidification existence | `INPUT_THEOREM_REGISTER.md` B.2, volume-2 D.1 | volume-2 V, Q, AA | S26 | L1 | 定位到 S26 的 solid abelian groups / solidification 具体讲次与定理 |
| solid kernel tensor ideal | B.3, volume-2 D.2 | volume-2 W, Q, AA | S26 | L1 | 定位到 S26 的 solid tensor product 构造与 kernel 证明 |
| profinite measure tensor formula | B.4, volume-2 D.3 | volume-2 W, Q, AA | S26, A23 | L1 | 定位测度对象公式和 Nöbeling 使用位置 |
| analytic ring localization | C.1, volume-2 D.4 | volume-2 X, R, AA | S26 | L1 | 定位 analytic ring 定义、localization 与 tensor compatibility |
| Huber pair rational descent | C.4, volume-2 D.7 | volume-2 Y, R, AA | S26 | L1 | 定位 rational localization、rational acyclicity 和 Čech descent |
| \(p\)-liquid analytic ring | C.3, volume-2 D.5 | volume-2 S, Z, AA | S26 | L1 | 定位 \(p\)-liquid measure theory 的定义与 analytic ring 验证 |
| liquid realization | C.2, volume-2 D.6 | volume-2 Z, S, AA | S26, CS26 | L1 | 定位 realization functor、Hom 判别和 exactness 范围 |
| \(f_!\), projection formula, \(f^!\) | volume-2 D.9 | volume-2 F, L, AA | S26, CS26 | L1 | 定位 finite type affine \(f_!\) 和相干对偶讲次 |

## 3. Clausen-Scholze 复几何输入定位

| 输入 | 本书编号 | 当前内部位置 | 外部文献 | 状态 | 下一步 |
| --- | --- | --- | --- | --- | --- |
| condensed/analytic complex geometry 建模 | volume-2 AA.12, volume-3 AR.1 | volume-3 B, AQ, AR | CS26 | L1 | 定位 CS26 中复解析对象、相干层和 analytic/liquid model 的定理 |
| Dolbeault-liquid comparison | volume-3 AR.2 | volume-2 Z, volume-3 N, R, AR | CS26 + classical Dolbeault | L1 | 分开定位 classical Dolbeault lemma 与 CS realization compatibility |
| coherent cohomology finite-dimensionality | volume-3 AR.3 | volume-3 L, M, X, AC, AN, AQ | CS26 + Grauert/Hodge | L1 | 区分 Clausen-Scholze compactness statement、Grauert、Hodge-Fredholm |
| Serre/Grothendieck duality | INPUT D.5, volume-3 AR.4 | volume-3 J, O, AA, AD, AQ | CS26 + classical duality | L1 | 定位 trace theorem、dualizing complex 与 \(f^!\) 相容 |
| GAGA | INPUT D.6, volume-3 AR.5 | volume-3 Q, Y, AI, AO, AQ | CS26 + Serre/Grothendieck | L1 | 定位 algebraic/analytic coherent comparison 与 formal GAGA 输入 |
| HRR/GRR | INPUT D.7, volume-3 AR.6 | volume-3 P, U, AE, AK, AP, AQ | CS26 + classical GRR | L1 | 定位 characteristic class formula 和 GRR 基本因子 |
| six functor interface | volume-3 AR.7 | volume-2 F, L; volume-3 AJ, AR | CS26 | L1 | 定位 analytic six-functor package 范围和未完成边界 |

## 4. 经典输入定位

| 输入 | 本书编号 | 当前内部位置 | 资料类型 | 状态 | 下一步 |
| --- | --- | --- | --- | --- | --- |
| Boolean prime ideal theorem | A.1 | volume-1 N | set-theoretic topology | L0 | 补标准集合论/Stone duality 引用 |
| Sikorski extension theorem | A.2 | volume-1 O | Boolean algebra | L0 | 补 Sikorski theorem 精确引用 |
| Gleason lifting theorem | A.3 | volume-1 J, O | compact Hausdorff topology | L0 | 补 Gleason projective cover 精确引用 |
| Cartan A/B | D.2 | volume-3 V, AB, AG, AH | several complex variables | L0 | 补 Cartan/Oka/Stein 教材引用 |
| Grauert direct image | D.3 | volume-3 AC, AN | complex analytic geometry | L0 | 补 Grauert theorem 精确引用 |
| Hodge-Fredholm theorem | D.4 | volume-2 P, Z; volume-3 Z, AA | elliptic theory | L0 | 补 elliptic parametrix / Hodge theorem 引用 |
| Serre duality | D.5 | volume-3 J, O, AD | algebraic/analytic geometry | L0 | 补 classical Serre duality 与 Grothendieck duality 引用 |
| GAGA | D.6 | volume-3 Q, Y, AI, AO | algebraic geometry | L0 | 补 Serre GAGA 与 Grothendieck existence 引用 |
| GRR | D.7 | volume-3 AE, AK, AP | intersection theory / K-theory | L0 | 补 GRR 和 localized Chern character 引用 |

## 5. 校对规则

出版级引用定位时采用以下规则：

1. 正文可以引用本书编号，例如“由输入定理 D.4”。
2. 输入登记表必须引用外部文献代号，例如 S26 或 CS26。
3. 每个 L1 条目后续至少应提升到 L2。
4. 所有“标准事实”应降为以下两类之一：书内证明，或本台账中的外部 locator。
5. 若找不到精确 locator，正文不得把该结论写成“已证”；只能写成输入定理或证明路线。
