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

当前目标不是一次性达到 L3，而是先保证所有核心输入至少达到 L2，并标出 L3 缺口。若只定位到 arXiv 号，不能算出版级引用；若定位到讲次、章节、源码 label 或 theorem package，至少算 L2。

## 1. 主资料

| 代号 | 文献 | 当前 locator | 状态 |
| --- | --- | --- | --- |
| S26 | Peter Scholze, *Lectures on Condensed Mathematics*, arXiv:2605.03658 | Lectures I-XI；TeX source `Condensed.tex` labels `def:condensed`, `thm:niceabcat`, `def:solid`, `thm:specker`, `thm:solid`, `thm:solidtensor`, `def:analytic`, `prop:analyticnice`, `thm:solidAR`, `thm:globalization`, `thm:openduality` | L2 |
| CS26 | Dustin Clausen and Peter Scholze, *Condensed Mathematics and Complex Geometry*, arXiv:2605.11731 | Lectures I-XV；TeX source `Complex.tex` labels `thm:liquidmain`, `thm:holomorphicdefinitionredux`, `thm:GAGAabstract`, `affinoidgiveslocale`, `thm:lowershriek`, `boundarylessuppershriek`, `thm:grauert`, Serre Duality theorem, GAGA theorem, `GRR`, `thm:GHRRfinal` | L2 |
| A23 | Dagur Asgeirsson, *Towards solid abelian groups: A formal proof of Nöbeling's theorem*, arXiv:2309.07252 | Sections `sec:preliminaries`, `sec:theorem`, `sec:formalisation`; TeX labels `Nobeling`, `NobelingClosed`, `profinite-profinite` | L3 |
| ABKMT24 | Asgeirsson-Brasca-Kuhn-Mortarino Majno di Capriglio-Topaz, *Categorical Foundations of Formalized Condensed Mathematics*, arXiv:2407.12840 | Sections on regular/extensive/coherent topology and `back_to_condensed.tex`; labels `prop:regular_extensive_sheaf`, `prop:sheafEquiv`, `thm:condensed set equivalence` | L3 |

## 1.1 凝聚基础与站点比较定位

| 输入 | 本书编号 | 当前内部位置 | 外部文献 | 状态 | 出版级 locator |
| --- | --- | --- | --- | --- | --- |
| condensed set on profinite site | volume-1 3, 5, B | volume-1 3, 5, B | S26, ABKMT24 | L2 | S26 Lecture I `def:condensed`; ABKMT24 `condensed.tex` definition and `back_to_condensed.tex` comparison |
| finite jointly surjective / coherent topology sheaf condition | volume-1 1, 5 | volume-1 1, 5, B | S26, ABKMT24 | L3 | S26 Lecture I; ABKMT24 `coherentSheaves.tex` `prop:regular_extensive_sheaf` and `condensed.tex` sheaf characterization |
| ProFin / CHaus / Stonean sheaf equivalence | volume-1 5, B | volume-1 5, B; volume-4 2 | ABKMT24, S26 | L3 | ABKMT24 `back_to_condensed.tex` theorem `thm:condensed set equivalence`; S26 Lectures I-II for profinite and \(\kappa\)-small setup |

## 2. Scholze 核心输入定位

| 输入 | 本书编号 | 当前内部位置 | 外部文献 | 状态 | 出版级 locator |
| --- | --- | --- | --- | --- | --- |
| solidification existence | `INPUT_THEOREM_REGISTER.md` B.2, volume-2 D.1 | volume-2 V, Q, AA | S26 | L2 | S26 Lecture V `def:solid`, theorem `thm:solid`; Lecture VI proof of `thm:solid` |
| solid kernel tensor ideal | B.3, volume-2 D.2 | volume-2 W, Q, AA | S26 | L2 | S26 Lecture VI `cor:solidproperties`, theorem `thm:solidtensor`; tensor ideal proof sits in proof of symmetric monoidality |
| profinite measure tensor formula | B.4, volume-2 D.3 | volume-2 W, Q, AA | S26, A23 | L3 | S26 Lecture VI `prop:tensorinfproducts`; A23 theorem `Nobeling` and formal proof section |
| analytic ring localization | C.1, volume-2 D.4 | volume-2 X, R, AA | S26 | L2 | S26 Lecture VII `def:analytic`, `prop:analyticnice`, `prop:functoriality`, `prop:exanalytic` |
| Huber pair rational descent | C.4, volume-2 D.7 | volume-2 Y, R, AA | S26 | L2 | S26 Lectures IX-X: discrete Huber pair definition, rational subsets, `prop:locfullyfaithful`, `prop:locbasechange`, `thm:globalization` |
| \(p\)-liquid analytic ring | C.3, volume-2 D.5 | volume-2 S, Z, AA | S26, CS26 | L2 | S26 Lecture VII theorem on \((\mathbb R,\mathcal M_{<p})\); CS26 Lectures II-III `thm:qspliquid`, `thm:liquidmain` |
| liquid realization | C.2, volume-2 D.6 | volume-2 Z, S, AA | CS26 | L2 | CS26 Lectures II-IV: \(p\)-liquid objects, flatness/tensor calculations and examples；精确 realization 适用子范畴仍需 source theorem number 或正式约定 |
| \(f_!\), projection formula, \(f^!\) | volume-2 D.9 | volume-2 F, L, AA | S26, CS26 | L2 | S26 Lecture VIII appendix `thm:solidAR`; Lecture XI `thm:openduality`, `def:flowershriek`; CS26 Lecture XII `thm:lowershriek`, `boundarylessuppershriek` |

## 3. Clausen-Scholze 复几何输入定位

| 输入 | 本书编号 | 当前内部位置 | 外部文献 | 状态 | 出版级 locator |
| --- | --- | --- | --- | --- | --- |
| condensed/analytic complex geometry 建模 | volume-2 AA.12, volume-3 AR.1 | volume-3 B, AQ, AR | CS26 | L2 | CS26 Lectures IX-XI: coherent sheaves, compact Stein descent, affinoid/categorified locale construction `affinoidgiveslocale` |
| Dolbeault-liquid comparison | volume-3 AR.2 | volume-2 Z, volume-3 N, R, AR | CS26 + classical Dolbeault | L2 | CS26 Lectures V-X explain holomorphic functions, analytic structure sheaves and coherent sheaves in liquid categories; classical Dolbeault lemma remains INPUT D.1 |
| coherent cohomology finite-dimensionality | volume-3 AR.3 | volume-3 L, M, X, AC, AN, AQ | CS26 + Grauert/Hodge | L3 | CS26 Lecture XII theorem `thm:grauert` and preceding proper pushforward package；classical Grauert/Hodge locator 已到 L2，仍需 theorem/page refinement |
| Serre/Grothendieck duality | INPUT D.5, volume-3 AR.4 | volume-3 J, O, AA, AD, AQ | CS26 + classical duality | L3 | CS26 Lecture XIII proposition `prop:serreduality0` and theorem “Serre Duality”; S26 Lecture XI `thm:openduality` for solid coherent duality analogue |
| GAGA | INPUT D.6, volume-3 AR.5 | volume-3 Q, Y, AI, AO, AQ | CS26 + Serre/Grothendieck | L3 | CS26 Lectures VI-VII `thm:GAGAabstract`; Lecture XIII theorem “GAGA” |
| HRR/GRR | INPUT D.7, volume-3 AR.6 | volume-3 P, U, AE, AK, AP, AQ | CS26 + classical GRR | L3 | CS26 Lecture XIV theorem `GRR`; Lecture XV theorem `thm:GHRRfinal`, propositions `prop:GHRRformal`, `prop:GHRRtodd`; classical GRR foundations still external |
| six functor interface | volume-3 AR.7 | volume-2 F, L; volume-3 AJ, AR | CS26, S26 | L2 | CS26 Lecture XII `thm:lowershriek`, `boundarylessuppershriek`; Lecture XIII proper pushforward/base-change/projection formula discussion; S26 Lecture XI `def:flowershriek` |

## 4. 经典输入定位

| 输入 | 本书编号 | 当前内部位置 | 资料类型 | 状态 | 出版级 locator |
| --- | --- | --- | --- | --- | --- |
| Boolean prime ideal theorem | A.1 | volume-1 N | set-theoretic topology | L1 | Johnstone, *Stone Spaces*, Stone duality/ultrafilter background；仍需补 theorem/page |
| Sikorski extension theorem | A.2 | volume-1 O | Boolean algebra | L1 | Sikorski, *Boolean Algebras*, extension theorem；仍需补 edition theorem/page |
| Gleason lifting theorem | A.3 | volume-1 J, O | compact Hausdorff topology | L3 | Gleason, *Projective topological spaces*, Illinois J. Math. 2 (1958), Theorem 2.5 |
| Dolbeault lemma | D.1 | volume-3 N, R | several complex variables | L2 | Wells, *Differential Analysis on Complex Manifolds*, Dolbeault theorem chapter；Huybrechts, *Complex Geometry*, Dolbeault cohomology section |
| Cartan A/B | D.2 | volume-3 V, AB, AG, AH | several complex variables | L2 | Cartan, *Varietes analytiques complexes et cohomologie*, Bruxelles colloquium；Grauert-Remmert, *Coherent Analytic Sheaves*, Cartan theorems chapter |
| Grauert direct image | D.3 | volume-3 AC, AN | complex analytic geometry | L2 | Grauert, *Ein Theorem der analytischen Garbentheorie und Modulraeume komplexer Strukturen*, Publ. Math. IHES 5 (1960), main theorem/Hauptsatz；CS26 Lecture XII `thm:grauert` |
| Hodge-Fredholm theorem | D.4 | volume-2 P, Z; volume-3 Z, AA | elliptic theory | L2 | Wells, *Differential Analysis on Complex Manifolds*, elliptic/Hodge chapters；Huybrechts, *Complex Geometry*, Hodge theorem section |
| Serre duality | D.5 | volume-3 J, O, AD | algebraic/analytic geometry | L3 | Serre, *Un theoreme de dualite*, Comment. Math. Helv. 29 (1955), 9-26；Hartshorne, *Algebraic Geometry*, Theorem III.7.6；Huybrechts, *Complex Geometry*, Theorem 3.12 |
| GAGA | D.6 | volume-3 Q, Y, AI, AO | algebraic geometry | L3 | Serre, *Geometrie algebrique et geometrie analytique*, Ann. Inst. Fourier 6 (1956), 1-42；Hartshorne, Appendix B, Theorems 3.1-3.2 |
| GRR | D.7 | volume-3 AE, AK, AP | intersection theory / K-theory | L2 | Borel-Serre, *Le theoreme de Riemann-Roch*, Bull. Soc. Math. France 86 (1958), 97-136；SGA 6, LNM 225, Expose III；Fulton, *Intersection Theory*, Riemann-Roch chapter |

## 5. 校对规则

出版级引用定位时采用以下规则：

1. 正文可以引用本书编号，例如“由输入定理 D.4”。
2. 输入登记表必须引用外部文献代号，例如 S26 或 CS26。
3. 每个 L1 条目后续至少应提升到 L2；当前凝聚主线 L1 已清零，经典输入中只剩 Boolean prime ideal theorem 与 Sikorski extension theorem 仍低于 L2。
4. 所有“标准事实”应降为以下两类之一：书内证明，或本台账中的外部 locator。
5. 若找不到精确 locator，正文不得把该结论写成“已证”；只能写成输入定理或证明路线。
