# 资料源账本

本账本记录本书使用的主要资料源、用途和当前定位状态。正文不得把未定位或未核查的近期结果写成无条件定理。

## Infinity-category、presentability 与稳定范畴基础

- Jacob Lurie, *Higher Topos Theory*, Annals of Mathematics Studies 170,
  Princeton University Press, 2009。
  - 用途：presheaf free cocompletion、presentable adjoint functor theorem、
    small-generated accessible localization、higher sheafification、
    compact generation 与 Ind-completion。
  - 精确定位：Theorem 5.1.5.6；Corollary 5.1.5.8；Corollary 5.5.2.9；
    Propositions 5.3.5.11、5.5.4.15、5.5.4.20、5.5.7.8、6.2.2.7。
  - 定位状态：located；见 `P0_REFERENCE_LOCATORS_BATCH_2.md`。

- Jacob Lurie, *Higher Algebra*, 2017-09-18 version。
  - 用途：stable infinity-category、exact functors、triangulated shadow、
    suspension stability criterion、stable compactness。
  - 精确定位：Theorem 1.1.2.14；Remark 1.1.2.15；Proposition 1.1.4.1；
    Corollary 1.4.2.27；Proposition 1.4.4.1。
  - 定位状态：located；见 `P0_REFERENCE_LOCATORS_BATCH_2.md`。

## 基础 motivic homotopy

- Fabien Morel, Vladimir Voevodsky, "A1-homotopy theory of schemes", Publications Mathematiques de l'IHES 90 (1999), 45-143.
  - 用途：Nisnevich site、simplicial sheaves、A1-localization、unstable motivic homotopy、homotopy purity 的 foundational source。
  - 精确定位：homotopy purity 为 Section 3, Theorem 2.23。
  - 定位状态：located；其他基础结果仍按章节继续补 locator。

- Vladimir Voevodsky, "A1-homotopy theory", ICM 1998 proceedings.
  - 用途：历史口径、核心构造和应用背景。
  - 定位状态：P1。

- Fabien Morel, "A1-Algebraic Topology over a Field", Lecture Notes in Mathematics 2052, Springer, 2012.
  - 用途：域上 A1-代数拓扑、Morel connectivity、homotopy sheaves、Milnor-Witt K-theory。
  - 精确定位：Corollary 6.43（稳定区间 sphere mapping groups）；Lemma
    3.10（`K_0^{MW}(k)\cong GW(k)`）；全书默认 perfect field。
  - 稳定链接：`https://doi.org/10.1007/978-3-642-29514-0`；可检索 PDF：
    `https://www.mahalex.net/teaching/seminars/morel/Morel.pdf`。
  - 定位状态：located；第十八章 P0 使用，见 batch 3。

- J. F. Jardine, motivic symmetric spectra 相关论文。
  - 用途：stable motivic model categories 与 spectra 模型。
  - 定位状态：P1。

- Marco Robalo, "Noncommutative Motives I: A Universal Characterization of
  the Motivic Stable Homotopy Theory of Schemes", arXiv:1206.3645.
  - 用途：presentable symmetric monoidal object-inversion、3-symmetric
    stabilization、symmetric spectra 模型比较。
  - 精确定位：Proposition 4.10；Corollary 4.24；Theorem 4.29。
  - 定位状态：located；见 `P0_REFERENCE_LOCATORS_BATCH_2.md`。

## 六操作和 motives

- Joseph Ayoub, "Les six operations de Grothendieck et le formalisme des cycles evanescents dans le monde motivique", Asterisque 314/315, 2007/2008.
  - 用途：motivic 六操作、nearby cycles、stable motivic formalism。
  - 定位状态：P0，需补卷册定理 locator。

- Denis-Charles Cisinski, Frederic Deglise, "Triangulated Categories of Mixed Motives", Springer, 2019.
  - 用途：motivic triangulated categories、six functors、absolute purity、motivic sheaves 和 mixed motives。
  - 定位状态：P0，需补定理号。

- Brad Drew, Martin Gallauer, "The universal six-functor formalism", arXiv:2009.13610.
  - 用途：stable A1-homotopy theory 的 universal coefficient system 和六操作普遍性质。
  - 核查：arXiv 摘要页显示作者、标题和 2020-09-28 日期，并说明 Morel-Voevodsky stable A1-homotopy theory gives the universal coefficient system for Grothendieck six operations。
  - 精确定位：Theorem 7.14；Proposition 7.13；Theorem 7.3。
  - 定位状态：located；注意 universal coefficient-system 结论不单独替代
    operation-specific base-change/projection/purity theorems。

- Marc Hoyois, "The six operations in equivariant motivic homotopy theory",
  Advances in Mathematics 305 (2017), 197-279。
  - 用途：本书取 trivial-group specialization，固定 motivic spectra 的
    compact generation、six operations、proper/smooth/base-change/
    projection/localization/continuity 精确 package。
  - 精确定位：Theorem 1.1；Propositions 3.15、4.2、6.4；Lemma 6.3；
    Corollaries 6.7、6.10、6.11、6.13、6.19；Theorem 6.18。另见 2018
    corrigendum。
  - 定位状态：located；见 `P0_REFERENCE_LOCATORS_BATCH_2.md`。

- Martin Gallauer, "An introduction to six-functor formalism".
  - 用途：six-functor formalism 的教学性入口和术语核对。
  - 定位状态：P2，不作为核心外部输入。

## 纯性、基本类、bivariant theory

- Frederic Deglise, Fangzhou Jin, Adeel A. Khan, "Fundamental classes in motivic homotopy theory", arXiv:1805.05920.
  - 用途：fundamental classes、Gysin maps、specialization、excess intersection、Euler classes、bivariant theory。
  - 核查：arXiv 摘要页显示作者、标题和 2018-05-15 日期；摘要说明在 motivic homotopy 中构造基本类并导出 Fulton-MacPherson 型 bivariant theory。
  - 精确定位：Example 2.3.10；Proposition 2.5.4；Definition 3.2.5；
    Theorem 3.3.2；Paragraph 3.3.3；Propositions 3.3.4、4.2.2；Theorems
    4.1.4、4.2.1；Paragraph 4.3.1；Definitions 4.3.7、4.3.11。
  - 层级边界：Remark 2.5.5 只把复合数据组织为
    `\operatorname{Ho}(\mathbf{SH})` 值逆变伪函子间的自然变换；transverse
    base change 是 Proposition 2.5.4(ii) 的交换方块。该文明确未完成
    infinity-category 层增强；本书不从这些三角影子推断 higher coherence。
  - Excess 边界：Propositions 3.3.4、4.2.2 要求 Cartesian square 中原
    morphism 与拉回 morphism 都是 smoothable lci s-morphisms，并使用
    Paragraph 3.3.3 的 excess bundle；4.2.2 的系数对象须有 unital
    associative commutative multiplication，proper push-pull 公式另要求两条
    竖边 proper。
  - 定位状态：located；第六章 purity morphism 与第十六章使用。

- Frederic Deglise, "Bivariant theories in motivic stable homotopy", arXiv:1705.01528.
  - 用途：bivariant theory、global complete intersection morphisms、Riemann-Roch 和 duality。
  - 定位状态：P1。

## Motivic cohomology、motives、K-theory、cobordism 和 slices

- Vladimir Voevodsky, "Motivic Eilenberg-MacLane spaces", arXiv:0805.4432.
  - 用途：`H\mathbb Z`、motivic operations、Eilenberg-Mac Lane objects。
  - 核查：arXiv 摘要页显示作者、标题和 2008-05-28 日期。
  - 定位状态：P0。

- Markus Spitzweck, "A commutative P1-spectrum representing motivic cohomology over Dedekind domains", arXiv:1207.4078.
  - 用途：一般基上的 highly structured `H\mathbb Z`、six functors、Hopkins-Morel 推广。
  - 精确定位：Theorem 7.18 与 Corollary 7.19；在
    mixed-characteristic Dedekind base 上对 smooth test schemes 表示
    cycle-complex motivic cohomology。Base change 见 Theorem 8.25；域上
    比较见 Theorem 6.7 and Lemma 8.23。
  - 稳定链接：`https://arxiv.org/abs/1207.4078`。
  - 定位状态：located；第九章 P0 使用，见 batch 3。

- Oliver Röndigs, Paul Arne Ostvaer, "Modules over motivic cohomology",
  Advances in Mathematics 219 (2008), 689--727。
  - 用途：characteristic zero field 上 Voevodsky big motives 与
    `H\mathbb Z`-modules 的 monoidal triangulated comparison。
  - 精确定位：Theorem 1.1；这是 model/homotopy-category 层面的比较。
  - 稳定链接：`https://doi.org/10.1016/j.aim.2008.05.013`。
  - 定位状态：located；第十、十四章 P0 使用，见 batch 3。

- Elden Elmanto, Hakon Kolderup, "On Modules Over Motivic Ring Spectra", arXiv:1708.05651.
  - 用途：motivic ring spectra module categories 的 monadic
    characterization，以及反演指数特征后 `DM` 比较的稳定
    infinity-categorical 口径。
  - 精确定位：Theorem 5.2；finite-correspondence application Corollary
    5.3。
  - 稳定链接：`https://arxiv.org/abs/1708.05651`。
  - 定位状态：located；第十、十四章 P0 使用，见 batch 3。

- Oliver Röndigs, Markus Spitzweck, Paul Arne Østvær, "Motivic strict ring models for K-theory", arXiv:0907.4121.
  - 用途：`KGL` strict ring models 和 K-theory 表示性。
  - 精确定位：Lemma 2.5；Theorems 3.6 and 4.1；Noetherian
    finite-dimensional base model-category 口径。
  - 稳定链接：`https://arxiv.org/abs/0907.4121`。
  - 定位状态：located；严格交换模型来源，见 batch 3。

- Denis-Charles Cisinski, "Descente par eclatements en K-theorie invariante
  par homotopie", Annals of Mathematics 177 (2013), 425--448。
  - 用途：`KGL` 表示 `KH` 与 `KH` 的 cdh descent。
  - 精确定位：全文 Noetherian finite-Krull-dimensional 约定；Theorem 2.20
    与 Theorem 3.9。
  - 稳定链接：`https://doi.org/10.4007/annals.2013.177.2.2`；作者 PDF：
    `https://www.math.univ-toulouse.fr/~dcisinsk/KHdescente.pdf`。
  - 定位状态：located；第十一章 P0 使用，见 batch 3。

- Charles Weibel, *The K-book: An Introduction to Algebraic K-theory*,
  Chapter IV。
  - 用途：regular Noetherian 情形的 `K\simeq KH`。
  - 精确定位：Corollary IV.12.3.1（rings）；Lemma IV.12.8(3)
    （quasi-projective regular schemes）。
  - 稳定链接：`https://sites.math.rutgers.edu/~weibel/Kbook/Kbook.IV.pdf`。
  - 定位状态：located；第十一章 P0 使用，见 batch 3。

- I. Panin, K. Pimenov, O. Röndigs, "A universality theorem for Voevodsky's algebraic cobordism spectrum", arXiv:0709.4116.
  - 用途：`MGL` orientation 泛性质。
  - 精确定位：Theorem 2.3.1；域上 motivic homotopy category 中
    commutative-monoid map sets 与 orientations 的双射。
  - 稳定链接：`https://arxiv.org/abs/0709.4116`。
  - 定位状态：located；不作任意基的 mapping-space equivalence，见 batch 3。

- Marc Hoyois, "From algebraic cobordism to motivic cohomology", arXiv:1210.7182.
  - 用途：Hopkins-Morel 型定理、`MGL/(a_i)\to H\mathbb Z` 比较。
  - 精确定位：Theorem 7.12；base essentially smooth over a field of
    characteristic exponent `c`，并反演 `c`。
  - 稳定链接：`https://arxiv.org/abs/1210.7182`；
    `https://doi.org/10.1515/crelle-2013-0038`。
  - 定位状态：located；第十二章 P0 使用，见 batch 3。

- Vladimir Voevodsky, "On the zero slice of the sphere spectrum", arXiv:math/0301013.
  - 用途：zero slice、sphere spectrum 与 motivic Eilenberg-Mac Lane spectrum 的联系。
  - 精确定位：Theorem 6.6（characteristic zero field 上
    `s_0(1)=H\mathbb Z`）；Introduction pp.106--107（slices 的
    `H\mathbb Z`-module 结构）。
  - 稳定链接：`https://arxiv.org/abs/math/0301013`。
  - 定位状态：located；第十三章 P0 使用，见 batch 3。

## Framed transfers、infinite loop spaces 与 norms

- Elden Elmanto, Marc Hoyois, Adeel A. Khan, Vladimir Sosnilo, Maria Yakerson, "Motivic infinite loop spaces", arXiv:1711.05248.
  - 用途：framed motivic spaces、P1-infinite loop recognition principle、suspension spectra representability。
  - 精确定位：Theorem 1.2.3 and Theorem 3.5.14；perfect field 上
    grouplike framed spaces/very-effective spectra 与 `S^1`-stable
    effective comparison。
  - 稳定链接：`https://arxiv.org/abs/1711.05248`。
  - 定位状态：located；very effective 层不误称 stable subcategory，见
    batches 1, 3。

- Tom Bachmann, Elden Elmanto, Marc Hoyois, Adeel A. Khan, Vladimir
  Sosnilo, Maria Yakerson, "On the infinite loop spaces of algebraic
  cobordism and the motivic sphere", arXiv:1911.02262。
  - 用途：framed finite-syntomic moduli/Hilbert models。
  - 精确定位：Theorems 1.1 and 1.4；域上版本。
  - 稳定链接：`https://arxiv.org/abs/1911.02262`。
  - 定位状态：P1；不参与 framed recognition 的 P0 闭合。

- Elden Elmanto, Marc Hoyois, Adeel A. Khan, Vladimir Sosnilo, Maria Yakerson, "Framed transfers and motivic fundamental classes", arXiv:1809.10666.
  - 用途：framed transfers 与 motivic fundamental classes 的比较。
  - 精确定位：Section 3；`https://arxiv.org/abs/1809.10666`。
  - 定位状态：P1；具体模型相容时另核对 theorem-level statement。

- Tom Bachmann, Marc Hoyois, "Norms in motivic homotopy theory", arXiv:1711.03061.
  - 用途：finite locally free/finite etale norm functors、normed motivic spectra、`H\mathbb Z`、`KGL`、`MGL` 的 normed structures。
  - 精确定位：Proposition 3.13（unstable quotient compatibility）；
    Proposition 4.5（finite-etale stable norm）；Definition 7.1（normed
    spectrum）；Theorems 14.5、14.14、15.22、16.19（`H\mathbb Z`/Chow、
    `KGL`、`MGL`）。
  - 稳定链接：`https://arxiv.org/abs/1711.03061`。
  - 定位状态：located；finite locally free/finite etale 层级分开，见
    batches 1, 3。

- Brian Shin, "Norms and Transfers in Motivic Homotopy Theory", arXiv:2305.12684.
  - 用途：norms 与 transfers 兼容、norm monoidal refinement。
  - 定位状态：P1。

## Finite correspondences、Milnor-Witt 与 quadratic refinements

- Andrei Suslin, Vladimir Voevodsky, relative cycles and finite
  correspondences papers; Voevodsky motives sources。
  - 用途：finite correspondences 的原始研究来源和 relative-cycle 推广。
  - 定位状态：P1/history；本书 P0 教学陈述改由下列 MVW 精确定位。

- Carlo Mazza, Vladimir Voevodsky, Charles Weibel, "Lecture Notes on Motivic Cohomology".
  - 用途：motivic complexes、`Z(q)`、高 Chow 群和 motivic cohomology 基础。
  - 精确定位：Lecture 1, Lemmas 1.4 and 1.7, Definition 1.5（finite
    correspondences）；Theorem 13.1（Nisnevich sheafification）；Theorem
    19.1 and Corollary 19.2（higher Chow）；Theorem 5.1（Milnor `K`）。
  - 稳定链接：`https://sites.math.rutgers.edu/~weibel/MVWnotes/xprova.pdf`。
  - 定位状态：located；第九、十四章 P0 使用，见 batch 3。

- Fabien Morel, "A1-Algebraic Topology over a Field".
  - 用途：Milnor-Witt K-theory、`A1`-homotopy sheaves、Grothendieck-Witt 值 endomorphisms。
  - 精确定位：Corollary 6.43；Lemma 3.10；链接和边界见本账本“基础
    motivic homotopy”条目。
  - 定位状态：located；第十八章 P0 使用。

- Jean Fasel、Barge-Morel、Calmes-Fasel、Deglise-Fasel 相关文献。
  - 用途：Chow-Witt groups、Milnor-Witt correspondences 和 Milnor-Witt motives。
  - 定位状态：P1。

- Jesse Kass, Kirsten Wickelgren, Marc Levine, Marc Hoyois 等 quadratic enumerative geometry 文献。
  - 用途：quadratic refinements of enumerative counts、Euler classes、motivic degree。
  - 定位状态：P1。

## Equivariant、stacky、log、perfect 与 analytic 扩展

- Marc Hoyois, "The six operations in equivariant motivic homotopy theory", arXiv:1509.02145.
  - 用途：此处使用其非平凡群版本处理 quotient stacks `[X/G]`、equivariant
    motivic spectra、equivariant six operations、cdh descent；基础章的
    trivial-group specialization 已在“六操作和 motives”登记。
  - 定位状态：P0/located；equivariant 章节仍须逐条保留 tame group 与
    resolution-property 假设。

- Adeel A. Khan, Charanya Ravi, "Generalized cohomology theories for algebraic stacks", arXiv:2106.15001.
  - 用途：scalloped algebraic stacks 上的 stable motivic homotopy、六操作、Gysin maps、localization、Mayer-Vietoris。
  - 核查：arXiv 摘要页显示作者、标题和 2021-06-28 日期；摘要说明把 stable motivic homotopy category 扩展到 scalloped algebraic stacks 并建立 Grothendieck 六操作。
  - 定位状态：P0。

- Chirantan Chowdhury, "Motivic Homotopy Theory of Algebraic Stacks", arXiv:2112.15097.
  - 用途：另一种 algebraic stacks 扩展和 enhanced operation map 方法。
  - 定位状态：研究边界/P1。

- Doosung Park, "A1-homotopy theory of log schemes", arXiv:2205.14750.
  - 用途：fs log schemes、log motivic homotopy、strict morphisms 的六操作。
  - 定位状态：研究边界/P1。

- Christian Dahlhausen, Jeroen Hekking, Storm Wolters, "Motivic homotopy theory for perfect schemes", arXiv:2510.01390.
  - 用途：positive characteristic perfect schemes、six-functor formalism、universal homeomorphism localization。
  - 核查：2025-10-01 arXiv 记录；截至 2026-07-08 写入研究边界，不作为基础定理。
  - 定位状态：研究边界。

- Roy Magen, "Geometric Criteria for 6-Functor Formalisms in the Setting of Pullback Formalisms", arXiv:2511.09371.
  - 用途：pullback formalisms 中的六函子几何判据、stacky/analytic universal property、Betti realization compatibility。
  - 核查：2025-11-12 arXiv 记录；截至 2026-07-08 写入研究边界，不作为基础定理。
  - 定位状态：研究边界。

- Roy Magen, "The Localization Theorem for the Motivic Homotopy Theory of Complex Analytic Stacks and other Geometric Settings", arXiv:2605.14470.
  - 用途：complex analytic stacks 的 Morel-Voevodsky localization analogue、为 2025 pullback formalism 结果提供输入。
  - 核查：2026-05-14 arXiv 记录；截至 2026-07-08 写入研究边界，不作为基础定理。
  - 定位状态：研究边界。

## Realization functors

- Ayoub 的 Betti realization 和 motivic six operations 兼容结果。
  - 用途：Betti realization、six-operation compatibility。
  - 定位状态：P0，需补精确 locator。

- Daniel C. Isaksen, etale realization on A1-homotopy theory of schemes.
  - 用途：etale realization、pro-spaces、completion restrictions。
  - 定位状态：P1。

- Tom Bachmann, real etale stable homotopy theory 相关文献。
  - 用途：real etale realization、real closed fields、real realization comparison。
  - 定位状态：P1。
