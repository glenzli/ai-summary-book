# P0 reference locators batch 3

核查日期：2026-07-11

本批定位覆盖第九至第十八章教学主线中的 cohomology、motives、`KGL`、
`MGL`、slice、finite transfers、framed recognition、fundamental classes、
norms 与 Milnor--Witt 输入。每条都区分原文定理的语境；三角同伦范畴中的
比较不自动升级为稳定 presentable infinity-范畴的等价。

## HZ-9.1, HZ-9.9, HZ-9.10 / motivic cohomology

- Markus Spitzweck, *A commutative P1-spectrum representing motivic
  cohomology over Dedekind domains*, arXiv:1207.4078v3，
  `https://arxiv.org/abs/1207.4078`。
  - Theorem 7.18 与 Corollary 7.19：若
    `S=\operatorname{Spec}(D)`，`D` 为 mixed-characteristic Dedekind
    domain，则所构造的交换 `P1`-谱对 `X\in\operatorname{Sm}_S` 表示由
    Bloch--Levine cycle complexes 给出的 motivic cohomology。
  - Theorem 8.25：mixed-characteristic Dedekind domains 之间的态射满足
    所构造谱的 base-change comparison；域上比较见 Theorem 6.7 and Lemma
    8.23。
  - 本书只据此在该基类上使用高度结构化的 `H\mathbb Z_S`；这不是任意概形
    上 `H\mathbb Z` 的无条件构造定理。
- Carlo Mazza, Vladimir Voevodsky, Charles Weibel, *Lecture Notes on
  Motivic Cohomology*，作者 PDF：
  `https://sites.math.rutgers.edu/~weibel/MVWnotes/xprova.pdf`。
  - Theorem 19.1：若 `k` 为 perfect field、`X` 为 smooth separated
    `k`-scheme，则
    `H^{p,q}(X,\mathbb Z) \cong CH^q(X,2q-p)`。
  - Corollary 19.2：同样假设下
    `H^{2q,q}(X,\mathbb Z) \cong CH^q(X)`。
  - Theorem 5.1：对任意域 `F`，
    `H^{n,n}(\operatorname{Spec}F,\mathbb Z) \cong K_n^M(F)`。

## DM-10.8 / motives as modules

- Oliver Röndigs, Paul Arne Ostvaer, *Modules over motivic cohomology*,
  Advances in Mathematics 219 (2008), 689--727，
  `https://doi.org/10.1016/j.aim.2008.05.013`。
  - Theorem 1.1：对 characteristic zero field，motivic
    Eilenberg--Mac Lane spectrum 的 modules 与 Voevodsky big motives 的
    模型给出 monoidal triangulated equivalence。
  - 该定理首先是 model/homotopy-category 层面的陈述；它本身不声称任意
    特征或任意基上的 presentably symmetric monoidal stable
    infinity-categorical equivalence。
- Elden Elmanto, Hakon Kolderup, *On Modules over Motivic Ring Spectra*,
  arXiv:1708.05651v4，`https://arxiv.org/abs/1708.05651`。
  - Theorem 5.2：满足论文 axioms 的 correspondence category 的稳定
    motivic category，与其关联 motivic ring spectrum 的 modules，等价为
    presentably symmetric monoidal stable infinity-categories。
  - Corollary 5.3：对域 `k` 和 finite correspondences 应用上述定理；反演
    `k` 的 exponential characteristic 后得到 `DM(k)` 与
    `H\mathbb Z`-modules 的稳定 infinity-范畴比较。
  - 本书用 Corollary 5.3 承担正特征的 infinity-categorical 主线；未反演
    指数特征的一般基比较仍在 P1 边界。

## KG-11.1, KG-11.6, KG-11.7, KG-11.12 / `KGL` and `KH`

- Oliver Röndigs, Markus Spitzweck, Paul Arne Ostvaer, *Motivic strict
  ring models for K-theory*, arXiv:0907.4121，
  `https://arxiv.org/abs/0907.4121`。
  - Lemma 2.5：Bott-periodized model `KGL^beta` 是交换 monoid。
  - Theorem 3.6：该严格模型的 motivic homotopy type 是 Bott 反演的
    suspension spectrum。
  - Theorem 4.1：其乘法与标准 `KGL` 的乘法相容。
  - 原文工作在 Noetherian finite-dimensional bases 的 motivic model
    categories；本书将其作为严格交换 ring model 的来源，不把“严格交换
    monoid”未经比较地当成现代 infinity-category 中任意基上的唯一
    `E_infinity` 结构。
- Denis-Charles Cisinski, *Descente par eclatements en K-theorie
  invariante par homotopie*, Annals of Mathematics 177 (2013), 425--448，
  `https://doi.org/10.4007/annals.2013.177.2.2`；作者 PDF：
  `https://www.math.univ-toulouse.fr/~dcisinsk/KHdescente.pdf`。
  - 全文约定：概形 Noetherian 且有限 Krull 维数。
  - Theorem 2.20：`KGL` 在 `\mathbf{SH}(S)` 中表示 `KH`。
  - Theorem 3.9：homotopy invariant K-theory `KH` 满足 cdh descent。
- Charles Weibel, *The K-book: An Introduction to Algebraic K-theory*,
  Chapter IV，作者 PDF：
  `https://sites.math.rutgers.edu/~weibel/Kbook/Kbook.IV.pdf`。
  - Corollary IV.12.3.1：regular Noetherian ring `R` 上 `K(R) -> KH(R)`
    为等价。
  - Lemma IV.12.8(3)：quasi-projective regular Noetherian scheme `X` 上
    `K(X) -> KH(X)` 为等价。
  - 因而正文不再对任意“正则概形”无有限性条件地使用此比较。

## MG-12.4, MG-12.11 / `MGL`

- Ivan Panin, Konstantin Pimenov, Oliver Röndigs, *A universality theorem
  for Voevodsky's algebraic cobordism spectrum*, arXiv:0709.4116，
  `https://arxiv.org/abs/0709.4116`。
  - Theorem 2.3.1：对 `S=\operatorname{Spec}(k)` 及交换 `P1`-ring spectrum
    `E`，同伦范畴中的交换 monoid maps `MGL -> E` 与 `E` 的 orientations
    自然双射。
  - 这是同伦范畴中映射集合的分类；本书不把它写成任意基上
    `\operatorname{Map}_{\operatorname{CAlg}}` 的空间等价。
- Marc Hoyois, *From algebraic cobordism to motivic cohomology*,
  arXiv:1210.7182v5，`https://arxiv.org/abs/1210.7182`，
  `https://doi.org/10.1515/crelle-2013-0038`。
  - Theorem 7.12：若 `S` essentially smooth over a field of
    characteristic exponent `c`，则 canonical map
    `MGL/(a_1,a_2,...) [1/c] -> H\mathbb Z[1/c]` 为等价。
  - 当 `c=1` 时无需反演；正文不把该结论推广到任意基或不反演的正特征。

## SL-13.8 / zero slice

- Vladimir Voevodsky, *On the zero slice of the sphere spectrum*,
  arXiv:math/0301013，`https://arxiv.org/abs/math/0301013`；IAS PDF：
  `https://www.math.ias.edu/vladimir/sites/math.ias.edu.vladimir/files/on_the_zero_slice_published.pdf`。
  - Theorem 6.6：对 characteristic zero field，
    `s_0(\mathbb 1) \simeq H\mathbb Z`。
  - Introduction, pp. 106--107：由零 slice 的乘法作用推出任意谱的 slices
    带 `H\mathbb Z`-module 结构。
  - 更一般基与正特征版本不是本章 P0 主线，保留为 P1 推广边界。

## TR-14.3, TR-14.10 / finite correspondences and sheafification

- Mazza--Voevodsky--Weibel, *Lecture Notes on Motivic Cohomology*，
  `https://sites.math.rutgers.edu/~weibel/MVWnotes/xprova.pdf`。
  - Lecture 1, Lemmas 1.4 and 1.7 证明复合循环仍满足 finite-over-source
    条件并验证结合性；Definition 1.5 定义加性范畴 `Cor_k`。
  - Theorem 13.1：presheaf with transfers 的 Nisnevich sheafification
    唯一继承 transfers。该定理本身不需要 perfectness；本书第十四章的
    perfect-field 约定是更强的统一口径。
  - Theorem 19.1 同时定位 motivic complexes/hypercohomology 与 higher
    Chow groups 的教学比较接口。

## FR-15.x, FC-16.2 and FC-16.13 / framed recognition, fundamental and excess

- Elden Elmanto, Marc Hoyois, Adeel A. Khan, Vladimir Sosnilo, Maria
  Yakerson, *Motivic infinite loop spaces*, arXiv:1711.05248v6，
  `https://arxiv.org/abs/1711.05248`。
  - Theorem 1.2.3：perfect field 上的 Motivic Recognition Principle。
  - Theorem 3.5.14：framed motivic spaces/spectra 与
    `SH^{veff}(k)`、`SH^{eff}(k)` 的 fully faithful/equivalence statements。
  - Section 2：finite syntomic correspondences 及 cotangent-complex
    framing 的三个等价模型。Finite syntomic 的 cotangent complex 是
    perfect of cohomological amplitude `[-1,0]`；`[0,0]` 只覆盖 smooth
    情形，不能作为定义。
- Frederic Deglise, Fangzhou Jin, Adeel A. Khan, *Fundamental classes in
  motivic homotopy theory*, arXiv:1805.05920v3，
  `https://arxiv.org/abs/1805.05920`。
  - Definition 3.2.5：regular closed immersion 的 fundamental class。
  - Theorem 3.3.2：smoothable lci fundamental classes 及 composition、
    Tor-independent transverse base change。
  - Theorems 4.1.4 and 4.2.1：带 motivic ring-spectrum coefficients 的
    fundamental classes 与 Gysin maps。
  - Paragraph 3.3.3 与 Proposition 3.3.4：若 Cartesian square 中 `f` 及其
    拉回 `g` 都是 smoothable lci s-morphisms，则法丛单射的向量丛商 `\xi`
    给出球谱 fundamental-class excess formula。
  - Proposition 4.2.2：上述公式的 coefficient 版本；`E` 需带 unital
    associative commutative multiplication，涉及 push-pull 的附加公式还要求
    两条竖边 proper。
  - 无修正 base change 需要 Tor-independence；一般 smoothable lci 只先有
    purity transformation，成为等价还需要相应 coefficientwise purity。任意
    non-Tor-independent Cartesian square 不自动具有 excess bundle 或 excess
    formula。

## NM-17.x / norms and normed spectra

- Tom Bachmann, Marc Hoyois, *Norms in motivic homotopy theory*,
  arXiv:1711.03061v5，`https://arxiv.org/abs/1711.03061`。
  - Theorem 3.3 及紧随其后的 motivic localization：finite locally free
    morphisms 上的 pointed unstable norms；Corollary 3.11 给出与 open-subpresheaf
    quotients 的相容。
  - Proposition 3.13：仅当 `p:T\to S` finite etale、`X\in\operatorname{Sm}_T`、
    `Z\subset X` closed 且 Weil restriction `R_pX` 存在时，才把上述商写成
    `R_pX/(R_pX\setminus R_pZ)`；Example 3.14 排除一般 finite locally free 推广。
  - Proposition 4.5：finite etale `p:T -> S` 的 norm 唯一稳定化为保持相关
    余极限的 symmetric monoidal functor `SH(T) -> SH(S)`。
  - Definition 7.1：normed spectrum 是 finite-etale span category 上满足
    cocartesian 条件的 section；这包含 composition、base change 与
    distributivity coherence，而不只是逐个 norm maps。
  - Theorem 14.5：Noetherian `S` 上 `H\mathbb Z_S` 具有相应 normed
    structure；Theorem 14.14 识别域上 Chow groups 的范数。
  - Theorem 15.22：任意 scheme `S` 上 `KGL_S` 为 normed spectrum。
  - Theorem 16.19：任意 scheme `S` 上 `MGL_S` 及其 periodization 为
    normed spectra。

## MW-18.2, MW-18.5 / Milnor--Witt and Grothendieck--Witt

- Fabien Morel, *A1-Algebraic Topology over a Field*, Lecture Notes in
  Mathematics 2052, Springer, 2012，
  `https://doi.org/10.1007/978-3-642-29514-0`；可检索作者版：
  `https://www.mahalex.net/teaching/seminars/morel/Morel.pdf`。
  - 全书约定 `k` 为 perfect field。
  - Corollary 6.43：计算 motivic spheres 的稳定区间映射群；其稳定化的
    对角项给出 sphere 的 degree-zero stem `K_0^{MW}(k)`。
  - Lemma 3.10：`K_0^{MW}(k) \cong GW(k)` 为环同构。
  - 合并两条得到
    `\operatorname{End}_{\mathbf{SH}(k)}(\mathbb 1_k) \cong GW(k)`；这是
    第十八章 Euler characteristic 主线的唯一 P0 quadratic 输入。

## 明确降为 P1 的研究/高级边界

- 第九章 etale cycle map/Bloch--Kato、第十一章 rational Chern character、
  第十三章 motivic Adams 收敛、第十五章 framed Hilbert-scheme 模型与
  framed/fundamental 的全相容、第十七章 norm/framed 的 Tambara 型细化，
  以及第十八章 Chow--Witt motives、一般 quadratic enumerative formulas、
  local-degree 与 Gauss--Bonnet 精化，均不参与第九至第十八章 P0 教学主线的
  证明闭合。它们保留为 P1 外部输入，并必须在实际调用时另补对应模型的
  假设与 locator。

## 本批对账本的影响

- `HZ-9.1`、`HZ-9.9`、`HZ-9.10`、`DM-10.8`、`KG-11.1`、
  `KG-11.6`、`KG-11.7`、`KG-11.12`、`MG-12.4`、`MG-12.11`、
  `SL-13.8`、`TR-14.3`、`TR-14.10`、`FR-15.x`、`FC-16.2`、`FC-16.13`、
  `NM-17.x`、`MW-18.2` 与 `MW-18.5` 可标为 `located`。
- `MW-18.7` 只代表高级 refinements 的合集，不再列入 P0 闭合队列。
