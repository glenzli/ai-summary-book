# P0 reference locators batch 1

核查日期：2026-07-08

本批定位已经核查原始 arXiv/PDF 页面。范围限于本书主链中最常调用的 modern P0 输入：universal six-functor formalism、framed recognition、norms、fundamental classes/Gysin maps。

## UF-23.2 / Drew-Gallauer universal six-functor formalism

- 资料源：Brad Drew, Martin Gallauer, "The universal six-functor formalism", arXiv:2009.13610v4；Ann. K-Theory 7 (2022), 599-649。
- arXiv 页面：`https://arxiv.org/abs/2009.13610`
- 核查信息：arXiv 页面标明 v4 日期 2023-08-07，期刊信息 Ann. K-Th. 7 (2022) 599-649。
- 精确 locator：
  - Theorem 7.14，PDF p.42：`SH` is initial in the infinity-category `CoSy^c_B` of cocomplete coefficient systems。
  - Proposition 7.13，PDF p.42：cocomplete coefficient systems fully faithfully embed into the relevant pullback formalism category; used to justify the passage from six-functor coefficient systems to pullback formalisms.
  - Theorem 7.3，PDF p.38：sequence of adjunctions/universal steps enforcing motivic axioms before the main theorem.
- 本书使用：
  - 第二十三章 universal coefficient system。
  - 第四、第五、第二十三章中“`\mathbf{SH}` 是 motivic coefficient systems 的 universal source”的外部输入。
- 限制：
  - Theorem 7.14 concerns morphisms of coefficient systems; the paper itself notes morphisms need not automatically commute with all six functors unless additional hypotheses hold. 本书第 23 章已保留该边界。

## FR-15.x / Elmanto-Hoyois-Khan-Sosnilo-Yakerson framed recognition

- 资料源：Elden Elmanto, Marc Hoyois, Adeel A. Khan, Vladimir Sosnilo, Maria Yakerson, "Motivic infinite loop spaces", arXiv:1711.05248v6。
- arXiv 页面：`https://arxiv.org/abs/1711.05248`
- 核查信息：arXiv 页面标明 v6 日期 2021-07-09，final version to appear in Cambridge Journal of Mathematics。
- 精确 locator：
  - Theorem 1.2.3，PDF pp.2-3：Motivic Recognition Principle, stated as equivalence between very effective motivic spectra and grouplike framed motivic spaces over a perfect field.
  - Theorem 3.5.14，PDF pp.39-40：fully faithful functors from framed motivic spaces/spectra into `SH(k)` and equivalences with `SH^{veff}(k)` and `SH^{eff}(k)`.
  - Section 2，PDF pp.9-27：notions of equational, normal and tangential framed correspondences.
- 本书使用：
  - 第十五章 recognition principle、grouplike condition、framed transfers。
  - 第十五章关于 finite syntomic morphisms with K-theoretic cotangent trivialization 的口径。
- 限制：
  - 主定理默认 perfect field；pro-smooth base extension 出现在后续 corollaries，不能无条件推广到任意基。

## NM-17.x / Bachmann-Hoyois norms

- 资料源：Tom Bachmann, Marc Hoyois, "Norms in motivic homotopy theory", arXiv:1711.03061v5。
- arXiv 页面：`https://arxiv.org/abs/1711.03061`
- 核查信息：arXiv 页面标明 v5 日期 2020-05-28，final version to appear in Asterisque。
- 精确 locator：
  - Abstract，PDF p.0：finite locally free unstable norm `f_\otimes:\mathcal H_*(S')\to\mathcal H_*(S)` and finite etale stable norm `f_\otimes:\mathcal{SH}(S')\to\mathcal{SH}(S)`.
  - Theorem 3.3，PDF pp.15--16，及其后的 localization 段落：universally
    clopen 的 presheaf-level pointed norm；对 integral universally open maps
    通过 Nisnevich/`\mathbb A^1` localization。Finite locally free maps 满足这些
    条件，故得到 unstable norm。
  - Corollary 3.11，PDF p.17：对 integral universally open `p`、presheaf `X`
    与 open subpresheaf `Y\subset X`，有
    `p_\otimes(X/Y)\simeq p_*X/p_*(X|Y)`。
  - Proposition 3.13，PDF p.18：`p` finite etale、`X\in\operatorname{Sm}_T`、
    `Z\subset X` closed 且 Weil restriction `R_pX` 存在（例如 `X/T`
    quasi-projective）时，norm 与 closed-complement quotient 的 Weil-restriction
    表达相容；Example 3.14 说明只假设 finite locally free 时该表达失败。
  - Proposition 4.5，PDF p.19：for finite etale `p:T\to S`, `\Sigma^\infty p_\otimes` has a unique symmetric monoidal extension `p_\otimes:\mathbf{SH}(T)\to\mathbf{SH}(S)` preserving filtered colimits and sifted colimits.
  - Definition 7.1，PDF p.35：definition of normed spectrum as a section of `SH^\otimes` over `Span(C, all, fet)` cocartesian over `C^{op}`.
- 本书使用：
  - 第十七章 norm functors、normed spectra、Tambara-like compatibility。
  - 第十七章 failure mode: finite locally free unstable norm does not automatically stabilize; stable statement is finite etale.
- 限制：
  - Stable norm functor locator is finite etale. Finite locally free appears at
    unstable level through Theorem 3.3/localization and Corollary 3.11；不得把
    Proposition 3.13 的 finite-etale、Weil-restriction-existence 结论扩大到该层级。

## FC-16.2/FC-16.13 / Deglise-Jin-Khan fundamental classes, Gysin and excess

- 资料源：Frederic Deglise, Fangzhou Jin, Adeel A. Khan, "Fundamental classes in motivic homotopy theory", arXiv:1805.05920v3；JEMS DOI linked on arXiv page.
- arXiv 页面：`https://arxiv.org/abs/1805.05920`
- 核查信息：arXiv 页面标明 v3 日期 2021-01-29；摘要说明构造 fundamental classes、bivariant theory、Gysin maps、specialization、excess/self-intersection/blow-up formulas 和 Euler classes。
- 精确 locator：
  - Definition 3.2.5，PDF p.21：fundamental class for regular closed immersions via specialization and Thom isomorphism.
  - Theorem 3.3.2，PDF pp.29-30：system of fundamental classes on smoothable lci morphisms, compatible with smooth morphisms, regular closed immersions and transverse base change.
  - Theorem 4.1.4，PDF pp.31-32：fundamental classes with coefficients in a motivic ring spectrum.
  - Theorem 4.2.1，PDF p.32：Gysin maps with coefficients for smoothable lci morphisms and their functoriality/transverse base change.
  - Paragraph 3.3.3 and Proposition 3.3.4：当 Cartesian square 中 `f` 与拉回
    `g` 都是 smoothable lci s-morphisms 时，由法丛商 `\xi` 得到球谱基本类的
    excess formula。
  - Proposition 4.2.2：带 unital associative commutative multiplication
    系数的 excess formula；proper push-pull 结论另要求两条竖边 proper。
  - Paragraph 4.3.3，PDF p.34：purity transformation induces Gysin maps for bivariant theory, cohomology with proper support, cohomology for proper maps, and bivariant theory with proper support.
- 本书使用：
  - 第十六章 fundamental classes、Gysin maps、bivariant theory、excess intersection、Riemann-Roch boundary。
  - 第十五章 framed transfers 与 fundamental classes 的相容边界。
- 限制：
  - Main class is for smoothable lci morphisms in the cited theorem. 本书不得把该结果写成任意 morphism 的 fundamental class。
  - Excess formula 不是任意 non-Tor-independent square 的形式结论；必须保留
    两条对应边 smoothable lci、excess bundle 存在、系数乘法和 properness 的
    分层假设。

## 本批对账本的影响

- `UF-23.2` 可标为 `located`。
- `FR-15.x` 可标为 `located`。
- `NM-17.x` 可标为 `located`，但需保留 finite locally free/finite etale 分界。
- `FC-16.2` 与 `FC-16.13` 可标为 `located`；前者保留 smoothable lci
  假设，后者还须保留双 lci、excess bundle、系数乘法与 properness 的分层
  假设。
