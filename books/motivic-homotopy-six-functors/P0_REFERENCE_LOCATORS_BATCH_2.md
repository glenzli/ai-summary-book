# P0 reference locators batch 2

核查日期：2026-07-11

本批定位覆盖本轮正文主链：presentability/localization、对称幺半
`T`-反演、motivic 六操作、homotopy/lci/smooth purity、compact generation
以及 stable infinity-category 的 triangulated shadow。

## CAT-A / Lurie presentability and localization

- 资料源：Jacob Lurie, *Higher Topos Theory*, Annals of Mathematics Studies 170,
  Princeton University Press, 2009；作者 PDF：
  `https://www.math.ias.edu/~lurie/papers/HTT.pdf`。
- 精确 locator：
  - Theorem 5.1.5.6：presheaf infinity-category 的自由余完备化泛性质。
  - Corollary 5.1.5.8：Yoneda 像在小余极限下生成 presheaves。
  - Corollary 5.5.2.9：presentable adjoint functor theorem。
  - Proposition 5.5.4.15：由一小集合 morphisms 生成的 accessible
    localization 存在且 presentable。
  - Proposition 5.5.4.20：localization 对 colimit-preserving functors 的
    泛性质。
  - Proposition 6.2.2.7：小 Grothendieck site 上 sheafification 是
    topological（accessible left exact）localization。
- 本书使用：附录 A、第一至第二章、第四章右伴随存在性。
- 边界：这些定理不把 ordinary sheaves 自动识别为 hypercomplete sheaves；
  本书默认不 hypercomplete。

## CAT-C / Lurie pointed smash products

- 资料源：Jacob Lurie, *Higher Algebra*，同下文 `TRI-F.1` 的作者 PDF。
- 精确 locator：
  - Proposition 4.8.1.15：`\operatorname{Pr}^{L}` 的 symmetric monoidal
    tensor product。
  - Example 4.8.1.21：`\mathcal C\otimes\mathcal S_*\simeq\mathcal C_*`。
  - Proposition 4.8.2.11：pointed presentable infinity-categories 是由
    `\mathcal S_*` 给出的 idempotent localization/module class。
  - Remark 4.8.2.14（并用 Proposition 4.8.2.9）：`\mathcal S_*` 的
    colimit-preserving symmetric monoidal structure 是 classical smash
    product。
- 本书使用：附录 C 的 C.4；Cartesian product 分变量保持小余极限时，
  把 `\mathcal C` 与 `\mathcal S_*` 的 commutative algebra structures
  在 `\operatorname{Pr}^{L}` 中张量。
- 边界：逐对象的 quotient 公式本身不提供 associativity/coherence；若
  Cartesian product 不分变量保持余极限，也不能推出 presentably
  symmetric monoidal smash product。

## CAT-K / compact generation and Ind-completion

- 资料源：Lurie, *Higher Topos Theory* 与 *Higher Algebra*，作者 PDF 同上。
- 精确 locator：
  - HTT Proposition 5.3.5.11：由小紧致子范畴给出的 Ind-completion
    equivalence criterion。
  - HTT Proposition 5.5.7.8：compactly generated presentable
    infinity-categories 与 essentially small idempotent-complete
    right-exact infinity-categories 的对应。
  - HA Proposition 1.4.4.1：stable category 中小 coproducts/colimits、
    exact colimit-preserving functors 和 compactness 的判据。
- 本书使用：第三章 3.19 与附录 F 的 compactness translation。对一组
  compact stable generators 取 finite cofibers、suspensions 和 retracts，
  再用 Ind-completion 识别全部 category，得到 compact objects 正是 thick
  closure。
- 边界：这些是一般范畴论输入；motivic 几何生成子本身的 compactness 与
  generation 仍由 Hoyois Proposition 6.4(2)-(3) 提供。

## MH-3.8 / Robalo symmetric monoidal inversion

- 资料源：Marco Robalo, *Noncommutative Motives I: A Universal
  Characterization of the Motivic Stable Homotopy Theory of Schemes*,
  arXiv:1206.3645；`https://arxiv.org/abs/1206.3645`。
- 精确 locator：
  - Proposition 4.10：presentable symmetric monoidal category 中形式反演
    任意对象及其泛性质。
  - Corollary 4.24：被反演对象满足 3-symmetry 时，形式反演与相应
    stabilization 等价。
  - Theorem 4.29：在相容 model presentation 下，symmetric spectrum
    objects 呈现同一 symmetric monoidal infinity-category。
- 本书使用：第三章 3.8、附录 C 的 C.8。
- 边界：任意对象的形式反演存在，但 sequential/symmetric spectrum 模型
  比较需要额外 symmetry/model hypotheses；形式反演本身也不自动 stable。

## MH-3.17 and MO-5 / Hoyois motivic spectra and six operations

- 资料源：Marc Hoyois, *The six operations in equivariant motivic homotopy
  theory*, Advances in Mathematics 305 (2017), 197--279,
  DOI 10.1016/j.aim.2016.09.031；作者 PDF：
  `https://hoyois.app.uni-regensburg.de/papers/equivariant.pdf`。本书只取
  trivial-group specialization；另注意 Adv. Math. 333 (2018), 1293--1296
  的 corrigendum。
- 精确 locator：
  - Theorem 1.1：基、群作用和 exceptional functors 定义域的总括版本。
  - Proposition 3.15：motivic localization 保持有限 Cartesian products。
  - Proposition 4.2 及 Proposition 6.4 后的稳定化段落：ordinary smooth
    base change 及其在 `\mathbf{SH}` 中的延拓。
  - Lemma 6.3：用于 symmetric stabilization 的 3-symmetry。
  - Proposition 6.4(1)--(4)：稳定化模型、生成子、smooth suspension
    spectra 的 compactness，以及 arbitrary pullback 的右伴随 `f_*`
    保持 colimits。
  - Corollary 6.7：形式 sphere-inversion 与 standard motivic stabilization
    的比较。
  - Corollaries 6.10、6.11：proper base change 与 proper projection
    formula。
  - Corollary 6.13：smooth proper Atiyah duality；`f_\sharp\mathbb 1_X` 的
    对偶是 `f_\sharp\Sigma^{-T_f}\mathbb 1_X\simeq f_*\mathbb 1_X`。
  - Theorem 6.18(1)--(7)：proper comparison、smooth purity、exceptional
    base change、gluing、immersive full faithfulness、monoidality 与
    exceptional projection formulas。
  - Corollary 6.19：compactifiable `f` 的 `f^!` 保持 colimits。
- 本书使用：第三章 compact generation；第四至第八章六操作、continuity、
  base change、projection formula、localization 和 smooth purity。
- 默认口径：固定 finite-dimensional Noetherian `B`，对象 finite type over
  `B`；exceptional morphisms separated。后者由 Nagata compactification
  落入 Hoyois 的 compactifiable class。
- 边界：Theorem 6.18 的 exceptional base change 不等于 arbitrary ordinary
  base change；ordinary `f_*` projection formula 对所有系数只在 proper
  情形由 Corollary 6.11 保证。

## PU-6.7 / Morel--Voevodsky homotopy purity

- 资料源：Fabien Morel, Vladimir Voevodsky, *A1-homotopy theory of
  schemes*, Publications Mathematiques de l'IHES 90 (1999), 45--143,
  DOI 10.1007/BF02698831；Numdam PDF：
  `https://www.numdam.org/article/PMIHES_1999__90__45_0.pdf`。
- 精确 locator：Section 3, Theorem 2.23。
- 本书使用：第六章 6.7 的
  `X/(X-Z)\simeq\operatorname{Th}(N_{Z/X})`。
- 边界：`X` 与 `Z` 均 smooth over the base，`i` 为 closed embedding；这是
  unstable pointed motivic equivalence，不是 arbitrary `i^!` purity。

## PU-6.12 / Deglise--Jin--Khan purity transformations

- 资料源：Frederic Deglise, Fangzhou Jin, Adeel A. Khan,
  *Fundamental classes in motivic homotopy theory*, JEMS 23 (2021),
  3935--3993, DOI 10.4171/JEMS/1094；arXiv:1805.05920v3。
- 精确 locator：
  - Example 2.3.10：virtual tangent class；smooth 时为 `[T_f]`，regular
    closed immersion 时为 `-[N_f]`。
  - Proposition 2.5.4：purity transformations 的复合与 transverse
    base-change 相容。
  - Remark 2.5.5：上述复合数据只在 homotopy categories 上组织为
    `\mathbf{Tri}` 值逆变伪函子间的自然变换；作者明确说所期待的
    infinity-category 层增强需要额外工作且本文不完成。
  - Theorem 3.3.2：smoothable lci fundamental classes。
  - Theorem 4.1.4：带 motivic ring-spectrum coefficients 的 fundamental
    classes。
  - Paragraph 4.3.1：purity transformation
    `\Sigma^{\tau_f}f^*\to f^!`。
  - Definition 4.3.7：`f`-pure coefficient。
  - Definition 4.3.11 与 Remark 4.3.12(i)：absolute purity 及 regular
    closed-immersion 检测。
- 本书使用：第六章 6.11--6.15，第十六章 fundamental classes/Gysin maps。
- 边界：smoothable lci 假设；一般只得到 transformation。Smooth case
  invertible；regular closed/lci case 的 invertibility 是 coefficientwise
  purity property。第六章只导入 `\operatorname{Ho}(\mathbf{SH})` 伪函子层的
  复合数据及同伦三角层的 base-change 交换方块，不宣称 infinity-natural
  enhancement。Base change 无修正公式要求 Tor-independence；非
  Tor-independent 情形也不对任意方块自动有 excess term：Propositions
  3.3.4、4.2.2 还要求原 morphism 与拉回 morphism 都 smoothable lci，并使用
  Paragraph 3.3.3 的 excess bundle；系数与 proper push-pull 另有各自假设。

## TRI-F.1 / stable infinity-categories and triangulated shadows

- 资料源：Jacob Lurie, *Higher Algebra*, 2017-09-18 version；作者 PDF：
  `https://www.math.ias.edu/~lurie/papers/HA.pdf`。
- 精确 locator：
  - Theorem 1.1.2.14 与 Remark 1.1.2.15：stable infinity-category 的
    homotopy category 带 canonical triangulation。
  - Proposition 1.1.4.1：stable infinity-categories 间 finite-limit、
    finite-colimit 与 exact functor 条件等价。
  - Corollary 1.4.2.27：有 cofibers 的 pointed infinity-category 由
    suspension equivalence 判别稳定性。
- 本书使用：附录 C 的稳定性论证、附录 F 的三角翻译。
- 边界：从 stable enhancement 到 triangulation；不提供 arbitrary
  triangulated category 的 enhancement 存在性或唯一性。

## 本批对账本的影响

- `MH-3.8`、`MH-3.17`、`MH-3.19`、`MO-5.2`、`MO-5.6`、`MO-5.14`、`BC-8.3`、
  `PF-8.5`、`PU-6.7`、`PU-6.12`、`PU-6.13`、`AD-7.3`、`AD-7.10`、
  `TRI-F.1` 可标为 `located`；其中 `AD-7.3` 是由已定位 smooth purity
  经书内伴随论证推出。
- `PU-6.12` 的内容必须登记为 homotopy-category/pseudofunctor 层的 purity
  transformation，而不是无条件 closed-immersion purity equivalence 或未经
  来源支持的 infinity-coherent transformation。
