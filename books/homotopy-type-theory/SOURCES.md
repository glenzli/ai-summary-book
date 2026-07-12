# 资料源

本书不是泛泛综述。每个核心定义、定理和证明路线都应能追溯到正式数学资料、教材、论文或经典文献。基础类型论、单值性、simplicial/cubical 语义与 HIT 条目于 2026-07-11 重新核查；附录 AO 的四组外部输入于 2026-07-12 按指定 arXiv 版本重新核查；其余近期研究条目沿用 2026-06-29 至 2026-06-30 的核查记录。后续扩写必须重新核查可能变化的版本信息。

## S0 核心教材

1.  The Univalent Foundations Program, *Homotopy Type Theory: Univalent Foundations of Mathematics*, Institute for Advanced Study, 2013, <https://homotopytypetheory.org/book/>.
    用途：第 1-2 章的 universe/family 与 identity/transport 规则定位到 §§1.3、1.12、2.3 和 Appendix A；附录 D.1 的固定端点 J 及严格 $\beta$ 精确定位到 §1.12.2；函数外延性与单值性定位到 §§2.9-2.10、4.9；圆及其 induction/computation 定位到 Chapter 6。第 6 章定理 6.11 精确使用 Theorems 4.9.4-4.9.5，并限制到基底与 fibers 同属一个单值 universe 的实例。

2.  Egbert Rijke, *Introduction to Homotopy Type Theory*, Cambridge University Press, 2025, DOI `10.1017/9781108933568`; prepublication version arXiv:2212.11082.
    用途：基础 HoTT、等价、h-level、univalence、truncation、circle 和 synthetic homotopy theory 的现代教材口径；涉及版本定位时优先使用 2025 年正式版。

3.  Vladimir Voevodsky 等关于 univalent foundations、h-level、SIP 和单值数学的资料。
    用途：单值基础的历史口径、结构等同性和基础哲学背景。

## S1 模型论与 Cubical Type Theory

4.  Chris Kapulkin, Peter LeFanu Lumsdaine, *The Simplicial Model of Univalent Foundations (after Voevodsky)*, arXiv:1211.2851.
    用途：第 16 章外部输入定理 16.1；一个 univalent universe 的 simplicial set 模型、contextual-category coherence 和相对于带两个不可达基数的 ZFC 的一致性推论。该来源不支撑 HIT、normalization 或 judgmental computation。

5.  Cyril Cohen, Thierry Coquand, Simon Huber, Anders Mörtberg, *Cubical Type Theory: a constructive interpretation of the univalence axiom*, LIPIcs TYPES 2015 (2018), DOI `10.4230/LIPIcs.TYPES.2015.5`, arXiv:1611.02108.
    用途：第 16 章定义 16.2 与外部输入定理 16.3；interval/face lattice、$\mathsf{Path}$、composition/transport、Glue、univalence 和构造性 cubical set 语义。精确落点为 §§4、6、7.2、8；该对象语言的 $\mathsf{Path}$ 不与第 2 章的归纳 $\mathsf{Id}$ 静默等同。

6.  Thierry Coquand, Simon Huber, Anders Mörtberg, *On Higher Inductive Types in Cubical Type Theory*, LICS 2018, DOI `10.1145/3209108.3209197`, arXiv:1802.01170.
    用途：第 9 章外部输入定理 9.11 和第 16 章 16.12.3；spheres、torus、suspensions、truncations、pushouts 的语法与语义、所有构造子的 judgmental computation、严格替换稳定性和 universe closure。论文不提供无条件的一般 HIT schema。

7.  Simon Huber, *Canonicity for Cubical Type Theory*, Journal of Automated Reasoning 63 (2019), DOI `10.1007/s10817-018-9469-1`.
    用途：第 16 章外部输入定理 16.6；CCHM calculus 中 name-variable contexts 的自然数 judgmental canonicity。不能自动推广到任意 cubical 变体或 HIT 扩展。

8.  Jonathan Sterling, Carlo Angiuli, *Normalization for Cubical Type Theory*, arXiv:2101.11479.
    用途：第 16 章外部输入定理 16.6.1；univalent Cartesian cubical type theory 的 normalization、judgmental equality 可判定性和类型构造子单射性。

9.  Michael Shulman, *The univalence axiom for elegant Reedy presheaves*.
    用途：presheaf models 和 univalence 的模型论背景；不替代第 16.1 节的 simplicial model 精确来源。

10. Ian Orton, Andrew M. Pitts, *Axioms for Modelling Cubical Type Theory in a Topos*, LIPIcs CSL 2016, DOI `10.4230/LIPIcs.CSL.2016.24`.
    用途：cubical topos 语义、uniform Kan filling 与模型假设的分解；用于第 16.4 节的模型比较边界。

## S2 单值范畴论与高阶范畴

11. HoTT Book 中的范畴论章节。
    用途：预范畴、单值范畴、Yoneda、Rezk completion 的教材基础。

12. Ahrens、Kapulkin、Shulman 等关于 univalent categories、displayed categories、bicategories 和 Rezk completion 的论文。
    用途：第十三至十四章、附录 BE 和附录 BB 的高阶范畴接口。

13. Emily Riehl, Michael Shulman, *A type theory for synthetic $\infty$-categories*.
    用途：Rezk/Segal、synthetic $\infty$-category type theory 和 directed/simplicial type theory 边界。

## S3 合成同伦论与代数拓扑

14. HoTT Book 中的 synthetic homotopy theory、circle、suspension、pushout 和 Blakers-Massey 相关章节。
    用途：第十至十二章和附录 AD、AI、AL、AU、AY。

15. Brunerie、Licata、Finster、Lumsdaine、Shulman 等关于合成同伦论、Blakers-Massey、Freudenthal、Hopf fibration 和球面同伦群的资料。
    用途：高级合成同伦论接口和低阶球面计算边界。

16. Hatcher, *Algebraic Topology*.
    用途：classical homotopy groups、fiber/cofiber sequences、spectral sequences、Postnikov tower 和 Steenrod operations 的传统数学背景。

17. May, *A Concise Course in Algebraic Topology*；May, *Simplicial Objects in Algebraic Topology*.
    用途：spectra、spectral sequences、Steenrod algebra、Ext 和 Adams spectral sequence 的经典来源。

18. Ravenel, *Complex Cobordism and Stable Homotopy Groups of Spheres*.
    用途：Adams spectral sequence、Ext 计算和稳定同伦论边界。

## S4 构造性数学与实数

19. HoTT Book 中关于 Cauchy reals、Dedekind reals 和 HIT/HIIT 构造的章节。
    用途：附录 AK、AR、AW 的构造性实数接口。

20. Bishop and Bridges, *Constructive Analysis*.
    用途：构造性连续性、紧致性、级数、积分和选择原则边界。

21. Troelstra and van Dalen, *Constructivism in Mathematics*.
    用途：构造性逻辑、选择原则、locatedness 和 classical principle 的背景。

## S5 模态、Cohesive HoTT 与合成几何

22. Shulman 等关于 modal HoTT、cohesive HoTT 和 real-cohesive foundations 的论文。
    用途：附录 AJ、AT、BD 的模态、cohesive 和 SDG/SAG 接口。

23. Cherubini、Coquand、Hutzler, *A Foundation for Synthetic Algebraic Geometry*.
    用途：合成代数几何的对象语言、Zariski 覆盖、环对象和模型边界。

## S6 逻辑、大小与集合层数学

24. HoTT Book 中关于 set-level mathematics、quotients、choice、resizing 和逻辑原则的章节。
    用途：第八章、附录 BH、BI、BL。

25. Aczel、Myhill、Bishop 等构造性集合论和选择原则相关文献。
    用途：有限集、基数、序数、选择原则和构造性边界。

## S7 2025-2026 模型论外部输入

26. Evan Cavallo, Jonas Höfer, *Univalence without function extensionality*, arXiv:2605.00812v1, 2026-05-01, <https://arxiv.org/abs/2605.00812v1>.
    用途：附录 AO.1。Definitions 1.1--1.4 区分 function extensionality、universe univalence、categorical univalence 与 familial categorical univalence；Theorem 1.5（Theorem 4.17）给出 polynomial model 的保持结论，Proposition 5.3 给出其中函数外延性的失败，Theorem 1.6（Theorem 5.6）给出 categorical univalence 不推出函数外延性的模型分离。
    采用/未采用：只采用 Theorem 5.6 的分离结论。Theorem 4.17 所需 extensive finite coproducts、strict $\eta$ 和 familial categorical univalence 是反模型构造的语义假设；本书不把它们加入基础语法，也不采用 polynomial model 的其他内部结构。

27. Rafaël Bocquet, *Strict Rezk completions of models of HoTT and homotopy canonicity*, arXiv:2311.05849v2, 2025-10-08, <https://arxiv.org/abs/2311.05849v2>.
    用途：附录 AO.2。Definitions 5.1--5.2 定义 complete model 与 strict Rezk completion；Theorem 5.18 对 global algebraically cofibrant、components fibrant 的 HoTT 模型给出 completion；Remark 5.19 处理自由语法；Theorem 6.1 与 §6.2 给出闭 Boolean 项的 homotopy canonicity。
    采用/未采用：采用 Theorems 5.18、6.1 的元理论路线。来源 §4 使用 cumulative univalent universes、函数外延性、W-types 等较强语法；这些不回流为本书非累积基础规则，且本书不把 completion 存在性推广到不满足 Theorem 5.18 假设的模型。

28. Evan Cavallo, Christian Sattler, *Eliminating reversals from cubical type theories*, arXiv:2605.15080v1, 2026-05-14, <https://arxiv.org/abs/2605.15080v1>.
    用途：附录 AO.3。§3.3 固定 opaque cubical theory；Definition 23 定义 self-dual interval theory；Definitions 25、27 与 Theorem 42 给出 reversal extension/twist interpretation；Theorem 65 给出 opaque 情形的 conservativity weak equivalence；§7 Theorem 71 给出特定 ABCHFL strict model。
    采用/未采用：采用 Theorem 65，并保留 self-dual 与 opaque 假设。不采用 strict-over-opaque 保守性；Theorem 71 的 $\infty$-groupoid 识别需要 classical logic，所得模型没有 connection，因而不支撑带 connections 的一般结论。

29. Nima Rasekh, *Non-Standard Models of Homotopy Type Theory*, arXiv:2508.07736v2, 2025-08-12, <https://arxiv.org/abs/2508.07736v2>.
    用途：附录 AO.4。Definition 2.7 定义 model/simplicial model filter；Theorems 2.13--2.14 给出模型范畴性质与类型构造的保持；Example 2.18 展示仍建模 HoTT 而失去 infinite (co)limits、local presentability、cofibrant generation 的 filter product；Corollaries 2.21--2.22 给出相应独立性与非标准自然数结论。
    采用/未采用：只在 Definition 2.7 的 model-filter 条件及需要时的 simplicial 条件下采用 Theorems 2.13--2.14；不推广到任意 filter、任意模型范畴或任意 HoTT 语法，也不把外部范畴性质转写成对象语言规则。

## 使用规则

- 若某个结果来自来源但本书暂不证明，必须标注为外部输入或研究边界。
- 若不同来源采用不同基础口径，必须在正文说明口径差异。
- 若某个经典定理被移植到 HoTT，需要补类型论中的定义翻译和依赖假设。
- 模型来源必须列出被解释语法、元理论假设、精确结论和非结论。
- 近期研究结果不得无条件升级为核心定理。
