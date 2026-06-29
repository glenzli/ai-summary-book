# 资料源

本书不是泛泛综述。每个核心定义、定理和证明路线都应能追溯到正式数学资料、教材、论文或经典文献。涉及近期研究的条目按 2026-06-29 至 2026-06-30 核查；后续扩写必须重新核查可能变化的版本信息。

## S0 核心教材

1.  The Univalent Foundations Program, *Homotopy Type Theory: Univalent Foundations of Mathematics*, Institute for Advanced Study, 2013.
    用途：本书基础规则、identity type、equivalence、univalence、HIT、truncation、circle、encode-decode、SIP 和基础范畴论的主资料源。

2.  Egbert Rijke, *Introduction to Homotopy Type Theory*.
    用途：基础 HoTT、等价、h-level、univalence、truncation、HIT 和 synthetic homotopy theory 的现代教材口径。

3.  Vladimir Voevodsky 等关于 univalent foundations、h-level、SIP 和单值数学的资料。
    用途：单值基础的历史口径、结构等同性和基础哲学背景。

## S1 模型论与 Cubical Type Theory

4.  Michael Shulman, *The univalence axiom for elegant Reedy presheaves*.
    用途：simplicial/presheaf model 和 univalence 的模型论背景。

5.  Cyril Cohen, Thierry Coquand, Simon Huber, Anders Mörtberg, *Cubical Type Theory: a constructive interpretation of the univalence axiom*, arXiv:1611.02108.
    用途：cubical type theory、interval、Glue、univalence 的计算解释和模型论边界。

6.  Thierry Coquand, Simon Huber, Anders Mörtberg, *On Higher Inductive Types in Cubical Type Theory*, arXiv:1802.01170.
    用途：HIT 的 cubical 元理论和计算边界。

7.  Daniel R. Licata, Ian Orton, Andrew M. Pitts, Bas Spitters 等关于 cubical、parametricity、univalence decomposition 和模型论的论文。
    用途：第十六章的计算性、模型边界和公理分离讨论。

## S2 单值范畴论与高阶范畴

8.  HoTT Book 中的范畴论章节。
    用途：预范畴、单值范畴、Yoneda、Rezk completion 的教材基础。

9.  Ahrens、Kapulkin、Shulman 等关于 univalent categories、displayed categories、bicategories 和 Rezk completion 的论文。
    用途：第十三至十四章、附录 BE 和附录 BB 的高阶范畴接口。

10.  Emily Riehl, Michael Shulman, *A type theory for synthetic $\infty$-categories*.
    用途：Rezk/Segal、synthetic $\infty$-category type theory 和 directed/simplicial type theory 边界。

## S3 合成同伦论与代数拓扑

11.  HoTT Book 中的 synthetic homotopy theory、circle、suspension、pushout 和 Blakers-Massey 相关章节。
    用途：第十至十二章和附录 AD、AI、AL、AU、AY。

12.  Brunerie、Licata、Finster、Lumsdaine、Shulman 等关于合成同伦论、Blakers-Massey、Freudenthal、Hopf fibration 和球面同伦群的资料。
    用途：高级合成同伦论接口和低阶球面计算边界。

13.  Hatcher, *Algebraic Topology*.
    用途：classical homotopy groups、fiber/cofiber sequences、spectral sequences、Postnikov tower 和 Steenrod operations 的传统数学背景。

14.  May, *A Concise Course in Algebraic Topology*；May, *Simplicial Objects in Algebraic Topology*.
    用途：spectra、spectral sequences、Steenrod algebra、Ext 和 Adams spectral sequence 的经典来源。

15.  Ravenel, *Complex Cobordism and Stable Homotopy Groups of Spheres*.
    用途：Adams spectral sequence、Ext 计算和稳定同伦论边界。

## S4 构造性数学与实数

16.  HoTT Book 中关于 Cauchy reals、Dedekind reals 和 HIT/HIIT 构造的章节。
    用途：附录 AK、AR、AW 的构造性实数接口。

17.  Bishop and Bridges, *Constructive Analysis*.
    用途：构造性连续性、紧致性、级数、积分和选择原则边界。

18.  Troelstra and van Dalen, *Constructivism in Mathematics*.
    用途：构造性逻辑、选择原则、locatedness 和 classical principle 的背景。

## S5 模态、Cohesive HoTT 与合成几何

19.  Shulman 等关于 modal HoTT、cohesive HoTT 和 real-cohesive foundations 的论文。
    用途：附录 AJ、AT、BD 的模态、cohesive 和 SDG/SAG 接口。

20.  Cherubini、Coquand、Hutzler, *A Foundation for Synthetic Algebraic Geometry*.
    用途：合成代数几何的对象语言、Zariski 覆盖、环对象和模型边界。

## S6 逻辑、大小与集合层数学

21.  HoTT Book 中关于 set-level mathematics、quotients、choice、resizing 和逻辑原则的章节。
    用途：第八章、附录 BH、BI、BL。

22.  Aczel、Myhill、Bishop 等构造性集合论和选择原则相关文献。
    用途：有限集、基数、序数、选择原则和构造性边界。

## 使用规则

- 若某个结果来自来源但本书暂不证明，必须标注为外部输入或研究边界。
- 若不同来源采用不同基础口径，必须在正文说明口径差异。
- 若某个经典定理被移植到 HoTT，需要补类型论中的定义翻译和依赖假设。
- 近期研究结果不得无条件升级为核心定理。
