# 资料源

本文档记录《Operad Theory》教材的主要资料源。正文不得复制原文；所有内容均应重写并标明依赖边界。

核查日期：2026-07-11。

## 经典来源

- J. P. May, *The Geometry of Iterated Loop Spaces*, Lecture Notes in Mathematics 271, Springer, 1972. 用途：operad 的早期形式化、iterated loop spaces。
- F. R. Cohen, “The homology of $C_{n+1}$-spaces, $n\ge0$,” in *The Homology of Iterated Loop Spaces*, Lecture Notes in Mathematics 533, Springer, 1976. 用途：little cubes operad 同调与 Poisson/Gerstenhaber 型结构。
- Jim Stasheff, “Homotopy associativity of H-spaces I, II,” *Transactions of the American Mathematical Society* 108, 1963. 用途：$A_\infty$ 结构和 associahedra 的经典来源。
- J. M. Boardman and R. M. Vogt, *Homotopy Invariant Algebraic Structures on Topological Spaces*, Lecture Notes in Mathematics 347, Springer, 1973. 用途：Boardman-Vogt tensor product、homotopy invariant structures。
- Martin Markl, Steve Shnider, and Jim Stasheff, *Operads in Algebra, Topology and Physics*, Mathematical Surveys and Monographs 96, AMS, 2002. 用途：经典定义、例子、树和同伦代数背景。
- Jean-Louis Loday and Bruno Vallette, *Algebraic Operads*, Grundlehren der mathematischen Wissenschaften 346, Springer, 2012；作者托管 draft v0.99. 用途：线性 operad、Koszul 对偶、bar-cobar 构造和同伦代数；LV-1/Theorem 6.6.2 是 connected weight-graded operadic twisting-morphism fundamental theorem，LV-2/Theorem 7.4.6 是 quadratic Koszul criterion，LV-3/Theorem 8.1.1 及其后 $\operatorname{As}$ 例子给出 nonsymmetric rewriting/Koszul 判据。链接：<https://www.math.univ-paris13.fr/~vallette/Operads.pdf>。
- Tom Leinster, *Higher Operads, Higher Categories*, London Mathematical Society Lecture Note Series 298, Cambridge University Press, 2004. 用途：multicategories、higher operads 和高阶结构背景。
- Bruno Vallette, “Algebra+Homotopy=Operad,” arXiv:1202.3245, 2012-02-15. 用途：历史和应用导览；不得作为核心定理的唯一依据。链接：<https://arxiv.org/abs/1202.3245>。
- Benoit Fresse, *Homotopy of Operads and Grothendieck-Teichmüller Groups*, Mathematical Surveys and Monographs 217, AMS, 2017. 用途：同伦 operad、$E_n$-operad 和 Grothendieck-Teichmüller 相关主题。

## Operad 的同伦理论

- Daniel G. Quillen, *Homotopical Algebra*, Lecture Notes in Mathematics 43, Springer, 1967. 用途：模型范畴与 Quillen equivalence 基础。
- Mark Hovey, *Model Categories*, Mathematical Surveys and Monographs 63, AMS, 1999. 用途：模型范畴、幺半模型范畴和 Quillen adjunction 背景。
- Philip S. Hirschhorn, *Model Categories and Their Localizations*, Mathematical Surveys and Monographs 99, AMS, 2003. 用途：局部化、小对象论证和模型范畴技术。
- Paul G. Goerss and John F. Jardine, *Simplicial Homotopy Theory*, Birkhauser, 1999. 用途：Kan-Quillen 模型结构、simplicial homotopy theory。
- W. G. Dwyer and D. M. Kan, “Simplicial localizations of categories,” *Journal of Pure and Applied Algebra* 17, 1980. 用途：Dwyer-Kan localization 和 hammock localization。
- Clark Barwick and Daniel M. Kan, “Relative categories: another model for the homotopy theory of homotopy theories,” arXiv:1011.1691, 2010. 用途：relative categories 与 homotopy theories 的模型。链接：<https://arxiv.org/abs/1011.1691>。
- Vladimir Hinich, “Dwyer-Kan localization revisited,” arXiv:1311.4128, 2013. 用途：infinity-categorical localization、hammock localization comparison、underlying infinity-category 和 Quillen-pair passage。链接：<https://arxiv.org/abs/1311.4128>。
- Clemens Berger and Ieke Moerdijk, “Axiomatic homotopy theory for operads,” *Commentarii Mathematici Helvetici* 78, 2003. 用途：模型范畴中 operad 的同伦理论。
- Clemens Berger and Ieke Moerdijk, “Resolution of coloured operads and rectification of homotopy algebras,” arXiv:math/0512576, 2005. 用途：colored operads 的 resolution 与 rectification。链接：<https://arxiv.org/abs/math/0512576>。
- Vladimir Hinich, “Homological algebra of homotopy algebras,” arXiv:q-alg/9702015, 1997. 用途：dg-operads、同伦代数和模型结构。链接：<https://arxiv.org/abs/q-alg/9702015>。
- Benoit Fresse, “Operadic cobar constructions, cylinder objects and homotopy morphisms of algebras over operads,” arXiv:0902.0177. 用途：operadic cobar construction、twisted composite object、quasi-free/cofibrant replacement 和 homotopy morphisms。链接：<https://arxiv.org/abs/0902.0177>。
- Dmitri Pavlov and Jakob Scholbach, “Admissibility and rectification of colored symmetric operads,” arXiv:1410.5675v4, 2022-03-27. 用途：Theorem 5.11 的 colored admissibility、Theorem 7.5 的 rectification 和 Theorem 7.11 的 simplicial strict-to-infinity algebra comparison；三者假设不得互换。链接：<https://arxiv.org/abs/1410.5675>。
- Dmitri Pavlov and Jakob Scholbach, “Homotopy theory of symmetric powers,” arXiv:1510.04969, 2015. 用途：symmetric h-monoidality、symmetric flatness 与 operad 代数模型结构。链接：<https://arxiv.org/abs/1510.04969>。
- David White, “Monoidal Bousfield Localizations and Algebras over Operads,” arXiv:1404.5197. 用途：模型范畴中 Bousfield localization preserves operad algebras、monoidal localization criteria；不替代 infinity-categorical algebra localization comparison。链接：<https://arxiv.org/abs/1404.5197>。
- David White and Donald Yau, “Bousfield localization and algebras over colored operads,” arXiv:1503.06720. 用途：colored operad algebra structures under Bousfield localization；不替代 dendroidal-Lurie 或 Lurie-style algebra comparison。链接：<https://arxiv.org/abs/1503.06720>。
- Victor Ginzburg and Mikhail Kapranov, “Koszul duality for operads,” *Duke Mathematical Journal* 76, 1994. 用途：二次 operad 与 Koszul 对偶的经典来源。链接：<https://arxiv.org/abs/0709.1228>。
- Murray Gerstenhaber, “The cohomology structure of an associative ring,” *Annals of Mathematics* 78, 1963. 用途：Hochschild cohomology 上的 Gerstenhaber 结构。
- Alexander A. Voronov, “Homotopy Gerstenhaber algebras,” 1990s. 用途：brace / homotopy Gerstenhaber 结构背景。
- James McClure and Jeffrey Smith, “A solution of Deligne's Hochschild cohomology conjecture,” arXiv:math/9910126, 2001. 用途：Deligne 猜想的 operadic/cosimplicial 证明。链接：<https://arxiv.org/abs/math/9910126>。
- Clemens Berger and Benoit Fresse, “Combinatorial operad actions on cochains,” arXiv:math/0109158, 2002. 用途：surjection/brace operad 与 Deligne 猜想相关模型。链接：<https://arxiv.org/abs/math/0109158>。
- Ezra Getzler and John D. S. Jones, “Operads, homotopy algebra and iterated integrals for double loop spaces,” 1990s preprint. 用途：同伦代数与 iterated integrals 的背景。
- Tornike Kadeishvili, “The algebraic structure in the homology of an $A(\infty)$-algebra,” 1980. 用途：$A_\infty$ 最小模型和同伦转移。
- Martin Markl, “Homotopy Algebras are Homotopy Algebras,” arXiv:math/9907138. 用途：strongly homotopy structures 在 chain homotopy equivalence 下的转移、side conditions 和 homotopy inverse moves。链接：<https://arxiv.org/abs/math/9907138>。
- Sergei Merkulov, “Strong homotopy algebras of a Kähler manifold,” 1999. 用途：同伦转移、最小模型和形式性背景。
- Maxim Kontsevich and Yan Soibelman, “Deformations of algebras over operads and Deligne's conjecture,” 2000. 用途：Deligne 猜想与 deformation theory 背景。

## Dendroidal sets 与 infinity-operads

- Ieke Moerdijk and Ittay Weiss, “Dendroidal Sets,” arXiv:math/0701293. 用途：树范畴 $\Omega$、dendroidal sets 和 dendroidal nerve。链接：<https://arxiv.org/abs/math/0701293>。
- Ieke Moerdijk and Ittay Weiss, “On inner Kan complexes in the category of dendroidal sets,” arXiv:math/0701295. 用途：dendroidal inner Kan 条件。链接：<https://arxiv.org/abs/math/0701295>。
- Denis-Charles Cisinski and Ieke Moerdijk, “Dendroidal sets as models for homotopy operads,” arXiv:0902.1954. 用途：dendroidal sets 的模型结构和 homotopy operads。链接：<https://arxiv.org/abs/0902.1954>。
- Gijs Heuts, Vladimir Hinich, and Ieke Moerdijk, “On the equivalence between Lurie's model and the dendroidal model for infinity-operads,” arXiv:1305.3658; *Advances in Mathematics*. 用途：HHM-1--HHM-5 的 Quillen-equivalence zig-zag；本书采用该来源的 open/no-constants 比较，不用于含 arity $0$ 的默认 operads。链接：<https://arxiv.org/abs/1305.3658>。
- Gijs Heuts, “Algebras over infinity-operads,” arXiv:1110.1776, 2011. 用途：dendroidal sets 中 infinity-operad 代数和 coCartesian fibrations。链接：<https://arxiv.org/abs/1110.1776>。
- Vladimir Hinich, “Rectification of algebras and modules,” arXiv:1311.4130, 2013. 用途：Lurie-style operad algebras 与经典 dg operad algebras 的 rectification 比较。链接：<https://arxiv.org/abs/1311.4130>。
- Francesca Pratali, “A straightening-unstraightening equivalence for infinity-operads,” arXiv:2501.05263, 2025. 用途：operadic straightening-unstraightening 的最新 spaces-valued locator；作为 P1/preprint 边界使用。链接：<https://arxiv.org/abs/2501.05263>。
- Jacob Lurie, *Higher Algebra*. 用途：Lurie-style infinity-operad、operadic fibration、higher algebra；HA-MON-1 = Proposition 4.1.7.4 + Example 4.1.7.6，HA-MON-2 = Corollary 4.1.7.16，支撑模型范畴的 underlying symmetric monoidal infinity-category。作者 PDF：<https://www.math.ias.edu/~lurie/papers/HA.pdf>。
- Jacob Lurie, *Higher Topos Theory*. 用途：quasi-category 和 straightening/unstraightening 背景。arXiv：<https://arxiv.org/abs/math/0608040>。
- Jacob Lurie, *Kerodon*. 用途：homotopy-coherent mathematics 在线参考；使用前需记录具体 tag。链接：<https://kerodon.net/>。

## 几何应用与 factorization

- Kevin Costello and Owen Gwilliam, *Factorization Algebras in Quantum Field Theory*, Volumes 1-2. 用途：prefactorization 乘法、Weiss cosheaf 条件、multiplicativity 与量子场论应用。作者稿链接：<https://people.math.umass.edu/~gwilliam/vol1may8.pdf>。
- David Ayala and John Francis, “Factorization homology of topological manifolds,” *Journal of Topology*, 2015; arXiv:1206.5522v6. 用途：AF-1 excision、AF-2 圆周 Hochschild 计算、AF-3 homology-theory characterization、AF-4 boundary version，以及 AF-5/Proposition 5.1 的 commutative coefficient 公式 $\int_MA\simeq M\otimes A$。链接：<https://arxiv.org/abs/1206.5522>。
- David Ayala, John Francis, and Hiro Lee Tanaka, “Factorization homology of stratified spaces,” 2010s. 用途：stratified factorization homology 和几何 gluing。
- Paul Seidel, *Fukaya Categories and Picard-Lefschetz Theory*, European Mathematical Society, 2008. 用途：Fukaya categories 的 $A_\infty$ 结构和 Picard-Lefschetz 应用。
- Kenji Fukaya, Yong-Geun Oh, Hiroshi Ohta, and Kaoru Ono, *Lagrangian Intersection Floer Theory: Anomaly and Obstruction*, AMS/IP, 2009. 用途：Floer theory、obstructions 和 Fukaya category 构造。
- Sheel Ganatra, John Pardon, and Vivek Shende, “Covariantly functorial wrapped Floer theory on Liouville sectors,” 2010s. 用途：wrapped Fukaya categories、sectorial descent 和几何 gluing。

## 近期研究资料与边界

以下资料只作为研究边界入口。除非后续章节完成独立验证，不得把其中新结果写成正文定理。

- Eric Hoffbeck and Ieke Moerdijk, “Homology of infinity-operads,” arXiv:2105.11943. 用途：infinity-operad 的同调、bar-cobar 和 Koszul 型结构。链接：<https://arxiv.org/abs/2105.11943>。
- Eric Hoffbeck and Ieke Moerdijk, “Koszul duality for algebras over infinity-operads,” arXiv:2602.08851, 2026-02-09. 用途：infinity-operad 上代数的 Koszul 对偶，当前前沿入口。链接：<https://arxiv.org/abs/2602.08851>。
- Daria Pavlova, “Boardman-Vogt tensor product and wreath product of operadic categories,” arXiv:2601.03985, 2026-01-07; v2 updated 2026-05-28. 用途：operadic categories 与 Boardman-Vogt tensor product 的近期工作。链接：<https://arxiv.org/abs/2601.03985>。
- Hang Yuan, “Higher operad structure for Fukaya categories,” arXiv:2603.08039, 2026-03-09. 用途：Fukaya categories 的高阶 operadic 结构。链接：<https://arxiv.org/abs/2603.08039>。
- Kensuke Arakawa, Victor Carmona, and Francesca Pratali, “Relative dendroidal Rezk nerve and applications,” arXiv:2606.11895, 2026-06-10. 用途：relative infinity-operads、localization 和 dendroidal Rezk nerve。链接：<https://arxiv.org/abs/2606.11895>。
- Michael Batanin, Joachim Kock, and Mark Weber, “Operadic categories as (pseudo)-simplicial groupoids,” arXiv:2606.15671, 2026-06-14. 用途：operadic categories 的 operadic nerve 和 simplicial groupoid 形式。链接：<https://arxiv.org/abs/2606.15671>。

最新版本核查记录见 [FRONTIER_SOURCE_AUDIT_2026_06_30.md](FRONTIER_SOURCE_AUDIT_2026_06_30.md)；前一轮记录见 [FRONTIER_SOURCE_AUDIT_2026_06_29.md](FRONTIER_SOURCE_AUDIT_2026_06_29.md)。这些记录不把上述条目提升为正文定理来源，只固定其研究边界状态。

## 当前章节依赖

- [SECOND_PASS_STRICTIFICATION_PLAN.md](SECOND_PASS_STRICTIFICATION_PLAN.md) 不新增数学定理；它记录第二轮审校处理过的定理编号、符号、模型假设和前沿版本核查。
- [DEPENDENCY_GRAPH.md](DEPENDENCY_GRAPH.md) 不新增数学定理；它记录定义、证明和外部输入的有向依赖，防止高级比较定理被倒用。
- [THEOREM_LEDGER.md](THEOREM_LEDGER.md) 不新增数学定理；它按章节区分内部证明、外部输入和只可作为边界说明的内容。
- [INTERNAL_OPERAD_CLOSURE_AUDIT.md](INTERNAL_OPERAD_CLOSURE_AUDIT.md) 不新增数学定理；它审计 operad theory 主体的定义链、类型链和公理链是否内部闭合。
- [INTERNAL_NUMBERING_AND_CROSSREF_AUDIT.md](INTERNAL_NUMBERING_AND_CROSSREF_AUDIT.md) 不新增数学定理；它审计第一至第七章编号和内部交叉引用是否可稳定使用。
- [LABEL_LEDGER_CH01_07.md](LABEL_LEDGER_CH01_07.md) 不新增数学定理；它把第一至第七章已编号声明登记为稳定交叉引用 label。
- [LABEL_LEDGER_CORE_APPENDICES.md](LABEL_LEDGER_CORE_APPENDICES.md) 不新增数学定理；它把核心附录 A/B/H/K/P/U/X 的已编号声明登记为稳定交叉引用 label。
- [LABEL_LEDGER_CH08_21.md](LABEL_LEDGER_CH08_21.md) 不新增数学定理；它把第八至第二十一章的 420 个正式编号项登记为稳定交叉引用 label。
- [LABEL_LEDGER_REMAINING_APPENDICES.md](LABEL_LEDGER_REMAINING_APPENDICES.md) 不新增数学定理；它把剩余附录 C/D/E/F/G/I/J/L/M/N/O/Q/R/S/T/V/W/Y/Z 的 398 个正式编号项登记为稳定交叉引用 label。
- [CROSSREF_REWRITE_AUDIT.md](CROSSREF_REWRITE_AUDIT.md) 不新增数学定理；它记录两轮散文交叉引用到编号引用的替换。
- [PUBLICATION_CLOSURE_MATRIX.md](PUBLICATION_CLOSURE_MATRIX.md) 不新增数学定理；它定义核心可读教材态、基本完本严格草稿态和最终出版态，并记录当前完成状态。
- [PUBLICATION_PROOFING_LEDGER.md](PUBLICATION_PROOFING_LEDGER.md) 不新增数学定理；它记录最终出版校对动作、locator 状态和局部指称判定。
- [REFERENCE_LOCATOR_LEDGER.md](REFERENCE_LOCATOR_LEDGER.md) 不新增数学定理；它把最终出版前必须定位的外部输入分为 P0/P1/P2/R 四类。
- [FINAL_OPERAD_THEORY_CLOSURE.md](FINAL_OPERAD_THEORY_CLOSURE.md) 不新增数学定理；它把剩余工作最终分类为内部证明、外部 locator、边界关闭或出版社级 production work。
- [P0_REFERENCE_LOCATORS_BATCH_1.md](P0_REFERENCE_LOCATORS_BATCH_1.md) 不新增数学定理；它记录 Berger-Moerdijk 与 Cisinski-Moerdijk 相关 P0 外部输入的第一批精确定位。
- [P0_REFERENCE_LOCATORS_BATCH_2.md](P0_REFERENCE_LOCATORS_BATCH_2.md) 不新增数学定理；它记录 Lurie *Higher Topos Theory* straightening/unstraightening 的 P0 精确定位。
- [P0_REFERENCE_LOCATORS_BATCH_3.md](P0_REFERENCE_LOCATORS_BATCH_3.md) 不新增数学定理；它记录 Ayala-Francis factorization homology、excision、圆周 Hochschild 计算、带边界版本和交换系数公式的 AF-0--AF-5 精确定位。
- [P0_REFERENCE_LOCATORS_BATCH_4.md](P0_REFERENCE_LOCATORS_BATCH_4.md) 不新增数学定理；它记录 Ginzburg-Kapranov GK-1--GK-7 与 Loday--Vallette LV-1--LV-3，覆盖 classical quadratic core、connected weight-graded twisting 四项等价、modern quadratic Koszul criterion 和 nonsymmetric rewriting criterion。
- [P0_REFERENCE_LOCATORS_BATCH_5.md](P0_REFERENCE_LOCATORS_BATCH_5.md) 不新增数学定理；它记录 Fresse operadic cobar/cofibrant replacement 和 Hinich dg-operad model context 的 P0 精确定位。Loday--Vallette 现代四项判别已由 batch 4 的 LV-1--LV-2 独立定位。
- [P0_REFERENCE_LOCATORS_BATCH_6.md](P0_REFERENCE_LOCATORS_BATCH_6.md) 不新增数学定理；它记录 Markl strongly homotopy transfer theorem 的 P0 精确定位，并把 HPL 显式公式、tree signs 和 minimal model uniqueness 归入 final sign/convention package。
- [P0_REFERENCE_LOCATORS_BATCH_7.md](P0_REFERENCE_LOCATORS_BATCH_7.md) 不新增数学定理；它记录 Moerdijk-Weiss dendroidal nerve fully faithfulness、$\Delta\subset\Omega$、strict nerve unique fillers 和 homotopy coherent nerve inner Kan 入口的 P0 精确定位。
- [P0_REFERENCE_LOCATORS_BATCH_8.md](P0_REFERENCE_LOCATORS_BATCH_8.md) 不新增数学定理；它记录 White/White--Yau 中 monoidal Bousfield localization preserves operad/colored-operad algebras 的模型范畴版本；strict-to-infinity 和 infinity-categorical comparison 的对应 locator 由批次 9/10 分层补齐。
- [P0_REFERENCE_LOCATORS_BATCH_9.md](P0_REFERENCE_LOCATORS_BATCH_9.md) 不新增数学定理；它记录 Pavlov--Scholbach modern colored admissibility/rectification、symmetric powers 技术条件，以及 Lurie strict-to-infinity algebra comparison 与 underlying symmetric monoidal infinity-category 的精确定位。
- [P0_REFERENCE_LOCATORS_BATCH_10.md](P0_REFERENCE_LOCATORS_BATCH_10.md) 不新增数学定理；它记录 Hinich Dwyer--Kan localization revisited、Heuts--Hinich--Moerdijk dendroidal-Lurie comparison、Lurie category of operators 和 Pratali operadic straightening 的定位。
- [P1_REFERENCE_LOCATORS_FINAL_SWEEP.md](P1_REFERENCE_LOCATORS_FINAL_SWEEP.md) 不新增数学定理；它记录 Dunn additivity、Deligne conjecture、May/Poisson/formality/framed BV 和几何边界 locator 的最终收口。
- [FRONTIER_SOURCE_AUDIT_2026_06_30.md](FRONTIER_SOURCE_AUDIT_2026_06_30.md) 依赖上述 arXiv 条目；用途是记录最新版本边界和禁止未核查定理化。
- [FRONTIER_SOURCE_AUDIT_2026_06_29.md](FRONTIER_SOURCE_AUDIT_2026_06_29.md) 保留为前一轮历史核查记录。
- 序章依赖 May、Boardman-Vogt、Markl-Shnider-Stasheff、Loday-Vallette、Fresse、Moerdijk-Weiss、Cisinski-Moerdijk、Lurie 和 2026 年近期 arXiv 入口。
- 第一章的对称序列、代入乘积、operad、endomorphism operad 和 Ass/Com 例子，主要依赖 Markl-Shnider-Stasheff 与 Loday-Vallette 的基础定义；允许 arity $0$ 的代入以所有有限集映射及空纤维重写，非空分块只用于内层 arity $0$ 为初对象的特例。
- 第二章的自由代数、operad monad 和 Ass/Com 自由代数例子，主要依赖 Markl-Shnider-Stasheff 与 Loday-Vallette 的基础理论。
- 第三章的非对称 operad、偏复合和树收缩口径，主要依赖 Markl-Shnider-Stasheff 与 Loday-Vallette 的树语言。
- 第四章的自由 operad、装饰树和生成元关系，主要依赖 Markl-Shnider-Stasheff、Loday-Vallette 和 Fresse 的标准构造。
- 第五章的 colored operad、multicategory 和带类型代数系统，主要依赖 Markl-Shnider-Stasheff、Loday-Vallette 和 Leinster 的多范畴语言。
- 第六章的线性 operad、Schur functor、Ass/Com/Lie/Pois 例子，主要依赖 Loday-Vallette、Ginzburg-Kapranov 和 Fresse。
- 第七章的 PROP、properad 和 wheeled 变体，主要依赖 Markl-Shnider-Stasheff、Loday-Vallette 和 Fresse。
- 第八章的二次 operad、Ginzburg-Kapranov 对偶和 Koszul 性，主要依赖 Ginzburg-Kapranov、Loday-Vallette 和 Fresse；classical Koszul core 由 GK-1--GK-5 定位，现代 $\mathcal P^¡=\mathcal C(sE,s^2R)$ 四项判别由 LV-2 定位。
- 第九章的 dg-operad、cooperad、bar-cobar 构造和 twisting morphism，主要依赖 Ginzburg-Kapranov、Loday-Vallette 和 Fresse；LV-1 精确定位 connected weight-graded twisting-morphism 四项等价，GK-6--GK-7 给出 classical dg-dual core，FRE-1--FRE-4 另行控制带 cofibrancy 假设的 modern model-category 版本。
- 第十章的 $A_\infty$、$L_\infty$、$C_\infty$ 与 $E_n$-operad，主要依赖 Stasheff、May、Cohen、Loday-Vallette 和 Fresse；recognition principle、$H_\*(E_n)\cong\operatorname{Pois}_n$ 和形式性均作为外部输入。
- 第十一章的 Gerstenhaber、BV 和 Deligne 猜想，主要依赖 Gerstenhaber、May-Cohen、Getzler-Jones、Kontsevich-Soibelman、Tamarkin、McClure-Smith、Berger-Fresse 和 Fresse；Deligne 猜想 locator 已由 MS-1--MS-3 与 BF-1--BF-4 定位，framed $E_2$ 同调仍作为 P1 边界。
- 第十二章的 brace operad 与 Hochschild cochains，主要依赖 Gerstenhaber、Voronov、McClure-Smith、Berger-Fresse 和 Kontsevich-Soibelman；brace/surjection operad 与 $E_2$ 链模型 locator 已由 MS-1--MS-3 与 BF-1--BF-3 定位，具体 suspended signs 仍需附录 W 核对。
- 第十三章的同伦转移和最小模型，主要依赖 Kadeishvili、Merkulov、Markl、Loday-Vallette 和 Fresse；Markl strongly homotopy transfer existence 已由 MHT-1--MHT-8 定位，homological perturbation lemma 显式公式、tree signs 和最小模型唯一性按最终收口文件保留为 sign/convention package。
- 第十四章的模型范畴中 operad、admissibility 和 rectification 主要依赖 Berger-Moerdijk、Hinich、Fresse、Pavlov-Scholbach；Berger--Moerdijk/Hinich/Fresse 基础 locator 已由批次 1/5 定位，Pavlov--Scholbach modern colored admissibility/rectification 已由 PSAR-1--PSAR-6 与 PSP-1--PSP-2 定位。Boardman--Vogt resolution 只保留为外部边界 14.31，未登记整套 $W$-construction 的可调用 theorem locator。
- 第十五章的 simplicial/topological operads、Sing-realization 比较和 little cubes operad，主要依赖 Quillen、Goerss-Jardine、Hovey、May、Boardman-Vogt、Berger-Moerdijk 和 Fresse；两类 operad transferred 模型结构使用 BM-1。Top 与 sSet 的底范畴 Quillen equivalence 不自动提升到 operad categories，后者暂作外部边界 15.18。
- 第十六章的树范畴 $\Omega$、dendroidal sets、dendroidal nerve、Segal core、faces 和 horns，主要依赖 Moerdijk-Weiss、Cisinski-Moerdijk 和 Heuts-Hinich-Moerdijk；dendroidal nerve fully faithfulness 和 $\Delta\hookrightarrow\Omega$ fully faithfulness 已由 MW-1--MW-2 定位，树范畴分解理论仍作为外部输入。
- 第十七章的 dendroidal inner Kan 条件、normal monomorphisms、inner anodynes 和 operadic model structure，主要依赖 Moerdijk-Weiss 与 Cisinski-Moerdijk；strict nerve unique fillers 已由 MW-4 定位，operadic model structure 已由 CM-1--CM-4 定位，weak equivalences 和 nerve/model comparison 仍作为外部输入。
- 第十八章的 Lurie-style infinity-operads、inert/active maps、symmetric monoidal infinity-categories 和 algebras over infinity-operads，主要依赖 Lurie、Heuts、Hinich 和 Heuts-Hinich-Moerdijk；category of operators nerve 已由 HA-OP-1--HA-OP-3 定位，dendroidal-Lurie 模型比较已由 HHM-1--HHM-5 定位，coCartesian fibration 技术仍按 Lurie/Hinich 外部输入处理。
- 第十九章的 Dwyer-Kan localization、模型范畴的 underlying infinity-category、straightening/unstraightening、monoidal localization 和 operadic localization，主要依赖 Dwyer-Kan、Barwick-Kan、Hovey、Hirschhorn、Lurie、Hinich、Pavlov-Scholbach、White 和 White--Yau；ordinary straightening 由 HTT-1 定位，underlying symmetric monoidal infinity-category 由 HA-MON-1--HA-MON-2 定位，Bousfield localization preservation 由 WHT-1--WHT-4 与 WY-1--WY-3 定位，Dwyer--Kan/coherent nerve 由 DKR-1--DKR-7 定位，strict-to-infinity algebra comparison 由 PSAR-5--PSAR-6 与 HA-ALG-1--HA-ALG-3 定位，spaces-valued operadic straightening 由 PRA-1--PRA-5 作 P1/preprint locator 定位。
- 第二十章的 factorization algebras、factorization homology、Dunn additivity、Fukaya categories 和几何 gluing，主要依赖 Costello-Gwilliam、Ayala-Francis、Lurie、Seidel、Fukaya-Oh-Ohta-Ono 和 Ganatra-Pardon-Shende；excision/圆周/边界/交换系数基础版本已由 AF-0--AF-5 定位，Dunn additivity 已由 DUNN-1 定位，locally constant multiplicative factorization algebra 与 $E_n$ 的比较、Fukaya category 构造和几何 gluing 定理仍作为外部几何边界。
- 附录 A 的 universes、finite set groupoids、symmetric group actions、coinvariants 和 coends 使用标准集合论、范畴论和表示论事实；Maschke 定理作为外部基础事实。
- 附录 B 的分块、代入乘积、arity coinvariants 公式和树代入主要依赖第一至第四章已采用的 Markl-Shnider-Stasheff、Loday-Vallette 和有限集群胚口径；证明在附录中直接给出。
- 附录 C 的模型范畴、weak factorization systems、Quillen adjunctions、Quillen equivalences 和 monoidal model categories 主要依赖 Quillen、Hovey 和 Hirschhorn；homotopy category 计算和 Quillen equivalence 判别作为标准外部基础事实。
- 附录 D 是正文外部输入定理的索引和引用包账本，不新增数学定理；最终版需要补全精确文献定位。
- 附录 E 的 Koszul sign rule、tensor differential、suspension、operadic suspension、Hochschild signs 和 suspended brace signs 主要依赖 Loday-Vallette、Fresse、Gerstenhaber-Voronov 和标准 dg category 约定；最终版仍需把这些符号与所选 brace/$E_2$ 链模型逐项核对。
- 附录 F 的 Ass、Com、Endomorphism、Lie 和 Poisson 例子主要依赖第一章、第六章、自由 operad 的生成元关系以及 Loday-Vallette 的经典例子；PBW、自由 Lie 代数模型和 little cubes 同调识别均作为外部输入。
- 附录 G 的模型结构假设、admissibility 和 rectification 检查表主要依赖 Berger-Moerdijk、Hinich、Fresse、Pavlov-Scholbach、Hovey、Hirschhorn 和 Lurie；G.11、G.12、G.13 分别固定 BM-1、PSAR-2、PSAR-4 假设包，G.16 内部证明正特征自由 commutative algebra 不保持一个 trivial cofibration。
- 附录 H 的平面树、叶标号树、自由 operad 树群胚商和 Moerdijk-Weiss 树范畴对照主要依赖 Markl-Shnider-Stasheff、Loday-Vallette、Fresse 和 Moerdijk-Weiss；自由性证明在附录中直接给出，dendroidal nerve fully faithfulness 已由 MW-2 定位。
- 附录 I 的 convolution Lie algebra、twisting morphism、twisted composite products、Koszul complex 和权重滤过主要依赖 Ginzburg-Kapranov、Loday-Vallette 和 Fresse；四项等价由 LV-1--LV-2 精确定位，classical 判别由 GK-3/GK-7 交叉核对，带 cofibrancy 的 model-category 版本由 FRE-2--FRE-4 控制。Bar 使用递增顶点滤过、cobar 使用递减顶点滤过；connectedness 保证逐 arity 有限，非 connected 完成化由反例 I.22.1 隔离。
- 附录 J 的 normalized contraction、$A_\infty/L_\infty$ 转移树公式和 minimal model 唯一性主要依赖 Kadeishvili、Gugenheim-Lambe-Stasheff、Markl、Merkulov、Loday-Vallette 和 Fresse；Markl existence theorem 已由 MHT-1--MHT-8 定位，完整 signs 与高阶恒等式按最终收口文件作为 sign/convention package。
- 附录 K 的 colored operad、自由 colored operad、模/双模例子和 enriched 版本主要依赖 Leinster、Markl-Shnider-Stasheff、Loday-Vallette、Berger-Moerdijk 和 Pavlov-Scholbach；enriched/colored admissibility 的 modern locator 已由 PSAR-1--PSAR-3 定位，但具体 enriched 底范畴仍需单独假设表。
- 附录 L 的 $\mathcal P_\infty$、suspended $A_\infty/L_\infty$、$E_n$、Poisson 同调、additivity 和 rectification 边界主要依赖 Stasheff、May、Cohen、Ginzburg-Kapranov、Loday-Vallette、Fresse、Dunn 和 Lurie；Dunn/Lurie additivity 已由 DUNN-1 定位，Poisson 同调和形式性仍作为 P1 边界。
- 附录 M 的 strict/dendroidal/Lurie/model category 比较图主要依赖 Moerdijk-Weiss、Cisinski-Moerdijk、Heuts-Hinich-Moerdijk、Lurie、Hinich、Pavlov-Scholbach、White 和 White--Yau；HHM-1--HHM-5 只支撑 open/no-constants dendroidal--Lurie zig-zag，HA-OP-1--HA-OP-3 是允许 constants 的独立 category-of-operators entry，PSAR/HA-ALG 支撑相应 strict-to-infinity algebra comparison。
- 附录 N 的 factorization homology、excision、圆周计算、切结构和 Fukaya gluing 边界主要依赖 Costello-Gwilliam、Ayala-Francis、Lurie、Seidel、Fukaya-Oh-Ohta-Ono 和 Ganatra-Pardon-Shende；AF-1、AF-2、AF-4、AF-5 分别定位 excision、圆周、boundary 和 commutative coefficients。球面计算使用 open collar pieces；Fukaya 型 gluing 保持为研究边界 N.30。
- 附录 O 的失败模式和反例边界不新增外部定理；它汇总附录 A、G、L、M、N 以及第十四至二十章中的不可混用约定。
- 附录 P 的低阶计算主要依赖第一、六、十、十二、十六、十七、二十章和附录 B、E、L、N；除 dendroidal/factorization 比较仍按正文标为外部输入外，其余计算在附录中直接给出。
- 附录 Q 的 Koszul complex、bar-cobar differential 和低权重计算主要依赖定义 8.4--定义 8.16、定义 9.14--定理 9.20 和定义 I.11--命题 I.21；完整 Koszul 四项判别由 LV-1--LV-2 定位，$\operatorname{Ass}_{ns}$ 的 rewriting/Koszul 步骤由 LV-3 定位，bar-cobar 的 model-context/cofibrancy 入口由 FRE-4--FRE-5 分开定位。
- 附录 R 的模型范畴案例主要依赖第十四章和附录 C、G、O；transferred model structure、admissibility、rectification 和 localization preservation 已有 BM/HIN/FRE/PSAR/White/White--Yau locator，只可用于对应模型范畴假设下的 operad/colored-operad algebra preservation。
- 附录 S 的同伦转移低阶计算主要依赖定义 13.1--外部输入定理 13.16、定义 E.18--定义 E.23、定义 J.1--外部输入定理 J.19、定义 L.4--定义 L.7 和附录 R 的 rectification 案例；Markl homotopy transfer existence 已定位，minimal model 唯一性、完整 sign convention 和 strict dg formality rectification 仍作为外部输入。
- 附录 T 的 dendroidal horn、Segal core 和 normality 样例主要依赖第十六、十七章和附录 M、O、P；fully faithfulness 和 strict nerve unique fillers 已由 MW-2、MW-4 定位，normal monomorphism 与 erratum 影响仍作为外部输入。
- 附录 U 的 PROP/properad 图计算主要依赖第七章和附录 O；自由 properad/PROP、Frobenius PROP 完整构造和 wheeled graph complex 仍作为外部输入或后续专题。
- 附录 V 的带边界与分层 factorization homology 样例主要依赖定义 20.3--研究边界 20.23、定义 N.3--研究边界 N.30 和错误命题 O.23--正确边界 O.28；V.3/V.6/V.8 已标为外部边界，V.11 已标为研究边界，不作为已定位定理调用。
- 附录 W 的符号交叉核对主要依赖约定 E.1--说明 E.25、定义 J.1--警告 J.20、定义 L.1--说明 L.20、命题 P.1--说明 P.9 和定义 S.1--说明 S.13；它不新增数学定理，只固定同调分次下的转换流程。
- 附录 X 的具体代数例子主要依赖附录 A、F、O、R、V 和第十二、十四、二十章；正特征 rectification、Morita invariance 和带边界 factorization homology 仍按相关附录作为外部输入或边界说明；$\operatorname{Sym}^p$ 不保持 acyclic complex 的计算在附录 X 内部给出。
- 附录 Y 的 infinity-operadic homology 与 Koszul 前沿接口主要依赖第八、九、十六至十九章和附录 D、I、M、Q；Hoffbeck-Moerdijk 的 infinity-operadic 结果仍作为研究边界，附录内部只证明 strict operad 的树指标线性化和特化检验。
- 附录 Z 的 operadic categories、relative dendroidal Rezk nerve 与 Fukaya 前沿接口主要依赖第五、十六至二十章和附录 D、M、N、O、V；Pavlova、Arakawa-Carmona-Pratali、Batanin-Kock-Weber 和 Yuan 的近期结果仍作为研究边界。

## 引用边界

- 本书严格草稿中的基础定义可由多部教材共同核对。
- 大型结构定理，例如自由 properad/PROP 的图群胚公式、Koszul 判别、bar-cobar Quillen 等价、Berger-Moerdijk 模型结构、Cisinski-Moerdijk dendroidal model structure、Lurie 与 dendroidal 模型比较，应在相应章节标为“外部输入定理”或给出完整证明路线。
- 2025-2026 预印本只在研究边界章节说明其问题、对象和大致贡献；正文定理化需要二次核验。
