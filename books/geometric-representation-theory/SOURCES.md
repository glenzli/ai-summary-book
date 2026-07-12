# 资料源、用途和边界

本文件记录本书的主要资料源。正文引用大型结论时，应同时检查 `THEOREM_LEDGER.md` 和后续附录 D 的精确 locator。

## 0. 使用分类

- **基础定义**：可作为定义和约定来源。
- **核心定理**：正文可使用，但必须标注为外部输入，除非本书给出完整证明。
- **研究边界**：只用于说明方向，不进入基础证明链。
- **历史说明**：用于说明发展脉络，不作为数学证明依据。

## 1. 经典代数群和表示

- Borel, *Linear Algebraic Groups*。用途：reductive groups、Borel subgroups、flag varieties、Bruhat decomposition。分类：基础定义和核心定理。
- Springer, *Linear Algebraic Groups*。用途：代数群基础、根数据、旗簇。分类：基础定义和核心定理。
- Humphreys, *Introduction to Lie Algebras and Representation Theory*；*Representations of Semisimple Lie Algebras in the BGG Category O*。用途：Lie 代数、Verma modules、category $\mathcal O$。分类：基础定义。
- Jantzen, *Representations of Algebraic Groups*。用途：代数群表示、highest weight theory。分类：核心定理。

## 2. Sheaves、D-modules 和几何表示论教材

- Beilinson, Bernstein, Deligne, *Faisceaux pervers*, Asterisque 100, 1982, https://www.numdam.org/item/AST_1982__100__1_0/。用途：`BBD-1` perverse t-structure、middle extension、simple objects；`BBD-2` decomposition theorem，正文定位到 6.2.5。分类：核心定理。
- Bernstein and Lunts, *Equivariant Sheaves and Functors*, Lecture Notes in Mathematics 1578, Springer, 1994。用途：Betti equivariant derived categories、finite-dimensional approximations、forgetful t-structure。分类：基础定义和核心 formalism。
- Kashiwara and Schapira, *Sheaves on Manifolds*。用途：constructible sheaves、microlocal language、Verdier duality。分类：基础定义和核心定理。
- Hotta, Takeuchi, Tanisaki, *D-Modules, Perverse Sheaves, and Representation Theory*。用途：D-modules、Riemann-Hilbert、localization 语境。分类：基础定义和核心定理。
- Chriss and Ginzburg, *Representation Theory and Complex Geometry*。用途：equivariant K-theory、Steinberg variety、Springer correspondence、Hecke algebras。分类：基础定义和核心定理。
- de Cataldo and Migliorini, *The hard Lefschetz theorem and the topology of semismall maps*, Ann. Sci. Ecole Norm. Sup. 35 (2002), 759--772, https://numdam.org/articles/10.1016/s0012-9593%2802%2901108-4/。用途：`BBD-SS-1` semismall decomposition、relevant strata 和 intersection-form nondegeneracy；第十二章 $GL_2$ convolution。分类：核心定理。
- Saavedra Rivano, *Categories Tannakiennes*, Lecture Notes in Mathematics 265, Springer, 1972；Deligne and Milne, *Tannakian Categories*, in *Hodge Cycles, Motives, and Shimura Varieties*, Lecture Notes in Mathematics 900, Springer, 1982。用途：`TANNAKA-1` neutral Tannakian reconstruction；只构造 affine group scheme，不承担 Satake root datum 识别。分类：核心外部输入。

## 3. Kazhdan-Lusztig、localization 和 Springer 理论

- Kazhdan and Lusztig, *Representations of Coxeter groups and Hecke algebras*。用途：Hecke algebra、Kazhdan-Lusztig basis。分类：核心定理。
- Kazhdan and Lusztig, *A topological approach to Springer's representation*。用途：Springer representation 和 intersection cohomology 接口。分类：核心定理。
- Beilinson and Bernstein, *Localisation de g-modules*；*A proof of Jantzen conjectures*。用途：Beilinson-Bernstein localization、Kazhdan-Lusztig conjectures。分类：核心定理。
- Brylinski and Kashiwara, *Kazhdan-Lusztig conjecture and holonomic systems*。用途：Kazhdan-Lusztig conjecture 的 D-module 证明。分类：核心定理。
- Lusztig, *Intersection cohomology complexes on a reductive group*。用途：character sheaves。分类：核心定理和研究边界。
- Joseph, primitive ideals and associated varieties 相关工作。用途：primitive ideals、Goldie rank、Joseph theory。分类：核心定理入口。
- Borho and Brylinski, primitive ideals、nilpotent orbits 和 characteristic varieties 相关工作。用途：第十章 associated varieties 和 localization 接口。分类：核心定理入口。

## 4. Geometric Satake 和仿射几何

- Mirkovic and Vilonen, *Geometric Langlands duality and representations of algebraic groups over commutative rings*, arXiv:math/0401222, v5 revised 2018-02-13, https://arxiv.org/abs/math/0401222。用途：`AFFGR-1` (§2 affine Grassmannian/orbits)、`GSAT-WEIGHT-1` (Theorem 3.6)、`GSAT-CONV-1` (Proposition 4.2, Lemma 4.4, Proposition 4.6, §§5--6)、`GSAT-FIBER-1` (Corollary 3.7, Proposition 6.3)、`GSAT-1` (主等价 (1.1), Theorem 12.1)。分类：核心定理。
- Ginzburg, affine Grassmannian 和 perverse sheaves 相关工作。用途：geometric Satake 早期构造。分类：核心定理。
- Zhu, *An introduction to affine Grassmannians and the geometric Satake equivalence*, arXiv:1603.05593, https://arxiv.org/abs/1603.05593。用途：`AFFGR-1`、`AFFGR-CONV-1` 的现代讲义入口及 geometric Satake；mixed-characteristic results 只作另一模型入口。分类：基础定义和核心定理入口。
- Iwahori and Matsumoto, affine Hecke algebra 相关原始工作。用途：affine Hecke algebra、Iwahori double cosets。分类：核心定理入口。
- Lusztig, affine Hecke algebras、character sheaves 和 cells 相关工作。用途：affine Kazhdan-Lusztig theory、character sheaves、cells。分类：核心定理和研究边界。
- Beilinson and Drinfeld, affine Grassmannian、factorization 和 chiral algebras 相关工作。用途：affine Grassmannian、geometric Satake、Kac-Moody localization 和 geometric Langlands 背景。分类：核心定理入口和研究边界。

## 5. 范畴化、辛几何和 Coulomb branches

- Soergel, bimodules and Kazhdan-Lusztig theory。用途：Hecke categorification。分类：核心定理。
- Elias and Williamson, *The Hodge theory of Soergel bimodules*, arXiv:1212.0791, https://arxiv.org/abs/1212.0791。用途：Soergel bimodule Hodge 理论、Soergel conjecture 和 KL 正性。分类：核心定理。
- Nakajima, quiver varieties and Kac-Moody representations。用途：quiver varieties 的表示构造。分类：核心定理。
- Rouquier, Khovanov-Lauda-Rouquier algebras。用途：categorification of quantum groups。分类：核心定理。
- Kang and Kashiwara, *Categorification of Highest Weight Modules via Khovanov-Lauda-Rouquier Algebras*, arXiv:1102.4677, https://arxiv.org/abs/1102.4677。用途：cyclotomic KLR categorification。分类：核心定理入口。
- Braden, Proudfoot, Webster, *Quantizations of conical symplectic resolutions I*, arXiv:1208.3863；Braden, Licata, Proudfoot, Webster, *Quantizations of conical symplectic resolutions II*, arXiv:1407.0964。用途：symplectic resolution category $\mathcal O$ 和 symplectic duality。分类：核心定理和研究边界。
- Braverman, Finkelberg, Nakajima, *Towards a mathematical definition of Coulomb branches of 3-dimensional $\mathcal N=4$ gauge theories, II*, arXiv:1601.03586, https://arxiv.org/abs/1601.03586。用途：Coulomb branch 的 BFN 构造。分类：研究边界到核心定理之间，进入正文前需补 locator。
- Braden, Licata, Proudfoot, Webster, symplectic duality。用途：symplectic duality 语言。分类：研究边界，进入正文前需补 locator。
- Lusztig, *Introduction to Quantum Groups* 及 canonical basis 相关论文。用途：quantum groups、canonical bases、quiver variety/perverse sheaf 模型。分类：核心定理入口。
- Kashiwara, crystal bases 和 global basis 相关工作。用途：crystal bases、global crystal basis。分类：核心定理入口。
- Kontsevich and Soibelman, cohomological Hall algebra 和 Donaldson-Thomas theory 相关工作。用途：CoHA、wall-crossing 和 BPS 结构。分类：研究边界和核心定理入口。
- Davison and Meinhardt, critical CoHA、vanishing cycles 和 BPS Lie algebra 相关工作。用途：第二十一章 critical CoHA 和 DT 接口。分类：研究边界和核心定理入口。
- Juteau, Mautner and Williamson, parity sheaves。用途：modular representation theory、parity sheaves、$p$-canonical basis 边界。分类：研究边界和后续核心定理入口。
- Williamson, torsion phenomena and $p$-canonical bases 相关工作。用途：torsion explosion、modular KL 边界。分类：研究边界。

## 6. Geometric Langlands 近期边界

- Gaitsgory and Raskin, *Proof of the geometric Langlands conjecture I: construction of the functor*, arXiv:2405.03599, submitted 2024-05-06, https://arxiv.org/abs/2405.03599。用途：2024 geometric Langlands proof series 的入口。分类：研究边界。
- Arinkin, Beraldo, Campbell, Chen, Faergeman, Gaitsgory, Lin, Raskin, Rozenblyum, *Proof of the geometric Langlands conjecture II: Kac-Moody localization and the FLE*, arXiv:2405.03648, submitted 2024-05-06, https://arxiv.org/abs/2405.03648。用途：critical level FLE、Kac-Moody localization、factorization categories。分类：研究边界。
- Gaitsgory and Raskin, *Proof of the geometric Langlands conjecture V: the multiplicity one theorem*, arXiv:2409.09856, submitted 2024-09-15, current viewed text dated 2026-03-22, https://arxiv.org/abs/2409.09856。用途：proof series 的末端和 multiplicity one。分类：研究边界。

## 7. 使用限制

1. 上述资料源在本阶段只给出全书方向和外部输入入口。除非章节内给出完整证明，本书不得把大型结果改写成内部定理。
2. 2024-2026 geometric Langlands proof series 只进入研究边界。若后续要写入正文定理链，必须建立独立 locator、版本、假设翻译和 proof-dependency ledger。
3. Coulomb branches、symplectic duality、categorical Langlands 等方向需要明确 dg/infinity category 模型；在模型未固定前，只能作为接口章节。
