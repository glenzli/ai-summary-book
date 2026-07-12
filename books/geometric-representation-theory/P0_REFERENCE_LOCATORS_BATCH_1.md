# P0 theorem locator 第一批

本文件给出正式教材化第一批 P0 外部输入定位。当前定位足以防止正文无源调用；出版级版本仍需补页码和版本交叉检查。

## 1. Beilinson-Bernstein localization

**BB-1.** Beilinson and Bernstein, *A proof of Jantzen conjectures*, I. M. Gelfand Seminar, Adv. Soviet Math. 16, Part 1, 1993.  
定位：Theorem 3.3.1 常作为 localization theorem 的定位入口；twisted version 需要同时记录 $\lambda$、$\rho$ shift 和 regular dominant 条件。  
本书用途：第八章 localization equivalence，第九章 wall crossing 几何解释，第八章到 KL character formula 的证明链。  
假设翻译：$k=\mathbb C$，$G$ reductive，$\mathcal B=G/B$，$\mathcal D_\lambda$ 为 twisted differential operators，$\lambda$ regular dominant。

## 2. Kazhdan-Lusztig 和 KL-IC

**KL-1.** Kazhdan and Lusztig, *Representations of Coxeter groups and Hecke algebras*, Invent. Math. 53, 1979.  
定位：Hecke algebra、KL basis、KL polynomials 的原始定义。  
本书用途：第四章 Hecke algebra 和 KL basis。

**KL-IC-1.** Kazhdan and Lusztig, *Schubert varieties and Poincare duality*, Proc. Sympos. Pure Math. 36, 1980；以及 *A topological approach to Springer's representations*, Adv. Math. 38, 1980。  
定位：Schubert varieties 的 intersection cohomology 与 KL polynomials 的关系。  
本书用途：第四章 IC sheaves 与 KL basis，第八章 KL character formula 的几何证明链。

## 3. BBD perverse sheaves 和 decomposition theorem

**BBD-1.** Beilinson, Bernstein, Deligne, *Faisceaux pervers*, Asterisque 100, 1982。
定位：§§2.1--2.2 的 perverse t-structures，§4.3 的 simple perverse sheaves 与 middle-extension classification；closed-embedding t-exactness 按 §4 formalism 调用。
本书用途：第三章 perverse sheaves、第四章 IC sheaves、第五章 Springer sheaf 分解。

**BBD-2.** 同上，decomposition theorem 定位到 6.2.5。
本书用途：Springer sheaf semisimplicity、KL-IC purity/semisimplicity，以及第十二章 characteristic-zero $GL_2$ semismall convolution。正文只对 $\operatorname{IC}_X$ 陈述默认 Betti 版本，并把 projective relative hard Lefschetz 单列。

**BBD-SS-1.** de Cataldo and Migliorini, *The hard Lefschetz theorem and the topology of semismall maps*, Ann. Sci. Ecole Norm. Sup. 35 (2002), 759--772。
定位：proper semismall maps、relevant strata、intersection forms 与 semismall decomposition。
本书用途：附录 C.7；第十二章 12.18 的 $GL_2$ convolution multiplicity-one decomposition。

## 4. Riemann-Hilbert

**RH-1.** Kashiwara, Mebkhout；教材入口：Hotta-Takeuchi-Tanisaki, *D-Modules, Perverse Sheaves, and Representation Theory*；Kashiwara-Schapira, *Sheaves on Manifolds*。  
定位：regular holonomic $\mathcal D$-modules 与 constructible sheaves/perverse sheaves 的 Riemann-Hilbert correspondence。  
本书用途：第七章、第八章。

## 5. Springer theory

**SPR-1.** Springer, Borho-MacPherson, Kazhdan-Lusztig；教材入口：Chriss-Ginzburg, *Representation Theory and Complex Geometry*。  
定位：Springer resolution、nilpotent-orbit fiber-dimension/semismallness、Springer sheaf 的 $W$-作用、Springer correspondence、Steinberg variety convolution。
本书用途：第五、六章。

## 6. Geometric Satake

**GSAT-1.** Mirkovic and Vilonen, *Geometric Langlands duality and representations of algebraic groups over commutative rings*, arXiv:math/0401222, v5, 2018-02-13。
定位：主等价 (1.1)；Theorem 12.1 识别 dual root datum。
本书用途：第十三章 13.6 和附录 I。
假设翻译：原文在 complex affine Grassmannian 的 classical topology 上允许更一般 coefficient rings；正文固定代数闭 characteristic-zero field $E$、reduced ind-scheme 和 finite support。

**AFFGR-1 / AFFGR-CONV-1.** 同一论文 §2 给出 reduced affine Grassmannian、orbit 与 closure convention；§4 的 Proposition 4.2、Lemma 4.4 给出 convolution perversity 与 stratified semismall estimate。
本书用途：第十二章 representability/orbits、finite-support properness 和 convolution 定义。

**GSAT-CONV-1 / GSAT-FIBER-1 / GSAT-WEIGHT-1.** Proposition 4.6 定位 associativity；§5 和 §6 定位 fusion 及 parity-corrected commutativity；Theorem 3.6、Corollary 3.7、Proposition 6.3、Proposition 6.4 定位 weight decomposition、faithfulness 和 tensor compatibility。
本书用途：第十三章 13.2、13.4、13.11 和附录 I。

**TANNAKA-1.** Saavedra Rivano, *Categories Tannakiennes*, Lecture Notes in Mathematics 265, 1972；Deligne--Milne, *Tannakian Categories*, Lecture Notes in Mathematics 900, 1982。
定位：neutral Tannakian reconstruction。
本书用途：第十三章 13.8；只从 fiber functor 构造 affine group scheme，不用于证明 Satake group 的 reductivity 或 root datum。

## 7. Soergel 和 Elias-Williamson

**SOERGEL-1.** Soergel, papers on bimodules and Kazhdan-Lusztig theory。  
用途：第十一章 Soergel categorification theorem。

**EW-1.** Elias and Williamson, *The Hodge theory of Soergel bimodules*, arXiv:1212.0791。  
用途：Soergel conjecture、KL positivity、Hodge-Riemann relations for Soergel bimodules。

## 8. BFN Coulomb branches

**BFN-1.** Braverman, Finkelberg, Nakajima, *Towards a mathematical definition of Coulomb branches of 3-dimensional $\mathcal N=4$ gauge theories, II*, arXiv:1601.03586。  
用途：第二十章 BFN space、Borel-Moore convolution、Coulomb branch algebra 和 quantization。

## 9. 当前限制

本批 locator 仍是“定理包级定位”，不是最终页码级定位。正式出版前还需：

1. 精确页码、定理编号或章节编号；
2. 版本差异检查；
3. 每条假设翻译到 `MODEL_HYPOTHESES_MATRIX.md`；
4. 正文每次调用处加稳定 label。
