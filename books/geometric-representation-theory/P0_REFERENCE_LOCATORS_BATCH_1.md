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
定位：perverse t-structures、middle extension、simple perverse sheaves 分类。  
本书用途：第三章 perverse sheaves、第四章 IC sheaves、第五章 Springer sheaf 分解。

**BBD-2.** 同上，decomposition theorem 入口通常定位到 BBD 的 decomposition theorem 部分，常用编号为 6.2.5。  
本书用途：Springer sheaf semisimplicity、KL-IC purity/semisimplicity、geometric Satake 的部分结构。

## 4. Riemann-Hilbert

**RH-1.** Kashiwara, Mebkhout；教材入口：Hotta-Takeuchi-Tanisaki, *D-Modules, Perverse Sheaves, and Representation Theory*；Kashiwara-Schapira, *Sheaves on Manifolds*。  
定位：regular holonomic $\mathcal D$-modules 与 constructible sheaves/perverse sheaves 的 Riemann-Hilbert correspondence。  
本书用途：第七章、第八章。

## 5. Springer theory

**SPR-1.** Springer, Borho-MacPherson, Kazhdan-Lusztig；教材入口：Chriss-Ginzburg, *Representation Theory and Complex Geometry*。  
定位：Springer resolution、Springer sheaf 的 $W$-作用、Springer correspondence、Steinberg variety convolution。  
本书用途：第五、六章。

## 6. Geometric Satake

**GSAT-1.** Mirkovic and Vilonen, *Geometric Langlands duality and representations of algebraic groups over commutative rings*, arXiv:math/0401222。  
定位：geometric Satake equivalence，perverse sheaves on affine Grassmannian 与 representations of $G^\vee$。  
本书用途：第十三章和附录 I。  
假设翻译：$G$ reductive，$\operatorname{Gr}_G$ affine Grassmannian，$L^+G$-equivariant perverse sheaves，系数可比本书默认域更一般，正文需按 $E$ 特化。

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

