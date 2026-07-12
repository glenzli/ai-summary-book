# 资料源与用途

本文件记录《Chromatic Homotopy Theory》教材的资料源。正文中的非平凡外部输入必须能在本文件和 `THEOREM_LEDGER.md` 中追溯。

## 1. 基础谱和稳定同伦

- J. F. Adams, *Stable Homotopy and Generalised Homology*。用途：广义同调、谱、Adams spectral sequence 的基础背景。
- M. Hovey, J. H. Palmieri, N. P. Strickland, *Axiomatic Stable Homotopy Theory*。用途：稳定同伦范畴、Bousfield 类、localization 和 thick/localizing 子范畴语言。
- D. C. Ravenel, *Complex Cobordism and Stable Homotopy Groups of Spheres*, 2nd ed., AMS Chelsea，[官方 PDF](https://www.sas.rochester.edu/mth/sites/doug-ravenel/mybooks/ravenel.pdf)。用途：$BP$ retract 与系数（Theorem 4.1.12(c), p. 108；Theorem 4.1.18(a), p. 111；Theorem A2.1.25, p. 349）、Hazewinkel/Araki convention（(A2.2.1) 与 Theorem A2.2.3, pp. 354--355）、Landweber 判据（Chapter 4, Section 2, pp. 115--116）、Adams--Novikov spectral sequence（Theorem 4.4.1, p. 130）和 chromatic spectral sequence（Definition 5.1.7、Proposition 5.1.8, p. 150）。
- D. C. Ravenel, *Nilpotence and Periodicity in Stable Homotopy Theory*。用途：Ravenel conjectures、nilpotence、periodicity、chromatic convergence、telescope conjecture 的历史和定理框架。
- M. Hovey, *Bousfield Localization Functors and Hopkins' Chromatic Splitting Conjecture*, Contemporary Mathematics 181 (1995), 225--250。用途：$E(n)$ 的 Bousfield 类（Corollary 1.12）、finite localization 和 chromatic splitting 背景。
- J. Lurie, *Chromatic Homotopy Theory* lecture notes (2010), Lecture 23。用途：$\langle E(n)\rangle=\langle E(n-1)\vee K(n)\rangle$（Proposition 2）、smash product theorem（Theorem 4）和任意谱上的 fracture square（Proposition 5）。
- J. Lurie, *Chromatic Homotopy Theory* lecture notes (2010), Lecture 32。用途：finite $p$-local spectra 的 chromatic convergence（Theorem 1）及 Adams--Novikov filtration 证明路线的交叉核对。
- T. Barthel and A. Beaudry, *Chromatic structures in stable homotopy theory*, arXiv:1901.09004。用途：现代综述和结构图；只作导引和交叉核对。

## 2. 形式群、复定向和 Brown-Peterson theory

- D. Quillen, *On the formal group laws of unoriented and complex cobordism theory*, Bulletin of the American Mathematical Society 75 (1969), 1293--1298，[Theorem 2, pp. 1294--1295；Theorem 4, pp. 1296--1297](https://people.math.rochester.edu/faculty/doug/otherpapers/QuillenBP.pdf)，[DOI](https://doi.org/10.1090/S0002-9904-1969-12401-8)。用途：$MU_*$ 的 Lazard universal formal group law，以及 $MU_{(p)}$ 的 $p$-typical idempotent/Brown--Peterson summand。
- P. S. Landweber, *Homological properties of comodules over $MU_*(MU)$ and $BP_*(BP)$*, American Journal of Mathematics 98 (1976), 591--610，[JSTOR 永久页](https://www.jstor.org/stable/2373808)，[DOI](https://doi.org/10.2307/2373808)。用途：Theorem 2.6 的 extension-of-scalars exactness 与 Corollary 2.7 的 homology-theory 结论；不作为结构化乘法来源。
- E. H. Brown and F. P. Peterson, Brown--Peterson spectrum 原始构造。用途：历史来源；本书主线的 $p$-typical summand 与系数定位采用 Quillen Theorem 4 和 Ravenel Theorems 4.1.12(c)、4.1.18(a)。
- M. Hazewinkel, formal groups and applications。用途：Hazewinkel generators 和 $p$-typical formal group law convention。

## 3. Morava K-theory、Morava E-theory 和局部化

- M. Hovey and N. P. Strickland, *Morava K-theories and localisation*, Memoirs AMS 139 (1999)。用途：Morava K-theories、localization、$K(n)$-local category 的基础定理。
- E. Devinatz and M. Hopkins, *Homotopy fixed point spectra for closed subgroups of the Morava stabilizer groups*, Topology 43 (2004), 1--47，[Ravenel 文献存档 PDF](https://people.math.rochester.edu/faculty/doug/otherpapers/dh04.pdf)，[DOI](https://doi.org/10.1016/S0040-9383(03)00029-6)。用途：Theorem 1(iii)--(iv), pp. 3--4、Definition 1.5, p. 4、Theorem 2(i)--(ii), p. 5 和 Proposition 6.7, pp. 34--35 所给 closed-subgroup fixed points、$K(n)$-local sphere 与连续 descent spectral sequence。
- P. G. Goerss and M. J. Hopkins, *Moduli Spaces of Commutative Ring Spectra*, in *Structured Ring Spectra*, LMS Lecture Note Series 315 (2004), 151--200，[Ravenel 文献存档 PDF](https://people.math.rochester.edu/faculty/doug/otherpapers/commring.PDF)。用途：Section 7、Proposition 7.1、Corollaries 7.6--7.7 所给 Lubin--Tate/Morava $E$-theory 的 $\mathbb E_\infty$ realization moduli 与结构化 stabilizer action。
- J. Rognes, *Galois extensions of structured ring spectra*, arXiv:math/0502183。用途：structured ring spectra 的 Galois 理论、Lubin-Tate examples 和 Goerss-Hopkins-Miller 映射空间背景。
- M. J. Hopkins, N. J. Kuhn, D. C. Ravenel, *Generalized group characters and complex oriented cohomology theories*, Journal of the American Mathematical Society 13 (2000), 553--594，[作者存档 PDF](https://people.math.rochester.edu/faculty/doug/mypapers/hkr.pdf)，[DOI](https://doi.org/10.1090/S0894-0347-00-00332-5)。用途：Theorem C、Sections 6.3--6.4 和 Theorem D 的固定高度有限群 character theory；登记为非主线 P1 接口。
- M. Ando, M. J. Hopkins, N. P. Strickland, *Elliptic spectra, the Witten genus and the theorem of the cube*, Inventiones Mathematicae 146 (2001), 595--687，[DOI](https://doi.org/10.1007/s002220100175)。用途：Definition 1.2、Definition 2.40、Corollary 2.50 和 Theorem 2.53 的 elliptic spectrum、cubical structure 与 sigma orientation。
- J. H. Silverman, *The Arithmetic of Elliptic Curves*, 2nd ed., Graduate Texts in Mathematics 106, Springer, 2009，[图书 DOI](https://doi.org/10.1007/978-0-387-09494-6)。用途：Chapter IV, Theorem 7.4 与 Corollary 7.5, p. 134 的椭圆曲线形式群高度 $1/2$ 结论。
- P. G. Goerss, *Topological modular forms [after Hopkins, Miller, and Lurie]*, Astérisque 332 (2010), Exp. no. 1005, 221--255，[机构存档 PDF](https://www.numdam.org/item/AST_2010__332__221_0.pdf)，[永久页](https://www.numdam.org/item/AST_2010__332__221_0/)。用途：Theorem 1.2, pp. 224--225 与 Definition 1.3, p. 225 的 derived moduli sheaf、tmf global sections 和 descent spectral sequence。

## 4. Nilpotence、periodicity、thick subcategories 和 convergence

- E. Devinatz, M. Hopkins, J. Smith, *Nilpotence and stable homotopy theory I*, Annals of Mathematics 128 (1988), 207--241。用途：ring-spectrum nilpotence；本书版本定位为 Theorem 1(i)。
- M. Hopkins and J. Smith, *Nilpotence and stable homotopy theory II*, Annals of Mathematics 148 (1998), 1--49。用途：Morava K field/Künneth 性质（Propositions 1.4、1.5）、nilpotence detection（Theorem 3）、thick subcategories（Theorem 7）、periodicity（Theorem 9）、self-map 唯一/自然性（Corollaries 3.7、3.8）和 finite-spectrum class invariance（Theorem 14）。
- D. C. Ravenel, *Nilpotence and Periodicity in Stable Homotopy Theory*。用途：$L_n$ smash product theorem（Theorem 7.5.6）、finite-spectrum chromatic convergence（Theorem 7.5.7）及其证明（Section 8.6，尤其 Lemma 8.6.5 与随后的 $\lim/\lim^1$ 论证）。
- S. K. Chebolu, *Thick subcategories in stable homotopy theory*, arXiv:math/0607245。用途：thick subcategory theorem 的教学性核对，不作为最终核心来源替代 Annals 原文。

## 5. Telescope、redshift 和近期前沿

- R. Burklund, J. Hahn, I. Levy, T. M. Schlank, *K-theoretic counterexamples to Ravenel's telescope conjecture*, arXiv:2310.17459。用途：2023 后 telescope conjecture 状态；研究边界和正文警示。
- J. Hahn and D. Wilson, *Redshift and multiplication for truncated Brown-Peterson spectra*, arXiv:2012.00864。用途：$BP\langle n\rangle$ 的 $\mathbb E_3$-$BP$ algebra structure 和 redshift 外部输入。
- R. Burklund, T. M. Schlank, A. Yuan, *The Chromatic Nullstellensatz*, arXiv:2207.09929。用途：$T(n)$-local $\mathbb E_\infty$-rings、Lubin-Tate theories、nilpotence detection 和 redshift for algebraic K-theory。
- S. Ben-Moshe, S. Carmeli, T. M. Schlank, L. Yanovski, *Descent and cyclotomic redshift for chromatically localized algebraic K-theory*, arXiv:2309.07123。用途：cyclotomic redshift、descent 和 telescope 反例的关系。
- S. Carmeli, T. M. Schlank, L. Yanovski, *Ambidexterity in Chromatic Homotopy Theory*, arXiv:1811.02057。用途：$T(n)$-local higher semiadditivity 背景。
- S. Ben-Moshe, *Higher Semiadditivity in Transchromatic Homotopy Theory*, arXiv:2411.00968。用途：transchromatic character 与 semiadditivity 前沿。
- S. Ben-Moshe, *Chromatic Higher Semiadditivity by Height Induction*, arXiv:2501.08092。用途：2025 高度归纳证明路线的边界记录。
- T. Barthel, T. M. Schlank, N. Stapleton, J. Weinstein, *On the rationalization of the $K(n)$-local sphere*, arXiv:2402.00960。用途：rational $K(n)$-local sphere 和 chromatic splitting 前沿。
- G. Angelini-Knoll, *Syntomic cohomology of truncated Brown-Peterson spectra*, arXiv:2602.14380v3。用途：2026 年 $BP\langle n\rangle$ syntomic/K-theory 结果的前沿记录；进入正文定理前需进一步 locator。

## 6. 使用级别

| 级别 | 说明 |
| --- | --- |
| Core | 可作为基础正文定义或定理来源 |
| External theorem | 可引用为外部输入，但正文不重证 |
| Survey cross-check | 用于核对路线和术语，不作为最终定理定位 |
| Frontier | 近期研究边界，默认不进入证明链 |

## 7. Splitting、duality、Picard 和计算

- N. P. Strickland, *Gross--Hopkins duality*, Topology 39 (2000), 1021--1033，[Ravenel 文献存档 PDF](https://www.sas.rochester.edu/mth/sites/doug-ravenel/otherpapers/gross.pdf)，[arXiv](https://arxiv.org/abs/math/0011108)，[DOI](https://doi.org/10.1016/S0040-9383(99)00049-X)。用途：Proposition 1、Theorem 2 和 Theorem 20 的 $K(n)$-local Brown--Comenetz dualizing object、determinant twist 与 grading convention；登记为非主线 P1 接口。
- S. K. Devalapurkar, *The Lubin-Tate stack and Gross-Hopkins duality*, arXiv:1711.04806。用途：Lubin-Tate stack、Gross-Hopkins duality 和 Picard group 的 descent-geometric 解释。
- P. G. Goerss, H.-W. Henn, M. Mahowald, C. Rezk, *On Hopkins' Picard groups for the prime 3 and chromatic level 2*, Journal of Topology 8 (2015), 267--294，[arXiv](https://arxiv.org/abs/1210.7033)，[DOI](https://doi.org/10.1112/jtopol/jtu024)。用途：Theorems 1.1--1.2 的高度 2、素数 3 exotic 与完整 Picard group 计算。
- I. Mor, *Picard and Brauer groups of $K(n)$-local spectra via profinite Galois descent*, arXiv:2306.05393v2 (2023-10-12)，[v2 永久页](https://arxiv.org/abs/2306.05393v2)，[v2 PDF](https://arxiv.org/pdf/2306.05393v2)，[Ravenel 文献存档 PDF](https://www.sas.rochester.edu/mth/sites/doug-ravenel/otherpapers/Mor.pdf)。用途：Theorem A、Proposition 3.21、Corollary 3.24 与 Theorem 4.4 的 Picard spectrum descent、连续 descent spectral sequence 和 exotic filtration；登记为非主线 P1 接口。
- A. Beaudry, I. Bobkova, P. Goerss, H.-W. Henn, V.-C. Pham, V. Stojanoska, *Cohomology of the Morava stabilizer group through the duality resolution at $n=p=2$*, arXiv:2210.15994。用途：高度 2、素数 2 的 Morava stabilizer cohomology 计算接口。

## 8. Equivariant、motivic 和 synthetic 前沿

- M. Behrens and J. Carlisle, *Periodic phenomena in equivariant stable homotopy theory*, arXiv:2406.19352。用途：equivariant chromatic theory 的近期综述和框架，当前作为前沿输入。
- K. Allen and L. Piessevaux, *Synthetic equivariant spectra for finite abelian groups and motivic homotopy theory*, arXiv:2510.20197。用途：equivariant synthetic spectra 与 complex motivic reconstruction 的前沿接口。
- A. Mazel-Gee, *$\mathbb E_\infty$ automorphisms of motivic Morava $E$-theories*, arXiv:1901.05713。用途：motivic Morava E-theory 的结构和 automorphism 前沿。

## 9. 当前引用边界

1. 本轮登记的主线 P0 外部输入均有 theorem/section/page 与稳定链接；编号映射见 `P0_REFERENCE_LOCATORS_BATCH_1.md`、`P0_REFERENCE_LOCATORS_BATCH_2.md` 和 `THEOREM_LEDGER.md`。
2. HKR、Gross--Hopkins duality、Picard/profinite descent 与计算扩展具有精确 locator，但保持非主线 P1 身份，不能扩大为书中未陈述的研究结论。
3. 2023--2026 预印本继续按版本、发表状态和假设翻译分类，默认属于 Frontier，不进入 P0 证明链。
4. equivariant/motivic 的同等深度展开属于独立扩展范围；现有条目只承担边界说明。
