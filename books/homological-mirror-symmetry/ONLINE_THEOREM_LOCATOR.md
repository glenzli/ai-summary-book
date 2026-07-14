# Online theorem locator

审计日期：2026-07-15
定位等级：在线教材最低闭合版。它要求读者能追溯到具体文献、主结果和本书使用位置；不要求出版社级页码、定理编号和版本差异校勘。

## 定位原则

- “Locator” 给出文献、arXiv/出版信息、主结果名称或章节主题。
- “本书使用” 给出引用该输入的章节或附录。
- 若结果依赖大型理论，本书不内部证明，而把它列为外部输入。
- 2024--2025 预印本默认标为研究边界，除非本书只使用其可明确陈述的主结果。

## 增强范畴与 Morita 理论

| 本书编号 | 外部输入 | Online locator | 本书使用 |
| --- | --- | --- | --- |
| 1.16, B.11 | $A_\infty$ Yoneda fully faithful | Lefevre-Hasegawa, *Sur les A-infini catégories*, arXiv:math/0310337；定位到 modules、Yoneda embedding、derived category of $A_\infty$ categories 的主体章节 | 1.4, B.5 |
| 1.21 | twisted complexes 与 pretriangulated envelope | Lefevre-Hasegawa arXiv:math/0310337；Keller, *On differential graded categories*, arXiv:math/0601185，定位到 pretriangulated dg categories 与 triangulated hull 讨论 | 1.5 |
| C.2 | dg quotient | Drinfeld, *DG quotients of DG categories*, arXiv:math/0210114；主构造为 dg category modulo full dg subcategory，并与 Verdier quotient 相容 | C.1 |
| 8.13, 18.2 | Hochschild Morita invariance | Keller, arXiv:math/0601185，定位到 dg Morita theory 与 Hochschild invariants 的 survey 部分 | 8.4, 18.1 |

## B-side 与 Fourier-Mukai / singularity

| 本书编号 | 外部输入 | Online locator | 本书使用 |
| --- | --- | --- | --- |
| 2.7 | regular scheme 上 $D^b_{\mathrm{Coh}}=D_{\mathrm{perf}}$ | Stacks Project, Tag 0FDC, Lemma 36.11.8 | 2.2 |
| 2.11 | h-injective dg enhancement | Spaltenstein, *Resolutions of unbounded complexes*, Compos. Math. 65 (1988), 121--154；Keller arXiv:math/0601185 | 2.3 |
| 2.12A | kernel transform 的 dg/Morita enhancement | Toen, *The homotopy theory of dg-categories and derived Morita theory*, arXiv:math/0408337，derived Morita 与 schemes/kernels 部分；Huybrechts | 2.4, D.1 |
| 2.14, D.3 | Orlov representability、adjoint kernels | Huybrechts, *Fourier--Mukai Transforms in Algebraic Geometry*, Theorem 5.14 (Orlov: fully faithful plus left/right adjoints)；adjoint-kernel chapter | 2.4, D.2 |
| 2.18 | affine regular hypersurface MF/singularity comparison | Orlov, *Triangulated categories of singularities and D-branes in Landau-Ginzburg models*, arXiv:math/0302304 | 2.5 |
| 19.4 | affine matrix-factorization/singularity comparison | Orlov, *Triangulated categories of singularities and D-branes in Landau-Ginzburg models*, arXiv:math/0302304；正文只取 finite-Krull-dimensional regular noetherian affine ring 与 non-zero-divisor 版本 | 19.2 |
| 18.4 | HKR theorem | 标准 Hochschild-Kostant-Rosenberg 定理；本书只使用光滑 proper 特征零情形与 Todd 修正提醒 | 18.2 |

## A-side Fukaya 与 Floer 输入

| 本书编号 | 外部输入 | Online locator | 本书使用 |
| --- | --- | --- | --- |
| 3.14, 4.9, E.6 | compact exact Floer/Fukaya analytic package | Seidel, *Fukaya Categories and Picard-Lefschetz Theory*, Chapters 8--12；Chapter 11 indices/determinant lines，Chapter 12 complete Fukaya category；DOI 10.4171/063 | 3.4, 4.2--4.3, E |
| 3.18, 4.12 | continuation 与 coherent-choice invariance | Seidel Chapters 8--12 的 exact compact model | 3.5, 4.4 |
| 4.14A | compact exact Fukaya category 的 cohomological units | Seidel, *Fukaya Categories and Picard-Lefschetz Theory*, Chapters 9--12 的 unit/complete-category package | 4.5 |
| 4.15 | homological/cohomological units strictification | Lefevre-Hasegawa, *Sur les $A$-infini categories*, arXiv:math/0310337，homological 与 strict unitality 的比较 | 4.5 |
| 5.13 | one-object filtered curved $A_\infty$ algebra、bounding cochains | FOOO, *Lagrangian Intersection Floer Theory: Anomaly and Obstruction*, filtered $A_\infty$ structures, Maurer-Cartan/bounding cochains, Kuranishi perturbations；多对象 category 需另加 coherent polygon data | 5.4 |

## Wrapped、stopped、sectorial 与 microlocal

| 本书编号 | 外部输入 | Online locator | 本书使用 |
| --- | --- | --- | --- |
| 6.8, 6.10 | telescope wrapped complexes；no-escape/continuation；wrapped $A_\infty$ category | Ganatra-Pardon-Shende, *Covariantly functorial wrapped Floer theory on Liouville sectors*, arXiv:1706.03152；Floer data、compactness、orientations、continuation 与 category construction | 6.2--6.3 |
| 6.13 | Liouville sector functoriality | GPS arXiv:1706.03152；主结果为 sector inclusions 诱导 wrapped Fukaya functors | 6.4 |
| 14.6 | wrapped open-closed/closed-open 与 duality | Ganatra, *Symplectic cohomology and duality for the wrapped Fukaya category*, arXiv:1304.7312；GPS arXiv:1706.03152 用于 sector 版本 | 14.2 |
| 14.7 | exact wrapped generation criterion | Abouzaid, *A geometric criterion for generating the Fukaya category*, arXiv:1001.4593 / Publ. Math. IHES 112 (2010), Theorem 1.1 and equation (1.2) | 14.3, K.2 |
| 18.8 | non-degenerate wrapped OC/CO isomorphisms | Ganatra, *Symplectic cohomology and duality for the wrapped Fukaya category*, arXiv:1304.7312；只用于来源规定的 non-degenerate Liouville/duality package | 18.3 |
| 6.17, 7.5, 7.7, 15.3, 15.6 | split-generation、stop removal、sectorial descent、Kunneth | GPS, *Sectorial descent for wrapped Fukaya categories*, arXiv:1809.03427；Weinstein cocore/mostly-Legendrian linking-disk split-generation、stop removal equals localization、descent 与 Kunneth results | 6.5, 7.2--7.3, 15 |
| 16.6 | wrapped/microlocal correspondence | GPS, *Microlocal Morse theory of wrapped Fukaya categories*, arXiv:1809.08807；主结果为 stopped cotangent wrapped category 与 microsupport-bounded sheaf category compact objects 的等价 | 16.2 |
| 16.5 | cotangent constructible/Fukaya correspondence | Nadler--Zaslow, arXiv:math/0604379，给出 quasi-embedding；Nadler, *Microlocal branes are constructible sheaves*, arXiv:math/0612399，证明本质满并升级为 quasi-equivalence | 16.2 |
| 17.5, 17.7 | Orlov/Viterbo functors | Sylvan, *Orlov and Viterbo functors in partially wrapped Fukaya categories*, arXiv:1908.02317；主结果为 Orlov functor spherical criterion 和 Viterbo transfer homological epimorphism/localization | 17.2--17.3 |

## 标准 HMS 例子

| 本书编号 | 外部输入 | Online locator | 本书使用 |
| --- | --- | --- | --- |
| 0.2, 8.1 | HMS 原始断言 | Kontsevich, *Homological Algebra of Mirror Symmetry*, arXiv:alg-geom/9411018；ICM 1994，同调代数化 mirror conjecture | 0, 8 |
| 9.12 | 椭圆曲线 HMS | Polishchuk-Zaslow, *Categorical Mirror Symmetry: The Elliptic Curve*, arXiv:math/9801119；对象字典、theta 函数乘法与 Fukaya 侧三角形计数 | 9, J, L |
| 10.12 | toric HMS | Abouzaid, *Morse Homology, Tropical Geometry, and Homological Mirror Symmetry for Toric Varieties*, arXiv:math/0610004；Morse/tropical 模型与 toric HMS | 10, J, L |
| 11.6, 11.8, 11.9A | Fukaya-Seidel、Picard-Lefschetz 与 Dehn-twist triangle | Seidel, *Fukaya Categories and Picard-Lefschetz Theory*；exact Lefschetz fibration、directed Fukaya category、Dehn twists 与 exact triangle | 11, J, L |
| 12.5 | quartic K3 HMS | Seidel, *Homological mirror symmetry for the quartic surface*, arXiv:math/0310414；quartic surface HMS | 12 |
| 12.6 | projective Calabi-Yau hypersurfaces | Sheridan, *Homological Mirror Symmetry for Calabi-Yau hypersurfaces in projective space*, arXiv:1111.0632；摘要主结果为 $d>2$ smooth Calabi-Yau hypersurface HMS | 12 |
| 13.7 | hypersurfaces in $(\mathbb C^\ast)^n$ | Abouzaid--Auroux, arXiv:2111.06543；maximally degenerating families 的 coherent category quasi-embeds 到 mirror fiberwise wrapped Fukaya category；来源未在此处断言本质满 | 13 |
| 13.9 | higher-dimensional pairs of pants | Lekili--Polishchuk, arXiv:1811.04264；partially wrapped category 给出 categorical resolution，移除 stops 后 fully wrapped category 对应奇异仿射簇的 derived category | 13 |
| 19.7 | Rabinowitz Fukaya / Brieskorn--Pham | Lekili--Ueda, arXiv:2406.15915；非 Calabi--Yau 型 Brieskorn--Pham 情形的 Rabinowitz Fukaya/equivariant-MF HMS | 19 |
| 12.7, 20.13(1) | Batyrev mirror pairs | Ganatra--Hanlon--Hicks--Pomerleano--Sheridan, arXiv:2406.05272；来源规定的大类 Batyrev pairs，特征零及除有限素数外的正特征 | 12, 20 |

## 稳定性、twists 与函子化边界

| 本书编号 | 外部输入 | Online locator | 本书使用 |
| --- | --- | --- | --- |
| 20.4--20.6 | Bridgeland stability conditions | Bridgeland, *Stability conditions on triangulated categories*, arXiv:math/0212237；正文采用有限秩数值格、HN filtration、local finiteness 与 support property 口径 | 20.2 |
| 20.8 | spherical twist autoequivalence | Seidel--Thomas, *Braid group actions on derived categories of coherent sheaves*, arXiv:math/0001043 | 20.3 |
| 20.13--20.14 | functorial HMS/BPS 研究视角 | Pasquarella, arXiv:2502.06951；其摘要强调统一形式主义仍需发展，故正文只据此标记开放边界 | 20.1, 20.6 |

## 在线收口判定

本文件满足在线教材引用闭合：每个外部输入定理均可追溯到具体一手资料和本书使用位置。
它不满足出版级 theorem locator：缺页码、出版版本差异、精确定理编号和逐条假设校勘。
