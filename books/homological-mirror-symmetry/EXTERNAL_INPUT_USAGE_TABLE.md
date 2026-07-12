# 外部输入使用表

本表回答三个问题：本书在哪里使用外部输入、输入假设是什么、该输入输出什么结论。

| 输入编号 | 使用章节 | 假设 | 输出 | 不内部证明原因 |
| --- | --- | --- | --- | --- |
| 1.16 / B.11 Yoneda | 1, B | small strictly unital $A_\infty$ category | Yoneda embedding 在 morphism complexes 上 quasi-isomorphic | 需要完整 $A_\infty$ module theory |
| 1.21 finite twisted-complex envelope | 1 | field $k$；small noncurved strictly unital $k$-linear $A_\infty$ category | $\operatorname{Tw}(\mathcal A)$ pretriangulated；$H^0$ triangulated；原 category cohomologically fully faithful | 属于 $A_\infty$ homological algebra；不含 arbitrary coproduct/idempotent completion |
| C.2 dg quotient | C, 7 | dg category 与 full dg subcategory | quotient models Verdier quotient/enhanced localization | Drinfeld quotient 构造技术性强 |
| 2.7 regular comparison | 2 | regular noetherian scheme | $D^b_{\mathrm{Coh}}=D_{\mathrm{perf}}$ | regular-local-ring resolution theorem；Stacks Tag 0FDC |
| 2.11 B-side enhancement | 2 | qcqs scheme；noetherian when using coherent subcategory | h-injective dg model enhances derived category；regular case gives Perf/DbCoh quasi-equivalence | 需要 unbounded h-injective resolution theory |
| 2.12A kernel enhancement | 2, D | smooth proper finite-type schemes；perfect kernel | Fourier--Mukai formula lifts to dg quasi-functor and convolution composes | 需要 derived Morita/h-flat/h-injective formalism |
| 2.14 Fourier-Mukai representability | 2 | characteristic-zero algebraically closed $k$；smooth projective varieties；$k$-linear exact fully faithful functor with left and right adjoints | functor represented by an isomorphism-unique perfect kernel | Huybrechts Theorem 5.14 (Orlov)；不覆盖缺 adjoints 或一般 exact functor |
| 2.18 Orlov affine hypersurface | 2 | finite Krull-dimensional regular noetherian $R$；$w$ non-zero-divisor | finite-rank MF 与 $D_{\mathrm{sg}}(R/(w))$ 在 idempotent completion 后等价 | 需要 Orlov/Buchweitz hypersurface theory |
| 19.4 Orlov singularity variants | 19 | 章内另定 equivariant/nonaffine hypotheses | 相应 MF/singularity comparison | 不能由 affine 2.18 自动推出 |
| 3.14 / 4.9 / E.6 compact exact package | 3, 4, E | Liouville completion；compact exact graded relative-Pin branes；3.14 取 transverse intersection/$H=0$ data，4.9 取 compact pair data 与 coherent regular universal perturbations | finite intersection/chord operations，$\mu^1{}^2=0$，完整 $A_\infty$ identities | 需要 Fredholm、no-escape、compactness、determinant orientations 与 gluing；Seidel Chapters 8--12 |
| 3.18 continuation | 3 | exact Floer data homotopy | Floer complexes quasi-isomorphic | 需要 continuation moduli spaces |
| 4.12 Fukaya invariance | 4 | 同一 compact exact geometry/brane background；两套 coherent perturbation systems | compact exact Fukaya categories quasi-equivalent | 需要 continuation higher homotopies；不比较改变 brane background 后的 categories |
| 4.14A cohomological units | 4 | compact exact package；与 strip-like ends 相容的 unit perturbation data | continuation classes are two-sided units in $H^*\mathcal F^c_{\mathrm{ex}}$ | unit moduli spaces 不包含在无单位 $A_\infty$ identities 中；Seidel Chapters 9--12 |
| 4.15 strictification | 4 | field $k$；small noncurved cohomologically unital $k$-linear $A_\infty$ category | quasi-equivalent small strictly unital model | 依赖 homological/strict unitality comparison；不声称在同一底层 graded category 上严格化 |
| 5.13 FOOO filtered theory | 5 | closed symplectic $M$；compact oriented relatively spin $L$；characteristic-zero completed Novikov coefficients；gapped filtration；coherent virtual perturbations | one-object unital gapped filtered curved $A_\infty$ algebra、MC deformation 与 Floer cohomology | Kuranishi/virtual 技术超出本书；多对象 category 还需 coherent polygon package |
| 6.8 / 6.10 wrapped analytic/categorical package | 6 | Liouville completion 或带 sectorial convexity 的 sector；exact conical graded relative-Pin branes；nondegenerate cofinal radial Hamiltonians；contact-type $J$；coherent continuation/polygon data | telescope $CW^*$、choice invariance、orientation/local-system operations 与精确 (B.3)；cohomological units | 需要 wrapped action bounds、no-escape、compactness、orientation、continuation/polygon gluing；GPS arXiv:1706.03152 |
| 6.13 sector functoriality | 6 | Liouville sector inclusions | covariant wrapped functors | GPS sector machinery |
| 6.17 cocore/linking-disk generation | 6, F | Weinstein sector with handle presentation；optional mostly Legendrian stop | cocores plus required linking disks split-generate $H^0\operatorname{Perf}\mathcal W$；inclusion Morita | GPS sectorial descent/generation arXiv:1809.03427；不输出 raw quasi-equivalence |
| 7.7 stop removal | 7, K | stops $\mathfrak f\subset\mathfrak g$ | removing stop equals localization by linking disks | 依赖 GPS stop removal theorem |
| 14.6 wrapped OC/CO package | 14, 18 | Liouville manifold；wrapped category/SH 的 coefficients、gradings、cofinal Hamiltonians、orientations 与 compactness data | degree-$n$ OC、CO、module/product/Cardy compatibility | 需要 punctured marked-disk moduli 与 wrapped duality |
| 14.7 Abouzaid criterion | 14, K | full $\mathcal B\subset\mathcal W(M)$；composite $HH_*(\mathcal B)\to HH_*(\mathcal W)\to SH^{*+n}$ 命中 $1_{SH}$ | $\mathcal B$ split-generates $\mathcal W(M)$，即 inclusion Morita | Abouzaid Theorem 1.1 / equation (1.2)，依赖 two-output disks 与 Cardy relation |
| 15.3 sectorial descent | 15, K | Weinstein sectorial cover | hocolim of local wrapped categories gives global | 依赖 GPS descent theorem |
| 15.6 Kunneth | 15 | product Liouville sectors | $\mathcal W(X\times Y)$ Morita tensor product | 需要 wrapped product analysis |
| 16.5 Nadler-Zaslow | 16 | compact real analytic $Q$ | constructible sheaves quasi-embed/equivalent to cotangent Fukaya model | 需要 microlocal sheaf/Fukaya correspondence |
| 16.6 GPS microlocal | 16 | stopped cotangent/Weinstein sector with polarization | partially wrapped category equals compact sheaf category with microsupport | 依赖 GPS microlocal Morse theorem |
| 17.5 Orlov functor | 17 | partially wrapped stop situation | spherical criterion for Orlov functor | 依赖 Sylvan partially wrapped functor theory |
| 17.7 Viterbo transfer | 17 | Weinstein domain/subdomain | homological epimorphism/localization after modules | 依赖 Sylvan/GPS functoriality |
| 18.2 HH Morita invariance | 18 | dg/$A_\infty$ Morita equivalence | Hochschild invariants preserved | 标准 derived Morita theorem |
| 18.4 HKR | 18 | smooth proper characteristic zero | $HH$ identified with Hodge cohomology up to standard corrections | 需要 algebraic HKR and Todd correction |
| 9.12 elliptic HMS | 9, J, L | mirror elliptic curve / symplectic torus model | HMS with theta multiplication | 完整证明是标准例子论文 |
| 10.12 toric HMS | 10, L | smooth projective toric variety | toric HMS via Morse/tropical model | 证明依赖 Abouzaid construction |
| 11.6 Fukaya-Seidel | 11, L | exact Lefschetz fibration | directed category and thimble generation | Seidel theory |
| 12.6 Sheridan hypersurfaces | 12 | smooth degree $d+2$? Calabi-Yau hypersurface in projective space, $d>2$ in source convention | HMS proof-type result | 技术跨度大，只作外部输入 |
| 13.7 Abouzaid-Auroux | 13 | maximally degenerating hypersurfaces in algebraic tori | quasi-embedding into fiberwise wrapped Fukaya category | 需要 fiberwise wrapped category |
| 19.7 Lekili-Ueda | 19 | Brieskorn-Pham Milnor fibers with stated non-CY hypotheses | Rabinowitz HMS with equivariant MF | 近期研究专题 |

## 在线收口判定

本表使每个外部输入的逻辑角色可追踪：本书不证明大型分析和深层外部定理，但不再把它们当作未定位的黑箱。
