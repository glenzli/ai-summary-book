# P0 引用定位批次 9：Modern admissibility, rectification, and strict-to-infinity algebra comparison

本文件记录第九批 P0 外部输入定位：Pavlov--Scholbach 的 colored symmetric operad admissibility/rectification 定理，以及 Lurie *Higher Algebra* 中 associative/commutative algebra 的模型范畴到 infinity-category 比较。它补齐第十四章、附录 G/R/X 和第十九章中“现代 admissibility/rectification”和“严格代数模型呈现 infinity-categorical algebra objects”的主要 theorem locator。

本批次不替代第八批 White/White--Yau 的 Bousfield localization preservation，也不证明任意底范畴中的 rectification。它只给出可引用的 theorem numbers 和使用边界。

## 1. Pavlov--Scholbach: colored operads

**来源.** Dmitri Pavlov and Jakob Scholbach, "Admissibility and rectification of colored symmetric operads," arXiv:1410.5675v4, 2022-03-27.

**PSAR-1.** Definition 2.1 gives the symmetricity hypotheses used later: admissibly generated, strongly admissibly generated, h-monoidal, symmetric h-monoidal, symmetroidal and symmetric flat. 本书附录 G 中的 T1--T3 假设包应回指此定义，而不能把这些词当作普通平坦性直觉。

**PSAR-2.** Theorem 5.11 is the principal admissibility theorem: under the stated combinatorial/admissibly-generated/tractable hypotheses, if $\mathcal C$ is symmetric h-monoidal, then every $W$-colored symmetric operad $\mathcal O$ in $\mathcal C$ is admissible. 本书第十四章外部输入定理 14.21、附录 G.12 和附录 R 的 colored/all-small admissibility 可引用此定理。

**PSAR-3.** Theorems 6.3 and 6.7 give strong admissibility/forgetful-functor control under projective or injective cofibrancy and symmetroidality-type hypotheses. 本书只用它们支撑“忘却 functor preserves cofibrant objects/cofibrations under stronger hypotheses”的检查表，不把它们改写成所有代数对象自动 cofibrant。

**PSAR-4.** Theorem 7.5 is the rectification theorem: for a map $f:\mathcal O\to\mathcal P$ of admissible colored operads in a tractable symmetric h-monoidal symmetric monoidal model category, the induced Quillen adjunction on algebras is a Quillen equivalence exactly under the stated weak-equivalence-on-free-cofibrant-algebras condition; symmetric flatness is a sufficient condition. 本书第十四章外部输入定理 14.26 和附录 G.13 可引用此定理。

**PSAR-5.** Theorem 7.11 compares strict algebras over a simplicial colored operad with Lurie-style quasicategorical algebras under $\mathcal C$-admissibility and symmetric-flatness of the projective cofibrant replacement $Q\mathcal O\to\mathcal O$. 本书第十九章外部输入定理 19.25 和附录 M.16 可把它作为 simplicial model category 情形的 strict-to-infinity algebra comparison locator。

**PSAR-6.** Theorem 8.10 transports colored operads and their algebra categories along weak symmetric monoidal Quillen adjunctions, with Quillen equivalence conclusions under the theorem's hypotheses. 本书不得把 Quillen equivalence of base categories 直接升级为 Quillen equivalence of algebra categories；必须引用 PSAR-6 或相应专门定理。

## 2. Pavlov--Scholbach: symmetric powers

**来源.** Dmitri Pavlov and Jakob Scholbach, "Homotopy theory of symmetric powers," arXiv:1510.04969v3, 2020-06-01.

**PSP-1.** Theorems 5.6 and 5.7 prove stability of symmetric h-monoidality/symmetroidality under transfer and monoidal Bousfield localization, in the notation of that paper.

**PSP-2.** Theorems 6.4 and 6.5 prove the analogous stability results for symmetric flatness. In [PSAR] these exact results are cited as the mechanism promoting admissibility and rectification hypotheses to transferred or localized model structures.

**本书使用边界.** PSP-1--PSP-2 justify the technical hypotheses in PSAR-2--PSAR-4; they do not by themselves create transferred algebra model structures. 在正文中应先说明底范畴满足这些 symmetricity properties，再引用 PSAR-2 或 PSAR-4。

## 3. Lurie: associative and commutative algebra comparison

**来源.** Jacob Lurie, *Higher Algebra*, September 18, 2017.

**HA-ALG-1.** Theorem 4.1.8.4 gives the associative algebra comparison for a combinatorial monoidal model category satisfying either all-objects-cofibrant or left proper/cofibrantly-generated/monoid-axiom assumptions. It identifies the underlying infinity-category of strict associative algebras with associative algebra objects in the localized monoidal infinity-category.

**HA-ALG-2.** Theorem 4.5.4.7 gives the commutative algebra comparison for a combinatorial symmetric monoidal model category that is freely powered and whose forgetful functor from commutative algebras preserves fibrant-cofibrant objects. It identifies the localization of strict commutative algebras with commutative algebra objects in the localized infinity-category.

**HA-ALG-3.** Lurie, *Higher Algebra*, Theorems 4.1.8.4 and 4.5.4.7 are special algebra comparison theorems, not universal colored-operad rectification theorems. For arbitrary colored operads in model categories, this book should use PSAR-2--PSAR-6 or Hinich/Fresse/White--Yau as appropriate.

**HA-MON-1.** Lurie, *Higher Algebra*, Proposition 4.1.7.4 and Example 4.1.7.6. For a symmetric monoidal model category in the convention of Section 4.1.7, the weak equivalences between cofibrant objects are stable under tensoring by cofibrant objects, and the localization of the cofibrant subcategory inherits the underlying symmetric monoidal infinity-category. 本书外部输入定理 19.21 使用此版本；它不需要把“足够良好”作为未展开假设。

**HA-MON-2.** Lurie, *Higher Algebra*, Corollary 4.1.7.16. For a simplicial symmetric monoidal model category with compatible simplicial and monoidal structures, the operadic nerve of the fibrant-cofibrant subcategory presents the underlying symmetric monoidal infinity-category. 本书只在需要显式 operadic-nerve 模型时使用这一加强。

## 4. 本书使用边界

**可用于以下位置.**

1. 第十四章：operad algebra admissibility 和 rectification criterion。
2. 附录 G：T1--T3 假设包的文献锚点。
3. 附录 R/X：正例和失败例子的边界检查。
4. 第十九章和附录 M：strict algebra model 与 infinity-categorical algebra objects 的比较。
5. 第十九章：模型范畴的 underlying symmetric monoidal infinity-category（HA-MON-1--HA-MON-2）。

**禁止用法.**

1. 不得把 PSAR-2 的 admissibility 结论用于不满足 symmetric h-monoidality、tractability、smallness 或 colored-set hypotheses 的底范畴。
2. 不得把 PSAR-4 的 rectification 结论用于任意 operad weak equivalence；必须检查 symmetric flatness 或 theorem 7.5 的 free-cofibrant-algebra condition。
3. 不得把 HA-ALG-1 或 HA-ALG-2 推广成所有 colored operads 的 algebra localization comparison。
4. 不得由本批次推出 Fukaya category、factorization algebra 或 geometric descent 的分析定理。
