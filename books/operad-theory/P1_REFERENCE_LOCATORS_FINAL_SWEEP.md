# P1 引用定位最终批：Deligne, Dunn additivity, and remaining geometry boundaries

本文件记录剩余 P1 外部输入的定位。它的目标是把“主题名引用”替换为明确来源和 theorem locator，同时把不能由 operad theory 自身闭合的几何/分析断言保留为外部边界。

## 1. Dunn/Lurie additivity

**来源.** Jacob Lurie, *Higher Algebra*, September 18, 2017.

**DUNN-1.** Theorem 5.1.2.2 (Dunn Additivity Theorem) states that the bifunctor
$$
E_k^\otimes\times E_{k'}^\otimes\to E_{k+k'}^\otimes
$$
of Construction 5.1.2.1 exhibits $E_{k+k'}^\otimes$ as the tensor product of the infinity-operads $E_k^\otimes$ and $E_{k'}^\otimes$.

**本书使用边界.** DUNN-1 supports 外部输入定理 L.19 and 外部输入定理 20.12. It is an infinity-operadic additivity theorem; for strict topological operad tensor products one must separately cite Dunn/Fiedorowicz--Vogt style cofibrancy statements.

## 2. Deligne conjecture and Hochschild cochains

**来源 A.** James E. McClure and Jeffrey H. Smith, "A solution of Deligne's Hochschild cohomology conjecture," arXiv:math/9910126v2, 2001-02-02.

**MS-1.** Theorem 1.1 states that the singular chain operad of the little 2-cubes operad is quasi-isomorphic, as a chain operad, to the brace/cup operad $H$ acting on desuspended normalized Hochschild cochains.

**MS-2.** Theorem 3.3 gives the operad action on totalizations of cosimplicial objects equipped with the required cup products and $\circ_k$ operations; Hochschild cochains are an application of this framework.

**MS-3.** Corollary 7.3 identifies the singular chains of the topological operad constructed by McClure--Smith with $H$ up to quasi-isomorphism, and Theorem 8.1 proves that the topological operad is weakly equivalent to the little 2-cubes operad.

**来源 B.** Clemens Berger and Benoit Fresse, "Combinatorial operad actions on cochains," arXiv:math/0109158v2, 2002-10-21.

**BF-1.** Proposition 1.2.7 proves that the surjection modules form a dg-operad.

**BF-2.** Theorem 1.3.2 proves that the table-reduction map from the Barratt--Eccles operad to the surjection operad is a surjective morphism of dg-operads.

**BF-3.** Theorem 1.6.7 states that the normalized Hochschild cochain complex of an associative algebra is an algebra over the operad $F_2E$, with the cup product and Getzler--Kadeishvili braces represented by specified elements.

**BF-4.** Theorem 3.1.3 gives a model structure on $P$-algebras under the paper's Assumption 3.1.2; this supports the model-category side of Barratt--Eccles type operadic actions, not arbitrary operad algebra admissibility.

**本书使用边界.** MS-1--MS-3 and BF-1--BF-3 close the Deligne-conjecture locator for Chapter 11/12. They do not fix this book's suspended brace signs; those still require the sign convention crosswalk in Appendix W.

## 3. Category of operators and strict-to-Lurie entry

This final sweep uses HA-OP-1--HA-OP-3 from [P0_REFERENCE_LOCATORS_BATCH_10.md](P0_REFERENCE_LOCATORS_BATCH_10.md): Lurie *Higher Algebra* Example 2.1.1.21, Definition 2.1.1.23 and Proposition 2.1.1.27. These locators replace the previous generic "Lurie/Hinich category of operators" placeholder for ordinary/simplicial colored operads.

## 4. Framed little disks, BV, formality, recognition

The following P1 topics remain external in mathematical substance but no longer have undefined status:

1. **May recognition principle.** Source: J. P. May, *The Geometry of Iterated Loop Spaces*, especially the recognition theorem for algebras over little-cubes/operad monads and iterated loop spaces. This book treats it as P1 background for Chapter 10/15, not as a proof-chain input for algebraic operad theory.
2. **$H_\*(E_n)\cong\operatorname{Pois}_n$.** Sources: F. R. Cohen's computation of the homology of little cubes and the standard Fresse/Loday--Vallette presentations. This book keeps the theorem as P1 because the proof is topological, not internal to algebraic operad definitions.
3. **$E_n$ formality.** Sources: Kontsevich, Tamarkin and Fresse. This book records the topic as P1 and uses no formality theorem in the core proof chain unless a specific chain model and characteristic-zero convention are chosen.
4. **Framed $E_2$ and BV.** Source: Getzler, "Batalin-Vilkovisky algebras and two-dimensional topological field theories." This book uses it only to identify the homology operad of framed little disks with BV after the chosen framed model is fixed.

These items are not closed as internal operad-theory proofs. They are closed as locator boundaries: any theorem-level use must cite the named source and check the model, coefficient ring and grading convention.

## 5. Stratified factorization and Fukaya geometry

The following are deliberately not absorbed into the operad-theory proof chain:

1. **Stratified factorization homology.** Sources: Ayala--Francis--Tanaka and Lurie. It remains a P1 geometry locator for Appendix V/Z; topological-manifold excision, circle, boundary, and commutative-coefficient calculations are covered by AF-0--AF-5.
2. **Wrapped Fukaya descent/gluing.** Sources: Ganatra--Pardon--Shende, Seidel, Nadler and related Liouville-sector literature. It remains a P1/geometric external locator.
3. **Fukaya category construction.** Sources: Seidel and Fukaya--Oh--Ohta--Ono. It remains a P0 geometric source boundary, not an operad-theory theorem.

**Boundary rule.** These geometry inputs cannot be "closed" by adding operad-theory definitions. They become usable only after choosing a geometric model and recording transversality, compactness, orientation and gluing hypotheses.
