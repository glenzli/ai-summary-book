# P0 引用定位批次 10：Dwyer--Kan localization, dendroidal-Lurie comparison, and operadic straightening

本文件记录第十批 P0/P1 外部输入定位。它覆盖第十八、十九章和附录 M/T 中仍缺的 localization、coherent nerve、dendroidal-Lurie comparison、category-of-operators nerve 和 operadic straightening locator。

本批次将经典 Dwyer--Kan/Hinich localization、Heuts--Hinich--Moerdijk 的模型比较、Lurie *Higher Algebra* 的 operadic nerve，以及 Pratali 2025 的 operadic straightening 分层列出。Pratali 2025 属于较新来源；在本书中可作为 P1/前沿增强 locator，若要进入核心证明链，应先检查出版状态。

## 1. Hinich: Dwyer--Kan localization revisited

**来源.** Vladimir Hinich, "Dwyer-Kan localization revisited," arXiv:1311.4128v4, 2015-09-18.

**DKR-1.** Section 1.1.2 and the Proposition in Section 1.1.3 define infinity-categorical localization $L(\mathcal C,W)$ by the universal property and identify it with fibrant replacement in marked simplicial sets. 本书定义 19.3 的 relative-category localization 可回指此处作 infinity-categorical formulation。

**DKR-2.** Proposition 1.2.1 states that for a fibrant simplicial category $\mathcal C$ and fibrant simplicial subcategory $W$ with the same objects, the hammock localization map induces a weak equivalence of marked simplicial sets
$$(N(\mathcal C),N(W))\to RN(L^H(\mathcal C,W))^\natural.$$
This is the comparison between hammock localization and infinity-categorical localization used in Chapter 19.

**DKR-3.** Definition 1.3.1 defines the underlying infinity-category $N(\mathcal C)$ of a model category as $RN(L^H(\mathcal C,W))$, where $W$ is the subcategory of weak equivalences.

**DKR-4.** Theorem 1.3.3, cited there as [DK3, 4.4], identifies mapping spaces in hammock localization with homotopy function complexes computed by cosimplicial and simplicial resolutions.

**DKR-5.** Proposition 1.3.4, cited there as [DK3, 5.2], says that passing to cofibrant or fibrant subcategories induces equivalences of hammock localizations.

**DKR-6.** Proposition 1.3.5, cited there as [DK3, 4.8], states that for a simplicial model category the simplicial category of fibrant-cofibrant objects, the hammock localizations of the fibrant-cofibrant/full simplicial categories, and the hammock localization of the underlying category give equivalent presentations.

**DKR-7.** Proposition 1.5.1 states that a Quillen pair of model categories induces an adjoint pair between the underlying infinity-categories. When the Quillen pair is a Quillen equivalence, the induced adjunction is the standard source for the equivalence used in Chapter 19, with ordinary derived-unit/counit hypotheses still to be checked in the chosen model-category setting.

## 2. Heuts--Hinich--Moerdijk: dendroidal and Lurie models

**来源.** Gijs Heuts, Vladimir Hinich, and Ieke Moerdijk, "On the equivalence between Lurie's model and the dendroidal model for infinity-operads," arXiv:1305.3658.

**HHM-1.** Theorem 2.4.1 gives a Quillen equivalence between dendroidal sets with the relevant model structure and simplicial operads with the referenced model structure.

**HHM-2.** Theorem 2.5.1 constructs the model structure on forest sets and gives a Quillen equivalence with dendroidal sets via the stated adjoint pair.

**HHM-3.** Theorem 2.5.3 gives the Quillen equivalence between the category of preoperads and marked open forest sets.

**HHM-4.** Corollary 2.5.4 assembles the preceding equivalences into a zig-zag of Quillen equivalences between the dendroidal and Lurie-style models, under the no-constants/open-operads restriction stated in the paper.

**HHM-5.** Theorem 5.3.14 records the main Quillen equivalence for the slice version over $\langle 0\rangle/PO^o$ and the corresponding marked forest model, and is one of the internal proof endpoints for HHM-3.

**本书使用边界.** HHM-1--HHM-5 support 外部输入定理 M.12 和第十八章/第十九章的 dendroidal-Lurie comparison. They do not identify individual dendroidal sets with Lurie preoperads on the nose; they give a zig-zag of Quillen equivalences and require the model restrictions in the source.

## 3. Lurie: operadic nerve and category-of-operators entry

**来源.** Jacob Lurie, *Higher Algebra*, September 18, 2017.

**HA-OP-1.** Example 2.1.1.21 constructs from an ordinary colored operad $\mathcal O$ the category $\mathcal O^\otimes$ over $\operatorname{Fin}_*$ and states that $N(\mathcal O^\otimes)\to N(\operatorname{Fin}_*)$ is an infinity-operad.

**HA-OP-2.** Definition 2.1.1.23 defines the operadic nerve $N^\otimes(\mathcal O)$ of a simplicial colored operad as the simplicial nerve of $\mathcal O^\otimes$.

**HA-OP-3.** Proposition 2.1.1.27 says that if a simplicial colored operad $\mathcal O$ is fibrant, then $N^\otimes(\mathcal O)$ is an infinity-operad.

**本书使用边界.** HA-OP-1--HA-OP-3 support the category-of-operators/operadic nerve entry in Chapter 18 and Appendix M. They do not by themselves prove dendroidal-Lurie comparison; for that use HHM-1--HHM-5.

## 4. Pratali: operadic straightening

**来源.** Francesca Pratali, "A straightening-unstraightening equivalence for infinity-operads," arXiv:2501.05263v2, 2025-02-26.

**PRA-1.** Theorem 2.10 states that the Hinich-Moerdijk comparison functors induce an equivalence between operadic left fibrations over a Lurie infinity-operad and dendroidal left fibrations over the corresponding dendroidal model.

**PRA-2.** Proposition 3.8 characterizes, for a symmetric monoidal infinity-category, the strong symmetric monoidal left fibrations that arise from monoidal unstraightening of strong monoidal functors.

**PRA-3.** Proposition 4.6 relates operadic left fibrations over $\mathcal O^\otimes$ to strong symmetric monoidal left fibrations over the symmetric monoidal envelope $\operatorname{Env}(\mathcal O)^\otimes$.

**PRA-4.** Theorem 5.1 gives the operadic straightening-unstraightening equivalence
$$
\operatorname{Left}^{opd}_{\mathcal O^\otimes}
\simeq
\operatorname{Alg}_{\mathcal O^\otimes}(\mathcal S^\times)
$$
for Lurie infinity-operads.

**PRA-5.** Corollary 5.2 gives an explicit formula for operadic straightening when the infinity-operad is discrete.

**本书使用边界.** PRA-1--PRA-5 can replace the earlier vague "operadic straightening" placeholder for spaces-valued algebras/operadic left fibrations. Because the source is a 2025 preprint, this book should treat it as P1/latest locator unless a later published version is checked.

## 5. 本书使用边界

1. DKR-1--DKR-7 close the Dwyer--Kan/coherent-nerve locator needed in Chapter 19.
2. HHM-1--HHM-5 close the dendroidal-Lurie model comparison locator, with the paper's no-constants/open-operad restriction.
3. HA-OP-1--HA-OP-3 close category-of-operators/operadic nerve entry for strict or simplicial colored operads.
4. PRA-1--PRA-5 close the spaces-valued operadic straightening locator as a latest-source P1 input; they do not prove arbitrary $\mathcal C$-valued algebra comparison.
