# 引用定位账本

本文件服务于最终出版态。附录 D 已经说明哪些结论是外部输入；本文件进一步规定哪些外部输入必须在出版前精确定位，哪些只需作为背景文献，哪些必须停留在研究边界。

## 0. 定位等级

**P0：证明链必需。** 若定位缺失，正文相关定理不能作为教材定理使用。必须补齐作者、题名、版本或出版信息、定理/命题/引理编号、假设转换。

**P1：结构解释必需。** 若定位缺失，正文可保留定义和接口，但不能宣称模型比较、等价或分类定理。

**P2：背景文献。** 用于历史、动机、例子或进一步阅读；不进入证明链。

**R：研究边界。** 近期预印本或尚未吸收的前沿结果；不得作为基础证明步骤。

## 1. P0 定位包

| 包 | 本书使用位置 | 必须定位的断言 | 当前来源/状态 | 当前允许用法/边界 |
| --- | --- | --- | --- | --- |
| Koszul 判别 | 第八、九章，附录 I/Q | $\Omega\mathcal P^¡\to\mathcal P$ 在 Koszul 情形为 quasi-isomorphism；Ass/Com/Lie Koszul | Ginzburg-Kapranov；Loday-Vallette；Fresse | GK classical core 为 GK-1--GK-5；Loday--Vallette Theorems 6.6.2/7.4.6 已登记为 LV-1--LV-2，Theorem 8.1.1 + following $\operatorname{As}$ example 已登记为 LV-3；Fresse model-category twisting acyclicity 为 FRE-1--FRE-3 |
| Bar-cobar resolution | 第九、十章，附录 I/Q | bar-cobar counit/unit 的 weak equivalence 与 cofibrant resolution 条件 | Ginzburg-Kapranov；Loday-Vallette；Fresse；Hinich | LV-1 给出 connected weight-graded quasi-isomorphism criterion；GK-6--GK-7 给出 classical core；FRE-4--FRE-6 给出带 cofibrancy 假设的 modern resolution 输入。Quasi-isomorphism 与 operad-model cofibrancy 分开登记 |
| Homological perturbation | 第十三章，附录 J/S | contraction 上的 perturbation lemma 与转移公式 | Gugenheim-Lambe-Stasheff；Kadeishvili；Markl；Merkulov | 低阶公式可内部用；Markl 的 operadic transfer existence 已由 [P0_REFERENCE_LOCATORS_BATCH_6.md](P0_REFERENCE_LOCATORS_BATCH_6.md) 中 MHT-1--MHT-8 定位；basic perturbation lemma 显式公式保留为 HPT convention translation，不作为 operad theory locator 空缺 |
| Homotopy transfer | 第十三章，附录 J/S/W | $A_\infty/L_\infty/\mathcal P_\infty$ 转移和 minimal model uniqueness | Kadeishvili；Markl；Merkulov；Loday-Vallette；Fresse | Markl strongly homotopy transfer over chain homotopy equivalence 已由 [P0_REFERENCE_LOCATORS_BATCH_6.md](P0_REFERENCE_LOCATORS_BATCH_6.md) 中 MHT-1--MHT-8 定位；Kadeishvili minimal model、tree signs 和 uniqueness 保留为 final sign/convention package，不作为 operad theory locator 空缺 |
| Operad transferred model structure | 第十四章，附录 G/R | operad 和 operad algebra 的 transferred model structures | Berger-Moerdijk；Hinich；Fresse；Pavlov-Scholbach | Berger-Moerdijk 部分已由 [P0_REFERENCE_LOCATORS_BATCH_1.md](P0_REFERENCE_LOCATORS_BATCH_1.md) 定位；Hinich dg-operad model structure 已由 [P0_REFERENCE_LOCATORS_BATCH_5.md](P0_REFERENCE_LOCATORS_BATCH_5.md) 中 HIN-1 定位；Pavlov--Scholbach modern colored/all-small admissibility 已由 [P0_REFERENCE_LOCATORS_BATCH_9.md](P0_REFERENCE_LOCATORS_BATCH_9.md) 中 PSAR-1--PSAR-3 与 PSP-1--PSP-2 定位；逐底范畴仍需假设翻译 |
| Rectification criterion | 第十四、十九章，附录 G/R/X | operad weak equivalence induces Quillen equivalence under flatness/cofibrancy assumptions | Hinich；Berger-Moerdijk；Pavlov-Scholbach | Berger-Moerdijk 早期版本已由 [P0_REFERENCE_LOCATORS_BATCH_1.md](P0_REFERENCE_LOCATORS_BATCH_1.md) 定位；Hinich $\Sigma$-split homotopy-category 版本已由 [P0_REFERENCE_LOCATORS_BATCH_5.md](P0_REFERENCE_LOCATORS_BATCH_5.md) 中 HIN-2 定位；Pavlov--Scholbach modern flatness/rectification 已由 [P0_REFERENCE_LOCATORS_BATCH_9.md](P0_REFERENCE_LOCATORS_BATCH_9.md) 中 PSAR-4--PSAR-6 定位；不得无假设推广 |
| Dendroidal model structure | 第十六、十七章，附录 T | Cisinski-Moerdijk operadic model structure；normal monomorphism facts | Moerdijk-Weiss；Cisinski-Moerdijk | Cisinski-Moerdijk operadic model structure 已由 [P0_REFERENCE_LOCATORS_BATCH_1.md](P0_REFERENCE_LOCATORS_BATCH_1.md) 定位；Moerdijk-Weiss strict nerve/inner Kan 前置已由 [P0_REFERENCE_LOCATORS_BATCH_7.md](P0_REFERENCE_LOCATORS_BATCH_7.md) 中 MW-4--MW-6 定位；erratum 影响仍需最终核查 |
| Dendroidal nerve fully faithful | 第十六、十七章，附录 M/T | strict colored operads 嵌入 dendroidal sets 的 fully faithfulness/Segal characterization | Moerdijk-Weiss | Moerdijk-Weiss fully faithfulness、$\Delta\subset\Omega$ 和 strict nerve unique fillers 已由 [P0_REFERENCE_LOCATORS_BATCH_7.md](P0_REFERENCE_LOCATORS_BATCH_7.md) 中 MW-1--MW-4 定位 |
| Dwyer-Kan localization | 第十九章 | relative category 的 hammock/simplicial localization | Dwyer-Kan；Hinich；Hirschhorn | Hinich 的 infinity-categorical localization、hammock localization comparison、underlying infinity-category 和 Quillen-pair passage 已由 [P0_REFERENCE_LOCATORS_BATCH_10.md](P0_REFERENCE_LOCATORS_BATCH_10.md) 中 DKR-1--DKR-7 定位；若最终版回溯 Dwyer--Kan 原文，可作为 bibliography 增强而非空缺 locator |
| Straightening/unstraightening | 第十九章 | coCartesian fibrations 与 functor categories 的等价 | Lurie HTT；Pratali | Ordinary straightening 已由 [P0_REFERENCE_LOCATORS_BATCH_2.md](P0_REFERENCE_LOCATORS_BATCH_2.md) 定位；spaces-valued operadic straightening 已由 [P0_REFERENCE_LOCATORS_BATCH_10.md](P0_REFERENCE_LOCATORS_BATCH_10.md) 中 PRA-1--PRA-5 作最新 P1/preprint locator 定位 |
| Monoidal/algebra localization | 第十九章 | localization 与 algebra objects 的比较 | White；White--Yau；Lurie HA；Hinich；Pavlov-Scholbach | 模型范畴中 Bousfield localization preserves operad/colored-operad algebras 已由 [P0_REFERENCE_LOCATORS_BATCH_8.md](P0_REFERENCE_LOCATORS_BATCH_8.md) 中 WHT-1--WHT-4 与 WY-1--WY-3 定位；strict associative/commutative algebra comparison 与 colored strict-to-infinity algebra comparison 已由 [P0_REFERENCE_LOCATORS_BATCH_9.md](P0_REFERENCE_LOCATORS_BATCH_9.md) 中 HA-ALG-1--HA-ALG-3 和 PSAR-5 定位；任意 colored/infinity algebra localization 仍需按模型假设选择具体定理 |
| Factorization homology excision | 第二十章，附录 N/V | excision 与 relative tensor product 公式 | Ayala-Francis；Lurie | Ayala-Francis topological manifolds 版本已由 [P0_REFERENCE_LOCATORS_BATCH_3.md](P0_REFERENCE_LOCATORS_BATCH_3.md) 定位；stratified 与 Fukaya 几何版本已在 [P1_REFERENCE_LOCATORS_FINAL_SWEEP.md](P1_REFERENCE_LOCATORS_FINAL_SWEEP.md) 中登记为边界 locator，不能由 operad theory 内部证明替代 |
| Locally constant factorization algebra | 第二十章，附录 N | locally constant factorization algebras on $\mathbb R^n$ 与 $E_n$-algebras 等价 | Costello-Gwilliam；Lurie；Ayala-Francis | Ayala-Francis homology theories 与 Disk$_n$-algebras 的刻画已由 [P0_REFERENCE_LOCATORS_BATCH_3.md](P0_REFERENCE_LOCATORS_BATCH_3.md) 定位；Costello-Gwilliam/Lurie locally constant factorization algebra 等价已在 [P1_REFERENCE_LOCATORS_FINAL_SWEEP.md](P1_REFERENCE_LOCATORS_FINAL_SWEEP.md) 中保留为外部几何/field-theoretic boundary locator，不作为 operad theory 内部证明 |
| Fukaya category construction | 第二十章，附录 N/Z | holomorphic polygon counts define $A_\infty$ category under analytic hypotheses | Seidel；Fukaya-Oh-Ohta-Ono | 已由 [FINAL_OPERAD_THEORY_CLOSURE.md](FINAL_OPERAD_THEORY_CLOSURE.md) 关闭为几何外部边界；不进入 operad theory 内部证明链 |

## 2. P1 定位包

| 包 | 本书使用位置 | 必须定位的断言 | 当前来源/状态 | 当前允许用法/边界 |
| --- | --- | --- | --- | --- |
| May recognition principle | 第十、十五章 | $E_n$-spaces 与 iterated loop spaces 的识别 | May；Boardman-Vogt | 已由 [P1_REFERENCE_LOCATORS_FINAL_SWEEP.md](P1_REFERENCE_LOCATORS_FINAL_SWEEP.md) 登记为 P1 背景 locator；不作为 algebraic operad proof-chain input |
| $H_\*(E_n)$ 为 Poisson | 第十章，附录 F/L | little disks/cubes 同调 operad 识别 | Cohen；Fresse | 已由 [P1_REFERENCE_LOCATORS_FINAL_SWEEP.md](P1_REFERENCE_LOCATORS_FINAL_SWEEP.md) 登记为 P1 topology locator；定理级使用仍需系数环和 degree convention |
| $E_n$ 形式性 | 第十、十一章，附录 L/W | 特征 $0$ 下链模型形式性 | Kontsevich；Tamarkin；Fresse | 已由 [P1_REFERENCE_LOCATORS_FINAL_SWEEP.md](P1_REFERENCE_LOCATORS_FINAL_SWEEP.md) 登记为 P1 characteristic-zero/formality boundary；本书核心证明链不使用该定理 |
| Deligne conjecture | 第十一、十二章 | Hochschild cochains 上的 $E_2$ action | McClure-Smith；Berger-Fresse；Kontsevich-Soibelman | McClure--Smith 和 Berger--Fresse locator 已由 [P1_REFERENCE_LOCATORS_FINAL_SWEEP.md](P1_REFERENCE_LOCATORS_FINAL_SWEEP.md) 中 MS-1--MS-3 与 BF-1--BF-4 定位；本书 suspended brace signs 仍需附录 W 逐项核对 |
| Framed $E_2$ 与 BV | 第十一章 | framed little disks 同调为 BV | Getzler；Fresse | 已由 [P1_REFERENCE_LOCATORS_FINAL_SWEEP.md](P1_REFERENCE_LOCATORS_FINAL_SWEEP.md) 登记为 P1 BV locator；需固定 framed model、circle action 和 grading convention |
| Category of operators comparison | 第十八章，附录 M | ordinary colored operad 的 operators nerve 给 Lurie-style 对象 | Lurie；Hinich | Lurie *Higher Algebra* Example 2.1.1.21、Definition 2.1.1.23 和 Proposition 2.1.1.27 已由 [P0_REFERENCE_LOCATORS_BATCH_10.md](P0_REFERENCE_LOCATORS_BATCH_10.md) 中 HA-OP-1--HA-OP-3 定位 |
| Dendroidal-Lurie comparison | 第十八、十九章，附录 M | dendroidal 与 Lurie-style infinity-operad 模型比较 | Heuts-Hinich-Moerdijk | HHM Quillen-equivalence zig-zag 已由 [P0_REFERENCE_LOCATORS_BATCH_10.md](P0_REFERENCE_LOCATORS_BATCH_10.md) 中 HHM-1--HHM-5 定位；使用时必须保留 open/no-constants 等来源限制 |
| Dunn/Lurie additivity | 第二十章，附录 L/N | $E_m\otimes E_n\simeq E_{m+n}$ 型结论 | Dunn；Lurie | Lurie *Higher Algebra* Theorem 5.1.2.2 已由 [P1_REFERENCE_LOCATORS_FINAL_SWEEP.md](P1_REFERENCE_LOCATORS_FINAL_SWEEP.md) 中 DUNN-1 定位；strict topological operad tensor product 版本仍需另引 |
| Stratified factorization homology | 附录 V/Z | conically smooth stratified spaces 上的 factorization homology | Ayala-Francis-Tanaka；Lurie | 已由 [P1_REFERENCE_LOCATORS_FINAL_SWEEP.md](P1_REFERENCE_LOCATORS_FINAL_SWEEP.md) 登记为几何边界 locator；不进入 operad theory 内部证明链 |
| Wrapped Fukaya descent | 第二十章，附录 N/V/Z | Liouville sectors/stops 下的 descent 或 gluing | Ganatra-Pardon-Shende；Seidel；Nadler | 已由 [P1_REFERENCE_LOCATORS_FINAL_SWEEP.md](P1_REFERENCE_LOCATORS_FINAL_SWEEP.md) 登记为几何边界 locator；需要具体几何模型和分析假设 |

## 3. P2 背景源

| 来源 | 用途 | 定位要求 |
| --- | --- | --- |
| May, *The Geometry of Iterated Loop Spaces* | operad 历史、recognition 背景 | 章节级即可，若作定理输入则升为 P1 |
| Boardman-Vogt, *Homotopy Invariant Algebraic Structures on Topological Spaces* | $W$-construction、tensor product、homotopy invariant structures | 章节级即可，若作定理输入则升为 P0/P1 |
| Markl-Shnider-Stasheff, *Operads in Algebra, Topology and Physics* | 基础定义、例子、PROP/properad 背景 | 章节级即可，若作图群胚定理输入则升为 P0 |
| Loday-Vallette, *Algebraic Operads* | 线性 operad、Koszul、bar-cobar、同伦代数 | 大量结论为 P0，背景叙述为 P2 |
| Fresse, *Homotopy of Operads and Grothendieck-Teichmüller Groups* | 同伦 operad、$E_n$、形式性 | 定理化使用升为 P0/P1 |
| Leinster, *Higher Operads, Higher Categories* | higher operads 和 multicategory 背景 | 背景使用为 P2 |
| Lurie, *Higher Algebra* / *Higher Topos Theory* | infinity-operad、straightening、factorization | 定理化使用为 P0/P1 |

## 4. R 研究边界源

| 来源 | 本书状态 | 进入正文所需 |
| --- | --- | --- |
| Hoffbeck-Moerdijk, “Homology of infinity-operads” | 研究边界/附录 Y 接口 | linear infinity-operad 定义、定理编号、与 strict Koszul 比较 |
| Hoffbeck-Moerdijk, “Koszul duality for algebras over infinity-operads” | 研究边界/附录 Y 接口 | algebra/coalgebra homotopy theory、bar-cobar 定理编号 |
| Pavlova, “Boardman-Vogt tensor product and wreath product of operadic categories” | 研究边界/附录 Z 接口 | operadic categories 公理、wreath product 与 BV tensor product 定理编号 |
| Arakawa-Carmona-Pratali, “Relative dendroidal Rezk nerve and applications” | 研究边界/附录 Z 接口 | relative infinity-operad 模型、Rezk nerve fibrancy、localization universal property |
| Batanin-Kock-Weber, “Operadic categories as (pseudo)-simplicial groupoids” | 研究边界/附录 Z 接口 | pseudo-simplicial groupoid 相干性、operadic nerve 定理编号 |
| Yuan, “Higher operad structure for Fukaya categories” | 研究边界/附录 Z 接口 | $\mathbf{fc}$-multicategory 定义、几何分析输入、符号转换 |

## 5. 引用升级规则

**规则 5.1.** P0 条目未定位时，正文可以保留定义、条件性命题和证明边界，但不得把相应结论作为已完成教材定理。

**规则 5.2.** P1 条目未定位时，正文可以说明用途和关系，但不得声称模型比较或等价已经由本书证明。

**规则 5.3.** P2 条目不得被引用为证明步骤；若某处需要它证明命题，必须先把该条目升级到 P0 或 P1。

**规则 5.4.** R 条目不得进入第一至二十章的证明链。若要进入，必须先通过流程 21.16，并按定义 D.0.2 在附录 D 和本文件中重新分类。

## 6. 当前出版状态

本文件完成后，书稿的引用状态为：

- 基本完本严格草稿：通过；
- operad theory 数学收口态：通过；
- camera-ready 出版态：未通过；
- 剩余 production work：HPT/minimal-model 符号约定、几何/Fukaya 假设包、bibliography normalization 以及 page/tag 级校对。Koszul/twisting 的现代书本编号已由 LV-1--LV-3 关闭；模型范畴 cofibrancy 仍使用 FRE/HIN 包。已定位批次见 [P0_REFERENCE_LOCATORS_BATCH_1.md](P0_REFERENCE_LOCATORS_BATCH_1.md)、[P0_REFERENCE_LOCATORS_BATCH_2.md](P0_REFERENCE_LOCATORS_BATCH_2.md)、[P0_REFERENCE_LOCATORS_BATCH_3.md](P0_REFERENCE_LOCATORS_BATCH_3.md)、[P0_REFERENCE_LOCATORS_BATCH_4.md](P0_REFERENCE_LOCATORS_BATCH_4.md)、[P0_REFERENCE_LOCATORS_BATCH_5.md](P0_REFERENCE_LOCATORS_BATCH_5.md)、[P0_REFERENCE_LOCATORS_BATCH_6.md](P0_REFERENCE_LOCATORS_BATCH_6.md)、[P0_REFERENCE_LOCATORS_BATCH_7.md](P0_REFERENCE_LOCATORS_BATCH_7.md)、[P0_REFERENCE_LOCATORS_BATCH_8.md](P0_REFERENCE_LOCATORS_BATCH_8.md)、[P0_REFERENCE_LOCATORS_BATCH_9.md](P0_REFERENCE_LOCATORS_BATCH_9.md)、[P0_REFERENCE_LOCATORS_BATCH_10.md](P0_REFERENCE_LOCATORS_BATCH_10.md)、[P1_REFERENCE_LOCATORS_FINAL_SWEEP.md](P1_REFERENCE_LOCATORS_FINAL_SWEEP.md) 和 [FINAL_OPERAD_THEORY_CLOSURE.md](FINAL_OPERAD_THEORY_CLOSURE.md)。

因此，下一阶段若继续，只应做 production/copy-editing，而不是继续增加新主题或继续寻找已经命名的 locator。
