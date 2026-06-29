# P0 引用定位批次 5：Modern cobar resolution and Hinich model context

本文件记录第五批已精确定位的 P0 外部输入：Fresse 的 operadic cobar construction/cofibrant replacement 定理，以及 Hinich 的 dg-operad 模型结构和 $\Sigma$-split operad algebra 比较。它推进上一批留下的 modern bar-cobar/cofibrant resolution 缺口。

本批次不声称已经完全替代 Loday--Vallette 书中 $\Omega\mathcal P^¡\to\mathcal P$ 的紧凑 Koszul 判别表述；它给出可直接引用的 modern cobar/cofibrant-replacement 来源，并把剩余缺口压缩为 convention translation 与 Loday--Vallette/Fresse book theorem locator。

## 1. Fresse：operadic cobar construction 与 quasi-free replacement

**主来源.** Benoit Fresse, “Operadic cobar constructions, cylinder objects and homotopy morphisms of algebras over operads,” arXiv:0902.0177.

**本书对应位置.** 第九、十、十三、十四章，附录 I/Q/L/R，附录 D.2、D.12.1，REFERENCE_LOCATOR_LEDGER 中 P0 “Bar-cobar resolution”和 P0 “Koszul 判别”的 modern cofibrant-resolution 部分。

### 1.1 Twisting cochains 与 cobar universal property

**定位 FRE-1.** Fresse, arXiv:0902.0177, Section 3.7.

**本书使用.** 第九章定理 9.20、推论 9.21 和附录 I 中 twisting morphism 与 cobar morphism 的对应：
$$
\operatorname{Mor}_{\mathcal O}(B^c(D),P)\cong \operatorname{Tw}_{\mathcal O}(D,P).
$$

**需要同时记录的假设.**

1. 来源使用 $B^c(D)$ 表示 cooperad $D$ 的 operadic cobar construction。
2. Twisting cochain 是定义在 cooperad 上、在 unit 上为零的次数 $-1$ 映射。
3. 本书的 $\Omega\mathcal C$ 记号与来源 $B^c(D)$ 需要通过 suspension/desuspension convention 对齐。

### 1.2 Universal twisted composite object 的 acyclicity

**定位 FRE-2.** Fresse, arXiv:0902.0177, Theorem 3.9.

**本书使用.** 附录 Q 中 bar-cobar counit 的低权重计算之后，用作 universal twisting cochain $\iota:D\to B^c(D)$ 给出的 twisted composite object
$$
(B^c(D)\circ D,\partial_\iota)
$$
到单位 $\Sigma_\ast$-object 的 weak equivalence 来源。

**需要同时记录的假设.**

1. $D$ 是 $C$-cofibrant connected cooperad。
2. 结论是 $\Sigma_\ast$-object 层面的 weak equivalence，不是无条件的 operad-level Quillen equivalence。
3. 证明使用 filtration/spectral sequence；本书使用时必须保留 connectedness 和 cofibrancy 条件。

### 1.3 Twisting weak equivalence 推出 twisted object acyclicity

**定位 FRE-3.** Fresse, arXiv:0902.0177, Theorem 3.10.

**本书使用.** 第九章外部输入定理 9.23 和附录 I/Q 中“若 $B^c(D)\to P$ 是 weak equivalence，则 $(P\circ D,\partial_\theta)\to I$ 是 weak equivalence”的 modern source。

**需要同时记录的假设.**

1. $D$ 是 $C$-cofibrant connected cooperad。
2. $P$ 是 $C$-cofibrant operad。
3. $\theta:D\to P$ 对应的 $\varphi_\theta:B^c(D)\to P$ 必须是 operad weak equivalence。
4. 该定理支撑 twisted composite acyclicity；若要得到 specific Koszul dual cooperad $\mathcal P^¡$ 的 resolution，还必须证明或引用 $\varphi_\kappa:B^c(\mathcal P^¡)\to\mathcal P$ 是 weak equivalence。

### 1.4 Bar duality 给出的 $B^cB(P)\to P$ 弱等价入口

**定位 FRE-4.** Fresse, arXiv:0902.0177, Section 3.14.

**本书使用.** 附录 Q 外部输入定理 Q.19 中
$$
B^c(B(P))\longrightarrow P
$$
作为 bar-cobar resolution 的 modern reference entry。

**需要同时记录的假设.**

1. $P$ 是 $C$-cofibrant augmented operad。
2. reduced 条件写作 $\widetilde P(0)=\widetilde P(1)=0$。
3. 来源在该段引用 Ginzburg--Kapranov Theorem 3.2.16 和 Fresse 早期结果；若最终出版要求完全独立证明，应继续定位 Fresse [5, §4.8] 或 Loday--Vallette 的相应 theorem。

### 1.5 Algebra 的 quasi-free/cofibrant replacement

**定位 FRE-5.** Fresse, arXiv:0902.0177, Theorem 4.2.4.

**本书使用.** 第十章和附录 L/R 中“由 cobar resolution 控制的 homotopy $\mathcal P$-algebra 有 quasi-free/cofibrant replacement”的主要来源之一。对于 $\Sigma_\ast$-cofibrant operad $P$、$\Sigma_\ast$-cofibrant connected cooperad $D$ 和 weak equivalence $B^c(D)\to P$，来源给出
$$
R_A=R_P(D(A),\partial_\alpha)\longrightarrow A
$$
是 $P$-algebras 中的 weak equivalence，且 $R_A$ 是 cofibrant replacement。

**需要同时记录的假设.**

1. $P$ 是 $\Sigma_\ast$-cofibrant operad。
2. $D$ 是 $\Sigma_\ast$-cofibrant connected cooperad。
3. $\theta:D\to P$ 对应的 $B^c(D)\to P$ 是 weak equivalence。
4. $A$ 是 $C$-cofibrant $P$-algebra。
5. 该定理处理 algebra-level replacement，不等于所有 operad algebra categories 的全局 rectification theorem。

### 1.6 Homotopy morphism 的同伦范畴入口

**定位 FRE-6.** Fresse, arXiv:0902.0177, Proposition 4.2.7 and Proposition 4.2.8.

**本书使用.** 第十三章和附录 S 中“quasi-cofree coalgebra morphism 表示 homotopy category 中的 algebra morphism”的来源入口。

**需要同时记录的假设.** 这些命题依赖 FRE-5 的 quasi-free replacement 设置；不得把它们独立用作任意 $\infty$-morphism 理论。

## 2. Hinich：dg-operad 模型结构和 $\Sigma$-split 比较

**主来源.** Vladimir Hinich, “Homological algebra of homotopy algebras,” arXiv:q-alg/9702015.

**本书对应位置.** 第十四、十九章，附录 G/R，附录 D.4、D.12.3，REFERENCE_LOCATOR_LEDGER 中 P0 “Operad transferred model structure”和 P0 “Rectification criterion”的 chain-complex model context。

### 2.1 Dg-operads 的 closed model category

**定位 HIN-1.** Hinich, arXiv:q-alg/9702015, Theorem 6.1.1.

**本书使用.** 第十四章和附录 G/R 中 $\mathbf{Ch}_k$ 上 dg-operads 的模型结构来源：weak equivalences 逐 arity 为 quasi-isomorphisms，fibrations 逐 arity 为 surjections。

**需要同时记录的假设.**

1. 来源语境是 complexes over a field/ring $k$ 的 operads，需按原文底环假设使用。
2. 该定理给出 operads 的模型结构，不自动给出所有 colored operads 或任意 symmetric monoidal model category 中的 admissibility。
3. 与 Berger--Moerdijk/Pavlov--Scholbach 的 general admissibility theorem 仍需分开引用。

### 2.2 $\Sigma$-split operad 弱等价诱导代数同伦范畴等价

**定位 HIN-2.** Hinich, arXiv:q-alg/9702015, Theorem 4.7.4.

**本书使用.** 第十四章、附录 G/R/X 中特征零或 $\Sigma$-split 条件下 rectification 的 classical dg source：若 $\alpha:O\to O'$ 是与 splittings 相容的 $\Sigma$-split operad quasi-isomorphism，则 induced adjunction 在 homotopy categories 上给出等价。

**需要同时记录的假设.**

1. Operads 必须是 $\Sigma$-split，且 morphism 与 splittings 相容。
2. 结论是 homotopy categories 的等价；若正文需要 Quillen equivalence 或 infinity-categorical equivalence，需要额外模型比较。
3. 该结果不能推广到正特征中 $\operatorname{Com}$ 与 $E_\infty$ 的无条件 rectification。

## 3. 与本书现有文件的替换规则

| 旧表述 | 替换为 |
| --- | --- |
| Modern cobar/twisting universal property | FRE-1 |
| Universal cobar twisted composite is acyclic | FRE-2 |
| Twisting weak equivalence implies twisted composite acyclicity | FRE-3 |
| Bar-cobar counit $B^cB(P)\to P$ as weak equivalence entry | FRE-4；完整书本 wording 已由 FINAL_OPERAD_THEORY_CLOSURE 关闭为 convention/bibliography production work |
| Algebra quasi-free/cofibrant replacement from cobar resolution | FRE-5 |
| Homotopy morphisms represented by quasi-cofree coalgebra morphisms | FRE-6 |
| Dg-operad model structure in Hinich's setting | HIN-1 |
| $\Sigma$-split operad rectification / homotopy category equivalence | HIN-2 |

## 4. 本批次未解决

本批次不解决：

1. Loday--Vallette *Algebraic Operads* 中 $\Omega\mathcal P^¡\to\mathcal P$、acyclic twisting morphism 和 bar-cobar counit 的 exact theorem numbering；
2. Fresse 书本版本中相同结论的 theorem locator；
3. Colored/all-small operad admissibility 的 Pavlov--Scholbach/Hinich modern flatness 版本；
4. Positive-characteristic divided-power、curved、unital 或 inhomogeneous Koszul duality；
5. Dendroidal/infinity-operadic Koszul duality 的 Hoffbeck--Moerdijk 前沿版本。

这些仍按 [REFERENCE_LOCATOR_LEDGER.md](REFERENCE_LOCATOR_LEDGER.md) 的 P0/P1 列表推进。
