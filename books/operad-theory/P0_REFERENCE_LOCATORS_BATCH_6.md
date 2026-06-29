# P0 引用定位批次 6：Homotopy transfer via Markl

本文件记录第六批已精确定位的 P0 外部输入：Markl 对 strongly homotopy $\mathcal P$-algebra 在 chain homotopy equivalence 下不变性的证明。它支撑第十三章和附录 J/S 中“homotopy transfer theorem 的存在性”这一核心外部输入。

本批次不完全覆盖 basic perturbation lemma 的显式公式、Kadeishvili minimal model uniqueness、Merkulov/Loday--Vallette/Fresse 中的完整 tree sign convention。它把 homotopy transfer 的一个 operadic/cofibrant-resolution 版本精确定位；剩余符号和 minimal model 唯一性按附录 W 与 `FINAL_OPERAD_THEORY_CLOSURE.md` 作为 sign/convention package 管理。

## 1. Markl：strongly homotopy structures transfer over homotopy equivalences

**主来源.** Martin Markl, “Homotopy Algebras are Homotopy Algebras,” arXiv:math/9907138v3.

**本书对应位置.** 第十三章，附录 J/S/W，附录 D.3、D.12.2，REFERENCE_LOCATOR_LEDGER 中 P0 “Homotopy transfer”。

### 1.1 Cofibrant colored-operad factorization technology

**定位 MHT-1.** Markl, arXiv:math/9907138v3, Definition 17.

**定位 MHT-2.** Markl, arXiv:math/9907138v3, Theorem 19 and Lemma 20.

**本书使用.** 第十三章外部输入定理 13.6 的 operadic proof framework：用 filtered/free colored-operad cofibrations 和 lifting/factorization 来扩展 strongly homotopy diagrams。

**需要同时记录的假设.**

1. 来源中的 cofibration 是 elemental cofibration，使用生成元滤过。
2. 底层语境是 characteristic zero field 上的 chain complexes。
3. 该技术是 Markl 证明 moves 的基础；它本身不是 HPL 的显式级数公式。

### 1.2 Homotopy extension property

**定位 MHT-3.** Markl, arXiv:math/9907138v3, Theorem 27.

**本书使用.** 第十三章中“homotopy extension problem for diagrams over cofibrant operads reduces to the classical extension problem”的来源。它是从 operad cofibrancy 推出 transfer moves 的关键抽象步骤。

**需要同时记录的假设.**

1. 使用 colored operads $P(S,D)$ 描述 diagrams。
2. 结论针对来源定义的 homotopy $P$-extension property。
3. 本书只用它支撑存在性，不用它替代具体 $A_\infty/L_\infty$ 树公式。

### 1.3 Side conditions 的作用

**定位 MHT-4.** Markl, arXiv:math/9907138v3, Proposition 31.

**本书使用.** 附录 J 中 normalized contraction side conditions 的外部边界：side conditions 不是装饰性假设，它们控制 ordinary SDR data 是否能延拓到来源的 $R_{\mathrm{iso}}$ action。

**需要同时记录的假设.**

1. 该命题解释 side conditions 与 strongly homotopy equivalence 的关系。
2. 它不替代 basic perturbation lemma，也不提供本书定义 J.6 的所有树符号。

### 1.4 Homotopy equivalence resolution

**定位 MHT-5.** Markl, arXiv:math/9907138v3, Theorem 33.

**本书使用.** 支撑 transfer moves 中把 cofibrant resolution 与 homotopy-equivalence operad $R_{\mathrm{iso}}$ 结合后仍得到 homology isomorphism 的步骤。

**需要同时记录的假设.**

1. $P$ 是 ordinary operad，满足 $P(0)=0$、$P(1)\cong k$ 且 differential trivial。
2. $S$ 是 $P$ 或 $P_{\bullet\to\bullet}$。
3. $R\to S$ 是 cofibrant resolution。

### 1.5 Transfer over chain homotopy equivalence

**定位 MHT-6.** Markl, arXiv:math/9907138v3, Proposition 34.

**本书使用.** 第十三章外部输入定理 13.6 的核心引用：给定 strongly homotopy $P$-algebra $V$、chain complex $W$ 和 chain homotopy equivalence $f:V\to W$，存在 $W$ 上的 strongly homotopy $P$-structure，并且 $f$ 获得 strongly homotopy $P$-morphism structure。

**需要同时记录的假设.**

1. 来源假设 characteristic zero field。
2. Strongly homotopy $P$-algebra 意味着代数 over a cofibrant model of $P$。
3. 来源中默认 non-unital 情形；unital 版本需另行核查。

### 1.6 Homotopic maps and inverse transfer moves

**定位 MHT-7.** Markl, arXiv:math/9907138v3, Proposition 35.

**定位 MHT-8.** Markl, arXiv:math/9907138v3, Proposition 36.

**本书使用.** 第十三章和附录 S 中“转移结构不仅存在，而且 homotopic underlying maps 与 homotopy inverse maps 也可获得 strongly homotopy morphism structure”的外部来源。

**需要同时记录的假设.**

1. MHT-7 是 move (M2)：若 chain map $g$ 与已有 sh morphism 的 underlying map chain homotopic，则 $g$ 可获得 strongly homotopy structure。
2. MHT-8 是 move (M3)：若 sh morphism 的 underlying map 是 chain homotopy equivalence，则其 homotopy inverse 可获得 strongly homotopy structure。
3. 这些结论不自动给出 minimal model uniqueness 或 explicit formula signs。

## 2. 与本书现有文件的替换规则

| 旧表述 | 替换为 |
| --- | --- |
| Homotopy transfer theorem, operadic existence version | MHT-1--MHT-6 |
| Strongly homotopy structures transfer over chain homotopy equivalences | MHT-6 |
| Side conditions for SDR data are necessary in transfer proofs | MHT-4 |
| Homotopic underlying maps inherit sh morphism structure | MHT-7 |
| Chain homotopy inverse of an sh equivalence inherits sh morphism structure | MHT-8 |

## 3. 本批次边界

本批次不直接解决：

1. Basic perturbation lemma 的 Gugenheim--Lambe--Stasheff/Huebschmann 版本 theorem locator；
2. Kadeishvili 原始 $A_\infty$ minimal model theorem 的 exact theorem locator；
3. Merkulov 或 Loday--Vallette 中 $L_\infty$ transfer tree formula 的 exact theorem locator；
4. Loday--Vallette/Fresse 书本版本的 full operadic homotopy transfer theorem 编号；
5. 本书附录 E/J/S/W 中 unsuspended sign convention 的逐项外部核对；
6. Minimal model uniqueness 与 formality obstruction 的 theorem locator。

这些不作为 operad theory 主证明链的未定位缺口；它们按 [REFERENCE_LOCATOR_LEDGER.md](REFERENCE_LOCATOR_LEDGER.md) 与 [FINAL_OPERAD_THEORY_CLOSURE.md](FINAL_OPERAD_THEORY_CLOSURE.md) 归入 sign/convention package 或 production-level bibliography work。
