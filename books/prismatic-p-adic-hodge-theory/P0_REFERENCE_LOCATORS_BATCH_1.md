# P0 外部输入 locator：第一批源码级核查

初次核查日期：2026-07-08。PDF 定理号复核：2026-07-11。

本文件记录 P0 外部输入定理的第一批 source-label locator。它不新增数学定理；它只把正文和附录中已经作为外部输入使用的结果定位到可复查的论文版本、TeX 标签和源码行附近。出版前仍需把这些 locator 转换为正式页码、定理号和期刊版本引用。

## Locator 等级说明

- `L1`：论文版本级 locator。
- `L2S`：源码标签级 locator，有 arXiv 版本、TeX 文件、label 和源码行附近位置。
- `L3`：正式出版页码、定理号、命题号或定义号 locator。

2026-07-08 的初次核查把核心 P0 条目从 `L1` 升级为 `L2S`；
2026-07-11 又把正文实际使用且已核对 PDF 编号的条目升级为 `L3`。
仍只有 TeX label/line 的 rows 不能冒充稳定编号，继续保持 `L2S`。

## P0-1：Bhatt-Scholze, Prisms and Prismatic Cohomology

来源版本：Bhatt-Scholze, arXiv:1905.08229, v4, 2022-01-12。
核查介质：上述 arXiv 版本的 PDF 与 TeX source snapshot；下表保留 source
labels/line neighborhoods，但不依赖临时解压路径。

| 本书 ID | 用途 | source label / PDF locator | 源码位置 | 当前等级 |
| --- | --- | --- | --- | --- |
| BS-DCOMP | Koszul derived completion、complete flatness、complete Tor-amplitude | §1.2, pp. 10-11 | PDF 与源码复核 | L3 |
| BS-PRISM-DEF | prism、perfect、bounded、orientable、crystalline、flat maps | `DefPrismCat`; Definition 3.2 | line 1008 附近 | L3 |
| BS-PRISM-DEF | prism map 的 $J=IB$ rigidity | Proposition 3.5 | PDF | L3 |
| BS-PRISM-DEF | bounded prisms 的经典完备性与 flat modules | `BoundedPrismProp`; Lemma 3.7 | line 1074 附近 | L3 |
| BS-PRISM-SITE | relative prismatic site、topology、structure sheaves | Corollary 3.12; Definition 4.1 | `PrismaticSiteIntro` 附近 | L3 |
| BS-PERF | perfect prisms 与 perfectoid rings 的等价 | `PerfdPrism`; Theorem 3.10 | line 1128 附近及 PDF | L3 |
| BS-COMP | 主比较定理总述 | `thm:A`; Theorem 1.8 | line 203 附近 | L3 |
| BS-COMP-CRYS | crystalline comparison | `CrysComp`; Theorem 5.2 | line 1507 附近 | L3 |
| BS-COMP-HT | Hodge-Tate comparison | `HTCompPrismatic`; Theorems 4.11, 6.3 | line 1687 附近 | L3 |
| BS-COMP-DR | general de Rham comparison | `generaldeRham`; Corollary 15.4 | line 3292 附近 | L3 |
| BS-COMP-ETALE | finite-level etale comparison | `EtaleCompThm`; Theorem 9.1 | line 2187 附近 | L3 |
| BS-COMP-BC | bounded prism base change | `BaseChangePrismCoh`; Corollary 4.12 | line 1374 附近 | L3 |
| BS-COMP-PHI | Frobenius image/isogeny | `ImageofPhi`; Corollary 15.5 | line 3303 附近 | L3 |
| BS-COMP-AINF | $A_{\inf}$-cohomology equals Frobenius pullback of prismatic cohomology | Theorem 17.2 and global descent | §17 | L3 |
| BS-COMP-BK | Breuil-Kisin/prismatic comparison | Example 1.9 (3), Proposition 15.7 and concluding paragraph of §15.2, p. 105 | PDF | L3 |
| BS-NYG | relative Nygaard theorem、graded pieces、Frobenius factorization | `thmCagain`; Theorem 1.16 / Theorem 15.3 | line 3248 附近及 PDF | L3 |

### 约定核查结果

在 `thm:A` 的 Hodge-Tate comparison 中，Bhatt-Scholze 采用
$$
M\{i\}=M\otimes_{A/I}(I/I^2)^{\otimes i}.
$$
在 affine smooth 情形，公式写成
$$
\Omega^i_{R/(A/I)}\{-i\}
\cong
H^i(R\Gamma_{\Prism}(X/A)\otimes_A^L A/I).
$$
因此本书附录 F 的 twist convention 与该版本一致。

在 `thmCagain` 中，Nygaard filtration 放在 Frobenius twist 后的 prismatic cohomology 上：
$$
\mathrm{Fil}^i_N R\Gamma_{\Prism}(X/A)^{(1)}
=R\Gamma(X_{\mathrm{qsyn}},\mathrm{Fil}^i_N\Prism^{(1)}_{-/A}),
$$
并有
$$
\operatorname{gr}^i_N R\Gamma_{\Prism}(X/A)^{(1)}
\cong
\tau^{\le i}\overline{\Prism}_{R/A}\{i\}.
$$
Frobenius 经由
$$
R\Gamma_{\Prism}(X/A)^{(1)}
\xrightarrow{\widetilde\varphi}
L\eta_I R\Gamma_{\Prism}(X/A)
\to
R\Gamma_{\Prism}(X/A),
$$
且 $\widetilde\varphi$ 为同构。第七章和第十一章凡使用 Nygaard 或 syntomic fibre 公式时，必须把上述 Frobenius twist 与 $\tau^{\le i}$ 写入假设或交叉引用附录 F。

## P0-2：Bhatt-Morrow-Scholze, Integral p-adic Hodge theory

来源版本：Bhatt-Morrow-Scholze, arXiv:1602.03148, v3 final, 2019-01-15。
核查介质：上述 arXiv 版本的 PDF 与 TeX source snapshot；临时 source
路径不作为出版 locator。

| 本书 ID | 用途 | label | 源码位置 | 当前等级 |
| --- | --- | --- | --- | --- |
| BMS1-AINF | proper smooth formal schemes 的 $A_{\inf}$ complex 与四个 comparisons | `ThmB`; Theorems 1.8, 14.3 | line 332 附近 | L3 |
| BMS1-AINF | sheaf-level $A\Omega_{\mathfrak X}$ comparisons | `ThmC` | line 384 附近 | L2S |
| BMS1-AINF | $A\Omega_{\mathfrak X}$ intro definition | `IntroAOmegaDef` | line 425 附近 | L2S |
| BMS1-QDR | torus 上的 $q$-de Rham computation | `ThmD` | line 503 附近 | L2S |
| BMS1-LETA | $L\eta$ functor existence、filtered colimits、truncations | `cor:LetaExists`; Corollary 6.5 | line 2252 附近及 PDF | L3 |
| BMS1-LETA | lax symmetric monoidal structure | `prop:Letalaxsymmmon`; Proposition 6.7 | line 2270 附近及 PDF | L3 |
| BMS1-LETA | Bockstein description modulo $I$ | `prop:LetaBock`; Proposition 6.12 | line 2371 附近及 PDF | L3 |
| BMS1-LETA | completeness preservation | `lem:Letapreservecompleteness`; Lemma 6.19 | line 2528 附近及 PDF | L3 |
| BMS1-LETA | same-ideal completion compatibility | `lem:Letacommutecompletion`; Lemma 6.20 | line 2540 附近及 PDF | L3 |
| BMS1-AINF | $A\Omega$ versus de Rham-Witt package | `thm:AOmegavsdRW` | line 3277 附近 | L2S |
| BMS1-AINF | Frobenius on $A\Omega$ | `prop:PhionAOmega` | line 3828 附近 | L2S |
| BMS1-AINF | crystalline comparison | `thm:cryscomp` | line 4669 附近 | L2S |
| BMS1-AINF | rational $B_{\mathrm{dR}}^+$ comparison | `thm:ratpadicHodgeC` | line 5027 附近 | L2S |
| BMS1-AINF | Hodge-Tate theorem | `thm:hodgetate` | line 5048 附近 | L2S |
| BMS1-BK | Kisin functor | `ThmKisin` | line 1501 附近 | L2S |
| BMS1-BKF | BKF definition、Fargues classification、cohomological torsion thresholds | Definition 1.5; `ThmFargues`/Theorem 4.28; Theorem 14.5 (iii) | line 1911 附近及 PDF | L3 |

### 对正文的约束

第七章关于 $L\eta$、Bockstein 和 syntomic 口径的陈述必须只作为 BMS 外部输入或形式推论使用。若需要证明 $L\eta$ 保完备化或 monoidal 结构，正文应引用本表中的 `BMS1-LETA` locator，而不在本书中伪装为内部证明。

## P0-3：Bhatt-Scholze, Prismatic F-crystals and crystalline Galois representations

来源版本：Bhatt-Scholze, arXiv:2106.14735, v2, 2023-09-12。
核查介质：上述 arXiv 版本的 PDF 与 TeX source snapshot；临时 source
路径不作为出版 locator。

| 本书 ID | 用途 | label | 源码位置 | 当前等级 |
| --- | --- | --- | --- | --- |
| BS-FCRYS | prismatic $F$-crystal formal definition | `PrismaticFCrysDef`; Definition 4.1 | line 483 附近 | L3 |
| BS-FCRYS | crystalline lattice equivalence | `MainThm`; Theorem 5.6 | line 657 附近 | L3 |
| BS-FCRYS-BK | Breuil-Kisin prism over $\mathcal O_K$ and evaluation | `BKAinfPrismOK`; Example 2.6, Example 4.3 | line 241 附近 | L3 |
| BS-FCRYS-BK | etale realization for BK prism | `EtaleRealizeBK` | line 1023 附近 | L2S |
| BS-FCRYS-BK | recovery of Kisin full faithfulness | `KisinFullyFaithfulBK`; Theorem 7.9 | line 1156 附近 | L3 |
| BS-FCRYS-BK | Breuil-Kisin/Breuil full faithfulness | `KisinBreuil` | line 1192 附近 | L2S |
| BS-FCRYS-BK | logarithmic connection from BK evaluation | `LogConnectionBK` | line 1294 附近 | L2S |

### 对正文的约束

本书第六章和第十二章可以把 prismatic $F$-crystal 定义写成：vector bundle $\mathcal E$ 加同构
$$
\varphi_{\mathcal E}:\varphi^*\mathcal E[1/\mathcal I_{\Prism}]
\xrightarrow{\sim}
\mathcal E[1/\mathcal I_{\Prism}],
$$
effective 条件为 Frobenius 把 $\varphi^*\mathcal E$ 送入 $\mathcal E$。主等价只应表述为外部输入定理：
$$
\mathrm{Vect}^{\varphi}(X_{\Prism},\mathcal O_{\Prism})
\simeq
\mathrm{Rep}^{\mathrm{crys}}_{\mathbf Z_p}(G_K)
$$
在论文假设下成立。

## 本批次结论

本批次及 2026-07-11 PDF 复核消除了以下 P0 风险：

- Hodge-Tate twist convention 已与 Bhatt-Scholze v4 源码一致。
- Nygaard graded formula 已定位到 `thmCagain`，并确认应带 Frobenius twist、$\tau^{\le i}$ 和 $\{i\}$。
- $A\Omega$、$L\eta$、BMS comparison 的基础引用已从书目级升级到源码 label 级。
- prismatic $F$-crystal 的对象定义、effective 条件和主等价已从书目级升级到源码 label 级。
- Derived completion、prism/site、基础 comparisons、BMS1/BKF 与
  prismatic $F$-crystal 主链已升级到 PDF numbered-statement level `L3`。

仍未完成：

- 未进入本轮正文主链的 $A\Omega$/de Rham--Witt 配套技术引理之 PDF numbered locator。
- Fontaine/Faltings/Tsuji classical comparison 的最终教材源选择。
- BMS2/THH-Breuil-Kisin comparison 与 Bhatt-Lurie prismatization 的第二批 locator 见 [P0_REFERENCE_LOCATORS_BATCH_2.md](P0_REFERENCE_LOCATORS_BATCH_2.md)；其中正文所用的 BMS2 主定理已完成 `L3` 复核，其余条目仍按该表分级。
