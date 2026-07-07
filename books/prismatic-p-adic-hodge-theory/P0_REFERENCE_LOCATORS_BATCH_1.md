# P0 外部输入 locator：第一批源码级核查

核查日期：2026-07-08。

本文件记录 P0 外部输入定理的第一批 source-label locator。它不新增数学定理；它只把正文和附录中已经作为外部输入使用的结果定位到可复查的论文版本、TeX 标签和源码行附近。出版前仍需把这些 locator 转换为正式页码、定理号和期刊版本引用。

## Locator 等级说明

- `L1`：论文版本级 locator。
- `L2S`：源码标签级 locator，有 arXiv 版本、TeX 文件、label 和源码行附近位置。
- `L3`：正式出版页码、定理号、命题号或定义号 locator。

本批次把核心 P0 条目从 `L1` 升级为 `L2S`。由于 arXiv TeX 源码行号不是出版稳定编号，本批次不能替代最终 `L3`。

## P0-1：Bhatt-Scholze, Prisms and Prismatic Cohomology

来源版本：Bhatt-Scholze, arXiv:1905.08229, v4, 2022-01-12。
本地核查源：`/private/tmp/prismatic_locator/1905/prisms.tex`。

| 本书 ID | 用途 | label | 源码位置 | 当前等级 |
| --- | --- | --- | --- | --- |
| BS-PRISM-DEF | prism、perfect、bounded、orientable、crystalline、flat maps | `DefPrismCat` | line 1008 附近 | L2S |
| BS-PRISM-DEF | bounded prisms 的基础性质 | `BoundedPrismProp` | line 1074 附近 | L2S |
| BS-PRISM-DEF | relative prismatic site 的引入 | `PrismaticSiteIntro` | line 184 附近 | L2S |
| BS-PERF | perfect prisms 与 perfectoid rings 的等价 | `PerfdPrism` | line 1128 附近 | L2S |
| BS-COMP | 主比较定理总述 | `thm:A` | line 203 附近 | L2S |
| BS-COMP | crystalline comparison | `CrysComp` | line 1507 附近 | L2S |
| BS-COMP | Hodge-Tate comparison | `HTCompPrismatic` | line 1687 附近 | L2S |
| BS-COMP | de Rham comparison 初步版本 | `dRComp1` | line 1704 附近 | L2S |
| BS-COMP | etale comparison | `EtaleCompThm` | line 2187 附近 | L2S |
| BS-COMP | bounded prism base change | `BaseChangePrismCoh` | line 1374 附近 | L2S |
| BS-COMP | general de Rham comparison | `generaldeRham` | line 3292 附近 | L2S |
| BS-COMP | Frobenius image | `ImageofPhi` | line 3303 附近 | L2S |
| BS-NYG | relative Nygaard theorem | `thmCagain` | line 3248 附近 | L2S |

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
本地核查源：`/private/tmp/prismatic_locator/1602/integralpadicHodge.tex`。

| 本书 ID | 用途 | label | 源码位置 | 当前等级 |
| --- | --- | --- | --- | --- |
| BMS1-AINF | proper smooth formal schemes 的 $A_{\inf}$ complex | `ThmB` | line 332 附近 | L2S |
| BMS1-AINF | sheaf-level $A\Omega_{\mathfrak X}$ comparisons | `ThmC` | line 384 附近 | L2S |
| BMS1-AINF | $A\Omega_{\mathfrak X}$ intro definition | `IntroAOmegaDef` | line 425 附近 | L2S |
| BMS1-QDR | torus 上的 $q$-de Rham computation | `ThmD` | line 503 附近 | L2S |
| BMS1-LETA | $L\eta$ functor existence | `cor:LetaExists` | line 2252 附近 | L2S |
| BMS1-LETA | lax symmetric monoidal structure | `prop:Letalaxsymmmon` | line 2270 附近 | L2S |
| BMS1-LETA | Bockstein description | `prop:LetaBock` | line 2371 附近 | L2S |
| BMS1-LETA | completeness preservation | `lem:Letapreservecompleteness` | line 2528 附近 | L2S |
| BMS1-LETA | completion compatibility | `lem:Letacommutecompletion` | line 2540 附近 | L2S |
| BMS1-AINF | $A\Omega$ versus de Rham-Witt package | `thm:AOmegavsdRW` | line 3277 附近 | L2S |
| BMS1-AINF | Frobenius on $A\Omega$ | `prop:PhionAOmega` | line 3828 附近 | L2S |
| BMS1-AINF | crystalline comparison | `thm:cryscomp` | line 4669 附近 | L2S |
| BMS1-AINF | rational $B_{\mathrm{dR}}^+$ comparison | `thm:ratpadicHodgeC` | line 5027 附近 | L2S |
| BMS1-AINF | Hodge-Tate theorem | `thm:hodgetate` | line 5048 附近 | L2S |
| BMS1-BK | Kisin functor | `ThmKisin` | line 1501 附近 | L2S |
| BMS1-BKF | Fargues BKF classification | `ThmFargues` | line 1911 附近 | L2S |

### 对正文的约束

第七章关于 $L\eta$、Bockstein 和 syntomic 口径的陈述必须只作为 BMS 外部输入或形式推论使用。若需要证明 $L\eta$ 保完备化或 monoidal 结构，正文应引用本表中的 `BMS1-LETA` locator，而不在本书中伪装为内部证明。

## P0-3：Bhatt-Scholze, Prismatic F-crystals and crystalline Galois representations

来源版本：Bhatt-Scholze, arXiv:2106.14735, v2, 2023-09-12。
本地核查源：`/private/tmp/prismatic_locator/2106/PrismaticCrystals.tex`。

| 本书 ID | 用途 | label | 源码位置 | 当前等级 |
| --- | --- | --- | --- | --- |
| BS-FCRYS | prismatic $F$-crystal intro definition | `PrismaticFCrysIntro` | line 113 附近 | L2S |
| BS-FCRYS | main theorem intro statement | `MainThmIntro` | line 123 附近 | L2S |
| BS-FCRYS | prismatic $F$-crystal formal definition | `PrismaticFCrysDef` | line 483 附近 | L2S |
| BS-FCRYS | crystalline lattice equivalence | `MainThm` | line 657 附近 | L2S |
| BS-FCRYS-BK | Breuil-Kisin prism over $\mathcal O_K$ | `BKAinfPrismOK` | line 241 附近 | L2S |
| BS-FCRYS-BK | etale realization for BK prism | `EtaleRealizeBK` | line 1023 附近 | L2S |
| BS-FCRYS-BK | recovery of Kisin full faithfulness | `KisinFullyFaithfulBK` | line 1156 附近 | L2S |
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

本批次消除了以下 P0 风险：

- Hodge-Tate twist convention 已与 Bhatt-Scholze v4 源码一致。
- Nygaard graded formula 已定位到 `thmCagain`，并确认应带 Frobenius twist、$\tau^{\le i}$ 和 $\{i\}$。
- $A\Omega$、$L\eta$、BMS comparison 的基础引用已从书目级升级到源码 label 级。
- prismatic $F$-crystal 的对象定义、effective 条件和主等价已从书目级升级到源码 label 级。

仍未完成：

- 正式出版页码、定理号、定义号的 `L3` locator。
- Fontaine/Faltings/Tsuji classical comparison 的最终教材源选择。
- BMS2/THH-Breuil-Kisin comparison 的源码级 locator。
- Bhatt-Lurie prismatization 的源码级 locator。
