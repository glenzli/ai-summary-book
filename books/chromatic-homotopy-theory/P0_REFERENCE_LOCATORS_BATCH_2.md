# P0/P1 Reference Locators Batch 2：Morava descent、tmf、Picard 与计算

核查日期：2026-07-12
状态：本批次七组外部输入均已定位到 theorem/definition/section/page，并附
DOI、期刊永久页、作者存档或机构存档链接。P1 项作为非主线接口保留，但其
引用定位已闭合。

## 1. Goerss-Hopkins-Miller theorem

**编号.** CHT-P0-12
**来源.** Paul G. Goerss and Michael J. Hopkins, *Moduli Spaces of
Commutative Ring Spectra*, in *Structured Ring Spectra*, London Mathematical
Society Lecture Note Series 315, Cambridge University Press, 2004,
151--200。
**本书用途.** 第三、六、八章：Morava E-theory 的 $\mathbb E_\infty$ 结构；extended Morava stabilizer group action；tmf 构造的 obstruction-theoretic 背景。
**可用陈述.** Lubin-Tate/Morava E-theory 可提升为结构化 commutative ring spectrum，且 stabilizer group action 可在结构化层面实现。
**禁止用途.** 不得由系数环 $W(k)[[u_i]][u^{\pm1}]$ 自动推出 $\mathbb E_\infty$ 结构；不得把 motivic Morava E-theory 自动纳入同一定理。
**精确 locator.** Section 7 “The Lubin--Tate Theories”；Proposition 7.1
给出所用 Lubin--Tate theory 的 Landweber-exact/Adams-type 输入；
Corollary 7.6, pp. 198--199，把其 $\mathbb E_\infty$ realization moduli
space 识别为 $B\operatorname{Aut}(\Gamma,k)$；Corollary 7.7, p. 199，
识别 Lubin--Tate spectra 间结构化映射空间的分支，从而提供 stabilizer
作用的结构化实现。稳定链接：
[Ravenel 文献存档 PDF](https://people.math.rochester.edu/faculty/doug/otherpapers/commring.PDF)。

## 2. Devinatz-Hopkins homotopy fixed points

**编号.** CHT-P0-13, CHT-P0-14
**来源.** Ethan S. Devinatz and Michael J. Hopkins, *Homotopy fixed point
spectra for closed subgroups of the Morava stabilizer groups*, Topology 43
(2004), no. 1, 1--47。
**本书用途.** 第六章、附录 J：闭子群 $H\le\mathbb G_n$ 的 homotopy fixed point spectra；$L_{K(n)}X$ 与 $(E_n\otimes X)^{h\mathbb G_n}$ 的比较；Morava descent spectral sequence。
**可用陈述.** 对合适对象和闭子群，Devinatz-Hopkins 构造 homotopy fixed point spectra，并提供 descent spectral sequence。
**禁止用途.** 不得把 closed profinite subgroup 当成有限离散群；不得省略连续性和收敛假设。
**精确 locator.** Theorem 1(iii)--(iv), pp. 3--4，给出 $F(*)\simeq
L_{K(n)}S^0$ 及开子群 fixed points 的连续谱序列；Definition 1.5,
p. 4，把构造扩展到闭子群，因而得到
$E_n^{h\mathbb G_n}\simeq L_{K(n)}S^0$；Theorem 2(i)--(ii), p. 5，
分别给出 Morava-module comparison 与对 CW spectrum $Z$ 的强收敛连续
cohomology spectral sequence；Proposition 6.7, pp. 34--35，在文中列明的
有限性/完备性假设下，把相应 $K(n)$-local $E_n$-Adams spectral
sequence 的 $E_2$ 页识别为连续群上同调。该命题的群、对象与收敛假设必须
按原文读取，不能把任意 $X$ 的比较无条件化。稳定链接：
[Ravenel 文献存档 PDF](https://people.math.rochester.edu/faculty/doug/otherpapers/dh04.pdf)，
[DOI](https://doi.org/10.1016/S0040-9383(03)00029-6)。

## 3. Hopkins-Kuhn-Ravenel character theory

**编号.** CHT-P1-10
**来源.** Michael J. Hopkins, Nicholas J. Kuhn, Douglas C. Ravenel, *Generalized group characters and complex oriented cohomology theories*, Journal of the American Mathematical Society 13 (2000), no. 3, 553-594。
**本书用途.** 第九章：transchromatic character 的经典高度固定版本；有限群 classifying spaces 上的 Morava E-theory character。
**可用陈述.** 对 Morava E-theory 和有限群，存在 generalized character map，把高度 $n$ 信息转化为 character-theoretic 数据。
**禁止用途.** 不得自动推广到所有 $\pi$-finite spaces 或 higher semiadditivity；transchromatic 版本需另引文献。
**精确 locator.** Theorem C, pp. 557--558：对有限群 $G$，generalized
character map 在扩张到 $L(E^*)$ 后成为同构，并描述相应不变量；
Sections 6.3--6.4, pp. 583--586，定义 generalized characters 并证明该
character theorem；Theorem D, p. 558，给出 induction/transfer 公式。
这是固定高度、有限群版本，不覆盖一般 transchromatic 或所有
$\pi$-finite spaces。稳定链接：
[作者存档 PDF](https://people.math.rochester.edu/faculty/doug/mypapers/hkr.pdf)，
[DOI](https://doi.org/10.1090/S0894-0347-00-00332-5)。

## 4. Elliptic spectra 和 tmf 构造背景

**编号.** CHT-P0-19, CHT-P0-20, CHT-P1-21
**来源.** Matthew Ando, Michael J. Hopkins, Neil P. Strickland, *Elliptic
spectra, the Witten genus and the theorem of the cube*, Inventiones
Mathematicae 146 (2001), 595--687；Paul G. Goerss, *Topological modular
forms [after Hopkins, Miller, and Lurie]*, Astérisque 332 (2010),
Exp. no. 1005, 221--255；Joseph H. Silverman, *The Arithmetic of Elliptic
Curves*, 2nd ed., Graduate Texts in Mathematics 106, Springer, 2009。
**本书用途.** 第八章、附录 K：椭圆曲线形式群、power operations、theorem of the cube、tmf sheaf construction。
**可用陈述.** 椭圆曲线形式群给出 height 1/2 分层；elliptic spectra 的 orientation 和 power operations 有几何约束；tmf 来自模栈上的结构化谱层。
**禁止用途.** 弱 elliptic datum 不等于 tmf；Landweber exact local chart 不等于 global $\mathbb E_\infty$ sheaf。
**精确 locator.** CHT-P0-19：Silverman, Chapter IV, Theorem 7.4 与
Corollary 7.5, p. 134，证明正特征域上椭圆曲线的形式群高度只能为 $1$
或 $2$；Ando--Hopkins--Strickland, Definition 1.2，定义 elliptic
spectrum 及其椭圆曲线形式群输入。CHT-P1-21：同文
Definition 2.40（cubical structure）、Corollary 2.50（cubical structures
与 $MU\langle6\rangle\to E$ 的乘法 orientation）和 Theorem 2.53
（theorem of the cube 所给唯一 cubical structure）；该包支持 sigma
orientation，不代表任意 power-operation 陈述。CHT-P0-20：Goerss,
Theorem 1.2, pp. 224--225，给出紧化椭圆模栈上的 derived structure sheaf
及其同伦层；Definition 1.3, p. 225，把 tmf 定义为 derived global
sections 并给出 descent spectral sequence。稳定链接：
[AHS DOI](https://doi.org/10.1007/s002220100175)，
[Silverman 图书 DOI](https://doi.org/10.1007/978-0-387-09494-6)，
[Goerss 机构存档 PDF](https://www.numdam.org/item/AST_2010__332__221_0.pdf)，
[Numdam 永久页](https://www.numdam.org/item/AST_2010__332__221_0/)。

## 5. Gross-Hopkins duality

**编号.** CHT-P1-12
**来源.** Neil P. Strickland, *Gross--Hopkins duality*, Topology 39 (2000),
1021--1033。
**本书用途.** 第十章、附录 L：$K(n)$-local Brown-Comenetz duality、determinant twist、dualizing objects。
**可用陈述.** Gross-Hopkins duality identifies Brown-Comenetz-type duality with Spanier-Whitehead duality up to invertible $K(n)$-local twists.
**禁止用途.** 不得在未指定 convention、悬挂、determinant sphere 和 exotic factor 时写简化公式。
**精确 locator.** Proposition 1, pp. 1021--1022，把 $K(n)$-local
Brown--Comenetz dualizing spectrum $\widehat I$ 证明为可逆对象，并比较
$\widehat I X$ 与 $\widehat I\widehat\wedge DX$；Theorem 2, p. 1022，
计算其 completed Morava $E$-homology 中的 determinant twist 和 grading；
Theorem 20 把 Kähler 微分的最高外幂识别为
$\Omega^{n-1}\cong\omega^{\otimes n}[\det]$。公式必须沿用原文 grading
与 determinant convention，不能据此省略可能的 exotic 信息。稳定链接：
[Ravenel 文献存档 PDF](https://www.sas.rochester.edu/mth/sites/doug-ravenel/otherpapers/gross.pdf)，
[arXiv](https://arxiv.org/abs/math/0011108)，
[DOI](https://doi.org/10.1016/S0040-9383(99)00049-X)。

## 6. Picard group 和 profinite descent

**编号.** CHT-P1-13, CHT-P1-14
**来源.** Paul G. Goerss, Hans-Werner Henn, Mark Mahowald, Charles Rezk,
*On Hopkins' Picard groups for the prime 3 and chromatic level 2*, Journal
of Topology 8 (2015), 267--294；Itamar Mor, *Picard and Brauer groups of
$K(n)$-local spectra via profinite Galois descent*, arXiv:2306.05393v2
(2023-10-12)。
**本书用途.** 第十章、附录 L：Picard comparison map、exotic Picard elements、Picard spectrum descent。
**可用陈述.** Picard groups can be studied through Morava-module algebraic approximations and profinite Galois descent; low-height computations reveal exotic elements.
**禁止用途.** 不得从 $(E_n)_*X$ rank-one 直接推出 $X$ 已由代数数据唯一决定。
**精确 locator.** Mor, Theorem A，给出
$\operatorname{pic}(\mathrm{Sp}_{K(n)})\simeq
\tau_{\ge0}\operatorname{pic}(E_n)^{h\mathbb G_n}$ 的 continuous
profinite Galois descent 比较，以及
$H^s_{\mathrm{cont}}(\mathbb G_n;\pi_t\operatorname{pic}(E_n))$
descent spectral sequence；Proposition 3.21 与 Corollary 3.24 精确比较其微分和
$K(n)$-local $E_n$-Adams spectral sequence；Theorem 4.4 把 algebraic
Picard 部分置于 filtration $\le1$，exotic kernel 置于 filtration
$\ge2$。低高度实例定位到 Goerss--Henn--Mahowald--Rezk, Theorems 1.1
和 1.2（$n=2,p=3$ 的 exotic 与完整 Picard groups）。稳定链接：
[Mor arXiv v2](https://arxiv.org/abs/2306.05393v2)，
[Mor v2 PDF](https://arxiv.org/pdf/2306.05393v2)，
[Mor 的 Ravenel 文献存档 PDF](https://www.sas.rochester.edu/mth/sites/doug-ravenel/otherpapers/Mor.pdf)，
[GHMR arXiv](https://arxiv.org/abs/1210.7033)，
[GHMR DOI](https://doi.org/10.1112/jtopol/jtu024)。

## 7. Adams-Novikov 和 chromatic spectral sequence

**编号.** CHT-P1-17, CHT-P1-22
**来源.** Douglas C. Ravenel, *Complex Cobordism and Stable Homotopy
Groups of Spheres*, 2nd ed., AMS Chelsea。
**本书用途.** 第十二章、附录 B/C/M：$BP_*BP$-comodule Ext、chromatic spectral sequence、low-stem table、hidden extensions。
**可用陈述.** Adams--Novikov spectral sequence 的 $E_2$ 页由 Hopf
algebroid comodule Ext 给出；其 abutment、过滤与收敛范围按来源定理的
connectivity 和 completion 假设读取。
**禁止用途.** 不得把 $E_\infty$ 页直接等同于最终 homotopy groups；不得忽略 hidden extensions。
**精确 locator.** Theorem 4.4.1, p. 130，给出 $BP$-based
Adams--Novikov spectral sequence、
$E_2=\operatorname{Ext}_{BP_*BP}(BP_*,BP_*X)$ 及 connective 情形的
过滤/收敛陈述；Chapter 2 的 completion 结果控制何时把该目标改写成特定
local 或 complete target。Definition 5.1.7 与 Proposition 5.1.8,
p. 150，由 chromatic resolution 构造 chromatic spectral
sequence 并给出其 $E_1$ 页与收敛到 $\operatorname{Ext}_{BP_*BP}$ 的陈述；
Definition 5.1.10 给出 chromatic cobar complex。稳定链接：
[Ravenel 官方 PDF](https://www.sas.rochester.edu/mth/sites/doug-ravenel/mybooks/ravenel.pdf)。

## 8. 闭合判定

本批次七组来源均具有可复核的一手 locator。CHT-P0-12、13、14、19、20
进入主线外部输入；CHT-P1-10、12、13、14、17、21、22 是已定位的非主线
接口。P1 标签限制其证明角色，不降低引用精度，也不得作为超出上述定理量词
和收敛范围的概括性许可证。
