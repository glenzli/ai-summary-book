# P0 Reference Locators Batch 1：基础 chromatic 定理包

核查日期：2026-07-12
状态：本批次主线 P0 输入均已定位到 theorem/section/page，并附 DOI 或
长期学术存档链接。Hovey--Strickland 中未被正文调用的全书级结构结果归入
非主线 P1 资料扩展，不构成本批次闭合条件。

本文件定位本书证明链中最基础的外部输入。定理编号以所列版本为准；页码均为
期刊或图书的印刷页码，而不是 PDF 阅读器页码。

## 1. Quillen theorem

**编号.** CHT-P0-01
**来源.** Daniel Quillen, *On the formal group laws of unoriented and complex
cobordism theory*, Bulletin of the American Mathematical Society 75 (1969),
1293--1298。
**本书用途.** 第二章：$MU_*$ 与 Lazard ring 的同构；$MU$ 的复定向给 universal one-dimensional commutative formal group law。
**可用陈述.** 复定向 ring spectrum $E$ 的形式群律由 $MU_*\to E_*$ 分类。
**禁止用途.** 不得由 Quillen theorem 直接推出 $BP\langle n\rangle$ 的 $\mathbb E_k$ quotient structure。
**精确 locator.** Theorem 2, pp. 1294--1295：由复 cobordism 的形式群律
诱导的 Lazard ring 到复 cobordism 系数环的同态是同构。正文从
$MU_*\cong L$ 到给定复定向的分类同态还使用复定向的 Thom 表示性质；这一步
不是额外的 $\mathbb E_\infty$ 结论。稳定链接：
[Ravenel 文献存档 PDF](https://people.math.rochester.edu/faculty/doug/otherpapers/QuillenBP.pdf)，
[DOI](https://doi.org/10.1090/S0002-9904-1969-12401-8)。

## 2. Brown-Peterson splitting 和 $BP_*$ 计算

**编号.** CHT-P0-02
**来源.** Quillen 上述论文；Douglas C. Ravenel, *Complex Cobordism and
Stable Homotopy Groups of Spheres*, 2nd ed., AMS Chelsea；Hazewinkel
formal-group convention。
**本书用途.** 第二章：固定素数 $p$ 后写
$$
BP_*\cong \mathbb Z_{(p)}[v_1,v_2,\ldots],\qquad |v_i|=2(p^i-1).
$$
**可用陈述.** $BP$ 是 $MU_{(p)}$ 的 $p$-typical summand；Hazewinkel generators 给出本书 convention。
**禁止用途.** 不得把 $BP_*$ 的代数商自动提升为结构化环谱商。
**精确 locator.** Quillen, Theorem 4, pp. 1296--1297，构造 $MU_{(p)}$
上的 $p$-typical 幂等元并把其像识别为 Brown--Peterson theory；Ravenel,
Theorem 4.1.12(c), p. 108，把 $BP$ 识别为 $MU_{(p)}$ 的 retract 并给出
$BP_*=\mathbb Z_{(p)}[v_1,v_2,\ldots]$ 及次数；Theorem 4.1.18(a),
p. 111，与 Theorem A2.1.25, p. 349，给出 universal $p$-typical
coefficient ring。Hazewinkel generators 精确位于 (A2.2.1), p. 354；
Theorem A2.2.3, pp. 354--355，证明 Hazewinkel 与 Araki 两组元素均生成
$V$ 且模 $p$ 同余。Ravenel 随后采用 Araki generators，因此本书把符号
$v_i$ 指定为 Hazewinkel 坐标时必须按附录 A 翻译，不能直接等同两套公式；
谱级 retract 也不推出结构化商。稳定链接：
[Quillen PDF](https://people.math.rochester.edu/faculty/doug/otherpapers/QuillenBP.pdf)，
[Ravenel 官方 PDF](https://www.sas.rochester.edu/mth/sites/doug-ravenel/mybooks/ravenel.pdf)。

## 3. Landweber exact functor theorem

**编号.** CHT-P0-03
**来源.** Peter S. Landweber, *Homological properties of comodules over
$MU_*(MU)$ and $BP_*(BP)$*, American Journal of Mathematics 98 (1976),
591--610；Ravenel 书中 Chapter 4, Section 2。
**本书用途.** 第二章、第八章、附录 K：从形式群数据构造同调理论，解释 elliptic cohomology 的 Landweber-exact 局部例子。
**可用陈述.** 满足 Landweber exactness 条件的 $MU_*$-代数给出 homology theory。
**禁止用途.** Landweber exactness 不给 $\mathbb E_\infty$ 结构，不给 tmf sheaf。
**精确 locator.** Landweber, Theorem 2.6（extension of scalars 的正合性）
与 Corollary 2.7（所得函子定义广义同调理论）；Ravenel, Chapter 4,
Section 2, pp. 115--116，以正则序列条件重述同一判据。该定位只给同调理论，
不提供结构化乘法或谱层。稳定链接：
[JSTOR 永久页](https://www.jstor.org/stable/2373808)，
[DOI](https://doi.org/10.2307/2373808)，
[Ravenel 官方 PDF](https://www.sas.rochester.edu/mth/sites/doug-ravenel/mybooks/ravenel.pdf)。

## 4. Devinatz-Hopkins-Smith nilpotence theorem

**编号.** CHT-P0-04
**来源.** Ethan S. Devinatz, Michael J. Hopkins, Jeffrey H. Smith, *Nilpotence and stable homotopy theory I*, Annals of Mathematics 128 (1988), 207-241。
**本书用途.** 第四章：nilpotence theorem；有限谱 periodicity/thick subcategory theorem 的基础输入之一。
**可用陈述.** 对 ring spectrum 中元素，$MU$-homology 检测 nilpotence；等价表述需按来源精确翻译。
**禁止用途.** 不得把 nilpotence theorem 改写成“$MU_*x=0$ 当且仅当 $x=0$”。结论是 nilpotent，不是零。
**精确 locator.** Devinatz--Hopkins--Smith I, Theorem 1(i)（ring-spectrum
kernel-of-Hurewicz 版本）。Hopkins--Smith II, Theorem 2 明确回引
“[DHS I, Theorem 1]”，可用于核对版本翻译。

## 5. Hopkins-Smith periodicity theorem

**编号.** CHT-P0-05
**来源.** Michael J. Hopkins, Jeffrey H. Smith, *Nilpotence and stable homotopy theory II*, Annals of Mathematics 148 (1998), 1-49；Ravenel *Nilpotence and Periodicity in Stable Homotopy Theory*。
**本书用途.** 第四章、附录 I：type $n$ 有限谱存在 $v_n$ self-map；$v_n$ self-map 唯一到幂；telescope Bousfield class 的选择无关。
**可用陈述.** 每个 type $n$ finite spectrum admits a $v_n$-self map，且任意两个在取幂后相容。
**禁止用途.** Periodicity theorem 不推出 telescope conjecture。
**精确 locator.** Hopkins--Smith II, Definition 8 与 Theorem 9（存在性
及可取规范幂）；Corollary 3.7（同一有限谱上唯一到幂）；Corollary 3.8
（不同 type-$n$ 有限谱间映射的幂次相容性）；Theorem 14 用于跨有限谱
的 Bousfield class comparison。

## 6. Hopkins-Smith thick subcategory theorem

**编号.** CHT-P0-06
**来源.** Hopkins-Smith, *Nilpotence and stable homotopy theory II*；Chebolu, *Thick subcategories in stable homotopy theory*, arXiv:math/0607245 作为教学性二级核对。
**本书用途.** 第四章：有限 $p$-局部谱的 thick subcategories 为
$$
\mathcal C_0\supsetneq\mathcal C_1\supsetneq\mathcal C_2\supsetneq\cdots.
$$
**可用陈述.** 每个非零 proper thick subcategory 等于某个 $\mathcal C_n$。
**禁止用途.** 不得用于分类一般 localizing subcategories 或非有限谱子范畴。
**精确 locator.** Hopkins--Smith II, Theorem 7；$\mathcal C_n$ 的定义
及嵌套链位于同文 Introduction、Theorem 7 之前。严格包含还需各 type
有限谱存在性的输入。

## 7. Morava K finite detection

**编号.** CHT-P0-07
**来源.** Devinatz-Hopkins-Smith/Hopkins-Smith 定理包；Ravenel 书；Hovey-Strickland Morava K localization 结果。
**本书用途.** 第三、四章：非零有限谱被某个 $K(n)$ 检测，从而 type 有定义。
**可用陈述.** 若 $X$ 为有限 $p$-局部谱且 $K(n)_*X=0$ 对所有 $n\ge0$，则 $X\simeq0$。
**禁止用途.** 不得用于任意无限谱。
**精确 locator.** Hopkins--Smith II, Theorem 14。若有限谱 $X$ 的
Morava 检测集合为空，则该定理把 $X$ 与零谱判为 Bousfield 等价；令
smash 因子为球谱即得 $X\simeq0$。

## 8. Ravenel-Hopkins chromatic convergence

**编号.** CHT-P0-09
**来源.** Ravenel, *Nilpotence and Periodicity in Stable Homotopy Theory*；Hopkins-Ravenel chromatic convergence theorem。
**本书用途.** 第五章：有限 $p$-局部谱满足
$$
X\simeq\operatorname*{holim}_nL_nX.
$$
**可用陈述.** finite spectra 的 chromatic tower 收敛。
**禁止用途.** 不得用于任意谱、filtered colimit 或未验证完备性的对象。
**精确 locator.** Ravenel, *Nilpotence and Periodicity in Stable Homotopy
Theory*, Theorem 7.5.7；完整证明在 Section 8.6，关键为 Lemma 8.6.5
及其后用 Adams filtration、Mittag--Leffler 与 $\lim^1$ 的论证。Lurie
Lecture 32, Theorem 1 为 infinity-category 记号下的交叉核对。

## 9. Hovey-Strickland Morava localization package

**编号.** CHT-P0-08, CHT-P0-08A, CHT-P0-10, CHT-P1-18
**来源.** Mark Hovey and Neil P. Strickland, *Morava K-theories and localisation*, Memoirs of the AMS 139 (1999)。
**本书用途.** 第三、五章、附录 H：Morava K-theory localization、$K(n)$-local category、$K(n)$-module field-like behavior、Bousfield class comparison 和 local duality 背景。
**可用陈述.** Morava K-theories and localizations 的标准结构定理；具体命题需按章节定位。
**禁止用途.** 不得把所有 $K(n)$-local 结论自动推广到 $T(n)$-local 或 $E(n)$-local。
**精确 locator（本轮闭合的核心子包）.**

- CHT-P0-08：Lurie Lecture 23, Proposition 2；Hovey,
  *Bousfield Localization Functors and Hopkins' Chromatic Splitting
  Conjecture*, Corollary 1.12。
- CHT-P0-08A（smash product）：Ravenel Theorem 7.5.6；Lurie Lecture 23,
  Theorem 4。
- CHT-P0-10（fracture）：Lurie Lecture 23, Proposition 5；量词为每个
  $n\ge1$、每个 $p$-局部谱 $X$。
- CHT-P1-18 的 field/Künneth 子包定位到 Hopkins--Smith II,
  Propositions 1.4、1.5。Hovey--Strickland 中其余未被正文逐项调用的
  $K(n)$-local category 结果登记为非主线 P1 资料扩展；在另行登记具体命题
  前，正文不得把这条概括性书目当作证明输入。

## 10. 闭合判定

CHT-P0-01、CHT-P0-02、CHT-P0-03 以及本文件原有的 finite chromatic
定理包均具有可复核的一手 locator。`THEOREM_LEDGER.md` 记录编号与正文用途，
`SOURCES.md` 记录版本和稳定链接。stable infinity-category 记号只作陈述翻译，
不会扩大原来源的对象范围、乘法结构或收敛结论。
