# P0 Reference Locators Batch 1：基础 chromatic 定理包

核查日期：2026-07-08
状态：bibliographic locator 已建立；精确 theorem/section/page locator 待下一批 PDF/书本逐条补齐。

本文件定位本书证明链中最基础的外部输入。当前不伪造 theorem number；若尚未逐页核对，则标为“精确编号待补”。

## 1. Quillen theorem

**编号.** CHT-P0-01
**来源.** Daniel Quillen, *On the formal group laws of unoriented and complex cobordism theory*, Bulletin of the American Mathematical Society.
**本书用途.** 第二章：$MU_*$ 与 Lazard ring 的同构；$MU$ 的复定向给 universal one-dimensional commutative formal group law。
**可用陈述.** 复定向 ring spectrum $E$ 的形式群律由 $MU_*\to E_*$ 分类。
**禁止用途.** 不得由 Quillen theorem 直接推出 $BP\langle n\rangle$ 的 $\mathbb E_k$ quotient structure。
**精确 locator.** 待补。

## 2. Brown-Peterson splitting 和 $BP_*$ 计算

**编号.** CHT-P0-02
**来源.** Brown-Peterson 原始构造；Ravenel, *Complex Cobordism and Stable Homotopy Groups of Spheres*；Hazewinkel formal group convention。
**本书用途.** 第二章：固定素数 $p$ 后写
$$
BP_*\cong \mathbb Z_{(p)}[v_1,v_2,\ldots],\qquad |v_i|=2(p^i-1).
$$
**可用陈述.** $BP$ 是 $MU_{(p)}$ 的 $p$-typical summand；Hazewinkel generators 给出本书 convention。
**禁止用途.** 不得把 $BP_*$ 的代数商自动提升为结构化环谱商。
**精确 locator.** 待补。

## 3. Landweber exact functor theorem

**编号.** CHT-P0-03
**来源.** Peter S. Landweber, exact functor theorem 相关论文；Ravenel 书中 Landweber exactness 章节。
**本书用途.** 第二章、第八章、附录 K：从形式群数据构造同调理论，解释 elliptic cohomology 的 Landweber-exact 局部例子。
**可用陈述.** 满足 Landweber exactness 条件的 $MU_*$-代数给出 homology theory。
**禁止用途.** Landweber exactness 不给 $\mathbb E_\infty$ 结构，不给 tmf sheaf。
**精确 locator.** 待补。

## 4. Devinatz-Hopkins-Smith nilpotence theorem

**编号.** CHT-P0-04
**来源.** Ethan S. Devinatz, Michael J. Hopkins, Jeffrey H. Smith, *Nilpotence and stable homotopy theory I*, Annals of Mathematics 128 (1988), 207-241。
**本书用途.** 第四章：nilpotence theorem；有限谱 periodicity/thick subcategory theorem 的基础输入之一。
**可用陈述.** 对 ring spectrum 中元素，$MU$-homology 检测 nilpotence；等价表述需按来源精确翻译。
**禁止用途.** 不得把 nilpotence theorem 改写成“$MU_*x=0$ 当且仅当 $x=0$”。结论是 nilpotent，不是零。
**精确 locator.** 待补。

## 5. Hopkins-Smith periodicity theorem

**编号.** CHT-P0-05
**来源.** Michael J. Hopkins, Jeffrey H. Smith, *Nilpotence and stable homotopy theory II*, Annals of Mathematics 148 (1998), 1-49；Ravenel *Nilpotence and Periodicity in Stable Homotopy Theory*。
**本书用途.** 第四章、附录 I：type $n$ 有限谱存在 $v_n$ self-map；$v_n$ self-map 唯一到幂；telescope Bousfield class 的选择无关。
**可用陈述.** 每个 type $n$ finite spectrum admits a $v_n$-self map，且任意两个在取幂后相容。
**禁止用途.** Periodicity theorem 不推出 telescope conjecture。
**精确 locator.** 待补。

## 6. Hopkins-Smith thick subcategory theorem

**编号.** CHT-P0-06
**来源.** Hopkins-Smith, *Nilpotence and stable homotopy theory II*；Chebolu, *Thick subcategories in stable homotopy theory*, arXiv:math/0607245 作为教学性二级核对。
**本书用途.** 第四章：有限 $p$-局部谱的 thick subcategories 为
$$
\mathcal C_0\supsetneq\mathcal C_1\supsetneq\mathcal C_2\supsetneq\cdots.
$$
**可用陈述.** 每个非零 proper thick subcategory 等于某个 $\mathcal C_n$。
**禁止用途.** 不得用于分类一般 localizing subcategories 或非有限谱子范畴。
**精确 locator.** 待补。

## 7. Morava K finite detection

**编号.** CHT-P0-07
**来源.** Devinatz-Hopkins-Smith/Hopkins-Smith 定理包；Ravenel 书；Hovey-Strickland Morava K localization 结果。
**本书用途.** 第三、四章：非零有限谱被某个 $K(n)$ 检测，从而 type 有定义。
**可用陈述.** 若 $X$ 为有限 $p$-局部谱且 $K(n)_*X=0$ 对所有 $n\ge0$，则 $X\simeq0$。
**禁止用途.** 不得用于任意无限谱。
**精确 locator.** 待补。

## 8. Ravenel-Hopkins chromatic convergence

**编号.** CHT-P0-09
**来源.** Ravenel, *Nilpotence and Periodicity in Stable Homotopy Theory*；Hopkins-Ravenel chromatic convergence theorem。
**本书用途.** 第五章：有限 $p$-局部谱满足
$$
X\simeq\operatorname*{holim}_nL_nX.
$$
**可用陈述.** finite spectra 的 chromatic tower 收敛。
**禁止用途.** 不得用于任意谱、filtered colimit 或未验证完备性的对象。
**精确 locator.** 待补。

## 9. Hovey-Strickland Morava localization package

**编号.** CHT-P0-08, CHT-P0-10, CHT-P1-18
**来源.** Mark Hovey and Neil P. Strickland, *Morava K-theories and localisation*, Memoirs of the AMS 139 (1999)。
**本书用途.** 第三、五章、附录 H：Morava K-theory localization、$K(n)$-local category、$K(n)$-module field-like behavior、Bousfield class comparison 和 local duality 背景。
**可用陈述.** Morava K-theories and localizations 的标准结构定理；具体命题需按章节定位。
**禁止用途.** 不得把所有 $K(n)$-local 结论自动推广到 $T(n)$-local 或 $E(n)$-local。
**精确 locator.** 待补。

## 10. 当前状态

本批次已经把 P0 基础定理包从“纯待办”推进到可追踪的 bibliographic locator。下一批必须补：

1. 每条的 theorem/section/page 精确定位；
2. 文献表述到本书 stable infinity-category notation 的假设翻译；
3. 与 `THEOREM_LEDGER.md` 的行级对应。
