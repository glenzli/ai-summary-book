# 附录 C：Hopf algebroid、comodules 与 change of rings

## C.1 Hopf algebroid 口径

**定义 C.1.** 一个 flat Hopf algebroid $(A,\Gamma)$ 包括交换环 $A$、交换 $A$-bialgebroid $\Gamma$、左右单位
$$
\eta_L,\eta_R:A\to\Gamma,
$$
counit、comultiplication 和 antipode 数据，使得它表示某个仿射群胚对象。Flat 表示 $\Gamma$ 作为左 $A$-模和右 $A$-模满足相应 flatness。

**例 C.2.** $(MU_*,MU_*MU)$ 和 $(BP_*,BP_*BP)$ 是 chromatic homotopy theory 的基本 Hopf algebroids。它们编码复定向形式群律及其坐标变化。

**警告 C.3.** Hopf algebroid 不是 Hopf algebra 的普通替换。左右单位 $\eta_L,\eta_R$ 不相同，comodules 的张量、Ext 和 invariant ideals 都必须记录左右结构。

## C.2 Comodules

**定义 C.4.** 左 $\Gamma$-comodule 是 $A$-模 $M$ 配同态
$$
\psi:M\to \Gamma\otimes_{\eta_L,A}M
$$
满足 counit 和 coassociativity 公理。

**定义 C.5.** 对 $(A,\Gamma)$-comodules $M,N$，记
$$
\operatorname{Ext}_{(A,\Gamma)}^{s,t}(M,N)
$$
为 graded comodule abelian category 中的 derived Ext。内部次数 $t$ 来自 $A,\Gamma,M,N$ 的 grading。

**警告 C.6.** $\operatorname{Ext}_{BP_*BP}$ 省略写法应读作
$$
\operatorname{Ext}_{(BP_*,BP_*BP)}.
$$
它不是普通环 $BP_*BP$ 上的模 Ext。

## C.3 Invariant ideals

**定义 C.7.** 理想 $I\subset A$ 称为 invariant，若左右单位诱导的两个理想在 $\Gamma$ 中一致：
$$
\eta_L(I)\Gamma=\eta_R(I)\Gamma.
$$

**命题 C.8.** 若 $I$ invariant，则 $A/I$ 自然成为 $(A,\Gamma)$-comodule quotient 的底环，并可形成 quotient Hopf algebroid
$$
(A/I,\Gamma/I\Gamma).
$$

**证明草图.** invariant 条件保证左右单位都下降到 $A/I$，并且 comultiplication、counit 和 antipode 与 quotient 相容。Flatness 需要单独检查或作为假设保留。证毕。

**外部输入 C.9.** 在 $BP$ convention 下，
$$
I_n=(p,v_1,\ldots,v_{n-1})
$$
是 invariant prime ideals。完整证明依赖 $BP_*BP$ 的结构公式。

## C.4 Change of rings

**定义 C.10.** Hopf algebroid 映射
$$
(A,\Gamma)\to(B,\Sigma)
$$
给出 comodules 的 restriction 和 extension of scalars。若满足合适 flatness、faithfulness 或 groupoid equivalence 条件，可诱导 Ext 群同构。

**外部输入 C.11 (change-of-rings theorem).** $MU$、$BP$ 和 Morava stabilizer group 的多个 change-of-rings 定理把 Hopf algebroid Ext 转换为更小的 Hopf algebroid Ext 或连续群上同调。典型高度 $n$ 形式为
$$
\operatorname{Ext}_{\text{height }n}\cong H_c^*(\mathbb G_n; -).
$$

**使用限制 C.12.** Change-of-rings 不是任意环映射下的代数技巧。每次使用必须说明：

1. Hopf algebroid 映射；
2. comodule 范畴；
3. flatness 或 groupoid equivalence 假设；
4. completion/localization；
5. grading 和拓扑。

## C.5 Cobar complex

**定义 C.13.** 对 Hopf algebroid $(A,\Gamma)$ 和 comodule $M$，normalized cobar complex 的 $s$ 次项形如
$$
\Gamma\otimes_A\overline\Gamma^{\otimes_A s}\otimes_A M
$$
或按左右 comodule convention 的等价版本。其 cohomology 计算 comodule Ext。

**警告 C.14.** Cobar differential 的符号和左右单位位置依赖 convention。正式计算必须固定 Ravenel convention 或另一个明确来源。

## 本附录小结

Adams-Novikov 和 chromatic spectral sequence 的代数核心不是普通环上同调，而是 Hopf algebroid comodule Ext。Invariant ideals、change-of-rings 和 Morava stabilizer cohomology 都必须在这个框架中表达。

## 练习

**练习 C.1.** 写出 Hopf algebra 是 Hopf algebroid 的特殊情形需要哪些条件。

**练习 C.2.** 证明若 $I$ invariant，则 $\eta_L(a)-\eta_R(a)$ 对 $a\in I$ 在 quotient 中为零。

**练习 C.3.** 查阅 $BP_*BP$ 的 Hazewinkel generator 公式，验证 $I_2=(p,v_1)$ 的 invariant 性。
