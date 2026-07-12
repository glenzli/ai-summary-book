# 第十七章：Norm functors、normed spectra 与 multiplicative transfers

## 本章目标

本章介绍 motivic norm functors。Norms 是 finite etale 或 finite locally free morphisms 上的乘法转移，是 motivic `E_\infty`-ring spectra 的增强结构。它们与 additive transfers、framed transfers 和 Galois theory 相互作用，但不能混同。

## 依赖前置知识

需要 finite etale morphisms、symmetric monoidal functors、commutative algebra objects、six operations、finite correspondences、framed transfers、Galois theory、`H\mathbb Z`、`KGL` 和 `MGL`。

## 17.1 Norm functor

**外部输入定理 17.1（NM-17.x）.** 若 `f:S'\to S` 为 finite locally free
morphism，则 Bachmann--Hoyois 在 pointed unstable motivic homotopy 中构造
symmetric monoidal norm functor

$$
f_\otimes:\mathbf H_*(S')\to\mathbf H_*(S).
$$

若 `f` finite etale，则该 functor 稳定化为

$$
f_\otimes:\mathbf{SH}(S')\to\mathbf{SH}(S).
$$

**精确来源与边界.** Bachmann--Hoyois, *Norms in motivic homotopy theory*,
Section 3 and Proposition 3.13（unstable finite-locally-free construction and
quotient compatibility），Proposition 4.5（finite-etale stable extension），
`https://arxiv.org/abs/1711.03061`。Proposition 4.5 的稳定结论只对 finite
etale morphisms；finite locally free 的 unstable norm 不能自动稳定化。

**定义 17.2.** 对 finite etale `f`，`f_\otimes` 称为 motivic norm functor。

**命题 17.3.** Norm functor 与 additive pushforward `f_*` 一般不同。

**证明.** 取 split degree-`d` cover
`p:\coprod_{r=1}^dS\to S`。有
`\mathbf{SH}(\coprod S)\simeq\mathbf{SH}(S)^d`。右伴随 `p_*` 把
`(E_1,\ldots,E_d)` 送到有限乘积 `\prod_rE_r`；在稳定范畴中这也等于有限
直和。Norm 则由对称幺半结构送到 `\bigotimes_rE_r`。当 `d=2`、
`E_1=0`、`E_2=\mathbb 1` 时，前者为 `\mathbb 1`，后者为 `0`，故两个
functors 不同。一般 finite etale 情形由局部 descent 粘合这些不同的加法和
乘法模型，因此也不能把 `f_*` 与 `f_\otimes` 按定义识别。`\square`

## 17.2 Normed spectra

**定义 17.4（精确口径）.** 按 Bachmann--Hoyois Definition 7.1，一个 normed
motivic spectrum 是 `\mathbf{SH}^{\otimes}` 在
`\operatorname{Span}(\mathcal C,\mathrm{all},\mathrm{f\acute et})` 上的 section，
并在 `\mathcal C^{op}` 方向 cocartesian。展开后，它给出各基上的 motivic
commutative ring spectrum、每个 finite etale morphism 的 norm，以及单位、
复合、base change 和 distributivity 的全部 coherence。只列逐个 norm maps
不足以构成 normed spectrum。

**外部输入定理 17.5（NM-17.x）.** 下列版本具有 Definition 17.4 的完整
normed structure：若 `S` Noetherian，则 `H\mathbb Z_S` 在论文指定的
`\operatorname{Sm}_S` 口径上为 normed spectrum；对任意 scheme `S`，
`KGL_S`、`MGL_S` 及 `MGL` 的 periodization 为 normed spectra。

**精确来源.** Bachmann--Hoyois, Theorem 14.5（`H\mathbb Z`）、Theorem
15.22（`KGL`）与 Theorem 16.19（`MGL`），同上 URL；域上 Chow norm 的
经典比较见 Theorem 14.14。三个例子的基假设不能压缩成同一个“适当假设”。

**命题 17.6.** Normed spectrum 的底层对象是 commutative motivic ring spectrum。

**证明.** 固定基 `S`，把 Definition 17.4 的 span section 限制到由有限个
`S` 的不交并及其 fold maps 组成的 finite-etale 子范畴。对
`\coprod_{r=1}^dS\to S` 的 norm 在 `\mathbf{SH}(S)` 中就是 `d` 重张量，
于是 section 的结构态射给出

$$
E_S^{\otimes d}\longrightarrow E_S\qquad(d\geq0).
$$

有限集合的恒等、复合和置换分别给出单位律、结合律和对称性；Definition
7.1 的 coherence 保证这些等式在 infinity-category 中相干。因而
`E_S` 是 `\operatorname{CAlg}(\mathbf{SH}(S))` 的对象。`\square`

## 17.3 与 Galois theory 的关系

**高级外部输入 17.7（P1）.** Norm functors 与 finite-etale/Galois
functoriality 及 classical multiplicative transfers 的逐理论比较，需要为所选
cohomology theory 单独验证。本章 P0 只使用 Proposition 4.5 与 Definition
7.1 的范畴性 norm package。

**命题 17.8.** 若 `L/k` 是 finite separable field extension，则 normed spectrum `E` 给出从 `E(L)` 到 `E(k)` 的 multiplicative norm map。

**证明.** 令 `p:\operatorname{Spec}L\to\operatorname{Spec}k`。可分有限扩张
使 `p` finite etale。一个 global section 是态射
`x:\mathbb 1_L\to E_L`。先应用 symmetric monoidal functor `p_\otimes`，
再接 normed section 的结构态射，得到

$$
\mathbb 1_k\simeq p_\otimes\mathbb 1_L
\xrightarrow{p_\otimes x}p_\otimes E_L
\longrightarrow E_k.
$$

这定义 `E(L)\to E(k)`；symmetric monoidality 保持乘法，复合相干由
Definition 17.4 给出。因此所得 operation 是 contravariant restriction
`p^*` 之外的、方向从 `L` 到 `k` 的 multiplicative norm。`\square`

## 17.4 Norms 与 transfers 的兼容

**高级外部输入 17.9（P1）.** Motivic norms 与 framed/additive transfer
结构的 norm-monoidal 或 Tambara 型 refinement 是额外比较定理，不由 normed
spectrum 定义自动产生。

**依赖源.** Brian Shin, "Norms and Transfers in Motivic Homotopy Theory"。

**命题 17.10.** 若一个理论同时有 additive transfers 和 norms，则需要验证 distributivity，而不能只验证二者分别存在。

**证明.** Additive transfer 控制加法结构，norm 控制乘法结构。一个 semiring-like 或 Tambara-like 结构要求乘法 norm 对加法 transfer 满足分配关系。分别存在两个操作只说明两个方向上有函子性；不保证混合表达式如 `N(a+b)` 与 transfers/products 的关系。因此 distributivity 是额外相干条件。`\square`

## 17.5 Tambara 型相干

**定义 17.11.** 若一个 cohomology theory 对 finite etale morphisms 同时有 additive transfer `tr_f`、restriction `f^*` 和 multiplicative norm `N_f`，并满足 base change、复合和分配律，则称它具有 Tambara 型结构。

**命题 17.12.** 对 finite etale 复合

$$
T\xrightarrow{g}S'\xrightarrow{f}S,
$$

normed structure 要求有相干等价

$$
N_{fg}\simeq N_fN_g.
$$

**证明.** Normed spectrum 的定义包含对 finite etale category 的 symmetric monoidal functoriality。复合态射在源范畴中等于 `fg`，functoriality 给出沿 `fg` 的 norm 与先沿 `g` 再沿 `f` 的 norm 的相干等价。`\square`

**命题 17.13.** 对 Cartesian 方块

$$
\begin{array}{c}
T'\longrightarrow T\\
\downarrow g'\qquad\downarrow g\\
S'\longrightarrow S
\end{array}
$$

其中 `g` finite etale，normed structure 要求 norm 与 base change 相容。

**证明.** Finite etale morphisms 对 base change 稳定，因此 `g'` 仍 finite etale。Normed functoriality 是在带 pullback 的几何范畴上定义的，故对方块给出 Beck-Chevalley 型相干：先拉回再 norm，与先 norm 再拉回在相应目标中比较为等价。该相干是定义的一部分，不由普通六操作自动推出。`\square`

**例子 17.14.** 对 split degree-`d` finite etale cover，constant integer
section `n\in H^{0,0}` 的 multiplicative norm 是 `n^d`，不是 additive
transfer 的 `dn`。对域上的 Chow groups，Bachmann--Hoyois Theorem 14.14
把 `H\mathbb Z`-norm 与 Fulton--MacPherson norm 比较；`KGL` 与 classical
algebraic K-theory norm 的进一步识别属于 P1。这个例子具体显示 norm 与
additive pushforward 的 variance 和代数性质不同。

## 17.6 失败模式

**命题 17.15.** 不能把 finite locally free morphism 上的 unstable norm 自动稳定化到 `\mathbf{SH}`。

**证明.** 定理 17.1 中，unstable norm 对 finite locally free morphisms 构造；稳定化到 `\mathbf{SH}` 需要 norm 与被反演的 Tate sphere/`\mathbb P^1` 稳定坐标相容。Bachmann-Hoyois 的结果在 finite etale 情形保证这种稳定化。没有该相容性时，unstable functor 不必通过稳定化泛性质因子化。`\square`

**命题 17.16.** Normed structure 不能由 commutative algebra object 结构恢复。

**证明.** Commutative algebra object 只给出每个基 `S` 上 `E_S` 的乘法

$$
E_S\otimes E_S\to E_S.
$$

Normed structure 还要求对每个 finite etale `f:S'\to S` 有跨基的 multiplicative transfer `f_\otimes` 作用和所有复合/base-change/distributivity 相干。这些数据不包含在单个 fiber `E_S` 的交换代数结构中。`\square`

## 17.7 本章小结

Norms 是 finite etale stable motivic homotopy 中的乘法转移；finite locally
free 的一般构造在本章只位于 pointed unstable 层。Normed spectrum 是 span
category 上带全部 coherence 的 section，不只是 commutative ring spectrum
加一族 maps。`H\mathbb Z`、`KGL`、`MGL` 的精确基假设分别由 Theorems
14.5、15.22、16.19 控制；framed/Tambara 兼容留在 P1。

## 练习

**练习 17.1.** 定义 finite etale norm functor。

**练习 17.2.** 比较 `f_*` 与 `f_\otimes`。

**练习 17.3.** 说明 normed spectrum 如何增强 commutative ring spectrum。

**练习 17.4.** 对 finite separable field extension 写出 norm operation 的几何来源。

**练习 17.5.** 解释 distributivity 相干为什么必要。

**练习 17.6.** 写出 finite etale 复合中 norm 的相干公式。

**练习 17.7.** 说明为什么 commutative ring spectrum 不自动是 normed spectrum。
