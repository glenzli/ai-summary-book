# 第十七章：Norm functors、normed spectra 与 multiplicative transfers

## 本章目标

本章介绍 motivic norm functors。Norms 是 finite etale 或 finite locally free morphisms 上的乘法转移，是 motivic `E_\infty`-ring spectra 的增强结构。它们与 additive transfers、framed transfers 和 Galois theory 相互作用，但不能混同。

## 依赖前置知识

需要 finite etale morphisms、symmetric monoidal functors、commutative algebra objects、six operations、finite correspondences、framed transfers、Galois theory、`H\mathbb Z`、`KGL` 和 `MGL`。

## 17.1 Norm functor

**外部输入定理 17.1.** 若 `f:S'\to S` 为 finite locally free morphism，则在 pointed unstable motivic homotopy 中存在 symmetric monoidal norm functor

$$
f_\otimes:\mathbf H_*(S')\to\mathbf H_*(S).
$$

若 `f` finite etale，则该 functor 稳定化为

$$
f_\otimes:\mathbf{SH}(S')\to\mathbf{SH}(S).
$$

**依赖源.** Bachmann-Hoyois, "Norms in motivic homotopy theory"。

**定义 17.2.** 对 finite etale `f`，`f_\otimes` 称为 motivic norm functor。

**命题 17.3.** Norm functor 与 additive pushforward `f_*` 一般不同。

**证明.** `f_*` 是 `f^*` 的右伴随，属于加性稳定范畴的六操作；`f_\otimes` 是 symmetric monoidal transfer，控制乘法结构。右伴随结构不包含 symmetric monoidal norm 的分配律；反之，norm 的乘法性也不等同于 additive mapping-space adjunction。因此二者不能由定义识别。`\square`

## 17.2 Normed spectra

**定义 17.4.** 一个 normed motivic spectrum 是 motivic commutative ring spectrum `E`，连同对 finite etale morphisms 的 norm maps，使得 `E` 对 `f_\otimes` 的作用满足单位、复合、base change 和 distributivity 相干。

**外部输入定理 17.5.** `H\mathbb Z`、`KGL` 和 `MGL` 在适当假设下具有 normed motivic spectrum structure。

**依赖源.** Bachmann-Hoyois；`H\mathbb Z` 的 normed structure 细化 Fulton-MacPherson multiplicative transfers 和 Voevodsky power operations。

**命题 17.6.** Normed spectrum 的底层对象是 commutative motivic ring spectrum。

**证明.** Normed spectrum 的定义包含对 identity morphism 和 finite etale self-products 的相干。特别地，在单位基上的平凡 finite etale operations 给出乘法、单位和对称幺半相干；这些正是 commutative algebra object 的结构。`\square`

## 17.3 与 Galois theory 的关系

**外部输入定理 17.7.** Norm functors 与 Grothendieck Galois theory 相容；对 finite etale covers，它们 categorify classical multiplicative transfers。

**命题 17.8.** 若 `L/k` 是 finite separable field extension，则 normed spectrum `E` 给出从 `E(L)` 到 `E(k)` 的 multiplicative norm map。

**证明.** `\operatorname{Spec}L\to\operatorname{Spec}k` 是 finite etale。对 normed spectrum `E`，定义 17.4 给出沿该 morphism 的 norm operation。把它应用到 `E` 在 `L` 上的 cohomology 或 global sections，得到到 `k` 上的乘法转移。`\square`

## 17.4 Norms 与 transfers 的兼容

**外部输入定理 17.9.** Motivic norms 与多种 transfer 结构兼容；特别存在对 framed transfers/infinite loop recognition 的 norm monoidal refinement。

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

**例子 17.14.** 若 `L/k` 是有限可分扩张，`E=H\mathbb Z` 的 norm 在 `H^{0,0}` 上应恢复整数乘法意义下的 degree 型 norm；在 `KGL` 上则与代数 K-theory 的 norm/transfer 结构比较。精确比较依赖 normed spectrum 结构和 classical transfer 的外部定理。

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

Norms 是 motivic homotopy 中的乘法转移。它们把 finite etale 几何、Galois theory、power operations 和 normed spectra 连接起来。`H\mathbb Z`、`KGL`、`MGL` 的 normed structures 是现代 motivic homotopy 的核心结果之一；与 additive/framed transfers 的兼容需要单独定理。

## 练习

**练习 17.1.** 定义 finite etale norm functor。

**练习 17.2.** 比较 `f_*` 与 `f_\otimes`。

**练习 17.3.** 说明 normed spectrum 如何增强 commutative ring spectrum。

**练习 17.4.** 对 finite separable field extension 写出 norm operation 的几何来源。

**练习 17.5.** 解释 distributivity 相干为什么必要。

**练习 17.6.** 写出 finite etale 复合中 norm 的相干公式。

**练习 17.7.** 说明为什么 commutative ring spectrum 不自动是 normed spectrum。
