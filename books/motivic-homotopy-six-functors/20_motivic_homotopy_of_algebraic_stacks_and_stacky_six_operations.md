# 第二十章：Algebraic stacks 上的 motivic homotopy 与六操作

## 本章目标

本章讨论 motivic homotopy theory 从 schemes 扩展到 algebraic stacks 的问题。Stacks 允许 quotient phenomena、moduli spaces 和 stabilizers，但也破坏了许多 scheme-level 证明中的有限性和覆盖性质。因此 stacky motivic homotopy 必须以外部输入定理和明确栈类为基础。

## 依赖前置知识

需要 algebraic stacks、smooth/lisse topology、quotient stacks、six operations、descent、stabilizers、Artin stacks、Deligne-Mumford stacks 和 motivic spectra。

## 20.1 栈上的 motivic homotopy

**定义 20.1.** 一个 stacky motivic homotopy theory 是把某类 algebraic stacks `\mathcal X` 送到 stable presentable category `\mathbf{SH}(\mathcal X)` 的构造，并带有 smooth descent、`\mathbb A^1`-invariance、stabilization 和六操作。

**注 20.2.** “某类 algebraic stacks”是定义的一部分。不同文献使用 scalloped stacks、local quotient stacks、quasi-separated algebraic spaces 或 lisse-extended constructions；这些类别不能混写。

**外部输入定理 20.3.** Stable motivic homotopy category 可扩展到 scalloped algebraic stacks，并支持 Grothendieck 六操作；所得对象表示 stacks 上的 generalized cohomology theories，如 K-theory、motivic cohomology 和 algebraic cobordism。

**依赖源.** Khan-Ravi, "Generalized cohomology theories for algebraic stacks"。

**外部输入定理 20.4.** 还有其他 algebraic stacks 扩展路线，例如利用 enhanced operation map 建立大类 stacks 上的六操作形式主义。

**依赖源.** Chowdhury, "Motivic Homotopy Theory of Algebraic Stacks"。

## 20.2 Lisse-extended 与 genuine stacky theories

**定义 20.5.** Lisse-extended motivic homotopy type 是通过 smooth maps from schemes to a stack 的 diagram 对 scheme-level motivic categories 做 descent/limit/colimit 得到的扩展。

**命题 20.6.** Lisse-extended theory 与 genuine stacky theory 的比较需要额外定理。

**证明.** Lisse extension 是从 smooth atlas 和 lisse site 重建 stack 上对象；genuine stacky theory 可能直接使用 stack 的几何态射、stabilizers 和六操作。二者构造方式不同，functors、compact generation 和 cohomology comparison 不会由定义自动一致。因此需要外部比较定理。`\square`

**外部输入定理 20.7.** 在 Khan-Ravi 的框架中，lisse-extended motivic homotopy type 可恢复若干已有的 stack motives 构造，并在 quotient stacks 上与 equivariant theories 相连。

## 20.3 Stacky 六操作

**定义 20.8.** Stacky 六操作要求对 stacks 间态射 `f:\mathcal X\to\mathcal Y` 给出

$$
f^*, f_*, f_!, f^!, \otimes, \underline{\operatorname{Hom}}
$$

并满足 base change、projection formula、localization、purity 和 descent。

**命题 20.9.** 若 stacky 六操作与 scheme-level 六操作相容，则任意 scheme `X` 视为 stack 时恢复前文六操作。

**证明.** Schemes 全嵌入 algebraic stacks。相容性假设说明对 schemes 间态射，stacky functors 限制为 scheme-level functors。因六操作由这些 functors 和相干同构组成，限制后得到前文的 `\mathbf{SH}(X)` 六操作。`\square`

## 20.4 Fixed point localization

**外部输入定理 20.10.** 对适当 torus actions，stacky motivic homotopy theory 中存在 fixed point localization formula。

**依赖源.** Khan-Ravi 摘要中包含 torus actions 的 fixed point localization formula。

**注 20.11.** Fixed point formula 依赖 equivariant/stacky purity、normal bundle weights 和 localization；不能从普通 open-closed localization 直接推出。

## 20.5 Atlas descent

**定义 20.12.** 设 `p:U\to\mathcal X` 是 algebraic stack 的 smooth atlas。其 Cech nerve 为 simplicial algebraic spaces 或 schemes

$$
U_\bullet,\qquad U_n=U\times_{\mathcal X}\cdots\times_{\mathcal X}U.
$$

**命题 20.13.** 若 stacky motivic theory 满足 smooth descent，则 `\mathbf{SH}(\mathcal X)` 可由 `\mathbf{SH}(U_\bullet)` 的 descent data 重建。

**证明.** Smooth descent 的含义是 `\mathcal X` 上的对象等价于 atlas Cech nerve 上带相干 descent datum 的对象。范畴值 sheaf 条件把 `\mathbf{SH}(\mathcal X)` 识别为 cosimplicial diagram `\mathbf{SH}(U_\bullet)` 的 limit。`\square`

**注 20.14.** 命题 20.13 是形式描述；实际证明需要 stacky motivic theory 是 lisse/smooth topology 上的 sheaf of categories。这是外部输入的一部分。

## 20.6 例子：classifying stack

**定义 20.15.** 对代数群 `G`，classifying stack `BG` 定义为 `[S/G]`，其中 `G` 平凡作用在 `S` 上。

**命题 20.16.** `\mathbf{SH}(BG)` 应理解为 `G`-equivariant motivic spectra over `S`。

**证明.** 按 quotient stack 口径，`[S/G]` 上的 motivic homotopy theory就是平凡 `G`-对象 `S` 的 equivariant motivic homotopy theory。对象记录的不只是 `S` 上谱，还记录 `G`-equivariant descent/stabilizer action。`\square`

**命题 20.17.** 若忽略 `BG` 的 stabilizer，只把其粗空间看作 `S`，会丢失 equivariant 信息。

**证明.** `BG` 的几何点自同构群为 `G`。粗空间 `S` 不记录该自同构群。Equivariant vector bundles、representation spheres 和 fixed-point operations 都依赖 stabilizer；因此粗空间不能恢复 `BG` 上的 motivic theory。`\square`

## 20.7 Scalloped stacks and local quotient conditions

**定义 20.18.** 一个 algebraic stack 称为具有 local quotient 性质，若它可由形如 `[U/G]` 的 quotient stacks 在合适拓扑下局部覆盖，其中 `U` 是 algebraic space 或 scheme，`G` 是满足有限性或可约性假设的群。

**定义 20.19.** Scalloped stack 是 Khan-Ravi 框架中的一类适于做 motivic homotopy 和六操作的 algebraic stacks。其精确定义包含对分层、局部商表示和稳定性性质的条件，本书把它作为外部输入定义使用。

**命题 20.20.** Local quotient 条件使 stacky motivic theory 可由 equivariant motivic theory 局部建模。

**证明.** 若 stack 在局部形如 `[U/G]`，则该局部片段上的 motivic theory 可用第十九章的 quotient stack/equivariant theory 表达。若 theory 满足 descent，则全局对象可由这些局部 equivariant pieces 及其交叠上的 descent data 粘合。`\square`

**注 20.21.** 这说明第十九章和第二十章不是两个孤立方向：equivariant theory 是 stacky theory 的局部模型之一。

## 20.8 Stacky purity

**外部输入定理 20.22.** 对适当 representable smooth morphisms 或 lci morphisms of stacks，stacky motivic homotopy theory 满足 purity 和 Gysin formalism。

**命题 20.23.** Stacky purity 需要 stabilizer 对 normal directions 的作用。

**证明.** 在 quotient stack `[X/G]` 中，闭子栈 `[Z/G]` 的 normal bundle 是 `G`-equivariant normal bundle `N_{Z/X}`。Thom twist 必须记住 `G` 对纤维的线性作用；否则会退化为粗空间上的普通 normal bundle，丢失 stabilizer 表示信息。`\square`

## 20.9 本章小结

Stacks 上的 motivic homotopy 是当前活跃方向之一。它把 quotient stacks、moduli stacks 和 equivariant phenomena 纳入六操作框架。由于 stack 类别和 lisse/genuine 构造差异很大，本书只在明确外部输入和假设下使用 stacky 结果。

## 练习

**练习 20.1.** 解释为什么必须声明允许的 algebraic stacks 类别。

**练习 20.2.** 定义 lisse-extended motivic homotopy type 的思想。

**练习 20.3.** 说明 stacky 六操作限制到 schemes 时应满足什么相容性。

**练习 20.4.** 为什么 quotient stack 是 equivariant motivic homotopy 的自然语言？

**练习 20.5.** 解释 fixed point localization 需要哪些额外几何输入。

**练习 20.6.** 写出 smooth atlas 的 Cech nerve。

**练习 20.7.** 解释为什么 `BG` 不是由粗空间 `S` 决定的。

**练习 20.8.** 说明 local quotient stack 如何由 equivariant theory 局部建模。

**练习 20.9.** 为什么 stacky purity 需要 stabilizer 表示？
