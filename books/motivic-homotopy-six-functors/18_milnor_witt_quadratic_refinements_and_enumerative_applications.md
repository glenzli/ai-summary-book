# 第十八章：Milnor-Witt refinements、quadratic refinements 与 enumerative applications

## 本章目标

本章介绍 motivic homotopy theory 的二次型方向。Morel 的基本结果把 sphere spectrum 的某些 endomorphisms 与 Grothendieck-Witt theory 联系起来，由此产生 Milnor-Witt K-theory、Chow-Witt groups、quadratic Euler characteristics 和 quadratic enumerative geometry。

## 依赖前置知识

需要 Grothendieck-Witt rings、Milnor K-theory、Milnor-Witt K-theory、stable motivic homotopy groups、Euler characteristics、fundamental classes、orientation、field extensions 和 trace forms。

## 18.1 Grothendieck-Witt 值的动机

**定义 18.1.** 对域 `k`，Grothendieck-Witt ring `GW(k)` 是非退化对称双线性型的 Grothendieck ring，乘法由张量积给出。

**外部输入定理 18.2（MW-18.2，Morel）.** 对 perfect field `k`，有
canonical ring isomorphism

$$
\operatorname{End}_{\mathbf{SH}(k)}(\mathbb 1_k)\cong GW(k).
$$

等价地，sphere spectrum 的 `\pi_{0,0}` 为 `GW(k)`。

**精确来源与推导.** Fabien Morel, *A1-Algebraic Topology over a Field*,
Corollary 6.43 计算足够悬挂后的 motivic sphere self-maps 为
`K_0^{MW}(k)`；沿 `S^1\wedge\mathbb G_m` 继续悬挂的 transition maps 与该
识别相容，所以稳定化 colimit 仍为 `K_0^{MW}(k)`。Lemma 3.10 在 `n=0`
时给出 ring isomorphism `K_0^{MW}(k)\cong GW(k)`。参见
`https://doi.org/10.1007/978-3-642-29514-0` 和可检索 PDF
`https://www.mahalex.net/teaching/seminars/morel/Morel.pdf`。这里使用的是
Morel 全书的 perfect-field 约定。

**命题 18.3.** 若定理 18.2 适用，则 smooth proper `k`-scheme 的 motivic Euler characteristic 可视为 `GW(k)` 中的元素。

**证明.** 第七章定义 `\chi_k(X)` 为 `\operatorname{End}_{\mathbf{SH}(k)}(\mathbb 1_k)` 中的元素。定理 18.2 把该 endomorphism ring 识别为 `GW(k)`，故 `\chi_k(X)` 得到 Grothendieck-Witt 值。`\square`

## 18.2 Milnor-Witt K-theory

**定义 18.4.** Milnor-Witt K-theory `K_*^{MW}(k)` 是同时记录 Milnor K-theory symbols 和二次型信息的分次环。它包含符号 `[a]` 和与 Hopf map 相关的元素，并满足 Morel 关系。

**外部输入定理 18.5（MW-18.5）.** 在定理 18.2 的 perfect-field 假设下，
若 `n\geq2`、`i>0`，则 Morel Corollary 6.43 给出

$$
\operatorname{Hom}_{\mathbf H_\bullet(k)}
\bigl(S^n\wedge\mathbb G_m^{\wedge j},
      S^n\wedge\mathbb G_m^{\wedge i}\bigr)
\cong K_{i-j}^{MW}(k).
$$

该等式是 pointed unstable motivic homotopy category 中进入稳定区间后的
映射群计算；定理 18.2 再取对角 `i=j` 并稳定化。它不能改写成所有双次数
stable stems 都等于 Milnor--Witt K-theory。

**注 18.6.** Milnor-Witt theory 是 motivic homotopy 区别于纯 topological stable homotopy 的重要现象：它保留了域上二次型和符号信息。

## 18.3 Rank、signature 与 fundamental ideal

**定义 18.7.** Rank map 是环同态

$$
\operatorname{rk}:GW(k)\to\mathbb Z
$$

把非退化双线性型送到其底层向量空间维数。

**定义 18.8.** 若 `k` 是有序域，特别是 `k=\mathbb R`，signature map

$$
\operatorname{sgn}:GW(k)\to\mathbb Z
$$

把二次型送到正负惯性指数之差。

**定义 18.9.** Fundamental ideal `I(k)\subset GW(k)` 是 rank 为偶数并在 Witt ring 中落入基本理想的二次型类所生成的理想；等价表述取决于 `GW` 与 `W(k)` 的精确模型。

**命题 18.10.** Rank 不能检测 `GW(k)` 的全部信息。

**证明.** 在 `\mathbb R` 上，`\langle1\rangle` 与 `\langle-1\rangle` 的 rank 都为 `1`，但 signature 分别为 `1` 与 `-1`。因此二者在 `GW(\mathbb R)` 中不同，而 rank 无法区分。`\square`

**命题 18.11.** 若 motivic Euler characteristic `\chi(X)\in GW(k)` 已定义，则 `\operatorname{rk}\chi(X)` 是其普通 Euler characteristic 的候选影子，而 signature 给出实拓扑或二次型修正信息。

**证明.** Rank map 忘记二次型结构，只保留维数，因此把 Grothendieck-Witt 值 invariant 降到整数值。Signature 在有序域上进一步区分正负方向；若比较定理适用，它与实 realization 的 Euler characteristic 或 trace formula 相连。具体相容性是外部输入，但两个 map 的信息含义由定义给出。`\square`

## 18.4 Chow-Witt 与 Milnor-Witt motives

**高级外部输入 18.12（MW-18.7，P1）.** 在指定 field、regularity、
orientation line bundle 与 coefficient conventions 后，可构造 Chow--Witt
groups、Milnor--Witt correspondences 和相应 motives/spectra。该合集不参与
定理 18.2--18.5 的 P0 sphere/Grothendieck--Witt 主线。

**来源边界.** Barge--Morel、Fasel、Calmes--Fasel、Deglise--Fasel 与
Deglise--Jin--Khan；不同模型的 twists 和 transfers 不同，实际调用时须为
所选模型另补 locator，不能用此 P1 汇总项支撑主线证明。

**命题 18.13.** 忘记二次型信息应给出从 Milnor-Witt refined theory 到 ordinary motivic cohomology/Chow theory 的映射。

**证明.** Grothendieck-Witt 或 Milnor-Witt 结构含有底层 rank/degree 或 ordinary cycle 信息。忘记双线性型只保留整数重数，因而诱导从 refined correspondences 或 refined cycle groups 到 ordinary correspondences/cycle groups 的函子或群同态。具体构造依赖所用模型，但逻辑上是忘却额外二次型装饰。`\square`

## 18.5 Quadratic enumerative geometry

**定义 18.14.** Quadratic enumerative invariant 是取值于 `GW(k)` 或相关二次型群的枚举不变量，其 rank 恢复经典计数，而完整二次型记录域和局部交数的信息。

**高级外部输入 18.15（P1）.** 在逐问题声明的 smooth/proper、孤立零点、
横截性、可分剩余域和相对定向假设下，motivic degree、Euler class 与
fundamental classes 可产生 `GW(k)`-值枚举精化。这里不存在一个覆盖所有
枚举问题的无条件定理。

**依赖源.** Kass-Wickelgren、Levine、Hoyois、Deglise-Jin-Khan 及后续 quadratic enumerative geometry 文献。

**命题 18.16.** 若 quadratic count `N\in GW(k)` 的 rank 为 `n`，则 `n` 是其底层经典计数的候选值；但 `N` 含有比 `n` 更多的信息。

**证明.** rank map `GW(k)\to\mathbb Z` 忘记双线性型而记录向量空间维数。若 invariant 构造与经典计数相容，则 rank 给出普通整数计数。不同二次型可以有同一 rank，例如各向异性部分或判别式不同，因此 `GW(k)` 元素保留额外算术信息。`\square`

## 18.6 Motivic local degree

**定义 18.17.** 设 `f:\mathbb A^n_k\to\mathbb A^n_k` 在孤立零点 `p` 处满足适当非退化条件。Motivic local degree 是 `GW(k(p))` 中的局部二次型，再沿 field trace 推到 `GW(k)`。

**高级外部输入 18.18（P1，simple-zero 版本）.** 设
`\operatorname{char}(k)\ne2`，`p` 为可分闭点，`f:\mathbb A_k^n\to
\mathbb A_k^n` 在 `p` 为 simple zero，即 `\det J_f(p)\ne0`；并采用标准
坐标给出的相对定向。则 motivic local degree 由 Jacobian determinant 的
一维二次型表示：

$$
\deg_p^{\mathbb A^1}(f)=\operatorname{Tr}_{k(p)/k}\langle \det J_f(p)\rangle.
$$

**来源边界.** Morel 的 `A1`-degree、Kass--Wickelgren 的 local-degree
比较及后续文献。非 simple zero 需要 Eisenbud--Khimshiashvili--Levine 型
局部代数双线性型，不能套用这一行公式；本条不参与 P0 闭合。

**命题 18.19.** 对代数闭域 `k`，rank of motivic local degree 恢复普通局部重数的非退化情形。

**证明.** 代数闭域上每个一维非退化二次型同构于 `\langle1\rangle`，其 rank 为 `1`。非退化孤立零点的普通局部重数为 `1`。若有有限多个非退化零点，Grothendieck-Witt 值求和的 rank 就是零点个数。`\square`

**命题 18.20.** 对非代数闭域，两个 rank 相同的 quadratic counts 可以不同。

**证明.** `GW(k)` 的 rank map 忘记判别式、Hasse invariant 和各向异性部分等二次型信息。例如在实数域上，`\langle1\rangle` 与 `\langle-1\rangle` rank 都为 `1`，但 signature 不同。因此 rank 相同不推出 `GW(k)` 元素相同。`\square`

## 18.7 Euler classes and orientation

**定义 18.21.** 若向量丛 `V\to X` 带相对定向，quadratic Euler class 是 Chow-Witt 或 Milnor-Witt cohomology 中的 Euler class，其 ordinary image 是通常 Euler class。

**命题 18.22.** 若 quadratic Euler class 在忘却二次型信息后为零，则不能推出 quadratic Euler class 本身为零。

**证明.** 忘却 map 从 refined group 到 ordinary Chow 或 motivic cohomology 可能有非零 kernel。Kernel 中的类正是二次型修正信息。故 ordinary image 为零只说明整数或 cycle-level 影子消失，不说明 refined class 消失。`\square`

## 18.8 与 fundamental classes 的关系

**命题 18.23.** Quadratic refinements 需要 orientation 或相对定向数据。

**证明.** 局部交数若要提升为二次型，需要把 determinant lines、法丛或 cotangent complex 的相关方向数据识别到可取 trace form 的标准对象上。没有 orientation，局部 contribution 只在扭曲系数中定义，不能自然落入统一的 `GW(k)`。因此 orientation 或相对定向是 quadratic refinement 的必要输入。`\square`

**高级外部输入 18.24（P1）.** 在选定 Milnor--Witt coefficient spectrum、
twists、properness、duality 和 orientation 假设后，fundamental classes 可与
Chow--Witt Euler classes 及 Gauss--Bonnet 型公式比较。该比较不是定理 16.2
的形式后果，也不作为本章 P0 输入。

## 18.9 本章小结

本章 P0 主线只有 Morel Corollary 6.43 与 Lemma 3.10：它们精确给出
`\operatorname{End}(\mathbb 1_k)\cong GW(k)`，从而使 smooth proper Euler
characteristic 取 Grothendieck--Witt 值。Chow--Witt motives、一般枚举计数、
local degrees 与 Gauss--Bonnet refinements 均为 P1；它们不能反向作为该
sphere-endomorphism 识别的依据。

## 练习

**练习 18.1.** 定义 `GW(k)` 并写出 rank map。

**练习 18.2.** 解释 Morel 定理如何把 Euler characteristic 变成二次型值。

**练习 18.3.** 比较 Milnor K-theory 和 Milnor-Witt K-theory 的信息量。

**练习 18.4.** 说明 quadratic count 的 rank 为什么只是部分信息。

**练习 18.5.** 解释 orientation 在 quadratic refinement 中的作用。

**练习 18.6.** 在 `k=\mathbb R` 上比较 `\langle1\rangle` 和 `\langle-1\rangle` 的 rank 与 signature。

**练习 18.7.** 写出非退化零点的 motivic local degree 公式。

**练习 18.8.** 说明 ordinary Euler class 为零为何不推出 quadratic Euler class 为零。
