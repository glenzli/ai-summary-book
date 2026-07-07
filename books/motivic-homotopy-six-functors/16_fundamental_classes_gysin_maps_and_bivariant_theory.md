# 第十六章：Fundamental classes、Gysin maps 与 bivariant theory

## 本章目标

本章介绍 motivic fundamental classes 和 bivariant theory。六操作提供 `f^!`，purity 提供 Thom twist，而 fundamental class 把几何态射转化为 cohomology operations。它是 Gysin maps、excess intersection、Riemann-Roch 和 quadratic refinements 的共同机制。

## 依赖前置知识

需要六操作、purity、Thom twists、local complete intersection morphisms、cotangent complex、Borel-Moore homology、orientation、intersection theory 和 stable motivic homotopy。

## 16.1 Fundamental class

**定义 16.1.** 对合适态射 `f:X\to Y`，特别是 local complete intersection morphism，motivic fundamental class 是六操作中连接 Thom twist 与 exceptional pullback 的自然变换，形式上可写作

$$
\eta_f:\Sigma^{L_f}\mathbb 1_X\longrightarrow f^!\mathbb 1_Y,
$$

其中 `L_f` 是 cotangent complex 或其对应的虚向量丛类。不同文献可能采用对偶符号 `T_f=-L_f`。

**外部输入定理 16.2.** 对大类态射存在 motivic fundamental classes，且满足复合、base change、excess intersection、self-intersection 和 blow-up formulas。

**依赖源.** Deglise-Jin-Khan, "Fundamental classes in motivic homotopy theory"；Deglise 的 bivariant theory；Fulton-MacPherson bivariant formalism。

**注 16.3.** 定义 16.1 是口径说明，不是构造。构造需要 deformation to the normal cone、six operations 和 purity 的相干。

## 16.2 Gysin maps

**定义 16.4.** 设 `E` 为 motivic ring spectrum。若 `f:X\to Y` 有 fundamental class，则它诱导 `E`-cohomology 或 Borel-Moore theory 上的 Gysin maps。典型形式由合成

$$
f^*E\longrightarrow \Sigma^{-L_f}f^!E
$$

及伴随得到。

**命题 16.5.** 若 fundamental classes 对复合相容，则 Gysin maps 对复合相容。

**证明.** 设 `X\xrightarrow{f}Y\xrightarrow{g}Z`。Gysin map 的构造由 `\eta_f`、`\eta_g`、六操作复合相干和 Thom twist 的加法性组成。若 fundamental classes 满足

$$
\eta_{gf}=\text{由 }\eta_f\text{ 与 }\eta_g\text{ 合成得到的类},
$$

则用 `gf` 一次构造 Gysin map，与先用 `f` 再用 `g` 构造的映射在每个组成结构态射上相同。由 functoriality 和相干性，两者相等。`\square`

**命题 16.6.** 若 `f` smooth，则 smooth purity 给出的 class 与第六章的 `f^!\simeq\Sigma^{T_f}f^*` 相容。

**证明.** smooth morphism 的 cotangent complex 是向量丛 `L_f`，其对偶为相对切丛 `T_f`。Smooth purity 已给出 `f^!` 与 Thom twist 后的 `f^*` 的等价。Fundamental class 在 smooth 情形应恢复该 purity equivalence；这正是定理 16.2 中复合和 purity normalization 的一部分。`\square`

## 16.3 Bivariant theory

**定义 16.7.** 对 motivic ring spectrum `E` 和态射 `f:X\to S`，定义 bivariant group 的一种六操作表达为

$$
E^{a,b}(X/S)=
\pi_0\operatorname{Map}_{\mathbf{SH}(X)}
(\mathbb 1_X,\Sigma^{a,b}f^!E_S).
$$

该记号把 cohomology、Borel-Moore homology 和 operational classes 放入同一框架。

**命题 16.8.** 若 `f=\operatorname{id}_S`，则 bivariant group 恢复 `E`-cohomology of `S`。

**证明.** 对恒等态射，`f^!=\operatorname{id}`。定义 16.7 变为

$$
\pi_0\operatorname{Map}_{\mathbf{SH}(S)}
(\mathbb 1_S,\Sigma^{a,b}E_S),
$$

即 `E^{a,b}(S)`。`\square`

**外部输入定理 16.9.** Motivic fundamental classes 赋予 `E`-bivariant groups 以乘法、proper pushforward、smooth/lci pullback、base change 和 excess intersection formulas。

## 16.4 Deformation to the normal cone

**定义 16.10.** 对闭嵌入 `i:Z\hookrightarrow X`，deformation to the normal cone 是一个族

$$
D_ZX\to\mathbb A^1
$$

其一般纤维为 `X`，特殊纤维为 normal cone `C_ZX`。若 `i` regular，则 `C_ZX` 为法丛 `N_{Z/X}`。

**外部输入定理 16.11.** Motivic purity 和 fundamental class 可通过 deformation to the normal cone 构造，并与 specialization maps 相容。

**命题 16.12.** 若 `i` 是 regular immersion，则 deformation 的特殊纤维给出 Thom twist 所需的法丛。

**证明.** Regular immersion 局部由 regular sequence 定义，其 normal cone 等于 normal bundle。Deformation to the normal cone 的特殊纤维为 `C_ZX`，因此在 regular 情形为 `N_{Z/X}`。Thom twist 正是由该向量丛产生。`\square`

## 16.5 Excess intersection

**外部输入定理 16.13（Excess intersection formula）.** 对满足假设的 Cartesian 方块和 lci morphisms，Gysin pullback 的交换缺陷由 excess bundle 的 Euler class 控制。

**注 16.14.** Excess formula 是 intersection theory 的核心，不是 base change 的形式后果。Base change 给出 functor exchange；excess formula 比较的是带 fundamental class 的 Gysin operations。

**定义 16.15.** 在 Cartesian 方块中，若预期 codimension 与实际拉回后的 codimension 不同，其差由 excess bundle `\mathcal E` 衡量。Excess formula 中出现的修正项为 Euler class `e(\mathcal E)`。

**命题 16.16.** 若 excess bundle 为零向量丛，则 excess formula 退化为 Gysin maps 与 base change 交换。

**证明.** 零向量丛的 Euler class 为单位。Excess formula 中的修正项乘以单位，不改变映射。因此得到无修正的交换公式。`\square`

## 16.6 Riemann-Roch 型公式

**定义 16.17.** 若 `\varphi:E\to F` 是 oriented motivic ring spectra 之间的 morphism，Riemann-Roch 问题询问 `\varphi` 与 Gysin maps 是否交换；通常需要 Todd class 修正。

**外部输入定理 16.18.** 在 motivic bivariant theory 中，orientation 改变和 ring spectrum morphism 导致的 Gysin 比较满足 Riemann-Roch 型公式。

**命题 16.19.** 若 `\varphi` 严格保持 orientation，则 Todd 修正为单位。

**证明.** Todd class 衡量两个 orientation 对 Thom classes 的差异。若 `\varphi` 把 `E` 的 Thom class 送到 `F` 的 Thom class，则差异类为单位。因此 Riemann-Roch 比较中不出现非平凡修正。`\square`

## 16.7 本章小结

Fundamental classes 把六操作和几何交理论连接起来。它们产生 Gysin maps、bivariant products、excess formulas 和 Riemann-Roch 型比较。所有这些结构都依赖深外部构造；本书内部可证明的是：一旦 fundamental classes 满足相干公理，Gysin functoriality 等结论就是形式后果。

## 练习

**练习 16.1.** 写出 fundamental class 的形式类型。

**练习 16.2.** 解释 `L_f` 与 `T_f` 的符号关系。

**练习 16.3.** 证明恒等态射的 bivariant group 恢复 cohomology。

**练习 16.4.** 说明 excess formula 为什么不是普通 base change。

**练习 16.5.** 解释 Todd class 衡量的是什么差异。

**练习 16.6.** 定义 deformation to the normal cone。

**练习 16.7.** 说明 excess bundle 为零时 excess formula 如何简化。
