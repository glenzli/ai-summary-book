# 第十三章：Slice filtration、effective categories 与 cellular methods

## 本章目标

本章介绍 Voevodsky slice filtration。它是 stable motivic homotopy category 中按 Tate twist 方向过滤谱的基本工具，并与 motivic cohomology、`H\mathbb Z`-modules、Adams 型谱序列和计算方法连接。本章给出形式定义和基本性质，把 zero slice 和 slices as motivic modules 等深结果列为外部输入。

## 依赖前置知识

需要 compact generation、localizing subcategories、adjoint functor theorem、Tate twists、stable infinity-categories、`H\mathbb Z`、spectral sequences 和 cellular subcategories。

## 13.1 Effective subcategories

**定义 13.1.** `\mathbf{SH}(S)^{eff}` 是由所有

$$
\Sigma_T^\infty X_+,\qquad X\in\operatorname{Sm}_S
$$

在小余极限、cofibers 和 retracts 下生成的 localizing subcategory。

**定义 13.2.** 对整数 `q`，定义

$$
\mathbf{SH}(S)^{eff}(q)=\Sigma^{2q,q}\mathbf{SH}(S)^{eff}.
$$

对象称为 `q`-effective。

**命题 13.3.** 若 inclusion

$$
i_q:\mathbf{SH}(S)^{eff}(q)\hookrightarrow\mathbf{SH}(S)
$$

保持小余极限且源范畴 presentable，则 `i_q` 有右伴随 `r_q`。

**证明.** `i_q` 是 presentable infinity-categories 之间保持小余极限的函子。由 presentable adjoint functor theorem，它有右伴随。`\square`

**定义 13.4.** 在命题 13.3 的假设下，定义 effective cover

$$
f_q(E)=i_qr_q(E).
$$

自然 counit 给出 `f_q(E)\to E`。

## 13.2 Slice tower

**定义 13.5.** 第 `q` 个 slice 定义为 cofiber

$$
s_q(E)=\operatorname{cofib}(f_{q+1}(E)\to f_q(E)).
$$

这些对象组成 tower

$$
\cdots\to f_{q+1}(E)\to f_q(E)\to f_{q-1}(E)\to\cdots
$$

称为 slice tower。

**命题 13.6.** `s_q(E)` 同时是 `q`-effective 的 quotient，并且被 `q+1`-effective 部分截去。

**证明.** 由定义，`f_q(E)` 属于 `\mathbf{SH}(S)^{eff}(q)`，`f_{q+1}(E)` 属于更高 effective 子范畴。`s_q(E)` 是二者之间映射的 cofiber，因此记录 `q` 层相对于 `q+1` 层的差。该表述是 localizing filtration 的形式后果；若要给出正交刻画，需要右伴随和局部化三角的额外结构。`\square`

**定义 13.7.** 若 slice tower 对 `E` 收敛，则其 slices 给出计算 `E`-cohomology 或 homotopy sheaves 的 spectral sequence。收敛性必须逐对象证明。

## 13.3 Zero slice 与 motivic cohomology

**外部输入定理 13.8（Voevodsky zero slice）.** 在特定基和特征假设下，sphere spectrum 的零 slice 满足

$$
s_0(\mathbb 1)\simeq H\mathbb Z.
$$

更一般版本需要额外基和系数限制。

**外部输入定理 13.9.** 在相应假设下，任意谱的 slices 具有 `H\mathbb Z`-module 结构。

**依赖源.** Voevodsky "On the zero slice of the sphere spectrum"、后续推广和 Spitzweck/Hoyois/Cisinski-Deglise 相关结果。

**命题 13.10.** 若定理 13.8 和 13.9 适用，则 slice filtration 把 sphere spectrum 的第一层近似与 motivic cohomology 联系起来。

**证明.** 定理 13.8 直接识别 `s_0(\mathbb 1)` 为 `H\mathbb Z`。定理 13.9 说明 slices 落在 `H\mathbb Z`-linear 世界。因此 sphere spectrum 的 slice tower 的零层就是 motivic cohomology，而高层 slices 是 `H\mathbb Z`-module 型对象。`\square`

## 13.4 Cellular subcategories

**定义 13.11.** Cellular motivic stable homotopy category `\mathbf{SH}(S)^{cell}` 是由 bigraded spheres

$$
S^{p,q}
$$

在小余极限、cofibers 和 retracts 下生成的 localizing subcategory。

**命题 13.12.** 若 `E\in\mathbf{SH}(S)^{cell}`，则任意由 spheres 检测的等价可通过 bigraded homotopy groups 检测。

**证明.** Cellular subcategory 由 spheres 生成。应用第三章命题 3.17，令生成子集合为所有 `S^{p,q}`。若态射在所有 spheres 映射空间上诱导等价，则其 fiber 被所有生成子检测为零；由生成性，fiber 为零，态射为等价。`\square`

**注 13.13.** 并非所有 motivic spectra 都 cellular。把计算限制到 cellular subcategory 是有效方法，但会丢失非 cellular 几何信息。

## 13.5 Slice spectral sequence

**定义 13.14.** 对 motivic spectrum `E` 和测试对象 `X`，slice tower 诱导的 exact couple 给出 slice spectral sequence，其 `E_1`-页形式为

$$
E_1^{p,q}(X;E)=\pi_{p+q}\operatorname{Map}(\Sigma_T^\infty X_+,s_q(E)).
$$

具体双次数约定依赖作者对 `S^{p,q}` 的记号。

**命题 13.15.** 若 slice tower 对 `E` 在 `X` 上强收敛，则 slice spectral sequence abut 到 `E` 在 `X` 上的同伦或 cohomology 群。

**证明.** Slice tower 是 `E` 的 filtration。对 `\operatorname{Map}(\Sigma_T^\infty X_+,-)` 应用得到 spectra tower。若该 tower 满足强收敛条件，则 exact couple 产生的 spectral sequence 收敛到 tower 极限的同伦群；若极限识别为 `\operatorname{Map}(\Sigma_T^\infty X_+,E)`，即得到 `E`-groups。`\square`

**注 13.16.** 收敛条件通常是计算中最难的部分之一。不能只写出 `E_1`-页就宣称完成计算。

## 13.6 与 Adams 型方法

**定义 13.17.** 对 ring spectrum `E`，`E`-based Adams tower 由单位 `\mathbb 1\to E` 的 fiber 迭代构造。若 `E=H\mathbb Z/\ell`，得到 motivic Adams spectral sequence 的输入。

**外部输入定理 13.18.** 在适当基域和完备化假设下，motivic Adams spectral sequence 由 motivic Steenrod algebra 控制，并可计算 sphere spectrum 的若干完成同伦群。

**注 13.19.** Adams 方法、slice 方法和 cellular 方法相互作用复杂。它们是计算 stable motivic homotopy groups 的主要工具，但收敛、隐藏扩张和基域依赖不能省略。

## 13.7 本章小结

Slice filtration 用 effective subcategories 按 Tate twist 方向过滤 motivic spectra。Zero slice 把 sphere spectrum 与 motivic cohomology 连接起来；cellular 方法把计算限制在由 bigraded spheres 生成的子范畴中。所有强计算结论都依赖额外基域、系数和收敛假设。

## 练习

**练习 13.1.** 证明 localizing subcategory 对 cofibers 和小余极限封闭。

**练习 13.2.** 写出 `f_q(E)` 的定义，并说明右伴随的作用。

**练习 13.3.** 解释 `s_q(E)` 是 filtration quotient 的意义。

**练习 13.4.** 用生成子检测定理证明命题 13.12。

**练习 13.5.** 说明 slice spectral sequence 的收敛性为什么不是形式自动结论。

**练习 13.6.** 写出 slice tower 产生 exact couple 的步骤。

**练习 13.7.** 比较 slice tower 和 Adams tower 的输入数据。
