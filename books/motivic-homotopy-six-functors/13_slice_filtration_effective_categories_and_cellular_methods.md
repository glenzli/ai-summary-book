# 第十三章：Slice filtration、effective categories 与 cellular methods

`\mathbf{SH}(S)` 中的谱通常远比它的同伦群更难直接计算。Slice filtration 试图沿
Tate 权重逐层逼近一个谱：先取由非负 Tate 悬挂生成的 effective 部分，再比较相邻
effective cover 的差。它与 Postnikov 塔相似，却不是同一个过滤；其层、收敛性和
基概形依赖都带有真正的 motivic 内容。

本章从 localizing subcategory 的右伴随构造 `f_qE`，证明 slice 对更高 effective
对象的正交性，再说明 characteristic zero 域上球谱零层为何是 `H\mathbb Z`。
Cellular 生成与 Adams 塔作为两种不同计算工具随后进入。紧生成、伴随函子定理和
谱序列的形式知识在本节直接使用；任何收敛结论都单独列为假设，不能由 tower 的存在
自动推出。

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

**命题 13.6.** 对任意 `E\in\mathbf{SH}(S)`，`s_q(E)` 属于
`\mathbf{SH}(S)^{eff}(q)`，并且对每个
`A\in\mathbf{SH}(S)^{eff}(q+1)` 有

$$
\operatorname{Map}_{\mathbf{SH}(S)}(A,s_q(E))\simeq *.
$$

因此 `s_q(E)` 精确记录 `q`-effective cover 中对所有 `q+1`-effective 测试对象
不可见的部分。

**证明.** 记 `\mathcal C_q=\mathbf{SH}(S)^{eff}(q)`。有包含关系
`\mathcal C_{q+1}\subseteq\mathcal C_q`。因为 `f_{q+1}(E)\in\mathcal C_q`，
其 counit `f_{q+1}(E)\to E` 由 `f_q(E)\to E` 的 `\mathcal C_q`-余反射泛性质
唯一因子化，得到定义 13.5 中的映射
`f_{q+1}(E)\to f_q(E)`。两项都属于稳定 localizing subcategory
`\mathcal C_q`，所以其 cofiber `s_q(E)` 也属于 `\mathcal C_q`。

现在取 `A\in\mathcal C_{q+1}`。由于 `A` 同时属于 `\mathcal C_q`，两次
余反射的伴随等价给出交换图中的等价

$$
\operatorname{Map}(A,f_{q+1}E)\simeq\operatorname{Map}(A,E)
\simeq\operatorname{Map}(A,f_qE).
$$

故 `\operatorname{Map}(A,f_{q+1}E)\to\operatorname{Map}(A,f_qE)` 为等价。
稳定范畴中的 mapping spectrum 把有限 cofiber sequence 送到 fiber sequence，
于是 `\operatorname{Map}(A,s_qE)` 为零谱，其底层空间可缩。`\square`

**定义 13.7.** 若 slice tower 对 `E` 收敛，则其 slices 给出计算 `E`-cohomology 或 homotopy sheaves 的 spectral sequence。收敛性必须逐对象证明。

## 13.3 Zero slice 与 motivic cohomology

**外部输入定理 13.8（SL-13.8，Voevodsky zero slice）.** 设 `k` 为
characteristic zero field。在 Voevodsky 的 effective slice filtration 中，
sphere spectrum 的零 slice 满足

$$
s_0(\mathbb 1)\simeq H\mathbb Z.
$$

**精确来源与边界.** Vladimir Voevodsky, *On the zero slice of the sphere
spectrum*, Theorem 6.6，`https://arxiv.org/abs/math/0301013`。更一般基和正
特征版本不由该定理给出，需要另行引用推广定理。

**外部输入定理 13.9.** 在定理 13.8 的 characteristic-zero field 口径中，
任意谱 `E` 的各个 `s_q(E)` 具有与 slice tower 相容的 `H\mathbb Z`-module
结构。

**精确来源.** Voevodsky，同上，Introduction, pp. 106--107；该处把
`s_0(\mathbb 1)=H\mathbb Z` 的作用作为 Theorem 6.6 的直接推论说明。本章
只在同一假设下调用它。

**命题 13.10.** 若定理 13.8 和 13.9 适用，则 slice filtration 把 sphere spectrum 的第一层近似与 motivic cohomology 联系起来。

**证明.** 定理 13.8 直接识别 `s_0(\mathbb 1)` 为 `H\mathbb Z`。定理 13.9 说明 slices 落在 `H\mathbb Z`-linear 世界。因此 sphere spectrum 的 slice tower 的零层就是 motivic cohomology，而高层 slices 是 `H\mathbb Z`-module 型对象。`\square`

## 13.4 Cellular subcategories

**定义 13.11.** Cellular motivic stable homotopy category `\mathbf{SH}(S)^{cell}` 是由 bigraded spheres

$$
S^{p,q}
$$

在小余极限、cofibers 和 retracts 下生成的 localizing subcategory。

**命题 13.12.** 若 `E\in\mathbf{SH}(S)^{cell}`，则任意由 spheres 检测的等价可通过 bigraded homotopy groups 检测。

**证明.** Cellular subcategory 由 spheres 生成。应用第三章命题 3.20，令生成子集合为所有 `S^{p,q}`。若态射在所有 spheres 映射空间上诱导等价，则其 fiber 被所有生成子检测为零；由生成性，fiber 为零，态射为等价。`\square`

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

**高级外部输入 13.18.** 在另行指定的基域、素数、完备化和收敛
假设下，motivic Adams spectral sequence 由相应 motivic Steenrod algebra
控制，并可计算 sphere spectrum 的若干完成同伦群。该计算 package 不参与
本章 effective/slice 定义与 zero-slice 定理的证明链。

**注 13.19.** Adams 方法、slice 方法和 cellular 方法相互作用复杂。它们是计算 stable motivic homotopy groups 的主要工具，但收敛、隐藏扩张和基域依赖不能省略。

## 13.7 从 effective cover 到可计算层

Slice filtration 用 effective subcategories 按 Tate twist 方向过滤 motivic
spectra。本章引用的 zero-slice 与 module 结论严格限于 characteristic-zero
field 的 Voevodsky 版本；更一般基和 Adams 型强计算需要独立的外部定理，并须
分别检查系数与收敛。

## 练习

**练习 13.1.** 证明 localizing subcategory 对 cofibers 和小余极限封闭。

**练习 13.2.** 写出 `f_q(E)` 的定义，并说明右伴随的作用。

**练习 13.3.** 解释 `s_q(E)` 是 filtration quotient 的意义。

**练习 13.4.** 用生成子检测定理证明命题 13.12。

**练习 13.5.** 说明 slice spectral sequence 的收敛性为什么不是形式自动结论。

**练习 13.6.** 写出 slice tower 产生 exact couple 的步骤。

**练习 13.7.** 比较 slice tower 和 Adams tower 的输入数据。
