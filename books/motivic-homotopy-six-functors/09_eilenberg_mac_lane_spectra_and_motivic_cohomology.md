# 第九章：Eilenberg-Mac Lane spectra、motivic cohomology 与 HZ

## 本章目标

本章引入 motivic Eilenberg-Mac Lane spectrum `H\mathbb Z`，并用 `\mathbf{SH}(S)` 中的映射群定义 motivic cohomology。重点是把 cohomology theory 的形式性质归约到 ring spectrum、六操作和 localization cofiber sequence，而把 `H\mathbb Z` 的构造及其与高 Chow 群、cycle complexes 的比较作为外部输入。

## 依赖前置知识

需要 stable motivic homotopy category、ring spectra、bigraded spheres、six operations、localization、Thom twists、ordinary cohomology theory 和基本代数循环背景。

## 9.1 Motivic Eilenberg-Mac Lane spectrum

**外部输入定理 9.1.** 在标准假设下，存在 motivic Eilenberg-Mac Lane spectrum

$$
H\mathbb Z_S\in\operatorname{CAlg}(\mathbf{SH}(S)),
$$

表示 motivic cohomology，并且随基变换具有相容结构。

**依赖源.** Voevodsky 的 motivic Eilenberg-Mac Lane spaces 和 motivic cohomology；Spitzweck 的一般基上 commutative `\mathbb P^1`-spectrum；Cisinski-Deglise 的 motivic sheaves/motives 口径。

**定义 9.2.** 对 `X\in\operatorname{Sm}_S`，定义 bigraded motivic cohomology 为

$$
H^{p,q}(X,\mathbb Z)=
\pi_0\operatorname{Map}_{\mathbf{SH}(S)}
(\Sigma_T^\infty X_+,\Sigma^{p,q}H\mathbb Z_S).
$$

若使用同伦范畴记号，则写作

$$
H^{p,q}(X,\mathbb Z)=
\operatorname{Hom}_{\operatorname{Ho}\mathbf{SH}(S)}
(\Sigma_T^\infty X_+,\Sigma^{p,q}H\mathbb Z_S).
$$

**定义 9.3.** 若 `E\in\operatorname{CAlg}(\mathbf{SH}(S))`，定义 `E`-cohomology

$$
E^{p,q}(X)=
\pi_0\operatorname{Map}_{\mathbf{SH}(S)}
(\Sigma_T^\infty X_+,\Sigma^{p,q}E).
$$

因此 motivic cohomology 是 `E=H\mathbb Z` 的特例。

**命题 9.4.** `H^{*,*}(X,\mathbb Z)` 是 bigraded ring。

**证明.** `H\mathbb Z_S` 是 commutative algebra object。乘法

$$
\mu:H\mathbb Z_S\otimes H\mathbb Z_S\to H\mathbb Z_S
$$

与 diagonal `X\to X\times_SX` 给出 cup product。交换性、结合律和单位律来自 `H\mathbb Z_S` 的 `E_\infty` 乘法相干和 diagonal 的余交换、余结合、余单位相干。双次数由 suspension 坐标加法给出。`\square`

## 9.2 支撑、紧支撑与 Borel-Moore 版本

**定义 9.5.** 设 `p:X\to S` 为 separated finite type morphism。对 `E\in\mathbf{SH}(S)`，定义带紧支撑 `E`-cohomology 为

$$
E_c^{a,b}(X/S)=
\pi_0\operatorname{Map}_{\mathbf{SH}(S)}
(p_!\mathbb 1_X,\Sigma^{a,b}E).
$$

**定义 9.6.** 定义 `E`-Borel-Moore homology 为

$$
E^{BM}_{a,b}(X/S)=
\pi_0\operatorname{Map}_{\mathbf{SH}(S)}
(\mathbb 1_S,p_!\Sigma^{a,b}p^!E).
$$

不同作者的双次数符号可能相差符号；本书以后续计算表统一。

**命题 9.7.** 若 `p:X\to S` proper，则 `E_c^{a,b}(X/S)` 可用 `p_*` 替代 `p_!`。

**证明.** proper compatibility 给出 `p_!\simeq p_*`。代入定义 9.5 即得。`\square`

**命题 9.8.** 对闭嵌入 `i:Z\hookrightarrow X` 和开补 `j:U\hookrightarrow X`，有带紧支撑 cohomology 的长正合列。

**证明.** 第五章 localization 给出 `\mathbf{SH}(X)` 中的 cofiber sequence

$$
j_!\mathbb 1_U\to \mathbb 1_X\to i_*\mathbb 1_Z.
$$

对结构态射推到 `S` 后，利用 `p_!j_!=(pj)_!` 和 `p_!i_*=(pi)_!`，得到 `\mathbf{SH}(S)` 中的 cofiber sequence

$$
(pj)_!\mathbb 1_U\to p_!\mathbb 1_X\to (pi)_!\mathbb 1_Z.
$$

对 `\operatorname{Map}(-,\Sigma^{a,b}E)` 取同伦群，稳定范畴的 cofiber sequence 给出长正合列。`\square`

## 9.3 与 Chow 群和 Milnor K-theory 的比较

**外部输入定理 9.9.** 若 `X` 是域 `k` 上光滑概形，则 motivic cohomology 与高 Chow 群/代数循环复形相容；特别在常见约定下有

$$
CH^n(X)\simeq H^{2n,n}(X,\mathbb Z).
$$

**依赖源.** Bloch、Voevodsky、Suslin-Voevodsky、Mazza-Voevodsky-Weibel、Cisinski-Deglise。

**外部输入定理 9.10.** 对域 `k`，有 Milnor K-theory 与 diagonal motivic cohomology 的比较

$$
K_n^M(k)\simeq H^{n,n}(\operatorname{Spec}k,\mathbb Z).
$$

**注 9.11.** 定理 9.9 和 9.10 是计算接口，不是定义。定义 9.2 已经在 `\mathbf{SH}(S)` 内部给出；比较定理说明该定义恢复循环和 Milnor K-theory 的经典对象。

## 9.4 Etale cycle maps 与 Bloch-Kato 边界

**外部输入定理 9.12.** 若 `m` 在基上可逆，则存在从 motivic cohomology 到 etale cohomology 的 cycle map

$$
H^{p,q}(X,\mathbb Z/m)\longrightarrow
H^p_{et}(X,\mu_m^{\otimes q}),
$$

并在 Beilinson-Lichtenbaum/Voevodsky norm residue theorem 的范围内成为等价或满足截断等价。

**注 9.13.** 本书不把 Bloch-Kato 定理作为 motivic homotopy 的内部结论。它是 arithmetic motivic cohomology 的深外部输入，只在相关比较章节中使用。

## 9.5 Coefficients

**定义 9.14.** 对交换环 `R`，若 `H R` 已构造为 `H\mathbb Z` 的 base change 或 Eilenberg-Mac Lane spectrum，则定义

$$
H^{p,q}(X,R)=
\pi_0\operatorname{Map}(\Sigma_T^\infty X_+,\Sigma^{p,q}HR).
$$

**命题 9.15.** 若 `R\to R'` 是交换环同态，并且 `HR'=HR\otimes_RR'` 在 `\mathbf{SH}(S)` 中成立，则有自然系数变换

$$
H^{p,q}(X,R)\to H^{p,q}(X,R').
$$

**证明.** 环谱映射 `HR\to HR'` 诱导映射

$$
\Sigma^{p,q}HR\to\Sigma^{p,q}HR'.
$$

对 `\Sigma_T^\infty X_+` 取映射空间并取 `\pi_0`，得到系数变换。`\square`

## 9.6 本章小结

`H\mathbb Z` 把 motivic cohomology 放入 `\mathbf{SH}(S)` 的表示性框架中。Ring spectrum 结构给出 cup product，六操作给出紧支撑、Borel-Moore、localization 长正合列和 Gysin 接口。与 Chow 群、Milnor K-theory、etale cohomology 的比较是外部深定理，必须保留假设。

## 练习

**练习 9.1.** 用定义 9.2 写出 `H^{0,0}(S,\mathbb Z)`。

**练习 9.2.** 证明 commutative ring spectrum 表示的 cohomology 有 cup product。

**练习 9.3.** 从 localization cofiber sequence 推导长正合列。

**练习 9.4.** 解释 `CH^n(X)\simeq H^{2n,n}(X,\mathbb Z)` 为什么是外部输入而不是定义。

**练习 9.5.** 写出系数变换 `\mathbb Z\to\mathbb Z/m` 对 motivic cohomology 的作用。
