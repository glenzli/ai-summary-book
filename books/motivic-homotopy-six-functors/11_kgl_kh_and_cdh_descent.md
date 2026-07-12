# 第十一章：Algebraic K-theory、homotopy K-theory 与 cdh descent

## 本章目标

本章说明 algebraic K-theory 在 motivic homotopy theory 中的表示性。关键区分是：普通 K-theory 对奇异对象不一定 `\mathbb A^1`-不变，而 homotopy K-theory `KH` 是 `\mathbb A^1`-不变的版本。稳定 motivic homotopy 中的谱 `KGL` 表示同伦不变 K-theory。

## 依赖前置知识

需要 ring spectra、motivic cohomology、six operations、proper/localization、Bott periodicity、algebraic K-theory、homotopy K-theory、cdh topology 的基本背景。

## 11.1 KGL

**外部输入定理 11.1（KG-11.1）.** 设 `S` 为 Noetherian finite-Krull-
dimensional scheme。Röndigs--Spitzweck--Ostvaer 构造了 Bott-periodized
严格交换 motivic ring model `KGL^\beta_S`；经对称幺半模型到
infinity-范畴的比较，其像给出

$$
KGL_S\in\operatorname{CAlg}(\mathbf{SH}(S))
$$

并具有标准 Bott-periodic homotopy type。Cisinski 在同一 Noetherian
finite-dimensional 口径中证明该 homotopy type 表示 homotopy invariant
algebraic K-theory `KH`。

**精确来源.** Röndigs--Spitzweck--Ostvaer, *Motivic strict ring models for
K-theory*, Lemma 2.5、Theorems 3.6 and 4.1，
`https://arxiv.org/abs/0907.4121`；Cisinski, *Descente par eclatements en
K-theorie invariante par homotopie*, Theorem 2.20，
`https://doi.org/10.4007/annals.2013.177.2.2`。第一篇给出严格交换模型；
`“\operatorname{CAlg}”` 的表述还使用对称幺半模型与其呈现的 stable
infinity-category 之间的比较。

**定义 11.2.** 定义

$$
KGL^{p,q}(X)=
\pi_0\operatorname{Map}_{\mathbf{SH}(S)}
(\Sigma_T^\infty X_+,\Sigma^{p,q}KGL_S).
$$

**外部输入定理 11.3（Bott periodicity）.** 在定理 11.1 的基假设下，
`KGL` 满足 `(2,1)`-周期性，即有等价

$$
KGL\simeq\Sigma^{2,1}KGL.
$$

因此 `KGL^{p,q}` 只依赖于组合次数 `p-2q` 的 K-theoretic 信息。

**命题 11.4.** `KGL^{*,*}(X)` 是 bigraded ring。

**证明.** 与命题 9.4 相同，`KGL` 是 commutative motivic ring spectrum，乘法和 `X` 的 diagonal 给出 cup product。`\square`

## 11.2 K-theory 与 KH 的区分

**定义 11.5.** `KH(X)` 表示 Weibel homotopy K-theory，它可视为普通 algebraic K-theory 沿 cosimplicial affine simplices做 `\mathbb A^1`-不变化得到的理论。

**外部输入定理 11.6（KG-11.6）.** 设 `S` Noetherian 且有限 Krull 维，
`X\in\operatorname{Sm}_S`。Cisinski Theorem 2.20 把 `KGL_S` 的表示群与
`KH_*(X)` 自然识别；特别

$$
KGL^{0,0}(X)\simeq KH_0(X)
$$

并且 suspension 次数恢复 `KH` 的全部分次群。这里等号是群同构；谱级对象
仍位于 `\mathbf{SH}(S)`。

**外部输入定理 11.7（KG-11.7）.** 下列两个有限性版本中，自然映射

$$
K(X)\longrightarrow KH(X)
$$

为谱等价：`X=\operatorname{Spec}(R)` 且 `R` regular Noetherian；或 `X`
为 quasi-projective regular Noetherian scheme。

**精确来源与边界.** Charles Weibel, *The K-book*, Chapter IV,
Corollary 12.3.1 与 Lemma 12.8(3)，
`https://sites.math.rutgers.edu/~weibel/Kbook/Kbook.IV.pdf`。本章因此不再把
该比较无条件表述为“任意正则概形”上的定理。

**命题 11.8.** 不能在奇异概形上无条件把 `KGL` 表示的理论称为普通 K-theory。

**证明.** 定理 11.6 说明 `KGL` 表示的是 homotopy invariant K-theory `KH`。定理 11.7 只在正则性等假设下把 `K` 与 `KH` 识别。若 `X` 奇异，普通 K-theory 可能不满足 `\mathbb A^1`-不变性，而 motivic spectra 表示的理论已经通过 `\mathbb A^1`-局部化。因此无条件识别会丢失奇异信息。`\square`

## 11.3 Bott element 与周期性

**定义 11.9.** Bott element 是 `KGL` 中给出 `(2,1)`-周期性等价的类

$$
\beta\in KGL^{2,1}(S)
$$

或等价地给出 map `\Sigma^{2,1}KGL\to KGL` 的可逆元素。

**命题 11.10.** 若 Bott element 可逆，则

$$
KGL^{p,q}(X)\simeq KGL^{p-2q,0}(X).
$$

**证明.** 乘以 `\beta^{-q}` 给出

$$
\Sigma^{p,q}KGL\simeq \Sigma^{p-2q,0}KGL.
$$

对 `\Sigma_T^\infty X_+` 取映射空间并取 `\pi_0`，得到同构。`\square`

**注 11.11.** 该周期性解释了为什么 `KGL` 虽在 bigraded motivic 范畴中定义，却主要恢复单分次的 K-theory 信息。

## 11.4 Localization 与 cdh descent

**外部输入定理 11.12（KG-11.12）.** 在 Noetherian finite-Krull-
dimensional schemes 上，homotopy K-theory 的 presheaf of spectra `KH`
满足 cdh descent。

**精确来源与边界.** Cisinski, 同上，Theorem 3.9；作者 PDF：
`https://www.math.univ-toulouse.fr/~dcisinsk/KHdescente.pdf`。本条不涵盖
stacky 或 equivariant 输入；这些扩展必须另行定位。

**命题 11.13.** 对 closed-open pair `Z\overset{i}\hookrightarrow X\overset{j}\hookleftarrow U`，`KGL`-cohomology 有 localization 长正合列。

**证明.** 这是命题 9.8 对 `E=KGL` 的特例。由 localization cofiber sequence 和映射到 `\Sigma^{p,q}KGL` 得到长正合列。`\square`

**定义 11.14.** 抽象 blow-up square 是 Cartesian 方块

$$
\begin{array}{c}
E\longrightarrow X'\\
\downarrow\qquad\downarrow\\
Z\longrightarrow X
\end{array}
$$

其中 `Z\hookrightarrow X` 闭嵌入，`X'\to X` proper，且 `X'\setminus E\to X\setminus Z` 为同构。

**推论 11.15.** 在定理 11.12 的假设下，`KH` 把定义 11.14 的抽象
blow-up square 送到 spectra 的 Cartesian square。

**证明.** Cisinski Theorem 3.9 的 cdh-descent 结论特别断言：对每个抽象
blow-up distinguished square，canonical descent map

$$
KH(X)\longrightarrow KH(X')\times_{KH(E)}KH(Z)
$$

是谱等价，其中 `E=X'\times_XZ`。在任意 infinity-category 中，一个方块
Cartesian 当且仅当其左上对象到其余三项的 pullback 的 canonical map 为
等价；将上一等价代入，正得 `KH` 像方块 Cartesian。此推论使用定理 11.12，
不是 closed--open localization 的形式后果。`\square`

**注 11.16.** cdh descent 比 closed-open localization 更强，涉及抽象 blow-up squares。它不是第五章 localization 的直接形式后果，必须作为 K-theory 的额外 descent 定理处理。

## 11.5 Chern character 与 rational comparison

**高级外部输入 11.17（P1）.** 在另外指定的有理化、基概形和有限性假设
下，motivic Chern character 把 `KGL_\mathbb Q` 与 motivic cohomology 的
Tate 分量联系起来。本节的条件命题不调用某个无条件分解；实际计算前须
另补所采用版本的 locator。

**命题 11.18.** 若有 ring spectrum map

$$
KGL_\mathbb Q\longrightarrow \prod_i\Sigma^{2i,i}H\mathbb Q
$$

并且它是等价，则 `KGL_\mathbb Q`-cohomology 分解为 motivic cohomology 的乘积。

**证明.** 对 `\Sigma_T^\infty X_+` 取映射空间。右侧乘积在稳定 presentable 范畴中由 mapping spaces 转为乘积：

$$
\operatorname{Map}(\Sigma_T^\infty X_+,\prod_i\Sigma^{2i,i}H\mathbb Q)
\simeq
\prod_i\operatorname{Map}(\Sigma_T^\infty X_+,\Sigma^{2i,i}H\mathbb Q).
$$

取同伦群后得到分解。`\square`

## 11.6 本章小结

`KGL` 在 Noetherian finite-dimensional 基上由严格交换 Bott 模型给出并表示
`KH`。`K\simeq KH` 的本章版本限于 regular Noetherian rings 或
quasi-projective regular Noetherian schemes。Localization 来自六操作，
而抽象 blow-up excision 来自 Cisinski 的 cdh descent；二者逻辑上独立。

## 练习

**练习 11.1.** 用 ring spectrum 结构证明 `KGL^{*,*}(X)` 有乘法。

**练习 11.2.** 解释为什么 `\mathbb A^1`-不变性会改变奇异 K-theory。

**练习 11.3.** 从 closed-open localization 推导 `KGL`-cohomology 长正合列。

**练习 11.4.** 说明 cdh descent 与 Nisnevich descent 的差别。

**练习 11.5.** 在有理 Chern character 等价假设下写出 `KGL_\mathbb Q^{0,0}(X)` 的 motivic cohomology 分解形式。

**练习 11.6.** 用 Bott element 推导 `KGL^{p,q}(X)\simeq KGL^{p-2q,0}(X)`。

**练习 11.7.** 写出抽象 blow-up square 并说明它为什么强于 closed-open localization。
