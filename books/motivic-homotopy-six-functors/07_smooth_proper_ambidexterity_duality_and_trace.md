# 第七章：Smooth/proper ambidexterity、duality 与 trace

## 本章目标

本章解释六操作和 purity 如何产生 ambidexterity、duality 与 trace。这里的核心原则是：properness 允许 `f_!` 与 `f_*` 比较，smoothness 允许 `f^!` 与 `f^*` 通过切丛 Thom twist 比较。二者合在一起给出 smooth proper 对象的对偶性和 Euler characteristic。

## 依赖前置知识

需要第五章的 proper compatibility、第六章的 smooth purity、closed symmetric monoidal stable categories、dualizable objects、trace、smooth morphisms、proper morphisms 和 Thom twists。

## 7.1 Smooth left adjoint

**外部输入定理 7.1.** 若 `f:X\to Y` 是 smooth morphism，则 pullback

$$
f^*:\mathbf{SH}(Y)\to\mathbf{SH}(X)
$$

有左伴随，常记为

$$
f_\sharp:\mathbf{SH}(X)\to\mathbf{SH}(Y).
$$

**定义 7.2.** `f_\sharp` 称为 smooth pushforward 或 smooth left adjoint。它不同于 `f_!`；二者通过相对切丛的 Thom twist 联系。

**外部输入定理 7.3（Smooth ambidexterity）.** 若 `f:X\to Y` smooth，相对切丛为 `T_f`，则有自然等价

$$
f_!\simeq f_\sharp\Sigma^{-T_f},
$$

等价地，由 smooth purity

$$
f^!\simeq\Sigma^{T_f}f^*
$$

通过伴随转置得到。

**命题 7.4.** 若 `f` etale，则 `f_!\simeq f_\sharp`。

**证明.** etale morphism 的相对切丛为零。由定理 7.3，

$$
f_!\simeq f_\sharp\Sigma^{-0}\simeq f_\sharp,
$$

其中 `\Sigma^0` 是恒等 Thom twist。`\square`

## 7.2 Smooth proper 情形

**命题 7.5.** 若 `f:X\to Y` 同时 smooth 且 proper，则有自然等价

$$
f_*\simeq f_\sharp\Sigma^{-T_f}.
$$

**证明.** smooth ambidexterity 给出 `f_!\simeq f_\sharp\Sigma^{-T_f}`。proper compatibility 给出 `f_!\simeq f_*`。合成得到所述等价。`\square`

**推论 7.6.** 若 `f` finite etale，则

$$
f_*\simeq f_\sharp.
$$

**证明.** finite etale 态射既 proper 又 etale。由命题 7.4 得 `f_!\simeq f_\sharp`，由 proper compatibility 得 `f_!\simeq f_*`，故 `f_*\simeq f_\sharp`。`\square`

**注 7.7.** finite etale 情形也是 norms 和 transfers 相互作用的入口，但 norm functor 是额外 multiplicative structure，不等同于 additive pushforward `f_*` 或 `f_\sharp`。

## 7.3 Dualizable objects

**定义 7.8.** 设 `\mathcal C` 是 symmetric monoidal infinity-category。对象 `A` 称为 dualizable，若存在对象 `A^\vee` 及 evaluation 和 coevaluation

$$
\operatorname{ev}:A^\vee\otimes A\to\mathbb 1,\qquad
\operatorname{coev}:\mathbb 1\to A\otimes A^\vee
$$

满足三角恒等式。

**命题 7.9.** 若 `A` dualizable，则 functor `A\otimes-` 的右伴随等价于 `A^\vee\otimes-`。

**证明.** 对任意 `B,C`，由 evaluation 和 coevaluation 可构造自然映射

$$
\operatorname{Map}(A\otimes B,C)\to
\operatorname{Map}(B,A^\vee\otimes C).
$$

三角恒等式保证该映射有逆，逆映射由 coevaluation 插入 `A\otimes A^\vee` 再用 evaluation 消去 `A` 给出。因此 `A^\vee\otimes-` 表示右伴随。`\square`

**外部输入定理 7.10.** 若 `f:X\to S` smooth proper，则

$$
\Sigma_T^\infty X_+=f_\sharp\mathbb 1_X
$$

在 `\mathbf{SH}(S)` 中 dualizable，其对偶可由

$$
f_*\Sigma^{-T_f}\mathbb 1_X
$$

或等价 Thom twist 表达。

**注 7.11.** 定理 7.10 是 Atiyah duality 的 motivic 形式。具体公式的正负号取决于 `T_f` 的 Thom twist 约定；本书采用第六章 `f^!\simeq\Sigma^{T_f}f^*` 的约定。

## 7.4 Trace 与 Euler characteristic

**定义 7.12.** 设 `A` 是 symmetric monoidal stable infinity-category 中的 dualizable object，且 `u:A\to A` 为 endomorphism。`u` 的 trace 定义为合成

$$
\mathbb 1
\xrightarrow{\operatorname{coev}}
A\otimes A^\vee
\xrightarrow{u\otimes\operatorname{id}}
A\otimes A^\vee
\xrightarrow{\operatorname{sym}}
A^\vee\otimes A
\xrightarrow{\operatorname{ev}}
\mathbb 1.
$$

当 `u=\operatorname{id}_A` 时，称为 `A` 的 Euler characteristic，记为 `\chi(A)`。

**命题 7.13.** trace 对 duality data 的等价替换不变。

**证明.** duality data 的等价替换由对偶对象之间的唯一等价给出，并与 evaluation、coevaluation 相容。trace 的定义只使用这些结构态射及 symmetric monoidal braiding。把所有结构态射沿该等价共轭，合成态射在 `\operatorname{End}(\mathbb 1)` 中保持不变。`\square`

**定义 7.14.** 若 `X` 是 `S` 上 smooth proper，定义 motivic Euler characteristic

$$
\chi_S(X)=\operatorname{tr}(\operatorname{id}_{\Sigma_T^\infty X_+})
\in\operatorname{End}_{\mathbf{SH}(S)}(\mathbb 1_S).
$$

**外部输入定理 7.15.** 在合适假设下，motivic trace formalism 与 fixed point formulas、Grothendieck-Witt-valued Euler characteristics 和 quadratic refinements 相容。

**依赖源.** Hoyois 的 trace formula、Deglise-Jin-Khan 的 motivic fundamental classes、相关 quadratic enumerative geometry 文献。后续章节细化。

## 7.5 Gysin 与 transfer 的边界

**定义 7.16.** 对 finite etale `f:X\to Y`，由 `f_\sharp\simeq f_*` 得到 additive transfer。若再有 norm functor

$$
f_\otimes:\mathbf{SH}(X)\to\mathbf{SH}(Y),
$$

则称其为 multiplicative transfer 或 norm。

**外部输入定理 7.17.** finite etale morphisms 上存在 motivic norm functors，并可用于定义 normed motivic spectra。

**依赖源.** Bachmann-Hoyois；与 transfers 的兼容依赖 Shin 等后续工作。

**命题 7.18.** additive transfer 与 norm 不能由符号相似性识别。

**证明.** additive transfer 来自 stable category 中的伴随和六操作，例如 `f_*, f_\sharp`；它保持加法结构并与直和/余积相容。norm 是对称幺半或 multiplicative 方向的 functor，目标是乘法转移。一个 functor 同时保留加法和乘法需要额外分配律和 normed structure；这些不由六操作伴随自动给出。`\square`

## 7.6 本章小结

Smoothness 给出 `f_\sharp` 与 purity，properness 给出 `f_!\simeq f_*`。smooth proper 情形把二者合并，得到 duality 和 trace。Finite etale 情形进一步连接 additive transfers 与 multiplicative norms，但 norms 是额外结构，必须由独立定理构造。

## 练习

**练习 7.1.** 证明 etale morphism 的相对切丛为零时，smooth ambidexterity 化为 `f_!\simeq f_\sharp`。

**练习 7.2.** 从 proper compatibility 和 smooth ambidexterity 推导命题 7.5。

**练习 7.3.** 写出 dualizable object 的三角恒等式。

**练习 7.4.** 对 `u=\operatorname{id}` 写出 Euler characteristic 的 trace 合成。

**练习 7.5.** 解释 additive transfer 与 norm 的区别。
