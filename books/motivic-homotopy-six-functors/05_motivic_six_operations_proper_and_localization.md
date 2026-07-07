# 第五章：Motivic 六操作、proper compatibility 与 localization

## 本章目标

本章把第四章的抽象六操作形式主义应用到 `\mathbf{SH}(-)`。本章不重证 motivic 六操作的存在性，而是把它作为 P0 外部输入定理，并严格推出 proper compatibility 与 open-closed localization 的基本形式后果。

## 依赖前置知识

需要前四章、稳定 motivic homotopy category、六操作形式主义、proper/open/closed morphisms、recollement、stable cofiber sequences 和 projection formula。

## 5.1 Motivic 系数系统

**约定 5.1.** 本章的基范畴 `\mathcal B` 取为默认有限维 Noetherian 概形及允许态射组成的范畴；若使用更一般 qcqs 概形或代数栈，必须另行声明。

**外部输入定理 5.2（Motivic 六操作）.** 赋值

$$
S\longmapsto\mathbf{SH}(S)
$$

扩展为六操作形式主义。对 `f:X\to Y`，有

$$
f^*,\quad f_*,\quad f_!,\quad f^!,\quad -\otimes_X-,\quad \underline{\operatorname{Hom}}_X(-,-),
$$

并满足复合相干、base change、projection formula、proper compatibility、localization 和 purity。

**依赖源.** Ayoub、Cisinski-Deglise、Drew-Gallauer；equivariant 和 stacky 变体分别依赖 Hoyois 与 Khan-Ravi 等资料。具体 locator 在后续 `REFERENCE_LOCATOR_LEDGER.md` 中补全。

**定义 5.3.** 对 `S`，记

$$
\mathbb 1_S\in\mathbf{SH}(S)
$$

为 motivic sphere spectrum。若 `f:X\to Y`，则 `f^*` 强对称幺半，故有自然等价

$$
f^*\mathbb 1_Y\simeq\mathbb 1_X.
$$

**命题 5.4.** 对任意 `E,F\in\mathbf{SH}(Y)`，有自然等价

$$
f^*(E\otimes_YF)\simeq f^*E\otimes_Xf^*F.
$$

**证明.** 定理 5.2 把 `f^*` 作为六操作形式主义中的强对称幺半 pullback。强对称幺半函子的结构态射正给出单位和张量积的自然等价。`\square`

## 5.2 Proper compatibility

**定义 5.5.** 若 `f:X\to Y` 为 proper 态射，proper compatibility 是六操作形式主义中的自然等价

$$
f_!\simeq f_*.
$$

**外部输入定理 5.6.** Motivic 六操作满足 proper compatibility：对 proper `f`，有自然等价 `f_!\simeq f_*`，并与复合相干。

**依赖源.** Ayoub、Cisinski-Deglise；在 universal six-functor formalism 中作为六操作结构的一部分出现。

**命题 5.7.** 若 `f:X\to Y` 与 `g:Y\to Z` 均 proper，则 `(gf)_!\simeq(gf)_*` 与 `g_!f_!\simeq g_*f_*` 相容。

**证明.** proper 态射在复合下封闭。六操作复合相干给出

$$
(gf)_!\simeq g_!f_!,\qquad (gf)_*\simeq g_*f_*.
$$

对 `f` 和 `g` 使用 proper compatibility 得到 `g_!f_!\simeq g_*f_*`。定理 5.6 要求 proper compatibility 与复合相干，因此这条等价与 `(gf)` 的 proper compatibility 一致。`\square`

**推论 5.8.** 若 `f` proper，则投影公式可写成普通推前形式

$$
f_*(A\otimes_X f^*B)\simeq f_*A\otimes_YB.
$$

**证明.** 定理 5.2 给出 `!`-投影公式。由 proper compatibility 把 `f_!` 替换为 `f_*`，得到所述等价。`\square`

## 5.3 Open immersion 与 extension by zero

**定义 5.9.** 对开嵌入 `j:U\hookrightarrow X`，`j_!` 称为 extension by zero。它是 `j^!` 的左伴随；在 motivic 六操作中还满足 `j^!\simeq j^*`。

**外部输入定理 5.10.** 对开嵌入 `j`，有自然等价

$$
j^!\simeq j^*
$$

并且 `j_!` fully faithful。

**命题 5.11.** 对开嵌入 `j:U\hookrightarrow X`，单位态射

$$
E\longrightarrow j^*j_!E
$$

为等价。

**证明.** `j_!` fully faithful 等价于伴随 `j_!\dashv j^!` 的单位 `\operatorname{id}\to j^!j_!` 为等价。由定理 5.10，`j^!\simeq j^*`，因此得到 `E\simeq j^*j_!E`。`\square`

**命题 5.12.** 若 `E\in\mathbf{SH}(X)` 且 `j^*E\simeq0`，则 `E` 支撑在闭补上，即在 localization recollement 下 `E\simeq i_*F` 对某个 `F\in\mathbf{SH}(Z)` 成立。

**证明.** 设 `i:Z\hookrightarrow X` 为闭补。由定理 5.2 的 localization recollement，`\mathbf{SH}(X)` 中被 `j^*` 杀掉的 full subcategory 等于 `i_*` 的本质像。故存在 `F` 使 `E\simeq i_*F`。`\square`

## 5.4 Closed immersion 与 localization cofiber sequence

**定义 5.13.** 对闭嵌入 `i:Z\hookrightarrow X` 和开补 `j:U\hookrightarrow X`，motivic localization triangle 指自然 cofiber sequence

$$
j_!j^*E\longrightarrow E\longrightarrow i_*i^*E.
$$

**外部输入定理 5.14（Motivic localization）.** 对任意闭嵌入 `i:Z\hookrightarrow X` 及其开补 `j:U\hookrightarrow X`，`\mathbf{SH}(X)`、`\mathbf{SH}(U)` 和 `\mathbf{SH}(Z)` 组成 open-closed recollement，特别有定义 5.13 中的 cofiber sequence。

**命题 5.15.** 对 `E=\mathbb 1_X`，有 cofiber sequence

$$
j_!\mathbb 1_U\longrightarrow\mathbb 1_X\longrightarrow i_*\mathbb 1_Z.
$$

**证明.** 由 `j^*` 和 `i^*` 强对称幺半，`j^*\mathbb 1_X\simeq\mathbb 1_U` 且 `i^*\mathbb 1_X\simeq\mathbb 1_Z`。代入定义 5.13。`\square`

**命题 5.16.** 若 `E\in\mathbf{SH}(X)` 且 `i^*E\simeq0`，则 counit

$$
j_!j^*E\longrightarrow E
$$

为等价。

**证明.** localization cofiber sequence 给出

$$
j_!j^*E\to E\to i_*i^*E.
$$

若 `i^*E\simeq0`，则第三项为零对象。稳定范畴中 cofiber 为零等价于第一箭头为等价。`\square`

## 5.5 支撑与 exceptional functors

**定义 5.17.** 对闭嵌入 `i:Z\hookrightarrow X`，定义带支撑对象的 full subcategory

$$
\mathbf{SH}_Z(X)=\{E\in\mathbf{SH}(X)\mid j^*E\simeq0\},
$$

其中 `j:X\setminus Z\hookrightarrow X` 是开补。

**命题 5.18.** `i_*:\mathbf{SH}(Z)\to\mathbf{SH}(X)` 诱导等价

$$
\mathbf{SH}(Z)\simeq\mathbf{SH}_Z(X).
$$

**证明.** 定理 5.14 的 recollement 包含 `i_*` fully faithful，并且其本质像正是被 `j^*` 杀掉的对象。定义 5.17 说明该本质像就是 `\mathbf{SH}_Z(X)`。`\square`

**注 5.19.** `i^!` 与 `i^*` 对闭嵌入一般不同。二者的关系由 purity 或 absolute purity 控制，需要第六章的 Thom twist 和法丛数据。

## 5.6 本章小结

Motivic 六操作把 `\mathbf{SH}(-)` 组织成几何系数系统。Proper compatibility 允许对 proper 态射把 `f_!` 与 `f_*` 识别；open-closed localization 把一个对象分解为开部分和闭支撑部分。第六章将解释闭嵌入和光滑态射中 `f^!` 与 `f^*` 的差异如何由 tangent/normal bundle 的 Thom twist 衡量。

## 练习

**练习 5.1.** 证明 proper morphisms 在复合下封闭，并说明该事实在命题 5.7 中的位置。

**练习 5.2.** 对开嵌入 `j`，从 `j_!` fully faithful 推出 `j^*j_!\simeq\operatorname{id}`。

**练习 5.3.** 用 localization cofiber sequence 证明命题 5.16。

**练习 5.4.** 解释为什么 `i^*E\simeq0` 与 `j^*E\simeq0` 表示不同支撑条件。

**练习 5.5.** 写出 `E=\Sigma_T^\infty X_+` 时命题 5.15 的形式。
