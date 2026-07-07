# 第八章：Base change、projection formula 与 Beck-Chevalley 相干

## 本章目标

本章系统整理 motivic 六操作中的 base change、projection formula 和 Beck-Chevalley 相干。第四章已经构造了抽象 mate；本章说明在 `\mathbf{SH}(-)` 中哪些 mate 被外部定理保证为等价，并证明若干组合稳定性。

## 依赖前置知识

需要六操作、Cartesian squares、伴随、mate calculus、projection formula、proper/open/smooth morphisms、stable infinity-categories 和 symmetric monoidal structures。

## 8.1 Cartesian 方块与交换变换

**约定 8.1.** 本章固定 Cartesian 方块

$$
\begin{array}{c}
X'\overset{g'}\longrightarrow X\\
\downarrow f'\qquad\downarrow f\\
Y'\overset{g}\longrightarrow Y.
\end{array}
$$

所有 functors 都在相应 `\mathbf{SH}` 范畴之间取值。

**定义 8.2.** Ordinary exchange transformation 是

$$
\operatorname{Ex}_*:g^*f_*\longrightarrow f'_*g'^*.
$$

Extraordinary exchange transformation 是

$$
\operatorname{Ex}_!:g^*f_!\longrightarrow f'_!g'^*.
$$

**外部输入定理 8.3（Motivic base change）.** 在 motivic 六操作形式主义允许的方块和态射假设下，`\operatorname{Ex}_*` 或 `\operatorname{Ex}_!` 为等价；特别地，proper base change、smooth base change、open base change 和 localization 中出现的 base-change maps 满足所需等价和复合相干。

**依赖源.** Ayoub、Cisinski-Deglise、Drew-Gallauer；特殊 equivariant/stacky 版本需单独定位。

**命题 8.4.** 若 `f` proper，且 extraordinary base change 对 `f` 成立，则 ordinary base change 对 `f` 成立。

**证明.** proper compatibility 给出 `f_!\simeq f_*` 和 `f'_!\simeq f'_*`，因为 proper morphism 在 base change 下仍 proper。将 `\operatorname{Ex}_!` 中的 `!` 替换为 `*`，得到

$$
g^*f_*\simeq g^*f_!\xrightarrow{\sim}f'_!g'^*\simeq f'_*g'^*.
$$

这正是 ordinary base-change map 的 proper 情形；相干性来自 proper compatibility 与 base-change transformation 的相干。`\square`

## 8.2 Projection formula

**外部输入定理 8.5（Motivic projection formula）.** 对六操作允许的 `f:X\to Y`，有自然等价

$$
f_!(A\otimes_X f^*B)\simeq f_!A\otimes_YB.
$$

在适当条件下也有普通推前版本

$$
f_*(A\otimes_X f^*B)\simeq f_*A\otimes_YB.
$$

**命题 8.6.** 若 `f` proper，则 ordinary projection formula 由 `!`-projection formula 推出。

**证明.** 与第五章推论 5.8 相同：proper compatibility 把 `f_!` 与 `f_*` 识别，代入定理 8.5。`\square`

**命题 8.7.** 若 `f` 与 `g` 的 `!`-projection formula 成立，且六操作复合相干成立，则 `gf` 的 `!`-projection formula 成立。

**证明.** 这正是第四章命题 4.15 的 motivic 特例。对 `A\in\mathbf{SH}(X)` 和 `C\in\mathbf{SH}(Z)`，使用

$$
(gf)_!\simeq g_!f_!,\qquad (gf)^*\simeq f^*g^*
$$

并依次应用 `f` 和 `g` 的投影公式。`\square`

## 8.3 Beck-Chevalley 条件

**定义 8.8.** 方块 8.1 满足 Beck-Chevalley 条件，若相应 exchange transformation 为等价。具体使用 `*` 还是 `!` 取决于讨论 ordinary 还是 extraordinary 六操作。

**命题 8.9.** 若两个可复合 Cartesian 方块分别满足 ordinary Beck-Chevalley 条件，则外矩形也满足 ordinary Beck-Chevalley 条件。

**证明.** 设两个方块横向复合。外矩形的 ordinary exchange transformation 由 mate calculus 构造。把外矩形的 `g^*(f_2f_1)_*` 先用复合相干写为 `g^*f_{2*}f_{1*}`，对右方块使用第一条 Beck-Chevalley 等价，再对左方块使用第二条 Beck-Chevalley 等价，最后用复合相干合并为外矩形右下角的推前。mate calculus 的粘合定理保证该合成正是外矩形的 exchange transformation。`\square`

**命题 8.10.** 命题 8.9 对 extraordinary Beck-Chevalley 同样成立。

**证明.** 把普通推前 `f_*` 全部替换为非常推前 `f_!`，并使用 `!`-复合相干。构造和粘合论证与命题 8.9 相同。`\square`

## 8.4 Open 与 closed 情形

**命题 8.11.** 开嵌入的 base change 稳定：若 `j:U\hookrightarrow X` 是开嵌入，并对任意 `g:Y\to X` 形成拉回

$$
\begin{array}{c}
U_Y\longrightarrow U\\
\downarrow j_Y\qquad\downarrow j\\
Y\overset{g}\longrightarrow X,
\end{array}
$$

则 `j_Y` 仍为开嵌入，且有 exchange equivalence

$$
g^*j_!\simeq j_{Y!}g_U^*.
$$

**证明.** 开嵌入在 base change 下保持为开嵌入。exchange equivalence 是 motivic base change 定理 8.3 在 open immersion 情形的特例。`\square`

**命题 8.12.** 闭嵌入的 base change 稳定：若 `i:Z\hookrightarrow X` 是闭嵌入，则任意 base change `i_Y:Z_Y\hookrightarrow Y` 仍为闭嵌入，并有相应的 `i_*` base-change equivalence。

**证明.** 闭嵌入在 base change 下保持为闭嵌入。由于闭嵌入 proper，`i_!\simeq i_*`。对 `i` 应用 proper base change，即得 `g^*i_*\simeq i_{Y*}g_Z^*`。`\square`

**命题 8.13.** Localization cofiber sequence 与 base change 相容。

**证明.** 设 `i:Z\hookrightarrow X`，`j:U\hookrightarrow X`，并沿 `g:Y\to X` 拉回得到 `i_Y` 与 `j_Y`。对

$$
j_!j^*E\to E\to i_*i^*E
$$

作用 `g^*`。因 `g^*` 正合，得到 cofiber sequence。由命题 8.11 和 8.12 的 exchange equivalences，第一项识别为 `j_{Y!}j_Y^*g^*E`，第三项识别为 `i_{Y*}i_Y^*g^*E`。这正是 `Y` 上对象 `g^*E` 的 localization cofiber sequence。`\square`

## 8.5 Internal Hom 与 projection formula

**定义 8.14.** `\mathbf{SH}(X)` 的 closed monoidal structure 给出 internal Hom

$$
\underline{\operatorname{Hom}}_X(A,B).
$$

**命题 8.15.** 若 `A\in\mathbf{SH}(Y)` dualizable 且 ordinary projection formula 对 `f` 成立，则有自然等价

$$
f_*\underline{\operatorname{Hom}}_X(f^*A,B)
\simeq
\underline{\operatorname{Hom}}_Y(A,f_*B).
$$

**证明.** 因 `A` dualizable，`\underline{\operatorname{Hom}}_X(f^*A,B)\simeq f^*(A^\vee)\otimes_XB`，且 `\underline{\operatorname{Hom}}_Y(A,f_*B)\simeq A^\vee\otimes_Yf_*B`。对左侧应用 projection formula：

$$
f_*(f^*A^\vee\otimes_XB)\simeq A^\vee\otimes_Yf_*B.
$$

这给出所述等价。`\square`

**注 8.16.** 若 `A` 不 dualizable，内部 Hom 公式需要更强假设或不同形式。不能把命题 8.15 无条件推广到任意 `A`。

## 8.6 本章小结

Base change 和 projection formula 是六操作可计算性的核心。它们的等价性本身是 motivic 六操作的外部输入；一旦给定，这些等价在复合、proper 情形、open/closed localization 和 dualizable internal Hom 计算中有严格的形式后果。

## 练习

**练习 8.1.** 构造 ordinary exchange transformation `g^*f_*\to f'_*g'^*`。

**练习 8.2.** 证明 proper morphism 的 base change 仍 proper。

**练习 8.3.** 写出命题 8.9 的两个横向复合方块，并逐步标出 exchange equivalences。

**练习 8.4.** 证明 localization cofiber sequence 与 base change 相容。

**练习 8.5.** 在命题 8.15 中指出 dualizable 假设被使用的位置。
