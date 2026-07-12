# 第二章：A1-局部化与 motivic spaces

## 本章目标

本章把 Nisnevich sheaves of spaces 进一步局部化，使仿射直线 `\mathbb A^1` 在同伦意义下成为单位区间。所得范畴 `\mathbf H(S)` 是非稳定 motivic homotopy theory 的基本环境。本章只使用 accessible localization 和 Yoneda lemma 证明基本性质。

## 依赖前置知识

需要第一章的光滑站点、Nisnevich sheaves、presentable infinity-categories、Bousfield/accesssible localization、Yoneda lemma、pointed objects 和 cofiber。

## 2.1 A1-局部化

**约定 2.1.** 本章仍固定 `\mathbb U`-小有限维 Noetherian 基概形 `S`。
对 `X\in\operatorname{Sm}_S`，记

$$
p_X:X\times_S\mathbb A^1_S\longrightarrow X
$$

为投影。

**定义 2.2.** 令 `W_{\mathbb A^1}` 为 `\operatorname{Shv}_{Nis}(\operatorname{Sm}_S)` 中由所有 `p_X` 的像组成的态射集合。`\mathbb A^1`-局部化定义为 accessible localization

$$
L_{\mathbb A^1}:\operatorname{Shv}_{Nis}(\operatorname{Sm}_S)
\longrightarrow
\mathbf H(S)
$$

其中 `\mathbf H(S)` 是 `W_{\mathbb A^1}`-局部对象的 full subcategory。

**定义 2.3.** `\mathbf H(S)` 称为 `S` 上的 motivic spaces 范畴。其对象称为 motivic spaces。

**命题 2.4.** `\mathbf H(S)` 是 presentable infinity-范畴。

**证明.** 第一章命题 1.9 给出 sheaf 范畴 presentable。
`\operatorname{Sm}_S` 是 `\mathbb U`-小骨架，所以
`W_{\mathbb A^1}` 是 `\mathbb U`-小态射集。对它应用外部输入定理 A.10
（HTT Proposition 5.5.4.15），得到局部对象组成的 presentable reflective
subcategory，即 `\mathbf H(S)`。`\square`

**注 2.5.** 本章不声称 `L_{\mathbb A^1}` 是 left exact localization。`A1`-局部化保留的结构需要逐项验证；不能把 `\mathbf H(S)` 自动当作 infinity-topos 使用。

## 2.2 局部对象的检测

**命题 2.6.** Nisnevich sheaf `F` 是 `\mathbb A^1`-局部对象，当且仅当对所有 `X\in\operatorname{Sm}_S`，自然映射

$$
F(X)\longrightarrow F(X\times_S\mathbb A^1_S)
$$

是 spaces 中的等价。

**证明.** 按 localization 的定义，`F` 是 `W_{\mathbb A^1}`-局部对象，当且仅当对每个 `p_X:X\times\mathbb A^1\to X`，诱导映射

$$
\operatorname{Map}(X,F)\longrightarrow
\operatorname{Map}(X\times\mathbb A^1,F)
$$

为等价。由 Yoneda lemma，左侧等价于 `F(X)`，右侧等价于 `F(X\times\mathbb A^1)`。因此得到所述条件。`\square`

**定义 2.7.** 若 Nisnevich sheaf `F` 满足命题 2.6 的条件，则称 `F` 为 `\mathbb A^1`-invariant。

**推论 2.8.** `\mathbf H(S)` 可识别为 `\mathbb A^1`-invariant Nisnevich sheaves 组成的反射 full subcategory。

**证明.** 命题 2.6 正是局部对象条件。accessible localization 的本质像为局部对象组成的 full subcategory，故得结论。`\square`

**命题 2.9.** 对每个 `X\in\operatorname{Sm}_S`，投影

$$
X\times_S\mathbb A^1_S\to X
$$

在 `\mathbf H(S)` 中成为等价。

**证明.** `\mathbf H(S)` 是把 `W_{\mathbb A^1}` 中所有态射反演得到的局部化。该投影属于 `W_{\mathbb A^1}`，故其像为等价。`\square`

**推论 2.10.** `\mathbb A^1_S` 在 `\mathbf H(S)` 中与终对象 `S` 等价。

**证明.** 取 `X=S`，命题 2.9 给出 `S\times_S\mathbb A^1_S\simeq S`。左侧即 `\mathbb A^1_S`。`\square`

## 2.3 泛性质

**命题 2.11.** 设 `\mathcal C` 为 presentable infinity-范畴。预合成局部化函子给出 full faithful 嵌入

$$
\operatorname{Fun}^L(\mathbf H(S),\mathcal C)
\hookrightarrow
\operatorname{Fun}^L(\operatorname{Shv}_{Nis}(\operatorname{Sm}_S),\mathcal C),
$$

其本质像由把所有 `p_X` 送为等价的保持小余极限函子组成。

**证明.** 这是 accessible localization 的泛性质。若 `L:\mathcal D\to\mathcal D[W^{-1}]` 是局部化，则保持小余极限函子 `\mathcal D[W^{-1}]\to\mathcal C` 与保持小余极限且反演 `W` 的函子 `\mathcal D\to\mathcal C` 等价。代入 `\mathcal D=\operatorname{Shv}_{Nis}(\operatorname{Sm}_S)` 和 `W=W_{\mathbb A^1}` 即可。`\square`

**推论 2.12.** 给出从 `\mathbf H(S)` 到 `\mathcal C` 的保持小余极限函子，等价于给出一个 Nisnevich-descent 且 `\mathbb A^1`-invariant 的几何理论，并使其对 sheaf 变量保持小余极限。

**证明.** 由命题 2.11，函子必须且只须对 Nisnevich sheaf 范畴定义并反演 `X\times\mathbb A^1\to X`。这正是 descent 与 `\mathbb A^1`-invariance 两个条件的范畴化表达。`\square`

## 2.4 Pointed motivic spaces

**定义 2.13.** `\mathbf H_*(S)` 是 `\mathbf H(S)` 的 pointed objects 范畴：

$$
\mathbf H_*(S)=\mathbf H(S)_{*/}.
$$

对象为带有基点 `* \to X` 的 motivic spaces。

**命题 2.14.** `\mathbf H_*(S)` 是 pointed presentable infinity-范畴，并有遗忘函子

$$
U:\mathbf H_*(S)\to\mathbf H(S)
$$

及左伴随 `(-)_+`。

**证明.** presentable infinity-范畴的 under-category `\mathcal C_{*/}` 仍 presentable。终对象上的 under-category 是 pointed 的，因为基点对象同时给出零对象。遗忘函子保持极限，左伴随把 `X` 送到 `X\amalg *` 并以第二个分量为基点，记为 `X_+`。`\square`

**定义 2.15.** 在 `\mathbf H_*(S)` 中，cofiber `Y/X` 表示态射 `X\to Y` 的推出

$$
Y\coprod_X *.
$$

**例子 2.16.** `\mathbb G_m` 默认以单位截面 `1:S\to\mathbb G_m` 为基点。商

$$
\mathbb A^1/(\mathbb A^1\setminus0)
$$

是 pointed motivic space，其中 `\mathbb A^1\setminus0=\mathbb G_m` 的像被压到基点。

## 2.5 失败模式

**命题 2.17.** 若 presheaf `F` 满足 `F(X)\simeq F(X\times\mathbb A^1)`，但不满足 Nisnevich descent，则不能由此推出 `F` 是 `\mathbf H(S)` 的对象。

**证明.** `\mathbf H(S)` 是 Nisnevich sheaves 的 `\mathbb A^1`-局部化。对象必须先位于 `\operatorname{Shv}_{Nis}(\operatorname{Sm}_S)`，再满足局部对象条件。presheaf 级别的 `\mathbb A^1`-不变性不包含 descent 条件，因此缺少进入 `\mathbf H(S)` 的第一步。`\square`

**命题 2.18.** 若 `E\to F` 在 `\mathbf H(S)` 中为等价，不可推出其在 `\operatorname{Shv}_{Nis}(\operatorname{Sm}_S)` 中为等价。

**证明.** `\mathbf H(S)` 是局部化后的范畴，局部化函子会反演 `W_{\mathbb A^1}` 中的态射。投影 `X\times\mathbb A^1\to X` 在 `\mathbf H(S)` 中为等价，但在 sheaf 范畴中通常不是等价，因为对测试对象 `Y`，映射空间 `\operatorname{Map}(Y,X\times\mathbb A^1)` 与 `\operatorname{Map}(Y,X)` 不必等价。`\square`

## 2.6 本章小结

Motivic spaces 由两步构成：先施加 Nisnevich descent，再反演所有 `X\times\mathbb A^1\to X`。局部对象正是 `\mathbb A^1`-invariant Nisnevich sheaves。这个定义的强点是泛性质清楚；弱点是很多几何结构不会自动保留，必须在后续章节中逐项建立。

## 练习

**练习 2.1.** 证明 `W_{\mathbb A^1}` 是一个集合而不是 proper class。

**练习 2.2.** 对 `X=S` 写出命题 2.9 的具体结论。

**练习 2.3.** 证明 pointed objects 范畴 `\mathbf H_*(S)` 有零对象。

**练习 2.4.** 给出一个 presheaf 级别性质不能直接下降到 motivic spaces 的例子。

**练习 2.5.** 用命题 2.11 说明 realization functor 若要从 `\mathbf H(S)` 因子化，必须满足哪些条件。
