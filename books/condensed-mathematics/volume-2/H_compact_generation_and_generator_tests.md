# 附录 H：紧生成、局部化子范畴与生成元检验

## H.0 目标

第二卷多次使用“在生成元上验证，然后推广到全范畴”的论证。本附录把这个模式写成可引用的定理。它服务于三类结论：

1. 投影公式从生成对象推广到所有对象。
2. 局部化核由一组 cone 生成。
3. 函子是否为等价可由生成元上的 Hom 检验判断。

本附录只证明范畴论形式命题；具体 condensed/solid/analytic 范畴是否紧生成、哪些对象构成生成元，仍按第二卷附录 D 的输入定理处理。

## H.1 紧对象与生成族

设 $\mathcal C$ 是有小余积的稳定范畴。

**定义 H.1.** 对象 $K\in\mathcal C$ 称为紧对象，如果对任意对象族 $\{X_i\}_{i\in I}$，自然映射

$$
\bigoplus_{i\in I}\operatorname{Hom}_{\mathcal C}(K,X_i)
\to
\operatorname{Hom}_{\mathcal C}\left(K,\bigoplus_{i\in I}X_i\right)
$$

是同构。在 $\infty$-范畴语言中，将 Hom 换成映射空间，并要求映射空间与 filtered colimit 或小直和按相应定义交换。

**定义 H.2.** 对象族 $\mathcal G$ 称为生成族，如果对任意 $X\in\mathcal C$，

$$
\operatorname{Hom}_{\mathcal C}(G,X[n])=0
\quad
\forall G\in\mathcal G,\ n\in\mathbb Z
$$

推出 $X\simeq0$。

**定义 H.3.** 若 $\mathcal C$ 有一组紧对象 $\mathcal G$ 构成生成族，则称 $\mathcal C$ 为紧生成。

**命题 H.4（生成族检测等价）.** 设 $f:X\to Y$ 是 $\mathcal C$ 中态射。若对所有 $G\in\mathcal G$ 和 $n\in\mathbb Z$，

$$
\operatorname{Hom}_{\mathcal C}(G,X[n])
\to
\operatorname{Hom}_{\mathcal C}(G,Y[n])
$$

是同构，则 $f$ 是等价。

**证明.** 令 $C=\operatorname{cofib}(f)$。对每个 $G,n$，长正合 Hom 列给出

$$
\operatorname{Hom}_{\mathcal C}(G,C[n])=0.
$$

由生成族定义，$C\simeq0$。稳定范畴中 cofiber 为零等价于 $f$ 是等价。证毕。

## H.2 Localizing subcategory

**定义 H.5.** $\mathcal C$ 的 full stable subcategory $\mathcal L$ 称为 localizing subcategory，如果它对所有小余积封闭。

给定对象族 $\mathcal S$，记

$$
\langle\mathcal S\rangle_{\operatorname{loc}}
$$

为包含 $\mathcal S$ 的最小 localizing subcategory。

**命题 H.6（由生成元推出全范畴）.** 若 $\mathcal C$ 由 $\mathcal G$ 紧生成，并且 $\mathcal L\subset\mathcal C$ 是 localizing subcategory，满足 $\mathcal G\subset\mathcal L$，则

$$
\mathcal L=\mathcal C.
$$

**证明.** 记 $\langle\mathcal G\rangle_{\operatorname{loc}}$ 为包含 $\mathcal G$ 的最小 localizing subcategory。由 $\mathcal G\subset\mathcal L$ 且 $\mathcal L$ localizing，有

$$
\langle\mathcal G\rangle_{\operatorname{loc}}\subset\mathcal L.
$$

在紧生成稳定范畴中，“$\mathcal G$ 是生成族”等价于

$$
\langle\mathcal G\rangle_{\operatorname{loc}}=\mathcal C.
$$

因此 $\mathcal C\subset\mathcal L$，故 $\mathcal L=\mathcal C$。证毕。

**证明边界.** 上述等价是紧生成三角范畴的标准定理；第二卷只使用其推论：包含生成元并对小余积、cofiber、shift 封闭的性质可推广到全范畴。

## H.3 由生成元检验自然变换

设 $F,G:\mathcal C\to\mathcal D$ 是稳定范畴之间的精确函子。

**命题 H.7（自然变换的生成元检验）.** 假设：

1. $\mathcal C$ 由 $\mathcal G$ 紧生成。
2. $F$ 和 $G$ 保持小余积。
3. $\eta:F\to G$ 是自然变换。
4. 对每个 $A\in\mathcal G$，$\eta_A:F(A)\to G(A)$ 是等价。

则 $\eta_X:F(X)\to G(X)$ 对所有 $X\in\mathcal C$ 是等价。

**证明.** 令 $\mathcal L$ 为所有使 $\eta_X$ 为等价的对象 $X$ 构成的 full subcategory。由于 $F,G$ 精确，$\mathcal L$ 对 shift 和 cofiber 封闭。由于 $F,G$ 保持小余积，$\mathcal L$ 对小余积封闭。由假设 $\mathcal G\subset\mathcal L$。命题 H.6 给出 $\mathcal L=\mathcal C$。证毕。

**推论 H.8（双变量自然变换）.** 设

$$
\eta_{X,Y}:F(X,Y)\to G(X,Y)
$$

是双变量精确且分别保持小余积的函子之间的自然变换。若 $\mathcal C,\mathcal D$ 分别由 $\mathcal G_\mathcal C,\mathcal G_\mathcal D$ 紧生成，且 $\eta_{X,Y}$ 在

$$
X\in\mathcal G_\mathcal C,\qquad Y\in\mathcal G_\mathcal D
$$

时为等价，则它对所有 $X,Y$ 为等价。

**证明.** 固定 $Y\in\mathcal G_\mathcal D$，对 $X$ 应用命题 H.7，得到所有 $X$ 上成立。再固定任意 $X$，对 $Y$ 应用命题 H.7，得到所有 $Y$ 上成立。证毕。

这正是附录 F 中投影公式生成元检验的抽象版本。

## H.4 全忠实与本质满

**命题 H.9（全忠实的生成元检验）.** 设

$$
F:\mathcal C\to\mathcal D
$$

是保持小余积的精确函子，$\mathcal C$ 由紧生成族 $\mathcal G$ 生成。若对任意 $G,G'\in\mathcal G$ 和 $n\in\mathbb Z$，

$$
\operatorname{Hom}_{\mathcal C}(G,G'[n])
\to
\operatorname{Hom}_{\mathcal D}(F G,F G'[n])
$$

是同构，并且每个 $FG$ 是紧对象，则 $F$ 全忠实。

**证明.** 固定 $G\in\mathcal G$。令 $\mathcal L_G$ 为所有 $X\in\mathcal C$ 使

$$
\operatorname{Hom}_{\mathcal C}(G,X[n])
\to
\operatorname{Hom}_{\mathcal D}(F G,F X[n])
$$

对所有 $n$ 为同构的对象。由于 $G$ 和 $FG$ 紧，左右两边都把小余积变为直和；精确性给出对 shift 和 cofiber 封闭。因此 $\mathcal L_G$ 是 localizing subcategory。它包含 $\mathcal G$，所以等于 $\mathcal C$。

现在固定任意 $X\in\mathcal C$，令 $\mathcal M_X$ 为所有 $Y$ 使

$$
\operatorname{Hom}_{\mathcal C}(X,Y[n])
\to
\operatorname{Hom}_{\mathcal D}(F X,F Y[n])
$$

对所有 $n$ 为同构的对象。由上一段当 $X\in\mathcal G$ 时成立。一般 $X$ 由生成元经 localizing 操作生成；用同样的 localizing 论证把 $X$ 从 $\mathcal G$ 推广到全体对象。于是 $F$ 全忠实。证毕。

**命题 H.10（等价判别）.** 在命题 H.9 的假设下，若 $\mathcal D$ 由 $\{FG\mid G\in\mathcal G\}$ 作为 localizing subcategory 生成，则 $F$ 是等价。

**证明.** 命题 H.9 给出全忠实。本质像包含所有 $FG$，且因 $F$ 保持小余积和精确三角，本质像是 $\mathcal D$ 的 localizing subcategory。它包含生成族 $\{FG\}$，故等于 $\mathcal D$。证毕。

## H.5 对 solid/analytic 的使用边界

在第二卷中，上述命题只能按以下方式使用：

1. 若要证明投影公式，只能在已经知道两边保持小余积且有明确生成元时使用 H.8。
2. 若要证明某个 localization 是幺半 localization，仍需单独证明核是张量理想；H.7 不能替代张量理想性。
3. 若要证明某个解析化函子是等价，必须同时证明全忠实和本质满；H.9 和 H.10 只给形式判别。
4. 具体生成元，例如 solid 理论中的 profinite 测度对象，来自 Scholze 输入定理，不由本附录构造。

## H.6 练习

**练习 H.1.** 证明紧对象的有限直和仍是紧对象。

**练习 H.2.** 设 $\eta:F\to G$ 满足命题 H.7。写出 $\mathcal L$ 对 cofiber 封闭的证明。

**练习 H.3.** 用推论 H.8 重写附录 F 命题 F.7 的证明。

**练习 H.4.** 给出一个例子说明：一个函子在生成元上本质满，不足以推出它在全范畴上本质满；还需要本质像对 localizing 操作封闭并生成目标范畴。
