# 第四章：六操作的抽象形式主义

## 本章目标

本章在不依赖 motivic 特殊构造的层面定义六操作形式主义，并证明若干纯形式后果。后续章节会把这些抽象符号应用到 `\mathbf{SH}(-)`。本章的重点是：六操作不是六个孤立函子，而是伴随、幺半结构、base change、projection formula、localization 和复合相干的整体。

## 依赖前置知识

需要伴随函子、presentable stable infinity-categories、closed symmetric monoidal categories、Cartesian square、mate calculus、Beck-Chevalley condition、recollement 和 stable cofiber sequences。

## 4.1 系数系统

**定义 4.1.** 设 `\mathcal B` 是有有限拉回的 infinity-范畴。一个稳定系数系统是函子

$$
\mathcal D:\mathcal B^{op}\longrightarrow
\operatorname{CAlg}(\operatorname{Pr}^L_{\operatorname{st}}).
$$

对 `X\in\mathcal B`，写 `\mathcal D(X)`；对 `f:X\to Y`，写

$$
f^*:\mathcal D(Y)\to\mathcal D(X)
$$

为对应的保持小余极限的强对称幺半正合函子。

**定义 4.2.** 若 `f^*` 有右伴随，记为

$$
f^*:\mathcal D(Y)\rightleftarrows\mathcal D(X):f_*.
$$

称 `f_*` 为普通推前。

**命题 4.3.** 在定义 4.1 的口径下，每个 `f^*` 有右伴随 `f_*`。

**证明.** `f^*` 是 `\operatorname{Pr}^L_{\operatorname{st}}` 中的态射，因此保持小余极限。其定义域和值域 presentable。由 presentable adjoint functor theorem，保持小余极限的函子有右伴随。`\square`

**定义 4.4.** `f_*` 只表示 `f^*` 的右伴随；它不自动满足 proper base change、projection formula 或 constructibility preservation。

**命题 4.5.** 若 `g:\mathcal D(X)\to\mathcal D(Y)` 与 `g':\mathcal D(X)\to\mathcal D(Y)` 都是 `f^*` 的右伴随，则 `g` 与 `g'` 由伴随结构唯一等价。

**证明.** 右伴随由映射空间等价

$$
\operatorname{Map}_{\mathcal D(X)}(f^*A,B)\simeq
\operatorname{Map}_{\mathcal D(Y)}(A,gB)
$$

表示。若 `g` 和 `g'` 都表示同一 functor `A\mapsto\operatorname{Map}(f^*A,B)`，则由 Yoneda lemma 对每个 `B` 得 `gB\simeq g'B`，且自然性给出函子等价。`\square`

## 4.2 六操作结构

**定义 4.6.** 一个六操作形式主义是在稳定系数系统 `\mathcal D` 上额外给出以下数据和公理：

1. 对指定态射类中的 `f:X\to Y`，给出伴随
   $$
   f_!:\mathcal D(X)\rightleftarrows\mathcal D(Y):f^!.
   $$
2. 每个 `\mathcal D(X)` 有 closed symmetric monoidal structure
   $$
   -\otimes_X-,\qquad \underline{\operatorname{Hom}}_X(-,-).
   $$
3. 对 Cartesian 方块给出 ordinary 和 extraordinary base-change transformations，并在规定条件下为等价。
4. 给出 projection formula、proper compatibility、open/closed localization、purity 和复合相干。

**注 4.7.** `f_!` 的存在不是 `f^*` 有右伴随的形式后果。`f_!` 是另一个方向的推前，通常由 proper compactification、gluing 和几何有限性定理构造。

**定义 4.8.** 若 `f` 为 proper 且六操作形式主义给出自然等价

$$
f_!\simeq f_*,
$$

称其满足 proper compatibility。

## 4.3 Base-change mate

**定义 4.9.** 对 Cartesian 方块

$$
\begin{array}{c}
X'\overset{g'}\longrightarrow X\\
\downarrow f'\qquad\downarrow f\\
Y'\overset{g}\longrightarrow Y
\end{array}
$$

ordinary base-change transformation 是自然变换

$$
g^*f_*\longrightarrow f'_*g'^*.
$$

**命题 4.10.** ordinary base-change transformation 可由伴随单位、余单位和伪函子相干构造。

**证明.** 从 `g^*f_*` 出发，插入伴随单位 `\operatorname{id}\to f'_*f'^*`，得到

$$
g^*f_*\longrightarrow f'_*f'^*g^*f_*.
$$

Cartesian 方块给出 pullback 伪函子相干等价

$$
f'^*g^*\simeq g'^*f^*.
$$

于是上式识别为

$$
g^*f_*\longrightarrow f'_*g'^*f^*f_*.
$$

再使用伴随余单位 `f^*f_*\to\operatorname{id}`，得到

$$
g^*f_*\longrightarrow f'_*g'^*.
$$

自然性来自单位、余单位和 pullback 相干的自然性。`\square`

**定义 4.11.** 若命题 4.10 的自然变换为等价，则称该方块满足 ordinary base change。

**定义 4.12.** extraordinary base-change transformation 是自然变换

$$
g^*f_!\longrightarrow f'_!g'^*.
$$

在六操作理论中，它通常是外部几何定理，而不是仅由 `f_!\dashv f^!` 自动为等价。

## 4.4 投影公式

**定义 4.13.** 对 `f:X\to Y`，`f_!` 的投影公式是自然变换

$$
f_!(A\otimes_X f^*B)\longrightarrow f_!A\otimes_Y B
$$

为等价。普通推前版本把 `f_!` 替换为 `f_*`。

**命题 4.14.** 若 `f_!:\mathcal D(X)\to\mathcal D(Y)` 是 `\mathcal D(Y)`-模函子，其中 `\mathcal D(X)` 通过 `f^*` 成为 `\mathcal D(Y)`-模范畴，则 `f_!` 满足投影公式。

**证明.** `\mathcal D(Y)`-模函子结构正是自然等价

$$
f_!(A\otimes_X f^*B)\simeq f_!A\otimes_YB
$$

并要求它与张量积的结合律和单位约束相容。因此投影公式是模函子结构的直接展开。`\square`

**命题 4.15.** 若 `f` 和 `g` 的 `!`-投影公式成立，且六操作复合相干给出 `(gf)_!\simeq g_!f_!` 与 `(gf)^*\simeq f^*g^*`，则 `gf` 的 `!`-投影公式成立。

**证明.** 对 `A\in\mathcal D(X)` 和 `C\in\mathcal D(Z)`，有自然等价链

$$
(gf)_!(A\otimes_X(gf)^*C)
\simeq
g_!f_!(A\otimes_X f^*g^*C)
$$

$$
\simeq
g_!(f_!A\otimes_Y g^*C)
\simeq
g_!f_!A\otimes_Z C
\simeq
(gf)_!A\otimes_Z C.
$$

第二个等价使用 `f` 的投影公式，第三个等价使用 `g` 的投影公式，其余等价来自复合相干。`\square`

## 4.5 Motivic 六操作作为外部输入

**外部输入定理 4.16.** 在本书默认基概形口径及其标准扩展中，赋值

$$
S\longmapsto\mathbf{SH}(S)
$$

支持 Grothendieck 六操作形式主义：对态射 `f:X\to Y` 存在

$$
f^*,\quad f_*,\quad f_!,\quad f^!,\quad -\otimes_X-,\quad \underline{\operatorname{Hom}}_X(-,-),
$$

并满足 base change、projection formula、proper compatibility、localization 和 purity 等公理。

**依赖源.** Ayoub 的 motivic six operations，Cisinski-Deglise 的 mixed motives 与 six functors，Hoyois 的 equivariant motivic six operations，Drew-Gallauer 的 universal six-functor formalism。具体基范畴和定理 locator 后续登记。

**注 4.17.** 定理 4.16 是本书 P0 外部输入。后续章节不会把它当作单一黑箱永久使用，而会拆成 proper、open、closed、smooth、base-change、projection formula、purity 和 duality 等模块。

## 4.6 Localization recollement 的形式后果

**定义 4.18.** 设 `j:U\hookrightarrow X` 为开嵌入，`i:Z\hookrightarrow X` 为闭补。六操作形式主义的 localization 数据包括 `j^*`、`j_!`、`i^*`、`i_*` 及相应伴随和 fully faithful 性，使 `\mathcal D(X)` 由 `\mathcal D(U)` 与 `\mathcal D(Z)` recollement。

**命题 4.19.** 在 localization recollement 假设下，对任意 `E\in\mathcal D(X)` 有自然 cofiber sequence

$$
j_!j^*E\longrightarrow E\longrightarrow i_*i^*E.
$$

**证明.** recollement 给出 `j_!` fully faithful、`i_*` fully faithful，并且 `j^*i_*=0`。伴随余单位给出 `j_!j^*E\to E`。令其 cofiber 为 `C`。对 `j^*` 作用，因 `j^*j_!\simeq\operatorname{id}`，得到 `j^*C\simeq0`。recollement 中被 `j^*` 杀掉的对象正是 `i_*` 的本质像，因此 `C\simeq i_*F`。再对 `i^*` 作用，`i^*j_!=0` 且 `i^*i_*\simeq\operatorname{id}`，得到 `F\simeq i^*E`。故 `C\simeq i_*i^*E`，形成所述 cofiber sequence。`\square`

**推论 4.20.** 若 `E=\mathbb 1_X`，则有 localization cofiber sequence

$$
j_!\mathbb 1_U\longrightarrow\mathbb 1_X\longrightarrow i_*\mathbb 1_Z.
$$

**证明.** `j^*\mathbb 1_X\simeq\mathbb 1_U` 来自 `j^*` 强对称幺半；`i^*\mathbb 1_X\simeq\mathbb 1_Z` 同理。代入命题 4.19。`\square`

## 4.7 本章小结

六操作形式主义由稳定系数系统、伴随、closed symmetric monoidal structures、base-change、projection formula、proper compatibility、localization 和 purity 组成。`f_*` 可由 `f^*` 的右伴随形式得到，但 `f_!`、`f^!` 及其几何性质是深外部输入。对 `\mathbf{SH}(-)` 而言，六操作存在性是本书后续所有几何应用的基础。

## 练习

**练习 4.1.** 证明 `f^*` 若为强对称幺半，则 `f^*\mathbb 1_Y\simeq\mathbb 1_X`。

**练习 4.2.** 写出命题 4.10 的 base-change transformation 在 1-categorical derived category 口径下的对应公式。

**练习 4.3.** 证明命题 4.14 的普通推前版本。

**练习 4.4.** 解释为什么 `f_!\simeq f_*` 应只对 proper `f` 期望成立。

**练习 4.5.** 在命题 4.19 中证明 `i^*j_!=0` 如何用于识别 cofiber。
