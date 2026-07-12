# 第四章：六操作的抽象形式主义

## 本章目标

本章在不依赖 motivic 特殊构造的层面定义六操作形式主义，并证明若干纯形式后果。后续章节会把这些抽象符号应用到 `\mathbf{SH}(-)`。本章的重点是：六操作不是六个孤立函子，而是伴随、幺半结构、base change、projection formula、localization 和复合相干的整体。

## 依赖前置知识

需要伴随函子、presentable stable infinity-categories、closed symmetric monoidal categories、Cartesian square、mate calculus、Beck-Chevalley condition、recollement 和 stable cofiber sequences。

## 4.1 系数系统

**定义 4.1.** 设 `\mathcal B` 是有有限拉回的 infinity-范畴，并固定一个
态射类 `\mathcal E`，它包含恒等态射且对复合和 base change 封闭。
`\mathcal E` 称为 **exceptional 态射类**。一个稳定系数系统是函子

$$
\mathcal D:\mathcal B^{op}\longrightarrow
\operatorname{CAlg}(\operatorname{Pr}^L_{\operatorname{st}}).
$$

对 `X\in\mathcal B`，写 `\mathcal D(X)`；对 `f:X\to Y`，写

$$
f^*:\mathcal D(Y)\to\mathcal D(X)
$$

为对应的保持小余极限的强对称幺半正合函子。

**定义 4.2.** 对任意 `f:X\to Y`，若 `f^*` 有右伴随，记为

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

**定义 4.6.** 一个关于 `(\mathcal B,\mathcal E)` 的六操作形式主义是在稳定
系数系统 `\mathcal D` 上额外给出以下数据和公理：

1. 对 `f\in\mathcal E`，给出伴随
   $$
   f_!:\mathcal D(X)\rightleftarrows\mathcal D(Y):f^!.
   $$
2. 每个 `\mathcal D(X)` 有 closed symmetric monoidal structure
   $$
   -\otimes_X-,\qquad \underline{\operatorname{Hom}}_X(-,-).
   $$
3. 对任意 Cartesian 方块给出 ordinary base-change transformation；当
   `f\in\mathcal E` 时，base change 后 `f'\in\mathcal E`，并给出
   extraordinary base-change transformation。哪些变换为等价必须作为
   公理或定理逐项列明。
4. 对 `f\in\mathcal E` 给出 `!`-projection formula；普通推前的
   projection map 另行定义，不能由符号替换得到。
5. 给出适用态射类上的 proper compatibility、open/closed localization、
   purity 和全部复合、单位、base-change pasting 相干。

六个基本函子的方差和类型为

| 操作 | 定义域 | 值域 | 方差/伴随 |
| --- | --- | --- | --- |
| `f^*` | `\mathcal D(Y)` | `\mathcal D(X)` | 对所有 `f` 反变；左伴随于 `f_*` |
| `f_*` | `\mathcal D(X)` | `\mathcal D(Y)` | 对所有 `f` 协变；右伴随 |
| `f_!` | `\mathcal D(X)` | `\mathcal D(Y)` | 仅 `f\in\mathcal E`；左伴随于 `f^!` |
| `f^!` | `\mathcal D(Y)` | `\mathcal D(X)` | 仅 `f\in\mathcal E`；反变的右伴随 |
| `-\otimes_X-` | `\mathcal D(X)^2` | `\mathcal D(X)` | 分变量协变 |
| `\underline{\operatorname{Hom}}_X(A,-)` | `\mathcal D(X)` | `\mathcal D(X)` | `A\otimes_X-` 的右伴随 |

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

**定义 4.12.** 若 `f\in\mathcal E`，则由 base-change 稳定性也有
`f'\in\mathcal E`。Extraordinary base-change transformation 是自然变换

$$
g^*f_!\longrightarrow f'_!g'^*.
$$

在六操作理论中，它通常是外部几何定理，而不是仅由 `f_!\dashv f^!` 自动为等价。

## 4.4 投影公式

**定义 4.13.** 对 `f\in\mathcal E`，`f_!` 的投影公式是自然变换

$$
f_!(A\otimes_X f^*B)\longrightarrow f_!A\otimes_Y B
$$

为等价。对任意 `f`，普通推前的 canonical projection map 方向相反：

$$
f_*A\otimes_YB\longrightarrow
f_*(A\otimes_Xf^*B).
$$

它由 `f^*\dashv f_*` 的 counit 构造。称 ordinary projection formula 对
`(A,B)` 成立，是指这条 map 为等价。只有在已经知道等价后，才可无歧义地
把等价逆写成 `f_*(A\otimes f^*B)\simeq f_*A\otimes B`。

**命题 4.14.** 若 `f_!:\mathcal D(X)\to\mathcal D(Y)` 是 `\mathcal D(Y)`-模函子，其中 `\mathcal D(X)` 通过 `f^*` 成为 `\mathcal D(Y)`-模范畴，则 `f_!` 满足投影公式。

**证明.** `\mathcal D(Y)`-模函子结构正是自然等价

$$
f_!(A\otimes_X f^*B)\simeq f_!A\otimes_YB
$$

并要求它与张量积的结合律和单位约束相容。因此投影公式是模函子结构的直接展开。`\square`

**命题 4.15（dualizable 系数的 ordinary projection formula）.** 设
`f:X\to Y` 为任意态射，`A\in\mathcal D(X)`，且
`B\in\mathcal D(Y)` dualizable。则 canonical map

$$
f_*A\otimes_YB\longrightarrow f_*(A\otimes_Xf^*B)
$$

为等价。

**证明.** 记 `B^\vee` 为 `B` 的对偶。强对称幺半函子 `f^*` 保持
duality data，故 `f^*B` 的对偶是 `f^*(B^\vee)`。对任意
`C\in\mathcal D(Y)`，连续使用两次 duality adjunction 和一次
`f^*\dashv f_*`，得到自然等价

$$
\begin{aligned}
\operatorname{Map}_Y(C,f_*A\otimes B)
&\simeq \operatorname{Map}_Y(C\otimes B^\vee,f_*A)\\
&\simeq \operatorname{Map}_X(f^*C\otimes f^*B^\vee,A)\\
&\simeq \operatorname{Map}_X(f^*C,A\otimes f^*B)\\
&\simeq \operatorname{Map}_Y(C,f_*(A\otimes f^*B)).
\end{aligned}
$$

该等价由 evaluation、coevaluation、伴随单位和余单位构成；展开其 Yoneda
对应态射正是定义 4.13 的 ordinary projection map。由 Yoneda lemma，该
map 为等价。`\square`

**命题 4.16.** 若 `f` 和 `g` 属于 `\mathcal E`，二者的 `!`-投影公式成立，
且六操作复合相干给出 `(gf)_!\simeq g_!f_!` 与
`(gf)^*\simeq f^*g^*`，则 `gf` 的 `!`-投影公式成立。

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

**外部输入定理 4.17（默认 motivic 六操作包）.** 固定附录 A 的有限维
Noetherian 概形 `B`，令 `\mathcal B` 为有限型 `B`-概形的范畴，令
`\mathcal E` 为 separated 态射。赋值

$$
S\longmapsto\mathbf{SH}(S)
$$

支持以下 Grothendieck 六操作形式主义：对每个 `f:X\to Y` 有
`f^*\dashv f_*`；对 `f\in\mathcal E` 有 `f_!\dashv f^!`；并有 fiberwise
closed symmetric monoidal operations。这里 `f\in\mathcal E` 自动为
separated finite type，且由 Nagata compactification 可 compactify。

该形式主义满足：`f_!` 对任意 Cartesian base change 的 exchange
equivalence；proper `f` 的 `f_!\simeq f_*`；`f_!` 的 projection formula；
open-closed gluing；以及 smooth purity。Ordinary base change 只在另行列明的
条件下使用，例如 proper `f`；ordinary projection formula 则至少在命题
4.15 的 dualizable 系数情形成立，对 proper `f` 可由 `!`-公式推出。

**依赖源.** Hoyois, *The six operations in equivariant motivic homotopy theory*,
Theorem 1.1、Theorem 6.18、Corollaries 6.10--6.11（取平凡群）；Ayoub 的
motivic six operations；Drew--Gallauer Theorem 7.14 只承担 universal
coefficient-system 口径，不能单独替代全部 operation compatibilities。
第五、八章逐条记录假设和 locator。

**注 4.18.** 定理 4.17 是本书 P0 外部输入。后续章节不会把它当作单一
黑箱永久使用，而会拆成 proper、open、closed、smooth、base-change、
projection formula、purity 和 duality 等模块。

## 4.6 Localization recollement 的形式后果

**定义 4.19.** 设 `j:U\hookrightarrow X` 为开嵌入，`i:Z\hookrightarrow X` 为闭补。六操作形式主义的 localization 数据包括 `j^*`、`j_!`、`i^*`、`i_*` 及相应伴随和 fully faithful 性，使 `\mathcal D(X)` 由 `\mathcal D(U)` 与 `\mathcal D(Z)` recollement。

**命题 4.20.** 在 localization recollement 假设下，对任意 `E\in\mathcal D(X)` 有自然 cofiber sequence

$$
j_!j^*E\longrightarrow E\longrightarrow i_*i^*E.
$$

**证明.** recollement 给出 `j_!` fully faithful、`i_*` fully faithful，并且 `j^*i_*=0`。伴随余单位给出 `j_!j^*E\to E`。令其 cofiber 为 `C`。对 `j^*` 作用，因 `j^*j_!\simeq\operatorname{id}`，得到 `j^*C\simeq0`。recollement 中被 `j^*` 杀掉的对象正是 `i_*` 的本质像，因此 `C\simeq i_*F`。再对 `i^*` 作用，`i^*j_!=0` 且 `i^*i_*\simeq\operatorname{id}`，得到 `F\simeq i^*E`。故 `C\simeq i_*i^*E`，形成所述 cofiber sequence。`\square`

**推论 4.21.** 若 `E=\mathbb 1_X`，则有 localization cofiber sequence

$$
j_!\mathbb 1_U\longrightarrow\mathbb 1_X\longrightarrow i_*\mathbb 1_Z.
$$

**证明.** `j^*\mathbb 1_X\simeq\mathbb 1_U` 来自 `j^*` 强对称幺半；`i^*\mathbb 1_X\simeq\mathbb 1_Z` 同理。代入命题 4.20。`\square`

## 4.7 本章小结

六操作形式主义由稳定系数系统、伴随、closed symmetric monoidal structures、base-change、projection formula、proper compatibility、localization 和 purity 组成。`f_*` 可由 `f^*` 的右伴随形式得到，但 `f_!`、`f^!` 及其几何性质是深外部输入。对 `\mathbf{SH}(-)` 而言，六操作存在性是本书后续所有几何应用的基础。

## 练习

**练习 4.1.** 证明 `f^*` 若为强对称幺半，则 `f^*\mathbb 1_Y\simeq\mathbb 1_X`。

**练习 4.2.** 写出命题 4.10 的 base-change transformation 在 1-categorical derived category 口径下的对应公式。

**练习 4.3.** 构造定义 4.13 的 ordinary projection map，并在 `B`
dualizable 时逐步核对命题 4.15 的映射空间等价链。

**练习 4.4.** 解释为什么 `f_!\simeq f_*` 应只对 proper `f` 期望成立。

**练习 4.5.** 在命题 4.20 中证明 `i^*j_!=0` 如何用于识别 cofiber。
