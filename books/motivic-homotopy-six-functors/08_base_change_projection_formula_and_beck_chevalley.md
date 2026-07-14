# 第八章：Base change、projection formula 与 Beck-Chevalley 相干

沿 Cartesian 方块先推前再拉回，与先拉回再推前之间总能写出一个 mate；真正有内容的
问题是这个 mate 何时可逆。类似地，投影公式的自然映射总可构造，但普通 `f_*` 版本
只在 properness、dualizability 或其他精确条件下成为等价。把“有交换态射”误写成
“任意换基等价”，会使后续每一次粘合计算失去依据。

我们固定一个 Cartesian 方块，分别追踪 ordinary 与 extraordinary exchange map，
并按竖直态射 proper、换基态射 smooth、或 exceptional 态射 separated 的情形调用
不同外部定理。第四章和附录 D 提供 mate 与 pasting calculus；本章的任务是把它们
落实到 `\mathbf{SH}(-)`，证明投影公式、开闭 localization 和 internal Hom 公式在
复合与换基下如何相容。

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

**定义 8.2.** Ordinary exchange transformation 对任意 `f,g` 都有定义：

$$
\operatorname{Ex}_*:g^*f_*\longrightarrow f'_*g'^*.
$$

若 `f` separated，则 `f'` 也 separated，extraordinary exchange
transformation 定义为

$$
\operatorname{Ex}_!:g^*f_!\longrightarrow f'_!g'^*.
$$

**外部输入定理 8.3（Motivic base change 的精确版本）.** 在约定 5.1 与
方块 8.1 下，有以下彼此不同的结论：

1. 若 `f` separated，则对任意 `g`，
   `\operatorname{Ex}_!:g^*f_!\xrightarrow{\sim}f'_!g'^*`；
2. 若 `f` proper，则对任意 `g`，
   `\operatorname{Ex}_*:g^*f_*\xrightarrow{\sim}f'_*g'^*`；
3. 若 base-change morphism `g` smooth，则对任意 `f`，ordinary
   exchange map `\operatorname{Ex}_*` 为等价。

三类等价都与 identity、composition 和 Cartesian-square pasting 相干。
第 1 项不需要 `g` flat 或 smooth；第 2 项不能删掉 `f` proper；第 3 项
不能删掉 `g` smooth。一般 Cartesian 方块只有 ordinary exchange map，
没有其可逆性的无条件结论。

**依赖源与边界.** Hoyois, Theorem 6.18(3) 给出 exceptional base change
及其右伴随版本，Corollary 6.10 给出 proper base change。第 3 项来自
Proposition 4.2 的 smooth base change；Proposition 6.4 后、Proposition
6.5 前的稳定化段落说明该交换等价沿 `\Sigma^\infty` 延拓到
`\mathbf{SH}`。因此此处的 smooth `g` 不需额外 separated 假设。均取
平凡群。Drew--Gallauer 的 universal theorem 不单独替代这些几何可逆性
结论。

**命题 8.4.** 若 `f` proper，且 extraordinary base change 对 `f` 成立，则 ordinary base change 对 `f` 成立。

**证明.** proper compatibility 给出 `f_!\simeq f_*` 和 `f'_!\simeq f'_*`，因为 proper morphism 在 base change 下仍 proper。将 `\operatorname{Ex}_!` 中的 `!` 替换为 `*`，得到

$$
g^*f_*\simeq g^*f_!\xrightarrow{\sim}f'_!g'^*\simeq f'_*g'^*.
$$

这正是 ordinary base-change map 的 proper 情形；相干性来自 proper compatibility 与 base-change transformation 的相干。`\square`

## 8.2 Projection formula

**外部输入定理 8.5（Exceptional projection formula）.** 若
`f:X\to Y` separated，则对任意 `A\in\mathbf{SH}(X)` 与
`B\in\mathbf{SH}(Y)`，有自然等价

$$
f_!(A\otimes_X f^*B)\simeq f_!A\otimes_YB.
$$

它是 `f_!` 的 `\mathbf{SH}(Y)`-module functor structure，并与复合和 base
change 相干。

**依赖源.** Hoyois, Theorem 6.18(7) 以及构造前的 module-functor
equivalence（该文公式 (6.16)--(6.17) 后的 projection formula），取平凡群。

**命题 8.6.** 若 `f` proper，则 ordinary projection formula 由 `!`-projection formula 推出。

**证明.** 与第五章推论 5.8 相同：proper compatibility 把 `f_!` 与 `f_*` 识别，代入定理 8.5。`\square`

**命题 8.7.** 对任意 `f:X\to Y`，若 `B\in\mathbf{SH}(Y)` dualizable，
则对所有 `A\in\mathbf{SH}(X)`，ordinary projection map

$$
f_*A\otimes_YB\longrightarrow f_*(A\otimes_Xf^*B)
$$

为等价。

**证明.** 这是第四章命题 4.15 对 motivic 系数系统的直接应用；该证明只用
`f^*\dashv f_*`、`f^*` 强对称幺半以及 `B` 的 duality data，不要求 `f`
proper 或 separated。`\square`

**命题 8.8.** 若 `f` 与 `g` 的 `!`-projection formula 成立，且六操作复合相干成立，则 `gf` 的 `!`-projection formula 成立。

**证明.** 这正是第四章命题 4.16 的 motivic 特例。对 `A\in\mathbf{SH}(X)` 和 `C\in\mathbf{SH}(Z)`，使用

$$
(gf)_!\simeq g_!f_!,\qquad (gf)^*\simeq f^*g^*
$$

并依次应用 `f` 和 `g` 的投影公式。`\square`

## 8.3 Beck-Chevalley 条件

**定义 8.9.** 方块 8.1 满足 Beck-Chevalley 条件，若相应 exchange transformation 为等价。具体使用 `*` 还是 `!` 取决于讨论 ordinary 还是 extraordinary 六操作。

**命题 8.10.** 若两个可复合 Cartesian 方块分别满足 ordinary Beck-Chevalley 条件，则外矩形也满足 ordinary Beck-Chevalley 条件。

**证明.** 设两个方块横向复合。外矩形的 ordinary exchange transformation 由 mate calculus 构造。把外矩形的 `g^*(f_2f_1)_*` 先用复合相干写为 `g^*f_{2*}f_{1*}`，对右方块使用第一条 Beck-Chevalley 等价，再对左方块使用第二条 Beck-Chevalley 等价，最后用复合相干合并为外矩形右下角的推前。mate calculus 的粘合定理保证该合成正是外矩形的 exchange transformation。`\square`

**命题 8.11.** 命题 8.10 对 extraordinary Beck-Chevalley 同样成立，前提是
两个竖直 exceptional morphisms 及其复合都在 `\mathcal E` 中。

**证明.** `\mathcal E` 对复合和 base change 封闭，所以三个
extraordinary pushforwards 都有定义。把普通推前 `f_*` 替换为
`f_!`，并使用 `!`-复合相干。附录 D 定理 D.11 的 mate-pasting 说明所得
合成恰是外矩形的 exchange map；两个等价的合成仍为等价。`\square`

## 8.4 Open 与 closed 情形

**命题 8.12.** 开嵌入的 base change 稳定：若 `j:U\hookrightarrow X` 是开嵌入，并对任意 `g:Y\to X` 形成拉回

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

**证明.** 开嵌入在 base change 下保持为开嵌入，且是 separated 态射。
因此两边 exceptional functors 都有定义，结论正是定理 8.3(1) 对 `f=j`
的特例。`\square`

**命题 8.13.** 闭嵌入的 base change 稳定：若 `i:Z\hookrightarrow X` 是闭嵌入，则任意 base change `i_Y:Z_Y\hookrightarrow Y` 仍为闭嵌入，并有相应的 `i_*` base-change equivalence。

**证明.** 闭嵌入在 base change 下保持为闭嵌入。由于闭嵌入 proper，`i_!\simeq i_*`。对 `i` 应用 proper base change，即得 `g^*i_*\simeq i_{Y*}g_Z^*`。`\square`

**命题 8.14.** Localization cofiber sequence 与 base change 相容。

**证明.** 设 `i:Z\hookrightarrow X`，`j:U\hookrightarrow X`，并沿 `g:Y\to X` 拉回得到 `i_Y` 与 `j_Y`。对

$$
j_!j^*E\to E\to i_*i^*E
$$

作用 `g^*`。因 `g^*` 是 stable infinity-categories 间保持小余极限的
函子，它正合，故得到 cofiber sequence。由命题 8.12 和 8.13 的 exchange
equivalences，并由 pullback 伪函子相干
`g_U^*j^*\simeq j_Y^*g^*`、`g_Z^*i^*\simeq i_Y^*g^*`，第一项识别为
`j_{Y!}j_Y^*g^*E`，第三项识别为 `i_{Y*}i_Y^*g^*E`。这正是 `Y` 上
`g^*E` 的 localization cofiber sequence。`\square`

## 8.5 Internal Hom 与 projection formula

**定义 8.15.** `\mathbf{SH}(X)` 的 closed monoidal structure 给出 internal Hom

$$
\underline{\operatorname{Hom}}_X(A,B).
$$

**命题 8.16（pullback--pushforward internal Hom）.** 对任意
`f:X\to Y`、`A\in\mathbf{SH}(Y)` 和 `B\in\mathbf{SH}(X)`，有自然等价

$$
f_*\underline{\operatorname{Hom}}_X(f^*A,B)
\simeq
\underline{\operatorname{Hom}}_Y(A,f_*B).
$$

**证明.** 对任意测试对象 `C\in\mathbf{SH}(Y)`，closed monoidal
adjunction、`f^*\dashv f_*` 和 `f^*` 的强对称幺半性给出自然等价链

$$
\begin{aligned}
\operatorname{Map}_Y(C,f_*\underline{\operatorname{Hom}}_X(f^*A,B))
&\simeq \operatorname{Map}_X(f^*C,
   \underline{\operatorname{Hom}}_X(f^*A,B))\\
&\simeq \operatorname{Map}_X(f^*C\otimes f^*A,B)\\
&\simeq \operatorname{Map}_X(f^*(C\otimes A),B)\\
&\simeq \operatorname{Map}_Y(C\otimes A,f_*B)\\
&\simeq \operatorname{Map}_Y(C,
   \underline{\operatorname{Hom}}_Y(A,f_*B)).
\end{aligned}
$$

该链对 `C` 自然；Yoneda lemma 给出所述对象等价。证明没有使用
dualizability、properness 或 projection formula。`\square`

**注 8.17.** 命题 8.16 对任意 `A` 成立，因为它是 closed adjunction 的
形式后果。Dualizability 只在把 internal Hom 进一步改写成
`A^\vee\otimes-` 时需要；这一步才与命题 8.7 的 ordinary projection
formula 联系。

**例子 8.18（换基保留非约化纤维）.** 设 `k` 为域，`n\geq2`，令
`i:\operatorname{Spec}k\hookrightarrow\mathbb A^1_k` 为原点，令
`g:\mathbb A^1_k\to\mathbb A^1_k` 由 `t\mapsto t^n` 给出。Cartesian
拉回不是约化原点，而是

$$
Z_n=\operatorname{Spec}k[t]/(t^n),
$$

并得到方块

$$
\begin{array}{c}
Z_n\overset{g'}\longrightarrow\operatorname{Spec}k\\
\downarrow i_n\qquad\downarrow i\\
\mathbb A^1_k\overset{g}\longrightarrow\mathbb A^1_k.
\end{array}
$$

闭嵌入 `i` proper，故定理 8.3(2) 给出对每个
`E\in\mathbf{SH}(k)` 的等价

$$
g^*i_*E\simeq i_{n*}g'^*E.
$$

这里右端由 scheme-theoretic fiber `Z_n` 决定，而不是先把它约化为一个点。
若随后使用某个 nil-invariance 定理把 `\mathbf{SH}(Z_n)` 与
`\mathbf{SH}(k)` 比较，那是另一个外部输入；它不属于 proper base change
本身。这一区分在 excess intersection 中尤其重要，因为非横截换基的厚化会进入
余法复形或 excess bundle。

## 8.6 可逆 mate 的适用边界

Base change 和 projection formula 是六操作可计算性的核心，但不存在一个
无条件的“所有 base change、所有普通投影公式”定理。Exceptional base
change 对 separated 推前成立；ordinary base change 在 proper 推前或 smooth
换基等精确情形成立。Exceptional projection formula 对 separated 推前成立；
ordinary 版本对 proper 推前的任意系数或任意推前的 dualizable 基系数成立。
Internal Hom 公式则直接来自 closed adjunction，逻辑上不依赖 ordinary
projection formula。

## 练习

**练习 8.1.** 构造 ordinary exchange transformation `g^*f_*\to f'_*g'^*`。

**练习 8.2.** 证明 proper morphism 的 base change 仍 proper。

**练习 8.3.** 写出命题 8.10 的两个横向复合方块，并逐步标出 exchange equivalences。

**练习 8.4.** 证明 localization cofiber sequence 与 base change 相容。

**练习 8.5.** 逐行验证命题 8.16 的映射空间等价链，并解释为何其中没有
dualizable 假设；再指出把 internal Hom 写成 `A^\vee\otimes-` 时该假设
出现在哪里。
