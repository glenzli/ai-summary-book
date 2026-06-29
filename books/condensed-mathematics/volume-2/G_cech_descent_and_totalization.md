# 附录 G：Cech 下降与 totalization

## G.0 目标

第二卷第六章使用 rational Cech 下降。本附录补充其中的形式部分：Cech nerve、totalization、ordinary sheaf descent、stable category valued descent，以及接受 rational descent 输入后的若干推论。

本附录不证明 Scholze 的 rational descent 定理；它只证明“如果某个对象满足 Cech descent，那么哪些结论可以形式推出”。

## G.1 Cech nerve

设 $\mathcal C$ 是有纤维积的范畴，$u:U\to X$ 是态射。定义 augmented simplicial object

$$
U^\bullet_X\to X
$$

为

$$
U^n_X
=\underbrace{U\times_X\cdots\times_XU}_{n+1\ \text{份}},
\qquad n\ge0.
$$

面映射删除某一个因子，退化映射重复某一个因子。

若覆盖以有限族 $\{U_i\to X\}_{i=1}^r$ 给出，则令

$$
U=\coprod_{i=1}^rU_i.
$$

此时

$$
U^n_X\cong
\coprod_{i_0,\ldots,i_n}
U_{i_0}\times_X\cdots\times_XU_{i_n}.
$$

**定义 G.1.** 设 $\mathcal D$ 有余单纯极限。对反变函子 $F:\mathcal C^{op}\to\mathcal D$，称 $F$ 对覆盖 $U\to X$ 满足 Cech descent，如果自然映射

$$
F(X)\longrightarrow
\operatorname{Tot}\bigl(F(U^\bullet_X)\bigr)
$$

是 $\mathcal D$ 中的同构或等价。

其中

$$
\operatorname{Tot}\bigl(F(U^\bullet_X)\bigr)
=
\lim_{[n]\in\Delta}F(U^n_X).
$$

## G.2 集合值 sheaf 的 Cech 下降

**命题 G.2.** 若 $F$ 是集合值 sheaf，$U\to X$ 是 site 中的覆盖，则

$$
F(X)\cong\operatorname{Tot}\bigl(F(U^\bullet_X)\bigr).
$$

**证明.** totalization 的元素可写成族

$$
(s_n)_{n\ge0},\qquad s_n\in F(U^n_X),
$$

满足所有余单纯相容条件。由退化映射相容，整个族由 $s_0\in F(U)$ 决定；由两个面映射 $U^1_X\rightrightarrows U$ 的相容性，$s_0$ 是覆盖 $U\to X$ 上的匹配族。

sheaf 条件给出唯一 $s\in F(X)$ 使

$$
u^\ast s=s_0.
$$

由函子性，$s$ 的各阶限制自动满足全部余单纯条件，并产生原来的族。因此 $F(X)\to\operatorname{Tot}(F(U^\bullet_X))$ 是双射。证毕。

**推论 G.3.** 若 $A$ 是阿贝尔群值 sheaf，则

$$
A(X)\cong\operatorname{Tot}\bigl(A(U^\bullet_X)\bigr)
$$

在阿贝尔群范畴中成立。

**证明.** 忘记群结构后由命题 G.2 得双射；限制映射均为群同态，因此该双射是群同构。证毕。

## G.3 稳定范畴值 descent

设 $\mathcal D$ 是有小极限的稳定 $\infty$-范畴。本书只使用以下形式性质；不需要在这里展开 $\infty$-范畴模型。

**定义 G.4.** 反变函子 $F:\mathcal C^{op}\to\mathcal D$ 对覆盖 $U\to X$ 满足 $\mathcal D$-值 descent，指自然映射

$$
F(X)\to\operatorname{Tot}F(U^\bullet_X)
$$

是 $\mathcal D$ 中的等价。

**命题 G.5（局部等价推出整体等价）.** 设 $f:F\to G$ 是两个 $\mathcal D$-值反变函子的态射。假设 $F$ 和 $G$ 对覆盖 $U\to X$ 都满足 descent，并且对每个 $n\ge0$，

$$
f(U^n_X):F(U^n_X)\to G(U^n_X)
$$

是等价。则

$$
f(X):F(X)\to G(X)
$$

是等价。

**证明.** 有交换图

$$
\begin{CD}
F(X) @>>> \operatorname{Tot}F(U^\bullet_X)\\
@V f(X) VV @VV \operatorname{Tot}f(U^\bullet_X) V\\
G(X) @>>> \operatorname{Tot}G(U^\bullet_X).
\end{CD}
$$

上下横箭头是等价。右侧竖箭头是逐阶等价的极限，因此是等价。由二出三性质，左侧竖箭头是等价。证毕。

**命题 G.6（纤维局部消失推出整体消失）.** 设 $F:\mathcal C^{op}\to\mathcal D$ 对覆盖 $U\to X$ 满足 descent。若对所有 $n\ge0$，

$$
F(U^n_X)\simeq0,
$$

则 $F(X)\simeq0$。

**证明.** 零余单纯对象的 totalization 是零对象。因此

$$
F(X)\simeq\operatorname{Tot}F(U^\bullet_X)\simeq0.
$$

证毕。

**命题 G.7（映射到 descent 对象）.** 设 $F$ 对 $U\to X$ 满足 descent，$T\in\mathcal D$。则

$$
\operatorname{Map}_{\mathcal D}(T,F(X))
\simeq
\operatorname{Tot}
\operatorname{Map}_{\mathcal D}(T,F(U^\bullet_X)).
$$

**证明.** 映射空间函子 $\operatorname{Map}_{\mathcal D}(T,-)$ 作为右伴随保持极限。于是

$$
\operatorname{Map}(T,F(X))
\simeq
\operatorname{Map}\left(T,\operatorname{Tot}F(U^\bullet_X)\right)
\simeq
\operatorname{Tot}\operatorname{Map}(T,F(U^\bullet_X)).
$$

证毕。

**警告 G.8.** 一般不能从

$$
F(X)\simeq\operatorname{Tot}F(U^\bullet_X)
$$

推出

$$
\operatorname{Map}(F(X),T)
\simeq
\operatorname{Tot}\operatorname{Map}(F(U^\bullet_X),T).
$$

因为 $\operatorname{Map}(-,T)$ 把余极限变成极限，而不是把极限变成极限。若要得到这种方向的公式，需要额外的 dualizability、compactness 或 Cech nerve 给出的余极限陈述。

## G.4 Rational Cech 下降的输入形式

令 $(A,A^+)$ 为离散 Huber pair，令

$$
X=\operatorname{Spa}(A,A^+).
$$

设 $U_i\subset X$ 是有限 rational 覆盖，$U=\coprod_iU_i$，并令 $U_X^\bullet$ 为 Cech nerve。

**输入定理 G.9（rational Cech descent）.** 对第二卷附录 D.6 的 analytic ring 构造，解析模范畴或其派生范畴满足

$$
D(X)\simeq
\operatorname{Tot}D(U_X^\bullet),
$$

其中 $D(U_X^n)$ 表示相应 rational localization 后的解析模派生范畴。

**本书使用的精确部分。** 后续只使用以下后果：

1. 解析模对象可在 rational cover 上限制，并由满足相容条件的局部对象恢复。
2. 解析模态射可在 rational cover 上检测。
3. 解析模态射若在所有 Cech 交叠上为等价，则整体为等价。

这些后果依赖输入定理 G.9；本附录不构造 $D(X)$ 到 totalization 的等价。

## G.5 形式推论

**推论 G.10（rational 局部等价检测）.** 设 $M,N\in D(X)$，$f:M\to N$ 为态射。若对 Cech nerve 的每一阶 rational localization，

$$
f|_{U^n_X}:M|_{U^n_X}\to N|_{U^n_X}
$$

都是等价，则 $f$ 是等价。

**证明.** 输入定理 G.9 是关于稳定范畴的 totalization 等价

$$
D(X)\simeq\operatorname{Tot}D(U_X^\bullet).
$$

在一个范畴的极限中，态射为等价当且仅当它在每个投影分量中为等价。假设正是说 $f$ 在 $D(U_X^n)$ 的每个分量中为等价，因此 $f$ 在 $D(X)$ 中为等价。证毕。

**推论 G.11（rational 局部零对象检测）.** 设 $M\in D(X)$。若对所有 $n\ge0$，

$$
M|_{U^n_X}\simeq0,
$$

则 $M\simeq0$。

**证明.** 由 G.9，$M$ 对应 totalization 中的相容族 $\{M|_{U_X^n}\}_n$。若每个分量为零对象，则该相容族是零对象，因此 $M\simeq0$。证毕。

**推论 G.12（局部 fiber 判别）.** 设 $f:M\to N$ 是 $D(X)$ 中态射。若对所有 $n\ge0$，

$$
\operatorname{Fib}(f)|_{U^n_X}\simeq0,
$$

则 $f$ 是等价。

**证明.** 由推论 G.11 得 $\operatorname{Fib}(f)\simeq0$。稳定范畴中，一个态射是等价当且仅当它的 fiber 为零。证毕。

## G.6 与第二卷正文的连接

第六章中的 Cech 下降公式不应被理解成 ordinary sheaf equalizer 的简单推广。正确层级如下：

1. 集合值或阿贝尔群值 sheaf 的 Cech 下降由 sheaf 条件证明，见命题 G.2。
2. 稳定范畴值 descent 以 totalization 表示，见定义 G.4。
3. rational analytic 模范畴满足 descent 是 Scholze 输入定理，见 G.9。
4. 一旦接受 G.9，局部等价、局部零对象和 fiber 判别都是形式推论。

## G.7 练习

**练习 G.1.** 对一个二元覆盖 $U_1,U_2\to X$ 写出 $U^0_X$ 和 $U^1_X$ 的分量。

**练习 G.2.** 证明命题 G.2 中 totalization 的高阶相容条件由一阶匹配条件推出。

**练习 G.3.** 设 $\mathcal D$ 是稳定范畴。证明若 $f$ 的 cofiber 为零，则 $f$ 是等价。

**练习 G.4.** 解释为什么警告 G.8 对投影公式和 Grothendieck duality 的方向很重要。
