# 第三十九章：Goodwillie calculus、excisive functors 与函子导数

## 本章目标

Goodwillie calculus 把同伦函子近似为“多项式函子”。它为 spaces、spectra、稳定 $\infty$-范畴和代数 $K$-理论中的非线性现象提供 Taylor tower。范畴论核心是：用 homotopy pushout cube 与 homotopy pullback cube 定义 excisive functors，用 cross-effects 和 spectra with symmetric group action 描述 homogeneous layers。

## 依赖前置知识

需要 $\infty$-范畴中的有限极限和余极限、稳定 $\infty$-范畴、spectrum、filtered colimit、Bousfield localization、functor categories 和 descent。

## 39.1 Excisive functors

**定义 39.1.** 设 $C,D$ 为有有限极限和余极限的 pointed $\infty$-范畴。函子 $F:C\to D$ 称为 reduced，若 $F(0)\simeq0$。

**定义 39.2.** 函子 $F:C\to D$ 称为 $1$-excisive，若它把 homotopy pushout squares 送到 homotopy pullback squares。若 $D$ 稳定，则 pullback squares 与 pushout squares 一致，所以 $1$-excisive 函子也称为 linear functor。

**命题 39.3.** 若 $D$ 稳定，正合函子 $F:C\to D$ 是 $1$-excisive。

**证明.** 正合函子保持有限极限和有限余极限。Homotopy pushout square 被送到 $D$ 中的 pushout square；因 $D$ 稳定，pushout square 等价于 pullback square。因此 $F$ 为 $1$-excisive。$\square$

**定义 39.4.** 对 $n\ge0$，函子 $F:C\to D$ 称为 $n$-excisive，若它把 strongly homotopy cocartesian $(n+1)$-cubes 送到 homotopy cartesian $(n+1)$-cubes。

**例子 39.5.** $0$-excisive 函子是常值函子。$1$-excisive 函子是线性近似。稳定范畴之间保持有限余极限的正合函子给出典型 $1$-excisive 函子。

## 39.2 Taylor 近似与 Goodwillie tower

**外部输入定理 39.6.** 在合适 presentability 和可达性假设下，对函子 $F:C\to D$ 存在 universal $n$-excisive approximation

$$
F\to P_nF,
$$

即任意从 $F$ 到 $n$-excisive 函子的自然变换唯一因子化经 $P_nF$。

**定义 39.7.** Goodwillie tower 是自然塔

$$
\cdots\to P_nF\to P_{n-1}F\to\cdots\to P_1F\to P_0F.
$$

其第 $n$ 层定义为纤维

$$
D_nF=\operatorname{fib}(P_nF\to P_{n-1}F).
$$

**外部输入定理 39.8（Goodwillie 层的齐次性）.** 在 Goodwillie tower 存在且满足标准稳定性假设时，第 $n$ 层

$$
D_nF=\operatorname{fib}(P_nF\to P_{n-1}F)
$$

是 $n$-homogeneous；特别地它是 $n$-excisive，并满足

$$
P_{n-1}(D_nF)\simeq0.
$$

此结论依赖 Goodwillie 近似函子与纤维、迭代近似和齐次层理论的相容性，本书把它作为外部输入。

**定义 39.9.** 函子 $H$ 称为 $n$-homogeneous，若它 $n$-excisive 且 $P_{n-1}H\simeq0$。

## 39.3 Cross-effects

**定义 39.10.** 对 reduced 函子 $F:C\to D$，二重 cross-effect 定义为

$$
\operatorname{cr}_2F(X,Y)=\operatorname{fib}\bigl(F(X\vee Y)\to F(X)\times F(Y)\bigr).
$$

更一般的 $\operatorname{cr}_nF(X_1,\dots,X_n)$ 是由 $F$ 在所有楔和子集上的值形成的全纤维。

**命题 39.11.** 若 $F$ 是 $1$-excisive reduced，则 $\operatorname{cr}_2F\simeq0$。

**证明.** 对 pointed $\infty$-范畴，$X\vee Y$ 是 pushout

$$
X\leftarrow0\to Y.
$$

$1$-excisive 把它送为 pullback square

$$
F(X\vee Y)\simeq F(X)\times_{F(0)}F(Y).
$$

因 $F$ reduced，$F(0)=0$，所以 $F(X\vee Y)\simeq F(X)\times F(Y)$，纤维为零。$\square$

**命题 39.12.** Cross-effects 对变量对称，并带有自然 $\Sigma_n$-作用。

**证明.** $\operatorname{cr}_nF$ 由所有子集 $S\subseteq\{1,\dots,n\}$ 的楔和 $\bigvee_{i\in S}X_i$ 组成的立方图的全纤维定义。置换 $\{1,\dots,n\}$ 会置换该立方图，因全纤维对图形同构自然，得到 $\Sigma_n$-作用。$\square$

## 39.4 导数与 homogeneous functors

**外部输入定理 39.13.** 对从 pointed spaces 到 spectra 的 reduced finitary functor $F$，其 $n$-homogeneous layer 由带 $\Sigma_n$-作用的 spectrum $\partial_nF$ 控制：

$$
D_nF(X)\simeq \bigl(\partial_nF\wedge X^{\wedge n}\bigr)_{h\Sigma_n}.
$$

谱 $\partial_nF$ 称为 $F$ 的第 $n$ 个 Goodwillie derivative。

**例子 39.14.** 恒等函子 $\operatorname{id}_{\mathcal S_*}$ 的 Goodwillie derivatives 形成 spectral Lie operad 的核心例子。其完整描述是 Goodwillie calculus 与 operad 理论交汇处的重要定理。

**命题 39.15.** 若 $F$ 是 reduced $1$-homogeneous functor from pointed spaces to spectra，则存在 spectrum $E$ 使

$$
F(X)\simeq E\wedge \Sigma^\infty X.
$$

**证明.** 这是定理 39.13 在 $n=1$ 的特例。此时 $\Sigma_1$ 平凡，公式变为

$$
D_1F(X)\simeq \partial_1F\wedge X.
$$

因 $F$ 已 $1$-homogeneous，$F\simeq D_1F$。令 $E=\partial_1F$，并把 pointed space 先稳定化为 $\Sigma^\infty X$，得到结论。$\square$

## 39.5 Chain rule 与 operad 结构

**外部输入定理 39.16（Goodwillie chain rule）.** 对合适 reduced finitary functors $F,G$，复合 $F\circ G$ 的 derivatives 由 symmetric sequences 的 composition product 给出：

$$
\partial_*(F\circ G)\simeq \partial_*F\circ \partial_*G.
$$

特别地，恒等函子的 derivatives 形成 operad，函子的 derivatives 形成其模。

**注 39.17.** 这说明 Goodwillie calculus 不只是逐层近似；它带有高阶链式法则。该法则把高阶范畴论、operad、stable homotopy 和 functor calculus 连接起来。

## 39.6 收敛性

**定义 39.18.** 若自然映射

$$
F(X)\to\lim_nP_nF(X)
$$

为等价，则称 Goodwillie tower 在 $X$ 处收敛。

**外部输入定理 39.19.** Goodwillie tower 的收敛性需要 connectivity 或 analytic 条件。对解析函子，在足够连通输入上 Taylor tower 收敛。

**命题 39.20.** 若 $F\simeq P_NF$ 为 $N$-excisive，则其 Goodwillie tower 从 $N$ 层起稳定，且对所有 $X$ 收敛。

**证明.** 若 $F$ 已 $N$-excisive，则对 $n\ge N$，$P_nF\simeq F$，因为 $F$ 本身满足 $n$-excisive 条件并由 universal property 接收 $F\to P_nF$ 的逆等价。因此塔从第 $N$ 层起常值为 $F$，其极限为 $F$。$\square$

## 39.7 本章小结

Goodwillie calculus 把同伦函子分解为多项式近似 $P_nF$ 和 homogeneous layers $D_nF$。Cross-effects 提取多变量非线性部分；derivatives 把 homogeneous 层表示为带对称群作用的谱；chain rule 则把函子复合变成 symmetric sequences 的复合。它是稳定化思想在函子范畴中的高阶推广。

## 练习

**练习 39.1.** 定义 reduced functor。

**练习 39.2.** 定义 $1$-excisive functor。

**练习 39.3.** 证明稳定目标中的正合函子是 $1$-excisive。

**练习 39.4.** 定义 $n$-excisive functor。

**练习 39.5.** 陈述 $P_nF$ 的泛性质。

**练习 39.6.** 定义 Goodwillie tower 和 $D_nF$。

**练习 39.7.** 定义 $n$-homogeneous functor。

**练习 39.8.** 写出 $\operatorname{cr}_2F$ 的定义。

**练习 39.9.** 证明 reduced $1$-excisive 函子的二重 cross-effect 为零。

**练习 39.10.** 说明 cross-effects 的 $\Sigma_n$-作用来源。

**练习 39.11.** 写出 homogeneous layer 的 derivative 公式。

**练习 39.12.** 陈述 Goodwillie chain rule。

**练习 39.13.** 定义 Goodwillie tower 在对象处收敛。
