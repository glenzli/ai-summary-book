# 第三十五章：Barr-Beck-Lurie 单子性、余单子下降与 descent

## 本章目标

本章把普通 Beck 单子性推广到 $\infty$-范畴语境。Barr-Beck-Lurie 定理是现代 descent、代数几何中的仿射性判别、模范畴重构和高阶代数的基本工具。核心思想是：一个右伴随若保守并保持特定几何实现，则目标范畴可由左伴随产生的 monad 的代数恢复；对偶地，comonadic descent 由余单子和 Cech nerve 控制。

## 依赖前置知识

需要伴随、单子、simplicial objects、geometric realization、presentable $\infty$-categories、Cartesian fibration、descent、QCoh 和 stable $\infty$-categories。

## 35.1 $\infty$-范畴中的 monad

**定义 35.1.** 设 $C$ 为 $\infty$-范畴。一个 monad $T$ 由函子 $T:C\to C$、单位

$$
\eta:\operatorname{id}_C\to T
$$

和乘法

$$
\mu:T^2\to T
$$

组成，满足结合律和单位律的同伦相干版本。

**定义 35.2.** $T$-algebra 是对象 $X\in C$ 连同作用

$$
TX\to X
$$

并满足相干结合律和单位律。$T$-algebras 组成的 $\infty$-范畴记作

$$
\operatorname{Alg}_T(C).
$$

**命题 35.3.** 若 $F:C\rightleftarrows D:G$ 是伴随，则 $T=GF$ 是 $C$ 上的 monad。

**证明.** 单位 $\eta:\operatorname{id}_C\to GF$ 是伴随单位。乘法

$$
GFGF\xrightarrow{G\varepsilon F}GF
$$

由伴随余单位 $\varepsilon:FG\to\operatorname{id}_D$ 给出。结合律和单位律正是伴随三角恒等式的相干形式。$\square$

**定义 35.4.** 伴随 $F\dashv G$ 的 comparison functor 为

$$
K:D\to\operatorname{Alg}_{GF}(C),
$$

把 $Y\in D$ 送到 $GY$，其 $GF$-作用由

$$
GFGY\xrightarrow{G\varepsilon_Y}GY
$$

给出。

## 35.2 Split simplicial objects 与几何实现

**定义 35.5.** $C$ 中的 augmented simplicial object 是函子

$$
X_\bullet:\Delta_+^{op}\to C
$$

其中 $\Delta_+$ 包含增广对象 $[-1]$。记增广目标为 $X_{-1}$。

**定义 35.6.** augmented simplicial object 称为 split，若存在额外退化映射使它由 $X_{-1}$ 可收缩地生成。等价地，它在同伦相干意义下有 contracting homotopy。

**命题 35.7.** 若 $X_\bullet\to X_{-1}$ split，且几何实现存在，则自然映射

$$
|X_\bullet|\to X_{-1}
$$

为等价。

**证明.** Split 条件给出增广单纯对象到常值单纯对象 $X_{-1}$ 的同伦收缩。几何实现是从单纯对象到 $C$ 的 colimit，因此把同伦收缩送为等价。更具体地，额外退化给出 simplicial indexing category 上的终同伦，使 colimit 与 $X_{-1}$ 等同。$\square$

## 35.3 Barr-Beck-Lurie 单子性定理

**外部输入定理 35.8（Barr-Beck-Lurie）.** 设 $F:C\rightleftarrows D:G$ 是 presentable $\infty$-categories 之间的伴随，且 $G$ 保持适当几何实现。若：

1. $G$ 保守；
2. $G$ 保持所有 $G$-split simplicial objects 的几何实现；

则 comparison functor

$$
D\to\operatorname{Alg}_{GF}(C)
$$

是等价。此时称 $G$ monadic。

**命题 35.9.** 若 $G$ monadic，则 $G$ 保守。

**证明.** 若 $D\simeq\operatorname{Alg}_T(C)$ 且 $G$ 识别为遗忘函子 $\operatorname{Alg}_T(C)\to C$，则一个 $T$-algebra 态射是等价当且仅当其底层 $C$ 中态射是等价。这由代数结构在映射空间上的全子空间条件给出。因此 $G$ 反映等价，即保守。$\square$

**例子 35.10.** 环谱 $A$ 的模范畴 $\operatorname{Mod}_A$ 在 $\mathbf{Sp}$ 上 monadic：遗忘函子

$$
\operatorname{Mod}_A\to\mathbf{Sp}
$$

对应 monad $A\otimes-$。

**证明.** 自由-遗忘伴随为

$$
A\otimes-:\mathbf{Sp}\rightleftarrows\operatorname{Mod}_A:U.
$$

复合 monad 是 $A\otimes-$。模对象正是该 monad 的代数。保守性和几何实现保持性在谱模范畴中由底层谱逐点计算给出。$\square$

## 35.4 Comonadic descent

**定义 35.11.** 对伴随 $F:C\rightleftarrows D:G$，余单子为 $FG:D\to D$。若 comparison functor

$$
C\to\operatorname{Coalg}_{FG}(D)
$$

是等价，则称 $F$ comonadic。

**外部输入定理 35.12（comonadic Barr-Beck-Lurie）.** 若 $F$ 保守并保持适当 totalizations，且满足对偶的 split 条件，则 $F$ comonadic。

**定义 35.13.** 对态射 $f:U\to X$，Cech nerve 为增广单纯对象

$$
U_\bullet=U\times_X\cdots\times_XU\to X.
$$

若一个系数系统 $\mathcal D$ 满足

$$
\mathcal D(X)\simeq\operatorname{Tot}\bigl(\mathcal D(U_\bullet)\bigr),
$$

则称 $\mathcal D$ 对 $f$ 满足 descent。

**命题 35.14.** 若 $f^*:\mathcal D(X)\to\mathcal D(U)$ comonadic，则 $\mathcal D(X)$ 等价于 Cech nerve 上的 descent data 范畴。

**证明.** Comonadicity 说明 $\mathcal D(X)$ 等价于余单子 $f^*f_*$ 的 coalgebras。Cech nerve 的 cosimplicial 范畴

$$
\mathcal D(U)\rightrightarrows\mathcal D(U\times_XU)\triplearrows\cdots
$$

正是该余单子的 cobar construction。Coalgebras 的 $\infty$-范畴由该 cobar cosimplicial object 的 totalization 给出，因此得到 descent data 的范畴。$\square$

## 35.5 有效下降与忠实平坦下降

**外部输入定理 35.15.** 对合适拓扑中的覆盖 $f:U\to X$，quasi-coherent sheaves 满足 faithfully flat descent：

$$
\operatorname{QCoh}(X)\simeq\operatorname{Tot}\operatorname{QCoh}(U_\bullet).
$$

在 derived/spectral 语境中，flat、fpqc 或 fppf 条件需用 connective $E_\infty$-rings 的同伦群条件表述。

**命题 35.16.** 若 $A\to B$ 是 faithfully flat ordinary ring map，则 $A$-modules 可由带 cocycle condition 的 $B$-modules 恢复。

**证明.** Cech nerve 为

$$
B\rightrightarrows B\otimes_AB\triplearrows B\otimes_AB\otimes_AB\cdots.
$$

Faithfully flat descent 定理给出

$$
A\operatorname{-Mod}\simeq
\operatorname{Tot}\bigl(B\operatorname{-Mod}\rightrightarrows (B\otimes_AB)\operatorname{-Mod}\triplearrows\cdots\bigr).
$$

右侧对象就是 $B$-模 $M$ 连同两个拉回到 $B\otimes_AB$ 后的同构，以及在三重张量上的 cocycle condition。$\square$

## 35.6 本章小结

Barr-Beck-Lurie 定理把“范畴是否由单子代数恢复”的问题化为保守性和几何实现保持性。其对偶形式给出 comonadic descent：对象可以从覆盖上的对象及其 Cech 相容数据恢复。现代代数几何、Tannaka duality、QCoh 下降、模范畴和高阶代数中的许多重构定理都依赖这一范畴论机制。

## 练习

**练习 35.1.** 定义 $\infty$-范畴中的 monad。

**练习 35.2.** 说明伴随 $F\dashv G$ 如何产生 monad $GF$。

**练习 35.3.** 定义 comparison functor。

**练习 35.4.** 定义 split augmented simplicial object。

**练习 35.5.** 证明 split augmented simplicial object 的几何实现等价于增广目标。

**练习 35.6.** 陈述 Barr-Beck-Lurie 单子性定理。

**练习 35.7.** 证明 monadic 遗忘函子保守。

**练习 35.8.** 说明 $\operatorname{Mod}_A\to\mathbf{Sp}$ 对应 monad $A\otimes-$。

**练习 35.9.** 定义 comonadic functor。

**练习 35.10.** 定义 Cech nerve 和 descent data。

**练习 35.11.** 证明 comonadicity 蕴含 Cech descent。

**练习 35.12.** 写出 faithfully flat descent 中的 cocycle data。
