# 第三十五章：Barr-Beck-Lurie 单子性、余单子下降与 descent

一个伴随 $F\dashv U$ 产生单子 $UF$，但比较函子 $\mathcal D\to\operatorname{Alg}_{UF}(\mathcal C)$ 并不自动是等价。Barr--Beck--Lurie 定理给出精确条件：$U$ 必须保守，并保存由 $U$-split simplicial objects 产生的几何实现。对偶的 comonadic 形式把对象重建为 Cech nerve 上 descent data 的全化。这个机制同时解释 faithfully flat descent、模范畴重构和许多仿射性判据。

本章使用伴随、单子、simplicial objects、geometric realization 与 presentable/stable $\infty$-范畴。我们会明确区分“保存全部几何实现”和定理真正要求的 split 类，并把 totalization 的收敛、保守性与可交换极限条件逐项列出。

## 35.1 $\infty$-范畴中的 monad

**定义 35.1.** 设 $C$ 为 $\infty$-范畴。Monad 是幺半 $\infty$-范畴 $\operatorname{Fun}(C,C)$（乘法为函子复合）中的结合代数对象。展开后，它含函子 $T:C\to C$、单位

$$
\eta:\operatorname{id}_C\to T
$$

和乘法

$$
\mu:T^2\to T
$$

以及所有高阶结合与单位相干。只给出 $\eta,\mu$ 而不指定这些相干数据，不能在任意 $\infty$-范畴中完整定义 monad。

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

给定函子 $G:D\to C$，$D$ 中单纯对象 $Y_\bullet$ 称为 $G$-split，若 $G(Y_\bullet)$ 可扩张为 $C$ 中的 split augmented simplicial object。

**命题 35.7.** 若 $X_\bullet\to X_{-1}$ split，且几何实现存在，则自然映射

$$
|X_\bullet|\to X_{-1}
$$

为等价。

**证明.** Split 条件给出增广单纯对象到常值单纯对象 $X_{-1}$ 的同伦收缩。几何实现是从单纯对象到 $C$ 的 colimit，因此把同伦收缩送为等价。更具体地，额外退化给出 simplicial indexing category 上的终同伦，使 colimit 与 $X_{-1}$ 等同。$\square$

## 35.3 Barr-Beck-Lurie 单子性定理

**外部输入定理 35.8（Barr--Beck--Lurie）.** 设 $F:C\rightleftarrows D:G$ 是 $\infty$-范畴之间的伴随。假设 $D$ 有所有 $G$-split 单纯对象的几何实现，且：

1. $G$ 保守；
2. $G$ 保持这些几何实现；

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

**外部输入定理 35.12（comonadic Barr--Beck--Lurie）.** 设 $F:C\rightleftarrows D:G$。若 $C$ 有所有 $F$-split cosimplicial objects 的 totalization，$F$ 保守并保持这些 totalizations，则 $F$ comonadic。

**定义 35.13.** 对态射 $f:U\to X$，Cech nerve 为增广单纯对象

$$
U_\bullet=U\times_X\cdots\times_XU\to X.
$$

若一个系数系统 $\mathcal D$ 满足

$$
\mathcal D(X)\simeq\operatorname{Tot}\bigl(\mathcal D(U_\bullet)\bigr),
$$

则称 $\mathcal D$ 对 $f$ 满足 descent。

**命题 35.14.** 若 $f^*:\mathcal D(X)\to\mathcal D(U)$ comonadic，并且系数系统 $\mathcal D$ 的 Beck--Chevalley 等价把余单子 $f^*f_*$ 的 cobar construction 逐层识别为 $\mathcal D(U_\bullet)$，则

$$
\mathcal D(X)\simeq\operatorname{Tot}\mathcal D(U_\bullet).
$$

**证明.** Comonadicity 说明 $\mathcal D(X)$ 等价于余单子 $f^*f_*$ 的 coalgebras。由附加的 Beck--Chevalley 假设，Cech nerve 的 cosimplicial 范畴

$$
\mathcal D(U)\rightrightarrows\mathcal D(U\times_XU)\triplearrows\cdots
$$

逐层等价于该余单子的 cobar construction。Coalgebras 的 $\infty$-范畴由 cobar totalization 给出，因此得到结论。Comonadicity 本身并不自动把任意几何 Cech nerve 识别为该 cobar 图形。$\square$

## 35.5 有效下降与忠实平坦下降

**外部输入定理 35.15.** 若 $f:U\to X$ 是 quasi-compact quasi-separated schemes 的 fpqc 覆盖，则 quasi-coherent complexes 满足 faithfully flat descent：

$$
\operatorname{QCoh}(X)\simeq\operatorname{Tot}\operatorname{QCoh}(U_\bullet).
$$

对 connective $E_\infty$-rings 的 faithfully flat 映射 $A\to B$，相应仿射陈述以导出张量幂 $B^{\otimes_A(n+1)}$ 形成的 Amitsur complex 表示；faithful flatness 要求 $\pi_0A\to\pi_0B$ faithfully flat 且 $\pi_*A\otimes_{\pi_0A}\pi_0B\simeq\pi_*B$。

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

## 35.6 单子性与下降的形式稳定性

**命题 35.17.** 伴随 $F:C\rightleftarrows D:G$ 的 comparison functor $K:D\to\operatorname{Alg}_{GF}(C)$ 满足

$$
U\circ K\simeq G,
$$

其中 $U:\operatorname{Alg}_{GF}(C)\to C$ 是遗忘函子。

**证明.** 按定义 35.4，$K(Y)$ 的底层对象就是 $GY$，其代数作用由 $G\varepsilon_Y$ 给出。因此遗忘代数结构后得到 $GY$。这对 $Y$ 自然，故 $U\circ K\simeq G$。$\square$

**命题 35.18（单子性在等价下不变）.** 设 $E:D'\simeq D$ 为 $\infty$-范畴等价。若 $G:D\to C$ monadic，则 $G\circ E:D'\to C$ monadic。

**证明.** 取 $G$ 的左伴随 $F$。则 $G\circ E$ 的左伴随为 $E^{-1}\circ F$。二者产生的 monad 都等价于 $GF$。Comparison functor

$$
D'\to\operatorname{Alg}_{GF}(C)
$$

等于 $E$ 后接 $D\simeq\operatorname{Alg}_{GF}(C)$ 的 comparison equivalence，因此是等价。故 $G\circ E$ monadic。$\square$

**命题 35.19.** 恒等覆盖的 Cech descent 是恒等命题：若 $f=\operatorname{id}_X$，则

$$
\mathcal D(X)\simeq\operatorname{Tot}\bigl(\mathcal D(X)\rightrightarrows\mathcal D(X)\triplearrows\cdots\bigr).
$$

**证明.** 恒等态射的 Cech nerve 是常值单纯对象 $X_\bullet=X$。因此 $\mathcal D(X_\bullet)$ 是常值 cosimplicial $\infty$-范畴。常值 cosimplicial 对象的 totalization 等于其常值项，因为终锥由恒等相容数据给出。故得到所需等价。$\square$

## 35.7 由单子与余单子重建

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

**练习 35.13.** 证明 comparison functor 与遗忘函子复合后等于右伴随。

**练习 35.14.** 证明 monadicity 在源范畴等价替换下保持。

**练习 35.15.** 证明恒等覆盖的 Cech descent 退化为恒等。
