# 第五十一章：范畴 Galois 理论、Descent 与有效下降

沿态射 $p:E\to B$ 拉回对象会产生带 cocycle 的 descent data，但并非每份数据都能下降为 $B$ 上对象。若比较函子到 descent-data 范畴是等价，称 $p$ 为有效下降态射；Barr--Beck 把这一条件与拉回函子的单子性联系起来。范畴 Galois 理论进一步把正规扩张、automorphism groupoids 与覆盖分类放入同一框架，并在 Galois categories 中恢复 profinite group actions。

本章使用 pullback、regular/effective epimorphisms、monads、Cech nerves、torsors 和 topoi。有效下降会按对象类别分别讨论；faithfully flat、open cover 与 regular epi 的结论只在相应几何或正合假设下调用。

## 51.1 Descent datum

**定义 51.1.** 设 $p:E\to B$ 为有 pullback 的范畴 $\mathcal C$ 中态射。对象 $X\to E$ 的 descent datum 是同构

$$
\theta:\pi_1^*X\xrightarrow{\sim}\pi_2^*X
$$

定义在 $E\times_BE$ 上，并满足 $E\times_BE\times_BE$ 上的 cocycle 条件。

**定义 51.2.** Descent category $\operatorname{Desc}(p)$ 的对象为带 descent datum 的 $E$ 上对象，态射为与 descent datum 相容的态射。

**命题 51.3.** 拉回函子

$$
p^*:\mathcal C_{/B}\to\operatorname{Desc}(p)
$$

自然定义。

**证明.** 给定 $Y\to B$，拉回得 $p^*Y=E\times_BY\to E$。在 $E\times_BE$ 上，两种拉回 $\pi_1^*p^*Y$ 与 $\pi_2^*p^*Y$ 都典范同构于 $E\times_BE\times_BY$，故有典范 descent 同构。三重交上的 cocycle 条件由 pullback 的自然结合性给出。$\square$

## 51.2 有效下降

**定义 51.4.** 态射 $p:E\to B$ 称为 effective descent morphism，若

$$
p^*:\mathcal C_{/B}\to\operatorname{Desc}(p)
$$

为范畴等价。

**命题 51.5.** 若 $p$ 为 isomorphism，则 $p$ 是 effective descent morphism。

**证明.** 若 $p$ 为同构，则 $E\times_BE\cong E$，descent datum 只有恒等相容性。拉回 $p^*:\mathcal C_{/B}\to\mathcal C_{/E}$ 是 slice 范畴等价，因此也是到 descent category 的等价。$\square$

**外部输入定理 51.6.** 在 Barr-exact category 中，每个 regular epimorphism 都是 effective descent morphism。特别地，Grothendieck topos 中每个 epimorphism 都是 regular epimorphism，因而是 effective descent morphism。仅假设范畴 regular 时，regular epimorphism 一般只保证 descent 的保守部分，不足以推出有效性。

## 51.3 单子性判别

**命题 51.7.** 设 $\mathcal C$ 有 pullback。后复合函子

$$
\Sigma_p:\mathcal C_{/E}\to\mathcal C_{/B}
$$

左伴随于 $p^*$，相应 descent monad 是

$$
T=p^*\Sigma_p
$$

在 $\mathcal C_{/E}$ 上的单子。其 Eilenberg--Moore 范畴典范等价于 $\operatorname{Desc}(p)$；因此 $p$ 是 effective descent morphism 当且仅当右伴随 $p^*$ 是 monadic。

**证明.** 伴随 $\Sigma_p\dashv p^*$ 给出比较函子

$$
K:\mathcal C_{/B}\longrightarrow\operatorname{Alg}_T(\mathcal C_{/E}).
$$

把一个 $T$-代数的结构映射沿 kernel pair $E\times_BE\rightrightarrows E$ 展开，正得到定义 51.1 的同构与 cocycle 条件；反过来，descent datum 也唯一确定该代数结构。这个对应在态射上同样成立，并给出典范等价

$$
\operatorname{Alg}_T(\mathcal C_{/E})\simeq\operatorname{Desc}(p).
$$

在此等价下，$K$ 正对应定义 51.3 的 descent 比较函子。因此，$p^*$ monadic，即 $K$ 为等价，当且仅当 $p$ 是 effective descent morphism。$\square$

## 51.4 Galois 结构

**定义 51.8.** 一个范畴 Galois 结构是资料

$$
\Gamma=(\mathcal C,\mathcal X,I,H,\eta,\varepsilon,\mathcal E,\mathcal F),
$$

其中 $I\dashv H$ 是伴随

$$
I:\mathcal C\rightleftarrows\mathcal X:H
$$

而 $\mathcal E\subseteq\operatorname{Mor}(\mathcal C)$、$\mathcal F\subseteq\operatorname{Mor}(\mathcal X)$ 是包含同构、对复合和 pullback 封闭的 extension 类，并满足 $I(\mathcal E)\subseteq\mathcal F$、$H(\mathcal F)\subseteq\mathcal E$。还要求所需 pullback 存在。称该结构 admissible，若对每个 $B\in\mathcal C$，诱导到 extension-slice 的右伴随全忠实。

**定义 51.9.** Extension $p:E\to B$ 称为 trivial covering，若单位自然性方块

$$
\begin{array}{c}
E\xrightarrow{\eta_E}HIE\\
\downarrow p\qquad\downarrow HIp\\
B\xrightarrow{\eta_B}HIB
\end{array}
$$

是 pullback。称 $p$ 为 covering，若存在属于 $\mathcal E$ 的 effective descent morphism $q:B'\to B$，使 $q^*p$ 为 trivial covering。

**命题 51.10.** 在 admissible Galois 结构中，trivial coverings 在沿 $\mathcal E$ 中态射的 pullback 下稳定。

**证明.** Admissibility 给出单位自然性方块与 extension-slice 拉回函子的 Beck--Chevalley 相容性。将定义 51.9 的 pullback 方块再沿 $\mathcal E$ 中态射拉回，pullback 的粘合引理说明所得单位方块仍为 pullback，故拉回后的 extension 仍 trivial。没有 admissibility 时，这一稳定性不由定义自动推出。$\square$

## 51.5 正规扩张与 Galois groupoid

**定义 51.11.** Covering $p:E\to B$ 称为 normal，若 $p^*p$ 是 trivial covering，并且 $p$ 满足有效下降。

**定义 51.12.** Normal extension $p:E\to B$ 的 Galois groupoid 是 kernel pair

$$
E\times_BE\rightrightarrows E
$$

在 Galois 结构下反射得到的内部 groupoid。

**外部输入定理 51.13（范畴 Galois 基本定理）.** 设 $\Gamma$ 是 admissible Galois 结构，并满足 extension 类所需的 pullback 与复合闭包公理。若 $p:E\to B$ 是 monadic extension，则被 $p$ 分裂的 coverings 所成范畴，等价于 Galois groupoid $\operatorname{Gal}_\Gamma(p)$ 上适当的内部离散纤维化范畴，也可等价地表述为该 groupoid 的内部作用范畴。若 $p$ 还是 normal extension，这个 groupoid 正是由 $p$ 的 kernel pair 经反射得到的定义 51.12 中的 Galois groupoid。

## 51.6 经典例子

**命题 51.14.** 设 $L/K$ 为有限 Galois 扩张，$G=\operatorname{Gal}(L/K)$。$K$-向量空间经标量扩张到 $L$ 后的 descent data，等价于 $L$-向量空间上的半线性 $G$-作用。

**证明.** 对有限 Galois 扩张，

$$
L\otimes_KL\cong\prod_{\sigma\in G}L
$$

其中 $G=\operatorname{Gal}(L/K)$。因此在 $L\otimes_KL$ 上给 descent 同构等价于给每个 $\sigma$ 的半线性自同构；三重张量积上的 cocycle 条件正是群作用条件。$\square$

**命题 51.15.** Effective descent morphism 在同构替换下不变：若 $e:E'\xrightarrow{\sim}E$ 为 $\mathcal C_{/B}$ 中同构，则 $p:E\to B$ effective descent 当且仅当 $pe:E'\to B$ effective descent。

**证明.** 同构 $e$ 诱导 slice 范畴等价

$$
e^*:\mathcal C_{/E}\simeq\mathcal C_{/E'}
$$

并且由 pullback 与同构的相容性诱导 descent categories 的等价

$$
\operatorname{Desc}(p)\simeq\operatorname{Desc}(pe).
$$

在这些等价下，两个比较函子 $p^*$ 与 $(pe)^*$ 相互对应。因此其中一个为等价当且仅当另一个为等价。$\square$

## 51.7 何时下降数据是有效的

Descent 把对象的局部数据和 cocycle 条件组织成 descent category；effective descent 要求这些数据真正来自全局对象。Barr-Beck 把有效下降与单子性联系起来。范畴 Galois 理论进一步把 covering、normal extension 和 automorphism groupoid 抽象化，统一了经典 Galois 理论、torsor 理论和 topos 中的覆盖下降。

## 练习

**练习 51.1.** 定义 descent datum。

**练习 51.2.** 定义 descent category。

**练习 51.3.** 构造 $p^*:\mathcal C_{/B}\to\operatorname{Desc}(p)$。

**练习 51.4.** 定义 effective descent morphism。

**练习 51.5.** 证明同构是 effective descent morphism。

**练习 51.6.** 陈述 topos 中 epimorphism 的有效下降性质。

**练习 51.7.** 用 monadicity 推出 effective descent。

**练习 51.8.** 定义范畴 Galois 结构。

**练习 51.9.** 定义 trivial covering 和 covering。

**练习 51.10.** 证明 trivial covering 在 pullback 下稳定。

**练习 51.11.** 定义 normal extension。

**练习 51.12.** 定义 Galois groupoid。

**练习 51.13.** 陈述范畴 Galois 基本定理，并说明其中 monadicity 的作用。

**练习 51.14.** 解释有限 Galois 扩张中的 descent datum 与群作用。

**练习 51.15.** 证明 effective descent morphism 在覆盖对象同构替换下不变。
