# 第五十一章：范畴 Galois 理论、Descent 与有效下降

## 本章目标

本章系统化 descent 的普通范畴论版本。有效下降研究对象能否由覆盖上的对象和 cocycle 数据恢复；范畴 Galois 理论把覆盖、单子性、正规扩张和 automorphism groupoid 统一为抽象 Galois 对应。它与 Barr-Beck、topos、torsors、Galois categories 和代数几何中的 faithfully flat descent 直接相连。

## 依赖前置知识

需要 pullback、regular epimorphism、monads、Barr-Beck、Cech nerve、fibered categories、effective epimorphism、topos、torsors 和 group actions。

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

**外部输入定理 51.6.** 在 regular category 中，regular epimorphisms 在合适条件下是 effective descent morphisms；在 Grothendieck topos 中，epimorphisms 是 effective descent morphisms。

## 51.3 单子性判别

**命题 51.7.** 若 $p^*:\mathcal C_{/B}\to\mathcal C_{/E}$ monadic，且其 comparison category 等价于 $\operatorname{Desc}(p)$，则 $p$ 为 effective descent morphism。

**证明.** Monadicity 给出

$$
\mathcal C_{/B}\simeq\operatorname{Alg}_T(\mathcal C_{/E})
$$

其中 $T=p^*p_*$ 或相应 descent monad。若 $\operatorname{Alg}_T(\mathcal C_{/E})\simeq\operatorname{Desc}(p)$，则复合给出 $\mathcal C_{/B}\simeq\operatorname{Desc}(p)$，这正是 effective descent。$\square$

## 51.4 Galois 结构

**定义 51.8.** 一个范畴 Galois 结构通常由 adjunction

$$
I:\mathcal C\rightleftarrows\mathcal X:H
$$

及两类 fibrations 或 extensions 组成，满足 pullback 稳定和反射相容条件。$\mathcal X$ 可理解为“trivial objects”的范畴。

**定义 51.9.** Extension $p:E\to B$ 称为 trivial covering，若由 $\mathcal X$ 中对象经 $H$ 和 pullback 得到；称为 covering，若存在 effective descent morphism $q$ 使 $q^*p$ trivial。

**命题 51.10.** Trivial covering 在 pullback 下稳定。

**证明.** Trivial covering 按定义由反射子范畴中的对象沿某态射 pullback 得到。再次 pullback 时，由 pullback 的粘合性质，复合 pullback 仍是同一反射对象沿复合态射的 pullback。因此仍 trivial。$\square$

## 51.5 正规扩张与 Galois groupoid

**定义 51.11.** Covering $p:E\to B$ 称为 normal，若 $p^*p$ 是 trivial covering，并且 $p$ 满足有效下降。

**定义 51.12.** Normal extension $p:E\to B$ 的 Galois groupoid 是 kernel pair

$$
E\times_BE\rightrightarrows E
$$

在 Galois 结构下反射得到的内部 groupoid。

**外部输入定理 51.13.** 在合适 Galois 结构中，normal extensions over $B$ 与相应 Galois groupoids 的 actions 之间存在等价。

## 51.6 经典例子

**命题 51.14.** 有限 Galois 扩张 $L/K$ 的 descent datum 等价于带 $\operatorname{Gal}(L/K)$-作用的 $L$-对象。

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

## 51.7 本章小结

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

**练习 51.13.** 陈述 normal extensions 与 groupoid actions 的关系。

**练习 51.14.** 解释有限 Galois 扩张中的 descent datum 与群作用。

**练习 51.15.** 证明 effective descent morphism 在覆盖对象同构替换下不变。
