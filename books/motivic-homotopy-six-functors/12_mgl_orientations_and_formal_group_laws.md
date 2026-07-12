# 第十二章：Algebraic cobordism、orientations 与 formal group laws

## 本章目标

本章引入 algebraic cobordism spectrum `MGL`。它是 motivic homotopy theory 中对应拓扑复 cobordism `MU` 的对象，并通过 orientation 的泛性质控制 oriented motivic ring spectra。本章只证明 orientation 的形式后果，把 `MGL` 的构造、泛性质和 Hopkins-Morel 型定理作为外部输入。

## 依赖前置知识

需要 Thom spaces、Grassmannians、commutative motivic ring spectra、projective bundle formula、Chern classes、formal group laws、`H\mathbb Z`、`KGL` 和 stable motivic homotopy。

## 12.1 MGL 的构造口径

**外部输入定理 12.1.** 存在 algebraic cobordism spectrum

$$
MGL_S\in\operatorname{CAlg}(\mathbf{SH}(S)),
$$

由 tautological vector bundles 的 Thom spaces 组装而成，并带有 canonical orientation。

**依赖源.** Voevodsky 的 `MGL` 构造，Levine-Morel algebraic cobordism，Panin-Pimenov-Röndigs 的 universality theorem，Hoyois 的 Hopkins-Morel 型比较。

**定义 12.2.** 对 `E\in\operatorname{CAlg}(\mathbf{SH}(S))`，一个 orientation 是满足标准归一化条件的类

$$
c\in E^{2,1}(\mathbb P^\infty)
$$

其在 `\mathbb P^1` 上的限制给出 `E^{2,1}(\mathbb P^1)` 中的 canonical generator。具体模型中可等价地用 Thom classes for line bundles 表述。

**注 12.3.** Orientation 的定义依赖 projective bundle formula 的背景。若某个 ring spectrum 不满足相应公式，则不能仅凭一个类 `c` 建立完整 Chern class formalism。

## 12.2 泛性质

**外部输入定理 12.4（MG-12.4，MGL universality）.** 设
`S=\operatorname{Spec}(k)`，其中 `k` 为域；设 `E` 为交换
`\mathbb P^1`-ring spectrum。在 motivic stable homotopy category 中，赋值
`\varphi\mapsto\varphi(th^{MGL})` 给出自然双射

$$
\operatorname{Hom}_{\mathrm{CMon}(\operatorname{Ho}\mathbf{SH}(k))}
(MGL,E)
\cong
\{E\text{ 的 orientations}\}.
$$

因此这是同伦范畴中 monoid-map **集合**的分类，不是任意基上
`\operatorname{CAlg}` mapping **space** 的等价。

**精确来源与边界.** Panin--Pimenov--Röndigs, *A universality theorem for
Voevodsky's algebraic cobordism spectrum*, Theorem 2.3.1，
`https://arxiv.org/abs/0709.4116`。更一般 regular bases 的 published 版本和
现代 infinity-categorical refinement 必须分别核对，不能从上述集合双射推出。

**命题 12.5.** 若 `E` 已定向，则存在自然的 first Chern class

$$
c_1^E(L)\in E^{2,1}(X)
$$

对每个线丛 `L\to X` 定义。

**证明.** 线丛 `L` 由 classifying map `X\to\mathbb P^\infty` 分类，至少在满足标准 descent 和分类空间构造的语境中如此。将 orientation 类 `c\in E^{2,1}(\mathbb P^\infty)` 沿该 classifying map 拉回，得到 `c_1^E(L)`。自然性来自 classifying map 的自然性和 cohomology pullback 的函子性。`\square`

**命题 12.6.** 若 `E` 已定向，则 tensor product of line bundles 诱导 formal group law

$$
c_1^E(L\otimes M)=F_E(c_1^E(L),c_1^E(M)).
$$

**证明.** 张量积给出 classifying spaces 上的乘法映射

$$
\mathbb P^\infty\times\mathbb P^\infty\to\mathbb P^\infty.
$$

将 orientation 类沿该映射拉回，得到 `E^{*,*}(\mathbb P^\infty\times\mathbb P^\infty)` 中的幂级数表达。projective bundle formula 把该 cohomology 识别为两个 Chern class 变量上的形式幂级数环，因此得到 formal group law。结合律、交换律和单位律来自线丛张量积的对应性质。`\square`

## 12.3 Projective bundle formula 与 Chern classes

**外部输入定理 12.7（定向理论 package）.** 在 Panin--Pimenov--Röndigs
Section 1 的 oriented representable cohomology theory 口径中，若 `E` 已
定向，`V\to X` 为秩 `r` 向量丛，则 projective bundle map 给出

$$
E^{*,*}(\mathbb P(V))\simeq
\bigoplus_{i=0}^{r-1}E^{*-2i,*-i}(X)\cdot \xi^i,
$$

其中 `\xi=c_1^E(\mathcal O(1))`。

**定义 12.8.** 向量丛 `V` 的 Chern classes `c_i^E(V)` 由 projective bundle formula 和 splitting principle 定义，满足

$$
c^E(V)=1+c_1^E(V)+c_2^E(V)+\cdots.
$$

**命题 12.9.** 对线丛 `L`，总 Chern class 满足

$$
c^E(L)=1+c_1^E(L).
$$

**证明.** 线丛秩为一，projective bundle formula 中只有一次 Chern root。高于一阶的 Chern class 按 splitting principle 为零，因此总类为 `1+c_1^E(L)`。`\square`

**命题 12.10.** 对短正合列 `0\to V'\to V\to V''\to0`，若 `E` 的 Chern class formalism 满足 Whitney formula，则

$$
c^E(V)=c^E(V')c^E(V'').
$$

**证明.** 这是 orientation 所诱导 Chern class theory 的公理性后果。通过 splitting principle 可把短正合列拉回到分裂为线丛直和的情形；线丛情形由 first Chern classes 和 formal group law 控制。再由 splitting principle 的保守性下降回 `X`。`\square`

## 12.4 HZ、KGL 与 MGL 的比较

**外部输入定理 12.11（MG-12.11，Hopkins--Morel 型定理）.** 设 `S`
essentially smooth over a field，且该域的 characteristic exponent 为 `c`。
若 `a_1,a_2,\ldots` 是 Lazard ring 的正次数 generators，则 canonical map

$$
MGL/(a_1,a_2,\ldots)[1/c]
\longrightarrow H\mathbb Z[1/c]
$$

是 `\mathbf{SH}(S)` 中的等价。当 `c=1` 时无需反演。

**精确来源与边界.** Marc Hoyois, *From algebraic cobordism to motivic
cohomology*, Theorem 7.12，`https://arxiv.org/abs/1210.7182`。本书不去掉
正特征中的 `1/c`，也不把此定理推广到任意基概形。

**命题 12.12.** 若定理 12.11 的等价成立，则 `H\mathbb Z` 继承 additive formal group law。

**证明.** `H\mathbb Z` 的 orientation 由 `MGL` 的 universal orientation 沿商映射诱导。商掉 Lazard ring 中对应非加性高阶项的 generators 后，universal formal group law 变为 additive formal group law。因此 `H\mathbb Z` 的 first Chern class 满足

$$
c_1(L\otimes M)=c_1(L)+c_1(M).
$$

该结论依赖定理 12.7 的 Chern-class 坐标和定理 12.11 的具体商识别。
`\square`

**外部输入定理 12.13.** `KGL` 带有 multiplicative orientation，其 formal group law 在合适坐标下为 multiplicative formal group law。

**注 12.14.** `H\mathbb Z`、`KGL` 和 `MGL` 分别对应 Chow/motivic cohomology、K-theory 和 cobordism 层面的定向理论；它们之间的比较需要系数、周期性和完成/局部化假设。

## 12.5 Thom isomorphism

**定义 12.15.** 若 `E` 是 oriented motivic ring spectrum，向量丛 `V\to X` 的 Thom class 是类

$$
u_V\in E^{2r,r}(\operatorname{Th}(V))
$$

其中 `r=\operatorname{rank}V`，满足对纤维的标准归一化和对 pullback 的自然性。

**外部输入定理 12.16.** 对 oriented motivic ring spectrum `E`，Thom class 诱导 Thom isomorphism

$$
E^{a,b}(X)\simeq E^{a+2r,b+r}(\operatorname{Th}(V)).
$$

**命题 12.17.** Thom isomorphism 与 direct sum of vector bundles 相容。

**证明.** Direct sum formula 给出 `\operatorname{Th}(V\oplus W)` 与迭代 Thom space 的等价。Orientation 的 Thom classes 满足乘法性 `u_{V\oplus W}=u_V\cup u_W`。因此先对 `V` 再对 `W` 应用 Thom isomorphism，与对 `V\oplus W` 一次应用给出同一 cup product by Thom class。`\square`

## 12.6 本章小结

`MGL` 的本章泛性质是域上 motivic homotopy category 中 monoid-map 集合的
分类，不是未加说明的 infinity-categorical mapping-space 等价。Hopkins--Morel
比较限于 essentially smooth bases，并在正特征反演 characteristic exponent。
Orientation 的 projective bundle、Chern class 与 Thom 后果均在所声明的
oriented-cohomology package 内使用。

## 练习

**练习 12.1.** 写出 orientation 如何定义线丛的 first Chern class。

**练习 12.2.** 从线丛张量积推导 formal group law 的结合律来源。

**练习 12.3.** 解释 `MGL\to E` 与 orientation 的关系为何是泛性质。

**练习 12.4.** 比较 additive 和 multiplicative formal group law 的首项。

**练习 12.5.** 说明 Thom isomorphism 为什么需要 orientation。

**练习 12.6.** 陈述 projective bundle formula 并解释 `\xi` 的来源。

**练习 12.7.** 用 splitting principle 说明 Whitney formula 的证明策略。
