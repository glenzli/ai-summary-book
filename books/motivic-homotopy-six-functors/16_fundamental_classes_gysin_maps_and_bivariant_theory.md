# 第十六章：Fundamental classes、Gysin maps 与 bivariant theory

## 本章目标

本章介绍 motivic fundamental classes 和 bivariant theory。六操作提供 `f^!`，purity 提供 Thom twist，而 fundamental class 把几何态射转化为 cohomology operations。它是 Gysin maps、excess intersection、Riemann-Roch 和 quadratic refinements 的共同机制。

## 依赖前置知识

需要六操作、purity、Thom twists、local complete intersection morphisms、cotangent complex、Borel-Moore homology、orientation、intersection theory 和 stable motivic homotopy。

## 16.1 Fundamental class

**定义 16.1.** 对 smoothable lci separated morphism `f:X\to Y`，沿用
第六章的 virtual tangent class
`\tau_f=\langle L_f\rangle\in K(X)`。Motivic fundamental class 是六操作中
连接 Thom twist 与 exceptional pullback 的类；在单位对象上可写作

$$
\eta_f:\Sigma^{\tau_f}\mathbb 1_X\longrightarrow f^!\mathbb 1_Y.
$$

这里 angle-bracket 是 Deglise--Jin--Khan 的 virtual-tangent convention：
smooth 时 `\tau_f=[T_f]`，regular closed immersion 时
`\tau_f=-[N_f]`，若 `f=p\circ i` 则
`\tau_f=i^*[T_p]-[N_i]`。不能把它简写成 K-theory 中字面上的
`-[L_f]`；dualization 与 additive inverse 是不同操作。

**外部输入定理 16.2（smoothable lci fundamental classes）.** 在本书默认
六操作口径中，smoothable lci separated morphisms 有与 smooth maps 和
regular closed immersions 的标准类相容的 fundamental classes。它们满足
identity、composition 与 Tor-independent transverse base change。对非
Tor-independent Cartesian square，只有当原 morphism `f` 与拉回 morphism
`g` 都是 smoothable lci separated finite type，且 Paragraph 3.3.3 的
excess bundle 存在时，才有 excess formula。带 unital associative
commutative multiplication 的 motivic-spectrum coefficients 版本诱导相应
Gysin operations。

**依赖源与边界.** Deglise--Jin--Khan, *Fundamental classes in motivic
homotopy theory*, Definition 3.2.5（regular closed immersions）、Theorem
3.3.2（smoothable lci system）、Theorem 4.1.4（coefficients）与 Theorem
4.2.1（Gysin maps）；复合和 transverse base change 另见 Proposition
2.5.4。球谱基本类的 excess formula 见 Paragraph 3.3.3 与 Proposition
3.3.4；带 unital associative commutative multiplication 系数的版本见
Proposition 4.2.2，
`https://arxiv.org/abs/1805.05920`。该定理不为任意 morphism 构造
fundamental class；无修正 transverse base change 要求 Tor-independence，
而 arbitrary Cartesian square 不能仅凭“非横截”就套用 excess term。

**注 16.3.** 定义 16.1 是口径说明，不是构造。构造需要 deformation to the normal cone、six operations 和 purity 的相干。

## 16.2 Gysin maps

**定义 16.4.** 设 `E` 为 motivic ring spectrum。若 `f:X\to Y` 有 fundamental class，则它诱导 `E`-cohomology 或 Borel-Moore theory 上的 Gysin maps。典型形式由合成

$$
f^*E\longrightarrow \Sigma^{-\tau_f}f^!E
$$

及伴随得到。

**命题 16.5.** 若 fundamental classes 对复合相容，则 Gysin maps 对复合相容。

**证明.** 设 `X\xrightarrow{f}Y\xrightarrow{g}Z`。Gysin map 的构造由 `\eta_f`、`\eta_g`、六操作复合相干和 Thom twist 的加法性组成。若 fundamental classes 满足

$$
\eta_{gf}=\text{由 }\eta_f\text{ 与 }\eta_g\text{ 合成得到的类},
$$

则用 `gf` 一次构造 Gysin map，与先用 `f` 再用 `g` 构造的映射在每个组成结构态射上相同。由 functoriality 和相干性，两者相等。`\square`

**命题 16.6.** 若 `f` smooth 且 separated，则 smooth purity 给出的 class
与第六章的 `f^!\simeq\Sigma^{T_f}f^*` 相容。

**证明.** smooth morphism 的 cotangent complex 是向量丛 `L_f`，其对偶为相对切丛 `T_f`。Smooth purity 已给出 `f^!` 与 Thom twist 后的 `f^*` 的等价。Fundamental class 在 smooth 情形应恢复该 purity equivalence；这正是定理 16.2 中复合和 purity normalization 的一部分。`\square`

## 16.3 Bivariant theory

**定义 16.7.** 对 motivic ring spectrum `E` 和态射 `f:X\to S`，定义 bivariant group 的一种六操作表达为

$$
E^{a,b}(X/S)=
\pi_0\operatorname{Map}_{\mathbf{SH}(X)}
(\mathbb 1_X,\Sigma^{a,b}f^!E_S).
$$

该记号把 cohomology、Borel-Moore homology 和 operational classes 放入同一框架。

**命题 16.8.** 若 `f=\operatorname{id}_S`，则 bivariant group 恢复 `E`-cohomology of `S`。

**证明.** 对恒等态射，`f^!=\operatorname{id}`。定义 16.7 变为

$$
\pi_0\operatorname{Map}_{\mathbf{SH}(S)}
(\mathbb 1_S,\Sigma^{a,b}E_S),
$$

即 `E^{a,b}(S)`。`\square`

**外部输入推论 16.9.** 在定理 16.2 的 smoothable-lci、系数与
Tor-independence/excess 假设下，motivic fundamental classes 赋予
`E`-bivariant groups 以乘法、proper pushforward 和 Gysin pullback；
Theorem 4.2.1 给出 Gysin functoriality 与 transverse base change，
Propositions 3.3.4、4.2.2 在各自假设下给出 excess 修正。这里的 proper
pushforward 来自六操作，而非 fundamental class 单独产生；Proposition
4.2.2 的 push-pull 等式还要求方块的两条竖边 proper。

## 16.4 Deformation to the normal cone

**定义 16.10.** 对闭嵌入 `i:Z\hookrightarrow X`，令 `\mathcal I\subset\mathcal O_X`
为其理想层，并记 `\widetilde{X\times\{0\}}` 为 `X\times\{0\}` 在相应 blow-up
中的严格变换。定义 deformation to the normal cone 为

$$
D_ZX:=
\operatorname{Bl}_{Z\times\{0\}}(X\times\mathbb A^1)
\setminus\widetilde{X\times\{0\}}
\longrightarrow\mathbb A^1.
$$

在 `\mathbb G_m\subset\mathbb A^1` 上，该族规范同构于
`X\times\mathbb G_m`；其 `0`-纤维规范同构于 normal cone

$$
C_ZX:=\operatorname{Spec}_Z
\left(\bigoplus_{n\geq0}\mathcal I^n/\mathcal I^{n+1}\right).
$$

若 `i` regular，则 `\mathcal I/\mathcal I^2` locally free，且规范映射
`\operatorname{Sym}_{\mathcal O_Z}(\mathcal I/\mathcal I^2)\to
\bigoplus_{n\geq0}\mathcal I^n/\mathcal I^{n+1}` 是同构；因此在 regular
情形，特殊纤维识别为法丛 `N_{Z/X}` 的总空间。

**外部输入定理 16.11.** Regular closed immersion 的 motivic fundamental
class 由 deformation to the normal cone、specialization 和 Thom
isomorphism 构造，并与 specialization maps 相容。

**精确来源.** Deglise--Jin--Khan, Section 3.2，尤其 Definition 3.2.5，同上
URL。一般 smoothable lci class 再由 Theorem 3.3.2 通过 smooth factorization
扩张。

**命题 16.12.** 若 `i` 是 regular immersion，则 deformation 的特殊纤维给出 Thom twist 所需的法丛。

**证明.** Regular immersion 局部由 regular sequence 定义，其 normal cone 等于 normal bundle。Deformation to the normal cone 的特殊纤维为 `C_ZX`，因此在 regular 情形为 `N_{Z/X}`。Thom twist 正是由该向量丛产生。`\square`

## 16.5 Excess intersection

**外部输入定理 16.13（Excess intersection formula）.** 固定基概形 `S`，设

$$
\begin{array}{ccc}
X'&\xrightarrow{g}&Y'\\
\downarrow v&&\downarrow u\\
X&\xrightarrow{f}&Y
\end{array}
$$

是 `S` 上 s-schemes 的 Cartesian square；这里 s-morphism 指 separated
finite-type morphism，s-scheme over `S` 指结构态射为 s-morphism 的
`S`-scheme。假设 `f` 与 `g` 都是 smoothable lci s-morphisms，且 `\xi` 是
定义 16.15 的 excess bundle。记 `\Delta^*` 为该方块诱导的 bivariant
change-of-base map。则在 canonical K-theory identification
`v^*\tau_f\simeq\tau_g-[\xi]` 下，球谱 fundamental classes 满足

$$
\Delta^*(\eta_f)\simeq e(\xi)\mathbin{:}\eta_g.
$$

这里 `H(-/-;-)` 表示球谱所表示的 twisted bivariant group，两边都属于
`H(X'/Y';v^*\tau_f)`；这是 Deglise--Jin--Khan Proposition 3.3.4。若
`E\in\mathbf{SH}(S)` 还带 unital associative commutative multiplication，
则在相同的 virtual-twist 记号下有

$$
\Delta^*(\eta_f^E)\simeq e(\xi;E)\mathbin{:}\eta_g^E
\quad\text{in }E(X'/Y';v^*\tau_f),
$$

这是 Proposition 4.2.2。若还要使用该命题中的 proper push-pull 公式，必须
另加 `u`、`v` proper。Tor-independent 时法丛单射为同构，`\xi=0`，从而恢复
无修正 transverse base change。

**注 16.14.** Deglise--Jin--Khan Definition 4.2.5 可在只假设 `f` 为
smoothable lci 的 Cartesian square 上定义 refined fundamental class；但把该
refined class 识别为 `e(\xi)\mathbin{:}\eta_g` 的 Proposition 4.2.2/4.2.6(ii)
还要求 `g` 也是 smoothable lci，并使用 Paragraph 3.3.3 的 excess bundle。
因此 refined base change 对任意方块的可定义性，不能被改写成任意方块都有
excess-bundle formula。

**定义 16.15（本章采用的 excess bundle）.** 在定理 16.13 的方块中，选择
smooth factorization `f=p\circ i`，其中 `i:X\hookrightarrow P` 为 regular
closed immersion、`p:P\to Y` smooth。拉回后 `g=q\circ k`，其中
`k:X'\hookrightarrow P\times_Y Y'` 仍为 regular closed immersion、`q`
smooth。Paragraph 3.3.3 给出法丛的 canonical monomorphism

$$
N_k\longrightarrow v^*N_i;
$$

其 locally free cokernel
`\xi=\operatorname{coker}(N_k\to v^*N_i)` 称为 excess bundle，修正项为
Euler class `e(\xi)`。若 `g` 不是 smoothable lci、拉回后的 `k` 不是 regular，
或该 cokernel 不是向量丛，则本定义与定理 16.13 均不适用。

**命题 16.16.** 在定理 16.13 的全部假设下，若 excess bundle 为零向量丛，
则 excess formula 退化为 Gysin maps 与 base change 交换。

**证明.** 零向量丛的 Euler class 为单位。Excess formula 中的修正项乘以单位，不改变映射。因此得到无修正的交换公式。`\square`

## 16.6 Riemann-Roch 型公式

**定义 16.17.** 若 `\varphi:E\to F` 是 oriented motivic ring spectra 之间的 morphism，Riemann-Roch 问题询问 `\varphi` 与 Gysin maps 是否交换；通常需要 Todd class 修正。

**高级外部输入 16.18（P1）.** 在另行指定的 oriented coefficient spectra、
可定向 morphisms 与 Todd class 假设下，orientation 改变导致的 Gysin 比较
满足 Riemann--Roch 型公式。该结果不参与第 16.2--16.16 节的 P0
fundamental-class 主线；具体应用须另补采用版本的 locator。

**命题 16.19.** 若 `\varphi` 严格保持 orientation，则 Todd 修正为单位。

**证明.** Todd class 衡量两个 orientation 对 Thom classes 的差异。若 `\varphi` 把 `E` 的 Thom class 送到 `F` 的 Thom class，则差异类为单位。因此 Riemann-Roch 比较中不出现非平凡修正。`\square`

## 16.7 本章小结

Fundamental classes 把六操作和几何交理论连接起来。P0 构造严格限于
smoothable lci morphisms；无修正 base change 只在 Tor-independent
方块成立。非 Tor-independent 方块只有在原 morphism 与拉回 morphism 都是
smoothable lci 且 excess bundle 存在时，才能使用 Propositions 3.3.4、4.2.2
的 excess Euler class；其余方块不在该公式范围内。Theorem 4.2.1 定位 Gysin
maps，而不是 excess theorem 的替代标签。Riemann--Roch 比较保留为 P1。

## 练习

**练习 16.1.** 写出 fundamental class 的形式类型。

**练习 16.2.** 解释 `\tau_f=\langle L_f\rangle` 在 smooth、regular closed
和一般 smoothable lci factorization 三种情形的公式，并说明为何不能写成
字面等式 `\tau_f=-[L_f]`。

**练习 16.3.** 证明恒等态射的 bivariant group 恢复 cohomology。

**练习 16.4.** 说明 excess formula 为什么不是普通 base change。

**练习 16.5.** 解释 Todd class 衡量的是什么差异。

**练习 16.6.** 定义 deformation to the normal cone。

**练习 16.7.** 说明 excess bundle 为零时 excess formula 如何简化。
