# 附录 C：Pointed presentable categories、stabilization 与 symmetric monoidal spectra

## 本附录目标

本附录补足从 `\mathbf H(S)` 到 `\mathbf{SH}(S)` 的一般范畴论。Motivic 稳定化不是形式上写一个 `T^{-1}`，而是一个带泛性质的 presentable symmetric monoidal stabilization。

## 依赖前置知识

需要 pointed objects、smash product、cofiber、suspension、spectrum objects、presentable symmetric monoidal infinity-categories 和 localization。

## C.1 Pointed objects

**定义 C.1.** 若 `\mathcal C` 有终对象 `*`，pointed objects 范畴为 under-category

$$
\mathcal C_*=\mathcal C_{*/}.
$$

对象是带基点 `*\to X` 的对象。

**命题 C.2.** 若 `\mathcal C` presentable，则 `\mathcal C_*` presentable，且有左伴随

$$
(-)_+:\mathcal C\rightleftarrows\mathcal C_*:U.
$$

**证明.** Presentable infinity-category 的 under-category presentable。遗忘函子 `U` 保持极限，其左伴随把 `X` 送到 `X\amalg *`，基点为第二个分量。`\square`

**定义 C.3.** 在 pointed category 中，cofiber of `A\to B` 定义为 pushout

$$
B/A=B\coprod_A *.
$$

Suspension 定义为 `\Sigma A=* \coprod_A *`。

## C.2 Smash product

**外部输入定理 C.4（pointed smash product）.** 若 `\mathcal C` 是
Cartesian symmetric monoidal presentable infinity-category，且 Cartesian
product 分变量保持小余极限（例如 `\mathcal C` 是 infinity-topos），则
`\mathcal C_*` 有 presentably symmetric monoidal smash product

$$
X\wedge Y=(X\times Y)/(X\vee Y),
$$

其中 `X\vee Y=X\amalg_*Y`。

单位是 `S^0=*_+`，且 `-\wedge-` 分变量保持小余极限。公式给出底层
二元函子；结合、交换和单位的全部高阶相干属于 pointed symmetric
monoidal construction 的结论，而不是由单个商对象公式自动得到。

**外部来源与边界.** Lurie, *Higher Algebra*, Proposition 4.8.1.15 给
`\operatorname{Pr}^{L}` 对称幺半结构，Example 4.8.1.21 给出
`\mathcal C\otimes\mathcal S_*\simeq\mathcal C_*`，Proposition 4.8.2.11
与 Remark 4.8.2.14 分别给出 pointed localization 和
`\mathcal S_*` 上的 smash product。把 `\mathcal C` 的 Cartesian
commutative-algebra structure 与 `\mathcal S_*` 张量，便得到上述
`\mathcal C_*` 的结构；商公式由两变量保余极限性从 pointed spaces 的
公式延拓。Motivic 情形还需 `\mathbb A^1`-局部化与乘积相容，统一并入
C.12。若 Cartesian product 不分变量保持余极限，上述商仍可逐对象形成，
但不能据此断言得到 presentably symmetric monoidal structure。

**命题 C.5（suspension 是 circle smash）.** 在 C.4 的假设下，对
`Y\in\mathcal C_*` 有自然等价

$$
\operatorname{cofib}(Y\to *)\simeq\Sigma Y
\simeq S^1\wedge Y,
$$

其中 pointed space `S^1=\Delta^1/\partial\Delta^1` 通过
`\mathcal C_*\simeq\mathcal C\otimes\mathcal S_*` 的 canonical
`\mathcal S_*`-tensoring 作用在 `\mathcal C_*` 上。

**证明.** Cofiber 定义为 `*\coprod_Y *`，这正是 suspension 的定义。在
pointed spaces 中

$$
S^1\simeq\operatorname{cofib}(S^0\longrightarrow *).
$$

由 C.4，`-\wedge Y` 保持小余极限，`S^0` 是 smash 单位，且
`*\wedge Y\simeq *`。所以

$$
S^1\wedge Y
\simeq\operatorname{cofib}(S^0\wedge Y\longrightarrow *\wedge Y)
\simeq\operatorname{cofib}(Y\longrightarrow *)
\simeq\Sigma Y.
$$

全部等价对 `Y` 自然。`\square`

## C.3 Spectrum objects

**定义 C.6.** 设 `\mathcal C` 是 pointed presentably symmetric monoidal
infinity-category，张量积分变量保持小余极限，且 `A\in\mathcal C`。若存在
presentably symmetric monoidal infinity-category $\mathcal C[A^{-1}]$ 和
保持小余极限的对称幺半函子

$$
\Sigma_A^\infty:\mathcal C\to\mathcal C[A^{-1}],
$$

使对每个 presentably symmetric monoidal $\mathcal D$，预合成给出等价

$$
\operatorname{Fun}^{L,\otimes}(\mathcal C[A^{-1}],\mathcal D)
\simeq
\left\{F\in\operatorname{Fun}^{L,\otimes}(\mathcal C,\mathcal D):
F(A)\text{ 张量可逆}\right\},
$$

则称它为 $\mathcal C$ 的 **$A$-反演**。右侧是所示函子范畴的 full
subcategory；该等价包括态射和高阶相干数据，而不只是对象集合的双射。
在尚未验证谱模型假设前，本书只写 $\mathcal C[A^{-1}]$，不先把它记成
$\operatorname{Sp}_A(\mathcal C)$。

**定义 C.7.** 对象 `A` 称为 **3-symmetric**，若 $A^{\otimes3}$ 上由三循环
`(123)` 诱导的自同构在 `h\mathcal C` 中等于恒等态射。该条件比
`\mathcal C` 的张量积具有对称约束更强；后者只给出 `\Sigma_3`-作用，不说
该作用平凡。

**外部输入定理 C.8（对象反演与谱模型）.** 在 C.6 的假设下：

1. presentable symmetric monoidal $A$-反演 $\mathcal C[A^{-1}]$ 存在；
2. 若 $A$ 3-symmetric，则其底层 infinity-category 与关于
   $\Sigma_A=A\wedge-$ 的稳定化 $\operatorname{Sp}_A(\mathcal C)$ 等价；
3. 在相容的 symmetric monoidal model category presentation 存在时，
   symmetric $A$-spectra 的底层 infinity-category 给出同一对象反演。

在第 2 项中，记

$$
\Sigma_A=A\wedge-:\mathcal C\to\mathcal C
$$

并以 $\Omega_A$ 表示其右伴随。谱模型的局部对象可表示为序列
$(E_n)_{n\ge0}$ 与结构映射 $A\wedge E_n\to E_{n+1}$，其伴随映射
$E_n\to\Omega_AE_{n+1}$ 均为等价。一般对象反演不自动是 stable；只有当
目标中的 ordinary suspension 可逆时才得到 stable infinity-category。

**外部来源与边界.** Robalo, *Noncommutative Motives I*, Proposition 4.10
（presentable 对象反演）、Corollary 4.24（3-symmetric 对象的反演与稳定化）
和 Theorem 4.29（symmetric spectra 模型比较）。没有 3-symmetry 时仍有第
1 项的形式反演，但本书不把朴素序列谱模型无条件套用到它。

**命题 C.9.** 若 `F:\mathcal C\to\mathcal D` 是保持小余极限的对称幺半
函子，`\mathcal D` 是 presentably symmetric monoidal infinity-category，且
`F(A)` 张量可逆，则 `F` 通过 `\Sigma_A^\infty` 在保持小余极限的对称幺半
函子意义下唯一因子化。

**证明.** 对称幺半性给出
$F(A\wedge X)\simeq F(A)\otimes F(X)$。由于 $F(A)$ 张量可逆，右端
关于 $X$ 的自函子是等价，因此 $F$ 把 $A\wedge-$ 送为等价。由
$\mathcal C[A^{-1}]$ 的反演泛性质，得到所述因子化及其可缩的唯一选择
空间。`\square`

## C.4 T-稳定化

**定义 C.10.** 对 `\mathbf H_*(S)`，取

$$
T=\mathbb A^1/(\mathbb A^1\setminus0).
$$

定义

$$
\mathbf{SH}(S)=\mathbf H_*(S)[T^{-1}].
$$

待 C.12 验证 `T` 的 3-symmetry 和模型比较后，也可把右端写成
`\operatorname{Sp}_T(\mathbf H_*(S))`。

**引理 C.11（可逆张量积的因子）.** 在 symmetric monoidal
infinity-category 中，若 `A\otimes B` 可逆，则 `A` 与 `B` 都可逆。

**证明.** 取 `C` 及等价
`(A\otimes B)\otimes C\simeq\mathbb 1`。由结合和对称约束，

$$
A\otimes(B\otimes C)\simeq\mathbb 1,
\qquad
(B\otimes C)\otimes A\simeq\mathbb 1,
$$

故 `B\otimes C` 是 `A` 的双边逆。同理，`C\otimes A` 是 `B` 的双边逆。
全部等价由 symmetric monoidal coherence 给出，故在 infinity-category 中成立，
而不只是 `h\mathcal C` 中的对象同构。`\square`

## C.5 Symmetric monoidal refinement

**外部输入定理 C.12（motivic 对称幺半稳定化）.** 对本书允许的基概形
$S$，Nisnevich 与 `\mathbb A^1`-局部化和 Cartesian product 相容，故
`\mathbf H_*(S)` 具有 C.4 的 presentable smash product。对象
$T\simeq S^{1,0}\wedge\mathbb G_m` 是 3-symmetric；上述 $T$-反演与标准
motivic symmetric $T$-spectra 构造等价。因此

$$
\mathbf{SH}(S)\simeq\operatorname{Sp}_T(\mathbf H_*(S))
$$

是 presentably symmetric monoidal infinity-category，且
`\Sigma_T^\infty` 保持小余极限并为强对称幺半函子。

**外部来源与边界.** Hoyois, Proposition 3.15 给出 motivic localization
与有限 Cartesian product 的相容性；Lemma 6.3 给出 spheres 的
3-symmetry；Proposition 6.4(1) 与 Corollary 6.7 给出稳定化和全部 motivic
spheres 反演的模型比较（均取平凡群）。形式对象反演及 symmetric spectra
比较的范畴论部分由 C.8 所列 Robalo Proposition 4.10、Corollary 4.24 与
Theorem 4.29 承担。没有这些 localization/model inputs，不能仅由商公式
断言得到 C.12 的 presentably symmetric monoidal category。

**命题 C.13（motivic 目标的稳定性）.** `\mathbf{SH}(S)` 是 stable
infinity-category，且 `S^{1,0}`、`\mathbb G_m` 与 `T` 的像都张量可逆。

**证明.** C.12 与对象反演的定义先给出 `T` 的像可逆。第三章命题 3.5 给出
`T\simeq S^{1,0}\wedge\mathbb G_m`；对其应用引理 C.11，得到
`S^{1,0}` 与 `\mathbb G_m` 的像分别可逆。C.12 的谱模型是 pointed，且因
smash product 分变量保持余极限，对任意 `E` 有

$$
\Sigma E\simeq S^{1,0}\wedge E.
$$

所以 ordinary suspension 是等价。由 stable infinity-category 的判别准则
（Lurie, *Higher Algebra*, Corollary 1.4.2.27），有 cofibers 的 pointed
infinity-category 中 suspension 为等价蕴含稳定性。`\square`

**注 C.14.** 深外部输入在于局部化的对称幺半相容、`T` 的 3-symmetry 和
谱模型比较。给定这些输入后，“`T` 可逆推出 ordinary suspension 可逆”是
引理 C.11 与上述稳定性判别的书内推导，不再被藏在“稳定化”一词中。

## C.6 本附录小结

`\mathbf{SH}(S)` 首先是反演 `T` 的 universal presentable symmetric
monoidal target；C.12 把该反演与 symmetric `T`-spectra 识别，C.13 再证明
其稳定性。这个分层避免把任意对象反演误写成无条件稳定化，也说明在
motivic 情形中 `T=S^{1,0}\wedge\mathbb G_m` 可逆事实上迫使两个因子分别
可逆。

## 练习

**练习 C.1.** 证明 `(-)_+` 是遗忘函子的左伴随。

**练习 C.2.** 写出 smash product 的 pushout 定义。

**练习 C.3.** 证明 `\operatorname{cofib}(Y\to *)\simeq\Sigma Y`，并用
smash product 分变量保余极限推出 `\Sigma Y\simeq S^1\wedge Y`。

**练习 C.4.** 完整写出引理 C.11 中 `A` 与 `B` 的双边逆及相干等价。

**练习 C.5.** 用泛性质解释 Betti realization 为什么从 `\mathbf{SH}` 因子化，并说明为何仅有一个非对称幺半函子还不足以搬运 commutative ring spectra。
