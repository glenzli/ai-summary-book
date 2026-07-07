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

**定理 C.4.** 若 `\mathcal C` 是 Cartesian symmetric monoidal presentable category，则 `\mathcal C_*` 有 smash product

$$
X\wedge Y=(X\times Y)/(X\vee Y),
$$

其中 `X\vee Y=X\amalg_*Y`。

**证明.** Pointed objects 上的 tensor product 由 Cartesian product 经 pointed quotient 得到。`X\vee Y\to X\times Y` 由两个基点嵌入给出；其 cofiber 定义 smash product。Presentability 和 Cartesian product 对 colimits 的相容性保证该构造给出 presentable symmetric monoidal structure。`\square`

**命题 C.5.** 在 pointed spaces 中，`\operatorname{cofib}(Y\to *)\simeq\Sigma Y`。

**证明.** Cofiber 定义为 `*\coprod_Y *`，这正是 suspension 的定义。`\square`

## C.3 Spectrum objects

**定义 C.6.** 设 `\mathcal C` pointed presentable，且 `A\in\mathcal C`。一个 `A`-spectrum 是序列 `(E_n)_{n\ge0}` 连同结构映射

$$
A\wedge E_n\to E_{n+1}
$$

满足稳定条件；其范畴记为 `\operatorname{Sp}_A(\mathcal C)`。

**定理 C.7.** `\operatorname{Sp}_A(\mathcal C)` 是 stable presentable infinity-category，并有 suspension spectrum functor

$$
\Sigma_A^\infty:\mathcal C\to\operatorname{Sp}_A(\mathcal C).
$$

**命题 C.8.** 若 `F:\mathcal C\to\mathcal D` 保持小余极限，`\mathcal D` stable，且 `F(A)` 可逆，则 `F` 通过 `\Sigma_A^\infty` 因子化。

**证明.** `A`-stabilization 是把 `A`-suspension 变为等价的 universal stable target。条件 `F(A)` 可逆保证 `F(A\wedge -)` 在 `\mathcal D` 中为等价，因此 `F` 满足稳定化关系。由泛性质得到唯一保持余极限因子化。`\square`

## C.4 T-稳定化

**定义 C.9.** 对 `\mathbf H_*(S)`，取

$$
T=\mathbb A^1/(\mathbb A^1\setminus0).
$$

定义

$$
\mathbf{SH}(S)=\operatorname{Sp}_T(\mathbf H_*(S)).
$$

**命题 C.10.** `\Sigma_T^\infty` 保持小余极限。

**证明.** `\Sigma_T^\infty` 是 stabilization 的左伴随结构函子。左伴随保持小余极限。`\square`

**命题 C.11.** `T` 在 `\mathbf{SH}(S)` 中可逆。

**证明.** `T`-spectrum construction 的目的就是使 `T\wedge-` 成为 suspension equivalence。故 `\Sigma_T^\infty T` 可逆。`\square`

## C.5 Symmetric monoidal refinement

**外部输入定理 C.12.** `\mathbf{SH}(S)` 的 `T`-稳定化可提升为 symmetric monoidal stable presentable infinity-category，且 `\Sigma_T^\infty` symmetric monoidal。

**注 C.13.** 该提升需要证明 `\mathbb A^1`-局部化、pointing、smash product 和 `T`-stabilization 彼此相容。基础章节将其作为外部输入。

## C.6 本附录小结

`\mathbf{SH}(S)` 是 `T`-稳定化的 universal stable target。这个泛性质解释了 realization、cohomology theories 和 ring spectra 为什么必须反演 `T`。对称幺半提升是 motivic ring spectra 和六操作的基础。

## 练习

**练习 C.1.** 证明 `(-)_+` 是遗忘函子的左伴随。

**练习 C.2.** 写出 smash product 的 pushout 定义。

**练习 C.3.** 证明 `\operatorname{cofib}(Y\to *)\simeq\Sigma Y`。

**练习 C.4.** 用泛性质解释 Betti realization 为什么从 `\mathbf{SH}` 因子化。

**练习 C.5.** 说明 symmetric monoidal refinement 为什么对 ring spectra 必要。
