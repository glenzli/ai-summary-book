# 第十九章：Equivariant motivic homotopy 与 quotient stacks

## 本章目标

本章介绍 equivariant motivic homotopy theory 的六操作版本。与普通 `G`-space 的 Borel construction 不同，genuine equivariant motivic theory 需要保留 stabilizers、representation spheres、quotient stacks 和 equivariant transfers。本章以 Hoyois 的 quotient stack 口径作为主要外部输入。

## 依赖前置知识

需要代数群、quotient stacks、linearly reductive groups、equivariant vector bundles、six operations、purity、cdh descent、Borel construction 和 stable motivic homotopy。

## 19.1 Quotient stack 口径

**约定 19.1.** 设 `G` 为基 `S` 上的线性代数群，`X` 为带 `G`-作用的 `S`-scheme。商栈记为

$$
[X/G].
$$

**定义 19.2.** Equivariant motivic homotopy category 的 quotient stack 口径把 `G`-equivariant geometry 视为栈 `[X/G]` 上的 motivic homotopy theory。

**外部输入定理 19.3.** 对 linearly reductive linear algebraic group 等假设，存在 parametrized motivic spaces/spectra over quotient stacks `[X/G]`，并满足 equivariant 版本的 purity、gluing、ambidexterity 和六操作。

**依赖源.** Marc Hoyois, "The six operations in equivariant motivic homotopy theory"。

**命题 19.4.** 若 `G` 为平凡群，则 quotient stack 口径恢复非 equivariant motivic homotopy。

**证明.** 若 `G=1`，则 `[X/G]\simeq X`。定理 19.3 的对象、态射和六操作都限制到 schemes 上的普通 motivic spaces/spectra。因此恢复前文的 `\mathbf H(X)` 与 `\mathbf{SH}(X)` 口径。`\square`

## 19.2 Genuine 与 Borel 的区别

**定义 19.5.** Borel-equivariant theory 通常把 `G`-对象 `X` 送到混合商 `EG\times_GX` 或相应近似。Genuine equivariant theory 则保留稳定子和表示球等信息。

**命题 19.6.** Borel-equivariant theory 不足以恢复 genuine equivariant motivic homotopy 的全部信息。

**证明.** Borel construction 把 equivariant 信息通过自由 `G`-空间近似转换为非 equivariant quotient 信息。该过程通常不会保留 representation spheres、fixed-point data 和 genuine transfers。Genuine equivariant theory 的六操作和 purity 依赖 quotient stack 的 stabilizer 结构，因此不能由 Borel theory 无条件恢复。`\square`

## 19.3 Equivariant six operations

**外部输入定理 19.7.** Equivariant motivic spectra 支持六操作，并且满足 equivariant base change、projection formula、localization、purity 和 cdh descent 的相应版本。

**命题 19.8.** Equivariant localization 对 invariant closed-open decomposition 成立。

**证明.** 若 `i:Z\hookrightarrow X` 是 `G`-invariant closed immersion，开补 `j:U\hookrightarrow X` 也带 `G`-作用。商后得到 closed-open pair `[Z/G]\hookrightarrow[X/G]\hookleftarrow[U/G]`。对 quotient stacks 应用 equivariant localization，得到相应 cofiber sequence。`\square`

## 19.4 Representation spheres

**定义 19.9.** 若 `V` 是 `G`-equivariant vector bundle 或 representation，其 Thom space 给出 equivariant sphere/twist，记为 `S^V` 或 `\Sigma^V`。

**命题 19.10.** Equivariant purity 中的 Thom twist 必须使用 equivariant normal bundle。

**证明.** 对 `G`-invariant closed immersion `Z\hookrightarrow X`，普通法丛 `N_{Z/X}` 自然带有 `G`-线性化。商栈上的 purity 记录 stabilizer 对法方向的作用；若忘记 equivariant structure，只保留底层向量丛，会丢失 representation sphere 信息。因此 Thom twist 必须是 equivariant normal bundle 的 Thom object。`\square`

## 19.5 Free actions 与 quotient schemes

**命题 19.11.** 若 `G` 在 `X` 上自由作用，且几何商 `X/G` 作为 scheme 存在并使 `X\to X/G` 为 `G`-torsor，则 quotient stack `[X/G]` 与 scheme `X/G` 表示的 stack 等价。

**证明.** 对任意测试对象 `T`，群胚 `\operatorname{Map}(T,[X/G])` 由 `G`-torsor `P\to T` 连同 equivariant map `P\to X` 给出。若 `X\to X/G` 为 `G`-torsor，则这样的数据等价于给出 `T\to X/G`，再取拉回 torsor `T\times_{X/G}X`。该构造给出群胚等价，故 stacks 等价。`\square`

**推论 19.12.** 在自由作用且好商存在时，equivariant motivic theory 应与商 scheme 上的 motivic theory 比较。

**证明.** 由命题 19.11，`[X/G]\simeq X/G`。若 equivariant motivic theory 采用 quotient stack 口径，则其在 `[X/G]` 上的值与在 `X/G` 上的值由 stack 等价识别。`\square`

**注 19.13.** 非自由作用时 `[X/G]` 保留 stabilizer 信息，而 coarse quotient 通常丢失该信息。因此 genuine equivariant theory 应使用 quotient stack，而不是只使用 coarse quotient。

## 19.6 Fixed points and isotropy

**定义 19.14.** 对子群 `H\subseteq G`，`H`-fixed locus `X^H` 是满足 `H` 逐点固定的子概形或子栈，若该对象存在。

**命题 19.15.** Genuine equivariant theory 中的 fixed-point information 不能由底层非 equivariant motivic spectrum 恢复。

**证明.** 底层非 equivariant functor 忘记 `G`-作用和所有 stabilizer 分层。两个不同 `G`-作用可能有同一底层 scheme 和同一底层 motivic spectrum，但 fixed loci、representation spheres 和 transfer data 不同。因此 fixed-point information 不由底层对象决定。`\square`

**例子 19.16.** `G_m` 作用在 `\mathbb A^1` 上：`t\cdot x=tx`。固定点为原点 `0`。商栈 `[ \mathbb A^1/G_m ]` 记录原点处的整个 `G_m` stabilizer；粗商只看到一个点状轨道闭包结构，不能保留同等 equivariant 信息。

## 19.7 Change of groups

**定义 19.17.** 若 `\varphi:H\to G` 是代数群同态，则 restriction functor 把 `G`-对象视为 `H`-对象，记为

$$
\operatorname{Res}^G_H.
$$

在 quotient stack 口径下，它由态射

$$
[X/H]\longrightarrow [X/G]
$$

诱导。

**外部输入定理 19.18.** 在 equivariant motivic six operations 的假设下，change-of-groups functors 具有与 pullback/pushforward 相容的六操作结构；对合适子群包含，还存在 induction、coinduction 或 norm 型结构。

**命题 19.19.** 若 `H=1`，restriction `\operatorname{Res}^G_1` 是忘却 equivariant structure 的 functor。

**证明.** 平凡群作用不包含非平凡 equivariance data。把 `G`-对象沿 `1\to G` 限制，只保留底层 scheme 或 spectrum，而忘记 `G`-action、stabilizer 和 representation-sphere 信息。`\square`

**命题 19.20.** 忘却 functor 通常不保守 fixed-point 信息。

**证明.** 忘却后两个不同 `G`-actions 可有同一底层对象。例如同一 scheme 上的平凡作用和非平凡作用底层 scheme 相同，但 fixed loci 和 quotient stacks 不同。因此从底层对象无法恢复 fixed-point data。`\square`

## 19.8 Compact generation and equivariant cells

**定义 19.21.** Equivariant cells 通常由轨道型对象 `G/H_+`、representation spheres `S^V` 及其 motivic suspensions 生成。代数几何中应以 quotient stacks 或 homogeneous spaces 的存在性为前提。

**外部输入定理 19.22.** 在 Hoyois 的假设下，equivariant motivic homotopy categories 具有适当的 compact generation/cdh descent 结构，可用 equivariant smooth schemes 或 quotient stack cells 控制。

**注 19.23.** Equivariant compact generation 的生成子依赖允许的群和基。不能直接把拓扑 equivariant stable homotopy 中的 orbit category 公式逐字搬到代数几何语境。

## 19.9 本章小结

Equivariant motivic homotopy theory 不是在普通 motivic theory 上附加一个群作用那么简单。Quotient stack、stabilizer、representation spheres 和 equivariant six operations 都是 genuine 结构。Hoyois 的六操作理论提供了标准入口；Borel 型理论只是某些线性化或近似版本。

## 练习

**练习 19.1.** 解释 `[X/1]\simeq X`。

**练习 19.2.** 比较 Borel-equivariant 与 genuine equivariant theory。

**练习 19.3.** 写出 invariant closed-open pair 的 localization sequence。

**练习 19.4.** 说明 equivariant normal bundle 在 purity 中的位置。

**练习 19.5.** 为什么 linearly reductive 假设会出现在 quotient stack 口径中？

**练习 19.6.** 证明自由作用且 torsor 商存在时 `[X/G]\simeq X/G`。

**练习 19.7.** 计算 `G_m` 作用在 `\mathbb A^1` 上的 fixed locus。

**练习 19.8.** 描述 `H\to G` 诱导的 `[X/H]\to[X/G]`。

**练习 19.9.** 解释为什么 equivariant cells 需要 representation spheres。
