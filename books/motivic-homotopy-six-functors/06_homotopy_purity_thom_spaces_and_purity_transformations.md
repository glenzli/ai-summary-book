# 第六章：Homotopy purity、Thom spaces 与 purity transformations

## 本章目标

本章引入 Thom spaces 和 purity。Purity 是 motivic homotopy theory 中连接局部化、法丛、Gysin maps 和 exceptional pullback 的核心定理。本章先证明 Thom spaces 的形式性质，再把 homotopy purity 和六操作中的 purity transformations 标记为外部输入。

## 依赖前置知识

需要 pointed motivic spaces、vector bundles、closed immersions、normal bundles、cofiber、stable motivic homotopy category、six operations、localization 和 tensoring by Thom spectra。

## 6.1 Thom spaces

**定义 6.1.** 设 `p:V\to X` 是 `X` 上向量丛，零截面为 `s:X\hookrightarrow V`。Thom space 定义为 pointed motivic space

$$
\operatorname{Th}_X(V)=V/(V\setminus s(X)).
$$

若上下文清楚，写作 `\operatorname{Th}(V)`。

**命题 6.2.** 对零向量丛 `0_X`，有自然等价

$$
\operatorname{Th}(0_X)\simeq X_+.
$$

**证明.** 零向量丛的总空间是 `X`，零截面为恒等嵌入，其补集为空。因此

$$
\operatorname{Th}(0_X)=X/\varnothing.
$$

在 pointed motivic spaces 中，把空子对象压缩到基点等于添加一个不相交基点，故得到 `X_+`。`\square`

**命题 6.3.** 若 `L` 是 `X` 上平凡线丛，则

$$
\operatorname{Th}(L)\simeq T\wedge X_+.
$$

**证明.** 平凡线丛总空间为 `X\times\mathbb A^1`，零截面的补为 `X\times\mathbb G_m`。因此

$$
\operatorname{Th}(L)=(X\times\mathbb A^1)/(X\times\mathbb G_m).
$$

由于 smash product 与 colimits 相容，该商等价于

$$
X_+\wedge(\mathbb A^1/\mathbb G_m)=X_+\wedge T.
$$

对称性给出 `T\wedge X_+`。`\square`

**外部输入定理 6.4（Thom direct sum formula）.** 对 `X` 上向量丛 `V,W`，有自然等价

$$
\operatorname{Th}(V\oplus W)\simeq \operatorname{Th}(V)\wedge_X\operatorname{Th}(W)
$$

在适当的相对 pointed motivic category 中成立；稳定化后给出 Thom twists 的可加性。

**注 6.5.** 本书后续在稳定范畴中把向量丛 `V` 的 Thom twist 写作 `\Sigma^V`。若 `V-W` 是虚向量丛，则写 `\Sigma^{V-W}=\Sigma^V\Sigma^{-W}`。

## 6.2 Homotopy purity

**定义 6.6.** 若 `i:Z\hookrightarrow X` 是闭嵌入，且 `Z`、`X` 光滑，记 `N_{Z/X}` 为法丛。

**外部输入定理 6.7（Morel-Voevodsky homotopy purity）.** 对光滑概形之间的闭嵌入 `i:Z\hookrightarrow X`，在 `\mathbf H_*(S)` 中有自然等价

$$
X/(X\setminus Z)\simeq \operatorname{Th}(N_{Z/X}).
$$

**依赖源.** Morel-Voevodsky foundational purity theorem；equivariant/stacky/log 变体需要各自外部输入。

**命题 6.8.** 若 `i:X\hookrightarrow X` 为恒等闭嵌入，则定理 6.7 化为 `X_+\simeq X_+`。

**证明.** 恒等闭嵌入的开补为空，故左侧为 `X/\varnothing\simeq X_+`。法丛为零向量丛 `0_X`，由命题 6.2，右侧为 `\operatorname{Th}(0_X)\simeq X_+`。`\square`

**命题 6.9.** 若 `i:S\hookrightarrow\mathbb A^1_S` 是零截面，则定理 6.7 给出

$$
\mathbb A^1/(\mathbb A^1\setminus0)\simeq T.
$$

**证明.** 左侧按定义就是 `T`。零截面的法丛是 `S` 上的平凡线丛。由命题 6.3，其 Thom space 为 `T\wedge S_+\simeq T`。`\square`

## 6.3 稳定 purity transformations

**定义 6.10.** 对向量丛 `V`，在 `\mathbf{SH}(X)` 中写

$$
\operatorname{Th}_X(V)=\Sigma_T^\infty \operatorname{Th}(V)
$$

并定义自函子

$$
\Sigma^V(E)=\operatorname{Th}_X(V)\otimes_X E.
$$

**外部输入定理 6.11（Smooth purity）.** 若 `f:X\to Y` 是 smooth morphism，相对切丛为 `T_f`，则有自然等价

$$
f^!\simeq \Sigma^{T_f}f^*.
$$

**外部输入定理 6.12（Closed immersion purity）.** 若 `i:Z\hookrightarrow X` 是合适正则闭嵌入，法丛为 `N_{Z/X}`，则有自然等价

$$
i^!\simeq \Sigma^{-N_{Z/X}}i^*
$$

或等价地

$$
i^!\mathbb 1_X\simeq \Sigma^{-N_{Z/X}}\mathbb 1_Z,
$$

具体表述依赖正则性、绝对纯性和所用六操作版本。

**注 6.13.** 定理 6.12 是高风险外部输入：对一般闭嵌入不能无条件使用。若只知道 homotopy purity，则得到的是光滑对象中的商与 Thom space 的等价；若要把它升级为 `i^!` 的公式，需要六操作和纯性相干。

## 6.4 Gysin maps 的形式来源

**定义 6.14.** 设 `E` 是 motivic ring spectrum。对 smooth morphism `f:X\to Y`，purity 等价给出从 `f^*` 到 `f^!` 的 Thom twist 识别。结合伴随，可构造 Gysin 型推前或拉回映射；具体方向取决于 `E` 的协变/反变约定和定向数据。

**命题 6.15.** 若 `f:X\to Y` proper 且 smooth，并且 `E` 是 `\mathbf{SH}(Y)` 中对象，则 trace/adjunction 给出映射

$$
f_*f^*E\longrightarrow E.
$$

**证明.** proper compatibility 给出 `f_!\simeq f_*`。伴随 `f^*\dashv f_*` 的 counit 正是自然变换 `f_*f^*E\to E`。该映射存在不需要 purity；若要把它解释为积分或 Gysin map，需要 smooth purity 和 orientation 等额外结构。`\square`

**命题 6.16.** 若向量丛 `V` 有 Thom class 使 `\operatorname{Th}(V)` 在某个 `E`-cohomology 理论中可定向，则 Thom isomorphism 是额外定向数据的后果，不是 Thom space 定义的形式后果。

**证明.** Thom space 的定义只给出对象 `V/(V\setminus X)`。Thom isomorphism 要求与系数理论 `E` 相关的类 `u\in E^{*,*}(\operatorname{Th}(V))` 使 cup product with `u` 诱导等价。该类的存在依赖 orientation；没有 orientation 时只能谈 Thom object 和 Thom twist，不能推出 cohomology 群同构。`\square`

## 6.5 失败模式

**命题 6.17.** 不能把 homotopy purity 的等价

$$
X/(X\setminus Z)\simeq\operatorname{Th}(N_{Z/X})
$$

直接替换为任意闭嵌入上的 `i^!\simeq\Sigma^{-N}i^*`。

**证明.** homotopy purity 的假设包含光滑性和法丛存在；`i^!` 的公式属于稳定六操作和绝对/相对纯性。一般闭嵌入可能不是正则嵌入，法丛未必以向量丛形式存在；即便存在，`i^!` 的相干公式还需要六操作中的 purity transformation。因此二者不能无条件互换。`\square`

## 6.6 本章小结

Thom spaces 把向量丛转换为 pointed motivic spaces，homotopy purity 把闭补商识别为法丛的 Thom space。稳定六操作中的 purity 进一步把 `f^!` 与 `f^*` 的差异编码为切丛或法丛的 Thom twist。Gysin maps、Euler classes 和 bivariant theory 都建立在这些结构上，但 orientation 和 absolute purity 是额外输入。

## 练习

**练习 6.1.** 证明零向量丛的 Thom space 为 `X_+`。

**练习 6.2.** 对平凡秩 `r` 向量丛，推导其 Thom space 与 `T^{\wedge r}\wedge X_+` 的关系。

**练习 6.3.** 说明 homotopy purity 中为什么需要法丛。

**练习 6.4.** 写出 smooth purity 在 etale morphism 情形下的形式。

**练习 6.5.** 解释 orientation 与 Thom space 定义之间的逻辑差异。
