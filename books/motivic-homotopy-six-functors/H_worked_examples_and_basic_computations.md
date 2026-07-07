# 附录 H：低阶例子、对象等价与基本计算

## 本附录目标

本附录把正文中反复使用的低阶计算集中写出。正式教材不能只给抽象定义；读者必须看到 `T`、`\mathbb P^1/\infty`、localization cofiber sequence、Thom spaces、finite etale transfer 和 norm 在最小例子中如何工作。

## 依赖前置知识

需要 pointed motivic spaces、cofiber、open-closed localization、Thom spaces、smooth/proper morphisms、finite etale morphisms 和 motivic ring spectra。

## H.1 `T`、`\Sigma\mathbb G_m` 与 `\mathbb P^1/\infty`

**定义 H.1.** Tate sphere 为

$$
T=\mathbb A^1/(\mathbb A^1\setminus0)=\mathbb A^1/\mathbb G_m.
$$

**命题 H.2.** 在 `\mathbf H_*(S)` 中有等价

$$
T\simeq\Sigma\mathbb G_m.
$$

**证明.** `T` 是包含 `\mathbb G_m\hookrightarrow\mathbb A^1` 的 cofiber。`\mathbb A^1` 在 `\mathbf H(S)` 中与终对象等价，所以该 cofiber 等价于 `\operatorname{cofib}(\mathbb G_m\to *)`。在 pointed category 中该 cofiber 为 `\Sigma\mathbb G_m`。`\square`

**命题 H.3.** 在 `\mathbf H_*(S)` 中有等价

$$
\mathbb P^1/\infty\simeq\mathbb A^1/(\mathbb A^1\setminus0)=T.
$$

**证明.** 用标准开覆盖

$$
\mathbb P^1=\mathbb A^1\cup\{\infty\}\text{ 的邻域}
$$

更具体地，`\mathbb P^1` 含开子概形 `\mathbb A^1`，闭补为无穷远点 `\infty\simeq S`。商 `\mathbb P^1/\infty` 把无穷远点压为基点。另一方面，`\mathbb A^1/(\mathbb A^1\setminus0)` 把 `\mathbb G_m` 压为基点。二者的等价可由 homotopy purity 应用于闭嵌入 `0:S\hookrightarrow\mathbb A^1` 与 `\infty:S\hookrightarrow\mathbb P^1` 的互补开集比较给出；两个闭点的法丛均为平凡线丛，其 Thom space 均为 `T`。`\square`

**注 H.4.** 若要完全避免 purity，可用显式坐标和 Nisnevich excision 构造该等价。本书采用 purity 作为 P0 外部输入，因此上述证明合法依赖第六章。

## H.2 最小 localization 计算

**例子 H.5.** 设 `i:Z\hookrightarrow X` 为闭嵌入，`j:U=X\setminus Z\hookrightarrow X`。对单位对象有 cofiber sequence

$$
j_!\mathbb 1_U\to\mathbb 1_X\to i_*\mathbb 1_Z.
$$

**命题 H.6.** 对任意 motivic spectrum `E\in\mathbf{SH}(X)`，有 cofiber sequence

$$
j_!j^*E\to E\to i_*i^*E.
$$

**证明.** 这是 localization recollement 的形式后果。第一箭头为 counit `j_!j^*E\to E`，其 cofiber 被 `j^*` 杀掉，因此位于 `i_*` 的本质像。对 `i^*` 作用后识别该 cofiber 为 `i_*i^*E`。`\square`

**例子 H.7.** 对 `X=\mathbb A^1`、`Z=\{0\}`、`U=\mathbb G_m`，得到

$$
j_!\mathbb 1_{\mathbb G_m}\to\mathbb 1_{\mathbb A^1}\to i_*\mathbb 1_S.
$$

在 `S` 上推前并使用 `\mathbb A^1`-contractibility，可把它看作连接 `\mathbb G_m`、单位对象和闭点贡献的基本 cofiber sequence。

## H.3 Thom spaces

**命题 H.8.** 平凡秩 `r` 向量丛 `\mathbb A^r_X\to X` 的 Thom space 为

$$
\operatorname{Th}(\mathbb A^r_X)\simeq T^{\wedge r}\wedge X_+.
$$

**证明.** 秩一情形由第六章命题 6.3 给出。一般 `r` 情形使用 direct sum formula

$$
\operatorname{Th}(V\oplus W)\simeq \operatorname{Th}(V)\wedge_X\operatorname{Th}(W)
$$

并对 `r` 做归纳。`\square`

**例子 H.9.** 对零截面 `S\hookrightarrow\mathbb A^r_S`，homotopy purity 给出

$$
\mathbb A^r/(\mathbb A^r\setminus0)\simeq T^{\wedge r}.
$$

这里法丛为平凡秩 `r` 向量丛。

## H.4 Smooth proper point 和 trace

**例子 H.10.** 恒等态射 `\operatorname{id}_S:S\to S` 同时 smooth 且 proper，`T_f=0`。因此

$$
f_!=f_*=f_\sharp=\operatorname{id}.
$$

其 trace 是 `\mathbb 1_S` 的恒等态射。

**例子 H.11.** 若 `L/k` 是 finite separable extension，则

$$
f:\operatorname{Spec}L\to\operatorname{Spec}k
$$

finite etale，故 `f_*\simeq f_\sharp`。对 cohomology theory `E`，这给出 additive transfer

$$
E(L)\to E(k).
$$

若 `E` 是 normed spectrum，还另有 multiplicative norm

$$
N_{L/k}:E(L)\to E(k).
$$

**命题 H.12.** 在例子 H.11 中，additive transfer 与 multiplicative norm 的相等不能由 finite etale 假设推出。

**证明.** Finite etale 假设给出 `f_*`、`f_\sharp` 和 norm functor 的定义域条件。`f_*`/`f_\sharp` 来自加性稳定范畴的伴随；norm 来自对称幺半乘法结构。两者作用在不同代数结构上，除非理论提供 Tambara-like 分配律或特定比较定理，否则不能相等。`\square`

## H.5 Motivic cohomology 最小计算

**例子 H.13.** 对终对象 `S`，

$$
H^{0,0}(S,\mathbb Z)=
\pi_0\operatorname{Map}_{\mathbf{SH}(S)}(\mathbb 1_S,H\mathbb Z_S).
$$

在常见连通正则基上，该群与 locally constant integer-valued functions 或 `\mathbb Z` 的相应全局截面比较；精确识别依赖 `H\mathbb Z` 的模型和基假设。

**命题 H.14.** 若 `X` smooth proper over `S`，则 `E^{*,*}(X)` 可同时由 `\Sigma_T^\infty X_+` 和结构态射 `p:X\to S` 的 `p_\sharp\mathbb 1_X` 表示。

**证明.** 对 smooth `p`，`p_\sharp` 是 `p^*` 的左伴随，并且 `p_\sharp\mathbb 1_X` 是 `X` 的 suspension spectrum 在 `S` 上的表达。故

$$
p_\sharp\mathbb 1_X\simeq \Sigma_T^\infty X_+.
$$

将其代入 `E`-cohomology 的映射空间定义即可。`\square`

## H.6 本附录小结

低阶计算说明抽象结构如何落到对象上：`T` 同时是 `\mathbb A^1/\mathbb G_m`、`\Sigma\mathbb G_m` 和 `\mathbb P^1/\infty`；localization 把开部分和闭支撑部分分解；Thom spaces 把向量丛转为 suspension 坐标；finite etale 态射同时支持 additive transfer 和 multiplicative norm，但二者结构不同。

## 练习

**练习 H.1.** 证明 `T\simeq\Sigma\mathbb G_m`。

**练习 H.2.** 用 homotopy purity 证明 `\mathbb P^1/\infty\simeq T`。

**练习 H.3.** 写出 `\mathbb A^2/(\mathbb A^2\setminus0)` 的 Thom space 表达。

**练习 H.4.** 对 finite separable extension `L/k`，区分 additive transfer 和 norm。

**练习 H.5.** 用 `p_\sharp\mathbb 1_X` 表达 smooth proper `X/S` 的 `E`-cohomology。
