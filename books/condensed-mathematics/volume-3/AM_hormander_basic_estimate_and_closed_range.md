# 附录 AM：Hörmander 基本估计与闭值域步骤

## AM.0 目标

附录 AH 把 Hörmander \(L^2\) estimate 作为输入。本附录补充该输入的内部结构：Bochner-Kodaira 型恒等式给基本估计，基本估计经闭值域定理给 \(\bar\partial\) 方程解。

本附录仍把 Kähler 恒等式、椭圆正则性和边界正则化作为分析输入；书内证明的是从基本估计到解算子的泛函分析部分。

## AM.1 加权 Hilbert 复形

设 \(X\) 是 complete Kähler 流形，\(\varphi\in C^\infty(X,\mathbb R)\)。定义加权内积

$$
\langle \alpha,\beta\rangle_\varphi
=\int_X \langle\alpha,\beta\rangle e^{-\varphi}dV.
$$

记

$$
L^2_{0,q}(X,\varphi)
$$

为 \((0,q)\)-形式的 Hilbert 空间。令

$$
T_q=\bar\partial:L^2_{0,q}(X,\varphi)\to L^2_{0,q+1}(X,\varphi)
$$

取 maximal closed extension，令 \(T_q^\ast\) 为 Hilbert 空间伴随。

**定义 AM.1.** 加权 \(\bar\partial\)-Laplacian 为

$$
\Box_{\varphi,q}=T_{q-1}T_{q-1}^\ast+T_q^\ast T_q.
$$

## AM.2 Bochner-Kodaira 基本估计

**输入定理 AM.2（Bochner-Kodaira-Nakano 恒等式）.** 对紧支光滑 \((0,q)\)-形式 \(u\)，有

$$
\|T_qu\|_\varphi^2+\|T_{q-1}^\ast u\|_\varphi^2
=
\|\nabla''u\|_\varphi^2+
\langle [i\partial\bar\partial\varphi,\Lambda]u,u\rangle_\varphi.
$$

若

$$
i\partial\bar\partial\varphi\ge c\omega,\qquad c>0,
$$

则对 \(q>0\) 有

$$
\langle [i\partial\bar\partial\varphi,\Lambda]u,u\rangle_\varphi
\ge cq\|u\|_\varphi^2.
$$

**推论 AM.3（基本估计）.** 在 AM.2 的正性假设下，对 \(q>0\) 和紧支光滑 \(u\) 有

$$
\|u\|_\varphi^2
\le
\frac1{cq}\left(\|T_qu\|_\varphi^2+\|T_{q-1}^\ast u\|_\varphi^2\right).
$$

若 \(X\) complete，则该估计延拓到

$$
u\in\operatorname{Dom}(T_q)\cap\operatorname{Dom}(T_{q-1}^\ast).
$$

**证明.** 第一式由 AM.2 直接推出。complete 条件允许用 Friedrichs mollifier 与 cutoff exhaustion 逼近图范数中的元素；估计在极限下保持。逼近定理属于 complete Kähler 分析输入。证毕。

## AM.3 闭值域与解方程

以下命题是 Hilbert 复形的纯泛函分析步骤。

**命题 AM.4（闭值域判别）.** 设 \(T:H_0\to H_1\) 是 closed densely defined operator。若存在 \(C>0\)，使

$$
\|v\|_{H_1}\le C\|T^\ast v\|_{H_0}
$$

对所有

$$
v\in\operatorname{Dom}(T^\ast)\cap(\ker T^\ast)^\perp
$$

成立，则 \(\operatorname{im}T\) 闭，并且

$$
\operatorname{im}T=\ker T^\perp{}^\perp=\ker S
$$

在复形 \(H_0\xrightarrow{T}H_1\xrightarrow{S}H_2\) 且 \(ST=0\) 中，对满足 \(\ker S\subset\overline{\operatorname{im}T}\) 的情形成立。

**证明.** 取 \(v_n=Tu_n\) 为 \(\operatorname{im}T\) 中收敛列，极限为 \(v\)。可令 \(u_n\in(\ker T)^\perp\)。闭图定理给反估计

$$
\|u_n-u_m\|\le C\|T(u_n-u_m)\|.
$$

故 \(u_n\) Cauchy，极限 \(u\in\operatorname{Dom}(T)\)，且 closedness 给 \(Tu=v\)。所以 \(\operatorname{im}T\) 闭。最后等式来自 Hilbert 空间基本恒等式

$$
\overline{\operatorname{im}T}=(\ker T^\ast)^\perp
$$

和闭性。证毕。

**命题 AM.5（Hahn-Banach 解算子）.** 设 \(T:H_0\to H_1\) closed densely defined。若对所有 \(v\in\operatorname{Dom}(T^\ast)\cap(\ker T^\ast)^\perp\) 有

$$
\|v\|\le C\|T^\ast v\|,
$$

则对每个 \(f\in(\ker T^\ast)^\perp\)，存在 \(u\in H_0\) 满足

$$
Tu=f,\qquad \|u\|\le C\|f\|.
$$

**证明.** 在 \(\operatorname{im}T^\ast\) 上定义线性泛函

$$
\lambda(T^\ast v)=\langle f,v\rangle.
$$

若 \(T^\ast v=0\)，则 \(v\in\ker T^\ast\)，而 \(f\perp\ker T^\ast\)，所以定义无关。估计给

$$
|\lambda(T^\ast v)|\le \|f\|\|v\|\le C\|f\|\|T^\ast v\|.
$$

由 Hahn-Banach，\(\lambda\) 延拓到 \(H_0\)，存在 \(u\in H_0\) 表示它，且 \(\|u\|\le C\|f\|\)。于是

$$
\langle Tu,v\rangle=\langle u,T^\ast v\rangle=\langle f,v\rangle
$$

对所有 \(v\in\operatorname{Dom}(T^\ast)\) 成立，故 \(Tu=f\)。证毕。

## AM.4 Hörmander 解定理的闭合

**定理 AM.6（基本估计推出 \(\bar\partial\) 解）.** 若 AM.3 对 \(q\)-次形式成立，且

$$
f\in L^2_{0,q}(X,\varphi),\qquad \bar\partial f=0,\qquad q>0,
$$

则存在

$$
u\in L^2_{0,q-1}(X,\varphi)
$$

满足

$$
\bar\partial u=f,\qquad
\|u\|_\varphi^2\le\frac1{cq}\|f\|_\varphi^2.
$$

**证明.** 在复形

$$
L^2_{0,q-1}\xrightarrow{T_{q-1}}L^2_{0,q}\xrightarrow{T_q}L^2_{0,q+1}
$$

中，\(\bar\partial f=0\) 表示 \(f\in\ker T_q\)。基本估计对

$$
v\in\operatorname{Dom}(T_q)\cap\operatorname{Dom}(T_{q-1}^\ast)\cap\ker T_q
$$

化为

$$
\|v\|\le(cq)^{-1/2}\|T_{q-1}^\ast v\|.
$$

命题 AM.5 给 \(u\) 使 \(T_{q-1}u=f\)，并给范数估计。证毕。

## AM.5 从 complete 情形到 Stein 域

**输入定理 AM.7（complete Kähler 化与 exhaustion）.** Stein 流形 \(X\) 上存在 complete Kähler metric 与严格 plurisubharmonic exhaustion；对相对紧强 pseudoconvex 子域，可用边界正则化获得同类估计，并令子域递增到 \(X\)。

**推论 AM.8（Stein 上 Hörmander 消没的分析核心）.** 在 AM.7 下，对每个 \(q>0\)，每个 \(\bar\partial\)-closed \(L^2\) \((0,q)\)-形式在加权空间中有 \(L^2\) 原像。若原形式光滑，则由椭圆正则性可取光滑原像。

**证明.** 对 complete metric 应用 AM.6。对 exhaustion 情形，在子域上解方程，利用统一估计取弱极限；closedness 保证极限仍解方程。光滑性由椭圆正则性输入。证毕。

## 练习

1. 证明 Hilbert 空间恒等式 \(\overline{\operatorname{im}T}=(\ker T^\ast)^\perp\)。
2. 在 AM.5 中检查 \(\lambda\) 的定义无关性。
3. 说明 complete 条件在 AM.3 的图范数逼近中起什么作用。
4. 对 \(X=\mathbb C\)、\(\varphi=|z|^2\)，写出 \(q=1\) 时基本估计的正性项。
