# 附录 AE：Grothendieck-Riemann-Roch 的一般形式

## AE.0 目标

附录 U 证明了 $\mathbb P^n$ 线丛情形的 HRR。本附录给出一般 Grothendieck-Riemann-Roch 的精确形式，并证明若接受 GRR 输入，如何推出 HRR、可加性和函子性。

GRR 本身作为输入定理。

## AE.1 K 理论与 Chern character

设 $X$ 是光滑 proper 复代数簇或紧复流形。记

$$
K^0(X)
$$

为向量丛 Grothendieck 群。

**输入定理 AE.1（Chern character）.** 存在环同态

$$
\operatorname{ch}:K^0(X)\to H^{2\ast}(X,\mathbb Q)
$$

满足：

1. $\operatorname{ch}(E\oplus F)=\operatorname{ch}(E)+\operatorname{ch}(F)$；
2. $\operatorname{ch}(E\otimes F)=\operatorname{ch}(E)\operatorname{ch}(F)$；
3. 对线丛 $L$，$\operatorname{ch}(L)=e^{c_1(L)}$。

**输入定理 AE.2（Todd class）.** 对每个向量丛 $E$，存在

$$
\operatorname{td}(E)\in H^{2\ast}(X,\mathbb Q)
$$

满足短正合列乘法性，并在线丛 $L$ 上满足

$$
\operatorname{td}(L)=\frac{x}{1-e^{-x}},
\qquad x=c_1(L).
$$

## AE.2 GRR statement

**输入定理 AE.3（Grothendieck-Riemann-Roch）.** 设

$$
f:X\to Y
$$

是 proper morphism，$X,Y$ 光滑。对任意 $E\in K^0(X)$，

$$
\operatorname{ch}(Rf_\ast E)\operatorname{td}(T_Y)
=
f_\ast\bigl(\operatorname{ch}(E)\operatorname{td}(T_X)\bigr)
$$

在 $H^{2\ast}(Y,\mathbb Q)$ 中成立。

这里

$$
Rf_\ast E=\sum_i(-1)^i[R^if_\ast E]
$$

作为 $K^0(Y)$ 中的类理解；在需要完备一般性时，应使用 coherent $G$-theory 或 perfect complexes。

## AE.3 HRR 作为到点映射

**定理 AE.4（HRR）.** 若 $X$ 光滑 proper，$E$ 为向量丛，则

$$
\chi(X,E)
=
\int_X\operatorname{ch}(E)\operatorname{td}(T_X).
$$

**证明.** 取 $f:X\to *$。点的 Todd class 为 $1$，且

$$
Rf_\ast E
=
\sum_i(-1)^iH^i(X,E)
$$

在 $K^0(*)\cong\mathbb Z$ 中对应整数 $\chi(X,E)$。AE.3 变为

$$
\operatorname{ch}(Rf_\ast E)
=
\int_X\operatorname{ch}(E)\operatorname{td}(T_X).
$$

左侧即 $\chi(X,E)$。证毕。

## AE.4 可加性

**命题 AE.5.** HRR 右侧定义的映射

$$
K^0(X)\to\mathbb Q,\qquad
[E]\mapsto\int_X\operatorname{ch}(E)\operatorname{td}(T_X)
$$

是群同态。

**证明.** Chern character 对直和可加，积分线性，Todd class 固定。因此

$$
\int_X\operatorname{ch}(E\oplus F)\operatorname{td}(T_X)
=
\int_X(\operatorname{ch}(E)+\operatorname{ch}(F))\operatorname{td}(T_X).
$$

证毕。

## AE.5 复合映射的相容

**命题 AE.6.** 若 $X\xrightarrow fY\xrightarrow gZ$ 都 proper 且光滑条件满足，则 GRR 与复合相容。

**证明.** 对 $f$ 应用 GRR：

$$
\operatorname{ch}(Rf_\ast E)\operatorname{td}(T_Y)
=
f_\ast(\operatorname{ch}(E)\operatorname{td}(T_X)).
$$

再对 $g$ 和 $Rf_\ast E$ 应用 GRR：

$$
\operatorname{ch}(Rg_\ast Rf_\ast E)\operatorname{td}(T_Z)
=
g_\ast(\operatorname{ch}(Rf_\ast E)\operatorname{td}(T_Y)).
$$

代入第一式并用投影/推前复合

$$
g_\ast f_\ast=(g\circ f)_\ast
$$

得到

$$
\operatorname{ch}(R(g\circ f)_\ast E)\operatorname{td}(T_Z)
=
(g\circ f)_\ast(\operatorname{ch}(E)\operatorname{td}(T_X)).
$$

证毕。

## AE.6 condensed/analytic 接口

在 condensed/analytic 框架中，GRR 需要以下结构同时相容：

1. coherent/perfect objects 的 $K$-theory；
2. trace map 与 Euler characteristic；
3. Chern character 的 analytic 或 topological realization；
4. proper pushforward 与 $f_!$；
5. duality 和 Todd correction。

附录 U 只验证 $\mathbb P^n$ 线丛模型；本附录说明一般 GRR 被接受后能推出哪些形式后果。

## 练习

1. 从 AE.3 推出 AE.4。
2. 证明 AE.5。
3. 检查 AE.6 中使用了哪两个 functoriality 公式。
4. 说明为什么 singular 情形需要 $G$-theory 或 perfect complex 语言。
