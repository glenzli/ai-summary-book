# 附录 AP：GRR 的局部化与推前相容

## AP.0 目标

附录 AK 把 deformation to the normal cone 作为 GRR 证明模块。本附录补充另一个组织层：\(K\)-理论局部化、Chow/cohomology 局部化、Chern character 与边界映射相容，以及 proper pushforward 的函子性。

本附录证明形式相容部分；deformation specialization、excess intersection 和 Chern character 的深层构造仍作为输入。

## AP.1 局部化序列

设 \(i:Z\hookrightarrow X\) 是 closed immersion，\(j:U=X\setminus Z\hookrightarrow X\) 为开补。

**输入定理 AP.1（\(K\)-理论局部化）.** 对 Noetherian scheme 或相应复解析空间，有长正合列

$$
K_1(U)\xrightarrow{\partial_K}K_0(Z)
\xrightarrow{i_\ast}K_0(X)
\xrightarrow{j^\ast}K_0(U)\to0.
$$

对 coherent sheaf 的 Grothendieck group，可写为

$$
G_0(Z)\xrightarrow{i_\ast}G_0(X)\xrightarrow{j^\ast}G_0(U)\to0
$$

并带有高阶边界。

**输入定理 AP.2（cohomology/Chow 局部化）.** 有长正合列

$$
A_\ast(Z)\xrightarrow{i_\ast}A_\ast(X)
\xrightarrow{j^\ast}A_\ast(U)\to0
$$

及高阶边界映射。复流形情形可替换为带支集 cohomology 的长正合列。

## AP.2 Chern character 与边界

**输入定理 AP.3（局部化 Chern character）.** Chern character 与 Todd class 给出的变换

$$
\tau_X:K_0(X)\to A_\ast(X)_\mathbb Q,
\qquad
\tau_X(E)=\operatorname{ch}(E)\operatorname{td}(T_X)\cap[X]
$$

在 smooth 情形下与 AP.1、AP.2 的局部化边界相容，即交换图

$$
\begin{CD}
K_1(U) @>{\partial_K}>> K_0(Z)\\
@V{\operatorname{ch}}VV @VV{\tau_Z}V\\
A_{\ast+1}(U)_\mathbb Q @>{\partial_A}>> A_\ast(Z)_\mathbb Q
\end{CD}
$$

成立。一般 lci 情形需用 virtual tangent bundle。

**命题 AP.4（局部相容推出闭开拼接）.** 若 GRR 等式对 \(Z\) 与 \(U\) 成立，并且 AP.3 的边界相容成立，则对由局部化序列拼接得到的类 \(\alpha\in K_0(X)\)，GRR 等式在 \(X\) 上成立。

**证明.** 设

$$
\Delta_X(\alpha)=
\operatorname{ch}(Rf_\ast\alpha)\operatorname{td}(T_Y)
-f_\ast(\operatorname{ch}(\alpha)\operatorname{td}(T_X)).
$$

GRR 要证 \(\Delta_X(\alpha)=0\)。若 \(j^\ast\Delta_X(\alpha)=0\)，则 \(\Delta_X(\alpha)\) 来自 \(A_\ast(Z)\)。局部化边界相容说明这个支撑在 \(Z\) 的误差等于 \(Z\) 上 GRR 误差的推前。后者为零，所以 \(\Delta_X(\alpha)=0\)。证毕。

## AP.3 Proper pushforward 的函子性

**命题 AP.5（\(K\)-理论推前复合）.** 若

$$
X\xrightarrow{f}Y\xrightarrow{g}Z
$$

proper，且 \(Rf_\ast\)、\(Rg_\ast\) 保持 perfect complex 或 bounded coherent complex 的指定类别，则

$$
R(g\circ f)_\ast=Rg_\ast Rf_\ast
$$

给出

$$
(g\circ f)_\ast=g_\ast f_\ast
$$

在 \(K_0\) 或 \(G_0\) 上的等式。

**证明.** 派生推前的复合由 derived category 的函子性给出。Grothendieck group 上推前定义为

$$
[E]\mapsto\sum_i(-1)^i[R^if_\ast E]
$$

或 perfect complex 的 \(Rf_\ast\) 类。两种定义都与三角形可加性相容，因此从派生函子同构降到 \(K\)-群等式。证毕。

**命题 AP.6（cohomology 推前复合）.** 对 proper maps，有

$$
(g\circ f)_\ast=g_\ast f_\ast
$$

在 Chow groups 或 compactly supported cohomology 上成立。

**证明.** Chow 情形中 proper pushforward 按闭子簇的函数域次数定义，复合时域扩张次数相乘。cohomology 情形中推前由 Poincare duality 或 trace 定义，trace/counit 的复合律给等式。证毕。

## AP.4 GRR 的因子分解证明

**输入定理 AP.7（因子分解）.** 每个 projective morphism 可分解为

$$
X\xrightarrow{\Gamma_f}X\times Y\xrightarrow{p}Y,
$$

其中 \(\Gamma_f\) 是 closed immersion，\(p\) 是 projective bundle 或射影空间投影后的组合。lci morphism 可进一步由 regular immersion 与 smooth morphism 组成。

**输入定理 AP.8（基本因子的 GRR）.** GRR 对以下 morphism 成立：

1. projective bundle projection；
2. regular closed immersion；
3. open/closed 局部化拼接中由 deformation to the normal cone 给出的 specialization。

**定理 AP.9（GRR 复合证明模块）.** 接受 AP.5、AP.6、AP.7、AP.8 后，GRR 对 projective morphism 成立。

**证明.** 由 AP.7，把 \(f\) 写成基本因子的复合。AP.8 给每个基本因子的 GRR。设 \(f\) 与 \(g\) 均满足 GRR，则

$$
\operatorname{ch}(R(gf)_\ast\alpha)\operatorname{td}(T_Z)
=
\operatorname{ch}(Rg_\ast Rf_\ast\alpha)\operatorname{td}(T_Z).
$$

对 \(g\) 应用 GRR，再对 \(f\) 应用 GRR，并用 Todd class 对 tangent exact sequence 的乘法性，得到

$$
(gf)_\ast(\operatorname{ch}(\alpha)\operatorname{td}(T_X)).
$$

AP.5 与 AP.6 保证两侧推前的复合无歧义。证毕。

## AP.5 与 condensed/analytic 追踪

第三卷在 condensed/analytic 框架中使用 GRR 时，应记录：

1. \(K\)-类所在范畴：perfect complex、coherent sheaf 的 \(G_0\)，或 analytic perfect object；
2. pushforward 是 \(Rf_\ast\)、\(f_!\) 还是 compactly supported trace；
3. Todd class 使用真实 tangent bundle、virtual tangent bundle 或 dualizing complex；
4. Chern character 取值的 cohomology 理论。

没有这四项，Riemann-Roch 公式只是同形状表达式，不是定理。

## 练习

1. 写出 closed-open 分解 \(Z\subset X\)、\(U=X\setminus Z\) 的 \(G_0\) 右正合列。
2. 证明 AP.5 中 Grothendieck group 推前与三角形可加性相容。
3. 对 closed immersion \(i:Z\hookrightarrow X\)，说明 \(i_\ast\) 在 Chow group 中如何作用于闭子簇。
4. 用 AP.9 解释为什么 AK 中 graph factorization 足以把 GRR 化为基本因子。
