# 附录 L：Fredholm-Hodge 有限性的形式证明层

## L.0 目标

第三卷第四章使用“elliptic regularity 或 Fredholm 性质证明上同调有限维”。本附录把这句话拆成可检查的形式证明：

1. Hilbert 复形的 Laplacian 与 harmonic vectors。
2. Hodge decomposition 推出 cohomology 与 harmonic space 同构。
3. Fredholm/椭圆输入给 harmonic space 有限维。
4. 因而 Dolbeault cohomology 有限维。

真正的分析输入是椭圆正则性、闭值域定理和 elliptic operator 的 Fredholm 性。本附录不重证偏微分方程估计。

## L.1 Hilbert 复形

设

$$
\cdots\to H^{q-1}\xrightarrow d H^q\xrightarrow d H^{q+1}\to\cdots
$$

是 Hilbert 空间上的有界或闭稠定算子复形，满足 $d^2=0$。设 $d^\ast$ 为 Hilbert adjoint。

**定义 L.1.** 第 $q$ 次 Laplacian 为

$$
\Delta_q=d_{q-1}d_{q-1}^\ast+d_q^\ast d_q.
$$

harmonic space 定义为

$$
\mathcal H^q=\ker\Delta_q.
$$

**引理 L.2.** 若 $x$ 属于 $\Delta_q$ 的定义域，则

$$
\langle\Delta_qx,x\rangle
=
\|d_qx\|^2+\|d_{q-1}^\ast x\|^2.
$$

因此

$$
\mathcal H^q=\ker d_q\cap\ker d_{q-1}^\ast.
$$

**证明.** 由 adjoint 定义，

$$
\langle d_{q-1}d_{q-1}^\ast x,x\rangle
=
\langle d_{q-1}^\ast x,d_{q-1}^\ast x\rangle,
$$

以及

$$
\langle d_q^\ast d_qx,x\rangle
=
\langle d_qx,d_qx\rangle.
$$

相加得公式。两项范数平方和为零当且仅当两项均为零。证毕。

## L.2 Hodge decomposition 的形式后果

**输入定理 L.3（Hodge decomposition hypothesis）.** 假设每个 $H^q$ 有正交分解

$$
H^q
=
\overline{\operatorname{im}d_{q-1}}
\oplus
\mathcal H^q
\oplus
\overline{\operatorname{im}d_q^\ast}.
$$

并且 $\operatorname{im}d_{q-1}$ 闭。

**定理 L.4.** 在 L.3 假设下，自然映射

$$
\mathcal H^q\to H^q_{\operatorname{coh}}
:=
\ker d_q/\operatorname{im}d_{q-1}
$$

为同构。

**证明.** 先证单射。若 $h\in\mathcal H^q$ 且 $h=d_{q-1}y$，则 $h\in\operatorname{im}d_{q-1}$。Hodge 分解是正交直和，而 $\mathcal H^q$ 与 $\operatorname{im}d_{q-1}$ 正交，故 $h=0$。

再证满射。取 $x\in\ker d_q$。按分解写

$$
x=d_{q-1}a+h+d_q^\ast b.
$$

对等式作用 $d_q$，得

$$
0=d_qx=d_qd_q^\ast b
$$

因为 $d_qd_{q-1}=0$ 且 $d_qh=0$。于是

$$
0=\langle d_qd_q^\ast b,b\rangle=\|d_q^\ast b\|^2.
$$

故 $d_q^\ast b=0$，所以 $x$ 与 $h$ 在商 $\ker d_q/\operatorname{im}d_{q-1}$ 中代表同一类。证毕。

**推论 L.5.** 若 $\mathcal H^q$ 有限维，则 $H^q_{\operatorname{coh}}$ 有限维。

**证明.** 由定理 L.4，二者同构。证毕。

## L.3 Fredholm 输入

**定义 L.6.** 闭算子 $T$ 称为 Fredholm，如果 $\ker T$ 有限维，$\operatorname{im}T$ 闭，且 cokernel 有限维。

**输入定理 L.7（椭圆 Fredholm 性）.** 紧光滑流形上的椭圆微分算子在合适 Sobolev 完备化之间给出 Fredholm 算子。特别地，Dolbeault Laplacian

$$
\Delta_{\bar\partial,q}
$$

的 kernel 有限维，并且光滑 harmonic forms 与 Sobolev harmonic vectors 一致。

**推论 L.8.** 在紧复流形上，全纯向量丛 $E$ 的 Dolbeault cohomology

$$
H^q(\Gamma(X,\mathcal A_X^{0,\bullet}(E)),\bar\partial)
$$

有限维。

**证明.** 取 Hermitian 度量，把 Dolbeault 复形放入 $L^2$ Hilbert 复形。椭圆 Fredholm 输入 L.7 给 harmonic space

$$
\mathcal H^{0,q}(X,E)=\ker\Delta_{\bar\partial,q}
$$

有限维。Hodge decomposition hypothesis L.3 在 Dolbeault elliptic complex 中由椭圆理论给出。由定理 L.4，Dolbeault cohomology 同构于 harmonic space。故有限维。证毕。

## L.4 从向量丛到相干层的边界

对一般相干解析层 $\mathcal F$，不能直接写成单个向量丛的 Dolbeault 复形。常用方法是：

1. 局部取有限自由分解。
2. 用 Cartan A/B 或解析有限自由 resolution 控制局部到整体。
3. 用谱序列或有限 resolution 把有限性从向量丛推广到相干层。

**输入定理 L.9（相干层有限性）.** 上述推广在紧复空间上成立，即 Grauert finiteness / coherent finiteness。

**本书证明的部分.** 若 $\mathcal F$ 在全局有有限长向量丛 resolution

$$
0\to E_m\to\cdots\to E_0\to\mathcal F\to0,
$$

且每个 $H^q(X,E_j)$ 有限维，则每个 $H^q(X,\mathcal F)$ 有限维。

**证明.** resolution 给出有限过滤或 hypercohomology spectral sequence

$$
E_1^{p,q}=H^q(X,E_p)
\Rightarrow
H^{p+q}(X,\mathcal F).
$$

每个 $E_1^{p,q}$ 有限维，且只有有限多个 $p,q$ 出现。因此每页有限维，极限项有限维，进而 $H^n(X,\mathcal F)$ 有限维。证毕。

## L.5 Condensed/analytic 翻译

在 condensed/analytic 语言中，有限性结论被翻译为：

1. Dolbeault 复形的各项是 liquid 向量空间。
2. $\bar\partial$ 是 liquid 范畴中的连续线性态射。
3. Fredholm/Hodge 分解说明 cohomology 是有限维离散或有限型对象。
4. 有限维对象在 analytic 派生范畴中是紧/dualizable 的基本例子。

**边界说明.** 本附录证明“Fredholm-Hodge 输入推出有限维上同调”。它没有证明 elliptic regularity，也没有证明 Clausen-Scholze 的 liquid 建模。

## 练习

**练习 L.1.** 证明引理 L.2 中 $\mathcal H^q=\ker d_q\cap\ker d_{q-1}^\ast$。

**练习 L.2.** 在定理 L.4 中，说明为什么 $d_qd_q^\ast b=0$ 推出 $d_q^\ast b=0$。

**练习 L.3.** 设有两项向量丛 resolution $0\to E_1\to E_0\to\mathcal F\to0$，写出推出 $\mathcal F$ 上同调有限性的长正合列证明。

**练习 L.4.** 解释为什么紧性在输入定理 L.7 中不可省略。
