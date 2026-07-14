# 附录 Z：椭圆复形与 Hodge 定理接口

## Z.0 目标

相干上同调有限性和 Serre 对偶都依赖分析输入：椭圆正则性、Fredholm 性和 Hodge decomposition。本附录把这些输入拆成精确命题，并证明它们推出有限维 cohomology。

本附录不证明 pseudodifferential parametrix 的构造；它作为椭圆分析输入。书内证明从 parametrix/Fredholm 输入开始。

## Z.1 Hilbert 复形

设

$$
0\to H^0\xrightarrow{d_0}H^1\xrightarrow{d_1}\cdots\xrightarrow{d_{m-1}}H^m\to0
$$

是 Hilbert 空间上的闭稠定线性算子复形，即 $d_{q+1}d_q=0$。定义 cohomology

$$
\mathcal H^q_d=\ker d_q/\operatorname{im}d_{q-1}.
$$

假设每个 $d_q$ 有 Hilbert adjoint $d_q^\ast$。定义 Laplacian

$$
\Delta_q=d_{q-1}d_{q-1}^\ast+d_q^\ast d_q.
$$

**引理 Z.1.** 有

$$
\ker\Delta_q=\ker d_q\cap\ker d_{q-1}^\ast.
$$

**证明.** 若 $\Delta_q u=0$，则

$$
0=\langle\Delta_q u,u\rangle
=
\|d_q u\|^2+\|d_{q-1}^\ast u\|^2.
$$

故 $d_q u=0$ 且 $d_{q-1}^\ast u=0$。反向代入定义即可。证毕。

## Z.2 椭圆输入

**输入定理 Z.2（椭圆 Fredholm-Hodge 输入）.** 设 $X$ 是紧光滑流形，$E^\bullet$ 是有限秩光滑向量丛复形，

$$
\Gamma(E^0)\xrightarrow{D_0}\Gamma(E^1)\to\cdots\to\Gamma(E^m),
$$

且 $D^\bullet$ 是椭圆微分算子复形。取 Sobolev 完备化后，Laplacian

$$
\Delta_q=D_{q-1}D_{q-1}^\ast+D_q^\ast D_q
$$

满足：

1. $\ker\Delta_q$ 有限维，并由光滑截面组成；
2. $\operatorname{im}D_{q-1}$ 闭；
3. 正交分解
   $$
   H^q=\operatorname{im}D_{q-1}
   \oplus
   \ker\Delta_q
   \oplus
   \operatorname{im}D_q^\ast
   $$
   成立。

## Z.3 Hodge 定理的形式结论

**定理 Z.3.** 在 Z.2 的假设下，自然映射

$$
\ker\Delta_q\to
\ker D_q/\operatorname{im}D_{q-1}
$$

为同构。特别地，cohomology 有限维。

**证明.** 若 $h\in\ker\Delta_q$，由引理 Z.1 得 $D_qh=0$，故定义 cohomology 类。

单射：若 $h=D_{q-1}v$，且 $h\in\ker D_{q-1}^\ast$，则

$$
\|h\|^2=\langle h,D_{q-1}v\rangle
=
\langle D_{q-1}^\ast h,v\rangle=0.
$$

故 $h=0$。

满射：取 $u\in\ker D_q$。由 Z.2 的正交分解写

$$
u=D_{q-1}a+h+D_q^\ast b.
$$

对该式施加 $D_q$，得

$$
0=D_qD_q^\ast b
$$

因为 $D_qD_{q-1}=0$ 且 $D_qh=0$。于是

$$
\|D_q^\ast b\|^2
=
\langle D_qD_q^\ast b,b\rangle
=0.
$$

故 $D_q^\ast b=0$，于是 $u$ 与 $h$ 相差 $D_{q-1}a$。证毕。

## Z.4 Dolbeault 复形情形

设 $X$ 是紧复流形，$E$ 是 Hermitian 全纯向量丛。Dolbeault 复形

$$
\Gamma(X,\mathcal A^{0,0}(E))
\xrightarrow{\bar\partial}
\Gamma(X,\mathcal A^{0,1}(E))
\to\cdots
$$

是椭圆复形。

**输入定理 Z.4（Dolbeault 椭圆性）.** Dolbeault Laplacian

$$
\Delta_{\bar\partial}
=
\bar\partial\bar\partial^\ast+\bar\partial^\ast\bar\partial
$$

满足 Z.2 的 Fredholm-Hodge 输入。

**推论 Z.5.** 对紧复流形 $X$ 和全纯向量丛 $E$，

$$
H^q(X,\mathcal O(E))
\cong
\ker\Delta_{\bar\partial,q}
$$

且左侧有限维。

**证明.** Dolbeault resolution 把 sheaf cohomology 识别为 Dolbeault cohomology。再用 Z.3。证毕。

## Z.5 condensed/analytic 边界

Z.5 是经典有限维向量空间结论。进入 condensed/analytic 语境时，还需验证：

1. Sobolev/Fréchet 空间的凝聚化及其 liquid membership。
2. $\bar\partial$、$\bar\partial^\ast$、Green operator 的连续性。
3. harmonic projection 的像取通常欧氏拓扑，并在目标范畴中对应有限自由系数模；这
   不等于把有限维向量空间离散化。
4. Hodge decomposition 与导出全局截面比较相容。

这些验证不是椭圆复形形式论的自动结论。

## 练习

1. 证明引理 Z.1 中反向包含。
2. 在有限维 Hilbert 复形中直接证明 Z.3。
3. 解释为什么 $\operatorname{im}D_{q-1}$ 闭性对 cohomology Hausdorff 性必要。
4. 说明 Z.5 与附录 X 的有限性传播如何结合。
