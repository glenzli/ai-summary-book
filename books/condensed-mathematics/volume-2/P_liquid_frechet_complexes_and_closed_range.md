# 附录 P：Liquid、Fréchet 复形与闭值域

## P.0 目标

第三卷频繁使用 Dolbeault 复形、椭圆算子和 Hodge 理论。它们的经典对象是 Fréchet 空间与连续线性算子；在 condensed/analytic 语言中，还要说明这些对象如何进入 liquid 或 analytic 模范畴。本附录补充两类严格检查：

1. Fréchet 复形何时给 Hausdorff cohomology。
2. liquid realization 与经典 Fréchet cohomology 比较时需要哪些假设。

## P.1 Fréchet 复形

设

$$
E^\bullet:\quad
E^0\xrightarrow{d^0}E^1\xrightarrow{d^1}E^2\to\cdots
$$

是 Fréchet 空间和连续线性映射组成的复形。

**定义 P.1.** 第 \(q\) 个 cohomology 的拓扑版本为

$$
H^q_{\mathrm{top}}(E^\bullet)
=
\ker d^q/\operatorname{im}d^{q-1}
$$

带商拓扑。若 \(\operatorname{im}d^{q-1}\) 在 \(\ker d^q\) 中闭，则该商为 Fréchet 空间。

**命题 P.2（闭像给 Hausdorff cohomology）.** 若 \(\operatorname{im}d^{q-1}\subset\ker d^q\) 闭，则 \(H^q_{\mathrm{top}}(E^\bullet)\) 是 Hausdorff Fréchet 空间。

**证明.** \(\ker d^q\) 是 Fréchet 空间 \(E^q\) 的闭子空间，因此为 Fréchet。闭子空间的商仍为 Hausdorff Fréchet 空间。证毕。

**边界 P.3.** 若像不闭，则代数 quotient 仍存在，但商拓扑非 Hausdorff。此时连续对偶、Fredholm 性和 Hodge 分解都不能按经典形式使用。

## P.2 Fredholm 复形的有限维性

**定义 P.4.** Fréchet 复形 \(E^\bullet\) 在次数 \(q\) Fredholm，若 \(\operatorname{im}d^{q-1}\) 闭且

$$
\dim_\mathbb C H^q_{\mathrm{top}}(E^\bullet)<\infty.
$$

**输入定理 P.5（椭圆 Fréchet 复形）.** 紧光滑流形上的椭圆微分复形，其全局光滑截面复形在每个次数有闭像和有限维 cohomology。更强地，存在 parametrix，给 Hodge 分解

$$
E^q=\operatorname{im}d^{q-1}\oplus\mathcal H^q\oplus\operatorname{im}(d^q)^\ast.
$$

**命题 P.6（Hodge 分解推出 Fredholm）.** 若 P.5 的分解成立且 \(\mathcal H^q\) 有限维，则 \(E^\bullet\) 在次数 \(q\) Fredholm，且

$$
H^q_{\mathrm{top}}(E^\bullet)\cong\mathcal H^q.
$$

**证明.** 若 \(u\in\ker d^q\)，按分解写

$$
u=d^{q-1}a+h+(d^q)^\ast b.
$$

对 \(d^q u=0\) 配合正交性得 \((d^q)^\ast b=0\)。于是 cohomology 类由唯一的 \(h\in\mathcal H^q\) 表示。分解中 \(\operatorname{im}d^{q-1}\) 是闭直和项，所以商 Hausdorff，且同构于有限维 \(\mathcal H^q\)。证毕。

## P.3 凝聚化与 liquid realization

对 Fréchet 空间 \(E\)，其凝聚化为

$$
\underline E(S)=\operatorname{Cont}(S,E).
$$

若选定 analytic ring \((\mathbb C,\mathcal M)\)，还需要一个 realization 过程把 \(\underline E\) 送入

$$
D(\mathbb C,\mathcal M).
$$

**输入定理 P.7（Fréchet-liquid realization）.** 在 Clausen-Scholze/Scholze 的 analytic-liquid 框架中，核 Fréchet 空间及其连续线性映射可函子性地实现为 analytic/liquid 模；该实现与有限极限、有限直和、闭子空间 kernel 和有限维 quotient 相容。

**命题 P.8（闭像复形的 cohomology 比较）.** 假设 \(E^\bullet\) 是 Fréchet 复形，且在次数 \(q\) 有闭像。再假设 P.7 的 realization 函子 \(\mathcal L\) 对短正合列

$$
0\to\operatorname{im}d^{q-1}\to\ker d^q\to H^q_{\mathrm{top}}(E^\bullet)\to0
$$

保持 exact triangle。则

$$
H^q(\mathcal L(E^\bullet))
\cong
\mathcal L(H^q_{\mathrm{top}}(E^\bullet)).
$$

**证明.** 闭像假设给 Fréchet 短正合列。realization 保持该短正合列对应的三角形，因此在 analytic 派生范畴中，\(q\) 次 cohomology 对象由同一个 quotient 表示。证毕。

**推论 P.9（有限维 cohomology 的类型闭合）.** 若 \(H^q_{\mathrm{top}}(E^\bullet)\) 有限维，则 \(\mathcal L(H^q_{\mathrm{top}}(E^\bullet))\) 是有限直和的单位对象，因而是 compact/perfect 对象。

**证明.** 有限维复向量空间同构于 \(\mathbb C^n\)。realization 保持有限直和，故得到单位对象的有限直和。compact/perfect 性来自单位对象 compact/perfect 且该性质对有限直和封闭。证毕。

## P.4 Dolbeault 复形的类型检查表

对紧复流形 \(X\) 与 holomorphic vector bundle \(E\)，使用 Dolbeault 复形时需记录：

1. \(\Gamma(X,\mathcal A^{0,q}(E))\) 的 Fréchet 拓扑；
2. \(\bar\partial\) 的连续性；
3. 椭圆 Hodge 输入给闭像和 finite-dimensional cohomology；
4. realization 函子 \(\mathcal L\) 的 exactness 范围；
5. finite-dimensional cohomology 在 analytic/liquid 范畴中对应 perfect 对象。

**边界 P.10.** 若只写代数复向量空间复形，则无法表达闭像；若只写 Fréchet 复形，则无法自动得到 analytic/liquid 范畴中的张量、Hom 和对偶公式。两层结构都要记录。

## 练习

1. 证明 Fréchet 空间闭子空间的 quotient 仍为 Fréchet 空间。
2. 在 P.6 中检查 \((d^q)^\ast b=0\) 的正交性步骤。
3. 给出一个连续线性映射像不闭导致 quotient 非 Hausdorff 的例子。
4. 对 compact Riemann surface 的 Dolbeault 复形，列出 P.4 的五项数据。
