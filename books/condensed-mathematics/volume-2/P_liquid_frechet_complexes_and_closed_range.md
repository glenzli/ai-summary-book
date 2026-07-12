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

**外部输入定理 P.5（椭圆 Fréchet 复形）.** 紧光滑流形上的有限阶椭圆微分
复形，在选定 Hermitian/Riemannian 度量后，其全局光滑截面复形在每个次数有闭像和
有限维 cohomology。Green operator 与正交投影在光滑 Fréchet 拓扑中连续，并给出
拓扑直和

$$
E^q=\operatorname{im}d^{q-1}\oplus\mathcal H^q\oplus\operatorname{im}(d^q)^\ast.
$$

特别地，\(\ker d^q=\operatorname{im}d^{q-1}\oplus\mathcal H^q\)，且
\(E^{q-1}\twoheadrightarrow\operatorname{im}d^{q-1}\) 与
\(\ker d^q\twoheadrightarrow\mathcal H^q\) 都有连续线性截面。

**来源与边界.** 本书把 parametrix、椭圆正则性、Green operator 连续性及 Hodge 分解
作为经典外部输入 D.8；命题 P.6 只证明这些输入的形式后果。

**命题 P.6（Hodge 分解推出 Fredholm）.** 若 P.5 的分解成立且 \(\mathcal H^q\) 有限维，则 \(E^\bullet\) 在次数 \(q\) Fredholm，且

$$
H^q_{\mathrm{top}}(E^\bullet)\cong\mathcal H^q.
$$

**证明.** P.5 已给出
\(\ker d^q=\operatorname{im}d^{q-1}\oplus\mathcal H^q\) 的拓扑直和。
因此第一项闭，商映射限制到 \(\mathcal H^q\) 是连续线性双射，其逆由连续 Hodge
projection 诱导。故
\(H^q_{\mathrm{top}}(E^\bullet)\cong\mathcal H^q\)，并因后者有限维而 Fredholm。
证毕。

## P.3 凝聚化、liquid membership 与严格性

固定 \(0<p\le1\)。对 Fréchet 空间 \(E\) 定义

$$
\mathcal L_p(E):=\underline E,
\qquad
\underline E(S)=\operatorname{Cont}(S,E).
$$

**外部输入定理 P.7（Fréchet 对象的 liquid membership）.** 每个实 Fréchet
空间的 \(\underline E\) 都是 \(p\)-liquid；连续线性映射诱导 liquid 态射。

**来源与边界.** 这是第五章定理 5.5 与推论 5.6：CS26 Theorem 2.14、Lemma 2.16
及逆极限稳定性。\(\mathcal L_p(E)\) 就是凝聚化，不是额外对象。P.7 只判断每一项
属于 \(\mathbf{Liquid}_p\)，不保证一个 Fréchet 正合列凝聚化后仍正合。

令

$$
B^q=\operatorname{im}d^{q-1},\qquad
Z^q=\ker d^q,\qquad
H^q_{\mathrm{top}}=Z^q/B^q.
$$

**定义 P.8（次数 \(q\) 的凝聚严格性）.** 假设 \(B^q\) 闭。称
\(E^\bullet\) 在次数 \(q\) 凝聚严格，如果两个连续满射

$$
E^{q-1}\twoheadrightarrow B^q,
\qquad
Z^q\twoheadrightarrow H^q_{\mathrm{top}}
$$

都满足第五章定义 5.8 的 \(\kappa\)-凝聚有效性。若二者都有连续截面，则该条件成立。

**命题 P.9（cohomology 比较的充要数据）.** 若 \(E^\bullet\) 在次数 \(q\)
凝聚严格，则在 \(\mathbf{Liquid}_p\) 中有自然同构

$$
H^q(\mathcal L_p(E^\bullet))
\cong
\mathcal L_p(H^q_{\mathrm{top}}(E^\bullet)).
$$

**证明.** 凝聚化逐测试对象保持 finite limits，故

$$
\ker(\underline d^q)=\underline{Z^q}.
$$

第一张凝聚有效满射使
\(\underline{E^{q-1}}\twoheadrightarrow\underline{B^q}\) 成为 sheaf
epimorphism；随后 \(\underline{B^q}\hookrightarrow\underline{Z^q}\) 为 monomorphism，
所以 \(\operatorname{im}(\underline d^{q-1})=\underline{B^q}\)。第二张有效满射和
第五章命题 5.9 给出短正合列

$$
0\to\underline{B^q}\to\underline{Z^q}
\to\underline{H^q_{\mathrm{top}}}\to0.
$$

因此导出范畴标准 \(t\)-结构中的 kernel modulo image 正是
\(\underline{H^q_{\mathrm{top}}}\)。P.7 保证这些对象和态射都位于满阿贝尔子范畴
\(\mathbf{Liquid}_p\)，同构也在其中成立。证毕。

**推论 P.10（有限维 cohomology）.** 在 P.9 假设下，若
\(H^q_{\mathrm{top}}(E^\bullet)\) 有限维，则比较同构的两侧都是 perfect liquid 对象。

**证明.** 有限维实空间的凝聚化是 liquid 单位
\(\underline{\mathbb R}\) 的有限直和。有限直和的单位是 dualizable 且 perfect；若
复形带复结构，则乘以 \(i\) 给该实 liquid 对象一个满足 \(i^2=-1\) 的内自同态。证毕。

## P.4 Dolbeault 复形的类型检查表

对紧复流形 \(X\) 与 holomorphic vector bundle \(E\)，使用 Dolbeault 复形时需记录：

1. \(\Gamma(X,\mathcal A^{0,q}(E))\) 的 Fréchet 拓扑；
2. \(\bar\partial\) 的连续复线性；
3. P.5 的椭圆 Hodge 输入给闭像、有限维 harmonic space 与连续 Green operators；
4. P.5 的连续 splittings 验证定义 P.8，而不是引用一个未指定的“realization exactness”；
5. P.9 给 cohomology 比较，P.10 给 perfect 性；复结构作为 liquid 内自同态保留。

**边界 P.11.** 闭值域只给 Hausdorff Fréchet quotient；P.9 还需要两张 quotient
映射的 profinite 局部提升。Hodge/Green splitting 是 Dolbeault 情形的充分理由，但一般
闭值域 Fréchet 复形不能省略这项验证。

## 练习

1. 证明 Fréchet 空间闭子空间的 quotient 仍为 Fréchet 空间。
2. 从 P.5 的拓扑直和逐步推出命题 P.6。
3. 给出一个连续线性映射像不闭导致 quotient 非 Hausdorff 的例子。
4. 对 compact Riemann surface 的 Dolbeault 复形，列出 P.4 的五项数据，并指出哪张
   连续 splitting 验证定义 P.8。
