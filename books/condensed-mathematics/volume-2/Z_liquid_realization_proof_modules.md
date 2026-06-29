# 附录 Z：Liquid realization 的证明模块

## Z.0 目标

附录 S 把 liquid realization 作为输入定理 S.2-S.3。本附录把它拆成三个层次：

1. 拓扑向量空间的凝聚化；
2. 凝聚化后进入 analytic/liquid 范畴的 realization；
3. 闭值域复形和 Dolbeault 复形的 cohomology 比较。

第一层可以书内证明。第二层是 Scholze/Clausen-Scholze 输入。第三层在接受第二层 exactness 后可书内证明。

## Z.1 拓扑向量空间的凝聚化

令 \(E\) 是 Hausdorff 拓扑实向量空间。定义预层

$$
\underline E(S)=\operatorname{Cont}(S,E)
$$

其中 \(S\) 为 compact Hausdorff 空间。

**命题 Z.1.** \(\underline E\) 是凝聚集合；若 \(E\) 是拓扑实向量空间，则 \(\underline E\) 是凝聚实向量空间对象。

**证明.** 对 compact Hausdorff 的有限联合满射覆盖 \(\coprod S_i\to S\)，连续映射 \(S\to E\) 等价于给出连续映射 \(S_i\to E\)，并在 fiber products \(S_i\times_SS_j\) 上相容。原因是 compact Hausdorff 空间中的有限联合满射是 quotient map；到 Hausdorff 空间的映射连续性可在 quotient 覆盖上检测。向量空间加法和数乘逐点定义，并由连续映射复合保持。证毕。

**边界 Z.2.** \(\underline E\) 是凝聚对象，不等于 liquid 对象。进入 liquid 范畴还需要 analytic/liquid realization。

## Z.2 Liquid realization 输入

**输入定理 Z.3（liquid realization functor）。** 对第二卷输入定理 D.6 中登记的拓扑向量空间子范畴 \(\mathcal T_p\)，存在函子

$$
\mathcal L_p:E\mapsto E_{\mathrm{liq}}
$$

从 \(\mathcal T_p\) 及其允许的连续线性映射到 \(D_{\mathrm{liq},p}\)。本书只使用以下性质：

1. \(\mathcal L_p(\mathbb R)\) 是单位对象；
2. 保持有限直和；
3. 保持有限极限；
4. 对闭嵌入和有限维 quotient 的短正合列给出 exact triangle；
5. 与经典连续线性映射的复合相容。

**说明.** 这是 liquid theory 的核心输入。本书后续只使用这些列出的性质。

## Z.3 有限维对象

**命题 Z.4.** 若 \(V\) 是有限维实向量空间，则

$$
\mathcal L_p(V)\simeq\mathbb R_{\mathrm{liq}}^{\oplus \dim V}.
$$

**证明.** 选取 \(V\cong\mathbb R^n\)。由 Z.3 的有限直和相容性，

$$
\mathcal L_p(V)\simeq\mathcal L_p(\mathbb R)^{\oplus n}
\simeq \mathbb R_{\mathrm{liq}}^{\oplus n}.
$$

证毕。

**推论 Z.5.** 有限维向量空间的 realization 是 perfect 对象。

**证明.** 单位对象 perfect，perfect 性对有限直和封闭。证毕。

## Z.4 闭值域复形

设 \(E^\bullet\) 是 Fréchet 复形。

**命题 Z.6.** 若 \(\operatorname{im}d^{q-1}\subset\ker d^q\) 闭，则

$$
0\to\operatorname{im}d^{q-1}\to\ker d^q\to H^q_{\mathrm{top}}(E^\bullet)\to0
$$

是 Fréchet 空间短正合列。

**证明.** \(\ker d^q\) 是 \(E^q\) 的闭子空间。闭值域假设说明 \(\operatorname{im}d^{q-1}\) 是 \(\ker d^q\) 的闭子空间。闭子空间商仍是 Hausdorff Fréchet 空间，因此得到 Fréchet 短正合列。证毕。

**命题 Z.7（realization 后 cohomology 比较）。** 假设 Z.3 适用于 Z.6 的短正合列，则

$$
H^q(\mathcal L_p(E^\bullet))
\simeq
\mathcal L_p(H^q_{\mathrm{top}}(E^\bullet)).
$$

**证明.** 由 Z.3，Z.6 的短正合列给出 liquid 范畴中的 exact triangle。cohomology 对象由 kernel modulo image 的同一三角形计算，因此得到等价。证毕。

## Z.5 Fredholm 复形

**定义 Z.8.** Fréchet 复形 \(E^\bullet\) 称为 Fredholm，若每个次数有闭值域且 \(H^q_{\mathrm{top}}(E^\bullet)\) 有限维。

**推论 Z.9.** 若 \(E^\bullet\) 是 Fredholm，且 Z.3 适用于相关短正合列，则每个

$$
H^q(\mathcal L_p(E^\bullet))
$$

是 perfect liquid 对象。

**证明.** 由 Z.7，它等价于有限维 \(H^q_{\mathrm{top}}\) 的 realization。由 Z.5 得 perfect。证毕。

## Z.6 Dolbeault realization

**输入定理 Z.10（Dolbeault-Fredholm 输入）。** 对 compact complex manifold \(X\) 和 holomorphic vector bundle \(E\)，Fréchet 复形

$$
\Gamma(X,\mathcal A^{0,\bullet}(E)),\bar\partial
$$

是 Fredholm，且其 cohomology 与 sheaf cohomology \(H^\bullet(X,E)\) 同构。

**定理 Z.11（Dolbeault liquid cohomology）。** 在 Z.3 和 Z.10 下，Dolbeault 复形的 liquid realization 满足

$$
H^q(\mathcal L_p\Gamma(X,\mathcal A^{0,\bullet}(E)))
\simeq
\mathcal L_p(H^q(X,E)).
$$

若 \(H^q(X,E)\) 有限维，则左侧 perfect。

**证明.** Z.10 给出 Fredholm 条件和经典 cohomology 同构。由 Z.7 得 realization 后 cohomology 比较；由 Z.9 得 perfect 性。证毕。

## Z.7 本附录闭包

**结论 Z.12.** 本书已证明：

1. 拓扑向量空间的凝聚化；
2. 闭值域 Fréchet cohomology 的 Hausdorff 性；
3. 接受 liquid realization exactness 后的 cohomology 比较；
4. Fredholm 有限维 cohomology 的 perfect 性；
5. Dolbeault 复形进入 liquid 范畴后的形式后果。

仍作为输入的是：

1. liquid realization functor 的构造；
2. realization 的 exactness 范围；
3. Dolbeault-Fredholm/Hodge 定理。

## 练习

1. 证明 compact Hausdorff 有限联合满射是 quotient map。
2. 证明 Z.1 的 sheaf 条件。
3. 给出像不闭时 quotient 非 Hausdorff 的例子。
4. 证明 Z.11 中 perfect 性的最后一步。
