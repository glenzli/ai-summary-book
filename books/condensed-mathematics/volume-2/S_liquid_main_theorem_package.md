# 附录 S：Liquid 主定理包

## S.0 目标

liquid theory 是第二卷主线的一部分，不只是第三卷 Dolbeault 应用的技术脚注。本附录把 liquid 向量空间、Fréchet/Banach 边界、闭值域复形和 Dolbeault 类型检查收束为一个主定理包。

本附录采取输入定理型写法：\(p\)-liquid 测度理论和 liquid realization 是 Scholze/Clausen-Scholze 的深层输入；本书证明接受这些输入后的范畴论、同调代数和函数分析形式后果。

## S.1 Liquid 输入数据

固定 \(0<p\le 1\) 的允许范围，并令

$$
(\mathbb R,\mathcal M_{<p})
$$

表示 \(p\)-liquid analytic ring。

**输入定理 S.1（\(p\)-liquid analytic ring）.** \((\mathbb R,\mathcal M_{<p})\) 是 analytic ring。相应 analytic 模范畴记为

$$
D_{\mathrm{liq},p}(\mathbb R)=D(\mathbb R,\mathcal M_{<p}).
$$

**输入定理 S.2（liquid realization）.** 存在从适当的拓扑向量空间范畴到 \(D_{\mathrm{liq},p}(\mathbb R)\) 或其心脏的 realization 过程

$$
\mathcal L_p:E\mapsto E_{\mathrm{liq}},
$$

在核 Fréchet 空间、Banach 空间的合适子类、有限维向量空间和连续线性映射上与经典结构相容。

**输入定理 S.3（exactness 范围）.** \(\mathcal L_p\) 保持有限直和、有限极限、闭嵌入的 kernel、有限维 quotient，并把满足闭值域条件的短正合列送到 liquid 范畴中的 fiber/cofiber sequence。

这些输入的精确范围依赖所选 \(p\) 和文献中的 liquid convention。正文使用时必须说明对象是否在 S.2-S.3 的范围内。

## S.2 Liquid 对象与 Hom 判别

**定义 S.4（liquid 对象）.** 在本书中，\(p\)-liquid 实向量空间是 \(D_{\mathrm{liq},p}(\mathbb R)\) 心脏中的对象，或由 S.2 realization 得到的对象。若不指定 \(p\)，默认存在一个固定允许的 \(p\) 使所有相关对象处于同一范畴。

**边界 S.5（Banach/Fréchet 不等于 liquid）.** 拓扑向量空间 \(E\) 本身不是 liquid 对象。必须给出 realization \(E_{\mathrm{liq}}\)，并说明 \(E\) 位于 S.2-S.3 的适用范围内。

**命题 S.6（有限维对象闭合）.** 有限维实向量空间 \(V\) 的 realization 是单位对象的有限直和：

$$
\mathcal L_p(V)\simeq \mathbb R_{\mathrm{liq}}^{\oplus \dim V}.
$$

因此它 compact、dualizable，并且 perfect。

**证明.** 有限维 \(V\cong\mathbb R^n\)。由 S.2-S.3，realization 保持有限直和并把 \(\mathbb R\) 送到单位对象。compact、dualizable 和 perfect 性对有限直和封闭。证毕。

## S.3 Fréchet 复形与闭值域

设

$$
E^\bullet:\cdots\to E^{q-1}\xrightarrow{d^{q-1}}E^q\xrightarrow{d^q}E^{q+1}\to\cdots
$$

是 Fréchet 空间和连续线性映射组成的复形。

**定义 S.7（闭值域条件）.** \(E^\bullet\) 在次数 \(q\) 满足闭值域条件，若

$$
\operatorname{im}d^{q-1}\subset\ker d^q
$$

为闭子空间。

**命题 S.8（闭值域给 Hausdorff cohomology）.** 若 \(E^\bullet\) 在次数 \(q\) 满足闭值域条件，则

$$
H^q_{\mathrm{top}}(E^\bullet)
=
\ker d^q/\operatorname{im}d^{q-1}
$$

是 Hausdorff Fréchet 空间。

**证明.** \(\ker d^q\) 是 Fréchet 空间的闭子空间，故为 Fréchet。闭子空间的 quotient 是 Hausdorff Fréchet 空间。证毕。

**命题 S.9（realization 与 cohomology 比较）.** 假设 \(E^\bullet\) 在次数 \(q\) 满足闭值域条件，且短正合列

$$
0\to\operatorname{im}d^{q-1}\to\ker d^q\to H^q_{\mathrm{top}}(E^\bullet)\to0
$$

处于 S.3 的 exactness 范围内。则

$$
H^q(\mathcal L_p(E^\bullet))
\simeq
\mathcal L_p(H^q_{\mathrm{top}}(E^\bullet)).
$$

**证明.** 闭值域给出 Fréchet 短正合列。S.3 把它送到 liquid 范畴中的 fiber/cofiber sequence。cohomology 对象由同一 quotient 表示，故得到等价。证毕。

## S.4 Fredholm 与 perfect 性

**定义 S.10（Fredholm Fréchet 复形）.** \(E^\bullet\) 称为 Fredholm，若每个次数满足闭值域条件，且每个 \(H^q_{\mathrm{top}}(E^\bullet)\) 有限维。

**推论 S.11（Fredholm 复形的 liquid perfect cohomology）.** 若 \(E^\bullet\) 是 Fredholm，且每个相关短正合列处于 S.3 的范围内，则

$$
H^q(\mathcal L_p(E^\bullet))
$$

是 perfect liquid 对象。

**证明.** 由 S.9，

$$
H^q(\mathcal L_p(E^\bullet))\simeq\mathcal L_p(H^q_{\mathrm{top}}(E^\bullet)).
$$

右侧由 S.6 perfect。证毕。

## S.5 Dolbeault 类型检查

令 \(X\) 是 compact complex manifold，\(E\) 是 holomorphic vector bundle。Dolbeault 复形的全局截面为

$$
\Gamma(X,\mathcal A^{0,\bullet}(E)),\bar\partial.
$$

**输入定理 S.12（Dolbeault Fréchet-Fredholm 输入）.** 上述 Fréchet 复形是 Fredholm，并且其 cohomology 与 \(H^\bullet(X,E)\) 同构。

**命题 S.13（Dolbeault 的 liquid 类型闭合）.** 在 S.1-S.3 和 S.12 下，Dolbeault 复形可实现为 \(D_{\mathrm{liq},p}(\mathbb C)\) 中的对象，且

$$
H^q(\mathcal L_p\Gamma(X,\mathcal A^{0,\bullet}(E)))
$$

是有限维 liquid 对象，并与经典 \(H^q(X,E)\) 的 realization 相容。

**证明.** S.12 给出 Fredholm 条件和经典 Dolbeault 同构。由 S.11，realization 后的 cohomology 是 perfect liquid 对象；由 S.9，它等于经典有限维 cohomology 的 realization。证毕。

## S.6 Liquid 主闭包定理

**定理 S.14（Liquid 主闭包）。** 接受 S.1-S.3 与 S.12 后，第二卷关于 liquid theory 的以下结构在书内闭合：

1. \(p\)-liquid analytic ring 的范畴位置；
2. 拓扑向量空间进入 liquid 范畴所需的 realization 输入；
3. Banach/Fréchet 与 liquid 的边界；
4. 闭值域 Fréchet 复形的 cohomology 比较；
5. Fredholm 复形有限维 cohomology 的 perfect 性；
6. Dolbeault 复形的 liquid 类型检查。

**证明.** S.1-S.4 给出范畴位置和对象定义。S.5 阻止把拓扑向量空间直接等同于 liquid 对象。S.8-S.11 证明闭值域、cohomology 比较和 perfect 性。S.12-S.13 处理 Dolbeault 应用。证毕。

## S.7 不能省略的假设

1. 不闭值域时，Fréchet cohomology 可能非 Hausdorff，不能直接进入 S.3。
2. Banach 或 Fréchet 空间不自动 liquid；必须给出 realization。
3. finite-dimensional cohomology 的 perfect 性依赖 realization 保持有限直和。
4. Dolbeault 复形的 analytic/liquid 使用依赖 Fredholm-Hodge 输入，不是纯形式结论。

## 练习

1. 证明 S.8。
2. 在 S.9 中写出 exact triangle 的 cohomology 长正合列。
3. 给出像不闭的连续线性映射例子，并说明 S.9 为什么不能用。
4. 对 compact Riemann surface 的 \(\mathcal O\) 写出 S.13 的对象和 cohomology。

