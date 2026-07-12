# 附录 S：Liquid 主定理包

## S.0 目标

本附录把 liquid 主线收束为一个不依赖虚构 realization functor 的定理包：经典空间先凝聚化，
外部定理判断其 \(p\)-liquid membership，书内再用显式局部提升或连续 splitting 验证
exactness。

## S.1 Liquid 输入与对象

固定 \(0<p\le1\)。

**外部输入定理 S.1（\(p\)-liquid analytic ring）.** 测度理论
\((\underline{\mathbb R},\mathcal M_{<p})\) 是 analytic ring。其解析模范畴等价于
\(\mathbf{Liquid}_p\)，且有全忠实嵌入

$$
D(\mathbf{Liquid}_p)\hookrightarrow D(\mathbf{CondAb}).
$$

本质像由 cohomology objects 检测，liquidification 与 liquid tensor 存在。

**来源与边界.** S26 Theorem 7.11 与 CS26 Theorem 3.11。本书不构造
\(\mathcal M_{<p}\)，也不重证 analytic 公理。

**定义 S.2.** 凝聚阿贝尔群 \(V\) 是 \(p\)-liquid，如果每个
\(f:\underline S\to V\)、\(S\in\mathbf{ProFin}_\kappa\)，唯一延拓为

$$
\mathcal M_{<p}[S]\longrightarrow V.
$$

由 S.1，此条件自动给 \(V\) 唯一的凝聚 \(\mathbb R\)-模结构，并等价于 analytic
module Hom 判别。

## S.2 经典空间的凝聚化

**外部输入定理 S.3（Banach/Fréchet membership）.** 每个 \(p\)-Banach 空间的
凝聚化 \(\underline E\) 都 \(p\)-liquid；逆极限保持 \(p\)-liquid。因此每个实
Fréchet 空间的凝聚化对所有 \(0<p\le1\) 都 \(p\)-liquid。

**来源与边界.** CS26 Theorem 2.14、Lemma 2.16 及其后的逆极限推论。该定理只判断
对象属于 \(\mathbf{Liquid}_p\)，不判断拓扑 cokernel 与 liquid cokernel 相同。

**定义 S.4（本书的 realization 记号）.** 对上述经典空间定义

$$
\mathcal L_p(E):=\underline E,
\qquad
\underline E(S)=\operatorname{Cont}(S,E).
$$

这是凝聚化，不是第二个完成化。连续线性映射逐测试对象给出 liquid 态射。

**边界 S.5（四种对象不能混同）.** 对同一底层向量空间必须区分：

1. 抽象向量空间；
2. 带拓扑的 Banach/Fréchet 空间 \(E\)；
3. 凝聚模 \(\underline E\)；
4. 断言 \(\underline E\in\mathbf{Liquid}_p\) 的额外性质。

S.3 在 Banach/Fréchet 情形证明第 4 项，但并不把这四个类型定义为相等。

## S.3 有限维对象

**命题 S.6.** 若 \(V\) 是 \(n\) 维实向量空间，则

$$
\mathcal L_p(V)\cong\underline{\mathbb R}^{\oplus n}.
$$

因而它 compact、dualizable 且 perfect。

**证明.** 任取线性同构 \(V\cong\mathbb R^n\)。有限维 Hausdorff 向量空间的线性
同构自动为同胚，凝聚化逐测试对象保持有限乘积；在阿贝尔范畴中有限乘积等于有限直和。
S.1 说明 \(\underline{\mathbb R}\) 是 liquid 单位和紧投射生成元，所列性质对有限直和
封闭。证毕。

## S.4 Fréchet 复形与严格性

设

$$
E^\bullet:\cdots\to E^{q-1}\xrightarrow{d^{q-1}}E^q
\xrightarrow{d^q}E^{q+1}\to\cdots
$$

是 Fréchet 复形，并记

$$
B^q=\operatorname{im}d^{q-1},\quad
Z^q=\ker d^q,\quad
H^q_{\mathrm{top}}=Z^q/B^q.
$$

**定义 S.7.** 称复形在次数 \(q\) **closed-range**，如果 \(B^q\) 在 \(Z^q\)
中闭；称它在次数 \(q\) **凝聚严格**，如果此外两张满射

$$
E^{q-1}\twoheadrightarrow B^q,
\qquad
Z^q\twoheadrightarrow H^q_{\mathrm{top}}
$$

都满足第五章定义 5.8 的 profinite 局部提升条件。

**命题 S.8.** Closed-range 条件使 \(H^q_{\mathrm{top}}\) 成为 Hausdorff Fréchet
空间；若再凝聚严格，则

$$
H^q(\mathcal L_p(E^\bullet))
\cong
\mathcal L_p(H^q_{\mathrm{top}}(E^\bullet)).
$$

**证明.** 第一项由闭子空间的 Fréchet quotient 定理。对第二项，凝聚化保持 kernel，
而两张局部有效满射分别识别 liquid image 为 \(\underline{B^q}\)、liquid cokernel 为
\(\underline{H^q_{\mathrm{top}}}\)。因此 kernel modulo image 给出所示同构；详细逐步
证明见附录 P.9。证毕。

**反例边界 S.9.** Closed-range 与凝聚严格是不同条件。前者只控制 quotient 的
Hausdorff/complete 拓扑，后者控制从 profinite 参数族的局部提升。若没有后者，S.8 的
cohomology 同构没有证明。连续 splitting 同时给全局提升，因而是凝聚严格的充分条件。

## S.5 Fredholm 与 Dolbeault

**定义 S.10.** Fréchet 复形称为 Fredholm，如果每次 closed-range 且所有
\(H^q_{\mathrm{top}}\) 有限维。这个定义本身不包含凝聚严格性。

**推论 S.11.** 若 \(E^\bullet\) Fredholm 且每次凝聚严格，则

$$
H^q(\mathcal L_p(E^\bullet))
$$

是 perfect liquid 对象。

**证明.** S.8 把它识别为有限维拓扑向量空间的凝聚化，再用 S.6。证毕。

**外部输入定理 S.12（Dolbeault--Hodge 输入）.** 设 \(X\) 是 compact complex
manifold，\(E\) 是 holomorphic vector bundle。Dolbeault Fréchet 复形

$$
\Gamma(X,\mathcal A^{0,\bullet}(E)),\bar\partial
$$

有连续 Green operators 和 Hodge projections；每个像闭，harmonic spaces 有限维，
并与 sheaf cohomology \(H^q(X,E)\) 同构。相应 exact/coexact/harmonic 分解在光滑
Fréchet 拓扑中连续分裂。

**来源与边界.** 这是经典 elliptic Hodge/Fredholm 与 Dolbeault theorem 输入 D.8；
本书不重证 parametrix 和正则性估计。

**定理 S.13（Dolbeault 的 liquid cohomology）.** 在 S.12 假设下，Dolbeault
复形每项 \(p\)-liquid、每次凝聚严格，并有

$$
H^q\!\left(\mathcal L_p
\Gamma(X,\mathcal A^{0,\bullet}(E))\right)
\cong
\mathcal L_p(H^q(X,E)).
$$

两侧是 perfect liquid 对象。

**证明.** 每项是 Fréchet，故由 S.3 为 \(p\)-liquid。S.12 的连续 splittings 使定义
S.7 中两张满射都有连续线性截面，因而凝聚严格。S.8 给比较同构，S.12 给
\(H^q_{\mathrm{top}}\cong H^q(X,E)\)，最后 S.6 给 perfect 性。证毕。

## S.6 Liquid 主闭包

**定理 S.14（Liquid 主闭包）.** 接受 S.1、S.3 与 S.12 后，第二卷已闭合：

1. \(p\)-liquid 的定义、analytic 范畴位置和派生 Hom；
2. Banach/Fréchet 对象的 liquid membership；
3. realization 记号的真实定义 \(\mathcal L_p(E)=\underline E\)；
4. exactness 的 profinite 局部提升判别；
5. 凝聚严格 Fréchet 复形的 cohomology 比较；
6. Dolbeault--Hodge 情形的 perfect liquid cohomology。

**证明.** S.1--S.6 处理对象和范畴；S.7--S.9 隔离 exactness；S.10--S.13 将
Hodge splitting 代入该判别。每个非形式步骤都已列为 S.1、S.3 或 S.12 的外部输入。
证毕。

## 练习

1. 证明 S.6，并指出 compact 性使用 S.1 的哪一项。
2. 展开附录 P.9，证明 S.8 的 cohomology 同构。
3. 给出非闭像算子，并说明它先在哪一步破坏 S.8。
4. 说明连续 splitting 为什么同时验证定义 S.7 中的两张局部提升。
