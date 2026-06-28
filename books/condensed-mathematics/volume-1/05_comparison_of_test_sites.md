# 第五章：测试站点的比较

## 本章目标

凝聚集合可以从紧 Hausdorff 空间站点进入，也常从 profinite 空间站点进入。本章解释这两个口径之间的关系：$\mathbf{CHaus}$ 给出直观定义，$\mathbf{ProFin}$ 给出更小、更可计算的测试范畴。

本章会使用一个标准的站点比较定理。完整证明属于一般 topos theory；本书给出足够用于后续的版本和证明思路。

## 依赖前置知识

需要第一章的站点和 sheaf 条件，第二章的 $\mathbf{CHaus}$ 与 $\mathbf{ProFin}$，以及第三章的凝聚集合定义。

## 5.1 为什么要换测试站点

第三章采用定义

$$
\mathbf{CondSet}
=
\operatorname{Sh}(\mathbf{CHaus},J_{\operatorname{surj}}).
$$

这个定义自然，因为任意拓扑空间 $T$ 都给出

$$
\underline T(S)=\operatorname{Cont}(S,T),
\qquad S\in \mathbf{CHaus}.
$$

但是在计算中，$\mathbf{CHaus}$ 太大。紧 Hausdorff 空间可以很复杂，纤维积、覆盖和连续映射的结构不总是容易掌控。相比之下，profinite 空间有更强的离散近似：

$$
S\simeq \varprojlim_i S_i,
$$

其中 $S_i$ 是有限离散集合。这使它们更接近代数对象。

因此，我们希望知道：是否可以只在 profinite 空间上测试 sheaf？

答案是肯定的，适当表述后，$\mathbf{ProFin}$ 是 $\mathbf{CHaus}$ 的一个基。

## 5.2 基与站点比较

**定义 5.1.** 设 $(\mathcal C,J)$ 为站点，$\mathcal D\subset\mathcal C$ 为全子范畴。称 $\mathcal D$ 是 $\mathcal C$ 的一个基，如果满足：

1. 对每个 $U\in\mathcal C$，存在覆盖族 $\{D_i\to U\}$，其中 $D_i\in\mathcal D$。
2. 若 $D\in\mathcal D$ 且 $U\to D$ 是 $\mathcal C$ 中的态射，则存在覆盖族 $\{D_j\to U\}$，其中 $D_j\in\mathcal D$。

第二条的意思是：把 $\mathcal D$ 中对象沿任意态射拉回后，仍能被 $\mathcal D$ 中对象覆盖。

**定理 5.2（站点比较定理，基版本）.** 设 $\mathcal D\subset \mathcal C$ 是站点 $(\mathcal C,J)$ 的基，并在 $\mathcal D$ 上赋予诱导拓扑。则限制函子

$$
\operatorname{Sh}(\mathcal C,J)
\longrightarrow
\operatorname{Sh}(\mathcal D,J|_{\mathcal D})
$$

是范畴等价。

**证明思路.** sheaf 由其在覆盖基上的取值决定。给定 $\mathcal C$ 上的 sheaf，限制到 $\mathcal D$ 后，对 $\mathcal D$ 中的覆盖族仍满足同一等化子条件，因此得到 $\mathcal D$ 上的 sheaf。反过来，给定 $\mathcal D$ 上的 sheaf $F$，对 $U\in\mathcal C$ 选择 $\mathcal D$-覆盖 $\{D_i\to U\}$，并定义 $F(U)$ 为匹配族集合：

$$
F(U)=
\operatorname{Eq}
\left(
\prod_i F(D_i)
\rightrightarrows
\prod_{i,j}F(D_i\times_U D_j)
\right),
$$

其中纤维积项再用 $\mathcal D$-覆盖计算。基条件保证这个定义与覆盖选择无关，并满足 sheaf 条件。完整证明需要检查自然性、覆盖独立性和拟逆。证毕。

**注 5.3.** 定理 5.2 是 sheaf 理论的标准工具。本书后续使用它时，会明确指出所用子范畴是否确实构成基。

## 5.3 Profinite 空间是一个基

要把 $\mathbf{CHaus}$ 换成 $\mathbf{ProFin}$，需要一个非平凡拓扑事实。

**定理 5.4（profinite 覆盖）.** 对每个紧 Hausdorff 空间 $K$，存在 profinite 空间 $P$ 和满射连续映射

$$
P\to K.
$$

此外，也存在极不连通紧 Hausdorff 空间 $E$ 和满射 $E\to K$。

**证明说明.** profinite 覆盖可由 Stone 型表示定理得到；极不连通覆盖来自 Gleason 关于 compact Hausdorff 范畴中投射覆盖的理论。完整证明超出本章范围，后续第六章会讨论极不连通空间与投射性。本书把该定理作为引用结果使用，依赖来源见 [SOURCES.md](SOURCES.md)。

**引理 5.5.** 若 $P,Q$ 是 profinite 空间，且有连续映射

$$
P\to K,\qquad Q\to K
$$

到紧 Hausdorff 空间 $K$，则纤维积 $P\times_K Q$ 是 profinite 空间。

**证明.** 纤维积可看作 $P\times Q$ 的闭子空间：

$$
P\times_K Q
=
\{(p,q)\in P\times Q\mid f(p)=g(q)\}.
$$

因为 $K$ Hausdorff，对角线 $\Delta_K\subset K\times K$ 闭，故上述集合是连续映射

$$
P\times Q\to K\times K
$$

下 $\Delta_K$ 的逆像，因此闭。$P\times Q$ 是 profinite 空间，闭子空间仍 profinite。证毕。

**命题 5.6.** $\mathbf{ProFin}$ 是 $\mathbf{CHaus}$ 在有限联合满射拓扑下的基。

**证明.** 第一条由定理 5.4 给出：任意 $K\in\mathbf{CHaus}$ 有 profinite 空间 $P$ 满射到 $K$，于是 $\{P\to K\}$ 是覆盖。

第二条设 $P\in\mathbf{ProFin}$，且 $K\to P$ 是 $\mathbf{CHaus}$ 中的态射。由定理 5.4，取 profinite 空间 $Q$ 和满射 $Q\to K$。则 $Q\to K$ 本身就是用 profinite 对象覆盖 $K$，满足基条件。若需要处理纤维积，相容性由引理 5.5 保证。证毕。

## 5.4 凝聚集合的 profinite 定义

由站点比较定理得到：

**定理 5.7.** 限制函子给出范畴等价

$$
\operatorname{Sh}(\mathbf{CHaus},J_{\operatorname{surj}})
\simeq
\operatorname{Sh}(\mathbf{ProFin},J_{\operatorname{surj}}).
$$

因此也可以定义

$$
\mathbf{CondSet}
\simeq
\operatorname{Sh}(\mathbf{ProFin},J_{\operatorname{surj}}).
$$

在 profinite 口径下，一个凝聚集合是反变函子

$$
X:\mathbf{ProFin}^{\operatorname{op}}\to\mathbf{Set}
$$

满足有限联合满射覆盖的 sheaf 条件：

$$
X(S)\to \prod_i X(S_i)
\rightrightarrows
\prod_{i,j}X(S_i\times_S S_j)
$$

是等化子。

**注 5.8.** 有些资料直接把凝聚集合定义为 profinite 集合站点上的 sheaf。本书从 $\mathbf{CHaus}$ 开始，是为了让拓扑空间 $T\mapsto\underline T$ 的构造更直观；定理 5.7 说明这不会改变最终范畴。

## 5.5 口径差异如何使用

后续章节采用如下约定：

- 写定义和直观例子时，可以使用 $\mathbf{CHaus}$。
- 做计算时，优先使用 $\mathbf{ProFin}$ 或极不连通空间。
- 若一个断言只在 profinite 测试对象上验证，必须说明它如何由站点比较提升到 $\mathbf{CondSet}$。

例如，对拓扑空间 $T$，其凝聚集合可写为

$$
\underline T(S)=\operatorname{Cont}(S,T),
\qquad S\in\mathbf{CHaus},
$$

也可限制为

$$
\underline T(P)=\operatorname{Cont}(P,T),
\qquad P\in\mathbf{ProFin}.
$$

第二种写法通常更便于计算，但第一种写法更清楚地说明它来自拓扑空间。

## 5.6 本章小结

本章说明了凝聚集合定义的两个等价口径：

$$
\operatorname{Sh}(\mathbf{CHaus},J_{\operatorname{surj}})
\simeq
\operatorname{Sh}(\mathbf{ProFin},J_{\operatorname{surj}}).
$$

关键输入是：每个紧 Hausdorff 空间都可由 profinite 空间满射覆盖。这一事实引向第六章的极不连通空间。

## 练习

**练习 5.1.** 证明引理 5.5 中 $P\times_K Q$ 是 $P\times Q$ 的闭子空间。

**练习 5.2.** 假设定理 5.4，证明任意紧 Hausdorff 空间 $K$ 的可表凝聚集合 $\underline K$ 由其在 profinite 空间上的取值决定。

**练习 5.3.** 查阅一般 sheaf 理论，写出站点比较定理的完整证明。

**练习 5.4.** 解释为什么“全子范畴对象覆盖所有对象”还不足以推出 sheaf 范畴等价，还需要拉回后的覆盖条件。
