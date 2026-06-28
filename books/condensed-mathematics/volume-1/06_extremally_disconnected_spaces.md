# 第六章：极不连通空间

## 本章目标

本章介绍极不连通紧 Hausdorff 空间（extremally disconnected compact Hausdorff spaces）。它们在凝聚数学中扮演“投射测试对象”的角色：覆盖它们的满射可以分裂，凝聚阿贝尔群在它们上取值具有良好的正合性。

## 依赖前置知识

需要第二章的紧 Hausdorff 空间与商映射，第五章的 profinite 测试站点，以及第四章的凝聚阿贝尔群。

## 6.1 定义

**定义 6.1.** 拓扑空间 $E$ 称为极不连通（extremally disconnected），如果任意开集 $U\subset E$ 的闭包 $\overline U$ 仍是开集。

若 $E$ 同时是紧 Hausdorff 空间，则称 $E$ 为极不连通紧 Hausdorff 空间。

**例 6.2.** 有限离散空间极不连通。更一般地，任何离散空间极不连通。

**例 6.3.** Cantor 集是 profinite 空间，但不是极不连通空间。极不连通比 profinite 更强。

**注 6.4.** 紧 Hausdorff 极不连通空间常称为 Stonean space。它们与 Stone 对偶理论和完备 Boolean 代数密切相关；本书只使用其投射性和覆盖存在性。

## 6.2 投射性

极不连通空间的重要性来自下面的经典定理。

**定理 6.5（Gleason）.** 设 $E$ 是紧 Hausdorff 空间。以下条件等价：

1. $E$ 极不连通。
2. 对任意紧 Hausdorff 空间满射 $q:X\to Y$ 和任意连续映射 $f:E\to Y$，存在连续映射 $\tilde f:E\to X$ 使得
   $$
   q\circ \tilde f=f.
   $$

也就是说，极不连通紧 Hausdorff 空间正是 $\mathbf{CHaus}$ 中关于满射的投射对象。

**证明说明.** 这是 Gleason 的经典定理。完整证明需要 compact Hausdorff 空间的投射覆盖理论。本书把它作为基础引用；后续只使用其提升性质。依赖来源见 [SOURCES.md](SOURCES.md) 中 Gleason 与 Johnstone 条目。

**推论 6.6.** 若 $E$ 极不连通，且 $q:X\to E$ 是紧 Hausdorff 空间满射，则 $q$ 有连续截面

$$
\sigma:E\to X,
\qquad q\circ\sigma=\operatorname{id}_E.
$$

**证明.** 在定理 6.5 中取 $Y=E$，$f=\operatorname{id}_E$。证毕。

## 6.3 覆盖在极不连通空间上分裂

**命题 6.7.** 设 $E$ 极不连通，且

$$
\{S_i\to E\}_{i=1}^n
$$

是有限联合满射覆盖。则诱导满射

$$
q:\coprod_i S_i\to E
$$

有连续截面。

**证明.** 有限不交并 $\coprod_i S_i$ 是紧 Hausdorff 空间，$q$ 是满射连续映射。由推论 6.6 得到截面。证毕。

截面 $\sigma:E\to\coprod_i S_i$ 等价于把 $E$ 分成有限个 clopen 部分，并在每部分上选择一个 $S_i$ 中的提升。因为 $\coprod_iS_i$ 的每个分量既开又闭，$\sigma^{-1}(S_i)$ 是 $E$ 的 clopen 子集。

**注 6.8.** 这就是极不连通空间在 sheaf 计算中好用的原因。一般覆盖只给局部数据；在极不连通测试对象上，局部提升常可通过截面拉回成全局提升。

## 6.4 ED 测试对象足够多

**定理 6.9（Gleason 覆盖）.** 对任意紧 Hausdorff 空间 $K$，存在极不连通紧 Hausdorff 空间 $E$ 和满射

$$
E\to K.
$$

**证明说明.** 这是 compact Hausdorff 空间的投射覆盖存在定理，也称 Gleason cover。完整证明不在本书第一卷展开；本书后续凡使用 ED 覆盖，都依赖此引用结果。

**推论 6.10.** 若凝聚集合 $X$ 满足对所有极不连通 $E$ 都有

$$
X(E)=\varnothing,
$$

则 $X$ 是初始 sheaf，即对每个 $S\in\mathbf{CHaus}$ 有 $X(S)=\varnothing$。

**证明.** 若存在 $x\in X(S)$，取极不连通覆盖 $q:E\to S$。限制得到

$$
x|_E\in X(E),
$$

矛盾。因此所有 $X(S)$ 为空。证毕。

类似地，若凝聚集合态射 $X\to Y$ 在所有极不连通 $E$ 上诱导单射，则它是单射；若局部满射性也能在极不连通覆盖上检验，则它可用于判断满射。这在阿贝尔群值情形尤其有效。

## 6.5 凝聚阿贝尔群在 ED 上取值的正合性

**定理 6.11.** 设 $E$ 是极不连通紧 Hausdorff 空间。取值函子

$$
\operatorname{ev}_E:\mathbf{CondAb}\to\mathbf{Ab},
\qquad A\mapsto A(E)
$$

是正合函子。

**证明.** 左正合性来自 sheaf 范畴中核的逐点计算：若

$$
0\to A'\to A\to A''
$$

左正合，则对任意 $S$，序列

$$
0\to A'(S)\to A(S)\to A''(S)
$$

左正合。

需要证明满射在 $E$ 上仍满。设 $f:A\to B$ 是 $\mathbf{CondAb}$ 中的满射，取 $b\in B(E)$。sheaf 范畴中的满射具有局部提升性质：存在覆盖

$$
\{S_i\to E\}_{i=1}^n
$$

以及元素 $a_i\in A(S_i)$，使得

$$
f(a_i)=b|_{S_i}.
$$

令 $q:\coprod_iS_i\to E$。由命题 6.7，存在截面 $\sigma:E\to\coprod_iS_i$。局部元素 $a_i$ 合成元素

$$
\tilde a\in A(\coprod_iS_i)
\cong \prod_i A(S_i).
$$

沿 $\sigma$ 限制，得

$$
a=\sigma^*(\tilde a)\in A(E).
$$

由于 $q\circ\sigma=\operatorname{id}_E$，并且 $f(\tilde a)=q^*b$，自然性给出

$$
f(a)=b.
$$

因此 $A(E)\to B(E)$ 满射，取值函子正合。证毕。

**注 6.12.** 这个定理是凝聚数学中极不连通空间的核心用途之一。它把 sheaf 范畴中的局部正合性转化为在特殊测试对象上的逐点正合性。

## 6.6 本章小结

极不连通紧 Hausdorff 空间具有两种关键性质：

1. 它们是 $\mathbf{CHaus}$ 中关于满射的投射对象。
2. 每个紧 Hausdorff 空间都可由它们满射覆盖。

因此，它们足以测试凝聚对象，并使凝聚阿贝尔群的取值函子正合。

## 练习

**练习 6.1.** 证明任意离散空间是极不连通的。

**练习 6.2.** 设 $E$ 极不连通，$\{S_i\to E\}$ 为有限联合满射覆盖。证明截面 $\sigma:E\to\coprod_iS_i$ 给出 $E$ 的有限 clopen 分解。

**练习 6.3.** 在定理 6.11 的证明中，详细验证 $f(a)=b$ 的自然性计算。

**练习 6.4.** 查阅 Gleason 定理，说明极不连通性如何与 compact Hausdorff 范畴的投射对象联系起来。
