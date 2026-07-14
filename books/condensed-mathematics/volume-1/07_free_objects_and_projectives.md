# 第七章：自由对象与投射生成元

第六章已经证明，对极不连通空间 $E$，取值函子 $A\mapsto A(E)$ 保持短正合列。
要把这一观察纳入同调代数，需要找出表示该函子的凝聚阿贝尔群。自由对象的伴随性与
Yoneda 引理给出答案：$\mathbb Z[\underline E]$ 满足
$\operatorname{Hom}(\mathbb Z[\underline E],A)\cong A(E)$，于是取值正合恰好
等价于它的投射性。

单个表示对象还不足以构造任意对象的投射分解。利用极不连通覆盖和所有局部截面，
可以把这些自由对象的直和满射到任意凝聚阿贝尔群，随后对核重复同一构造。以下证明
依赖第四章的自由阿贝尔群对象、第五章的站点比较与第六章的正合取值，并最终得到

$$
\mathbb Z[\underline E]
$$

组成的一族投射生成元。

## 7.1 自由阿贝尔群函子

普通集合范畴中，有自由阿贝尔群函子

$$
\mathbb Z[-]:\mathbf{Set}\to\mathbf{Ab},
$$

它左伴随于忘却函子

$$
U:\mathbf{Ab}\to\mathbf{Set}.
$$

也就是说，对集合 $X$ 和阿贝尔群 $A$，有自然双射

$$
\operatorname{Hom}_{\mathbf{Ab}}(\mathbb Z[X],A)
\cong
\operatorname{Hom}_{\mathbf{Set}}(X,U(A)).
$$

凝聚阿贝尔群中的自由对象是这个伴随的 sheaf 版本。

## 7.2 构造自由凝聚阿贝尔群

设 $X\in\mathbf{CondSet}$。先定义预层

$$
P_X:\mathbf{CHaus}^{\operatorname{op}}\to\mathbf{Ab}
$$

为

$$
P_X(S)=\mathbb Z[X(S)].
$$

这通常只是阿贝尔群值预层，不一定是 sheaf。

**定义 7.1.** 由凝聚集合 $X$ 生成的自由凝聚阿贝尔群定义为 $P_X$ 的 sheafification：

$$
\mathbb Z[X]=a(P_X),
$$

其中

$$
a:\operatorname{PSh}(\mathbf{CHaus};\mathbf{Ab})
\to
\mathbf{CondAb}
$$

是 sheafification 函子。

**命题 7.2.** 函子

$$
\mathbb Z[-]:\mathbf{CondSet}\to\mathbf{CondAb}
$$

左伴随于忘却函子

$$
U:\mathbf{CondAb}\to\mathbf{CondSet}.
$$

也就是说，有自然双射

$$
\operatorname{Hom}_{\mathbf{CondAb}}(\mathbb Z[X],A)
\cong
\operatorname{Hom}_{\mathbf{CondSet}}(X,U(A)).
$$

**证明.** 由 sheafification 的泛性质，

$$
\operatorname{Hom}_{\mathbf{CondAb}}(a(P_X),A)
\cong
\operatorname{Hom}_{\operatorname{PSh}(\mathbf{CHaus};\mathbf{Ab})}(P_X,A).
$$

右侧按逐点自由阿贝尔群伴随等价于给出自然变换

$$
X\to U(A).
$$

即

$$
\operatorname{Hom}_{\mathbf{CondSet}}(X,U(A)).
$$

自然性由各层伴随和 sheafification 泛性质给出。证毕。

## 7.3 可表对象生成的自由对象

设 $S\in\mathbf{CHaus}$，记 $\underline S=h_S$。由 Yoneda 引理和命题 7.2，有：

**命题 7.3.** 对任意 $A\in\mathbf{CondAb}$，有自然同构

$$
\operatorname{Hom}_{\mathbf{CondAb}}(\mathbb Z[\underline S],A)
\cong
A(S).
$$

**证明.**

$$
\operatorname{Hom}_{\mathbf{CondAb}}(\mathbb Z[\underline S],A)
\cong
\operatorname{Hom}_{\mathbf{CondSet}}(\underline S,U(A))
\cong
U(A)(S)
=A(S),
$$

其中第二个同构是 Yoneda 引理。证毕。

这条公式非常重要。它说明 $\mathbb Z[\underline S]$ 是“在 $S$ 上取值”函子的表示对象。

## 7.4 极不连通空间给出投射对象

**定义 7.4.** 阿贝尔范畴 $\mathcal A$ 中对象 $P$ 称为投射对象，如果函子

$$
\operatorname{Hom}_{\mathcal A}(P,-):\mathcal A\to\mathbf{Ab}
$$

是正合函子。

**定理 7.5.** 若 $E$ 是极不连通紧 Hausdorff 空间，则

$$
\mathbb Z[\underline E]
$$

是 $\mathbf{CondAb}$ 中的投射对象。

**证明.** 由命题 7.3，

$$
\operatorname{Hom}_{\mathbf{CondAb}}(\mathbb Z[\underline E],-)
\cong
\operatorname{ev}_E.
$$

第六章定理 6.11 已证明 $\operatorname{ev}_E$ 是正合函子。因此 $\mathbb Z[\underline E]$ 投射。证毕。

## 7.5 足够多的投射对象

**定义 7.6.** 阿贝尔范畴 $\mathcal A$ 有足够多的投射对象，如果对任意对象 $A\in\mathcal A$，存在投射对象 $P$ 和满射

$$
P\to A.
$$

**定理 7.7.** $\mathbf{CondAb}$ 有足够多的投射对象。更具体地，每个 $A\in\mathbf{CondAb}$ 都存在形如

$$
\bigoplus_{\alpha}\mathbb Z[\underline{E_\alpha}]
\to A
$$

的满射，其中每个 $E_\alpha$ 都是极不连通紧 Hausdorff 空间。

**证明.** 对所有二元组 $(E,a)$ 取直和，其中 $E$ 遍历一个代表集合中的极不连通紧 Hausdorff 空间，$a\in A(E)$。由 $a\in A(E)$ 和命题 7.3，对应一个态射

$$
\mathbb Z[\underline E]\to A.
$$

把这些态射相加，得到

$$
\Phi:\bigoplus_{(E,a)}\mathbb Z[\underline E]\to A.
$$

需要证明 $\Phi$ 是满射。按 sheaf 满射的局部判据，取任意 $S\in\mathbf{CHaus}$ 与 $b\in A(S)$。由定理 6.9，存在极不连通覆盖 $q:E\to S$。限制得到

$$
q^*b\in A(E).
$$

在直和中有对应生成态射

$$
\mathbb Z[\underline E]\to A
$$

代表元素 $q^*b$。因此 $b$ 在覆盖 $E\to S$ 上局部来自 $\Phi$ 的源。故 $\Phi$ 为 sheaf 意义下的满射。源是投射对象的直和；投射对象直和仍投射，因为 Hom 从直和出发变为乘积，而 $\mathbf{Ab}$ 中乘积保持满射。证毕。

**注 7.8.** 证明中“代表集合”隐藏了 universe 选择。严格处理大小问题时，应固定足够大的 Grothendieck universe，或采用 $\kappa$-小版本的凝聚数学。本书第一卷按 [NOTATION.md](NOTATION.md) 的约定处理。

## 7.6 投射分解

定理 7.7 允许对任意凝聚阿贝尔群 $A$ 构造投射分解。先取满射

$$
P_0\to A,
$$

令 $K_1=\ker(P_0\to A)$。再取满射

$$
P_1\to K_1.
$$

如此继续，得到链复形

$$
\cdots\to P_2\to P_1\to P_0\to A\to 0,
$$

其中每个 $P_i$ 是投射对象。

这是后续定义导出函子和 $\operatorname{Ext}$ 的基础。

## 7.7 取值函子的表示对象

自由凝聚阿贝尔群把 Yoneda 取值公式提升为

$$
\operatorname{Hom}(\mathbb Z[\underline S],A)\cong A(S).
$$

当 $S=E$ 极不连通时，右侧正合，故 $\mathbb Z[\underline E]$ 投射；极不连通覆盖
又使这些对象的直和能够满射到任意 $A$。反复覆盖核便得到投射分解。第八章因而可以
在不假设逐点余核的前提下，用这些表示对象检测正合性并定义
$\operatorname{Ext}$。

## 练习

**练习 7.1.** 证明命题 7.2 中自然双射的两个方向，并检查它们互逆。

**练习 7.2.** 设 $E$ 极不连通。用定理 6.11 直接证明 $\mathbb Z[\underline E]$ 投射。

**练习 7.3.** 在定理 7.7 的证明中，详细写出为什么局部提升意味着 $\Phi$ 是 sheaf 意义下的满射。

**练习 7.4.** 证明阿贝尔范畴中任意投射对象族的直和仍投射，前提是相关直和存在且 Hom 将直和转成乘积。
