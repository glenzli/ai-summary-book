# 第三章：凝聚集合

## 本章目标

本章正式定义凝聚集合（condensed set），并验证几个基本例子：可表凝聚集合、拓扑空间关联的凝聚集合、离散集合关联的凝聚集合。本章的核心公式是

$$
\mathbf{CondSet}
=
\operatorname{Sh}(\mathbf{CHaus},J_{\operatorname{surj}}).
$$

## 依赖前置知识

需要第一章的 sheaf 条件，以及第二章中有限联合满射覆盖和子典范性定理。

## 3.1 定义

**定义 3.1.** 凝聚集合是站点

$$
(\mathbf{CHaus},J_{\operatorname{surj}})
$$

上的集合值 sheaf。也就是说，它是一个反变函子

$$
X:\mathbf{CHaus}^{\operatorname{op}}\to \mathbf{Set}
$$

满足：对每个有限联合满射覆盖

$$
\{S_i\to S\}_{i=1}^n,
$$

序列

$$
X(S)\longrightarrow \prod_{i=1}^n X(S_i)
\rightrightarrows
\prod_{i,j}X(S_i\times_S S_j)
$$

是 $\mathbf{Set}$ 中的等化子。

凝聚集合范畴记为

$$
\mathbf{CondSet}.
$$

**注 3.2.** 这个定义有两个层次。第一，$X$ 是预层，因此每个紧 Hausdorff 空间 $S$ 都给出一个集合 $X(S)$。第二，$X$ 满足 sheaf 条件，因此 $X(S)$ 中的元素可以由覆盖 $S$ 的测试对象上的相容局部元素唯一粘合。

不要把 $X(S)$ 理解为“$S$ 中的点”。更合适的说法是：$S$ 是测试空间，$X(S)$ 是 $S$ 参数化的 $X$-族。

## 3.2 可表凝聚集合

由定理 2.8，$\mathbf{CHaus}$ 的每个对象都给出凝聚集合。

**定义 3.3.** 对 $K\in\mathbf{CHaus}$，记

$$
\underline K=h_K,
\qquad
\underline K(S)=\operatorname{Hom}_{\mathbf{CHaus}}(S,K).
$$

这称为由紧 Hausdorff 空间 $K$ 表示的凝聚集合。

**命题 3.4.** 函子

$$
\mathbf{CHaus}\to \mathbf{CondSet},
\qquad
K\mapsto \underline K
$$

是全忠实的。

**证明.** 由 Yoneda 引理，在预层范畴中有自然双射

$$
\operatorname{Hom}_{\widehat{\mathbf{CHaus}}}(h_K,h_L)
\cong h_L(K)
=\operatorname{Hom}_{\mathbf{CHaus}}(K,L).
$$

由于 $h_K,h_L$ 都是 sheaf，且 sheaf 范畴是预层范畴的满子范畴，二者之间的自然变换集合不变。因此该函子全忠实。证毕。

**注 3.5.** 这说明紧 Hausdorff 空间没有在凝聚化过程中丢失。凝聚集合范畴包含 $\mathbf{CHaus}$ 的一个全忠实影像，但比 $\mathbf{CHaus}$ 大得多，也更适合代数操作。

## 3.3 拓扑空间给出的凝聚集合

紧 Hausdorff 空间不是唯一能进入凝聚世界的拓扑对象。

**定义 3.6.** 设 $T$ 是拓扑空间。定义预层

$$
\underline T:\mathbf{CHaus}^{\operatorname{op}}\to \mathbf{Set}
$$

为

$$
\underline T(S)=\operatorname{Cont}(S,T),
$$

其中 $\operatorname{Cont}$ 表示连续映射集合。若 $f:S'\to S$ 是连续映射，则限制映射为预合成：

$$
\underline T(S)\to \underline T(S'),
\qquad
\varphi\mapsto \varphi\circ f.
$$

**命题 3.7.** 对任意拓扑空间 $T$，预层 $\underline T$ 是凝聚集合。

**证明.** 设 $\{S_i\to S\}_{i=1}^n$ 是有限联合满射覆盖，并令

$$
q:\coprod_i S_i\to S
$$

为诱导商映射。给定相容族

$$
\varphi_i:S_i\to T,
$$

相容性表示在 $S_i\times_S S_j$ 上有

$$
\varphi_i\circ \operatorname{pr}_1
=
\varphi_j\circ \operatorname{pr}_2.
$$

于是它们给出连续映射

$$
\tilde\varphi:\coprod_i S_i\to T
$$

并且 $\tilde\varphi$ 在 $q$ 的纤维上为常值。因此存在唯一集合映射

$$
\varphi:S\to T
$$

使得

$$
\tilde\varphi=\varphi\circ q.
$$

由推论 2.7，$q$ 是商映射。由于 $\tilde\varphi$ 连续，商拓扑的定义推出 $\varphi$ 连续。唯一性由 $q$ 满射得到。故 sheaf 条件成立。证毕。

**注 3.8.** 命题 3.7 的证明不要求 $T$ 紧或 Hausdorff。紧 Hausdorff 条件加在测试空间 $S$ 上，而不是目标空间 $T$ 上。

## 3.4 离散集合

**定义 3.9.** 设 $A$ 是集合，赋予离散拓扑。其对应凝聚集合仍记为

$$
\underline A.
$$

对 $S\in\mathbf{CHaus}$，

$$
\underline A(S)=\operatorname{Cont}(S,A_{\operatorname{disc}}).
$$

由于 $A$ 离散，连续映射 $S\to A$ 等价于 $S$ 被 clopen 子集分解并在每个分解块上取常值。

**例 3.10.** 若 $S$ 连通，则任意连续映射 $S\to A_{\operatorname{disc}}$ 为常值。因此

$$
\underline A(S)\cong A
$$

当 $S$ 非空且连通时成立。

若 $S$ 是有限离散集合，有 $m$ 个点，则

$$
\underline A(S)\cong A^m.
$$

这说明离散集合在凝聚世界中并不只是一个常值函子；它仍然看见测试空间的 clopen 分解。

## 3.5 点与全局截面

**定义 3.11.** 凝聚集合 $X$ 的全局截面集合定义为

$$
\Gamma(X)=X(*),
$$

其中 $*$ 是单点紧 Hausdorff 空间。

若 $X=\underline T$ 来自拓扑空间 $T$，则

$$
\Gamma(\underline T)=\operatorname{Cont}(*,T)
\cong |T|,
$$

即 $T$ 的底层集合。

**注 3.12.** 全局截面只看见点，不看见拓扑。拓扑信息隐藏在所有测试空间 $S$ 上的集合 $X(S)$ 以及限制映射中。例如两个拓扑空间若底层集合相同，但连续映射 $S\to T$ 的集合随 $S$ 不同而不同，则它们给出不同凝聚集合。

## 3.6 凝聚集合的态射

**定义 3.13.** 凝聚集合 $X,Y$ 之间的态射是 sheaf 之间的自然变换

$$
\eta:X\to Y.
$$

也就是说，对每个 $S\in\mathbf{CHaus}$，有映射

$$
\eta_S:X(S)\to Y(S),
$$

并且对每个 $f:S'\to S$，交换图
$$
Y(f)\circ \eta_S=\eta_{S'}\circ X(f).
$$

成立。这就是自然变换的自然性条件。

**命题 3.14.** 对拓扑空间 $T,U$，任意连续映射 $a:T\to U$ 给出凝聚集合态射

$$
\underline a:\underline T\to \underline U.
$$

**证明.** 对每个 $S\in\mathbf{CHaus}$，定义

$$
\underline a_S:\operatorname{Cont}(S,T)\to \operatorname{Cont}(S,U),
\qquad
\varphi\mapsto a\circ \varphi.
$$

若 $f:S'\to S$，则

$$
(a\circ \varphi)\circ f=a\circ(\varphi\circ f).
$$

因此自然性成立，得到凝聚集合态射。证毕。

## 3.7 本章小结

本章完成了凝聚集合的正式定义：

$$
\mathbf{CondSet}
=
\operatorname{Sh}(\mathbf{CHaus},J_{\operatorname{surj}}).
$$

我们证明了三类基本对象：

1. 每个紧 Hausdorff 空间 $K$ 给出可表凝聚集合 $\underline K$。
2. 每个拓扑空间 $T$ 给出凝聚集合 $\underline T$。
3. 每个离散集合 $A$ 给出凝聚集合 $\underline A$，其值记录测试空间的 clopen 分解。

下一章将把集合值 sheaf 替换为阿贝尔群值 sheaf，定义凝聚阿贝尔群。

## 练习

**练习 3.1.** 设 $T$ 是拓扑空间。详细检查命题 3.7 中 $\varphi:S\to T$ 连续性的商拓扑论证。

**练习 3.2.** 设 $A$ 为离散集合，$S=S_1\sqcup S_2$ 为两个非空 clopen 子空间的不交并。证明

$$
\underline A(S)\cong \underline A(S_1)\times \underline A(S_2).
$$

**练习 3.3.** 设 $K,L\in\mathbf{CHaus}$。用 Yoneda 引理证明

$$
\operatorname{Hom}_{\mathbf{CondSet}}(\underline K,\underline L)
\cong
\operatorname{Hom}_{\mathbf{CHaus}}(K,L).
$$

**练习 3.4.** 给出两个底层集合相同但拓扑不同的空间 $T,U$，并说明为什么 $\underline T$ 与 $\underline U$ 可能不同。
