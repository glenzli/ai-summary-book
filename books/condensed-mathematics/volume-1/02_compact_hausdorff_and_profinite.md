# 第二章：紧 Hausdorff 空间与 profinite 空间

第一章的 sheaf 条件只有在覆盖可拉回、可复合且局部映射能够沿商粘合时才真正可用。
紧 Hausdorff 空间恰好同时提供这些性质：有限极限仍留在该类中，有限个映射的联合
满射可合并成紧空间到 Hausdorff 空间的商映射，而商映射能把纤维上相容的连续函数
降到目标空间。这里的拓扑事实将直接承担凝聚站点的预拓扑公理与子典范性证明。

计算时却不希望每次面对任意紧 Hausdorff 空间。profinite 空间作为有限离散空间的
逆极限，保留足够的覆盖信息并带来 clopen 分解；它们后来还可由极不连通空间进一步
细化。以下论证使用紧致、Hausdorff、商拓扑、闭映射、全不连通性与逆极限的标准
性质，并在固定的大小层级中进行。

## 2.1 紧 Hausdorff 空间范畴

**定义 2.1.** 记 $\mathbf{CHaus}_\kappa$ 为底层集合基数小于附录 A 中固定
\(\kappa\) 的紧 Hausdorff 空间骨架，态射为连续映射。本章简写为
\(\mathbf{CHaus}\)；所有极限和覆盖仍在这个固定层级内计算。

有限极限不会越过该层级：有限个基数小于无限基数 \(\kappa\) 的集合，其有限乘积和
子集仍有基数小于 \(\kappa\)。

**命题 2.2.** $\mathbf{CHaus}$ 有有限极限。

**证明.** 终对象是单点空间 $*$，它紧且 Hausdorff。若 $X,Y$ 是紧 Hausdorff 空间，则乘积 $X\times Y$ 仍紧且 Hausdorff。若有两条连续映射

$$
f,g:X\to Y,
$$

则等化子为

$$
\operatorname{Eq}(f,g)=\{x\in X\mid f(x)=g(x)\}.
$$

因为 $Y$ Hausdorff，对角线 $\Delta_Y\subset Y\times Y$ 闭，而 $(f,g):X\to Y\times Y$ 连续，所以等化子是 $X$ 的闭子空间。闭子空间仍紧且 Hausdorff。有限极限可由终对象、有限乘积和等化子构造。证毕。

## 2.2 有限联合满射覆盖

**定义 2.3.** 设 $S\in\mathbf{CHaus}$。一族有限态射

$$
\{S_i\to S\}_{i=1}^n,\qquad n\ge 0,
$$

称为有限联合满射覆盖，如果诱导映射

$$
\coprod_{i=1}^n S_i\longrightarrow S
$$

在底层集合上是满射。

这里 $\coprod$ 是有限不交并。有限不交并的紧 Hausdorff 空间仍是紧 Hausdorff 空间。
当 \(n=0\) 时，源是空空间；因此空族恰好覆盖 \(S=\varnothing\)。

**来源定位.** 这一口径对应 S26 Definitions 1.2 与 2.1 的有限联合满射拓扑；其中
sheaf 的显式条件包括 \(F(\varnothing)=*\)。附录 A 解释了本书的基数截断。

**命题 2.4.** 有限联合满射覆盖在拉回下稳定。

**证明.** 设 $\{S_i\to S\}_{i=1}^n$ 是有限联合满射覆盖，且 $T\to S$ 是任意连续映射。需要证明

$$
\{T\times_S S_i\to T\}_{i=1}^n
$$

联合满射。

若 \(n=0\)，则 \(S=\varnothing\)，从而存在 \(T\to S\) 强制
\(T=\varnothing\)；拉回后的空族覆盖 \(T\)。以下设 \(n>0\)。

任取 $t\in T$。其像为 $s\in S$。由于 $\coprod_i S_i\to S$ 满射，存在某个 $i$ 和 $s_i\in S_i$ 使得 $s_i$ 映到 $s$。于是 $(t,s_i)\in T\times_S S_i$ 映到 $t$。故拉回族联合满射。原覆盖族只有有限多个对象，拉回后仍只有同样有限多个对象，因此有限性保持。证毕。

**命题 2.5.** 有限联合满射覆盖满足复合稳定性。

**证明.** 设 $\{S_i\to S\}_{i=1}^n$ 联合满射，且对每个 $i$，$\{S_{ij}\to S_i\}_{j=1}^{m_i}$ 联合满射。任取 $s\in S$。先选 $i$ 与 $s_i\in S_i$ 映到 $s$，再选 $j$ 与 $s_{ij}\in S_{ij}$ 映到 $s_i$。则 $s_{ij}$ 经复合映到 $s$。所以复合族联合满射。证毕。

由此，有限联合满射覆盖给出 $\mathbf{CHaus}$ 上的预拓扑。记该站点为

$$
(\mathbf{CHaus},J_{\operatorname{surj}}).
$$

## 2.3 满射与商映射

后续证明可表预层是 sheaf 时，需要一个基础拓扑事实。

**引理 2.6.** 若 $q:X\to Y$ 是紧 Hausdorff 空间之间的满射连续映射，则 $q$ 是闭映射，因此是商映射。

**证明.** 设 $C\subset X$ 闭。因为 $X$ 紧，$C$ 紧。连续映射保持紧性，所以 $q(C)$ 是 $Y$ 中紧子集。由于 $Y$ Hausdorff，紧子集闭。因此 $q$ 是闭映射。

连续满射闭映射是商映射：若 $U\subset Y$ 且 $q^{-1}(U)$ 开，则 $X\setminus q^{-1}(U)=q^{-1}(Y\setminus U)$ 闭。因为 $q$ 是闭映射，$q(q^{-1}(Y\setminus U))=Y\setminus U$ 闭，故 $U$ 开。证毕。

**推论 2.7.** 若 $\{S_i\to S\}_{i=1}^n$ 是有限联合满射覆盖，则

$$
q:\coprod_{i=1}^n S_i\to S
$$

是商映射。

**证明.** 有限不交并 $\coprod_i S_i$ 紧 Hausdorff，$q$ 是紧 Hausdorff 空间之间的满射连续映射。由引理 2.6 得证。证毕。

## 2.4 可表预层是 sheaf

**定理 2.8.** 站点 $(\mathbf{CHaus},J_{\operatorname{surj}})$ 是子典范的。也就是说，对每个 $K\in\mathbf{CHaus}$，可表预层

$$
h_K(S)=\operatorname{Hom}_{\mathbf{CHaus}}(S,K)
$$

是 sheaf。

**证明.** 设 $\{S_i\to S\}_{i=1}^n$ 是有限联合满射覆盖。令

$$
q:\coprod_i S_i\to S
$$

为诱导商映射。

首先证明唯一性。若 $f,g:S\to K$ 是连续映射，且对每个 $i$ 有

$$
f|_{S_i}=g|_{S_i},
$$

则由于 $\coprod_i S_i\to S$ 满射，$f$ 与 $g$ 在每个点上相等，故 $f=g$。

再证明存在性。设给定连续映射族

$$
f_i:S_i\to K
$$

并且在每个纤维积 $S_i\times_S S_j$ 上相容：

$$
f_i\circ \operatorname{pr}_1
=
f_j\circ \operatorname{pr}_2.
$$

这些映射合成一个连续映射

$$
\tilde f:\coprod_i S_i\to K.
$$

相容条件说明：若两个点在 $\coprod_i S_i$ 中经 $q$ 映到 $S$ 的同一点，则它们在 $\tilde f$ 下有相同像。因此存在唯一集合映射

$$
f:S\to K
$$

使得

$$
\tilde f=f\circ q.
$$

由于 $q$ 是商映射，且 $\tilde f$ 连续，按商拓扑定义，$f$ 连续。故 $f\in h_K(S)$，并且限制到每个 $S_i$ 等于 $f_i$。于是 sheaf 条件成立。证毕。

这一定理解释了为什么有限联合满射覆盖是正确的覆盖：它使紧 Hausdorff 空间本身通过 Yoneda 嵌入成为凝聚集合。

## 2.5 Profinite 空间

**定义 2.9.** 一个 profinite 空间是有限离散空间的逆极限。等价地，它是紧、Hausdorff、全不连通的拓扑空间。

记固定层级的 profinite 空间范畴为 $\mathbf{ProFin}_\kappa$，并在本卷中简写为
\(\mathbf{ProFin}\)。

**注 2.10.** “全不连通”有不同强弱版本。本书采用紧 Hausdorff 情形中的标准等价表述：连通分支均为单点，并且具有足够多的 clopen 子集分离点。严格证明属于 Stone 对偶理论，本章只记录后续需要的结论。

**命题 2.11.** 有限离散空间是 profinite 空间；profinite 空间的闭子空间和有限乘积仍是 profinite 空间。

**证明.** 有限离散空间是自身构成的平凡逆系统极限。若 $X=\varprojlim_i X_i$ 与 $Y=\varprojlim_j Y_j$，其中 $X_i,Y_j$ 有限离散，则

$$
X\times Y\simeq \varprojlim_{i,j}(X_i\times Y_j),
$$

而 $X_i\times Y_j$ 有限离散。闭子空间仍紧 Hausdorff；全不连通性由子空间继承。故闭子空间仍 profinite。证毕。

## 2.6 为什么 profinite 空间会出现

凝聚数学有多个等价或近似等价的测试范畴口径。紧 Hausdorff 空间最直观，因为任何拓扑空间 $T$ 都给出

$$
S\mapsto \operatorname{Cont}(S,T),
\qquad S\in\mathbf{CHaus}.
$$

但 profinite 空间更接近代数和逻辑：它们由有限集合逆极限构成，clopen 分解丰富，常常更适合计算 sheaf 条件。

本书后续会使用如下思想：

- $\mathbf{CHaus}$ 给出概念上自然的定义。
- $\mathbf{ProFin}$ 给出更可控的测试对象。
- 极不连通紧 Hausdorff 空间给出类似投射对象的计算工具。

这些说法暂时只是路线图。正式证明需要更多 sheaf 理论和 Stone 对偶背景。

## 2.7 商映射使站点成立

有限极限保证覆盖可拉回，有限联合满射的稳定性给出预拓扑，而紧空间到 Hausdorff
空间的满射为商映射，正好把匹配族粘合成连续映射。因此

$$
(\mathbf{CHaus},J_{\operatorname{surj}}),
$$

不仅是一个合法站点，而且是子典范站点。profinite 空间则提供同一几何的有限离散
近似。下一章不再把这些结果当作背景说明，而直接用它们定义

$$
\mathbf{CondSet}_\kappa
=
\operatorname{Sh}(\mathbf{CHaus}_\kappa,J_{\operatorname{surj}}).
$$

## 练习

**练习 2.1.** 证明有限不交并的紧 Hausdorff 空间仍是紧 Hausdorff 空间。

**练习 2.2.** 证明 $\mathbf{CHaus}$ 中任意两个对象的纤维积可以看作乘积空间中的闭子空间。

**练习 2.3.** 在定理 2.8 的证明中，详细说明相容条件如何保证 $\tilde f$ 在 $q$ 的等价关系上为常值。

**练习 2.4.** 设 $A$ 为无限离散集合。它是否是紧 Hausdorff 空间？它是否是 profinite 空间？说明理由。

**练习 2.5.** 查阅 Stone 对偶理论，证明紧 Hausdorff 全不连通空间具有 clopen 基。

**练习 2.6.** 验证空族覆盖 \(\varnothing\) 在拉回与复合下满足预拓扑公理，并在
定理 2.8 中单独检查这个退化情形。
