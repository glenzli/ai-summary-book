# 附录 D：Stone 对偶与 Gleason cover

## D.0 目标

正文中多次使用 profinite 空间、极不连通紧 Hausdorff 空间和 Gleason cover。本附录给出这些事实在本书所需范围内的证明路线。Gleason 投射性定理本身作为经典拓扑输入使用；本附录把它与 Stone 对偶、regular open 代数和正文中的 sheaf 计算连接起来。

## D.1 Boolean 代数

**定义 D.1.** Boolean 代数是带有运算

$$
\wedge,\quad \vee,\quad \neg,\quad 0,\quad 1
$$

的集合 $B$，满足有限交、有限并、补元、分配律以及

$$
b\wedge\neg b=0,\qquad b\vee\neg b=1.
$$

若任意子集 $A\subset B$ 都有上确界 $\bigvee A$ 和下确界 $\bigwedge A$，则称 $B$ 是完备 Boolean 代数。

**例 D.2.** 若 $X$ 是拓扑空间，则开闭子集构成 Boolean 代数

$$
\operatorname{Clop}(X).
$$

若 $X$ 是离散有限集合，则

$$
\operatorname{Clop}(X)=\mathcal P(X).
$$

## D.2 Stone 空间

**定义 D.3.** 对 Boolean 代数 $B$，定义其 Stone 空间

$$
\operatorname{Stone}(B)
$$

为所有 Boolean 代数同态

$$
\varphi:B\to\{0,1\}
$$

构成的集合。对 $b\in B$，记

$$
U_b=\{\varphi\mid \varphi(b)=1\}.
$$

以 $U_b$ 为开闭基定义拓扑。

**命题 D.4.** $\operatorname{Stone}(B)$ 是紧 Hausdorff 且全不连通的空间。

**证明.** Hausdorff 性来自 Boolean 代数分离：若 $\varphi\ne\psi$，取 $b$ 使 $\varphi(b)\ne\psi(b)$，则 $U_b$ 与 $U_{\neg b}$ 分离二者。全不连通性来自 $U_b$ 构成开闭基。

紧性可用 Alexander 子基定理。若一族基本闭集具有有限交性质，它对应于 $B$ 中一个真滤子；由 Boolean 代数的素滤子定理，可扩张为超滤子，等价于一点 $\varphi:B\to\{0,1\}$。证毕。

**定理 D.5（Stone 对偶）.** 函子

$$
X\mapsto \operatorname{Clop}(X)
$$

给出 profinite 空间范畴与 Boolean 代数范畴反向等价。

**证明.** 对 Boolean 代数 $B$，映射

$$
B\to \operatorname{Clop}(\operatorname{Stone}(B)),\qquad b\mapsto U_b
$$

是 Boolean 代数同构。单射由超滤子分离元素给出；满射由 $U_b$ 构成紧空间的开闭基给出：任意开闭集被有限多个 $U_b$ 覆盖，故等于某个有限并 $U_{b_1\vee\cdots\vee b_n}$。

反向地，若 $X$ 是 profinite 空间，则点 $x\in X$ 给出同态

$$
\operatorname{Clop}(X)\to\{0,1\},\qquad U\mapsto 1_{x\in U}.
$$

这给出同胚

$$
X\cong \operatorname{Stone}(\operatorname{Clop}(X)).
$$

自然性直接来自连续映射对开闭集的逆像。证毕。

## D.3 Profinite 空间的等价刻画

**命题 D.6.** 对紧 Hausdorff 空间 $X$，以下条件等价：

1. $X$ 是 profinite 空间。
2. $X$ 是紧、Hausdorff、全不连通空间。
3. $X$ 可写为有限离散空间的逆极限：
   $$
   X\simeq \varprojlim_i X_i.
   $$

**证明.** $(3)\Rightarrow(2)$ 因有限离散空间紧、Hausdorff、全不连通，且这些性质在逆极限下保持。

$(2)\Rightarrow(3)$：对 $X$ 的有限开闭划分 $\mathcal P$，令 $X_{\mathcal P}$ 为划分块构成的有限离散集合。若 $\mathcal Q$ 细化 $\mathcal P$，有自然满射 $X_{\mathcal Q}\to X_{\mathcal P}$。所有有限开闭划分构成滤过系统，且自然映射

$$
X\to\varprojlim_{\mathcal P}X_{\mathcal P}
$$

是同胚。单射来自开闭集分离点；满射由紧性和有限交性质给出。$(1)$ 与 $(2)$ 的等价是定义约定。证毕。

## D.4 极不连通空间与完备 Boolean 代数

**定义 D.7.** 拓扑空间 $X$ 称为极不连通，如果任意开集 $U\subset X$ 的闭包 $\overline U$ 仍是开集。

紧 Hausdorff 的极不连通空间也常称为 Stonean 空间。

**命题 D.8.** 若 $B$ 是完备 Boolean 代数，则 $\operatorname{Stone}(B)$ 极不连通。反过来，若 $X$ 是极不连通 profinite 空间，则 $\operatorname{Clop}(X)$ 是完备 Boolean 代数。

**证明.** 设 $X=\operatorname{Stone}(B)$，$U\subset X$ 是开集。取所有满足 $U_b\subset U$ 的 $b$，令

$$
c=\bigvee_{U_b\subset U} b.
$$

则 $\overline U=U_c$。包含 $\overline U\subset U_c$ 由 $U\subset U_c$ 得到。反向包含用基本邻域检验：若 $\varphi(c)=1$ 且 $U_d$ 是 $\varphi$ 的基本邻域，则 $d\wedge c\ne0$；由完备性，

$$
d\wedge c=\bigvee_{U_b\subset U}(d\wedge b),
$$

故存在某个 $b$ 使 $d\wedge b\ne0$，于是 $U_d\cap U_b\ne\varnothing$，从而 $U_d\cap U\ne\varnothing$。这说明 $\varphi\in\overline U$。

反过来，若 $X$ 极不连通且 $\{V_i\}_{i\in I}$ 是开闭子集族，则

$$
\bigvee_i V_i=\overline{\bigcup_i V_i}
$$

是开闭集；下确界用补集转化为上确界。这给出任意上确界和下确界。证毕。

## D.5 Gleason cover

设 $X$ 是紧 Hausdorff 空间。记 $\operatorname{RO}(X)$ 为 $X$ 的 regular open subsets：

$$
U=\operatorname{int}\overline U.
$$

在运算

$$
U\wedge V=U\cap V,\qquad
U\vee V=\operatorname{int}\overline{U\cup V},\qquad
\neg U=\operatorname{int}(X\setminus U)
$$

下，$\operatorname{RO}(X)$ 是完备 Boolean 代数。

**定理 D.9（Gleason cover）.** 对任意紧 Hausdorff 空间 $X$，存在极不连通紧 Hausdorff 空间 $E_X$ 和满射

$$
p:E_X\to X.
$$

可取

$$
E_X=\operatorname{Stone}(\operatorname{RO}(X)).
$$

**证明.** 由命题 D.8，$E_X$ 极不连通。若 $\mathfrak u$ 是 $\operatorname{RO}(X)$ 上的超滤子，则闭集族

$$
\{\overline U\mid U\in\mathfrak u\}
$$

有有限交性质，因此交非空。Hausdorff 正则性和 regular open 集分离点可证明该交恰有一个点，记为 $p(\mathfrak u)$。这定义了映射 $p:E_X\to X$。

连续性可在开集基上检查：若 $V\subset X$ 是 regular open，则

$$
p^{-1}(V)\subset U_V
$$

并且对适当缩小的 regular open 子集可得到邻域控制。满射由每个 $x\in X$ 处的邻域滤子扩张为超滤子得到。完整证明还说明 $p$ 是 irreducible map，即任意真闭子集都不能满射到 $X$。

本书正文只需要满射存在性与 $E_X$ 的极不连通性。

这里补充满射性的验证。给定 $x\in X$，令

$$
\mathcal F_x=\{U\in\operatorname{RO}(X)\mid x\in U\}
$$

并把它扩张为 $\operatorname{RO}(X)$ 上的超滤子 $\mathfrak u_x$。则对任意 $U\in\mathfrak u_x$，闭包 $\overline U$ 含有 $x$；因此

$$
x\in\bigcap_{U\in\mathfrak u_x}\overline U.
$$

按照上面构造的 $p$，得到 $p(\mathfrak u_x)=x$。所以 $p$ 是满射。

## D.6 极不连通对象的投射性

**定理 D.10（Gleason）.** 在 $\mathbf{CHaus}$ 中，紧 Hausdorff 空间 $E$ 关于满射投射，当且仅当 $E$ 极不连通。

也就是说，对任意满射 $q:Y\to X$ 和连续映射 $f:E\to X$，存在连续映射 $\tilde f:E\to Y$ 使

$$
q\circ\tilde f=f.
$$

**证明说明.** 若 $E$ 极不连通，可把 $Y\to X$ 的闭图关系拉回到 $E$，再用极不连通性在闭包和开集之间选择连续截面。这是 Gleason 原定理的核心内容，本书把它作为经典拓扑输入。反向地，若 $E$ 对满射投射，把 Gleason cover $p:E_E\to E$ 提升出截面 $E\to E_E$；极不连通性作为 retract 性质从 $E_E$ 传给 $E$。

本书后续只使用这个定理的提升性质，不使用其证明内部的选择构造。因此第一卷的逻辑依赖是：

$$
\text{Gleason 定理}\Rightarrow \mathbf{ED}\text{ 对满射投射}\Rightarrow (-)(E)\text{ 正合}.
$$

## D.7 与正文的关系

正文用到本附录的方式如下：

1. 第二章使用命题 D.6，把 profinite 空间写成有限离散空间逆极限。
2. 第六章使用定理 D.9，使任意紧 Hausdorff 空间可由极不连通空间覆盖。
3. 第七章使用定理 D.10，把极不连通空间作为投射测试对象。
4. 第十二章以后使用极不连通/profinite 对象计算 solid 与 analytic 结构。

## 练习

**练习 D.1.** 证明 $\operatorname{RO}(X)$ 在上述运算下是 Boolean 代数。

**练习 D.2.** 对有限离散集合 $S$，写出 $\operatorname{Stone}(\mathcal P(S))$ 与 $S$ 的同胚。

**练习 D.3.** 设 $X$ 是 Cantor 集。说明 $X$ 是 profinite，但不是极不连通。

**练习 D.4.** 证明极不连通紧 Hausdorff 空间的 retract 仍极不连通。
