# 附录 G：形式概形、site 与导出整体截面

## 本附录目标

本附录补齐正文默认使用的几何语言：$p$-adic formal schemes、Grothendieck site、sheaves、hypercohomology 和 derived global sections。它不替代 EGA/SGA，而是给出本书内部可用的最小严格版本。

## G.1 $p$-adic formal schemes

**定义 G.1.** 令 $A$ 为 derived $p$-complete 的离散环。若在本书中写
$$
\operatorname{Spf}(A),
$$
默认指以 $p$-adic topology 为定义拓扑的 affine formal scheme。其 underlying topological space 可由 $\operatorname{Spec}(A/p)$ 表示，结构层由 compatible system
$$
\{A/p^nA\}_{n\ge1}
$$
控制。

**定义 G.2.** 一个 $p$-adic formal scheme $X$ 称为 affine，如果 $X\simeq\operatorname{Spf}(A)$。称 $X$ locally affine，如果它有由 affine opens 组成的覆盖。

**命题 G.3.** 若 $A\to B$ 是 $p$-adically continuous ring map，则诱导态射
$$
\operatorname{Spf}(B)\to\operatorname{Spf}(A).
$$

**证明.** 对每个 $n$，连续性给出兼容的环同态 $A/p^n\to B/p^n$，从而给出 schemes 的反向态射 $\operatorname{Spec}(B/p^n)\to\operatorname{Spec}(A/p^n)$。这些态射与 transition maps 相容，取形式极限得到 formal schemes 的态射。证毕。

## G.2 Formal schemes over a quotient

**定义 G.4.** 若 $(A,I)$ 是 prism，则 $X$ over $A/I$ 是指给定态射
$$
X\to\operatorname{Spf}(A/I).
$$
Affine 情形写作 $X=\operatorname{Spf}(R)$，其中 $R$ 是 $p$-complete $A/I$-algebra。

**警告 G.5.** $X$ over $A/I$ 并不意味着 $X$ over $A$。Prismatic site 正是通过 prisms $(B,J)$ 和 maps $\operatorname{Spf}(B/J)\to X$ 来探测 $X$ 的 prism thickenings。

## G.3 Grothendieck site

**定义 G.6.** 一个 Grothendieck site 是范畴 $\mathcal C$ 及覆盖族类，使得：

1. 同构族是覆盖；
2. 覆盖在 pullback 后仍为覆盖；
3. 覆盖的覆盖仍为覆盖。

在 $\mathcal C$ 上的 sheaf 是 presheaf $F:\mathcal C^{op}\to\mathbf{Set}$，满足每个覆盖族 $\{U_i\to U\}$ 的 equalizer 条件：
$$
F(U)\to\prod_iF(U_i)\rightrightarrows\prod_{i,j}F(U_i\times_UU_j)
$$
为 equalizer。

**定义 G.7.** 若 $\mathcal C$ 上有 sheaf of rings $\mathcal O$，则 $\mathcal O$-module sheaf 是 sheaf $\mathcal F$，使得每个 $\mathcal F(U)$ 是 $\mathcal O(U)$-module，并且 restriction maps 线性。

## G.4 Derived global sections

**定义 G.8.** 令 $(\mathcal C,\mathcal O)$ 为 ringed site。对 $\mathcal O$-module complex $\mathcal F^\bullet$，其 derived global sections 记作
$$
R\Gamma(\mathcal C,\mathcal F^\bullet).
$$
它可通过 injective resolution、K-injective resolution 或适当 hypercover descent 模型计算。

**命题 G.9.** 若 $f:\mathcal F^\bullet\to\mathcal G^\bullet$ 是 complexes of sheaves 的 quasi-isomorphism，则
$$
R\Gamma(\mathcal C,\mathcal F^\bullet)\to R\Gamma(\mathcal C,\mathcal G^\bullet)
$$
是 quasi-isomorphism。

**证明.** 取 $\mathcal F^\bullet$ 和 $\mathcal G^\bullet$ 的 K-injective resolutions。Quasi-isomorphism 在 derived category 中成为同构，$R\Gamma$ 是 global sections functor 的右导出函子，因此保持 derived category 中的同构。证毕。

## G.5 Cech 计算与限制

**定义 G.10.** 对覆盖 $U_\bullet\to X$，Cech complex 为
$$
\check C^n(U_\bullet,\mathcal F)=\prod_{i_0,\ldots,i_n}\mathcal F(U_{i_0}\times_X\cdots\times_XU_{i_n}),
$$
d=\sum_{r=0}^{n+1}(-1)^r d_r$。

**警告 G.11.** Cech complex 不总是计算 sheaf cohomology。只有在覆盖 acyclic、site 有足够 descent 性，或使用 hypercover 时，才能无条件计算 derived global sections。

**说明 G.12.** Prismatic cohomology 的 $R\Gamma_\Delta(X/A)$ 是 derived global sections，不是普通 global sections。Affine smooth 情形常有显式 complex 计算，但那是 comparison 或 descent theorem 的内容。

## 本附录小结

正文中的 $R\Gamma_\Delta$、$R\Gamma_{\mathrm{dR}}$、crystal cohomology 和 syntomic complexes 都属于 derived global sections 的语境。任何把它们降为普通截面的论证，都必须额外证明 acyclicity 或 descent。

## 练习

**练习 G.1.** 对 $A=\mathbf Z_p[[T]]$，写出 $\operatorname{Spf}(A)$ 的 truncations $\operatorname{Spec}(A/p^n)$。

**练习 G.2.** 证明 sheaf 条件中的 equalizer 对单覆盖同构族自动成立。

**练习 G.3.** 给出一个理由，说明为什么 prismatic cohomology 必须用 derived global sections 定义。

