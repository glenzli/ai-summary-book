# 第二十二章：Quantum groups、crystals 与 canonical bases 的几何模型

Canonical basis 同时出现在 quiver representation varieties 的 IC 层、Nakajima Lagrangian 的不可约分支和 KLR algebra 的不可分 projectives 中，但“这些对象给出同一组基”不是定义，而是一系列比较定理。Crystal 提供较粗却更可计算的共同影子：它忘掉作用系数与 $q$-高阶信息，只保留 weight 以及沿每个 simple root 的有限 strings。单顶点 quiver 已能把这条联系完整画出。对 framing $w=n$，零 fibers 是 $\operatorname{Gr}(v,n)$，$0\le v\le n$；相邻 Grassmannians 的 incidence correspondence 形成一条长度 $n+1$ 的链，权正是 $n-2v$。这将从第十七章的 Hamiltonian reduction 推出 $\mathfrak{sl}_2$ crystal $B(n)$，再说明更一般 canonical-basis 识别必须保留何种外部输入。

## 22.1 Quantum group 和 integral form

**定义 22.1.** 给定 symmetrizable Cartan datum，quantum group $U_q(\mathfrak g)$ 是由生成元
$$
E_i,F_i,K_i^{\pm1}
$$
和 quantum Serre relations 定义的 $\mathbb Q(q)$-代数。其负半部记为 $U_q^-(\mathfrak g)$。

**定义 22.2.** Lusztig integral form 是 $U_q(\mathfrak g)$ 中由 divided powers
$$
E_i^{(n)}=\frac{E_i^n}{[n]_{q_i}!},\qquad
F_i^{(n)}=\frac{F_i^n}{[n]_{q_i}!}
$$
以及 Cartan integral part（$K_i^{\pm1}$ 和 quantum binomial coefficients）生成的 $\mathbb Z[q,q^{-1}]$-子代数。若只讨论正半部或负半部，则相应 integral form 只需对应的 divided powers。

**外部输入定理 22.3.** Lusztig canonical basis 和 Kashiwara global crystal basis 给出 $U_q^-(\mathfrak g)$ 及 integrable highest weight modules 的特殊基，并满足 positivity、bar-invariance 和 crystal limit 性质。

## 22.2 Crystals

**定义 22.4.** crystal 是集合 $B$ 连同 maps
$$
e_i,f_i:B\to B\sqcup\{0\},\qquad
\operatorname{wt}:B\to P,
$$
以及 $\varepsilon_i,\varphi_i$ 等函数，满足 Kashiwara axioms。它记录 $q\to0$ 极限下的表示组合结构。

**命题 22.5.** 若一个 integrable representation 有 crystal basis，则每个 $f_i$ 沿 $i$-string 给出有限链。

**证明.** integrability 意味着每个 $\mathfrak{sl}_2$-方向局部有限，故对任意向量反复施加 lowering operator 只能产生有限长 string。crystal basis 是该结构的 $q\to0$ 记录，因此 $f_i$ 迭代在有限步后到达 $0$。$\square$

## 22.3 几何模型

**例 22.6.** 对 $\mathfrak{sl}_2$，最高权为 $n$ 的 irreducible representation 的 crystal 为
$$
B(n)=\{b_0,b_1,\ldots,b_n\},
$$
其中
$$
\operatorname{wt}(b_r)=n-2r,\qquad
f(b_r)=b_{r+1}\ (r<n),\qquad f(b_n)=0,
$$
$$
e(b_r)=b_{r-1}\ (r>0),\qquad e(b_0)=0.
$$
这是长度为 $n+1$ 的一条 string。

**命题 22.7.** $B(n)$ 满足 $\mathfrak{sl}_2$ crystal 的基本 string 条件。

**证明.** 对每个 $b_r$，可向左施加 $e$ 共 $r$ 次，向右施加 $f$ 共 $n-r$ 次。因此
$$
\varepsilon(b_r)=r,\qquad \varphi(b_r)=n-r.
$$
权满足
$$
\varphi(b_r)-\varepsilon(b_r)=n-2r=\langle h,\operatorname{wt}(b_r)\rangle.
$$
同时 $e$ 与 $f$ 在非端点处互逆，端点送到 $0$。这正是 $\mathfrak{sl}_2$ crystal axioms 的内容。$\square$

**例 22.7.1（$A_1$ quiver 的 Lagrangian fibers）.** 取第十七章的单顶点无边 quiver，令 $W=\mathbb C^n$、$\dim V=v$，并采用 $i:W\twoheadrightarrow V$ 的稳定 chamber。第十七章给出
$$
\mathfrak M(v,n)\simeq T^*\operatorname{Gr}(v,W).
$$
到 affine quotient 的映射由 $ji\in\operatorname{End}(W)$ 给出。其零 fiber 满足 $ji=0$；因为 $i$ 满射，这迫使 $j=0$。因此
$$
\mathfrak L(v,n)=\pi^{-1}(0)\simeq\operatorname{Gr}(v,W)
$$
是零截面。它在 $0\le v\le n$ 时非空且不可约，在其余 $v$ 上为空，所以
$$
H^{BM}_{2v(n-v)}(\mathfrak L(v,n),E)
$$
由 fundamental class $c_v=[\operatorname{Gr}(v,n)]$ 生成。

**命题 22.7.2（从 incidence 得到 $B(n)$）.** 把 $c_v$ 的 weight 定为 $n-2v$。对 $0\le v<n$，令
$$
Z_v=\{(K_v,K_{v+1})\mid
K_{v+1}\subset K_v\subset W,
\ \dim K_v=n-v,
\ \dim K_{v+1}=n-v-1\},
$$
其中 $K_v$ 是 $v$-维 quotient $W\twoheadrightarrow W/K_v$ 的 kernel。以 $Z_v$ 连接 $c_v$ 与 $c_{v+1}$，所得有向图同构于 crystal $B(n)$：
$$
c_0\longrightarrow c_1\longrightarrow\cdots\longrightarrow c_n.
$$

**证明.** 每个 $\operatorname{Gr}(v,n)$ 只有一个不可约分支，所以每个允许的 $v$ 只给一个顶点。若 $v<n$，$Z_v$ 是参数化嵌套 kernels 的二步 partial flag variety，因而非空且不可约，只连接相邻两个顶点；当 $v=n$ 时不存在更小 kernel，箭头终止。沿箭头 $v\mapsto v+1$ 后，weight 由 $n-2v$ 变为 $n-2v-2$，正是 $\mathfrak{sl}_2$ 的 lowering step。反向 incidence 给 raising step。因此该图与例 22.6 的 $B(n)$ 逐项相同。Nakajima 定理把这些 correspondences 提升为表示作用；本命题只识别其不可约分支图与 crystal combinatorics。$\square$

这个推导还解释了 crystal 为何比表示更粗：incidence 只告诉我们某条边是否存在，实际 Chevalley operator 在 fundamental classes 上的系数、量子 grading 与 divided powers 需要同调 correspondence 或 KLR projectives 才能恢复。

**外部输入定理 22.8.** Lusztig 用 quiver varieties 和 perverse sheaves 构造 canonical bases；Nakajima quiver varieties 给出 highest weight representations 的几何模型；KLR algebras 的 indecomposable projectives 在合适情形中对应 canonical basis。

**边界说明 22.9.** “canonical basis 的几何实现”不是单一陈述。必须说明是：

1. Lusztig perverse sheaf model；
2. Nakajima quiver variety homology model；
3. KLR projective module model；
4. cluster/DT/CoHA model；
5. Satake/MV cycle model。

**表 22.10.** 几何模型与基的对应。

| 模型 | 几何/代数对象 | 基向量候选 | 外部输入 |
| --- | --- | --- | --- |
| Lusztig perverse sheaves | quiver representation varieties | simple perverse sheaves | Lusztig canonical basis |
| Nakajima quiver varieties | Lagrangian varieties $\mathfrak L(v,w)$ | irreducible components / homology classes | Nakajima theorem |
| KLR algebras | projective module categories | indecomposable projectives | KLR/Rouquier theorem |
| MV cycles | affine Grassmannian | MV cycle classes | geometric Satake |
| CoHA | critical cohomology of stacks | BPS classes | KS/DT theory |

该表用于定位，不表示这些基在没有额外定理时自动相同。

$A_1$ quiver 的零 fibers $\operatorname{Gr}(v,n)$ 各只有一个不可约分支，相邻 kernel 的 incidence 因而直接给出 $B(n)$ 的整条 string；weight $n-2v$ 也由 dimension vector 读出。一般 quiver、KLR 与 MV 模型中，crystal 仍记录 simple-root 方向的组合，但 canonical basis 还依赖 IC、projective 或 cycle 类及其 duality convention。第二十三章将不再罗列新模型，而按这些比较尚缺少的数学障碍组织全书的研究边界。

## 练习

**练习 22.1.** 对 $\mathfrak{sl}_2$ 写出 $U_q(\mathfrak{sl}_2)$ 的生成元和关系。

**练习 22.2.** 画出最高权为 $n$ 的 $\mathfrak{sl}_2$-module 的 crystal graph。

**练习 22.3.** 比较 KLR indecomposable projectives 和 simple modules 分别对应哪一种 dual canonical basis convention。

**练习 22.4.** 对 $B(2)$ 画出 crystal graph，并验证 $\varepsilon,\varphi$ 与 weight 的关系。

**练习 22.5.** 对 $n=3$ 写出四个 $\operatorname{Gr}(v,3)$ 及 kernels 的维数，画出 $Z_0,Z_1,Z_2$ 给出的 incidence graph，并与 $B(3)$ 比较。
