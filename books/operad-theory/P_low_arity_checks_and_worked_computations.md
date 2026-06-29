# 附录 P：低阶计算、逐项验算与小模型

本附录把前文若干抽象定义压缩到低阶可手算情形。它的目标是检验符号和公理，而不是提供新的理论。每个计算都只使用已定义的结构：有限集分块、树代入、operad 乘法、bar construction、inner horn 或 derived tensor product。

## P.1 代入乘积的 arity $0,1,2$

设 $X,Y:\mathbf B_{\mathcal U}\to\mathbf{Set}_{\mathcal U}$ 是集合值对称序列。按照定义 B.5，
$$
(X\circ Y)(S)
=
\coprod_{\pi\in\operatorname{Part}(S)}
X(\operatorname{Bl}(\pi))\times
\prod_{B\in\operatorname{Bl}(\pi)}Y(B).
$$

### Arity $0$

当 $S=\varnothing$ 时，$\varnothing$ 只有一个分块，其块集合为空集。因此
$$
(X\circ Y)(\varnothing)
\cong
X(\varnothing)\times\prod_{B\in\varnothing}Y(B)
\cong
X(0).
$$
这里空乘积为 singleton。结论说明：在一次代入中，arity $0$ 输出只来自外层 $X$ 的 arity $0$ 运算，而不是来自某个内层输入。

### Arity $1$

设 $S=\{s\}$。分块只有
$$
\pi=\{\{s\}\}.
$$
因此
$$
(X\circ Y)(1)\cong X(1)\times Y(1).
$$
在 operad $\mathcal O$ 中，乘法
$$
\mu:\mathcal O\circ\mathcal O\to\mathcal O
$$
在 arity $1$ 上给出 unary operation 的复合。

### Arity $2$

设 $S=\{s_1,s_2\}$。有两类分块：

1. 单块分块 $\{\{s_1,s_2\}\}$；
2. 双块分块 $\{\{s_1\},\{s_2\}\}$。

故
$$
(X\circ Y)(S)
\cong
X(1)\times Y(S)
\;\amalg\;
X(\{B_1,B_2\})\times Y(B_1)\times Y(B_2),
$$
其中第二项还携带块集合双射导致的 $\Sigma_2$-自然性。若改用骨架 arity 公式，第二项要写成对 $\Sigma_2$ 作用的 coinvariants。该例是左右作用约定最早出现风险的位置。

## P.2 Endomorphism operad 的结合律检查

设 $A$ 是集合。Endomorphism operad 定义为
$$
\operatorname{End}_A(S)=\mathbf{Set}(A^S,A).
$$
给定分块 $\pi$ of $S$、外层函数
$$
f:A^{\operatorname{Bl}(\pi)}\to A
$$
以及内层函数
$$
g_B:A^B\to A\qquad(B\in\operatorname{Bl}(\pi)),
$$
复合为
$$
\mu(f;(g_B)_{B})
:A^S\to A,\qquad
(a_s)_{s\in S}\mapsto
f\big((g_B((a_s)_{s\in B}))_{B\in\operatorname{Bl}(\pi)}\big).
$$

**命题 P.1.** 第 P.2 节定义的复合满足 operad 结合律。

**证明.** 设 $\rho$ 是 $S$ 的 refinement，$\rho$ 先按小块 $C$ 分组，再按大块 $B$ 分组，最后按 $\pi$ 的块集合分组。取
$$
h_C:A^C\to A,\qquad
g_B:A^{\operatorname{Bl}(\rho|_B)}\to A,\qquad
f:A^{\operatorname{Bl}(\pi)}\to A.
$$
先复合内两层再复合外层，得到
$$
(a_s)\mapsto
f\left(
\left(
g_B\left(
\left(h_C((a_s)_{s\in C})\right)_{C\subset B}
\right)
\right)_{B}
\right).
$$
先复合外两层再代入 $h_C$，得到同一个函数表达式。两个括号方案在 $\mathbf{Set}$ 中给出相等函数，因此 endomorphism operad 的结合律成立。单位函数来自恒等映射 $\operatorname{id}_A:A\to A$；代入恒等映射不改变该函数表达式。$\square$

## P.3 $\operatorname{Ass}$ 的低阶运算

集合值 associative operad 可取
$$
\operatorname{Ass}(S)=\operatorname{Lin}(S),
$$
即有限集 $S$ 上的全序集合。若 $S=\{a,b\}$，则
$$
\operatorname{Ass}(S)=\{a<b,\; b<a\}.
$$

给定分块 $\pi$，外层全序 $<$ on $\operatorname{Bl}(\pi)$，以及每个块 $B$ 上的内层全序 $<_B$，复合全序定义为 lexicographic block order：
$$
s<t
$$
当且仅当

1. $s,t$ 在同一块 $B$ 且 $s<_B t$；或
2. $s\in B_s,t\in B_t,B_s\ne B_t$ 且 $B_s<B_t$。

**命题 P.2.** 该复合定义 $\operatorname{Ass}$ 的 operad 结构。

**证明.** 需要检查两件事。

第一，lexicographic block order 是全序。任取 $s,t$。若二者在同一块，使用该块上的全序比较；若在不同块，使用块集合上的全序比较。反对称性、传递性分别由内层全序和外层全序的反对称性、传递性给出。

第二，结合律来自多层 lexicographic order 的拉平。若 $S$ 先分为小块 $C$，小块再聚成中块 $B$，中块再聚成大块，则比较两个元素 $s,t$ 时，两个复合顺序都按如下优先级决定：

1. 先比较包含二者的大块；
2. 大块相同则比较中块；
3. 中块相同则比较小块；
4. 小块相同则使用最内层全序。

因此两个复合顺序产生相同全序。单位是一元素集上的唯一全序。$\square$

**推论 P.3.** 对 $R$-模 $V$，
$$
\bigoplus_{n\ge0}R[\operatorname{Ass}(n)]\otimes_{\Sigma_n}V^{\otimes n}
\cong
\bigoplus_{n\ge0}V^{\otimes n}=T(V).
$$

**证明.** $\operatorname{Ass}(n)$ 是 $\Sigma_n$ 的自由传递右作用集合。故
$$
R[\operatorname{Ass}(n)]\otimes_{\Sigma_n}V^{\otimes n}
\cong
V^{\otimes n}.
$$
对 $n$ 求直和即得张量代数。$\square$

## P.4 $\operatorname{Com}$ 与 arity $0$

集合值 commutative operad 定义为
$$
\operatorname{Com}(S)=\{*\}
$$
对每个有限集 $S$。复合只能把唯一元素送到唯一元素。

**命题 P.4.** $\operatorname{Com}$-代数是含单位交换幺半群。

**证明.** 设 $A$ 是 $\operatorname{Com}$-代数。对每个有限集 $S$ 有运算
$$
\mu_S:A^S\to A.
$$
当 $S=\varnothing$ 时，得到元素
$$
e=\mu_\varnothing(*)\in A.
$$
当 $S=\{1,2\}$ 时，得到二元乘法 $a\cdot b=\mu_{\{1,2\}}(a,b)$。对称群等变性给出 $a\cdot b=b\cdot a$。分块结合律应用于三元素集的分块
$$
\{\{1,2\},\{3\}\}
$$
与
$$
\{\{1\},\{2,3\}\}
$$
给出 $(a\cdot b)\cdot c=a\cdot(b\cdot c)$。arity $0$ 与 arity $1$ 的分块组合给出 $e\cdot a=a=a\cdot e$。反向地，含单位交换幺半群用有限乘积定义 $\mu_S$，交换性保证对枚举选择无依赖。$\square$

## P.5 Lie operad 的 arity $2,3$ 检查

在线性语境中，Lie operad 可由一个二元生成元
$$
[-,-]\in\operatorname{Lie}(2)
$$
和关系给出：

1. 反对称性 $[x,y]+[y,x]=0$；
2. Jacobi 关系
   $$
   [x,[y,z]]+[y,[z,x]]+[z,[x,y]]=0.
   $$

**说明 P.5.** 若底环中 $2$ 不可逆，反对称性与 alternating 条件 $[x,x]=0$ 需要分开。第六章和附录 F 默认在安全语境中使用 Lie operad；一般底环必须重新声明约定。

**命题 P.6.** Lie algebra $L$ 给出 $\operatorname{Lie}$-代数。

**证明.** $\operatorname{Lie}$ 是由二元括号生成并除以反对称与 Jacobi 关系的 operad。给定 Lie algebra $L$，把生成元送到 $L$ 的括号
$$
[-,-]_L:L\otimes L\to L.
$$
反对称性和 Jacobi 关系保证该赋值杀掉 defining ideal，因此由商的泛性质诱导 operad morphism
$$
\operatorname{Lie}\to\operatorname{End}_L.
$$
这正是 $\operatorname{Lie}$-代数结构。反向方向由 arity $2$ 生成元在 $\operatorname{End}_L$ 中的像恢复括号，关系恢复 Lie 公理。$\square$

## P.6 Suspended $A_\infty$ 低阶关系

采用附录 L 的 suspended 定义。$A_\infty$-结构是 reduced tensor coalgebra
$$
T^c(sA)=\bigoplus_{n\ge1}(sA)^{\otimes n}
$$
上的 degree $-1$ coderivation
$$
b:T^c(sA)\to T^c(sA)
$$
满足
$$
b^2=0.
$$
其 Taylor 分量为
$$
b_n:(sA)^{\otimes n}\to sA.
$$

在 suspended convention 下，低阶关系写为
$$
\sum_{r+s+t=n}
b_{r+1+t}
\big(
\operatorname{id}^{\otimes r}\otimes b_s\otimes
\operatorname{id}^{\otimes t}
\big)=0
$$
作用在 $(sA)^{\otimes n}$ 上；把它转成 $m_n:A^{\otimes n}\to A$ 时才出现附录 E 的 suspension signs。

### $n=1$

$$
b_1b_1=0.
$$
因此 $b_1$ 是 $sA$ 上的 differential。转回 $A$，得到 $m_1^2=0$。

### $n=2$

$$
b_1b_2
+b_2(b_1\otimes\operatorname{id})
+b_2(\operatorname{id}\otimes b_1)=0.
$$
转回 unsuspended convention 后，这表示 $m_1$ 是 $m_2$ 的 derivation，符号由同调分次决定。

### $n=3$

$$
\begin{aligned}
0={}&
b_1b_3
+b_2(b_2\otimes\operatorname{id})
+b_2(\operatorname{id}\otimes b_2)\\
&+b_3(b_1\otimes\operatorname{id}\otimes\operatorname{id})
+b_3(\operatorname{id}\otimes b_1\otimes\operatorname{id})
+b_3(\operatorname{id}\otimes\operatorname{id}\otimes b_1).
\end{aligned}
$$
若 $b_1=0$ 且 $b_3=0$，该式退化为 suspended 乘法的结合律。若 $b_3$ 非零，它记录结合律只在 homotopy 意义下成立。

## P.7 Hochschild cochains 的低阶 brace

设 $A$ 是未分次结合代数。对 $f\in C^p(A,A)$、$g\in C^q(A,A)$，单插入为
$$
(f\circ_i g)(a_1,\ldots,a_{p+q-1})
=
f(a_1,\ldots,a_{i-1},g(a_i,\ldots,a_{i+q-1}),a_{i+q},\ldots).
$$

总插入
$$
f\circ g=\sum_{i=1}^{p}(-1)^{(q-1)(i-1)}f\circ_i g.
$$
Gerstenhaber bracket 为
$$
[f,g]=f\circ g-(-1)^{(p-1)(q-1)}g\circ f.
$$

**计算 P.7.** 若 $m\in C^2(A,A)$ 是乘法，则
$$
[m,m]=2(m\circ m)
$$
且
$$
m\circ m
=
m\circ_1m-m\circ_2m.
$$
作用在 $(a,b,c)$ 上得到
$$
(ab)c-a(bc).
$$
因此结合律等价于 $m\circ m=0$。在特征不为 $2$ 的情形，也等价于 $[m,m]=0$；在一般底环上应直接使用 $m\circ m=0$ 或 Hochschild differential 的 square-zero 形式。

## P.8 Dendroidal inner horn 的最小例子

令 $T$ 是有两个顶点和一条 inner edge 的树：
$$
\begin{array}{c}
\text{two corollas glued along one edge}.
\end{array}
$$
它的 inner horn $\Lambda^e[T]\subset\Omega[T]$ 删除收缩 inner edge $e$ 的 face，而保留其他 faces。

**解释 P.8.** 对 strict operad $\mathcal P$，给出 map
$$
\Lambda^e[T]\to N_d(\mathcal P)
$$
等价于给出两个可复合 operations 及其外边界数据。填充
$$
\Omega[T]\to N_d(\mathcal P)
$$
等价于给出它们的复合 operation。Strict operad 中复合由 operad 乘法唯一决定，因此填充唯一。

对一般 dendroidal inner Kan object $X$，同一 horn 只要求至少一个填充存在。不同填充对应不同 homotopy coherent composites。该差异正是 strict operad 与 infinity-operad 的分界。

## P.9 圆周 Hochschild 模型的 simplicial levels

设 $A$ 是 dg associative algebra。Cyclic bar construction 的第 $q$ 层可写作
$$
B_q^{cy}(A)=A^{\otimes(q+1)}.
$$
Face maps 由相邻乘法给出：
$$
d_i(a_0\otimes\cdots\otimes a_q)
=
a_0\otimes\cdots\otimes a_ia_{i+1}\otimes\cdots\otimes a_q
\quad(0\le i<q),
$$
最后一个 face 使用循环乘法：
$$
d_q(a_0\otimes\cdots\otimes a_q)
=
(-1)^\epsilon a_qa_0\otimes a_1\otimes\cdots\otimes a_{q-1},
$$
其中 $\epsilon$ 由 dg Koszul rule 决定；未分次情形 $\epsilon=0$。

**说明 P.9.** 几何实现或 normalized chains of $B_\bullet^{cy}(A)$ 给出 Hochschild chains 的模型。附录 N 中的
$$
\int_{S^1}A\simeq HH_\*(A)
$$
可通过一维 factorization homology 的 excision 与该 cyclic bar 模型比较得到。该比较仍是外部输入；本节只记录低阶链模型。

## P.10 二重相对张量积的括号

在稳定 monoidal infinity-category 中，derived relative tensor product 可由 geometric realization
$$
M\otimes_B^{\mathbf L}N
\simeq
\left|[q]\mapsto M\otimes B^{\otimes q}\otimes N\right|
$$
表示。

若出现三重表达式
$$
(M\otimes_B^{\mathbf L}N)\otimes_C^{\mathbf L}P
$$
和
$$
M\otimes_B^{\mathbf L}(N\otimes_C^{\mathbf L}P),
$$
二者的比较不是普通括号等式，而是由 bisimplicial bar construction 的 Fubini theorem for colimits 给出的 canonical equivalence。factorization homology 的迭代 excision 使用的是这种等价。

## P.11 小结

本附录验证了以下低阶事实：

1. arity $0$ 项来自外层 nullary operation；
2. endomorphism operad 的结合律是函数复合的逐项相等；
3. $\operatorname{Ass}$ 的 operad 复合是多层字典序拉平；
4. $\operatorname{Com}$ 的 arity $0$ 元素给出单位；
5. Lie 代数结构由生成元关系的泛性质给出；
6. $A_\infty$ 低阶关系应先在 suspended convention 中写出；
7. Hochschild bracket 的低阶计算把结合律写成 $m\circ m=0$；
8. dendroidal inner horn 的最小例子区分唯一填充和存在填充；
9. $\int_{S^1}A$ 的链级模型经过 cyclic bar construction 表示。

这些计算是后续严格化中检查符号、单位和跨模型比较的基准。
