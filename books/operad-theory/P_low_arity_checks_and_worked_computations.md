# 附录 P：低阶计算、逐项验算与小模型

本附录把前文若干抽象定义压缩到低阶可手算情形。它的目标是检验符号和公理，而不是提供新的理论。每个计算都只使用已定义的结构：有限集映射及纤维、树代入、operad 乘法、bar construction、inner horn 或 derived tensor product。

## P.1 代入乘积的 arity $0,1,2$

设 $X,Y:\mathbf B_{\mathcal U}\to\mathbf{Set}_{\mathcal U}$ 是集合值对称序列。按照定义 B.5，
$$
(X\circ Y)(S)
=
\operatorname*{colim}_{(f:S\to T)\in\operatorname{Fib}(S)}
X(T)\times\prod_{t\in T}Y(f^{-1}(t)).
$$

### Arity $0$

对每个 $k\ge0$，存在唯一函数 $\varnothing\to[k]$，其 $k$ 个纤维全为空。因此
$$
(X\circ Y)(\varnothing)
\cong
\coprod_{k\ge0}X(k)\times_{\Sigma_k}Y(0)^k.
$$
这里 $Y(0)^0$ 是 singleton。故 arity $0$ 不只来自 $X(0)$：一个 $k$-ary 外层运算可在全部 $k$ 个输入槽中代入 nullary $Y$-运算。

**反例 P.0（非空分块公式的左单位失败）.** 取 $X=I$ 且 $Y(0)\ne\varnothing$。上式只有 $k=1$ 项有贡献，给出
$$
(I\circ Y)(0)\cong Y(0),
$$
与左单位一致。若误用非空分块公式，则会得到 $I(0)=\varnothing$。因此省略空纤维会直接破坏幺半单位。

### Arity $1$

对函数 $f:[1]\to[k]$，必须有 $k\ge1$；恰有一个纤维是单点，其余 $k-1$ 个纤维为空。固定单点纤维为第一个槽后，其稳定子为置换其余槽的 $\Sigma_{k-1}$。因此
$$
(X\circ Y)(1)
\cong
\coprod_{k\ge1}
\bigl(X(k)\times Y(1)\times Y(0)^{k-1}\bigr)_{\Sigma_{k-1}}.
$$
若 $Y(0)=\varnothing$，只有 $k=1$ 项保留，才得到 $X(1)\times Y(1)$。在 operad $\mathcal O$ 中，乘法
$$
\mu:\mathcal O\circ\mathcal O\to\mathcal O
$$
在 arity $1$ 上既包含 unary operation 的复合，也包含在高 arity 运算的其余槽中代入 nullary operations。

### Arity $2$

设 $S=[2]$。函数 $f:[2]\to[k]$ 在目标双射下有两类：

1. 两个元素落入同一目标点；此时有一个二元纤维和 $k-1$ 个空纤维；
2. 两个元素落入不同目标点；此时有两个带源标号的单点纤维和 $k-2$ 个空纤维。

故
$$
(X\circ Y)(2)
\cong
\coprod_{k\ge1}
\bigl(X(k)\times Y(2)\times Y(0)^{k-1}\bigr)_{\Sigma_{k-1}}
\;\amalg\;
\coprod_{k\ge2}
\bigl(X(k)\times Y(1)^2\times Y(0)^{k-2}\bigr)_{\Sigma_{k-2}},
$$
其中稳定子固定非空纤维对应的目标点，并置换空槽。若 $Y(0)=\varnothing$，该式才缩成
$$
X(1)\times Y(2)\;\amalg\;X(2)\times Y(1)^2.
$$
输出的 $\Sigma_2$-作用在第一项作用于 $Y(2)$，在第二项交换两个带源标号的单点纤维并同步使用 $X(2)$ 的作用。该例是左右作用和空槽约定最早同时出现的位置。

### 自由 operad 权重的二槽检查

设 $E(2)$ 含二元生成元 $x,y_1,y_2$，三者在自由 operad 中均有权重 $1$。完整代入
$$
\gamma(x;y_1,y_2)
$$
由一个外层顶点和两个内层顶点组成，故权重为 $1+1+1=3$。相反，偏复合
$$
x\circ_1y_1=\gamma(x;y_1,\mathbf 1)
$$
的第二个内层因子是权重 $0$ 的单位树，所以权重为 $1+1+0=2$。一般地，外层权重为 $r$、第 $t$ 个内层权重为 $s_t$ 时，grafting 后的顶点不交并给出权重
$$
r+\sum_t s_t;
$$
若外层 arity 为 $n$ 且所有 $s_t=s$，该值就是 $r+ns$。这也直接解答练习 8.1。

## P.2 Endomorphism operad 的结合律检查

设 $A$ 是集合。Endomorphism operad 定义为
$$
\operatorname{End}_A(S)=\mathbf{Set}(A^S,A).
$$
给定有限集映射 $q:S\to T$、外层函数
$$
F:A^T\to A
$$
以及内层函数
$$
G_t:A^{q^{-1}(t)}\to A\qquad(t\in T),
$$
复合为
$$
\mu_q(F;(G_t)_{t\in T})
:A^S\to A,\qquad
(a_s)_{s\in S}\mapsto
F\big((G_t((a_s)_{s\in q^{-1}(t)}))_{t\in T}\big).
$$
空纤维给出 $A^\varnothing\to A$，即常量函数。

**命题 P.1.** 第 P.2 节定义的复合满足 operad 结合律。

**证明.** 取可复合映射 $S\xrightarrow{g}U\xrightarrow{p}T$ 以及函数
$$
h_u:A^{g^{-1}(u)}\to A,\qquad
G_t:A^{p^{-1}(t)}\to A,\qquad
F:A^T\to A.
$$
先复合内两层再复合外层，得到
$$
(a_s)\mapsto
F\left(
\left(
G_t\left(
\left(h_u((a_s)_{s\in g^{-1}(u)})\right)_{u\in p^{-1}(t)}
\right)
\right)_{t\in T}
\right).
$$
先复合外两层再代入 $h_u$，得到同一个函数表达式，包括 $g$ 或 $p$ 有空纤维的情形。两个括号方案在 $\mathbf{Set}$ 中给出相等函数，因此 endomorphism operad 的结合律成立。单位函数来自恒等映射 $\operatorname{id}_A:A\to A$；插入双射层或单点目标层不改变该表达式。$\square$

## P.3 $\operatorname{Ass}$ 的低阶运算

集合值 associative operad 可取
$$
\operatorname{Ass}(S)=\operatorname{Lin}(S),
$$
即有限集 $S$ 上的全序集合。若 $S=\{a,b\}$，则
$$
\operatorname{Ass}(S)=\{a<b,\; b<a\}.
$$

给定函数 $q:S\to T$，$T$ 上的外层全序 $<$，以及每个纤维 $q^{-1}(t)$ 上的内层全序 $<_t$，复合全序定义为 lexicographic fiber order：
$$
s<t
$$
当且仅当

1. $q(s)=q(t)$ 且 $s<_{q(s)}t$；或
2. $q(s)\ne q(t)$ 且 $q(s)<q(t)$。

空纤维有唯一全序，并且不向最终的 $S$-次序贡献元素。

**命题 P.2.** 该复合定义 $\operatorname{Ass}$ 的 operad 结构。

**证明.** 需要检查两件事。

第一，lexicographic fiber order 是全序。任取 $s,t$。若二者在同一纤维，使用该纤维上的全序比较；若在不同纤维，使用目标集上的全序比较。反对称性、传递性分别由内层全序和外层全序的反对称性、传递性给出。

第二，对可复合映射 $S\to U\to T$，结合律来自多层 lexicographic order 的拉平。比较两个元素 $s,t$ 时，两个复合顺序都按如下优先级决定：

1. 先比较二者在 $T$ 中的像；
2. $T$-像相同则比较其 $U$-像；
3. $U$-像相同则使用最内层纤维全序。

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
当 $S=\{1,2\}$ 时，得到二元乘法 $a\cdot b=\mu_{\{1,2\}}(a,b)$。对称群等变性给出 $a\cdot b=b\cdot a$。比较无空纤维的三元代入所对应的两个分组
$$
\{\{1,2\},\{3\}\}
$$
与
$$
\{\{1\},\{2,3\}\}
$$
可得 $(a\cdot b)\cdot c=a\cdot(b\cdot c)$。为检查单位，取两个函数 $[1]\to[2]$，分别把唯一元素送入第一槽和第二槽；另一槽是空纤维。把 nullary 运算 $e$ 代入该空槽，并把一元单位代入非空槽，operad 代数相容性分别给出 $a\cdot e=a$ 与 $e\cdot a=a$。非空分块无法表达这一步。反向地，含单位交换幺半群用有限乘积定义 $\mu_S$；交换性保证对枚举选择无依赖，空纤维的空乘积取 $e$，所以与全部有限集映射的代入相容。$\square$

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
2. 自由 operad 的完整代入把权重相加为 $r+\sum_t s_t$；
3. endomorphism operad 的结合律是函数复合的逐项相等；
4. $\operatorname{Ass}$ 的 operad 复合是多层字典序拉平；
5. $\operatorname{Com}$ 的 arity $0$ 元素给出单位；
6. Lie 代数结构由生成元关系的泛性质给出；
7. $A_\infty$ 低阶关系应先在 suspended convention 中写出；
8. Hochschild bracket 的低阶计算把结合律写成 $m\circ m=0$；
9. dendroidal inner horn 的最小例子区分唯一填充和存在填充；
10. $\int_{S^1}A$ 的链级模型经过 cyclic bar construction 表示。

这些计算是后续严格化中检查符号、单位和跨模型比较的基准。
