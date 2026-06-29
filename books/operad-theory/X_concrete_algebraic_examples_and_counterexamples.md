# 附录 X：具体代数例子、反例与边界计算

本附录收集小型代数例子，用于检查 arity $0$、coinvariants、底环特征、Lie 约定、rectification 和 Hochschild/factorization 直觉。所有例子都只使用前文已经定义的结构；涉及深层同伦理论的地方只给边界说明。

## X.1 Arity $0$ 改变自由交换代数

令 $R$ 是交换环，$V$ 是 $R$-模。

若使用含 arity $0$ 的 $\operatorname{Com}$，自由交换代数为
$$
\operatorname{Sym}_R(V)=\bigoplus_{n\ge0}(V^{\otimes n})_{\Sigma_n}.
$$
其中 $n=0$ 项为
$$
(V^{\otimes0})_{\Sigma_0}=R.
$$

若使用 reduced nonunital $\operatorname{Com}_{nu}$，自由非含单位交换代数为
$$
\operatorname{Sym}^{+}_R(V)=\bigoplus_{n\ge1}(V^{\otimes n})_{\Sigma_n}.
$$

**命题 X.1.** 两个自由对象的泛性质不同。

**证明.** $\operatorname{Sym}_R(V)$ 接收到 $R$ 中单位 $1$，任意代数映射必须把它送到目标交换代数的单位。$\operatorname{Sym}^{+}_R(V)$ 没有常数项，只分类到非含单位交换代数的 $R$-线性映射。若目标是含单位代数，非含单位代数映射不要求保存单位，因为源中没有单位。因此两个泛性质不同。$\square$

## X.2 Tensor algebra 与 symmetric algebra 的差异

对 $R$-模 $V$，
$$
T(V)=\bigoplus_{n\ge0}V^{\otimes n}
$$
是自由 associative algebra，
$$
\operatorname{Sym}_R(V)=\bigoplus_{n\ge0}(V^{\otimes n})_{\Sigma_n}
$$
是自由 commutative algebra。

**例 X.2.** 令 $V=R\cdot x\oplus R\cdot y$。在 $T(V)$ 中，$xy$ 与 $yx$ 是不同张量；在 $\operatorname{Sym}_R(V)$ 中，它们在 coinvariants 中相同。

**说明 X.3.** 这就是 operad $\operatorname{Ass}$ 与 $\operatorname{Com}$ 的差异在自由代数上的最小表现。若底环中特征为 $2$，交换性 $xy=yx$ 与 graded-commutativity 的符号规则还需另行区分。

## X.3 Coinvariants 不 exact 的最小风险

令 $k=\mathbb F_p$，$G=C_p=\langle g\rangle$。考虑 $k[G]$-模上的 coinvariants functor
$$
(-)_G:k[G]\operatorname{-Mod}\to k\operatorname{-Mod}.
$$

**命题 X.4.** $(-)_G$ 不是 exact。

**证明.** 令 $I=(g-1)\subset k[G]$ 为 augmentation ideal。有短正合列
$$
0\to I\to k[G]\to k\to0
$$
其中 $k$ 是平凡模。取 coinvariants 得到右正合列
$$
I_G\to k[G]_G\to k_G\to0.
$$
$k[G]_G\cong k$，$k_G\cong k$。映射 $k[G]_G\to k_G$ 由 augmentation 诱导，是同构。若 coinvariants exact，则 $I_G\to k[G]_G$ 的像应为 kernel，即 $0$，并且左侧 exact 还要求 $0\to I_G$ 注入。但直接计算
$$
I_G=I/(g-1)I.
$$
在 $k[C_p]$ 中，$I$ 由 $g-1$ 生成，而 $(g-1)^p=g^p-1=0$。因此 $I/(g-1)I$ 非零。故左 exactness 失败。$\square$

**推论 X.5.** 在正特征中，对称幂
$$
(X^{\otimes n})_{\Sigma_n}
$$
不应被当作 exact functor 使用。

## X.4 特征 $2$ 中 Lie 反对称的边界

设 $k$ 是特征 $2$ 的域。若只要求 bracket 满足
$$
[x,y]+[y,x]=0,
$$
则因为 $-1=1$，该条件变成
$$
[x,y]=[y,x].
$$
它不推出
$$
[x,x]=0.
$$

**例 X.6.** 令 $V=k\cdot e$，定义
$$
[e,e]=e.
$$
则 bracket 是对称的，因此满足 $[x,y]+[y,x]=0$ 形式的反对称式，但不满足 alternating 条件 $[e,e]=0$。

**说明 X.7.** 因此一般底环上的 Lie operad 必须指定采用 alternating 关系还是只采用 antisymmetry relation。第六章中的 Lie 例子默认处在安全约定下；进入特征 $2$ 时必须重写。

## X.5 Strict commutative 与 $E_\infty$ 的边界例子

在特征 $0$ 链复形中，许多 rectification 定理允许把适当 $E_\infty$-algebra 与 commutative dg algebra 比较。正特征中不能无条件这样做。

**边界 X.8.** 设 $k=\mathbb F_p$。即使有 operad map
$$
E_\infty\to\operatorname{Com}
$$
逐 arity 为 quasi-isomorphism，也不能只由此推出
$$
\operatorname{Alg}_{E_\infty}(\mathbf{Ch}_k)
\simeq
\operatorname{Alg}_{\operatorname{Com}}(\mathbf{Ch}_k)
$$
为 Quillen equivalence。

**原因.** 自由 $\operatorname{Com}$-algebra 使用对称幂 coinvariants；命题 X.4 表明这些 functors 在正特征中有 exactness 风险。此外 $E_\infty$-结构可携带 power operations 等严格交换 dg algebra 未必保留的同伦信息。完整反例和正确定理必须引用附录 D/R 中的外部来源。

## X.6 Hochschild 的最小计算

令 $A=k$。作为 associative algebra，$k$ 的 enveloping algebra 为
$$
k\otimes k^{op}\cong k.
$$
因此
$$
HH_\*(k)\cong k\otimes^{\mathbf L}_{k}k\cong k.
$$

**命题 X.9.** 对 $A=k$，
$$
\int_{S^1}A\simeq k
$$
在外部输入定理 N.18 的圆周计算和 AF-2 定位下成立。

**证明.** 外部输入定理 N.18 给出
$$
\int_{S^1}k\simeq HH_\*(k).
$$
本节开头的 Hochschild 计算给出 $HH_\*(k)\cong k$。$\square$

**说明 X.10.** 该例不能推广为“$\int_{S^1}A$ 总等于 $A$”。对一般非交换代数，圆周积分是 Hochschild homology。

## X.7 非交换代数的圆周警告

设 $A$ 是非交换代数。Hochschild chains 的 $0$ 阶同调满足
$$
HH_0(A)\cong A/[A,A],
$$
其中 $[A,A]$ 是由 $ab-ba$ 张成的子空间。

**命题 X.11.** 若 $A=M_n(k)$，则
$$
HH_0(A)\cong k
$$
由矩阵 trace 诱导。

**证明边界.** $M_n(k)$ 的 commutator quotient 由普通 trace 识别为 $k$；证明使用矩阵单位 $E_{ij}$ 的交换子计算。完整 Morita invariance 给出 $HH_\*(M_n(k))\cong HH_\*(k)$，但 Morita invariance 是外部标准定理，本命题只记录 $HH_0$ 的低阶检查。$\square$

**说明 X.12.** 若错误地把 $\int_{S^1}A$ 当成 ordinary homology with coefficients in $A$，就不会出现 commutator quotient。这说明 factorization homology 的非交换性质是真实的。

## X.8 Module 边界条件的区间例子

令 $A$ 是 associative algebra，$M$ 是右 $A$-module，$N$ 是左 $A$-module。带边界 factorization homology 的区间公式为外部输入：
$$
\int_{[0,1]}(M,A,N)\simeq M\otimes_A^{\mathbf L}N.
$$

**例 X.13.** 若 $M=A$、$N=A$，使用正则左右作用，则
$$
A\otimes_A^{\mathbf L}A\simeq A.
$$
若 $M=k$、$N=k$ 是经 augmentation $A\to k$ 得到的 modules，则
$$
k\otimes_A^{\mathbf L}k
$$
一般不是 $A$。

**说明 X.14.** 区间值取决于端点边界条件。无边界 disk 归一化不能替代该计算。

## X.9 对称幂不保持准同构的链复形计算

令 $k=\mathbb F_p$。取同调分次链复形 $C$：
$$
0\longrightarrow k\cdot y \xrightarrow{d} k\cdot x\longrightarrow0,
\qquad |y|=2,\quad |x|=1,\quad d y=x.
$$
则 $C$ 是 acyclic。

考虑第 $p$ 对称幂
$$
\operatorname{Sym}^p(C)=(C^{\otimes p})_{\Sigma_p}
$$
中的元素 $y^p$。

**命题 X.15.** $y^p$ 在 $\operatorname{Sym}^p(C)$ 中给出非零同调类。因此 $\operatorname{Sym}^p$ 不保持准同构 $C\to0$。

**证明.** 微分满足 Leibniz rule。于是
$$
d(y^p)=\sum_{i=1}^{p}y^{i-1}(dy)y^{p-i}
=p\,y^{p-1}x=0
$$
因为 $\operatorname{char} k=p$。所以 $y^p$ 是 cycle。另一方面，$\operatorname{Sym}^p(C)$ 的最高同调次数为 $2p$，而 $y^p$ 正处在该最高次数；不存在次数 $2p+1$ 的元素，其微分可以等于 $y^p$。故 $y^p$ 不是 boundary。

又 $C$ acyclic，故 $C\to0$ 是 quasi-isomorphism。若 $\operatorname{Sym}^p$ 保持该 quasi-isomorphism，则 $\operatorname{Sym}^p(C)$ 应 acyclic；这与 $[y^p]\ne0$ 矛盾。$\square$

**推论 X.16.** 在 $\mathbf{Ch}_k$ 正特征中，自由 commutative dg algebra functor
$$
\operatorname{Sym}(C)=\bigoplus_{n\ge0}\operatorname{Sym}^n(C)
$$
不保持所有 acyclic chain complexes 到 $0$ 的 quasi-isomorphism。任何把 $E_\infty$-algebras rectifies to strict commutative dg algebras 的断言都必须加入额外假设或改用适当模型。

**说明 X.17.** 这个计算不是 Mandell 型 power operations 的完整反例；它只是最小代数风险：对称 coinvariants 在正特征中不具备特征 $0$ 下的 exact/homotopical 行为。

## X.10 小结

本附录给出的反例和样例说明：

1. arity $0$ 决定单位；
2. tensor algebra 与 symmetric algebra 由 $\Sigma_n$ coinvariants 区分；
3. 正特征中 coinvariants 不 exact；
4. 特征 $2$ 中 Lie 反对称与 alternating 不等价；
5. $E_\infty$ rectification 需要底范畴假设；
6. $\int_{S^1}A$ 是 Hochschild homology，不是普通同调；
7. 带边界 factorization homology 需要 module 边界条件；
8. 正特征中对称幂可以把 acyclic complex 送到有非零同调的 complex。
