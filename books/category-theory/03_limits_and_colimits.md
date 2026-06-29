# 第三章：极限与余极限

## 本章目标

本章用泛性质定义极限和余极限，并证明若范畴有终对象、二元积和等化子，则有所有有限极限。余极限部分通过对偶原则给出。

## 依赖前置知识

需要第二章的可表性、终对象和 Yoneda 观点。

## 3.1 锥与极限

**定义 3.1.** 设 $\mathcal J$ 为小范畴，$D:\mathcal J\to\mathcal C$ 为图形。对象 $X\in\mathcal C$ 到 $D$ 的锥（cone）由一族态射

$$
\lambda_j:X\to D(j),\qquad j\in\mathcal J
$$

组成，使得对任意态射 $\alpha:j\to k$，有

$$
D(\alpha)\circ\lambda_j=\lambda_k.
$$

**定义 3.2.** 图形 $D$ 的极限是一个锥 $(L,\pi_j:L\to D(j))$，使得对任意锥 $(X,\lambda_j)$，存在唯一态射 $u:X\to L$ 满足

$$
\pi_j\circ u=\lambda_j
$$

对所有 $j$ 成立。记作

$$
L=\lim_{\mathcal J}D.
$$

**命题 3.3.** 极限若存在，则在唯一同构意义下唯一。

**证明.** 极限是锥范畴中的终对象。由命题 2.2，终对象在唯一同构意义下唯一。展开说，两个极限锥 $(L,\pi_j)$ 和 $(L',\pi'_j)$ 互相诱导唯一态射 $u:L\to L'$ 与 $v:L'\to L$；复合 $v\circ u$ 与 $\operatorname{id}_L$ 都是从极限锥 $L$ 到自身并保持投影的态射，故相等。同理 $u\circ v=\operatorname{id}_{L'}$。$\square$

## 3.2 基本例子

**例子 3.4.** 若 $\mathcal J=\varnothing$，则从 $X$ 到空图形的锥没有额外数据。空图形的极限就是终对象。

**定义 3.5.** 对象 $A,B\in\mathcal C$ 的积是离散二点图形的极限，记作 $A\times B$。它带有投影

$$
p_A:A\times B\to A,\qquad p_B:A\times B\to B
$$

并满足自然双射

$$
\mathcal C(X,A\times B)\cong\mathcal C(X,A)\times\mathcal C(X,B).
$$

**定义 3.6.** 两个平行态射 $f,g:A\to B$ 的等化子（equalizer）是对象 $E$ 和态射 $e:E\to A$，满足 $f e=g e$，且对任意 $h:X\to A$ 若 $fh=gh$，存在唯一 $u:X\to E$ 使 $e u=h$。

**例子 3.7.** 在 $\mathbf{Set}$ 中，$f,g:A\to B$ 的等化子是子集

$$
\{a\in A\mid f(a)=g(a)\}\subseteq A
$$

连同包含映射。

## 3.3 有限极限的构造

**定理 3.8.** 若范畴 $\mathcal C$ 有终对象、所有二元积和所有等化子，则 $\mathcal C$ 有所有有限极限。

**证明.** 设 $D:\mathcal J\to\mathcal C$，其中 $\mathcal J$ 有有限多个对象和态射。构造两个有限积

$$
P=\prod_{j\in\operatorname{Ob}\mathcal J}D(j),
\qquad
Q=\prod_{\alpha:j\to k}D(k).
$$

对每个态射 $\alpha:j\to k$，有两个从 $P$ 到 $D(k)$ 的态射：

$$
D(\alpha)\circ p_j:P\to D(k),
\qquad
p_k:P\to D(k).
$$

由积的泛性质，这些分量给出两个态射

$$
s,t:P\rightrightarrows Q.
$$

取其等化子

$$
E\xrightarrow{e}P.
$$

令 $\pi_j=p_j\circ e:E\to D(j)$。等化子条件 $s e=t e$ 精确表示对每个 $\alpha:j\to k$ 有

$$
D(\alpha)\pi_j=\pi_k,
$$

所以 $(E,\pi_j)$ 是锥。

若 $(X,\lambda_j)$ 是任意锥，由积的泛性质得到唯一 $h:X\to P$ 使 $p_j h=\lambda_j$。锥条件说明 $s h=t h$，故由等化子泛性质得到唯一 $u:X\to E$ 使 $e u=h$。于是 $\pi_j u=\lambda_j$。唯一性由 $e$ 作为等化子态射和 $P$ 的积投影共同检测。$\square$

## 3.4 余锥与余极限

**定义 3.9.** 图形 $D:\mathcal J\to\mathcal C$ 的余锥（cocone）到对象 $X$ 是态射族

$$
\iota_j:D(j)\to X
$$

使得对任意 $\alpha:j\to k$，

$$
\iota_k\circ D(\alpha)=\iota_j.
$$

**定义 3.10.** 余极限是余锥 $(\operatorname{colim}D,\iota_j)$，使得对任意余锥 $(X,\mu_j)$，存在唯一态射 $u:\operatorname{colim}D\to X$ 满足 $u\iota_j=\mu_j$。

**定义 3.11.** 余积、余等化子分别是积、等化子的对偶概念。若范畴有始对象、所有二元余积和所有余等化子，则有所有有限余极限。

**证明.** 对定理 3.8 取反范畴。$\square$

## 3.5 函子范畴中的逐点极限

**命题 3.12.** 设 $\mathcal A$ 小，$\mathcal C$ 有形状 $\mathcal J$ 的极限。则函子范畴 $\operatorname{Fun}(\mathcal A,\mathcal C)$ 有形状 $\mathcal J$ 的极限，并逐点计算：

$$
(\lim_{\mathcal J}D)(A)\cong\lim_{\mathcal J}(D(-)(A)).
$$

**证明.** 对每个 $A\in\mathcal A$，令

$$
L(A)=\lim_{j\in\mathcal J}D(j)(A)
$$

并记其投影为

$$
\pi_{j,A}:L(A)\to D(j)(A).
$$

若 $u:A\to B$ 是 $\mathcal A$ 中态射，则对每个 $j$ 有态射

$$
D(j)(u):D(j)(A)\to D(j)(B).
$$

族

$$
D(j)(u)\circ\pi_{j,A}:L(A)\to D(j)(B)
$$

是到图形 $j\mapsto D(j)(B)$ 的锥，因为对 $\alpha:j\to k$，

$$
D(k)(u)\pi_{k,A}
=D(k)(u)D(\alpha)_A\pi_{j,A}
=D(\alpha)_B D(j)(u)\pi_{j,A}.
$$

由 $L(B)$ 的极限泛性质，存在唯一态射

$$
L(u):L(A)\to L(B)
$$

使得

$$
\pi_{j,B}L(u)=D(j)(u)\pi_{j,A}
$$

对所有 $j$ 成立。若 $u=\operatorname{id}_A$，则 $L(u)$ 与 $\operatorname{id}_{L(A)}$ 对所有投影 $\pi_{j,A}$ 有相同复合，故相等。若 $A\xrightarrow{u}B\xrightarrow{v}C$，则 $L(v)L(u)$ 与 $L(vu)$ 对每个投影 $\pi_{j,C}$ 的复合均为

$$
D(j)(v)D(j)(u)\pi_{j,A}=D(j)(vu)\pi_{j,A},
$$

故由极限唯一性相等。因此 $L:\mathcal A\to\mathcal C$ 是函子。

对每个 $j$，投影族 $\pi_{j,A}$ 对 $A$ 自然，因为上式正是自然性方块

$$
D(j)(u)\pi_{j,A}=\pi_{j,B}L(u).
$$

于是 $\pi_j:L\Rightarrow D(j)$ 组成函子范畴中的锥。

设 $M:\mathcal A\to\mathcal C$ 带有锥 $\lambda_j:M\Rightarrow D(j)$。对每个 $A$，族 $\lambda_{j,A}:M(A)\to D(j)(A)$ 是 $\mathcal C$ 中到 $j\mapsto D(j)(A)$ 的锥，故存在唯一

$$
\lambda_A:M(A)\to L(A)
$$

使 $\pi_{j,A}\lambda_A=\lambda_{j,A}$。若 $u:A\to B$，则 $L(u)\lambda_A$ 与 $\lambda_B M(u)$ 在投影 $\pi_{j,B}$ 下都等于

$$
D(j)(u)\lambda_{j,A}=\lambda_{j,B}M(u),
$$

其中用到 $\lambda_j$ 的自然性。故 $\lambda_A$ 组成自然变换 $\lambda:M\Rightarrow L$。它显然满足 $\pi_j\lambda=\lambda_j$；唯一性也逐对象由 $L(A)$ 的极限唯一性给出。所以 $L$ 是函子范畴中的极限。$\square$

## 3.6 本章小结

极限是锥范畴的终对象，余极限是余锥范畴的始对象。积、等化子、终对象足以构造所有有限极限；对偶地，余积、余等化子、始对象足以构造所有有限余极限。函子范畴中的极限在目标范畴有相应极限时逐点计算。

## 练习

**练习 3.1.** 写出拉回 $A\times_C B$ 作为极限的图形和泛性质。

**练习 3.2.** 证明在 $\mathbf{Set}$ 中，拉回集合是
$$
\{(a,b)\in A\times B\mid f(a)=g(b)\}.
$$

**练习 3.3.** 对偶化定义 3.6，写出余等化子的完整泛性质。

**练习 3.4.** 证明若 $\mathcal C$ 有所有小积和等化子，则 $\mathcal C$ 有所有小极限。

**练习 3.5.** 证明预层范畴 $\widehat{\mathcal C}$ 的小极限和小余极限逐点计算。
