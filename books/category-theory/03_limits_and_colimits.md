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

## 3.6 表示性刻画与保存极限

**命题 3.13.** 若 $D:\mathcal J\to\mathcal C$ 有极限 $L$，则对每个对象 $X\in\mathcal C$ 有自然同构

$$
\mathcal C(X,L)\cong \lim_{j\in\mathcal J}\mathcal C(X,Dj)
$$

其中右边是在 $\mathbf{Set}$ 中的极限。对偶地，若 $D$ 有余极限 $Q$，则

$$
\mathcal C(Q,X)\cong \lim_{j\in\mathcal J^{op}}\mathcal C(Dj,X).
$$

**证明.** 一个元素 $u\in\mathcal C(X,L)$ 给出族

$$
\pi_j u:X\to D(j).
$$

由于 $(L,\pi_j)$ 是锥，该族满足对任意 $\alpha:j\to k$ 有

$$
D(\alpha)\pi_j u=\pi_k u.
$$

这正是 $\mathbf{Set}$ 中极限

$$
\lim_j\mathcal C(X,Dj)
$$

的元素条件。反过来，一个相容族 $\lambda_j:X\to D(j)$ 是从 $X$ 到 $D$ 的锥，因此由 $L$ 的泛性质唯一给出 $u:X\to L$ 使 $\pi_j u=\lambda_j$。两种构造互逆。若 $f:X'\to X$，预复合 $u\mapsto uf$ 与族 $\lambda_j\mapsto \lambda_j f$ 相容，故同构对 $X$ 自然。余极限情形把箭头方向对偶即可。$\square$

**定义 3.14.** 设 $F:\mathcal C\to\mathcal D$ 为函子，且 $D:\mathcal J\to\mathcal C$ 有极限锥 $(L,\pi_j)$。若

$$
(F L,F\pi_j)
$$

是 $FD$ 的极限锥，则称 $F$ 保持该极限。若任意锥只要其 $F$-像是极限锥便自身是极限锥，则称 $F$ 反映该极限。余极限的保持与反映对偶定义。

命题 3.13 说明反变 Hom 函子 $\mathcal C(X,-)$ 保持一切存在的极限；协变 Hom 函子 $\mathcal C(-,X)$ 把余极限送为 $\mathbf{Set}$ 中的极限。

## 3.7 共尾函子

**定义 3.15.** 函子 $U:\mathcal I\to\mathcal J$ 称为终函子或共尾函子（final functor），若对每个 $j\in\mathcal J$，逗号范畴 $j/U$ 非空且其底层无向图连通。这里 $j/U$ 的对象是二元组

$$
(i,\alpha:j\to U i).
$$

对偶地，$U$ 称为始函子（initial functor），若对每个 $j$，逗号范畴 $U/j$ 非空且连通。

**定理 3.16（共尾性）.** 若 $U:\mathcal I\to\mathcal J$ 是共尾函子，且 $D:\mathcal J\to\mathcal C$，则预复合给出对每个 $X\in\mathcal C$ 自然的双射

$$
\operatorname{Cocone}(D,X)\cong \operatorname{Cocone}(DU,X).
$$

因此，只要相应余极限存在，就有典范同构

$$
\operatorname{colim}_{i\in\mathcal I}D(Ui)\cong
\operatorname{colim}_{j\in\mathcal J}D(j).
$$

对偶地，若 $U$ 是始函子，则

$$
\lim_{j\in\mathcal J}D(j)\cong
\lim_{i\in\mathcal I}D(Ui)
$$

在相应极限存在时成立。

**证明.** 从 $D$ 到 $X$ 的余锥 $\lambda_j:D(j)\to X$ 限制为 $DU$ 的余锥 $\lambda_{Ui}:D(Ui)\to X$。

反过来设给定 $DU$ 的余锥

$$
\mu_i:D(Ui)\to X.
$$

对每个 $j$，选取 $j/U$ 的一个对象 $(i,\alpha:j\to Ui)$，并定义

$$
\lambda_j=\mu_i\circ D(\alpha):D(j)\to X.
$$

该定义与选择无关。事实上，若

$$
u:(i,\alpha)\to(i',\alpha')
$$

是 $j/U$ 中的态射，则 $\alpha'=U(u)\alpha$，而余锥条件给出

$$
\mu_i=\mu_{i'}D(Uu).
$$

于是

$$
\mu_iD(\alpha)
=\mu_{i'}D(Uu)D(\alpha)
=\mu_{i'}D(\alpha').
$$

由于 $j/U$ 连通，沿任意有限折线反复使用此等式，任意两个选择得到同一态射 $\lambda_j$。

现在验证 $\lambda$ 是 $D$ 的余锥。若 $\beta:j'\to j$，并且 $(i,\alpha:j\to Ui)$ 是 $j/U$ 的对象，则 $(i,\alpha\beta:j'\to Ui)$ 是 $j'/U$ 的对象。因此由定义的选择无关性，

$$
\lambda_{j'}=\mu_iD(\alpha\beta)=\mu_iD(\alpha)D(\beta)=\lambda_jD(\beta).
$$

所以 $\lambda$ 是余锥。两个构造互逆是直接的：从 $D$ 的余锥限制再按上述公式恢复时，余锥条件给出同一族；从 $DU$ 的余锥扩张再限制时，取对象 $Ui$ 时可用 $(i,\operatorname{id}_{Ui})$，得到 $\mu_i$。自然性来自所有构造只用后复合 $X\to X'$。余极限陈述是表示性结论；始函子和极限的陈述为对偶命题。$\square$

**命题 3.17.** 若 $U:\mathcal I\to\mathcal J$ 有左伴随 $L:\mathcal J\to\mathcal I$，则 $U$ 是共尾函子。若 $U$ 有右伴随，则 $U$ 是始函子。

**证明.** 设 $L\dashv U$，单位为 $\eta_j:j\to U L j$。对每个 $j$，对象

$$
(L j,\eta_j)
$$

是 $j/U$ 的始对象。事实上，对任意 $(i,\alpha:j\to Ui)$，伴随双射给出唯一态射 $\bar\alpha:L j\to i$，满足 $U(\bar\alpha)\eta_j=\alpha$；这正是 $j/U$ 中从 $(L j,\eta_j)$ 到 $(i,\alpha)$ 的唯一态射。因此 $j/U$ 非空且连通，$U$ 共尾。右伴随情形对偶，得到 $U/j$ 中的终对象。$\square$

**例子 3.18.** 若 $\mathcal J$ 有终对象 $t$，则选择 $t$ 的函子 $* \to \mathcal J$ 是共尾函子；因此任意 $D:\mathcal J\to\mathcal C$ 的余极限若存在，则为 $D(t)$。对偶地，若 $\mathcal J$ 有始对象 $s$，则 $\lim_{\mathcal J}D\cong D(s)$。

## 3.8 例子、反例与创造极限

**例子 3.19（Set 中的余等化子）.** 在 $\mathbf{Set}$ 中，平行函数 $f,g:A\rightrightarrows B$ 的余等化子是商映射

$$
q:B\to B/{\sim},
$$

其中 $\sim$ 是由关系 $f(a)\sim g(a)$ 对所有 $a\in A$ 生成的最小等价关系。

确实，$qf=qg$。若 $h:B\to X$ 满足 $hf=hg$，则 $h$ 对生成关系取相同值，因此对其生成的等价关系取相同值。于是存在唯一函数 $\bar h:B/{\sim}\to X$ 使 $\bar hq=h$。这正是余等化子的泛性质。

**定义 3.20.** 函子 $F:\mathcal C\to\mathcal D$ 称为创造某类极限（creates limits），若对任意图形 $D:\mathcal J\to\mathcal C$，只要 $F D$ 在 $\mathcal D$ 中有极限锥，且该极限锥的顶点和投影可提升为 $\mathcal C$ 中的锥，则这个提升锥自动是 $\mathcal C$ 中的极限锥，并且这种提升唯一。余极限的创造对偶定义。

**例子 3.21（代数范畴中的逐底层极限）.** 忘却函子

$$
U:\mathbf{Grp}\to\mathbf{Set}
$$

创造小极限。给定群值图形 $D:\mathcal J\to\mathbf{Grp}$，先在 $\mathbf{Set}$ 中取底层集合极限

$$
L=\lim_{\mathcal J}U D.
$$

把 $L$ 看作所有相容族 $(x_j)_j$ 的集合。逐坐标定义乘法、单位和逆：

$$
(x_j)_j(y_j)_j=(x_jy_j)_j,\qquad
e=(e_j)_j,\qquad
(x_j)_j^{-1}=(x_j^{-1})_j.
$$

因为 $D(\alpha)$ 是群同态，相容族在这些运算下仍相容。投影 $L\to D(j)$ 是群同态。若 $G$ 是任意群且给出到 $D$ 的相容群同态族，则其底层函数唯一分解经集合极限；逐坐标公式说明该分解仍是群同态。因此 $L$ 是 $\mathbf{Grp}$ 中的极限。

**例子 3.22（假设不能删）.** 设 $\mathbf{Set}_{\ne\varnothing}$ 为非空集合和函数组成的范畴。它有终对象和二元积：单点集仍终，两个非空集合的笛卡尔积仍非空。但是它没有所有等化子。取两函数

$$
f,g:\{*\}\rightrightarrows\{0,1\},
\qquad f(*)=0,\quad g(*)=1.
$$

在 $\mathbf{Set}$ 中等化子是空集。若在 $\mathbf{Set}_{\ne\varnothing}$ 中存在等化子 $e:E\to\{*\}$，则 $E$ 非空，取 $x\in E$，有

$$
f(e(x))=g(e(x)),
$$

即 $0=1$，矛盾。因此定理 3.8 中“等化子”假设不能由终对象和二元积推出。

## 3.9 本章小结

极限是锥范畴的终对象，余极限是余锥范畴的始对象。积、等化子、终对象足以构造所有有限极限；对偶地，余积、余等化子、始对象足以构造所有有限余极限。函子范畴中的极限在目标范畴有相应极限时逐点计算。表示性把极限转化为 Hom 集的极限，共尾函子允许在不改变余极限的情况下缩小指标范畴；始函子给出极限的对偶缩小原则。

## 练习

**练习 3.1.** 写出拉回 $A\times_C B$ 作为极限的图形和泛性质。

**练习 3.2.** 证明在 $\mathbf{Set}$ 中，拉回集合是
$$
\{(a,b)\in A\times B\mid f(a)=g(b)\}.
$$

**练习 3.3.** 对偶化定义 3.6，写出余等化子的完整泛性质。

**练习 3.4.** 证明若 $\mathcal C$ 有所有小积和等化子，则 $\mathcal C$ 有所有小极限。

**练习 3.5.** 证明预层范畴 $\widehat{\mathcal C}$ 的小极限和小余极限逐点计算。

**练习 3.6.** 证明命题 3.13 的余极限版本，并明确自然性变量。

**练习 3.7.** 证明若 $U:\mathcal I\to\mathcal J$ 有左伴随，则对任意 $j$，$j/U$ 有始对象。

**练习 3.8.** 设 $A$ 是偏序集 $P$ 的子偏序。把包含 $A\hookrightarrow P$ 为共尾函子的条件翻译成偏序语言。

**练习 3.9.** 若 $\mathcal J$ 有终对象 $t$，直接证明 $D(t)$ 满足 $\operatorname{colim}_{\mathcal J}D$ 的泛性质。

**练习 3.10.** 设 $A\subseteq P$ 是滤过偏序集 $P$ 的共尾子偏序，$D:P\to\mathbf{Set}$。证明自然映射
$$
\operatorname{colim}_{a\in A}D(a)\to\operatorname{colim}_{p\in P}D(p)
$$
是双射。

**练习 3.11.** 证明例子 3.19 中的等价关系确实是满足 $qf=qg$ 的最小等价关系。

**练习 3.12.** 证明忘却函子 $\mathbf{Ring}\to\mathbf{Set}$ 创造小极限，其中环取含幺环且同态保持单位。

**练习 3.13.** 证明 $\mathbf{Set}_{\ne\varnothing}$ 有所有有限积，但没有所有有限极限。
