# 附录 A：导出完备化、Koszul complex 与 $p^\infty$-torsion

## 本附录目标

本附录固定正文所用的导出完备化模型，并区分普通 Tor-amplitude 与
complete Tor-amplitude。这里采用 Bhatt-Scholze 的 Koszul tower 口径；除非
另有假设，不能把它替换成商环 $A/J^n$ 的 tower。

## A.1 Koszul tower 与 derived completion

**定义 A.1（Koszul complex）.** 令 $A$ 为环，$f\in A$。本书把
$$
K_A(f)=[A\xrightarrow{f}A]
$$
放在 cohomological degrees $[-1,0]$，右端的 $A$ 位于 degree $0$。对有限
序列 $\mathbf f=(f_1,\ldots,f_r)$，定义
$$
K_A(\mathbf f)=K_A(f_1)\otimes_A\cdots\otimes_AK_A(f_r).
$$
特别地，$H^0(K_A(f))=A/fA$，而
$H^{-1}(K_A(f))=A[f]$；只有当 $f$ 是 nonzerodivisor 时，
$K_A(f)\simeq A/fA$。

对 $n\ge1$，从 $K_A(f^{n+1})$ 到 $K_A(f^n)$ 的 transition map 在 degree
$-1$ 上为乘以 $f$，在 degree $0$ 上为恒等。对多个生成元取这些映射的
张量积，得到 inverse system $\{K_A(f_1^n,\ldots,f_r^n)\}_n$。

**定义 A.2（derived $J$-completion）.** 设
$J=(f_1,\ldots,f_r)\subset A$ 为有限生成理想，$M\in D(A)$。定义
$$
M^{\wedge,L}_J
=R\varprojlim_n\left(M\otimes_A^LK_A(f_1^n,\ldots,f_r^n)\right).
$$
若自然映射 $M\to M^{\wedge,L}_J$ 是 $D(A)$ 中的同构，则称 $M$ derived
$J$-complete。生成元无关性以及 completion functor 的函子性作为外部输入
使用；来源为 Bhatt-Scholze, *Prisms and Prismatic Cohomology*, §1.2，
locator `BS-DCOMP`。

**警告 A.3（商幂 tower 不是一般定义）.** 一般没有公式
$$
M^{\wedge,L}_J\simeq
R\varprojlim_n(M\otimes_A^LA/J^n).
$$
若 $J$ 由 regular sequence 生成，或 $A$ noetherian，则该商幂模型与
Koszul 模型相容；非 noetherian 情形没有这些假设时不得使用它。

**命题 A.4（regular principal case）.** 设 $f$ 是 $A$ 的
nonzerodivisor，且 $A\xrightarrow{\sim}\varprojlim_nA/f^nA$。则 $A$ derived
$(f)$-complete。

**证明.** 因 $f^n$ 仍为 nonzerodivisor，定义 A.1 给出
$K_A(f^n)\simeq A/f^nA$。商映射 $A/f^{n+1}A\to A/f^nA$ 均满射，故该
inverse system 的 $\varprojlim^1$ 为零。因此
$$
R\varprojlim_nK_A(f^n)
\simeq \varprojlim_nA/f^nA
\simeq A.
$$
复合映射正是定义 A.2 的 completion map。证毕。

**命题 A.5（principal bounded-torsion criterion）.** 设 $M$ 为离散
$A$-module，且存在 $N\ge0$ 使
$$
M[f^\infty]=M[f^N].
$$
则 $M^{\wedge,L}_{(f)}$ 集中在 degree $0$，并且自然同构于 ordinary
completion：
$$
M^{\wedge,L}_{(f)}\simeq\varprojlim_nM/f^nM.
$$

**证明.** 令 $K_n=M\otimes_AK_A(f^n)=[M\xrightarrow{f^n}M]$，仍置于
degrees $[-1,0]$。于是
$$
H^{-1}(K_n)=M[f^n],\qquad H^0(K_n)=M/f^nM.
$$
在 $H^{-1}$ 上，transition map 是乘以 $f$。当 $n\ge N$ 时，所有
$M[f^n]$ 都等于 $M[f^N]$。若 $N=0$，这些 groups 已全为零；若
$N\ge1$，任意连续 $N$ 个 transition maps 的复合为乘以 $f^N$，因而为
零。故此 inverse system pro-zero；它的 $\varprojlim$ 与
$\varprojlim^1$ 均为零。另一方面，$M/f^{n+1}M\to M/f^nM$ 满射，所以
$\varprojlim^1_nM/f^nM=0$。对 $R\varprojlim K_n$ 使用 Milnor 短正合列，
可得除 degree $0$ 外的 cohomology 全部为零，且 degree $0$ 等于
$\varprojlim_nM/f^nM$。证毕。

**外部输入定理 A.6（noetherian comparison）.** 设 $A$ noetherian，
$J\subset A$ 为理想，$M\in D^b(A)$，且每个 $H^i(M)$ 有限生成。则 Koszul
completion 与商幂 completion 相容，并有
$$
H^i(M^{\wedge,L}_J)\cong
\varprojlim_n H^i(M)/J^nH^i(M).
$$
这是 Artin-Rees 控制下的外部输入；本书不重证一般多生成元情形。

## A.2 $p^\infty$-torsion 与 boundedness

**定义 A.7.** 对环或模 $M$，定义
$$
M[p^n]=\{x\in M\mid p^nx=0\},\qquad
M[p^\infty]=\bigcup_{n\ge0}M[p^n].
$$
称 $M$ 的 $p^\infty$-torsion 有界，如果存在 $N$ 使
$M[p^\infty]=M[p^N]$。

**命题 A.8.** 若 $M$ 无 $p$-torsion，则
$M[p^\infty]=0=M[p^0]$，因而其 $p^\infty$-torsion 有界。

**证明.** 乘以 $p$ 在 $M$ 上单射。若 $p^nx=0$，从
$p(p^{n-1}x)=0$ 开始反复使用单射性，得到 $x=0$。证毕。

**说明 A.9.** Bounded prism 的 boundedness 恰好要求 $A/I$ 的
$p^\infty$-torsion 有界。命题 A.5 说明，在 principal 情形，这一条件会
消除 completion 的负次数 torsion 项；它不是 “$A/I$ 无 $p$-torsion” 的
同义词，也不允许把所有 derived tensor products 改成 ordinary tensor products。

## A.3 $p$-、$I$-与 $(p,I)$-完备

**定义 A.10.** 若 $I=(d_1,\ldots,d_r)$，则 derived $(p,I)$-completion
指对理想 $(p,d_1,\ldots,d_r)$ 使用定义 A.2。Prism 定义要求环 $A$ 对该
联合理想 derived complete，而不只是 derived $p$-complete 或 derived
$I$-complete。

**警告 A.11.** 三种完备性不能省略下标。例如
$\mathbf Z_p\langle t\rangle$ 是 $p$-adically complete，但其 $(t)$-adic
completion 是更大的 $\mathbf Z_p[[t]]$；反向，
$\mathbf Z_{(p)}[[t]]$ 是 $(t)$-adically complete，却不是 $p$-adically
complete。因此正文中的 “complete” 必须写出理想。

**外部输入定理 A.12（bounded prism 的经典完备性）.** 若 $(A,I)$ 是
bounded prism，则 $A$ 不仅 derived $(p,I)$-complete，而且 classically
$(p,I)$-complete。更进一步，若 $M$ 同时 derived $(p,I)$-complete 且
$(p,I)$-completely flat，则 $M$ 是离散且 classically $(p,I)$-complete，
并且 $M[I^n]=0$ 对所有 $n\ge1$；此外各 $M/I^nM$ 的
$p^\infty$-torsion 有界。来源为 Bhatt-Scholze, Lemma 3.7，locator
`BS-PRISM-DEF`。这些结论依赖 boundedness；本书不把它们并入 prism 的
形式定义。

Derived completeness 不能省略：例如 $A[1/f]$ 对 $J=(f)$ 测试所有
$J$-torsion modules 时 tensor 为零，因而是 $J$-completely flat，但它的
$J$-adic completion 为零，通常不等于 $A[1/f]$。

## A.4 Complete flatness 与 Tor-amplitude

**定义 A.13（$J$-completely flat）.** 令 $J\subset A$ 为有限生成理想，
$M\in D(A)$。若对每个 $J$-torsion 离散 $A$-module $N$，
$$
M\otimes_A^LN
$$
都集中在 degree $0$，则称 $M$ 为 $J$-completely flat。若此外
$M\otimes_A^LA/J$ 是 faithfully flat $A/J$-module，则称 $M$
$J$-completely faithfully flat。环映射 $A\to B$ 的相应性质指 $B$ 作为
$A$-complex 具有该性质。

这里 $J$-torsion 指每个元素都被某个 $J$ 的幂杀死。只测试被单个固定
$J^n$ 整体杀死的 modules（允许 $n$ 随 module 变化）也给出同一条件，
因为任意 $J$-torsion module 是这些 bounded-power-torsion 子模的 filtered
colimit。

**命题 A.14.** 若 $M$ 是 $J$-completely flat，则
$M\otimes_A^LA/J$ 集中在 degree $0$，且其 $H^0$ 是 flat
$A/J$-module。

**证明.** 取 $N=A/J$ 得集中性。再令 $Q$ 为任意离散
$A/J$-module；它也是 $J$-torsion $A$-module，并且结合律给出
$$
(M\otimes_A^LA/J)\otimes_{A/J}^LQ
\simeq M\otimes_A^LQ.
$$
右侧集中在 degree $0$，故 $H^0(M\otimes_A^LA/J)$ 对所有 $Q$ 无高阶
Tor，即为 flat。证毕。

**定义 A.15（finite complete Tor-amplitude）.** 称 $M$ 有有限
$J$-complete Tor-amplitude，如果
$M\otimes_A^LA/J\in D(A/J)$ 有有限 ordinary Tor-amplitude。等价地，
存在统一常数 $c\ge0$，使 $M\otimes_A^L-$ 把离散的
$J$-power-torsion modules 送入 $D^{[-c,c]}(A)$；该等价作为
`BS-DCOMP` 的外部输入。$c=0$ 正是定义 A.13 的 complete flatness。

**警告 A.16.** Ordinary Tor-amplitude 与 complete Tor-amplitude 是不同
条件。前者测试所有 $A$-modules；后者只控制 modulo $J$ 或 $J$-power
torsion 的行为。Prismatic site 的 covers 使用 complete flatness，而第五章
的 universal-coefficient 短正合列要求 ordinary Tor-dimension 至多一。
此外，derived tensor functor 本来就保持 exact triangles；“保持 exact
triangles” 不能作为 flatness 定义。

## 本附录小结

Derived completion 的基础模型是 Koszul tower。商幂 tower 只在
noetherian、regular-sequence 或其他 weak-proregular 情形下可安全替代。
Boundedness 控制 principal completion 中的高阶 torsion，bounded prism
还由 Bhatt-Scholze 的深引理获得经典 $(p,I)$-完备性。Complete flatness
与 ordinary flatness、ordinary Tor-amplitude 必须分开记录。

## 练习

**练习 A.1.** 对 $A=\mathbf Z_p$、$J=(p)$、$M=A$，用 Koszul tower
计算 $M^{\wedge,L}_J$。

**练习 A.2.** 若 $M$ 无 $p$-torsion，证明 $M/p^{n+1}M\to M/p^nM$
满射，并说明该 inverse system 满足 Mittag-Leffler 条件。

**练习 A.3.** 写出 $K_A(f,g)$ 在 degrees $[-2,0]$ 的三项 complex，
并固定 differential 的符号。
