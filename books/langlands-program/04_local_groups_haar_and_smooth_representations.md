# 第四章：局部紧群、Haar 测度与光滑表示

## 本章目标

本章建立局部 Langlands 所需的表示论语言：局部紧群、Haar 测度、卷积代数、光滑表示、可容许表示、Hecke 代数和球表示。后续 `GL(n)` 和一般还原群的自守表示都以这些概念为局部基础。

## 依赖前置知识

需要局部紧拓扑群、向量空间和基本表示论。非 Archimedean 局部域上的代数群 $G(F)$ 是全不连通局部紧群；Archimedean 情形需要 Lie 群表示论，本章只给出接口。附录 Z 记录 Harish-Chandra character、Plancherel 和 Bernstein-Paley-Wiener 接口；附录 AA 记录 Bruhat-Tits、parahoric 和 hyperspecial 的结构来源。

收口归一化回指：本章卷积、Hecke 幂等元、开紧子群体积和归一化抛物诱导的 convention 见 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 第 3、4 节。

## 4.1 局部紧群和 Haar 测度

**定义 4.1.** 局部紧群是拓扑群 $G$，其拓扑空间为 Hausdorff 且每点有紧邻域。若单位元有一组由开紧子群组成的邻域基，则称 $G$ 为局部 profinite 群（locally profinite group）。每个局部 profinite 群都是全不连通局部紧群；反过来，Van Dantzig 定理说明全不连通局部紧群也有开紧子群邻域基。

**例 4.2.** 若 $F$ 为非 Archimedean 局部域，则 $F$ 的加法群、$F^\times$、$\operatorname{GL}_n(F)$ 以及更一般的 $G(F)$ 都是局部 profinite 群。$\operatorname{GL}_n(\mathcal O_F)$ 是 $\operatorname{GL}_n(F)$ 的开紧子群。

**外部输入定理 4.3（Haar 测度）.** 设 $G$ 为局部紧群。存在非零左不变正则 Borel 测度 $dg$，称为左 Haar 测度，且任意两个左 Haar 测度只差一个正实数倍。右 Haar 测度同理存在。存在连续同态
$$
\delta_G:G\to\mathbb R_{>0}
$$
称为模函数，使得
$$
d(gh)=\delta_G(h)^{-1}dg
$$
描述左 Haar 测度在右平移下的变化。若 $\delta_G=1$，则称 $G$ 为 unimodular。

**例 4.4.** 紧群、离散群、交换局部紧群和 reductive $p$-adic 群都是 unimodular。若 $P=MN$ 是 reductive 群的抛物子群，则 $P(F)$ 通常不是 unimodular；其模函数记为 $\delta_P$。

## 4.2 卷积代数

本节设 $G$ 为局部 profinite 群，并固定左 Haar 测度 $dg$。

**定义 4.5.** 记 $C_c^\infty(G)$ 为所有紧支撑、局部常值的复值函数。对 $f_1,f_2\in C_c^\infty(G)$，定义卷积
$$
(f_1*f_2)(g)=\int_G f_1(x)f_2(x^{-1}g)\,dx.
$$
若 $G$ 为 unimodular，定义反合 involution
$$
f^\vee(g)=f(g^{-1}).
$$

**命题 4.6.** $C_c^\infty(G)$ 在卷积下构成结合代数。

**证明.** 对 $f_1,f_2,f_3\in C_c^\infty(G)$，由紧支撑性，相关积分都在紧集上进行；由局部常值性，积分可化为有限个开紧双陪集上的有限和。利用左 Haar 测度的左不变性和 Fubini 定理，有
$$
\begin{aligned}
((f_1*f_2)*f_3)(g)
&=\int_G\int_G f_1(y)f_2(y^{-1}x)f_3(x^{-1}g)\,dy\,dx\\
&=\int_G\int_G f_1(y)f_2(z)f_3(z^{-1}y^{-1}g)\,dz\,dy\\
&=(f_1*(f_2*f_3))(g),
\end{aligned}
$$
其中第二步令 $z=y^{-1}x$ 并使用左不变性 $dx=dz$。$\square$

**定义 4.7.** 若 $J\subset G$ 为开紧子群，归一化 Haar 测度使 $\operatorname{vol}(J)=1$。定义
$$
e_J=\mathbf 1_J\in C_c^\infty(G).
$$
则 $e_J$ 称为 $J$ 的归一化幂等元。

**命题 4.8.** 在 $\operatorname{vol}(J)=1$ 的归一化下，
$$
e_J*e_J=e_J.
$$

**证明.** 对 $g\in G$，
$$
(e_J*e_J)(g)=\int_G\mathbf 1_J(x)\mathbf 1_J(x^{-1}g)\,dx.
$$
被积函数非零当且仅当 $x\in J$ 且 $x^{-1}g\in J$，即 $x\in J\cap gJ$。若 $g\in J$，则 $J\cap gJ=J$，积分为 $\operatorname{vol}(J)=1$。若 $g\notin J$，则 $J\cap gJ=\varnothing$，积分为 $0$。故卷积等于 $\mathbf 1_J$。$\square$

## 4.3 光滑表示

**定义 4.9.** 设 $G$ 为局部 profinite 群。$G$ 的复表示是复向量空间 $V$ 及群同态
$$
\pi:G\to\operatorname{Aut}_{\mathbb C}(V).
$$
表示 $(\pi,V)$ 称为光滑的（smooth），若对每个 $v\in V$，稳定子
$$
G_v=\{g\in G:\pi(g)v=v\}
$$
是 $G$ 的开子群。

**定义 4.10.** 对开紧子群 $J\subset G$，$J$-不变量空间定义为
$$
V^J=\{v\in V:\pi(j)v=v\text{ for all }j\in J\}.
$$
光滑表示 $(\pi,V)$ 称为可容许的（admissible），若对每个开紧子群 $J\subset G$，$V^J$ 是有限维复向量空间。

**命题 4.11.** 设 $(\pi,V)$ 为光滑表示。卷积公式
$$
\pi(f)v=\int_G f(g)\pi(g)v\,dg
$$
定义了 $C_c^\infty(G)$ 在 $V$ 上的代数作用。

**证明.** 因为 $f$ 紧支撑且局部常值，而 $v$ 的稳定子开，函数 $g\mapsto f(g)\pi(g)v$ 在紧支撑上只取有限多个值；积分因此是有限线性组合，定义良好。卷积作用的结合性由命题 4.6 的同一 Fubini 计算给出：
$$
\pi(f_1*f_2)=\pi(f_1)\pi(f_2).
$$
$\square$

**命题 4.12.** 若 $J\subset G$ 为开紧子群且 $\operatorname{vol}(J)=1$，则 $\pi(e_J)$ 是 $V$ 到 $V^J$ 的投影。

**证明.** 对 $v\in V$，
$$
\pi(e_J)v=\int_J\pi(j)v\,dj.
$$
若 $j_0\in J$，则左不变性给出
$$
\pi(j_0)\pi(e_J)v
=\int_J\pi(j_0j)v\,dj
=\int_J\pi(j)v\,dj
=\pi(e_J)v,
$$
故 $\pi(e_J)v\in V^J$。若 $v\in V^J$，则
$$
\pi(e_J)v=\int_Jv\,dj=\operatorname{vol}(J)v=v.
$$
所以 $\pi(e_J)$ 是投影到 $V^J$ 的算子。$\square$

## 4.4 Hecke 代数和球表示

**定义 4.13.** 设 $J\subset G$ 为开紧子群。Hecke 代数
$$
\mathcal H(G,J)=e_J*C_c^\infty(G)*e_J
$$
等同于所有紧支撑、双 $J$-不变函数构成的卷积代数。

若 $(\pi,V)$ 为光滑表示，则 $\mathcal H(G,J)$ 作用在 $V^J$ 上。

**定义 4.14.** 设 $F$ 为非 Archimedean 局部域，$G$ 为 $F$ 上非分歧 reductive 群，$K=G(\mathcal O_F)$ 为选定 hyperspecial maximal compact subgroup。不可约光滑表示 $(\pi,V)$ 称为球表示（spherical representation）或非分歧表示，若
$$
V^K\ne 0.
$$

**外部输入定理 4.15（Satake 同构，接口形式）.** 对非分歧 reductive 群 $G/F$，球 Hecke 代数
$$
\mathcal H(G(F),G(\mathcal O_F))
$$
是交换代数，并与对偶群 $\widehat G$ 的表示环或其 Weyl 不变量坐标环有典范同构。不可约球表示的 Hecke 本征值等价于 $\widehat G$ 中的半单共轭类。

本定理是“非分歧局部 Langlands 参数”的表示论入口。第五章将把该半单共轭类写成 Frobenius 参数。

**注 4.15.1.** 附录 P 把本定理拆成球 Hecke 代数、Cartan 分解、Satake 变换和 `GL(n)` 非分歧公式；附录 AA 解释 hyperspecial subgroup 和 Cartan 分解的 Bruhat-Tits 来源。第四章只使用其接口；涉及具体 eigenvalue 和 Euler 因子的计算应引用附录 P/AA。

## 4.5 抛物诱导的最小接口

**定义 4.16.** 设 $G$ 为局部域 $F$ 上 reductive 群，$P=MN$ 为抛物子群，其中 $M$ 为 Levi 因子，$N$ 为 unipotent radical。若 $(\sigma,W)$ 是 $M(F)$ 的光滑表示，则归一化抛物诱导定义为满足以下条件的光滑函数空间：
$$
\operatorname{Ind}_{P(F)}^{G(F)}(\sigma)
=
\left\{
f:G(F)\to W:
f(mng)=\delta_P(m)^{1/2}\sigma(m)f(g)
\right\},
$$
并由右平移给出 $G(F)$ 作用。

**注 4.17.** 归一化因子 $\delta_P^{1/2}$ 的作用是使诱导表示与局部函数方程和酉性更好相容。不同资料可能使用非归一化诱导；本书默认使用归一化诱导。

**定义 4.18.** 对 $G=\operatorname{GL}_n(F)$，主级数表示是从 Borel 子群 $B=TN$ 的特征
$$
\chi_1\otimes\cdots\otimes\chi_n:T(F)\to\mathbb C^\times
$$
归一化诱导到 $\operatorname{GL}_n(F)$ 所得的表示。

**例 4.19.** 对 $\operatorname{GL}_2(F)$，若 $\chi_1,\chi_2:F^\times\to\mathbb C^\times$ 为非分歧特征，则
$$
\operatorname{Ind}_{B(F)}^{\operatorname{GL}_2(F)}(\chi_1\otimes\chi_2)
$$
含有非零 $\operatorname{GL}_2(\mathcal O_F)$-不变量。其 Satake 参数应为对角半单共轭类
$$
\operatorname{diag}(\chi_1(\varpi),\chi_2(\varpi))
\in\operatorname{GL}_2(\mathbb C)
$$
在适当归一化下的共轭类。

## 4.6 Archimedean 位置的接口

若 $F=\mathbb R$ 或 $\mathbb C$，群 $G(F)$ 是实 Lie 群，不属于本章前几节的局部 profinite 口径。光滑表示需要替换为 Fréchet 表示、Harish-Chandra 模和 $(\mathfrak g,K)$-模。

**外部输入定理 4.20（Harish-Chandra 理论，接口形式）.** 对实 reductive 群，合适的不可约可容许表示可由 Harish-Chandra 模研究；其无穷小特征和 Langlands 分类给出 Archimedean 局部 Langlands 参数所需的表示论输入。

本书在进入 Archimedean 局部 Langlands 时会单独声明所需版本。

## 4.7 本章小结

本章建立了非 Archimedean 局部表示论的最低语言。光滑表示把拓扑群作用转化为开紧不变量的代数结构；Hecke 代数把双陪集卷积变成算子；Satake 同构把非分歧表示的 Hecke 本征值翻译为对偶群中的半单共轭类。这正是局部 Langlands 参数的自守侧入口。

## 练习

**练习 4.1.** 证明 $\operatorname{GL}_n(\mathcal O_F)$ 是 $\operatorname{GL}_n(F)$ 的开紧子群。

**练习 4.2.** 设 $G$ 为离散群。说明本章的 $C_c^\infty(G)$ 与通常的群代数中有限支撑函数代数一致。

**练习 4.3.** 证明若 $(\pi,V)$ 是光滑表示，则
$$
V=\bigcup_J V^J
$$
其中 $J$ 遍历 $G$ 的开紧子群。

**练习 4.4.** 对 $G=F^\times$ 和 $J=\mathcal O_F^\times$，计算 $\mathcal H(G,J)$ 的基，并说明它如何记录非分歧特征的值 $\chi(\varpi)$。
