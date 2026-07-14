# 第十三章：全局自守表示和标准 L 函数

局部参数只有在能够同时装配到所有位置时，才会产生全局算术信息。自守表示 $\pi=\bigotimes_v'\pi_v$ 正是这种装配：几乎所有 $\pi_v$ 非分歧，其 Satake 参数与 L 群表示 $r$ 共同定义局部 Euler 因子，继而形成部分 L 函数。这里必须把形式上的 Euler 乘积与解析延拓、函数方程和有界性分开；前者由局部数据定义，后者在一般群上属于深定理或猜想。本章从自守商和尖点条件出发，直到完整 L 函数及其解析问题。

所需的 adelic、L 群和局部参数语言已经在第一、十一、十二章建立。离散谱分解、强重数一、Godement--Jacquet、Langlands--Shahidi 与 Rankin--Selberg 方法均按外部输入标记。本文采用 automorphic normalization；Haar 测度、Satake 参数和经典模形式变量之间的转换见 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 第 3、4、5、8 节。

## 13.1 自守商与中心特征

本章固定整体域 $K$，其 adele 环为 $\mathbb A_K$。设 $G/K$ 为 connected reductive group，中心为 $Z_G$。

**本章约定.** 本章写
$$
r:{}^LG\to\operatorname{GL}(V)
$$
时，含义是：固定 $G$ 的对偶群 $\widehat G$、Galois 作用和一个可在每个位置 $v$ 局部化为
$$
r_v:{}^LG_v\to\operatorname{GL}(V)
$$
的 L 群表示数据，并额外要求该数据在几乎所有位置非分歧。这里不假设数域情形已经存在一个完整的全局 Langlands 群。所有局部因子均通过局部 L 群 ${}^LG_v$ 定义；若只给任意局部表示族而没有“几乎处处非分歧”条件，命题 13.14 和 Euler 乘积都不成立。

**定义 13.1.** $G$ 的自守商是拓扑商
$$
[G]=G(K)\backslash G(\mathbb A_K).
$$
若固定中心特征
$$
\omega:Z_G(K)\backslash Z_G(\mathbb A_K)\to\mathbb C^\times,
$$
则考虑函数 $f:G(\mathbb A_K)\to\mathbb C$ 满足
$$
f(\gamma zg)=\omega(z)f(g),
\qquad
\gamma\in G(K),\ z\in Z_G(\mathbb A_K),\ g\in G(\mathbb A_K).
$$

**注 13.2.** 若 $Z_G(K)\backslash Z_G(\mathbb A_K)$ 非紧，通常还要对中心方向施加增长条件，或固定 unitary central character 后在 $L^2$ 空间中工作。不同群的中心会影响离散谱和残余谱的表述。

**定义 13.3.** 右正则作用定义为
$$
(R(h)f)(g)=f(gh),\qquad h,g\in G(\mathbb A_K).
$$
若函数空间 $V$ 在所有 $R(h)$ 下稳定，则 $V$ 成为 $G(\mathbb A_K)$ 的表示。

**定义 13.4.** 一个光滑自守形式是函数 $f:G(K)\backslash G(\mathbb A_K)\to\mathbb C$，满足：

1. $f$ 在有限 adele 方向右平移下 locally constant，且存在开紧子群 $K_f\subset G(\mathbb A_{K,f})$ 使 $f(gk)=f(g)$；
2. 在 Archimedean 位置，$f$ 为 $C^\infty$，且在 maximal compact subgroup 和 enveloping algebra 作用下满足 $K_\infty$-finite 与 $Z(\mathfrak g_\infty)$-finite 条件；
3. $f$ 满足 moderate growth 条件：相对于某个 adelic height function $\|g\|$，存在常数 $C,N$ 使得 $|f(g)|\le C\|g\|^N$；
4. 若固定中心特征，则满足定义 13.1 的中心变换律。

自守形式空间记为
$$
\mathcal A(G,\omega).
$$

**注 13.5.** 对函数域没有 Archimedean 位置，因此第 2 项应删除并以有限处 smoothness 替代。为了统一叙述，本章默认数域和函数域的差异在涉及无穷处时单独说明。

## 13.2 尖点条件与尖点自守形式

设 $P\subset G$ 为 proper parabolic subgroup，Levi 分解为
$$
P=MN,
$$
其中 $N$ 是 unipotent radical。

**定义 13.6.** 自守形式 $f\in\mathcal A(G,\omega)$ 沿 $P$ 的常数项是函数
$$
f_P(g)=\int_{N(K)\backslash N(\mathbb A_K)} f(ng)\,dn.
$$
这里 $N(K)\backslash N(\mathbb A_K)$ 对 unipotent $N$ 为紧商；本书把 quotient Haar measure 归一化为体积 $1$。常数项为零不依赖该测度的非零标量倍，但与 Eisenstein series 的精确常数项公式比较时必须使用同一测度。

**定义 13.7.** 自守形式 $f$ 称为尖点的（cuspidal），若对所有 proper parabolic subgroups $P\subsetneq G$，
$$
f_P(g)=0
$$
对所有 $g\in G(\mathbb A_K)$ 成立。尖点自守形式空间记为
$$
\mathcal A_0(G,\omega)\subset\mathcal A(G,\omega).
$$

**命题 13.8.** 若 $G$ 没有定义在 $K$ 上的 proper parabolic subgroup，则所有自守形式都是尖点形式。

**证明.** 定义 13.7 要求对所有定义在 $K$ 上的 proper parabolic subgroup 检验常数项。假设说明该索引集合为空，所以该全称命题为真。$\square$

**外部输入定理 13.8.1（各向异性与紧商判准）.** 对整体域上的 connected reductive group，在除去适当 split central direction 后，“没有 proper $K$-parabolic”与“自守商 modulo center 紧”由 reduction theory 联系。该等价不是命题 13.8 的形式逻辑证明的一部分；后文需要从紧商推出无抛物子群时，必须引用本外部输入。

**注 13.9.** 对 $G=\operatorname{GL}_n$，proper parabolic subgroups 存在，尖点条件是强约束。对 $G=\operatorname{GL}_2$，它退化为第七章沿上三角 Borel 的 unipotent radical 积分为零。

**注 13.9.1.** 完整 $L^2$ 自守谱还包含 Eisenstein series 产生的连续谱与残余谱。附录 L 给出常数项公式、intertwining operators 和 residual spectrum 的接口；本章后续主要聚焦 cuspidal representations。

## 13.3 自守表示与张量积分解

**定义 13.10.** 固定 unitary central character $\omega$。一个 cuspidal automorphic representation of $G(\mathbb A_K)$ 是右正则酉表示在
$L_0^2(G(K)Z_G(\mathbb A_K)\backslash G(\mathbb A_K),\omega)$ 中出现的不可约闭子表示；其 smooth、$K_\infty$-finite vectors 给出正文使用的代数表示。有限位置分量要求 smooth admissible，无穷位置分量按 admissible
$(\mathfrak g_v,K_v)$-module/Fréchet globalization 理解。若 $\pi$ 是这样的表示，称 $\omega_\pi=\omega$ 为其中心特征。

**外部输入定理 13.11（自守表示的 restricted tensor product 分解）.** 设 $\pi$ 为 cuspidal automorphic representation of $G(\mathbb A_K)$。则存在局部不可约可容许表示 $\pi_v$，使
$$
\pi\cong\bigotimes_{v\in V_K}'\pi_v.
$$
对几乎所有非 Archimedean 位置 $v$，$G$ 在 $K_v$ 上 unramified，存在 hyperspecial maximal compact subgroup $K_v\subset G(K_v)$，并且
$$
\pi_v^{K_v}\ne0.
$$

**注 13.12.** Restricted tensor product 的参考向量是几乎所有非分歧位置的 spherical vector。若 $G$ 在某些位置 ramified，或 $\pi_v$ ramified，则这些位置必须放入有限集合 $S$。

**定义 13.13.** 一个有限位置集合 $S$ 称为适合 $(G,\pi,r)$，若满足：

1. $S$ 包含所有 Archimedean 位置；
2. 对 $v\notin S$，$G$ 在 $K_v$ 上 unramified；
3. 对 $v\notin S$，$\pi_v$ spherical；
4. 对 $v\notin S$，局部 L 群表示 $r_v:{}^LG_v\to\operatorname{GL}(V)$ 为非分歧数据。

**命题 13.14.** 对满足本章约定的 $(G,\pi,r)$，适合的有限集合 $S$ 存在。

**证明.** 由 $G/K$ 为有限型 reductive group，除有限多个位置外可选 unramified integral model。由定理 13.11，除有限多个位置外 $\pi_v$ spherical。本章约定明确要求 $r_v$ 几乎处处非分歧。把这三个有限例外集与所有 Archimedean 位置合并，即得 $S$。注意“pinned root datum 的作用通过有限商”本身不能约束任意人为给定的局部表示族；这里必须使用本章对 $r$ 的有限分歧假设。$\square$

## 13.4 非分歧局部因子

设 $v$ 为非 Archimedean 位置，$v\notin S$。令 $q_v$ 为剩余域基数。由第十一、十二章，spherical 表示 $\pi_v$ 给出 Satake parameter
$$
s_v(\pi)\rtimes\operatorname{Fr}_v\in{}^LG_v
$$
的半单共轭类。

**定义 13.15.** 设
$$
r_v:{}^LG_v\to\operatorname{GL}(V)
$$
为有限维局部 L 群表示。非分歧局部 L 因子定义为
$$
L(s,\pi_v,r)=
\det\left(1-q_v^{-s}r_v(s_v(\pi)\rtimes\operatorname{Fr}_v)\mid V\right)^{-1}.
$$
若 $r_v$ 在非分歧位置只看 $\widehat G$ 的半单类，则同一公式写为
$$
L(s,\pi_v,r)=
\det\left(1-q_v^{-s}r_v(s_v(\pi))\mid V\right)^{-1}.
$$

**注 13.16.** 第二个公式适用于 split 情形或 $r$ 对 Weil 分量的作用已并入 Satake 元的情形。非 split unramified 群中，Satake parameter 自然位于 $\widehat G\rtimes\operatorname{Fr}_v$ 的连通分量，不能随意丢弃 $\operatorname{Fr}_v$。

**命题 13.17.** 对 $G=\operatorname{GL}_n$ 且 $r=\operatorname{Std}$，若 $\pi_v$ 的 Satake 参数为
$$
\operatorname{diag}(\alpha_{1,v},\ldots,\alpha_{n,v}),
$$
则
$$
L(s,\pi_v,\operatorname{Std})
=
\prod_{i=1}^n(1-\alpha_{i,v}q_v^{-s})^{-1}.
$$

**证明.** 标准表示把 diagonal semisimple element 映为同一矩阵在 $\mathbb C^n$ 上的线性作用。其 characteristic polynomial 为
$$
\prod_{i=1}^n(1-\alpha_{i,v}X).
$$
取 $X=q_v^{-s}$ 并代入定义 13.15。$\square$

**例 13.18.** 当 $G=\mathbb G_m$ 且 $\pi=\chi$ 为 Hecke 特征时，$\widehat G=\mathbb C^\times$，标准表示为恒等表示。若 $\chi_v$ 非分歧，则
$$
L(s,\chi_v)=(1-\chi_v(\varpi_v)q_v^{-s})^{-1},
$$
这与第二、三章的局部因子一致。

## 13.5 部分 L 函数与 Euler 乘积

**定义 13.19.** 设 $S$ 适合 $(G,\pi,r)$。部分 L 函数定义为 Euler 乘积
$$
L^S(s,\pi,r)=\prod_{v\notin S}L(s,\pi_v,r).
$$

**命题 13.20.** 在 $v\notin S$ 的非分歧位置，$L(s,\pi_v,r)$ 只依赖于 $\pi_v$ 的 spherical Hecke eigencharacter。

**证明.** 由 Satake 同构，spherical Hecke eigencharacter 等价于 L 群中的 Satake parameter 半单共轭类。定义 13.15 只使用该半单共轭类在 $r_v$ 下的 characteristic polynomial。Characteristic polynomial 对共轭不变，因此局部因子只依赖 Hecke eigencharacter。$\square$

**外部输入定理 13.21（标准 Euler 乘积的初始收敛）.** 采用 unitary normalization 时，下列 Euler 乘积在
$\operatorname{Re}(s)>1$ 绝对收敛：酉 Hecke 特征的 L 函数、unitary cuspidal
$\operatorname{GL}_n$ 表示的标准 L 函数，以及 unitary cuspidal
$\operatorname{GL}_n\times\operatorname{GL}_m$ Rankin-Selberg L 函数。若表示或特征再张量实次 norm character，半平面按该实次作相应平移。对一般 reductive group 与任意 $r$，本书不从形式 Satake 参数推出任何收敛半平面；必须逐项登记外部估计或保留为形式 Euler 乘积。

**注 13.22.** 不能把形式 Euler 乘积自动视为全平面解析函数。Euler 乘积首先只在某个可能存在的收敛半平面定义；解析延拓和函数方程是额外深性质。

## 13.6 Ramified 因子与完全 L 函数

为了得到完整函数方程，需要补上 $S$ 中的位置。

**定义 13.23.** 若局部 LLC 和局部因子理论已经给出 $\pi_v$ 的参数 $\varphi_v$，则定义
$$
L(s,\pi_v,r)=L(s,r_v\circ\varphi_v)
$$
其中右侧为线性 Weil-Deligne 或 Archimedean Weil 参数的局部 L 因子。

**定义 13.24.** 在所有局部因子已经定义且 Euler 乘积在某个右半平面绝对收敛的情形，完全 L 函数首先在该半平面定义为
$$
L(s,\pi,r)=\prod_v L(s,\pi_v,r).
$$
若只补上有限集合 $S$ 中的局部因子，则
$$
L(s,\pi,r)=L^S(s,\pi,r)\prod_{v\in S}L(s,\pi_v,r).
$$

**注 13.25.** 对一般 $G$ 和一般 $r$，ramified 局部因子的定义可来自不同理论：局部 LLC、Langlands-Shahidi 方法、Rankin-Selberg 积分、doubling method 或其他局部 zeta integrals。不同构造必须满足相同的 $\gamma$ 因子和函数方程相容性，才可视为同一 Langlands L 因子。

**定义 13.26.** 给定非平凡整体加法特征
$$
\psi:K\backslash\mathbb A_K\to\mathbb C^\times,
$$
局部 epsilon 因子若已定义，则全局 epsilon 因子定义为乘积
$$
\varepsilon(s,\pi,r)=\prod_v\varepsilon(s,\pi_v,r,\psi_v).
$$
若 $v$ 同时满足 $\pi_v,r_v,\psi_v$ 非分歧、$\mathfrak c(\psi_v)=\mathcal O_v$ 且采用自对偶加法测度，则标准归一化给出局部 epsilon 因子 $1$。这些条件在几乎所有 $v$ 成立，故乘积是有限乘积；缺少加法特征 conductor 或测度条件时不能直接作此断言。

## 13.7 解析延拓与函数方程

**猜想 13.27（Langlands L 函数的解析性质）.** 设 $\pi$ 为 cuspidal automorphic representation of $G(\mathbb A_K)$，设
$$
r:{}^LG\to\operatorname{GL}(V)
$$
为有限维 L 群表示。补入定义 13.24 的所有 ramified 与 Archimedean 局部因子后的 L 函数 $L(s,\pi,r)$ 应具有 meromorphic continuation 到整个复平面，并满足函数方程
$$
L(s,\pi,r)
=
\varepsilon(s,\pi,r)
L(1-s,\pi,r^\vee),
$$
其中 $r^\vee$ 为对偶表示。若已知
$r\circ\varphi_{\pi^\vee}\cong(r\circ\varphi_\pi)^\vee$，可把右端等价写成
$L(1-s,\pi^\vee,r)$。一般不能同时写 $\pi^\vee$ 与 $r^\vee$，因为那会把线性参数对偶两次。若 $r$ 和 $\pi$ 满足额外非平凡性条件，预期除可由中心或平凡表示解释的极点外，$L(s,\pi,r)$ 是 entire。

**注 13.28.** 函数方程中的 $s\mapsto1-s$ 依赖归一化。本书采用自守归一化；若把经典模形式的权吸收到无穷处 gamma 因子或 Tate twist 中，函数方程中心会相应平移。

**外部输入定理 13.29（Godement-Jacquet）.** 设 $K$ 为整体域，$n\ge2$，$\pi$ 为 unitary cuspidal automorphic representation of
$\operatorname{GL}_n(\mathbb A_K)$。其完成标准 L 函数 $\Lambda(s,\pi)$ 整，并满足
$$
\Lambda(s,\pi)=\varepsilon(s,\pi)\Lambda(1-s,\pi^\vee).
$$
对 $n=1$ 应改用定理 2.13，纯 norm character 允许极点。Godement-Jacquet zeta integrals 还给出局部标准因子和全局积分的 Euler 分解；本书把矩阵空间 Poisson summation 及 Archimedean 分析保留为外部输入。

**外部输入定理 13.30（Rankin-Selberg 与 Langlands-Shahidi，限定接口）.** 对 unitary cuspidal
$\pi$ of $\operatorname{GL}_n(\mathbb A_K)$ 和 $\pi'$ of
$\operatorname{GL}_m(\mathbb A_K)$，完成 Rankin-Selberg L 函数有亚纯延拓和对偶函数方程；其可能极点按 $\pi'$ 与 $\pi^\vee$ 的 twist-equivalence 条件控制。对 Langlands-Shahidi 方法，只有当
$r$ 是某个已固定 quasi-split group 的 maximal parabolic Levi 对 $\operatorname{Lie}({}^LN)$ 的不可约分量、且局部 genericity 与全局 cuspidality 假设满足该版本定理时，才能断言相应 L 函数的亚纯延拓和函数方程。本条不覆盖任意 $G$ 与任意 $r$。

**注 13.31.** 定理 13.30 不是任意 reductive group 与任意 L 群表示的完整定理。一般解析性质通常需要 functoriality，或需要单独构造相应 integral representation。附录 I 展开 Godement-Jacquet、Rankin-Selberg 和 converse theorem 的积分接口；附录 M 展开 Langlands-Shahidi local coefficient、局部 $\gamma$ 因子和由 Eisenstein 函数方程产生 L 因子的接口。

## 13.8 标准例子

### 13.8.1 `GL(1)`：Hecke L 函数

设 $G=\mathbb G_m$，自守表示即 Hecke 特征
$$
\chi:K^\times\backslash\mathbb A_K^\times\to\mathbb C^\times.
$$
取 $r=\operatorname{id}:\mathbb C^\times\to\mathbb C^\times$。则
$$
L(s,\chi,r)=L(s,\chi)
$$
就是第二章的 Hecke L 函数。解析延拓和函数方程由 Tate thesis 给出。

### 13.8.2 `GL(2)`：模形式的标准 L 函数

设 $K=\mathbb Q$，$\pi_f$ 来自权 $k$、级 $N$ 的归一化 cuspidal newform $f$，并采用 unitary normalization。在 $p\nmid N$ 处，若 $(\alpha_p,\beta_p)$ 是 classical Hecke roots，则 Satake roots 为
$(\alpha_pp^{-(k-1)/2},\beta_pp^{-(k-1)/2})$，于是
$$
L_p(s,\pi_f,\operatorname{Std})
=
\left((1-\alpha_pp^{-(s+(k-1)/2)})
(1-\beta_pp^{-(s+(k-1)/2)})\right)^{-1}.
$$
因此精确关系是
$L(s,\pi_f,\operatorname{Std})=L(f,s+(k-1)/2)$；不是在同一变量中“忽略 convention 后相同”。

### 13.8.3 Adjoint L 函数

设 $G=\operatorname{GL}_n$。对偶群为 $\operatorname{GL}_n(\mathbb C)$。Adjoint representation
$$
\operatorname{Ad}:\operatorname{GL}_n(\mathbb C)\to\operatorname{GL}(\mathfrak{sl}_n(\mathbb C))
$$
给出 adjoint L 函数
$$
L(s,\pi,\operatorname{Ad}).
$$
该 L 函数控制若干算术和谱论问题，例如 Petersson norm、deformation theory 中的 adjoint Selmer groups，以及 trace formula 中的归一化因子。

**注 13.32.** Adjoint L 函数的解析性质可由 Rankin-Selberg L 函数与中心 character 的因子关系在 `GL(n)` 情形中推出；一般群的 adjoint L 函数属于更广的 Langlands-Shahidi 或 functoriality 框架。

## 13.9 强重数一与全局表示的局部确定性

**外部输入定理 13.33（强重数一 for `GL(n)`）.** 设 $\pi$ 和 $\pi'$ 为 cuspidal automorphic representations of $\operatorname{GL}_n(\mathbb A_K)$。若对几乎所有位置 $v$ 有
$$
\pi_v\cong\pi'_v,
$$
则
$$
\pi\cong\pi'.
$$

**推论 13.34.** 对 `GL(n)`，几乎所有非分歧 Satake 参数确定 cuspidal automorphic representation。

**证明.** 若两个表示在几乎所有位置有相同 Satake 参数，则对应 spherical 表示同构。由强重数一定理 13.33，两个全局表示同构。$\square$

**注 13.35.** 对一般 reductive groups，强重数一可能失败或需要以 packet、稳定迹和端oscopic 数据修正。后续 trace formula 与 endoscopy 章节会解释这种失败如何与 functoriality 相连。

## 13.10 局部因子如何组成全局 L 函数

全局自守表示 $\pi$ 是 $G(\mathbb A_K)$ 在自守形式空间中出现的不可约表示，并分解为局部分量
$$
\pi=\otimes_v'\pi_v.
$$
在几乎所有非分歧位置，$\pi_v$ 由 Satake parameter 描述。给定 L 群表示 $r:{}^LG\to\operatorname{GL}(V)$，这些 Satake 参数定义局部因子和部分 Euler 乘积
$$
L^S(s,\pi,r)=\prod_{v\notin S}L(s,\pi_v,r).
$$
补上 ramified 和 Archimedean 因子后得到完全 L 函数。其解析延拓和函数方程在 `GL(1)`、`GL(n)` 标准 L 函数、Rankin-Selberg L 函数和若干 Langlands-Shahidi 情形中是定理；对一般还原群和任意 $r$，这是 Langlands 纲领的核心猜想之一。

## 练习

**练习 13.1.** 对 $G=\operatorname{GL}_2$，把定义 13.6 的常数项写成第七章上三角 unipotent radical 上的积分。

**练习 13.2.** 证明若 $G/Z_G$ anisotropic，则尖点条件自动满足。

**练习 13.3.** 对 $G=\operatorname{GL}_n$，由 diagonal Satake parameter 推导标准局部 L 因子公式。

**练习 13.4.** 设 $S\subset S'$ 为两个适合 $(G,\pi,r)$ 的有限集合。写出 $L^S(s,\pi,r)$ 与 $L^{S'}(s,\pi,r)$ 的关系。

**练习 13.5.** 解释为什么非分歧 Euler 因子只依赖 spherical Hecke eigencharacter。

**练习 13.6.** 对 $G=\mathbb G_m$，验证本章定义的全局 L 函数退化为 Tate thesis 中的 Hecke L 函数。

**练习 13.7.** 说明为什么一般 reductive group 的任意 L 群表示 $r$ 的解析延拓不能仅由 Euler 乘积形式推出。
