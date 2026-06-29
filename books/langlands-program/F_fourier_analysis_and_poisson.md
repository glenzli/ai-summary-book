# 附录 F：Fourier 分析、Pontryagin 对偶和 Poisson 求和

本附录固定第一、二章所用的 Fourier 分析口径。完整的局部紧 Abel 群 Fourier 分析是一门独立理论；本附录证明本书反复使用的形式性质，并把深层分析定理明确标为外部输入。

## F.1 局部紧 Abel 群和对偶群

**定义 F.1.** 设 $G$ 为局部紧 Abel 群。其 Pontryagin 对偶定义为
$$
\widehat G=\operatorname{Hom}_{\operatorname{cont}}(G,S^1),
$$
配备 compact-open topology。元素 $\chi\in\widehat G$ 称为 $G$ 的连续 unitary character。

**定义 F.2.** 若 $H\subset G$ 为闭子群，定义 annihilator
$$
H^\perp=\{\chi\in\widehat G:\chi(h)=1\text{ for all }h\in H\}.
$$

**命题 F.3.** $H^\perp$ 是 $\widehat G$ 的闭子群。

**证明.** 对每个 $h\in H$，映射
$$
\operatorname{ev}_h:\widehat G\to S^1,\qquad \chi\mapsto\chi(h)
$$
连续。于是
$$
H^\perp=\bigcap_{h\in H}\operatorname{ev}_h^{-1}(\{1\})
$$
为闭集，并且逐点乘法和取逆保持在 $H$ 上取值 $1$，故为闭子群。$\square$

**外部输入定理 F.4（Pontryagin duality）.** 对每个局部紧 Abel 群 $G$，自然映射
$$
G\to\widehat{\widehat G},\qquad
g\mapsto(\chi\mapsto\chi(g))
$$
是拓扑群同构。

**外部输入定理 F.5（闭子群对偶正合列）.** 若 $H\subset G$ 为闭子群，则有自然拓扑同构
$$
\widehat{G/H}\simeq H^\perp,
$$
以及
$$
\widehat H\simeq \widehat G/H^\perp
$$
在适当的商拓扑意义下成立。

**命题 F.6.** 若 $H\subset G$ 离散且 $G/H$ 紧，则 $H^\perp\subset\widehat G$ 离散且 $\widehat G/H^\perp$ 紧。

**证明草图.** 由 F.5，
$$
H^\perp\simeq\widehat{G/H}.
$$
紧 Abel 群的对偶是离散群，故 $H^\perp$ 离散。又
$$
\widehat G/H^\perp\simeq\widehat H.
$$
离散 Abel 群的对偶是紧群，故商紧。$\square$

## F.2 Fourier 变换和自对偶测度

**定义 F.7.** 设 $G$ 为局部紧 Abel 群，$dx$ 为 Haar 测度。对 $f\in L^1(G,dx)$，其 Fourier 变换定义为
$$
\widehat f(\chi)=\int_G f(x)\chi(x)\,dx,\qquad \chi\in\widehat G.
$$
若使用加法特征 $\psi:G\to S^1$ 把 $G$ 与 $\widehat G$ 识别，则也写
$$
\widehat f(y)=\int_G f(x)\psi(xy)\,dx.
$$

**外部输入定理 F.8（Fourier inversion and Plancherel）.** 对局部紧 Abel 群 $G$，存在与 $dx$ 对偶的 Haar 测度 $d\chi$ on $\widehat G$，使得合适函数 $f$ 满足 Fourier inversion：
$$
f(x)=\int_{\widehat G}\widehat f(\chi)\chi(x)^{-1}\,d\chi.
$$
并且 Fourier 变换延拓为 $L^2(G)$ 与 $L^2(\widehat G)$ 间的 unitary isomorphism。

**定义 F.9.** 若 $G$ 通过非平凡加法特征 $\psi$ 与 $\widehat G$ 同构，则称 Haar 测度 $dx$ 为 $\psi$-self-dual measure，若 Fourier inversion 在同一个测度归一化下成立。

**命题 F.10（非 Archimedean 基本计算）.** 设 $F$ 为非 Archimedean 局部域，$\psi:F\to S^1$ 为非平凡加法特征。令
$$
\mathfrak d_\psi^{-1}=\{x\in F:\psi(x\mathcal O_F)=1\}.
$$
若 $dx$ 满足 $\operatorname{vol}(\mathcal O_F,dx)=\operatorname{vol}(\mathfrak d_\psi^{-1},dx)^{-1}$，则
$$
\widehat{\mathbf 1_{\mathcal O_F}}(y)=\operatorname{vol}(\mathcal O_F)\mathbf 1_{\mathfrak d_\psi^{-1}}(y).
$$

**证明.** 有
$$
\widehat{\mathbf 1_{\mathcal O_F}}(y)=\int_{\mathcal O_F}\psi(xy)\,dx.
$$
若 $y\in\mathfrak d_\psi^{-1}$，则 $\psi(xy)=1$ 对所有 $x\in\mathcal O_F$ 成立，积分为 $\operatorname{vol}(\mathcal O_F)$。若 $y\notin\mathfrak d_\psi^{-1}$，则 character $x\mapsto\psi(xy)$ 在紧群 $\mathcal O_F$ 上非平凡。取 $a\in\mathcal O_F$ 使 $\psi(ay)\ne1$，平移 $x\mapsto x+a$ 给出
$$
I=\int_{\mathcal O_F}\psi(xy)\,dx
=\int_{\mathcal O_F}\psi((x+a)y)\,dx
=\psi(ay)I.
$$
故 $I=0$。$\square$

**推论 F.11.** 若 $\psi$ 的 conductor 为 $\mathcal O_F$，即
$$
\{x\in F:\psi(x\mathcal O_F)=1\}=\mathcal O_F,
$$
并取 $\operatorname{vol}(\mathcal O_F)=1$，则
$$
\widehat{\mathbf 1_{\mathcal O_F}}=\mathbf 1_{\mathcal O_F}.
$$

**证明.** 命题 F.10 中 $\mathfrak d_\psi^{-1}=\mathcal O_F$ 且 $\operatorname{vol}(\mathcal O_F)=1$。$\square$

## F.3 Schwartz-Bruhat 空间

**定义 F.12.** 对局部域 $F$，Schwartz-Bruhat 空间 $\mathcal S(F)$ 定义如下：

- 若 $F$ 为 Archimedean，则 $\mathcal S(F)$ 为通常的 Schwartz rapidly decreasing smooth functions。
- 若 $F$ 为非 Archimedean，则 $\mathcal S(F)=C_c^\infty(F)$，即紧支撑局部常值函数。

**命题 F.13.** Fourier 变换保持 $\mathcal S(F)$。

**证明草图.** Archimedean 情形是经典 Schwartz 空间 Fourier 理论。非 Archimedean 情形中，任意 $f\in C_c^\infty(F)$ 可写成有限个紧开陪集特征函数的线性组合；命题 F.10 和缩放换元说明这些特征函数的 Fourier 变换仍为紧支撑局部常值函数。$\square$

**定义 F.14.** 对整体域 $K$，定义
$$
\mathcal S(\mathbb A_K)=\bigotimes_v'\mathcal S(K_v)
$$
相对于标准向量 $\mathbf 1_{\mathcal O_v}$ 取 restricted tensor product，非 Archimedean 好位置使用 conductor 为 $\mathcal O_v$ 的局部加法特征和 $\operatorname{vol}(\mathcal O_v)=1$ 的 self-dual 测度。

**命题 F.15.** 若 $\Phi=\otimes_v\Phi_v\in\mathcal S(\mathbb A_K)$ 为纯张量，则
$$
\widehat\Phi=\otimes_v\widehat{\Phi_v}.
$$

**证明.** 取有限集合 $S$ 包含所有 $\Phi_v\ne\mathbf 1_{\mathcal O_v}$、特征或测度非标准的位置。对 $v\notin S$，推论 F.11 给出 $\widehat{\mathbf 1_{\mathcal O_v}}=\mathbf 1_{\mathcal O_v}$。整体 Fourier 积分在柱状支撑上化为有限乘积积分，故由 Fubini 得到张量分解。$\square$

## F.4 Adeles 的自对偶性

**外部输入定理 F.16（adeles 的自对偶性）.** 设 $K$ 为整体域。存在非平凡连续加法特征
$$
\psi:\mathbb A_K/K\to S^1
$$
使映射
$$
\mathbb A_K\to\widehat{\mathbb A_K},\qquad
y\mapsto(x\mapsto\psi(xy))
$$
为拓扑群同构。该同构下，$K\subset\mathbb A_K$ 的 annihilator 正是 $K$。

**命题 F.17.** 在 F.16 的同构下，
$$
\widehat{\mathbb A_K/K}\simeq K
$$
作为离散群。

**证明.** 由 F.5，
$$
\widehat{\mathbb A_K/K}\simeq K^\perp.
$$
F.16 说明 $K^\perp=K$，其中右侧通过对角嵌入视为 $\mathbb A_K$ 的子群。由于 $\mathbb A_K/K$ 紧，F.6 也说明其对偶离散。$\square$

**注 F.18.** 第一章的自对偶性定理 1.21 即 F.16 的正文版本。第二章 Tate thesis 使用该同构把 Poisson summation 写成 $\mathbb A_K$ 上 Schwartz-Bruhat 函数的公式。

## F.5 Poisson 求和

**外部输入定理 F.19（LCA Poisson summation）.** 设 $G$ 为局部紧 Abel 群，$H\subset G$ 为离散闭子群且 $G/H$ 紧。对足够好的函数 $f$，有
$$
\sum_{h\in H}f(h)=\operatorname{vol}(G/H)^{-1}\sum_{\chi\in H^\perp}\widehat f(\chi),
$$
其中测度和对偶测度按 F.8 归一化。

**推论 F.20（adele Poisson summation）.** 对 $\Phi\in\mathcal S(\mathbb A_K)$，若 measure normalization 取使 $\operatorname{vol}(\mathbb A_K/K)=1$，则
$$
\sum_{\gamma\in K}\Phi(\gamma)=\sum_{\gamma\in K}\widehat\Phi(\gamma).
$$

**证明.** 取 $G=\mathbb A_K$、$H=K$。由第一章外部输入定理 1.15，$K$ 离散且 $\mathbb A_K/K$ 紧。由 F.16，$H^\perp=K$。代入 F.19，并使用体积归一化 $\operatorname{vol}(\mathbb A_K/K)=1$。$\square$

## F.6 Tate Thesis 中的用法

**命题 F.21.** 第二章整体 zeta 积分
$$
Z(\Phi,\chi,s)=\int_{\mathbb A_K^\times}\Phi(x)\chi(x)|x|_{\mathbb A}^s\,d^\times x
$$
在纯张量和绝对收敛半平面中分解为局部 zeta 积分乘积。

**证明.** 这是命题 2.11 的分析基础。取有限集合 $S$ 包含所有非标准局部数据。对 $v\notin S$，$\Phi_v=\mathbf 1_{\mathcal O_v}$ 且 $\chi_v$ 非分歧，局部积分给出标准 Euler 因子。restricted product 测度和附录 B 的命题 B.15 把整体积分化为有限个非标准积分与标准局部积分的乘积。绝对收敛保证 Fubini 交换合法。$\square$

**外部输入定理 F.22（Tate 整体函数方程的 Fourier 分析核心）.** 对 $\Phi\in\mathcal S(\mathbb A_K)$ 和 Hecke 特征 $\chi$，Poisson summation 应用于函数
$$
x\mapsto \Phi(tx)
$$
并结合局部函数方程，给出完成 zeta 积分的 meromorphic continuation 和函数方程。精确极点取决于 $\chi$ 是否平凡以及 $K$ 的体积归一化。

**注 F.23.** 本书第二章把 Tate thesis 作为外部输入定理 2.13。本附录说明其 Fourier 分析骨架，但不替代 Tate thesis 的完整证明；完整证明还需要局部 zeta 积分的有理性、Archimedean gamma 因子分析和整体积分截断。

## F.7 本附录小结

本附录给出四个接口：

1. 局部紧 Abel 群的对偶和 Fourier inversion。
2. 非 Archimedean 局部域上 $\mathbf 1_{\mathcal O_F}$ 的 Fourier 变换计算。
3. $\mathcal S(\mathbb A_K)$ 的 restricted tensor product 与 Fourier 变换相容。
4. Adele Poisson summation 作为 Tate thesis 整体函数方程的入口。

## 练习

**练习 F.1.** 证明若 $H\subset G$ 为闭子群，则 $H^\perp$ 的 annihilator 在 $\widehat{\widehat G}\simeq G$ 下等于 $H$。

**练习 F.2.** 设 $F$ 为非 Archimedean 局部域，$a\in F^\times$。计算 $\widehat{\mathbf 1_{a\mathcal O_F}}$。

**练习 F.3.** 对 $K=\mathbb Q$，取标准加法特征，说明 $\prod_p\mathbf 1_{\mathbb Z_p}$ 在有限 adele Fourier 变换下保持不变。

**练习 F.4.** 从 F.20 推出 $\mathbb Q$ 上 classical Poisson summation 的形式
$$
\sum_{n\in\mathbb Z}f(n)=\sum_{n\in\mathbb Z}\widehat f(n)
$$
在合适的 Archimedean Schwartz 函数上成立。

**练习 F.5.** 说明 Tate thesis 中平凡特征的极点为什么来自 Poisson summation 中的零点项。
