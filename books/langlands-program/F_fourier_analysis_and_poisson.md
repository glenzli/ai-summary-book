# 附录 F：Fourier 分析、Pontryagin 对偶和 Poisson 求和

本附录固定第一、二章所用的 Fourier 分析口径。完整的局部紧 Abel 群 Fourier 分析是一门独立理论；本附录证明本书反复使用的形式性质，并把深层分析定理明确标为外部输入。

收口归一化回指：本附录所有 Fourier 变换、自对偶测度、Poisson summation 和 Tate theta 公式均按 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 第 3、8 节处理。

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
在 quotient topology 意义下成立。

**命题 F.6.** 若 $H\subset G$ 离散且 $G/H$ 紧，则 $H^\perp\subset\widehat G$ 离散且 $\widehat G/H^\perp$ 紧。

**证明.** 由外部输入 F.5，
$$
H^\perp\simeq\widehat{G/H}.
$$
紧 Abel 群的对偶是离散群，故 $H^\perp$ 离散。又
$$
\widehat G/H^\perp\simeq\widehat H.
$$
离散 Abel 群的对偶是紧群，故商紧。这里“紧群的对偶离散、离散群的
对偶紧”是 Pontryagin 对偶定理包的直接部分；除此之外没有额外输入。
$\square$

## F.2 Fourier 变换和自对偶测度

**定义 F.7.** 设 $G$ 为局部紧 Abel 群，$dx$ 为 Haar 测度。对 $f\in L^1(G,dx)$，其 Fourier 变换定义为
$$
\widehat f(\chi)=\int_G f(x)\chi(x)\,dx,\qquad \chi\in\widehat G.
$$
若使用加法特征 $\psi:G\to S^1$ 把 $G$ 与 $\widehat G$ 识别，则也写
$$
\widehat f(y)=\int_G f(x)\psi(xy)\,dx.
$$

**外部输入定理 F.8（Fourier inversion and Plancherel）.** 对局部紧 Abel 群 $G$，存在与 $dx$ 对偶的 Haar 测度 $d\chi$ on $\widehat G$，使得 $f,\widehat f\in L^1$ 的函数满足 Fourier inversion：
$$
f(x)=\int_{\widehat G}\widehat f(\chi)\chi(x)^{-1}\,d\chi.
$$
并且 Fourier 变换延拓为 $L^2(G)$ 与 $L^2(\widehat G)$ 间的 unitary isomorphism。

**命题 F.8.1（对偶测度的缩放）.** 若 $dx'=c\,dx$，其中 $c>0$，则与 $dx'$ 对偶的 Haar 测度为
$$
d\chi'=c^{-1}d\chi.
$$
特别地，在给定同构 $G\simeq\widehat G$ 后，自对偶测度至多唯一。

**证明.** 用 $dx'$ 定义的 Fourier 变换满足
$$
\widehat f_{dx'}=c\,\widehat f_{dx}.
$$
Fourier 反演要求
$$
f(x)=\int_{\widehat G}\widehat f_{dx'}(\chi)\chi(x)^{-1}\,d\chi'
=\int_{\widehat G}c\,\widehat f_{dx}(\chi)\chi(x)^{-1}\,d\chi'.
$$
与 $dx$ 的反演公式比较，必须有 $d\chi'=c^{-1}d\chi$。若 $G$ 已与 $\widehat G$ 识别且 $dx$ 与 $dx'$ 都自对偶，则同时要求 $dx'=c\,dx$ 和 $dx'=c^{-1}dx$，故 $c=1$。$\square$

**命题 F.8.2（有限 Abel 群 Fourier 反演）.** 设 $A$ 为有限 Abel 群。对函数 $f:A\to\mathbb C$，定义
$$
\widehat f(\chi)=\sum_{a\in A}f(a)\chi(a),\qquad \chi\in\widehat A.
$$
则
$$
f(a)=\frac{1}{|A|}\sum_{\chi\in\widehat A}\widehat f(\chi)\chi(a)^{-1}.
$$

**证明.** 代入定义得
$$
\frac{1}{|A|}\sum_{\chi\in\widehat A}\widehat f(\chi)\chi(a)^{-1}
=
\sum_{b\in A}f(b)
\left(
\frac{1}{|A|}\sum_{\chi\in\widehat A}\chi(ba^{-1})
\right).
$$
字符正交关系给出括号中表达式在 $b=a$ 时为 $1$，在 $b\ne a$ 时为 $0$。因此右端等于 $f(a)$。$\square$

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

**命题 F.11.1（紧开陪集的 Fourier 变换）.** 设 $F$ 为非 Archimedean 局部域，$L\subset F$ 为加法紧开子群，$a\in F$。记
$$
L^\perp=\{y\in F:\psi(xy)=1\text{ for all }x\in L\}.
$$
则
$$
\widehat{\mathbf 1_{a+L}}(y)=\psi(ay)\operatorname{vol}(L)\mathbf 1_{L^\perp}(y).
$$
若 $L=b\mathcal O_F$，则
$$
L^\perp=b^{-1}\mathfrak d_\psi^{-1}.
$$

**证明.** 直接计算
$$
\widehat{\mathbf 1_{a+L}}(y)
=\int_{a+L}\psi(xy)\,dx
=\psi(ay)\int_L\psi(uy)\,du.
$$
当 $y\in L^\perp$ 时，后一个积分为 $\operatorname{vol}(L)$。当 $y\notin L^\perp$ 时，存在 $u_0\in L$ 使 $\psi(u_0y)\ne1$；由平移不变性，
$$
I=\int_L\psi(uy)\,du
=\int_L\psi((u+u_0)y)\,du
=\psi(u_0y)I,
$$
所以 $I=0$。若 $L=b\mathcal O_F$，条件 $\psi(yb\mathcal O_F)=1$ 等价于 $yb\in\mathfrak d_\psi^{-1}$，故 $L^\perp=b^{-1}\mathfrak d_\psi^{-1}$。$\square$

## F.3 Schwartz-Bruhat 空间

**定义 F.12.** 对局部域 $F$，Schwartz-Bruhat 空间 $\mathcal S(F)$ 定义如下：

- 若 $F$ 为 Archimedean，则 $\mathcal S(F)$ 为通常的 Schwartz rapidly decreasing smooth functions。
- 若 $F$ 为非 Archimedean，则 $\mathcal S(F)=C_c^\infty(F)$，即紧支撑局部常值函数。

**命题 F.13.** Fourier 变换保持 $\mathcal S(F)$。

**证明.** Archimedean 情形调用外部输入 F.8 的 Schwartz 版本：分部积分
把多项式乘权转化为导数，故 Fourier 变换保持快速衰减及光滑性。
非 Archimedean 情形中，$f$ 的局部常值性和紧支撑性给出一个开紧子群
$L\subset F$，使 $f$ 在有限多个 $L$-陪集上常值并在其余陪集上为零。
因此 $f$ 是有限个 $\mathbf1_{a+L}$ 的线性组合。命题 F.10 的公式说明
每个此类特征函数的 Fourier 变换支撑在开紧子群 $L^\perp$ 上，并在其
陪集上局部常值。有限线性组合仍属于 $C_c^\infty(F)$。$\square$

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

**命题 F.18.1（$\mathbb A_\mathbb Q/\mathbb Q$ 的基本域）.** 对 $K=\mathbb Q$，对角嵌入下 $\mathbb Q$ 是 $\mathbb A_\mathbb Q$ 的离散子群，并且
$$
\mathbb A_\mathbb Q/\mathbb Q
$$
紧。更具体地，每个 adele 都可模去某个 $q\in\mathbb Q$ 后落入
$$
[0,1]\times\prod_p\mathbb Z_p.
$$

**证明.** 先证离散性。设
$$
U=(-1/2,1/2)\times\prod_p\mathbb Z_p\subset\mathbb A_\mathbb Q.
$$
若 $q\in\mathbb Q\cap U$，则 $q\in\mathbb Z_p$ 对所有素数 $p$ 成立，所以 $q\in\mathbb Z$。又 $q\in(-1/2,1/2)$，故 $q=0$。因此 $\mathbb Q$ 在 $\mathbb A_\mathbb Q$ 中离散。

再证商紧。取 $x=(x_\infty,x_p)_p\in\mathbb A_\mathbb Q$。存在有限集合 $S$ 使 $x_p\in\mathbb Z_p$ 对 $p\notin S$ 成立。对每个 $p\in S$，取 $n_p$ 使 $p^{n_p}x_p\in\mathbb Z_p$。商群 $p^{-n_p}\mathbb Z_p/\mathbb Z_p$ 可由 $\mathbb Z[1/p]/\mathbb Z$ 的元素表示。由中国剩余定理，可取 $q_0\in\mathbb Q$，其分母只含 $S$ 中素数，且
$$
x_p-q_0\in\mathbb Z_p,\qquad p\in S.
$$
对 $p\notin S$，因 $q_0\in\mathbb Z_p$ 且 $x_p\in\mathbb Z_p$，也有 $x_p-q_0\in\mathbb Z_p$。于是 $x-q_0$ 的所有有限分量都在 $\mathbb Z_p$ 中。

最后取 $m\in\mathbb Z$ 使 $x_\infty-q_0-m\in[0,1]$。因为 $m\in\mathbb Z_p$ 对所有 $p$ 成立，$x-q_0-m$ 仍有所有有限分量在 $\mathbb Z_p$ 中。故自然映射
$$
[0,1]\times\prod_p\mathbb Z_p\to\mathbb A_\mathbb Q/\mathbb Q
$$
满射。源空间紧，商是其连续像，因此紧。$\square$

**注 F.18.2.** 一般数域情形的 $\mathbb A_K/K$ 紧性可用 Minkowski 理论和整数环格点基本域证明；函数域情形可用 Riemann-Roch 证明。本书把一般情形保留为第一章外部输入定理 1.15，但命题 F.18.1 给出 $\mathbb Q$ 情形的完整模型。

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

**命题 F.20.1（idele 缩放下的 Poisson 公式）.** 设 $t\in\mathbb A_K^\times$，并令
$$
\Phi_t(x)=\Phi(tx).
$$
则
$$
\widehat{\Phi_t}(y)=|t|_{\mathbb A}^{-1}\widehat\Phi(t^{-1}y),
$$
并且
$$
\sum_{\gamma\in K}\Phi(t\gamma)
=
|t|_{\mathbb A}^{-1}
\sum_{\gamma\in K}\widehat\Phi(t^{-1}\gamma).
$$

**证明.** 在 Fourier 积分中作变量替换 $u=tx$。加法 Haar 测度满足 $dx=|t|_{\mathbb A}^{-1}du$，于是
$$
\widehat{\Phi_t}(y)
=\int_{\mathbb A_K}\Phi(tx)\psi(xy)\,dx
=|t|_{\mathbb A}^{-1}\int_{\mathbb A_K}\Phi(u)\psi(u\,t^{-1}y)\,du
=|t|_{\mathbb A}^{-1}\widehat\Phi(t^{-1}y).
$$
对 $\Phi_t$ 应用 F.20 即得第二个公式。$\square$

**命题 F.20.2（经典 Poisson 求和的 adele 推导）.** 取 $K=\mathbb Q$ 和标准加法特征。若 $f\in\mathcal S(\mathbb R)$，并令
$$
\Phi=f\otimes\prod_p\mathbf 1_{\mathbb Z_p}\in\mathcal S(\mathbb A_\mathbb Q),
$$
则 F.20 给出
$$
\sum_{n\in\mathbb Z}f(n)=\sum_{n\in\mathbb Z}\widehat f(n),
$$
其中实 Fourier 变换采用与标准加法特征相容的归一化。

**证明.** 对 $q\in\mathbb Q$，有限部分
$$
\prod_p\mathbf 1_{\mathbb Z_p}(q)
$$
非零当且仅当 $q\in\mathbb Z$。因此
$$
\sum_{q\in\mathbb Q}\Phi(q)=\sum_{n\in\mathbb Z}f(n).
$$
推论 F.11 和 F.15 给出
$$
\widehat\Phi=\widehat f\otimes\prod_p\mathbf 1_{\mathbb Z_p}.
$$
同理
$$
\sum_{q\in\mathbb Q}\widehat\Phi(q)=\sum_{n\in\mathbb Z}\widehat f(n).
$$
代入 F.20 得结论。$\square$

## F.6 Tate Thesis 中的用法

**命题 F.21.** 第二章整体 zeta 积分
$$
Z(\Phi,\chi,s)=\int_{\mathbb A_K^\times}\Phi(x)\chi(x)|x|_{\mathbb A}^s\,d^\times x
$$
在纯张量和绝对收敛半平面中分解为局部 zeta 积分乘积。

**证明.** 这是命题 2.11 的分析基础。取有限集合 $S$ 包含所有非标准局部数据。对 $v\notin S$，$\Phi_v=\mathbf 1_{\mathcal O_v}$ 且 $\chi_v$ 非分歧，局部积分给出标准 Euler 因子。restricted product 测度和附录 B 的命题 B.15 把整体积分化为有限个非标准积分与标准局部积分的乘积。绝对收敛保证 Fubini 交换合法。$\square$

**命题 F.21.1（Tate theta 恒等式）.** 对 $\Phi\in\mathcal S(\mathbb A_K)$ 和 $t\in\mathbb A_K^\times$，定义
$$
\Theta_\Phi(t)=\sum_{\gamma\in K}\Phi(\gamma t).
$$
则
$$
\Theta_\Phi(t)
=
|t|_{\mathbb A}^{-1}\Theta_{\widehat\Phi}(t^{-1}).
$$
若
$$
\Theta_\Phi^\times(t)=\sum_{\gamma\in K^\times}\Phi(\gamma t),
$$
则
$$
\Theta_\Phi^\times(t)+\Phi(0)
=
|t|_{\mathbb A}^{-1}
\left(\Theta_{\widehat\Phi}^\times(t^{-1})+\widehat\Phi(0)\right).
$$

**证明.** 第一式正是命题 F.20.1。第二式由
$$
\Theta_\Phi(t)=\Theta_\Phi^\times(t)+\Phi(0),
\qquad
\Theta_{\widehat\Phi}(t^{-1})
=\Theta_{\widehat\Phi}^\times(t^{-1})+\widehat\Phi(0)
$$
代入第一式得到。$\square$

**外部输入定理 F.22（Tate 整体函数方程的 Fourier 分析核心）.** 对 $\Phi\in\mathcal S(\mathbb A_K)$ 和 Hecke 特征 $\chi$，Poisson summation 应用于函数
$$
x\mapsto \Phi(tx)
$$
并结合局部函数方程，给出完成 zeta 积分的 meromorphic continuation 和函数方程。精确极点取决于 $\chi$ 是否平凡以及 $K$ 的体积归一化。

**注 F.23.** 本书第二章把 Tate thesis 作为外部输入定理 2.13。本附录说明其 Fourier 分析骨架，但不替代 Tate thesis 的完整证明；完整证明还需要局部 zeta 积分的有理性、Archimedean gamma 因子分析和整体积分截断。

**注 F.23.1.** 命题 F.21.1 是 Tate thesis 中极点来源的形式核心。平凡特征时，$\Phi(0)$ 与 $\widehat\Phi(0)$ 的两项在 idele norm 方向积分后产生 $s=1$ 与 $s=0$ 的可能极点；非平凡特征在 $C_K$ 上积分这些常数项时消失。

## F.7 本附录小结

本附录给出五个接口：

1. 局部紧 Abel 群的对偶和 Fourier inversion。
2. 非 Archimedean 局部域上 $\mathbf 1_{\mathcal O_F}$ 的 Fourier 变换计算。
3. 紧开陪集、有限 Abel 群和 $\mathbb A_\mathbb Q/\mathbb Q$ 基本域的可计算模型。
4. $\mathcal S(\mathbb A_K)$ 的 restricted tensor product 与 Fourier 变换相容。
5. Adele Poisson summation、idele 缩放公式和 Tate theta 恒等式作为整体函数方程的入口。

## 练习

**练习 F.1.** 证明若 $H\subset G$ 为闭子群，则 $H^\perp$ 的 annihilator 在 $\widehat{\widehat G}\simeq G$ 下等于 $H$。

**练习 F.2.** 设 $F$ 为非 Archimedean 局部域，$a\in F^\times$。计算 $\widehat{\mathbf 1_{a\mathcal O_F}}$。

**练习 F.3.** 对 $K=\mathbb Q$，取标准加法特征，说明 $\prod_p\mathbf 1_{\mathbb Z_p}$ 在有限 adele Fourier 变换下保持不变。

**练习 F.4.** 从 F.20 推出 $\mathbb Q$ 上 classical Poisson summation 的形式
$$
\sum_{n\in\mathbb Z}f(n)=\sum_{n\in\mathbb Z}\widehat f(n)
$$
对 $f\in\mathcal S(\mathbb R)$ 成立。

**练习 F.5.** 说明 Tate thesis 中平凡特征的极点为什么来自 Poisson summation 中的零点项。

**练习 F.6.** 用命题 F.11.1 计算 $\widehat{\mathbf 1_{a+b\mathcal O_F}}$。

**练习 F.7.** 证明命题 F.18.1 中使用的局部条件 $x_p-q_0\in\mathbb Z_p$ 可由中国剩余定理同时满足。

**练习 F.8.** 从命题 F.21.1 推出平凡特征时 Tate 整体 zeta 积分中常数项对 $s=0,1$ 的贡献形式。
