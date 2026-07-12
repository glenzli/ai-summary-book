# 习题解答与提示

本文档给出核心习题的解答或严格提示。它优先覆盖全书主线中反复使用的基本计算：restricted product、局部特征、Fourier/Hecke 计算、Frobenius 归一化、L 因子、费马应用链。其余习题应按相同格式继续补入。

## 第一章

**练习 1.1.** 对 $x=18/25\in\mathbb Q^\times$，计算 $\prod_v|x|_v$。

**解答.** 写
$$
x=\frac{18}{25}=2\cdot 3^2\cdot 5^{-2}.
$$
有限素数处采用 $|p|_p=p^{-1}$。于是
$$
|x|_2=2^{-1},\qquad |x|_3=3^{-2},\qquad |x|_5=5^2,
$$
其余素数处 $|x|_p=1$。Archimedean 处
$$
|x|_\infty=\frac{18}{25}=2\cdot3^2\cdot5^{-2}.
$$
故
$$
|x|_\infty\prod_p|x|_p
=\left(2\cdot3^2\cdot5^{-2}\right)\left(2^{-1}3^{-2}5^2\right)=1.
$$
$\square$

**练习 1.2.** 证明 $\prod_p\mathbb Z_p$ 是 $\mathbb A_{\mathbb Q,f}$ 的开紧子环。

**解答.** 有限 adele 环定义为 restricted product
$$
\mathbb A_{\mathbb Q,f}=\prod_p' \mathbb Q_p
$$
相对于开紧子环 $\mathbb Z_p$。在 restricted product 拓扑中，基本开集形如
$$
\prod_{p\in S}U_p\times\prod_{p\notin S}\mathbb Z_p,
$$
其中 $S$ 有限且 $U_p\subset\mathbb Q_p$ 开。因此取 $S=\varnothing$ 得 $\prod_p\mathbb Z_p$ 是开集。每个 $\mathbb Z_p$ 紧，Tychonoff 定理给出 $\prod_p\mathbb Z_p$ 紧。逐坐标加法和乘法保持 $\mathbb Z_p$，所以它是子环。$\square$

**练习 1.3.** 设 $x=(x_v)_v\in\mathbb A_K^\times$。证明 $x^{-1}=(x_v^{-1})_v$ 仍属于 $\mathbb A_K^\times$。

**解答.** Idele 群是 restricted product
$$
\mathbb A_K^\times=\prod_v'(K_v^\times,\mathcal O_v^\times).
$$
这意味着 $x_v\in\mathcal O_v^\times$ 对几乎所有非 Archimedean 位置 $v$ 成立。对这些 $v$，$x_v^{-1}\in\mathcal O_v^\times$。对剩下有限多个位置，$x_v\in K_v^\times$，所以 $x_v^{-1}\in K_v^\times$。故 $x^{-1}$ 满足同一个 restricted product 条件。$\square$

**练习 1.4.** 对 $K=\mathbb Q$，证明 $\mathbb Q$ 在 $\mathbb A_\mathbb Q$ 中是离散的。

**解答.** 令
$$
U=(-1/2,1/2)\times\prod_p\mathbb Z_p\subset\mathbb A_\mathbb Q.
$$
若 $q\in\mathbb Q\cap U$，则 $q\in\mathbb Z_p$ 对所有素数 $p$ 成立。于是 $q$ 的分母不含任何素因子，所以 $q\in\mathbb Z$。同时 $q\in(-1/2,1/2)$，故 $q=0$。因此 $U\cap\mathbb Q=\{0\}$。平移给出每个有理点在对角嵌入下都是孤立点，所以 $\mathbb Q$ 离散。$\square$

## 第二章

**练习 2.1.** 若 $\chi$ 分歧，说明标准函数 $\mathbf 1_{\mathcal O_F}$ 的局部积分可能为 $0$，但可取其他 $\phi$ 使积分非零。

**解答.** 设 $F$ 为非 Archimedean 局部域。若 $\chi$ 在 $\mathcal O_F^\times$ 上非平凡，则
$$
Z(\mathbf 1_{\mathcal O_F},\chi,s)
=\sum_{n\ge0}\int_{\varpi^n\mathcal O_F^\times}\chi(x)|x|^s\,d^\times x.
$$
在每个壳 $\varpi^n\mathcal O_F^\times$ 上写 $x=\varpi^nu$，积分为
$$
\chi(\varpi)^nq^{-ns}\int_{\mathcal O_F^\times}\chi(u)\,d^\times u.
$$
紧群 $\mathcal O_F^\times$ 上非平凡连续特征的积分为 $0$，故标准函数给出 $0$。另一方面，因 $\chi$ 连续，存在开紧子群 $U\subset\mathcal O_F^\times$ 使 $\chi|_U=1$。取 $\phi=\mathbf 1_U$，则
$$
Z(\phi,\chi,s)=\int_U1\,d^\times x=\operatorname{vol}(U)\ne0.
$$
$\square$

**练习 2.2.** Dirichlet 特征如何给出 $\mathbb Q$ 上有限阶 Hecke 特征。

**解答.** 先设 $\chi:(\mathbb Z/N\mathbb Z)^\times\to\mathbb C^\times$ 的精确导子为 $N$；若原特征 imprimitive，则先换成其 primitive associate。令
$$
\widehat\chi:\widehat{\mathbb Z}^\times\to\mathbb C^\times,
\qquad
u=(u_p)_p\longmapsto\chi(u\bmod N).
$$
由中国剩余定理，$u\bmod N$ 良定义。每个 idèle $x\in\mathbb A_\mathbb Q^\times$ 可唯一写成
$$
x=qru,
\qquad
q\in\mathbb Q^\times,\qquad
r\in\mathbb R_{>0},\qquad
u\in\widehat{\mathbb Z}^\times.
$$
定义
$$
\omega_\chi(x):=\widehat\chi(u)^{-1}.
$$
该式忽略 $q$ 与 $r$，故对对角 $\mathbb Q^\times$ 平凡并下降为
$$
\omega_\chi:C_\mathbb Q
=\mathbb Q^\times\backslash\mathbb A_\mathbb Q^\times
\longrightarrow\mathbb C^\times.
$$
其像包含于 $\chi$ 的有限像，所以 $\omega_\chi$ 有限阶。

令 $a_\chi\in\{0,1\}$ 由 $\chi(-1)=(-1)^{a_\chi}$ 定义。对只在无穷处取值 $x_\infty<0$ 的 idèle，上述分解中必须取 $q=-1$，因而
$$
\omega_{\chi,\infty}(x_\infty)
=\operatorname{sgn}(x_\infty)^{a_\chi}.
$$
所以除非 $\chi(-1)=1$，无穷分量不能声明为平凡。对 $p\nmid N$，局部 character 非分歧，而把 $p$ 放在第 $p$ 个 idèle 坐标的分解给出
$$
\omega_{\chi,p}(p)=\chi(p).
$$
因此
$$
L(s,\omega_\chi)
=\prod_{p\nmid N}(1-\chi(p)p^{-s})^{-1}
=L(s,\chi).
$$
这解释了为什么 finite-unit restriction 要取 $\widehat\chi^{-1}$；若取 $\widehat\chi$，则一致化元取值变为 $\chi(p)^{-1}$，Euler 乘积对应 $L(s,\overline\chi)$。若原特征模 $M$ 但由导子 $N\mid M$ 的 $\chi$ 诱导，则该 Hecke L 函数是 primitive $L(s,\chi)$；模 $M$ 的 imprimitive Dirichlet L 函数还要删去 $p\mid M, p\nmid N$ 的 Euler factors。$\square$

**练习 2.3.** 平凡 Hecke 特征的完成 L 函数为何允许在 $s=0,1$ 出现极点。

**解答.** 平凡 Hecke 特征对应 Dedekind zeta 函数 $\zeta_K(s)$。Tate thesis 的整体 zeta 积分在平凡特征情形含有来自 idele norm 方向的常数项。Poisson summation 把 $s$ 与 $1-s$ 联系起来；$s=1$ 的简单极点因而与 $s=0$ 的简单极点配对。精确的整性判准不是“特征非平凡”，而是
$$
\chi|_{C_K^1}\ne1.
$$
若酉特征为非平凡纯 norm 特征 $\chi=|\cdot|_{\mathbb A}^{it}$，完成 L 函数仍是 $\zeta_K(s+it)$ 的平移，在数域情形于 $-it,1-it$ 有极点；函数域情形还要加入 $2\pi i/\log q$ 的周期。$\square$

**练习 2.4.** `GL(1)` Langlands 中三侧对象。

**解答.** Galois 侧在有限阶情形为连续有限像特征
$$
\rho:G_K\to\mathbb C^\times.
$$
自守侧为有限阶 Hecke 特征
$$
\chi:C_K\to\mathbb C^\times.
$$
相容性是指 $\chi=\rho\circ\operatorname{rec}_K$，并且在几乎所有非分歧位置 $v$，
$$
L(s,\rho_v)=L(s,\chi_v).
$$
若允许一般 quasi-character，则 Galois 侧应换成 Weil 侧，避免把非有限像连续复特征误写为 profinite Galois 表示。$\square$

## 第三章

**练习 3.1.** 非分歧特征由 $\chi(\varpi)$ 唯一决定。

**解答.** 非 Archimedean 局部域满足
$$
F^\times=\varpi^\mathbb Z\times\mathcal O_F^\times.
$$
若 $\chi$ 非分歧，则 $\chi|_{\mathcal O_F^\times}=1$。任意 $x\in F^\times$ 可唯一写成 $x=\varpi^nu$，$u\in\mathcal O_F^\times$，于是
$$
\chi(x)=\chi(\varpi)^n.
$$
若换用 $\varpi'=u_0\varpi$，则 $\chi(\varpi')=\chi(u_0)\chi(\varpi)=\chi(\varpi)$。$\square$

**练习 3.2.** 几何 Frobenius 与局部 L 因子公式。

**解答.** 本书约定局部 reciprocity map 把一致化元 $\varpi$ 送到几何 Frobenius $\operatorname{Fr}_F$。若非分歧特征 $\chi$ 对应 Weil 特征 $\rho$，则
$$
\rho(\operatorname{Fr}_F)=\chi(\varpi).
$$
因此
$$
L(s,\rho)=\left(1-\rho(\operatorname{Fr}_F)q^{-s}\right)^{-1}
=\left(1-\chi(\varpi)q^{-s}\right)^{-1}.
$$
若改用算术 Frobenius $\operatorname{Frob}^{\operatorname{arith}}_F=\operatorname{Fr}_F^{-1}$，同一个表示在算术 Frobenius 上的值为 $\chi(\varpi)^{-1}$，公式必须相应改写。$\square$

**练习 3.3.** 解释全局类域论在有限 Abel 商上的同构为何与局部 Artin 映射相容。

**解答.** 全局 reciprocity map
$$
\operatorname{rec}_K:C_K\to G_K^{\operatorname{ab}}
$$
由各局部映射 $\operatorname{rec}_{K_v}:K_v^\times\to G_{K_v}^{\operatorname{ab}}$ 拼合而成，并要求对每个位置 $v$，下图在有限 Abel 商中交换：
$$
\begin{array}{ccc}
K_v^\times & \longrightarrow & C_K\\
\downarrow \operatorname{rec}_{K_v} && \downarrow \operatorname{rec}_K\\
G_{K_v}^{\operatorname{ab}} & \longrightarrow & G_K^{\operatorname{ab}} .
\end{array}
$$
这里下方水平箭头由选取嵌入 $\overline K\hookrightarrow\overline {K_v}$ 后的分解群映射给出，在 Abel 化后与选择无关。该相容性是全局类域论陈述的一部分，而不是从抽象拓扑同构自动推出。$\square$

**练习 3.4.** Dirichlet 特征对应的一维 Galois 表示在哪些素数处分歧。

**解答.** 若 Dirichlet 特征 $\chi$ 的导子为 $N_\chi$，则对应 Hecke 特征在 $p\nmid N_\chi$ 处非分歧，在 $p\mid N_\chi$ 处可能分歧且导子指数等于局部字符在 $\mathbb Z_p^\times$ 上的导子指数。通过局部类域论，Galois 表示 $\rho_\chi$ 在 $p\nmid N_\chi$ 处惯性作用平凡，在 $p\mid N_\chi$ 处惯性作用由 $\chi_p|\mathbb Z_p^\times$ 给出。因此分歧素数正是 $N_\chi$ 的素因子。$\square$

**练习 3.5.** 有限阶 Hecke 特征的 conductor 和 ray class factorization。

**解答.** 对非 Archimedean 位置 $v$，令 $n_v$ 为最小非负整数，使得
$$
\chi_v|_{1+\mathfrak p_v^{n_v}}=1,
$$
其中 $n_v=0$ 表示 $\chi_v$ 在 $\mathcal O_v^\times$ 上平凡。因为 $\chi$ 是 idele class character，$\chi_v$ 对几乎所有 $v$ 非分歧，所以只有有限多个 $n_v$ 非零。定义
$$
\mathfrak f(\chi)=\prod_{v<\infty}\mathfrak p_v^{n_v}.
$$
设 $U(\mathfrak f)=\prod_{v<\infty}U_v$，其中
$$
U_v=
\begin{cases}
1+\mathfrak p_v^{n_v},& n_v>0,\\
\mathcal O_v^\times,& n_v=0.
\end{cases}
$$
则 $\chi$ 在 $U(\mathfrak f)$ 上平凡，并且在 $K^\times$ 上平凡，所以它通过商
$$
K^\times\backslash\mathbb A_K^\times/
\left(K_\infty^\times\cdot U(\mathfrak f)\right)
$$
的有限部分分解。加入实位符号条件后，该商就是相应 modulus 的 ray class group。因此 conductor 是使 $\chi$ 通过 ray class group 分解的最小有限模。类域论再把该 ray class group 的 character 解释为 ray class field 的 Abel Galois character。$\square$

## 第四章

**练习 4.1.** 设 $G$ 为 locally profinite group，$J\subset G$ 开紧。证明 $e_J=\mathbf 1_J$ 在 $\operatorname{vol}(J)=1$ 时为幂等元。

**解答.** 对 $g\in G$，
$$
(e_J*e_J)(g)=\int_G\mathbf 1_J(x)\mathbf 1_J(x^{-1}g)\,dx.
$$
积分域为 $J\cap gJ$。若 $g\in J$，则 $J\cap gJ=J$，积分为 $1$。若 $g\notin J$，则 $J\cap gJ=\varnothing$，因为 $x\in J$ 且 $x^{-1}g\in J$ 会推出 $g\in J$。故 $e_J*e_J=e_J$。$\square$

**练习 4.2.** 证明 $\pi(e_J)$ 是到 $V^J$ 的投影。

**解答.** 对 $v\in V$，
$$
\pi(e_J)v=\int_J\pi(j)v\,dj.
$$
若 $j_0\in J$，利用左不变性得
$$
\pi(j_0)\pi(e_J)v=\int_J\pi(j_0j)v\,dj=\int_J\pi(j)v\,dj=\pi(e_J)v.
$$
故像包含在 $V^J$。若 $v\in V^J$，则
$$
\pi(e_J)v=\int_Jv\,dj=v
$$
因为 $\operatorname{vol}(J)=1$。再由 $e_J*e_J=e_J$ 得 $\pi(e_J)^2=\pi(e_J)$。$\square$

## 第五章

**练习 5.1.** 证明 $W_F$ 在 $G_F$ 中稠密，并说明 $W_F/I_F\simeq\mathbb Z$。

**解答.** 非 Archimedean 局部域的 Weil 群定义为
$$
W_F=\{g\in G_F:\text{其在 }G_F/I_F\simeq\widehat{\mathbb Z}\text{ 中的像属于 }\mathbb Z\}.
$$
其中 $\mathbb Z\subset\widehat{\mathbb Z}$ 由 Frobenius 的幂生成。因为 $\mathbb Z$ 在 $\widehat{\mathbb Z}$ 中稠密，$W_F$ 在 $G_F$ 中稠密。商 $W_F/I_F$ 是 Frobenius 幂的离散无限循环群，所以同构于 $\mathbb Z$；而 $G_F/I_F\simeq\widehat{\mathbb Z}$ 是 profinite completion。$\square$

**练习 5.2.** 非分歧特征对应 Weil 参数在 $\operatorname{Fr}_F$ 上的值。

**解答.** 本书采用几何 Frobenius 归一化，局部 reciprocity map 满足
$$
\operatorname{rec}_F(\varpi)=\operatorname{Fr}_F.
$$
若 $\chi:F^\times\to\mathbb C^\times$ 非分歧，则对应参数为
$$
\rho_\chi=\chi\circ\operatorname{rec}_F^{-1}
$$
在 Abel 化意义下定义。因此
$$
\rho_\chi(\operatorname{Fr}_F)=\chi(\varpi).
$$
$\square$

**练习 5.3.** 证明 $(\ker N)^{I_F}$ 被 $r(\operatorname{Fr}_F)$ 保持。

**解答.** Weil-Deligne 数据满足
$$
r(w)Nr(w)^{-1}=|w|N.
$$
若 $v\in\ker N$，则
$$
N(r(w)v)=|w|^{-1}r(w)Nv=0,
$$
所以 $\ker N$ 被 $r(w)$ 保持。若 $v$ 还被 $I_F$ 固定，取 $\tau\in I_F$，则
$$
r(\tau)r(\operatorname{Fr}_F)v
=r(\operatorname{Fr}_F)r(\operatorname{Fr}_F^{-1}\tau\operatorname{Fr}_F)v.
$$
惯性群正规，$\operatorname{Fr}_F^{-1}\tau\operatorname{Fr}_F\in I_F$，故右端等于 $r(\operatorname{Fr}_F)v$。因此 $r(\operatorname{Fr}_F)v\in(\ker N)^{I_F}$。$\square$

**练习 5.4.** `GL(2)` 非分歧主级数的二维非分歧 Weil 参数。

**解答.** 设
$$
\pi=\operatorname{Ind}_B^{\operatorname{GL}_2(F)}(\chi_1\otimes\chi_2)
$$
为归一化非分歧主级数，$\chi_i$ 非分歧。对应 Weil-Deligne 表示可取 $N=0$，惯性作用平凡，并满足
$$
r(\operatorname{Fr}_F)
=\begin{pmatrix}
\chi_1(\varpi)&0\\
0&\chi_2(\varpi)
\end{pmatrix}
$$
在与正文 Satake 归一化一致的 convention 下成立。若采用 unitary normalization 或算术 Frobenius，需要按章节约定加入 $q^{\pm1/2}$ 或取逆。$\square$

## 第六章至第七章

**练习 6.1.** 证明 slash 算子满足右作用律。

**解答.** 对 $\gamma_i=\begin{pmatrix}a_i&b_i\\c_i&d_i\end{pmatrix}\in\operatorname{GL}_2^+(\mathbb R)$，设
$$
j(\gamma,z)=cz+d.
$$
有 cocycle 恒等式
$$
j(\gamma_1\gamma_2,z)=j(\gamma_1,\gamma_2z)j(\gamma_2,z)
$$
并且 $\det(\gamma_1\gamma_2)=\det(\gamma_1)\det(\gamma_2)$。代入权 $k$ slash 算子定义，得到
$$
(f|_k\gamma_1)|_k\gamma_2=f|_k(\gamma_1\gamma_2).
$$
$\square$

**练习 6.2.** 说明 $T_\ell$ 的 Fourier 系数公式如何给出 Hecke 关系。

**解答.** 当 $\ell\nmid N$ 时，命题 6.12 给出
$$
a_n(T_\ell f)=a_{\ell n}(f)+\varepsilon(\ell)\ell^{k-1}a_{n/\ell}(f),
$$
其中若 $\ell\nmid n$ 则第二项为 $0$。若 $f$ 是归一化本征形式且 $T_\ell f=a_\ell(f)f$，比较第 $n$ 个 Fourier 系数得
$$
a_\ell(f)a_n(f)=a_{\ell n}(f)+\varepsilon(\ell)\ell^{k-1}a_{n/\ell}(f).
$$
这正是 Euler 乘积中局部二次因子的递推来源。$\square$

**练习 7.1.** 证明 $K_0(N)$ 是开紧子群。

**解答.** 在有限 adele 群中
$$
K_0(N)=\{g\in\operatorname{GL}_2(\widehat{\mathbb Z}):c(g)\equiv0\pmod N\}.
$$
它是紧群 $\operatorname{GL}_2(\widehat{\mathbb Z})$ 中模 $N$ 约化映射
$$
\operatorname{GL}_2(\widehat{\mathbb Z})\to\operatorname{GL}_2(\mathbb Z/N\mathbb Z)
$$
下某个子集的逆像。因此它既开又闭；作为紧群的闭子集紧。子群性质来自矩阵乘法中左下角模 $N$ 的闭合性。$\square$

**练习 7.2.** 说明经典尖点条件如何变成 adelic 尖点积分。

**解答.** 对 $G=\operatorname{GL}_2$，标准上三角 unipotent 群 $N$ 的有理点商
$$
N(\mathbb Q)\backslash N(\mathbb A_\mathbb Q)
$$
对应经典变量中的 $x$ 平移周期。adelic 尖点条件要求
$$
\int_{N(\mathbb Q)\backslash N(\mathbb A_\mathbb Q)}\Phi(ng)\,dn=0
$$
对所有 $g$ 成立。取 $g$ 的无穷分量给出某个尖点的坐标后，该积分抽取 Fourier 展开中的常数项。所有尖点常数项为零等价于经典 cusp form 条件。$\square$

**练习 7.6.** Classical normalization 与 unitary automorphic normalization 的变量平移。

**解答.** 好素数 $p\nmid N$ 处，classical Satake roots 满足
$$
X^2-a_pX+\varepsilon(p)p^{k-1}=(X-\alpha_p)(X-\beta_p),
$$
因此 classical Euler factor 为
$$
L_p(f,s)=(1-\alpha_pp^{-s})^{-1}(1-\beta_pp^{-s})^{-1}.
$$
对应的 unitary automorphic normalization 把 Satake 参数改为
$$
\left(\alpha_pp^{-(k-1)/2},\beta_pp^{-(k-1)/2}\right).
$$
于是
$$
L_p(s,\pi_f,\operatorname{Std})
=\prod_{\gamma\in\{\alpha_p,\beta_p\}}
\left(1-\gamma p^{-(k-1)/2}p^{-s}\right)^{-1}
=L_p(f,s+(k-1)/2).
$$
对所有好素数取乘积得到未完成 L 函数的相容公式。坏素数和 Archimedean gamma factors 需按 newform theory 和所选完成因子另行声明。$\square$

## 第八章至第十章

**练习 8.1.** 好约化处局部因子的 Frobenius 表达式。

**解答.** 若 $E/\mathbb Q$ 在 $p$ 处好约化，定义
$$
a_p=p+1-\#E(\mathbb F_p).
$$
Frobenius 在 $T_\ell(E)$ 上的特征多项式为
$$
X^2-a_pX+p.
$$
因此
$$
L_p(E,s)=\det(1-\operatorname{Frob}_p^{\operatorname{arith}}p^{-s}\mid V_\ell(E))^{-1}
=(1-a_pp^{-s}+p^{1-2s})^{-1}.
$$
$\square$

**练习 9.1.** 为什么稳定格约化后需要半单化。

**解答.** 同一个 $\ell$-adic 表示可能有不同的稳定格。两个稳定格给出的模 $\lambda$ 表示不必同构，但其 Jordan-Holder 因子相同；这是 Brauer-Nesbitt 型结论的内容。因此残余表示在本书中默认取半单化同构类
$$
\overline\rho^{\operatorname{ss}}.
$$
这样残余模性、导子和 Hecke eigenvalue 比较才不依赖格的非本质选择。$\square$

**练习 10.1.** 解释 Ribet 降层为什么需要局部-整体相容。

**解答.** 降层要判断某个素数 $q$ 是否能从级中删除。级的 $q$-部分来自局部表示 $\pi_q$ 或局部 Galois 表示 $\rho|_{G_{\mathbb Q_q}}$ 的分歧。局部-整体相容把
$$
\operatorname{WD}(\rho_{f,\lambda}|_{G_{\mathbb Q_q}})
$$
与 $\pi_{f,q}$ 的局部 Langlands 参数比较，使 Galois 侧残余导子变化能翻译成 automorphic 侧级的变化。没有该相容性，残余表示在 $q$ 处分歧降低并不能推出存在较低级 newform。$\square$

## 第十一章至第十五章

**练习 11.1.** 对 $T=\mathbb G_m^r$ 计算 $X^*(T)$、$X_*(T)$ 和配对。

**解答.** 任一代数 character 形如
$$
(t_1,\ldots,t_r)\mapsto t_1^{a_1}\cdots t_r^{a_r}
$$
其中 $(a_1,\ldots,a_r)\in\mathbb Z^r$。故 $X^*(T)\simeq\mathbb Z^r$。任一 cocharacter 形如
$$
z\mapsto (z^{b_1},\ldots,z^{b_r})
$$
其中 $(b_1,\ldots,b_r)\in\mathbb Z^r$。故 $X_*(T)\simeq\mathbb Z^r$。配对为
$$
\langle(a_i),(b_i)\rangle=\sum_{i=1}^ra_ib_i.
$$
$\square$

**练习 12.1.** 为什么 `GL(n)` 的 L-packet 为单元素。

**解答.** 对 `GL(n)`，局部 Langlands 已知为
$$
\operatorname{Irr}(\operatorname{GL}_n(F))\longleftrightarrow \operatorname{WDRep}_n(F).
$$
右侧每个参数给出唯一不可约可容许表示的同构类。以一般还原群语言看，$\widehat G=\operatorname{GL}_n(\mathbb C)$，一个半单 WD 参数的中心化子在相关情形中给出的 component group 不产生多个表示标签。因此 packet 退化为单元素集合。$\square$

**练习 13.1.** 说明部分 L 函数为何需要去掉有限集合 $S$。

**解答.** 若 $v$ 为 Archimedean、表示分歧、群分歧或 L 群表示 $r$ 分歧的位置，则非分歧 Satake 参数公式不直接适用。由于自守表示和代数群在几乎所有有限位置非分歧，这些坏位置可收入有限集合 $S$。在 $v\notin S$ 时可用统一公式
$$
L(s,\pi_v,r)=\det(1-r(s_v)q_v^{-s})^{-1}.
$$
于是先定义
$$
L^S(s,\pi,r)=\prod_{v\notin S}L(s,\pi_v,r),
$$
再单独处理坏局部因子。$\square$

**练习 15.1.** 强转移推出弱转移。

**解答.** 强转移要求每个位置 $v$ 的局部参数满足
$$
\varphi_{\Pi_v}=\xi\circ\varphi_{\sigma_v}
$$
或相应 packet 形式。弱转移只要求几乎所有非分歧位置的 Satake 参数满足该等式。因为强条件对所有位置成立，限制到几乎所有非分歧位置即得到弱条件。$\square$

**练习 15.3.** 弱转移推出非分歧部分 L 函数相容。

**解答.** 取有限集合 $S$，使得 $v\notin S$ 时 $\sigma_v,\Pi_v$ 均非分歧，且弱转移条件成立。记 $\sigma_v$ 的 Satake 参数为 $s_{\sigma_v}\in{}^LH$，则 $\Pi_v$ 的 Satake 参数为
$$
s_{\Pi_v}=\xi(s_{\sigma_v})
$$
的共轭类。于是
$$
L(s,\Pi_v,r)
=\det(1-r(s_{\Pi_v})q_v^{-s})^{-1}
=\det(1-(r\circ\xi)(s_{\sigma_v})q_v^{-s})^{-1}
=L(s,\sigma_v,r\circ\xi).
$$
对所有 $v\notin S$ 取乘积，得
$$
L^S(s,\Pi,r)=L^S(s,\sigma,r\circ\xi).
$$
该结论只使用非分歧 Satake 参数相容；坏位置的局部因子需要强转移或额外局部理论。$\square$

## 第十六章至第二十二章

**练习 16.1.** 为什么 trace formula 比较可产生函子性转移。

**解答.** Trace formula 的几何侧由轨道积分组成，谱侧由自守表示或 packet 的 character 组成。若 endoscopic 数据给出匹配测试函数，并且 transfer factor 使几何侧稳定轨道积分相等，则稳定 trace formula 给出谱侧稳定分布相等。若谱侧稳定分布已经按 L-packet 分解，就能识别哪些 $H$ 上的表示应转移到 $G$ 上。这正是 endoscopic functoriality 的机制。$\square$

**练习 19.1.** 对 $G=\operatorname{GL}_n$，解释 dominant coweight 与 $\widehat G$ 的 dominant weight。

**解答.** 对 $G=\operatorname{GL}_n$ 的 diagonal torus，$X_*(T)\simeq\mathbb Z^n$，dominant coweight 可写为
$$
\lambda=(\lambda_1\ge\cdots\ge\lambda_n).
$$
对偶群仍为 $\widehat G=\operatorname{GL}_n(\mathbb C)$，其 maximal torus 的 character lattice 也可识别为 $\mathbb Z^n$。根资料对偶把 $G$ 的 coweight 变成 $\widehat G$ 的 weight，因此同一个整数序列 $\lambda$ 给出 $\widehat G$ 的 dominant highest weight。$\square$

**练习 20.1.** 证明 $V\mapsto V_{\mathcal E}$ 是张量函子。

**解答.** 设 $\mathcal E$ 为主 $\widehat G$-local system。对表示 $V$，关联局部系统为
$$
V_{\mathcal E}=\mathcal E\times^{\widehat G}V.
$$
关联丛构造与直和、张量积和对偶相容：
$$
(V\oplus W)_{\mathcal E}\simeq V_{\mathcal E}\oplus W_{\mathcal E},
$$
$$
(V\otimes W)_{\mathcal E}\simeq V_{\mathcal E}\otimes W_{\mathcal E},
$$
并且平坦连接也按这些操作诱导。因此 $V\mapsto V_{\mathcal E}$ 是从 $\operatorname{Rep}(\widehat G)$ 到局部系统范畴的张量函子。$\square$

**练习 20.4.** Frobenius trace 如何把 Hecke eigensheaf 变成 Hecke eigenfunction。

**解答.** 设底域为 $\mathbb F_q$。对 $\mathcal F$ 在 $\operatorname{Bun}_G$ 上的 $\ell$-adic complex，定义 trace function
$$
f_{\mathcal F}(x)=\operatorname{Tr}(\operatorname{Fr}_x\mid \mathcal F_x),
\qquad x\in\operatorname{Bun}_G(\mathbb F_q),
$$
其中 Frobenius convention 按 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 第 9 节固定。若 $\mathcal F$ 是 eigenvalue 为 $\mathcal E$ 的 Hecke eigensheaf，则对每个 $V\in\operatorname{Rep}(\widehat G)$ 有
$$
\mathsf H_V(\mathcal F)\cong \mathcal F\boxtimes V_{\mathcal E}.
$$
对 $\mathbb F_q$-点取 Frobenius trace，左侧给出 Hecke correspondence 诱导的 Hecke operator 作用在函数 $f_{\mathcal F}$ 上；右侧的 trace 等于
$$
f_{\mathcal F}(x)\cdot
\operatorname{Tr}(\operatorname{Fr}_y\mid (V_{\mathcal E})_y)
$$
的形式。因此 $f_{\mathcal F}$ 是 Hecke eigenfunction，其 eigenvalue 由 $\widehat G$-local system $\mathcal E$ 在点 $y$ 处的 Frobenius conjugacy class 通过表示 $V$ 取 trace 给出。$\square$

**练习 22.1.** 函数域如何同时具有 adelic 和曲线几何描述。

**解答.** 若 $X/\mathbb F_q$ 为光滑射影曲线，$K=\mathbb F_q(X)$。闭点 $x\in X$ 给出离散赋值、完备化 $K_x$ 和整数环 $\mathcal O_x$。于是有 adele 环
$$
\mathbb A_K=\prod_x'(K_x,\mathcal O_x).
$$
同时，$G$-bundles on $X$ 可通过 Beauville-Laszlo 粘合和局部平凡化与 adelic 双商
$$
G(K)\backslash G(\mathbb A_K)/\prod_xG(\mathcal O_x)
$$
联系起来。因此函数域既是整体域，也是曲线的函数域；这是数论 Langlands 和几何 Langlands 在有限域上相接的原因。$\square$

## 第九十章

**练习 90.1.** 证明指数归约。

**解答.** 若存在 $a^n+b^n=c^n$ 的非零整数解，取素数 $\ell\mid n$。写 $n=\ell m$，则
$$
(a^m)^\ell+(b^m)^\ell=(c^m)^\ell.
$$
因此若所有奇素数指数情形无解，则任何含奇素因子的 $n$ 无解。若 $n$ 是 $2$ 的幂且 $n>2$，则 $4\mid n$，写 $n=4m$ 得
$$
(a^m)^4+(b^m)^4=(c^m)^4.
$$
所以指数 $4$ 无解推出所有 $2$ 的高次幂指数无解。$\square$

**练习 90.2.** $\#E(\mathbb F_\ell)$ 与 Frobenius trace 的关系。

**解答.** 若 $E/\mathbb Q$ 在 $\ell$ 处好约化，令
$$
a_\ell=\ell+1-\#E(\mathbb F_\ell).
$$
对任意辅助素数 $p\ne\ell$，算术 Frobenius 在 $V_p(E)$ 上的特征多项式为
$$
X^2-a_\ell X+\ell.
$$
因此
$$
\operatorname{tr}\rho_{E,p}(\operatorname{Frob}_\ell^{\operatorname{arith}})=a_\ell.
$$
$\square$

**练习 90.3.** 验证 $X_0(2)$ 的 genus 为 $0$。

**解答.** 由附录 D，
$$
\mu=[\operatorname{SL}_2(\mathbb Z):\Gamma_0(2)]=3,\quad c=2,\quad e_2=1,\quad e_3=0.
$$
代入
$$
g=1+\frac{\mu}{12}-\frac{e_2}{4}-\frac{e_3}{3}-\frac c2
$$
得
$$
g=1+\frac14-\frac14-1=0.
$$
所以
$$
S_2(\Gamma_0(2))\simeq H^0(X_0(2),\Omega^1)=0.
$$
$\square$

**练习 90.4.** 为什么“费马大定理由 Langlands 纲领证明”不够精确。

**解答.** 第九十章实际使用的输入是半稳定椭圆曲线模性、Ribet 降层、Frey 曲线局部性质和 $S_2(\Gamma_0(2))=0$。这些内容属于 `GL(2)/\mathbb Q` 的模性和局部-整体相容方向，与 Langlands 纲领密切相关，但并不等于一般 Langlands functoriality、一般还原群 LLC 或几何 Langlands。严格说法应是：费马大定理由 `GL(2)/\mathbb Q` 模性定理、Ribet 降层和 Frey 曲线构造推出；这些定理处在 Langlands 纲领的核心脉络中。$\square$

## 附录 F

**练习 F.1.** 证明若 $H\subset G$ 为闭子群，则 $H^\perp$ 的 annihilator 在 $\widehat{\widehat G}\simeq G$ 下等于 $H$。

**解答.** 由 Pontryagin duality，把 $G$ 识别为 $\widehat{\widehat G}$。若 $h\in H$ 且 $\chi\in H^\perp$，则 $\chi(h)=1$，所以 $H\subset(H^\perp)^\perp$。反向包含使用闭子群对偶正合列：F.5 给出
$$
\widehat{G/H}\simeq H^\perp.
$$
若 $g\notin H$，则其像 $\bar g\in G/H$ 非零。Pontryagin duality 保证存在 $\eta\in\widehat{G/H}$ 使 $\eta(\bar g)\ne1$。把 $\eta$ 视为 $H^\perp$ 中 character，则 $\eta(g)\ne1$，故 $g\notin(H^\perp)^\perp$。因此 $(H^\perp)^\perp=H$。$\square$

**练习 F.2.** 设 $F$ 为非 Archimedean 局部域，$a\in F^\times$。计算 $\widehat{\mathbf 1_{a\mathcal O_F}}$。

**解答.** 按定义
$$
\widehat{\mathbf 1_{a\mathcal O_F}}(y)
=\int_{a\mathcal O_F}\psi(xy)\,dx.
$$
令 $x=az$，则 $dx=|a|\,dz$，所以
$$
\widehat{\mathbf 1_{a\mathcal O_F}}(y)
=|a|\int_{\mathcal O_F}\psi(azy)\,dz.
$$
若 $\psi$ 的 conductor 为 $\mathcal O_F$ 且 $\operatorname{vol}(\mathcal O_F)=1$，命题 F.10 给出
$$
\widehat{\mathbf 1_{a\mathcal O_F}}(y)=|a|\mathbf 1_{a^{-1}\mathcal O_F}(y).
$$
一般 conductor 情形中，把 $\mathcal O_F$ 替换为 $\mathfrak d_\psi^{-1}$，得到
$$
\widehat{\mathbf 1_{a\mathcal O_F}}(y)=|a|\operatorname{vol}(\mathcal O_F)\mathbf 1_{a^{-1}\mathfrak d_\psi^{-1}}(y).
$$
$\square$

**练习 F.3.** 对 $K=\mathbb Q$，取标准加法特征，说明 $\prod_p\mathbf 1_{\mathbb Z_p}$ 在有限 adele Fourier 变换下保持不变。

**解答.** 标准加法特征在每个 $\mathbb Q_p$ 上的 conductor 为 $\mathbb Z_p$，并取 Haar 测度使 $\operatorname{vol}(\mathbb Z_p)=1$。由推论 F.11，
$$
\widehat{\mathbf 1_{\mathbb Z_p}}=\mathbf 1_{\mathbb Z_p}
$$
对每个 $p$ 成立。有限 adele 上
$$
\Phi_f=\prod_p\mathbf 1_{\mathbb Z_p}
$$
是 restricted tensor product 的标准向量。由 F.15，
$$
\widehat{\Phi_f}=\prod_p\widehat{\mathbf 1_{\mathbb Z_p}}
=\prod_p\mathbf 1_{\mathbb Z_p}.
$$
$\square$

**练习 F.4.** 从 F.20 推出 $\mathbb Q$ 上 classical Poisson summation 的形式
$$
\sum_{n\in\mathbb Z}f(n)=\sum_{n\in\mathbb Z}\widehat f(n)
$$
对 $f\in\mathcal S(\mathbb R)$ 成立。

**解答.** 取
$$
\Phi=f\otimes\prod_p\mathbf 1_{\mathbb Z_p}\in\mathcal S(\mathbb A_\mathbb Q).
$$
对 $q\in\mathbb Q$，有限部分非零当且仅当 $q\in\mathbb Z$，所以
$$
\sum_{q\in\mathbb Q}\Phi(q)=\sum_{n\in\mathbb Z}f(n).
$$
由 F.3 的解答和张量分解，
$$
\widehat\Phi=\widehat f\otimes\prod_p\mathbf 1_{\mathbb Z_p}.
$$
因此
$$
\sum_{q\in\mathbb Q}\widehat\Phi(q)=\sum_{n\in\mathbb Z}\widehat f(n).
$$
把这两个等式代入 adele Poisson 公式 F.20 得到 classical Poisson summation。$\square$

**练习 F.5.** 说明 Tate thesis 中平凡特征的极点为什么来自 Poisson summation 中的零点项。

**解答.** Tate theta 恒等式写为
$$
\Theta_\Phi^\times(t)+\Phi(0)
=|t|_{\mathbb A}^{-1}
\left(\Theta_{\widehat\Phi}^\times(t^{-1})+\widehat\Phi(0)\right).
$$
把整体 zeta 积分改写为 idele class group 上的 theta 积分并按 $|t|_{\mathbb A}\ge1$ 与 $|t|_{\mathbb A}<1$ 分裂时，非零项给出在大区域快速衰减的积分。平凡特征下，常数项 $\Phi(0)$ 与 $\widehat\Phi(0)$ 不被 character 积分消掉；沿 idele norm 方向积分这些常数项会产生形如
$$
\frac{\Phi(0)}{s}
\quad\text{和}\quad
\frac{\widehat\Phi(0)}{1-s}
$$
的简单极点项。非平凡特征在紧的 norm-one idele class 方向上积分常数项为 $0$，因此没有同一来源的极点。$\square$

**练习 F.6.** 用命题 F.11.1 计算 $\widehat{\mathbf 1_{a+b\mathcal O_F}}$。

**解答.** 在 F.11.1 中取 $L=b\mathcal O_F$。则
$$
L^\perp=b^{-1}\mathfrak d_\psi^{-1},
$$
并且
$$
\operatorname{vol}(L)=|b|\operatorname{vol}(\mathcal O_F).
$$
因此
$$
\widehat{\mathbf 1_{a+b\mathcal O_F}}(y)
=
\psi(ay)|b|\operatorname{vol}(\mathcal O_F)
\mathbf 1_{b^{-1}\mathfrak d_\psi^{-1}}(y).
$$
若 $\psi$ 的 conductor 为 $\mathcal O_F$ 且 $\operatorname{vol}(\mathcal O_F)=1$，该式化为
$$
\widehat{\mathbf 1_{a+b\mathcal O_F}}(y)
=
\psi(ay)|b|\mathbf 1_{b^{-1}\mathcal O_F}(y).
$$
$\square$

**练习 F.7.** 证明命题 F.18.1 中使用的局部条件 $x_p-q_0\in\mathbb Z_p$ 可由中国剩余定理同时满足。

**解答.** 对有限集合 $S$ 中每个 $p$，取 $n_p$ 使 $p^{n_p}x_p\in\mathbb Z_p$。在商群 $p^{-n_p}\mathbb Z_p/\mathbb Z_p$ 中，$x_p$ 的类可写成
$$
\frac{c_p}{p^{n_p}}\pmod{\mathbb Z_p},
\qquad c_p\in\mathbb Z.
$$
令
$$
N=\prod_{p\in S}p^{n_p}.
$$
希望 $q_0=A/N$ 满足
$$
\frac{A}{N}-\frac{c_p}{p^{n_p}}\in\mathbb Z_p.
$$
这等价于同余
$$
A\equiv c_p\frac{N}{p^{n_p}}\pmod{p^{n_p}}.
$$
这些模数 $p^{n_p}$ 两两互素，所以中国剩余定理给出整数 $A$ 同时满足所有同余。取 $q_0=A/N$ 即得所需局部条件。$\square$

**练习 F.8.** 从命题 F.21.1 推出平凡特征时 Tate 整体 zeta 积分中常数项对 $s=0,1$ 的贡献形式。

**解答.** 平凡特征时，整体 zeta 积分可写成
$$
\int_{K^\times\backslash\mathbb A_K^\times}\Theta_\Phi^\times(t)|t|_{\mathbb A}^s\,d^\times t
$$
的 Tate thesis normalization。用 F.21.1 把 $|t|_{\mathbb A}<1$ 的部分换成 $\widehat\Phi$ 在 $t^{-1}$ 处的 theta series。该替换产生常数项
$$
-\Phi(0)
\quad\text{和}\quad
|t|_{\mathbb A}^{-1}\widehat\Phi(0).
$$
沿正实 norm 方向积分时，第一类项给出 $s=0$ 附近的简单极点贡献，第二类项给出 $s=1$ 附近的简单极点贡献。非零 theta 项在 Schwartz 衰减和截断后给出全纯部分；因此平凡特征的极点只来自这两个零点项。$\square$

## 附录 G

**练习 G.1.** 对 `GL(3)`，写出所有正根、simple roots 和 Weyl group 的 simple reflections。

**解答.** 对 diagonal torus，根为 $e_i-e_j$。以上三角 Borel 为正根选择时，正根为
$$
e_1-e_2,\qquad e_2-e_3,\qquad e_1-e_3.
$$
Simple roots 为
$$
\alpha_1=e_1-e_2,\qquad \alpha_2=e_2-e_3.
$$
Weyl group 为 $S_3$。Simple reflections 为相邻换位
$$
s_{\alpha_1}=(12),\qquad s_{\alpha_2}=(23).
$$
$\square$

**练习 G.3.** 对 $T=\operatorname{Res}_{E/F}\mathbb G_m$，在 $E/F$ 二次 Galois 时写出 ${}^LT$ 中非平凡 Galois 元素对 $(z_1,z_2)$ 的作用。

**解答.** 若 $\operatorname{Gal}(E/F)=\{1,\sigma\}$，则
$$
\widehat T\simeq\mathbb C^\times\times\mathbb C^\times.
$$
非平凡元素 $\sigma$ 交换两个嵌入 $E\hookrightarrow\overline F$，因此在对偶 torus 上作用为
$$
\sigma\cdot(z_1,z_2)=(z_2,z_1).
$$
所以
$$
{}^LT=(\mathbb C^\times\times\mathbb C^\times)\rtimes W_F,
$$
其中 $W_F$ 经 $\Gamma_F\to\operatorname{Gal}(E/F)$ 非平凡元素时交换两个因子。$\square$

**练习 G.4.** 验证 determinant L 同态在非分歧参数上的 Satake 参数作用。

**解答.** Split `GL(n)` 的非分歧 Satake 参数可写为半单共轭类
$$
s=\operatorname{diag}(\alpha_1,\ldots,\alpha_n)\in\operatorname{GL}_n(\mathbb C).
$$
determinant L 同态的对偶群部分为
$$
\det:\operatorname{GL}_n(\mathbb C)\to\mathbb C^\times.
$$
因此推前参数的 Satake 参数为
$$
\det(s)=\alpha_1\cdots\alpha_n.
$$
$\square$

**练习 G.5.** 对 `GL(2)` Satake 参数 $\operatorname{diag}(\alpha,\beta)$，计算 $\operatorname{Sym}^2$ 推前后的 `GL(3)` Satake 参数。

**解答.** $\operatorname{Sym}^2$ 作用在二元二次齐次多项式空间，基可取
$$
X^2,\quad XY,\quad Y^2.
$$
若
$$
g=\operatorname{diag}(\alpha,\beta),
$$
则
$$
X^2\mapsto \alpha^2X^2,\qquad
XY\mapsto \alpha\beta XY,\qquad
Y^2\mapsto \beta^2Y^2.
$$
故推前后的 `GL(3)` Satake 参数为
$$
\operatorname{diag}(\alpha^2,\alpha\beta,\beta^2).
$$
$\square$

## 附录 H

**练习 H.1.** 证明命题 H.2 中右乘 $\gamma\in\Gamma$ 确实置换右陪集集合。

**解答.** 右陪集集合是
$$
\Gamma\backslash \Gamma\alpha\Gamma.
$$
若 $\gamma\in\Gamma$，则
$$
(\Gamma\alpha\Gamma)\gamma=\Gamma\alpha(\Gamma\gamma)=\Gamma\alpha\Gamma.
$$
因此映射
$$
\Gamma x\mapsto \Gamma x\gamma
$$
从该有限集合到自身。其逆映射为右乘 $\gamma^{-1}$，因为 $\gamma^{-1}\in\Gamma$。所以它是置换。$\square$

**练习 H.3.** 用 H.7 和 H.8 重新证明第六章命题 6.12。

**解答.** 若 $\ell\nmid N$，由 H.4，$T_\ell$ 的代表为
$$
\alpha_b=\begin{pmatrix}1&b\\0&\ell\end{pmatrix},\quad 0\le b<\ell,
\qquad
\beta=\begin{pmatrix}\ell&0\\0&1\end{pmatrix}.
$$
H.7 给出 $\alpha_b$ 部分对第 $n$ 个 Fourier 系数贡献 $a_{\ell n}$。H.8 给出 $\beta$ 部分贡献 $\ell^{k-1}a_{n/\ell}$。带 nebentypus 时该方向还乘以 $\varepsilon(\ell)$。因此
$$
a_n(T_\ell f)=a_{\ell n}+\varepsilon(\ell)\ell^{k-1}a_{n/\ell}.
$$
若 $\ell\mid N$，$U_\ell$ 只使用 $\alpha_b$ 代表族，由 H.7 得
$$
a_n(U_\ell f)=a_{\ell n}.
$$
$\square$

**练习 H.4.** 设 $f$ 为归一化 Hecke eigenform。由 H.9 推出当 $(m,n)=1$ 且 $(mn,N)=1$ 时 $a_{mn}=a_ma_n$。

**解答.** 先对 $m=\ell$ 为素数且 $\ell\nmid nN$ 证明。由 $T_\ell f=a_\ell f$ 和 H.9，比较第 $n$ 个系数：
$$
a_\ell a_n=a_{\ell n}+\varepsilon(\ell)\ell^{k-1}a_{n/\ell}.
$$
因 $\ell\nmid n$，第二项为 $0$，故 $a_{\ell n}=a_\ell a_n$。对一般互素 $m,n$ 且 $(mn,N)=1$，把 $m$ 分解为素数幂，并对 $m$ 的素因子数归纳，反复使用同一公式和 Hecke 算子交换性，得到
$$
a_{mn}=a_ma_n.
$$
$\square$

## 附录 I

**练习 I.1.** 对 $n=1$，说明 Godement-Jacquet 局部积分退化为 Tate thesis 的局部 zeta integral。

**解答.** 当 $n=1$ 时，$M_1(F)=F$，$\operatorname{GL}_1(F)=F^\times$。不可约表示 $\pi$ 是 character $\chi:F^\times\to\mathbb C^\times$，matrix coefficient 就是 $\chi(x)$ 乘以标量。Godement-Jacquet 局部积分变为
$$
Z(s,\Phi,\chi)=\int_{F^\times}\Phi(x)\chi(x)|x|^s\,d^\times x
$$
至多差本附录采用的平移归一化。此即 Tate thesis 的局部 zeta integral。$\square$

**练习 I.2.** 若 `GL(n)` 和 `GL(m)` 的非分歧 Satake 参数分别为 $(\alpha_i)$ 和 $(\beta_j)$，推导 Rankin-Selberg 局部因子。

**解答.** 局部 Langlands 下，非分歧参数可对角化为
$$
\operatorname{diag}(\alpha_1,\ldots,\alpha_n),
\qquad
\operatorname{diag}(\beta_1,\ldots,\beta_m).
$$
Rankin-Selberg L 函数对应 tensor product 参数。其 eigenvalues 为所有乘积
$$
\alpha_i\beta_j,\qquad 1\le i\le n,\ 1\le j\le m.
$$
因此局部 Euler 因子为
$$
L(s,\pi_v\times\pi_v')
=\prod_{i=1}^n\prod_{j=1}^m
(1-\alpha_i\beta_jq_v^{-s})^{-1}.
$$
$\square$

**练习 I.4.** 说明 converse theorem 中为什么需要对足够多 $\tau$ 的 twists，而不是只检查 $L(s,\Pi)$。

**解答.** 标准 L 函数 $L(s,\Pi)$ 只检测 $\Pi$ 的标准 Satake 多项式和一个方向的解析性质。非自守的候选 restricted tensor product 也可能形式上拥有看似良好的标准 Euler 乘积。对所有足够多 cuspidal $\tau$ 检查
$$
L(s,\Pi\times\tau)
$$
会测试 $\Pi$ 与许多不同局部和全局频率的相互作用；这些 twists 足以通过 converse theorem 重构自守性所需的 Fourier-Whittaker 展开和函数方程体系。因此单个标准 L 函数条件太弱，而一族 twists 的解析性质足以强制 automorphy。$\square$

**练习 I.5.** 设 symmetric square lift $\operatorname{Sym}^2\pi$ 已存在。写出非分歧 L 函数相容公式。

**解答.** 若 $v\notin S$ 且 $\pi_v$ 的 Satake 参数为
$$
\operatorname{diag}(\alpha_v,\beta_v),
$$
则附录 G 的 symmetric square L 同态把它送到
$$
\operatorname{diag}(\alpha_v^2,\alpha_v\beta_v,\beta_v^2).
$$
因此
$$
L^S(s,\operatorname{Sym}^2\pi,\operatorname{Std})
=
\prod_{v\notin S}
\left((1-\alpha_v^2q_v^{-s})(1-\alpha_v\beta_vq_v^{-s})(1-\beta_v^2q_v^{-s})\right)^{-1}.
$$
右侧正是 $\pi$ 的 symmetric square L 函数非分歧部分：
$$
L^S(s,\pi,\operatorname{Sym}^2).
$$
$\square$

## 附录 J

**练习 J.3.** 解释为什么 oldforms 与 newforms 的区分不是由好素数 Hecke eigenvalues 决定的。

**解答.** 若 $f$ 是低级 $M$ 的 newform，$M\mid N$，其 degeneracy image 在级 $N$ 中是 oldform。对所有 $p\nmid N$，该 oldform 的 $T_p$ 本征值与原来的 $f$ 相同，因为 degeneracy maps 只改变 $p\mid N/M$ 处的 level vector。强重数一说明几乎所有好素数 Hecke eigenvalues 决定的是全局自守表示 $\pi_f$，而不是该表示中选择的具体 level vector。New/old 区分发生在坏素数处的最小级，即导子层。$\square$

**练习 J.4.** 设 $\pi_p$ 非分歧。用 J.10 说明 $a(\pi_p)=0$。

**解答.** 非分歧表示的定义是
$$
\pi_p^{\operatorname{GL}_2(\mathbb Z_p)}\ne0.
$$
而
$$
K_1(p^0)=\operatorname{GL}_2(\mathbb Z_p).
$$
所以满足 J.10 中不变量非零的最小 $m$ 至多为 $0$。因 $m\ge0$，得到
$$
a(\pi_p)=0.
$$
$\square$

**练习 J.5.** 用 J.16 重写费马应用章中“级 $2$ 无 newform”的矛盾。

**解答.** Frey 曲线和 Ribet 降层给出一个权 $2$、级 $2$ 的 newform，因此由 J.16，
$$
S_2(\Gamma_0(2))_{\operatorname{new}}\ne0.
$$
但
$$
S_2(\Gamma_0(2))_{\operatorname{new}}\subset S_2(\Gamma_0(2)).
$$
附录 D 证明 $S_2(\Gamma_0(2))=0$，于是 new subspace 也为 $0$，矛盾。$\square$

## 附录 K

**练习 K.1.** 对 $k[\epsilon]/(\epsilon^2)$，验证 lift 条件等价于 $1$-cocycle 条件。

**解答.** 写 lift 为
$$
\rho_\epsilon(g)=(1+\epsilon c(g))\overline\rho(g).
$$
表示条件 $\rho_\epsilon(gh)=\rho_\epsilon(g)\rho_\epsilon(h)$ 给出
$$
1+\epsilon c(gh)
=(1+\epsilon c(g))\overline\rho(g)(1+\epsilon c(h))\overline\rho(g)^{-1}.
$$
比较 $\epsilon$ 系数：
$$
c(gh)=c(g)+\operatorname{Ad}(\overline\rho(g))c(h).
$$
这正是 $c$ 为 $Z^1(G,\operatorname{ad}\overline\rho)$ 中 cocycle 的条件。$\square$

**练习 K.3.** 写出 fixed determinant 变形问题中 tangent representation 为什么是 $\operatorname{ad}^0\overline\rho$。

**解答.** 对双数 lift
$$
\rho_\epsilon(g)=(1+\epsilon c(g))\overline\rho(g),
$$
行列式为
$$
\det\rho_\epsilon(g)=\det(1+\epsilon c(g))\det\overline\rho(g)
=(1+\epsilon\operatorname{tr}c(g))\det\overline\rho(g).
$$
若固定 determinant，则 $\operatorname{tr}c(g)=0$ 对所有 $g$ 成立。因此 tangent cocycle 取值于 trace-zero adjoint representation
$$
\operatorname{ad}^0\overline\rho.
$$
$\square$

**练习 K.5.** 用命题 K.19 解释“$R=T$ 推出模性提升”的逻辑。

**解答.** Lift $\rho$ 对应 deformation ring 的一个点
$$
x_\rho:R_{\mathcal S}\to\mathcal O.
$$
若 $R_{\mathcal S}\simeq T_\mathfrak m$，该点就是 Hecke algebra 的一组特征值。Hecke algebra 作用在模形式或相应同调空间上，所以该特征值系统来自 Hecke eigenclass。由 Hecke 侧 Galois 表示，该 eigenclass 对应的 Galois 表示在几乎所有 Frobenius trace 上与 $\rho$ 相同。Chebotarev 给出两者半单同构，因此 $\rho$ 来自模形式。$\square$

## 附录 L

**练习 L.1.** 对 `GL(2)`，写出 Bruhat decomposition 并说明为何常数项有两个 Weyl 项。

**解答.** 对 $G=\operatorname{GL}_2$ 和 Borel $B$，
$$
G(K)=B(K)\sqcup B(K)wB(K),
\qquad
w=\begin{pmatrix}0&1\\-1&0\end{pmatrix}.
$$
Weyl group 有两个元素：单位元和 $w$。计算 Eisenstein series 沿 $B$ 的常数项时，对
$$
B(K)\backslash G(K)/B(K)
$$
分解求和，因此只有两个贡献：单位元项给出原 section，非平凡 Weyl 元项给出 standard intertwining operator。$\square$

**练习 L.4.** 比较 cuspidal spectrum、residual spectrum 和 continuous spectrum 的定义差异。

**解答.** Cuspidal spectrum 由常数项全为零的 cusp forms 生成，属于离散谱。Residual spectrum 由 Eisenstein series 的 poles 的 residues 生成，也属于离散 $L^2$ 谱，但通常来自 proper Levi 的 cuspidal data。Continuous spectrum 由 Eisenstein series 在 unitary axis 上的连续参数积分给出，不是离散直和，而是直接积分。三者都出现在完整自守谱分解中。$\square$

**练习 L.5.** 解释第十七章非 tempered Arthur 参数为什么需要一个额外的 $\operatorname{SL}_2(\mathbb C)$ 因子。

**解答.** 普通 Langlands 参数记录局部或全局 Weil 型数据。Residual spectrum 中的许多表示来自 parabolic induction 和 Eisenstein residues，局部分量常偏离 tempered。Arthur 参数额外加入
$$
\operatorname{SL}_2(\mathbb C)
$$
因子，用其非平凡代数表示记录这种非 tempered 偏移的大小。若该因子平凡，参数接近 tempered Langlands 参数；若非平凡，则对应的 packet 可能含 residual 或非 tempered 离散谱表示。$\square$

## 附录 M

**练习 M.1.** 对 maximal parabolic $P=MN$，解释为何 $\operatorname{Lie}({}^LN)$ 上的 adjoint action 给出若干 L 群表示。

**解答.** Levi 对偶群 ${}^LM$ 通过共轭作用作用在 ${}^LN$ 的 Lie algebra 上：
$$
\operatorname{Ad}:{}^LM\to\operatorname{GL}(\operatorname{Lie}({}^LN)).
$$
按中心 cocharacter 或根高度分解 $\operatorname{Lie}({}^LN)$，可得有限个半单成分
$$
\operatorname{Lie}({}^LN)=\bigoplus_i V_i.
$$
每个成分给出表示
$$
r_i:{}^LM\to\operatorname{GL}(V_i).
$$
Langlands-Shahidi 方法正是为 generic cuspidal data $\pi$ 构造与这些 $r_i$ 相联的局部和全局因子。$\square$

**练习 M.2.** 说明 local coefficient 为什么应控制局部 $\gamma$ 因子。

**解答.** Intertwining operator 的局部函数方程把同一个诱导表示沿 Weyl 元前后比较。Whittaker functional 在 generic 情形唯一，所以归一化 intertwining operator 对 Whittaker functional 的影响只能是一个标量。这个标量就是 local coefficient。全局 Eisenstein 函数方程分解为局部 intertwining operators 的乘积；把全局函数方程按 Euler product 分解后，每个局部标量必须给出相应的局部 $\gamma(s,\pi,r,\psi)$ 因子。$\square$

**练习 M.5.** 若 functorial lift $\Pi$ 到 `GL(N)` 已知，解释附录 M 的 L 因子与 `GL(N)` 标准 L 因子为何相容。

**解答.** Functorial lift 的定义要求几乎所有非分歧位置满足
$$
\varphi_{\Pi_v}=r\circ\varphi_{\pi_v}.
$$
因此在这些位置，
$$
L(s,\Pi_v,\operatorname{Std})
=\det(1-\operatorname{Std}(\varphi_{\Pi_v}(\operatorname{Fr}_v))q_v^{-s})^{-1}
=L(s,\pi_v,r).
$$
两侧的 ramified 因子还需要局部理论归一化。Langlands-Shahidi 方法提供一套由 $\gamma$ 因子和函数方程刻画的归一化；若该归一化与 `GL(N)` LLC 和 Rankin-Selberg 归一化相容，则得到完全 L 函数相等。$\square$

## 附录 N

**练习 N.1.** 解释为什么 split torus 的 L-packet 为单元素。

**解答.** 设 $T\simeq\mathbb G_m^r$。局部类域论给出
$$
\operatorname{Hom}_{\operatorname{cont}}(T(F),\mathbb C^\times)
\simeq
H^1(W_F,\widehat T)
$$
在 split 情形即 $r$ 个 `GL(1)` 对应的乘积。由于 $T(F)$ Abel，每个不可约光滑表示就是一个 character。给定参数只对应这个 character，没有非平凡 Weyl 群或非连通 centralizer 造成多个表示。因此 packet 为单元素。$\square$

**练习 N.4.** 说明 local Jacquet-Langlands 为什么迫使 LLC 同时考虑内形式。

**解答.** 设 $D/F$ 为 quaternion division algebra。$D^\times$ 是 $\operatorname{GL}_2(F)$ 的内形式而非同构群。Local Jacquet-Langlands 把 $D^\times$ 的某些不可约表示对应到 $\operatorname{GL}_2(F)$ 的离散系列表示。若只在 quasi-split 群 $\operatorname{GL}_2$ 上记录参数，就无法说明同一类离散参数在内形式上出现的表示。Enhanced LLC 因此把相关内形式一起纳入，并用 component group 或 Kottwitz 数据切分每个内形式上的成员。$\square$

**练习 N.5.** 解释 fundamental lemma 在稳定 trace formula 中的局部角色。

**解答.** 稳定 trace formula 比较要求在每个局部位置选择匹配测试函数，使 orbital integrals 经 transfer factor 后相等。非分歧位置占几乎所有位置，通常选择 hyperspecial compact subgroup 的单位元函数。Fundamental lemma 断言这些单位元函数在 endoscopic transfer 下匹配。于是全局几何侧比较可以在几乎所有位置无额外误差地分解，只剩有限多个 ramified 位置需要单独控制。$\square$

## 附录 O

**练习 O.2.** 把 Hecke functor 写成 kernel transform。

**解答.** Hecke stack 给出 correspondence
$$
\operatorname{Bun}_G\xleftarrow{h_1}\operatorname{Hecke}_G\xrightarrow{h_2}\operatorname{Bun}_G\times X.
$$
几何 Satake 给每个 $V\in\operatorname{Rep}(\widehat G)$ 一个在 Hecke stack 相对位置方向上的 kernel $\mathcal K_V$。于是 Hecke functor 形如
$$
\mathsf H_V(\mathcal F)=h_{2,!}\bigl(h_1^!\mathcal F\otimes\mathcal K_V\bigr)
$$
或按所选 sheaf theory 改用相应的 $*$、$!$ 版本。这正是 correspondence kernel transform 的标准形式。$\square$

**练习 O.3.** 说明为什么谱侧常取 $\operatorname{IndCoh}_{\mathcal N}$ 而不是全部 $\operatorname{QCoh}$。

**解答.** $\operatorname{LocSys}_{\widehat G}(X)$ 通常是奇异 derived stack，特别在 reducible local systems 处有非平凡 automorphism 和 obstruction theory。全部 $\operatorname{QCoh}$ 对这些奇异方向过大，不能正确匹配自动侧由 Eisenstein、constant term 和 nilpotent 现象控制的范畴。$\operatorname{IndCoh}$ 能记录奇异方向，而下标 $\mathcal N$ 限制 singular support 落在 nilpotent cone 中，从而保留几何 Langlands 需要的谱侧对象并排除过大的非 nilpotent 方向。$\square$

**练习 O.4.** 解释 Riemann-Hilbert 对应在复曲线几何 Langlands 中的作用。

**解答.** 当 $k=\mathbb C$ 且考虑正则 holonomic D-modules 时，Riemann-Hilbert 对应把 D-modules 与 constructible sheaves 或 perverse sheaves 联系起来。因此同一个 Hecke eigensheaf 可以在 de Rham 语言中写成 D-module，也可以在 Betti 语言中用拓扑局部系统和 constructible sheaves 表示。几何 Langlands 的不同版本使用这些口径时，必须通过 Riemann-Hilbert 或其派生增强来比较 Hecke 作用和本征条件。$\square$

## 附录 P

**练习 P.1.** 对 `GL(2)` 写出 Cartan 分解中的双陪集代表。

**解答.** 对 $G=\operatorname{GL}_2(F)$ 和 $K=\operatorname{GL}_2(\mathcal O_F)$，Smith normal form 给出
$$
G=\bigsqcup_{a\ge b}K
\begin{pmatrix}\varpi^a&0\\0&\varpi^b\end{pmatrix}
K.
$$
若模去中心方向，也可把代表写成
$$
\begin{pmatrix}\varpi^m&0\\0&1\end{pmatrix},
\qquad m\ge0,
$$
再乘以中心元素 $\varpi^b I_2$。$\square$

**练习 P.3.** 对 `GL(2)` 的非分歧主级数，写出标准 L 因子的 Satake 参数公式。

**解答.** 设
$$
\pi=\operatorname{Ind}_B^{\operatorname{GL}_2(F)}(\chi_1\otimes\chi_2)
$$
为归一化非分歧主级数，令
$$
\alpha=\chi_1(\varpi),\qquad \beta=\chi_2(\varpi).
$$
Satake 参数为无序二元组 $\{\alpha,\beta\}$，标准表示下的局部 L 因子为
$$
L(s,\pi,\operatorname{Std})
=(1-\alpha q^{-s})^{-1}(1-\beta q^{-s})^{-1}.
$$
$\square$

**练习 P.5.** 解释为什么几何 Satake 是 Satake 同构的范畴化。

**解答.** Classical Satake 同构把球 Hecke 代数识别为 $\widehat G$ 的表示环或 Weyl 不变量坐标环。几何 Satake 把这个环提升为范畴等价：
$$
\operatorname{Perv}_{G(\mathcal O)}(\operatorname{Gr}_G)
\simeq
\operatorname{Rep}(\widehat G).
$$
左侧的卷积对应右侧的张量积。对有限域上的对象取 Frobenius trace 函数，卷积范畴降到球 Hecke 代数，张量范畴的 Grothendieck ring 降到表示环。因此 classical Satake 是几何 Satake 的函数迹影子。$\square$

## 附录 Q

**练习 Q.1.** 对 `GL(2)`，列出 principal series、Steinberg twist 和 supercuspidal 三类不可约表示在 LLC 中的大致参数形状。

**解答.** Principal series 对应可约二维 Weil-Deligne 参数
$$
\chi_1\oplus\chi_2,\qquad N=0.
$$
Steinberg twist $\operatorname{St}\otimes\chi$ 对应同一个 character 上的 special 参数，半单部分为
$$
\chi|\cdot|^{1/2}\oplus\chi|\cdot|^{-1/2}
$$
并带非零 nilpotent operator $N$。按本书的几何 Frobenius convention，其两个特征值为
$\chi(\varpi)q^{-1/2}$ 与 $\chi(\varpi)q^{1/2}$，且 $\ker N$ 取前一个特征线，故非分歧 $\chi$ 给出
$L(s,\operatorname{St}\otimes\chi)=(1-\chi(\varpi)q^{-s-1/2})^{-1}$。Supercuspidal 表示对应不可约二维 Weil 表示，因惯性不变量为零而有标准局部 L 因子 $1$。$\square$

**练习 Q.3.** 对 segment $[\rho,\rho\nu]$，说明它的 degree 是 $2\deg\rho$。

**解答.** 若 $\rho$ 是 $G_m=\operatorname{GL}_m(F)$ 的 cuspidal representation，则 $\deg\rho=m$。Segment $[\rho,\rho\nu]$ 含两个 cuspidal twists，均属于 $G_m$。因此该 segment 对应的总 degree 为
$$
m+m=2m=2\deg\rho.
$$
$\square$

**练习 Q.5.** 解释非分歧主级数的 Satake 参数和 Weil-Deligne 参数为何给出同一 Euler 因子。

**解答.** 非分歧主级数由非分歧 characters $\chi_i$ 给出。Satake 参数记录数值
$$
\alpha_i=\chi_i(\varpi).
$$
局部类域论把 $\chi_i$ 变成 Weil 群的非分歧 character，其在几何 Frobenius 上的值同为 $\alpha_i$。因此 Weil-Deligne 参数的 Frobenius 半单特征值与 Satake 参数一致，标准 Euler 因子两种写法都为
$$
\prod_i(1-\alpha_iq^{-s})^{-1}.
$$
$\square$

## 附录 R

**练习 R.1.** 在紧商情形，从核 $K_f(x,y)$ 推导几何侧 orbital integral。

**解答.** 紧商时
$$
\operatorname{tr}R(f)=\int_{[G]}K_f(x,x)\,dx
=\int_{[G]}\sum_{\gamma\in G(K)}f(x^{-1}\gamma x)\,dx.
$$
按 $G(K)$-共轭类分组。固定 $\gamma$ 后，共轭类求和等价于 $G_\gamma(K)\backslash G(K)$ 上求和，故该部分为
$$
\int_{G_\gamma(K)\backslash G(\mathbb A)}f(x^{-1}\gamma x)\,dx.
$$
再用 Weil 积分公式分解为
$$
\operatorname{vol}(G_\gamma(K)\backslash G_\gamma(\mathbb A))
\int_{G_\gamma(\mathbb A)\backslash G(\mathbb A)}
f(x^{-1}\gamma x)\,dx.
$$
后一个积分就是 orbital integral。$\square$

**练习 R.3.** 说明 ordinary orbital integral 与 stable orbital integral 的差别。

**解答.** Ordinary orbital integral 固定一个 $G(F)$-共轭类：
$$
O_\gamma(f)=\int_{G_\gamma(F)\backslash G(F)}f(x^{-1}\gamma x)\,dx.
$$
Stable orbital integral 则把同一个 stable conjugacy class 内的若干 $G(F)$-共轭类按符号或 transfer factor 归一化加总。Endoscopy 比较的是 stable distributions；因此单个 ordinary orbital integral 通常不是正确的不变量，必须组合成 stable orbital integral。$\square$

**练习 R.5.** 把 base change 的 trace formula 证明框架拆成三步。

**解答.** 第一步，构造源群和目标群上的匹配测试函数，通常一侧为 twisted orbital integrals，另一侧为普通 orbital integrals。第二步，应用 twisted trace formula 和 ordinary trace formula，把几何侧匹配转化为谱侧分布恒等式。第三步，利用 characters 或 pseudo-coefficients 的线性无关性，从谱侧分布恒等式中分离出表示，得到 base change lift，并检查非分歧位置的参数限制公式。$\square$

## 附录 S

**练习 S.1.** 说明 $\operatorname{Bun}_G(\mathbb F_q)$ 与 adelic 双商的关系。

**解答.** 对 $G$-bundle 选择泛点平凡化，可在每个闭点得到局部粘合数据，即 $G(K_x)$ 中的元素。几乎所有闭点处 bundle 可由 $G(\mathcal O_x)$ 平凡化，所以得到一个 adele 元素。改变泛点平凡化对应左乘 $G(K)$，改变局部平凡化对应右乘 $G(\mathcal O_{\mathbb A})$。因此得到
$$
\operatorname{Bun}_G(\mathbb F_q)
\sim
G(K)\backslash G(\mathbb A_K)/G(\mathcal O_{\mathbb A})
$$
的 groupoid 形式对应；严格计数时需保留 automorphism group。$\square$

**练习 S.3.** 说明 Lafforgue 定理中 Satake polynomial 与 Frobenius characteristic polynomial 的相容性。

**解答.** 若 $\pi$ 是 $\operatorname{GL}_n(\mathbb A_K)$ 的 cuspidal automorphic representation，Lafforgue 对应给出 $\ell$-adic 表示 $\rho_\pi$。在 $\pi$ 与 $\rho_\pi$ 均非分歧的闭点 $v$，$\pi_v$ 有 Satake 参数
$$
\operatorname{diag}(\alpha_{1,v},\ldots,\alpha_{n,v}).
$$
相容性要求几何 Frobenius 或算术 Frobenius按约定调整后，$\rho_\pi(\operatorname{Frob}_v)$ 的 characteristic polynomial 为
$$
\prod_i(1-\alpha_{i,v}T).
$$
因此局部 Hecke L 因子和 Galois L 因子相同。$\square$

**练习 S.4.** 解释 excursion operators 为什么需要多个 Galois 元素。

**解答.** 对一般对偶群 $\widehat G$，单个元素在某个固定表示下的 trace 不足以恢复半单共轭类，更不足以记录多个 Galois 元素之间的乘法关系。Excursion operators 允许取有限集合 $I$、多个 Galois 元素 $(\gamma_i)$ 和任意 invariant function
$$
f\in\mathcal O(\widehat G\backslash\widehat G^I/\widehat G).
$$
这些函数同时检测多个元素的相对位置。由 invariant theory，它们能刻画 semisimple global Langlands parameter 的 $\widehat G$-共轭类。$\square$

## 附录 T

**练习 T.1.** 从 trace 和 determinant 推导定理 T.8 的 Euler 因子公式。

**解答.** 对 $p\nmid N\ell$，定理 T.8 给出
$$
\operatorname{tr}\rho_{f,\ell}(\operatorname{Frob}^{\operatorname{arith}}_p)=a_p,\qquad
\det\rho_{f,\ell}(\operatorname{Frob}^{\operatorname{arith}}_p)=\varepsilon(p)p^{k-1}.
$$
二维矩阵的 characteristic polynomial 为
$$
X^2-a_pX+\varepsilon(p)p^{k-1}.
$$
因此
$$
\det(1-\rho_{f,\ell}(\operatorname{Frob}^{\operatorname{arith}}_p)p^{-s})
=1-a_pp^{-s}+\varepsilon(p)p^{k-1}p^{-2s},
$$
取倒数即得 Euler 因子。$\square$

**练习 T.3.** 说明 residual representation 为什么只在半单化意义下与稳定格无关。

**解答.** 同一个 $\ell$-adic 表示可能有多个 Galois 稳定格。不同稳定格的模 $\lambda$ 约化不必同构，因为扩张类可能改变。但 Brauer-Nesbitt 定理说明，若两个约化表示来自同一个 characteristic-zero 表示，则它们的半单化由几乎所有 Frobenius characteristic polynomials 决定，因此相同。所以 residual representation 通常定义为约化后的半单化。$\square$

**练习 T.5.** 说明 Deligne 表示在费马应用链中的最小使用范围。

**解答.** 费马应用只需要从权 $2$ newform 得到二维 Galois 表示，并在好素数处满足 trace 等于 Hecke eigenvalue；再取稳定格约化得到 residual representation。Ribet 降层比较的是这些 residual representations 的 ramification 和导子。应用章不需要 Deligne 构造的完整 cohomological proof，也不需要一般权或一般 level 的全部精细性质。$\square$

## 附录 U

**练习 U.2.** 说明 Shimura variety cohomology 为什么同时携带 Hecke 和 Galois 作用。

**解答.** Shimura variety 的 canonical model 定义在 reflex field $E$ 上，因此其 $\ell$-adic cohomology 自然带有 $\operatorname{Gal}(\overline E/E)$ 作用。另一方面，改变有限 adelic level 的 correspondences 给出 Hecke operators，它们作为代数 correspondences 作用在同一 cohomology 上。这两类作用都来自定义在数域上的几何 correspondences 和 functoriality，因此彼此交换。$\square$

**练习 U.4.** 说明局部-整体相容在非分歧位置退化为 Satake-Frobenius 相容。

**解答.** 若 $v\nmid\ell$ 且 $\pi_v$ 非分歧，则 $\pi_v$ 的局部 Langlands 参数由 Satake parameter 给出，惯性作用平凡且 monodromy 为 $0$。局部-整体相容要求 $r_{\ell,\iota}(\pi)|_{G_{K_v}}$ 的 Weil-Deligne 参数等于该局部参数。于是该条件只剩下 Frobenius 半单共轭类相等，也就是 Galois 表示的 Frobenius characteristic polynomial 等于自守侧 Satake polynomial。$\square$

**练习 U.5.** 解释 p-adic Hodge 条件在 automorphy lifting 中的角色。

**解答.** Automorphy lifting 比较的是满足指定局部条件的 deformation ring 和 Hecke algebra。在 $v\mid\ell$ 处，局部条件不能只用 ramification 描述，而要指定 de Rham、crystalline、ordinary 或 potentially semistable 等 p-adic Hodge 类型。这些条件定义局部变形环，并控制其维数、不可约分支和与 automorphic local factors 的相容性。没有这些局部 p-adic Hodge 条件，patching 中的全局变形问题不会有正确的几何大小。$\square$

## 附录 V

**练习 V.1.** 用局部 reciprocity 计算非分歧 character 的 Frobenius 值。

**解答.** 设 $\chi:F^\times\to\mathbb C^\times$ 非分歧。局部 reciprocity 采用几何 Frobenius 归一化：
$$
\operatorname{rec}_F(\varpi)=\operatorname{Fr}_F.
$$
对应 Weil character 为
$$
\rho_\chi=\chi\circ\operatorname{rec}_F^{-1}.
$$
因此
$$
\rho_\chi(\operatorname{Fr}_F)=\chi(\varpi).
$$
$\square$

**练习 V.3.** 对 $K=\mathbb Q$，把 Dirichlet characters 写成 finite idele class characters。

**解答.** 取精确导子为 $N$ 的 primitive Dirichlet character
$\chi:(\mathbb Z/N\mathbb Z)^\times\to\mathbb C^\times$，并记其在 finite units 上的提升为
$$
\widehat\chi:
\widehat{\mathbb Z}^\times\to
(\mathbb Z/N\mathbb Z)^\times
\xrightarrow{\chi}\mathbb C^\times.
$$
用唯一分解
$\mathbb A_\mathbb Q^\times=\mathbb Q^\times\mathbb R_{>0}\widehat{\mathbb Z}^\times$
定义
$$
\omega_\chi(qru)=\widehat\chi(u)^{-1}.
$$
它对 $\mathbb Q^\times$ 平凡且有限阶，因而给出
$$
\omega_\chi:
\mathbb Q^\times\backslash\mathbb A_\mathbb Q^\times
\longrightarrow\mathbb C^\times.
$$
若 $\chi(-1)=(-1)^{a_\chi}$，则
$\omega_{\chi,\infty}=\operatorname{sgn}^{a_\chi}$；对 $p\nmid N$，
$\omega_{\chi,p}$ 非分歧且 $\omega_{\chi,p}(p)=\chi(p)$。因此本书的 Euler convention 给出
$L(s,\omega_\chi)=L(s,\chi)$。若在 finite units 上不取逆，则所得是 $L(s,\overline\chi)$ 的 convention；imprimitive modulus 的额外删因子与相应 primitive Hecke L 函数也必须分开。$\square$

**练习 V.5.** 说明为什么一般 Hecke quasi-character 不应写成 profinite Galois character。

**解答.** Profinite Galois group 的连续复 character 有紧像；在有限阶 Artin 情形中像为有限群。一般 Hecke quasi-character 可含
$$
|\cdot|^s:C_K\to\mathbb C^\times
$$
这样的非有限阶非紧方向。它自然是 idele class group 或 Weil group 的 character，但不是普通 profinite Galois group 的连续复 character。因此 `GL(1)` 的一般 quasi-character 必须放在 Weil 侧。$\square$

## 附录 W

**练习 W.1.** 用 genus formula 重新计算 $X_0(2)$ 的 genus。

**解答.** 对 $\Gamma_0(2)$，
$$
\mu=3,\qquad c=2,\qquad e_2=1,\qquad e_3=0.
$$
代入
$$
g=1+\frac{\mu}{12}-\frac{e_2}{4}-\frac{e_3}{3}-\frac c2
$$
得
$$
g=1+\frac14-\frac14-1=0.
$$
$\square$

**练习 W.4.** 说明 oldforms 为什么不改变几乎所有好素数 Hecke eigenvalues。

**解答.** Oldform 来自低级 $M\mid N$ 的 form 通过 degeneracy maps 拉回。若 $p\nmid N$，Hecke correspondence $T_p$ 与这些 degeneracy maps 在 moduli 解释下交换，因为 degree $p$ isogeny 与改变 level 的操作发生在互素 level 上。因此 oldform 在所有 $p\nmid N$ 的 Hecke eigenvalue 与原低级 newform 相同。Old/new 差别反映在坏素数和最小 level，即导子处。$\square$

**练习 W.5.** 用 W.19 重述费马应用中的最终矛盾。

**解答.** Frey 曲线、半稳定模性和 Ribet 降层给出一个权 $2$、级 $2$ newform，即
$$
S_2(\Gamma_0(2))_{\operatorname{new}}\ne0.
$$
但 W.9 证明
$$
S_2(\Gamma_0(2))=0.
$$
由 W.15，new subspace 也为 $0$。这与降层结论矛盾，因此原 Fermat 反例不存在。$\square$

## 附录 X

**练习 X.1.** 对 $\psi=(\pi,1)$，说明 Arthur 参数何时是 tempered。

**解答.** 参数 $\psi=(\pi,1)$ 中第二个分量 $1$ 表示 Arthur $\operatorname{SL}_2(\mathbb C)$ 只取一维平凡表示。因此它不引入 $|w|$ 的非零幂。若 $\pi$ 本身对应 tempered cuspidal datum，则局部化后的 Arthur 参数给出的 Langlands 参数在 Weil 群上 bounded modulo center，所以是 tempered。非 tempered 现象在这种记号中来自某个 $b_i>1$。$\square$

**练习 X.2.** 解释 component group character 为什么进入 multiplicity formula。

**解答.** 同一个 Arthur parameter 通常给出一个 packet，而不是单个表示。Packet 内成员由 component group 的 characters 区分。Arthur multiplicity formula 要决定 restricted tensor product
$$
\pi=\otimes_v'\pi_v
$$
是否出现在离散谱中，以及重数是多少。这个判定由各局部 component group characters 的乘积与 global Arthur sign character 比较给出。因此没有 component group character，就无法从 packet 集合层面计算 multiplicity。$\square$

**练习 X.3.** 写出 standard transfer 在非分歧 Satake 参数上的公式。

**解答.** 若 $\pi_v$ 是 $G(K_v)$ 的 spherical member，Satake 参数为
$$
s_v\rtimes\operatorname{Fr}_v\in{}^LG.
$$
标准转移到 $\operatorname{GL}_N$ 后，非分歧 Satake 参数为
$$
\operatorname{Std}(s_v\rtimes\operatorname{Fr}_v)
\in \operatorname{GL}_N(\mathbb C).
$$
因此标准 L 因子满足
$$
L(s,\pi_v,\operatorname{Std})
=
\det(1-\operatorname{Std}(s_v)q_v^{-s})^{-1},
$$
这等于转移后的 $\operatorname{GL}_N$ 标准局部因子。$\square$

## 附录 Y

**练习 Y.1.** 解释 Ran space 为什么适合记录多点 Hecke 修改。

**解答.** Hecke 修改可以发生在曲线的任意有限多个点上，而点数本身也可变化。Ran space 的点正是非空有限子集，因此它同时记录“修改发生在哪里”和“有多少个修改点”。当有限点集分裂成互不相交的两部分时，factorization 结构表达这些修改彼此独立。$\square$

**练习 Y.3.** 解释 fusion 如何给出 convolution 的交换约束。

**解答.** 两个 Hecke 修改在不同点时作用在 disjoint formal discs 上，因此交换次序给出同构。让两个点在 $X^2$ 中沿对角线相碰，BD Grassmannian 的 nearby cycles 把“不同点处的外积”专化为单点 affine Grassmannian 上的 convolution。不同点处的交换同构专化后成为 convolution product 的交换约束。$\square$

**练习 Y.5.** 说明 Hecke eigensheaf 条件为什么必须对所有 $V\in\operatorname{Rep}(\widehat G)$ 张量相容。

**解答.** 一个 $\widehat G$-local system 等价于张量函子
$$
\operatorname{Rep}(\widehat G)\to\operatorname{Loc}(X).
$$
若只给单个表示 $V$ 的 Hecke 本征同构，只能得到 associated local system $V_{\mathcal E}$，不能恢复主 $\widehat G$-local system。要求所有 $V$ 的本征同构并与 $V\otimes W$、直和、对偶相容，正是保证这些 associated local systems 来自同一个 tensor functor。$\square$

## 附录 Z

**练习 Z.1.** 证明 $\pi(f)$ 的像落在某个开紧不变量空间中。

**解答.** 因为 $f\in C_c^\infty(G)$，存在开紧子群 $J$ 使 $f$ 左 $J$-不变。对任意 $j\in J$，
$$
\pi(j)\pi(f)v
=\int_G f(g)\pi(jg)v\,dg
=\int_G f(j^{-1}g')\pi(g')v\,dg'
=\pi(f)v.
$$
所以 $\pi(f)v\in\pi^J$。若 $\pi$ admissible，则 $\pi^J$ 有限维。$\square$

**练习 Z.3.** 说明 character distribution 为什么是 trace formula 谱侧的局部输入。

**解答.** Trace formula 的谱侧包含局部测试函数在局部表示上的 trace。对 admissible representation $\pi$，该 trace 是
$$
\Theta_\pi(f)=\operatorname{tr}\pi(f).
$$
Harish-Chandra character theorem 保证 $\Theta_\pi$ 是良定义 invariant distribution，并在 regular semisimple locus 上由函数表示。这样谱侧才能作为分布与几何侧 orbital integrals 比较。$\square$

**练习 Z.5.** 说明 Paley-Wiener theorem 在构造测试函数时的作用。

**解答.** 在 trace formula 应用中，常要选择局部测试函数，使它在某些表示上 trace 非零，在其他 Bernstein components 上消失或满足指定标量。Local Paley-Wiener theorem 描述哪些 spectral-side functions 来自 compactly supported smooth functions。因此它提供从“想要的谱侧过滤条件”反推“存在局部测试函数”的定理依据。$\square$

## 附录 AA

**练习 AA.1.** 对 $\operatorname{GL}_n$，说明 hyperspecial subgroup 的 integral model 来源。

**解答.** 取
$$
\mathcal G=\operatorname{GL}_{n,\mathcal O_F}.
$$
这是 $\mathcal O_F$ 上的 reductive group scheme，generic fiber 为 $\operatorname{GL}_{n,F}$，特殊纤维为 $\operatorname{GL}_{n,k_F}$。因此
$$
\mathcal G(\mathcal O_F)=\operatorname{GL}_n(\mathcal O_F)
$$
是 hyperspecial maximal compact subgroup。$\square$

**练习 AA.3.** 用 Cartan decomposition 说明球 Hecke 代数有双陪集基。

**解答.** Cartan decomposition 给出
$$
G(F)=\bigsqcup_{\lambda\in X_*(T)^+}K\lambda(\varpi)K.
$$
球 Hecke 代数由 compactly supported bi-$K$-invariant functions 组成，因此每个函数在这些双陪集上常值，且只支持有限多个双陪集。故特征函数
$$
\mathbf 1_{K\lambda(\varpi)K}
$$
构成向量空间基。$\square$

**练习 AA.4.** 说明 spherical representation 的 depth 为 $0$。

**解答.** 若 $\pi$ spherical，则存在 hyperspecial vertex $x$ 对应的 $K=G(F)_{x,0}$，使
$$
\pi^K\ne0.
$$
Moy-Prasad filtration 满足 $G(F)_{x,0+}\subset K$，所以同一非零向量也被 $G(F)_{x,0+}$ 固定。Depth 非负，故 $\operatorname{depth}(\pi)=0$。$\square$

## 附录 AB

**练习 AB.1.** 说明 derived structure 如何记录 obstruction data。

**解答.** 对 derived moduli problem $X$，点 $x\in X$ 处的 tangent complex
$$
T_xX=\operatorname{RHom}(x^*L_X,k)
$$
不是只在次数 $0$ 有同调。其 $H^0$ 给出一阶变形，负次数或正次数的相应 convention 给出 automorphisms 与 obstruction groups。若只取 classical truncation $t_0X$，则 cotangent complex 被截断，非平凡 obstruction class 只能表现为 classical moduli 的奇异性，而不再作为 functorial complex 参与 pullback、base change 和 singular support。因此 derived structure 以 cotangent complex 的形式保存 obstruction data。$\square$

**练习 AB.2.** 解释 smooth stack 情形下 QCoh 与 IndCoh 的关系。

**解答.** 若 $X$ smooth，则 dualizing object $\omega_X$ 是可逆的移位 line object。Gaitsgory 的比较函子
$$
\Upsilon_X:\operatorname{QCoh}(X)\to\operatorname{IndCoh}(X),
\qquad
\mathcal F\mapsto \mathcal F\otimes\omega_X
$$
在这种情形是等价。几何原因是 smoothness 使 !-pullback 与 $*$-pullback 只差相对维数移位和 line twist，coherent sheaves 的 ind-completion 不产生额外 singular directions。Singular stack 中该结论失效，差异由 singularity stack 记录。$\square$

**练习 AB.3.** 把 Hecke functor 写成 correspondence kernel functor。

**解答.** Hecke stack 给出 correspondence
$$
\operatorname{Bun}_G\xleftarrow{h_1}\operatorname{Hecke}_G\xrightarrow{h_2}\operatorname{Bun}_G\times X.
$$
对 $V\in\operatorname{Rep}(\widehat G)$，几何 Satake 给出 Hecke correspondence 上的 kernel $\mathcal S_V$。在选定 sheaf theory 中，kernel functor 为
$$
\Phi_{\mathcal S_V}(\mathcal F)
=h_{2,!}(h_1^!\mathcal F\otimes\mathcal S_V)
$$
或在相应有限性条件下使用 $h_{2,*}$ 的版本。该函子正是第二十章使用的
$$
\mathsf H_V:\mathcal D(\operatorname{Bun}_G)\to \mathcal D(\operatorname{Bun}_G\times X).
$$
几何 Satake 的 convolution tensor structure 保证 $V\mapsto\mathsf H_V$ 是张量作用。$\square$

**练习 AB.4.** 说明 $\operatorname{Bun}_G$ 非 quasi-compact 对 D-module category 的影响。

**解答.** $\operatorname{Bun}_G$ 按 Harder-Narasimhan type 分层，通常含有无限多个 quasi-compact substacks。因此普通 D-module category 的 compact generation、连续 functor 的定义以及 $!$-extension 的存在性不能直接由 finite type stack 的定理推出，而需要 truncatable/co-truncative 理论。Eisenstein series 和 constant term 又需要沿非 proper correspondence 做推拉操作。与此同时，改变 compact generators 可定义另一个
$\operatorname{DMod}_{1/2}(\operatorname{Bun}_G)_{\operatorname{ren}}$，并与普通范畴由
$\operatorname{ren}\dashv\operatorname{un\text{-}ren}$ 比较。主几何 Langlands 公式使用普通范畴指向 $\operatorname{IndCoh}_{\mathcal N}$；renormalized companion 指向全部 $\operatorname{IndCoh}$。非 quasi-compact 性说明两类构造为何出现，但不给出两范畴的识别。$\square$

**练习 AB.5.** 解释 nilpotent singular support 在谱侧的作用。

**解答.** 谱侧 $\operatorname{LocSys}_{\widehat G}(X)$ 一般是 derived singular stack。全部 $\operatorname{IndCoh}$ 允许任意 singular directions，范畴会过大；$\operatorname{QCoh}$ 又忽略一部分 !-functorial behavior。Nilpotent singular support 条件在
$$
\operatorname{Sing}(\operatorname{LocSys}_{\widehat G}(X))
$$
中选出由 nilpotent cone 控制的闭锥，定义
$$
\operatorname{IndCoh}_{\mathcal N}(\operatorname{LocSys}_{\widehat G}(X)).
$$
该条件与 Eisenstein/constant term、Arthur 型非离散现象和自动侧的连续谱相匹配，因此是谱侧大小控制条件。$\square$

## 附录 AC

**练习 AC.1.** 在 AC.4 的完备代数闭设定下，说明 Fargues-Fontaine 曲线如何把 Newton slope 几何化。

**解答.** 固定 AC.4 的完备代数闭非 Archimedean 扩张 $C/F$、其 tilt $C^\flat$ 与曲线 $X_{FF}=X_{C^\flat,F}$。Isocrystal 带有 Frobenius slope decomposition，其数值由 Newton polygon 记录。Fargues-Fontaine construction 把 isocrystal 送到 $X_{FF}$ 上的 vector bundle；slope 为 $\lambda$ 的 isocrystal 分量对应 semistable vector bundle $\mathcal O(\lambda)$ 型分量。因此 isocrystal 的 Newton slope 在几何侧变为 vector bundle 的 Harder-Narasimhan slope，Newton polygon 变为 Harder-Narasimhan polygon。若基底未代数闭，这个分解须先在完备代数闭扩张上取得并附加 descent datum。$\square$

**练习 AC.2.** 在 AC.4 的完备代数闭设定下，解释 $B(G)$ 与 $G$-bundles 同构类的关系，并说明一般 perfectoid 基底为何还需要 descent 数据。

**解答.** 固定 AC.4 的完备代数闭几何点。对局部域 $F$ 上的 connected reductive group $G$，Kottwitz 集合 $B(G)$ 分类 $G$-isocrystals up to $\sigma$-conjugacy。Fargues-Fontaine 理论给出同构类的双射
$$
b\in B(G)\longmapsto \mathcal E_b.
$$
Newton point 给出 $\mathcal E_b$ 的 Harder-Narasimhan type，Kottwitz invariant 给出连通分量数据。Basic 元素对应 semistable $G$-bundles，其 automorphism group 是内形式 $J_b(F)$。对一般 perfectoid 基底，$\operatorname{Bun}_G(S)$ 还记录 families 与 v-descent 数据，不能直接等同于集合 $B(G)$。$\square$

**练习 AC.3.** 说明 local Shimura variety cohomology 为什么同时有 $G(F)$、$J_b(F)$ 和 $W_{E_\mu}$ 作用，并解释 $E_\mu$ 为什么由 $\mu$ 而不是由 $b$ 定义。

**解答.** 按 AC.10 的归一化，$b\in B(G,\mu)$ 且 local Shimura diamond
$\operatorname{Sht}_{G,b,\mu}/\operatorname{Spd}\breve E_\mu$ 参数化
$$
\mathcal E_b\dashrightarrow\mathcal E_1
$$
的 $\mu$-有界 modifications。改变平凡目标丛的 level trivialization 给出 $G(F)$ 作用，改变源丛 $\mathcal E_b$ 的 self-quasi-isogeny 给出 $J_b(F)$ 作用。$\mu$ 的共轭类的定义域是 reflex field $E_\mu$，因而相对 $E_\mu$ 的 Weil descent datum，而非 $b$ 本身，在几何纤维上给出 $W_{E_\mu}$ 作用。三个作用两两交换，于是对含 $\breve E_\mu$ 的完备代数闭扩张 $C$ 有
$$
R\Gamma_c(\operatorname{Sht}_{G,b,\mu,C},\mathcal L_\xi)
$$
自然成为 $G(F)\times J_b(F)\times W_{E_\mu}$ 的表示对象。$\square$

**练习 AC.4.** 解释 Fargues-Scholze 结果为什么主要给出 semisimple 参数化。

**解答.** Fargues-Scholze 通过 stack of Langlands parameters 与 $\operatorname{Bun}_G(X_{FF})$ 上 sheaves 的 spectral action 构造不可约光滑表示的参数。该 construction 捕捉的是参数的半单谱支撑，足以得到广义的 semisimple LLC map。完整 LLC 还需要 monodromy、component group representations、Whittaker normalization、inner twist normalization 和 endoscopic character identities 等附加结构。因此该结果是局部 Langlands 的强几何框架，但不自动给出猜想 12.19 的全部增强数据。$\square$

**练习 AC.5.** 比较全局几何 Langlands 与 Fargues-Fontaine 几何局部 Langlands 的曲线和谱侧。

**解答.** 全局几何 Langlands 的曲线是代数闭域或有限域上的光滑射影曲线 $X$；自动侧是 $\operatorname{Bun}_G(X)$ 上的 sheaves，谱侧是 $\widehat G$-local systems on $X$。Fargues-Fontaine 几何局部 Langlands 的曲线是由 $p$-adic 局部域构造的 $X_{FF}$；自动侧是 $X_{FF}$ 上 $G$-bundles 的 stack 及其 sheaves，谱侧是局部 Weil group 或 L 群参数的 stack。二者都把 Hecke 修改和谱作用作为核心结构，但全局理论编码函数域或代数曲线上的全局局部系统，局部理论编码单个 $p$-adic 域的局部参数。$\square$

## 附录 AD

**练习 AD.1.** 用变量变换权重解释为什么判别式按 $u^{-12}$ 缩放。

**解答.** Weierstrass 变换
$$
x=u^2x'+r,\qquad y=u^3y'+u^2sx'+t
$$
把 $x$ 赋予权重 $2$，把 $y$ 赋予权重 $3$。判别式是三次多项式根差平方乘积的相对不变量；对短方程
$$
y^2=(x-e_1)(x-e_2)(x-e_3),
$$
若 $x=u^2x'$，则每个根差乘以 $u^2$，三个根差平方总共乘以 $u^{12}$。因此旧判别式满足 $\Delta=u^{12}\Delta'$，即新方程判别式为 $\Delta'=u^{-12}\Delta$。长 Weierstrass 方程的标准不变量给出同一权重结论。$\square$

**练习 AD.2.** 用 Kodaira 表说明半稳定椭圆曲线的导子是坏乘法素数的一次乘积。

**解答.** 半稳定表示每个局部约化类型只可能是 $I_0$ 或 $I_n$。Kodaira 表中 $I_0$ 为好约化，导子指数 $f=0$；$I_n$，$n\ge1$，为乘法约化，导子指数 $f=1$。因此全局导子
$$
N_E=\prod_p p^{f_p(E)}
$$
只在坏乘法素数处出现，且每个这样的素数指数为 $1$。$\square$

**练习 AD.3.** 说明 split multiplicative 与 nonsplit multiplicative 的局部 L 因子为什么只差一个符号。

**解答.** Split multiplicative reduction 由 Tate curve 给出，$V_\ell(E)^{I_F}$ 上 Frobenius 的有效特征值为 $1$，故局部因子为
$$
(1-q^{-s})^{-1}.
$$
Nonsplit multiplicative reduction 在一个非分歧二次扩张后变为 split；相应局部表示是 split 情形再张量非平凡 unramified quadratic character。该 character 在 Frobenius 上取值 $-1$，所以特征值由 $1$ 变为 $-1$，局部因子变为
$$
(1+q^{-s})^{-1}.
$$
$\square$

**练习 AD.4.** 对 Frey 曲线 $y^2=x(x-a^p)(x+b^p)$ 计算三根差并推出判别式。

**解答.** 三根为
$$
0,\qquad a^p,\qquad -b^p.
$$
根差为
$$
0-a^p=-a^p,\qquad 0-(-b^p)=b^p,\qquad a^p-(-b^p)=a^p+b^p=c^p.
$$
对三次方程 $y^2=\prod_i(x-e_i)$，判别式为
$$
16\prod_{i<j}(e_i-e_j)^2.
$$
所以
$$
\Delta=16a^{2p}b^{2p}c^{2p}.
$$
$\square$

**练习 AD.5.** 解释为什么 $v_q(\Delta_E)$ 被 $p$ 整除会导致 residual conductor 可能小于 $E$ 的 conductor。

**解答.** 在乘法约化处，$E$ 局部上由 Tate curve 描述，扩张类由 Tate parameter $q_E$ 控制，且
$$
v(q_E)=v_q(\Delta_E).
$$
$p$-torsion 表示的惯性作用由该 Tate parameter 在 $F^\times/(F^\times)^p$ 中的类控制。若 $v_q(\Delta_E)$ 被 $p$ 整除，则 valuation 部分在模 $p$ 后消失；在 Frey 曲线的局部情形中，剩余单位部分也满足降层所需条件，于是模 $p$ 表示的 ramification 比 $\ell$-adic lift 更小。故曲线本身的 conductor 在 $q$ 处有指数 $1$，但 residual conductor 可以不含 $q$。$\square$

## 附录 AE

**练习 AE.1.** 设 $\chi_1,\chi_2$ 非分歧，计算 principal series 的 Satake 参数和 L 因子。

**解答.** 非分歧 character 由 $\chi_i(\varpi)=\alpha_i$ 决定。归一化 principal series $I(\chi_1,\chi_2)$ 的 Satake 参数为
$$
\operatorname{diag}(\alpha_1,\alpha_2)
$$
在 $\operatorname{GL}_2(\mathbb C)$ 中的共轭类。对应 Weil-Deligne 参数为
$$
\varphi_{\chi_1}\oplus\varphi_{\chi_2},\qquad N=0.
$$
因此标准 L 因子为
$$
L(s,I(\chi_1,\chi_2))
=
(1-\alpha_1q^{-s})^{-1}(1-\alpha_2q^{-s})^{-1}.
$$
$\square$

**练习 AE.2.** 解释 Steinberg 表示为什么需要非零 monodromy $N$。

**解答.** Steinberg 表示出现在 principal series 的 reducibility point，此时两个 inducing characters 相差 $|\cdot|$。若只取半单直和参数，就不能区分 Steinberg factor 和同一半简化下的一维 quotient/character 型对象。Weil-Deligne 参数中的 nilpotent operator
$$
N\ne0
$$
把两个相邻 character 连成一个 indecomposable Jordan block，并满足
$$
r(w)Nr(w)^{-1}=|w|N.
$$
这个非零 monodromy 正是 Steinberg 表示本质平方可积和 conductor 指数 $1$ 的参数侧记录。$\square$

**练习 AE.3.** 说明 supercuspidal 参数为什么不能是两个 character 的直和。

**解答.** 两个 character 的直和参数
$$
\varphi=\chi_1\oplus\chi_2
$$
来自 Levi subgroup $\operatorname{GL}_1\times\operatorname{GL}_1$。LLC 与 parabolic induction 相容，因此对应表示应属于 Borel parabolic induction 的 subquotient，即 principal series 或其极限情形。Supercuspidal 的定义正是“不作为 proper parabolic induction 的 subquotient 出现”。所以 supercuspidal 参数不能是两个 character 的直和，而应为不可约二维 Weil 表示。$\square$

**练习 AE.4.** 比较乘法约化椭圆曲线与 Steinberg twist 的共同特征。

**解答.** 乘法约化椭圆曲线由 Tate curve 描述，其 Tate module 给出带非平凡扩张结构的 Weil-Deligne 参数。该参数半简化可约，但 monodromy $N$ 非零。Steinberg twist 的参数也正是可约 Weil 半简化加非零 $N$ 的 special parameter。因此乘法约化在自守侧对应 Steinberg twist；split/nonsplit 的差异由一个非分歧 quadratic character 的符号记录。$\square$

**练习 AE.5.** 用 AE.13 解释 `GL(2)` LLC 比 `GL(1)` 多出的现象。

**解答.** `GL(1)` LLC 只有 characters 与一维 Weil 参数，既没有 parabolic induction，也没有 monodromy block 或 supercuspidal 二维不可约参数。`GL(2)` 已出现三种新现象：两个 character 直和对应 principal series；两个相邻 character 可形成带非零 $N$ 的 Steinberg 参数；不可约二维 Weil 表示对应 supercuspidal。它们分别反映 induction、monodromy 和 genuinely two-dimensional Galois/Weil 参数，是高阶 Langlands 理论的最小模型。$\square$
