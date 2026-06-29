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

**解答.** 设 $\chi:(\mathbb Z/N\mathbb Z)^\times\to\mathbb C^\times$ 为 Dirichlet 特征。定义有限 adele 单位上的特征
$$
\chi_f:\widehat{\mathbb Z}^\times\to\mathbb C^\times,\qquad
u=(u_p)_p\mapsto \chi(u\bmod N).
$$
由中国剩余定理，$u\bmod N$ 是良定义的。把任意 $x\in\mathbb A_{\mathbb Q,f}^\times$ 写成
$$
x=q\cdot u\cdot r_\infty
$$
的等价类时，更标准的定义是在
$$
C_\mathbb Q=\mathbb Q^\times\backslash\mathbb A_\mathbb Q^\times
$$
上令无穷处分量平凡、有限单位部分按上式作用。局部地，若 $p\nmid N$，则 $\chi_p$ 在 $\mathbb Z_p^\times$ 上平凡；若 $p\mid N$，则 $\chi_p$ 由 $(\mathbb Z_p/N\mathbb Z_p)^\times$ 上的相应因子给出。该 Hecke 特征有限阶，因为其像包含在有限群 $\chi((\mathbb Z/N\mathbb Z)^\times)$ 中。$\square$

**练习 2.3.** 平凡 Hecke 特征的完成 L 函数为何允许在 $s=0,1$ 出现极点。

**解答.** 平凡 Hecke 特征对应 Dedekind zeta 函数 $\zeta_K(s)$。Tate thesis 的整体 zeta 积分在平凡特征情形含有来自 $K^\times\backslash\mathbb A_K^\times$ 体积方向的常数项。Poisson summation 把 $s$ 与 $1-s$ 联系起来；若在 $s=1$ 有简单极点，则函数方程强制在 $s=0$ 出现对应极点或余项。非平凡酉特征没有同样的全局常数项贡献，因此完成 L 函数通常为 entire。$\square$

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

**练习 3.3.** 解释全局类域论同构为何与局部 Artin 映射相容。

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
