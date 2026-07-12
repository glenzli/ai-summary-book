# 第七章：adelic 模形式与 `GL(2)` 自守表示

## 本章目标

本章把第六章的经典模形式翻译为 adelic 语言，并定义 `GL(2)` 的自守形式、自守表示、尖点条件、局部分解和标准 L 函数。核心目标是区分三类对象：经典函数 $f:\mathfrak H\to\mathbb C$、adelic 函数 $\Phi:\operatorname{GL}_2(\mathbb Q)\backslash\operatorname{GL}_2(\mathbb A_\mathbb Q)\to\mathbb C$、以及右正则表示生成的不可约自守表示 $\pi_f$。

## 依赖前置知识

需要第一章的 adeles、第四章的光滑表示、第六章的经典模形式和 Hecke 算子。Strong approximation、经典-adelic 对应和 newform theory 在本章作为外部输入使用。

收口归一化回指：本章是 classical normalization 与 automorphic normalization 的主要转换点；Satake roots、unitary normalization、标准 L 函数和平移公式见 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 第 4、5、8 节。

## 7.1 `GL(2)` 的 adelic 商

本章固定
$$
G=\operatorname{GL}_2,\qquad Z=\text{center of }G.
$$
对任意环 $R$，有
$$
G(R)=\operatorname{GL}_2(R),\qquad Z(R)=R^\times\cdot I_2.
$$

**定义 7.1.** `GL(2)` 的 adelic automorphic quotient 是商空间
$$
G(\mathbb Q)\backslash G(\mathbb A_\mathbb Q).
$$
若固定中心特征 $\omega:\mathbb A_\mathbb Q^\times/\mathbb Q^\times\to\mathbb C^\times$，则考虑函数 $\Phi:G(\mathbb Q)\backslash G(\mathbb A_\mathbb Q)\to\mathbb C$ 满足
$$
\Phi(zg)=\omega(z)\Phi(g),
\qquad z\in Z(\mathbb A_\mathbb Q).
$$

**定义 7.2.** 右正则作用定义为
$$
(R(h)\Phi)(g)=\Phi(gh),
\qquad h,g\in G(\mathbb A_\mathbb Q).
$$
若 $V$ 是一族函数构成的向量空间，并且 $R(h)V\subseteq V$ 对所有 $h$ 成立，则 $V$ 是 $G(\mathbb A_\mathbb Q)$ 的表示空间。

## 7.2 无穷处的权

设
$$
K_\infty=\operatorname{SO}(2)\subset\operatorname{GL}_2^+(\mathbb R),
$$
其中
$$
r(\theta)=
\begin{pmatrix}
\cos\theta&\sin\theta\\
-\sin\theta&\cos\theta
\end{pmatrix}.
$$

**定义 7.3.** Adelic 函数 $\Phi$ 称为无穷处权 $k$，若
$$
\Phi(gr(\theta))=e^{ik\theta}\Phi(g)
$$
对所有 $\theta\in\mathbb R$ 成立。若还要求对正实中心 $aI_2\in Z(\mathbb R)^+$ 有指定中心特征，则该条件和中心特征共同决定无穷处的 $K_\infty$-type。

**注 7.4.** 经典 slash 算子中的权 $k$ 在 adelic 语言中变成右 $K_\infty$-type。经典模形式是上半平面上的函数；adelic 模形式是 `GL(2)` adelic 商上的函数，其无穷处行为由 $K_\infty$-type 控制。

## 7.3 有限处的紧开子群

**定义 7.5.** 对整数 $N\ge1$，定义紧开子群
$$
K_0(N)=
\left\{
g=\begin{pmatrix}a&b\\ c&d\end{pmatrix}\in\operatorname{GL}_2(\widehat{\mathbb Z}):
c\equiv0\pmod N
\right\},
$$
其中
$$
\widehat{\mathbb Z}=\prod_p\mathbb Z_p.
$$
同理定义
$$
K_1(N)=
\left\{
g\in K_0(N):d\equiv1\pmod N
\right\}.
$$

**命题 7.6.** $K_0(N)$ 和 $K_1(N)$ 是 $G(\mathbb A_{\mathbb Q,f})$ 的开紧子群。

**证明.** $\operatorname{GL}_2(\widehat{\mathbb Z})=\prod_p\operatorname{GL}_2(\mathbb Z_p)$ 是开紧子群。条件 $c\equiv0\pmod N$ 和 $d\equiv1\pmod N$ 只依赖于模 $N\widehat{\mathbb Z}$ 的有限商，因此定义出 $\operatorname{GL}_2(\widehat{\mathbb Z})$ 中的开闭子集，并且在乘法和逆下稳定。故 $K_0(N)$ 和 $K_1(N)$ 是开紧子群。$\square$

**定义 7.7.** Adelic 函数 $\Phi$ 称为有限处级 $K\subset G(\mathbb A_{\mathbb Q,f})$，若
$$
\Phi(gk)=\Phi(g)
$$
对所有 $k\in K$ 成立。

## 7.4 经典到 adelic 的提升

设 $f\in M_k(\Gamma_0(N),\varepsilon)$。给定 $g\in G(\mathbb A_\mathbb Q)$，用 strong approximation 将 $g$ 写成
$$
g=\gamma g_\infty k_f,
\qquad
\gamma\in G(\mathbb Q),\quad
g_\infty\in \operatorname{GL}_2^+(\mathbb R),\quad
k_f\in K_0(N).
$$

**定义 7.8.** 在上述分解下，定义 adelic 提升的候选函数
$$
\Phi_f(g)=
\varepsilon(d_f)^{-1}
\det(g_\infty)^{k/2}
j(g_\infty,i)^{-k}
f(g_\infty i),
$$
其中 $d_f$ 是 $k_f$ 的右下角在 $(\mathbb Z/N\mathbb Z)^\times$ 中的像，$j(g_\infty,z)=cz+d$。

**外部输入定理 7.9（经典-adelic 对应）.** 定义 7.8 给出良定义函数，并诱导经典模形式空间与满足以下条件的 adelic 函数空间之间的同构：

1. 左 $G(\mathbb Q)$-不变。
2. 右 $K_0(N)$-变换由 nebentypus $\varepsilon$ 控制。
3. 无穷处为权 $k$ 的 $K_\infty$-type。
4. 在中心上具有由 $\varepsilon$ 和权 $k$ 决定的中心特征。
5. 满足经典模形式在尖点处的 holomorphy 或 cuspidality 转译得到的 moderate growth 条件。

在该对应下，尖点形式对应 adelic 尖点形式。

**注 7.10.** 定义 7.8 的分解不是唯一的；良定义性正是经典变换律和 strong approximation 的共同结果。不同文献对 $\varepsilon$、中心特征和 $K_\infty$-type 的归一化略有差异。本书只固定一种与第六章 Hecke 算子相容的接口口径。

## 7.5 尖点条件

**定义 7.11.** 设 $\Phi$ 是 $G(\mathbb Q)\backslash G(\mathbb A_\mathbb Q)$ 上的自守函数。称 $\Phi$ 为尖点的（cuspidal），若对所有 $g\in G(\mathbb A_\mathbb Q)$，
$$
\int_{\mathbb Q\backslash\mathbb A_\mathbb Q}
\Phi\left(
\begin{pmatrix}1&x\\0&1\end{pmatrix}g
\right)\,dx=0.
$$
其中取标准非平凡特征 $\psi:\mathbb A_\mathbb Q/\mathbb Q\to\mathbb C^\times$ 的自对偶 Haar 测度；由第一章和附录 F 的归一化，
$\operatorname{vol}(\mathbb Q\backslash\mathbb A_\mathbb Q)=1$。尖点条件本身不随 $dx$ 的非零标量倍改变，但后续 Fourier coefficient 公式使用这个固定归一化。

**命题 7.12.** 在经典-adelic 对应下，经典尖点形式对应 adelic 尖点函数。

**证明路线（外部输入）.** 上式是沿 Borel 子群 unipotent radical 的常数项积分。把 $g$ 通过 strong approximation 化到无穷处代表后，该积分等价于经典 Fourier 展开在相应尖点处的常数项。经典尖点条件要求所有尖点处常数项为 $0$，因此 adelic 常数项积分为 $0$；反向使用同一尖点双商比较。完整证明需要逐尖点追踪 $\Gamma_0(N)\backslash\mathbb P^1(\mathbb Q)$ 与 adelic 双商，本段只记录该外部输入的证明路线。

## 7.6 自守表示

**定义 7.13.** 固定中心特征 $\omega$。`GL(2)` 的 cuspidal automorphic forms 空间记为
$$
\mathcal A_0(G,\omega).
$$
它由满足下列条件的函数 $\Phi:G(\mathbb Q)\backslash G(\mathbb A_\mathbb Q)\to\mathbb C$ 构成：

1. $\Phi(zg)=\omega(z)\Phi(g)$。
2. $\Phi$ 在有限处右平移下光滑，即存在开紧 $K_f$ 使 $\Phi(gk)=\Phi(g)$。
3. $\Phi$ 在无穷处为 $C^\infty$、$K_\infty$-finite，并满足 moderate growth 条件。
4. $\Phi$ 满足尖点条件。

右正则作用使 $\mathcal A_0(G,\omega)$ 成为 $G(\mathbb A_\mathbb Q)$ 的表示。

**定义 7.14.** 一个 cuspidal automorphic representation of `GL(2)` 是在 $\mathcal A_0(G,\omega)$ 的右正则表示中出现的不可约可容许 $G(\mathbb A_\mathbb Q)$-表示的同构类。若 $\pi$ 是这样的表示，则写
$$
\pi=\bigotimes_v'\pi_v
$$
表示其 restricted tensor product 分解，其中 $\pi_v$ 是 $G(\mathbb Q_v)$ 的不可约可容许表示。

**外部输入定理 7.15（张量积分解）.** Cuspidal automorphic representation $\pi$ 可分解为 restricted tensor product
$$
\pi\cong\bigotimes_v'\pi_v.
$$
对几乎所有素数 $p$，$\pi_p$ 是非分歧球表示，并含有非零 $\operatorname{GL}_2(\mathbb Z_p)$-不变量。

该定理依赖局部表示论、全局自守表示的可容许性和 restricted tensor product 理论。

## 7.7 Hecke 本征形式到自守表示

**外部输入定理 7.16（newform 生成的自守表示）.** 设 $f\in S_k(\Gamma_0(N),\varepsilon)$ 是归一化 cuspidal newform，$k\ge2$，$\varepsilon$ 为 Dirichlet character。令 $\Phi_f$ 为按定理 7.9 归一化的 adelic 提升。则其 $K_\infty$-finite 右平移张成空间
$$
\pi_f=\langle R(g)\Phi_f:g\in G(\mathbb A_\mathbb Q)\rangle
$$
是不可约 cuspidal automorphic representation；本书把它置于 unitary automorphic normalization。它满足：

1. 中心特征是由 $\varepsilon$ 给出的酉 Hecke 特征，并带与权 $k$ 相容的无穷处分量。
2. 无穷处分量 $\pi_{f,\infty}$ 是与权 $k$ 对应的离散系列或其极限情形。
3. 若 $p\nmid N$，则 $\pi_{f,p}$ 非分歧，其 unitary Satake 参数由定义 7.18.1 给出。
4. 若 $p\mid N$，则 $\pi_{f,p}$ 的导子记录 $f$ 的局部级结构。

**注 7.17.** 对非 newform 的旧形式，不能直接套用定理 7.16 的不可约性陈述。Oldspace 中的 degeneracy images 是某个 $\pi_f$ 的高 level vectors；newform theory 选出导子级别的一维 newvector line。

**注 7.17.1.** 附录 H 证明经典 Hecke 双陪集代表如何嵌入有限 adelic Hecke algebra，并说明好素数处 $\mathbf 1_{K_p\operatorname{diag}(1,p)K_p}$ 的作用如何给出经典 $T_p$ 本征值。

**注 7.17.2.** 附录 J 进一步解释 oldforms、newforms、Atkin-Lehner operators 和 local newvectors 的关系；特别地，classical newform 的级对应 $\pi_f$ 的 adelic conductor。

**收口精修 7.A（classical-to-adelic 检查表）.** 后续把经典 newform 送入自守表示时，逐项使用以下翻译：

| 经典对象 | adelic 对象 | 后续使用 |
|---|---|---|
| $\Gamma_0(N)$ 变换律 | 左 $G(\mathbb Q)$-不变和右 $K_0(N)$-型 | 定义全局表示 $\pi_f$ |
| 权 $k$ 与 nebentypus | 无穷处 $K_\infty$-type 和中心特征 | 固定代数性、行列式和 Hodge-Tate 权 |
| 尖点条件 | 沿 unipotent radical 的常数项为 $0$ | 保证表示位于离散 cuspidal 谱 |
| 好素数 $T_p$ 本征值 | 球 Hecke 算子本征值 | 给出 Satake 参数和 Euler 因子 |
| newform 的最小级 | local newvector 的 conductor | 与 Galois 表示导子和降层比较 |

## 7.8 好素数处的两种 Satake 归一化

设 $f(q)=\sum_{n\ge1}a_nq^n$ 是归一化 Hecke eigenform，权 $k$、nebentypus $\varepsilon$、级 $N$。设 $p\nmid N$。

**定义 7.18.** 好素数 $p$ 处的 classical Hecke roots 是无序二元组 $(\alpha_p,\beta_p)$，满足
$$
\alpha_p+\beta_p=a_p,\qquad
\alpha_p\beta_p=\varepsilon(p)p^{k-1}.
$$
等价地，$\alpha_p,\beta_p$ 是多项式
$$
X^2-a_pX+\varepsilon(p)p^{k-1}
$$
的根。

**定义 7.18.1.** Unitarily normalized automorphic representation $\pi_f$ 的 Satake roots 定义为
$$
\alpha_p^{\mathrm u}=\alpha_pp^{-(k-1)/2},\qquad
\beta_p^{\mathrm u}=\beta_pp^{-(k-1)/2}.
$$
它们满足
$$
\alpha_p^{\mathrm u}+\beta_p^{\mathrm u}=a_pp^{-(k-1)/2},
\qquad
\alpha_p^{\mathrm u}\beta_p^{\mathrm u}=\varepsilon(p).
$$
Classical roots 与 unitary Satake roots 属于不同归一化，后文不再都称为
“$\pi_f$ 的 Satake 参数”。

**命题 7.19.** 好素数处的 Euler 因子可写为
$$
L_p(f,s)
=
\left(1-a_pp^{-s}+\varepsilon(p)p^{k-1-2s}\right)^{-1}
=
\left((1-\alpha_pp^{-s})(1-\beta_pp^{-s})\right)^{-1}.
$$

相应的 unitary automorphic local factor 为
$$
L_p(s,\pi_f,\operatorname{Std})
=\left((1-\alpha_p^{\mathrm u}p^{-s})
(1-\beta_p^{\mathrm u}p^{-s})\right)^{-1},
$$
即更直接地
$$
L_p(s,\pi_f,\operatorname{Std})
=L_p(f,s+(k-1)/2).
$$

**证明.** 由定义 7.18，
$$
(1-\alpha_pX)(1-\beta_pX)
=1-(\alpha_p+\beta_p)X+\alpha_p\beta_pX^2
=1-a_pX+\varepsilon(p)p^{k-1}X^2.
$$
取 $X=p^{-s}$ 得到 classical 公式。再把
$\alpha_p^{\mathrm u},\beta_p^{\mathrm u}$ 代入标准局部因子定义：
$$
(1-\alpha_p^{\mathrm u}p^{-s})(1-\beta_p^{\mathrm u}p^{-s})
=(1-\alpha_pp^{-(s+(k-1)/2)})(1-\beta_pp^{-(s+(k-1)/2)}),
$$
即得变量平移。$\square$

**外部输入定理 7.20（局部 Langlands 相容，好素数接口）.** 在 $p\nmid N$ 时，$\pi_{f,p}$ 的非分歧局部 Langlands 参数使用第五章的几何 Frobenius convention，并满足
$$
\varphi_{f,p}(\operatorname{Fr}_p)
\sim
\operatorname{diag}(\alpha_p^{\mathrm u},\beta_p^{\mathrm u}).
$$
若 $\lambda\mid\ell$ 且 $p\ne\ell$，则与 Deligne 表示的算术 Frobenius 归一化比较为
$$
\varphi_{f,p}
\cong
\iota\,\operatorname{WD}_p(\rho_{f,\lambda}^{\vee})^{\mathrm{F\text{-}ss}}
\otimes|\cdot|^{(k-1)/2}.
$$
这里对偶把 $\rho(\operatorname{Fr}_p)$ 的逆特征值变回 classical roots，随后 norm twist 给出 unitary roots。第十章将把该式扩展到坏位置并明确排除 $p=\ell$ 的额外 p-adic Hodge 责任。

## 7.9 标准 L 函数

**定义 7.21.** 设 $\pi=\otimes_v'\pi_v$ 为 `GL(2)` cuspidal automorphic representation。其标准 L 函数定义为 Euler 乘积
$$
L(s,\pi)=\prod_v L(s,\pi_v,\operatorname{Std})
$$
在绝对收敛半平面中成立。若 $\pi=\pi_f$ 来自归一化 newform $f$，本书默认 unitary normalization，故
$$
L(s,\pi_f,\operatorname{Std})=L(f,s+(k-1)/2).
$$
若定义非酉算术归一化
$\pi_f^{\mathrm{alg}}=\pi_f\otimes|\det|^{-(k-1)/2}$，则且仅在该归一化下有
$L(s,\pi_f^{\mathrm{alg}},\operatorname{Std})=L(f,s)$。

**外部输入定理 7.22（`GL(2)` 标准 L 函数）.** 设 $\pi$ 为 unitary cuspidal automorphic representation of
$\operatorname{GL}_2(\mathbb A_K)$。标准 Euler 乘积在 $\operatorname{Re}(s)>1$ 绝对收敛；补入 Archimedean 因子所得 $\Lambda(s,\pi)$ 整，并满足
$$
\Lambda(s,\pi)=\varepsilon(s,\pi)\Lambda(1-s,\pi^\vee),
$$
其中局部因子、加法特征和 Haar 测度采用归一化总表。该结论可由 Godement-Jacquet 或 Rankin-Selberg 理论得到；本书不重证其全局解析估计。

**注 7.23.** 附录 I 在 `GL(n)` 口径下给出 Godement-Jacquet 和 Rankin-Selberg 积分的定义、Euler 分解、非分歧计算和函数方程接口；本章的 `GL(2)` 标准 L 函数是其中 $n=2$ 的特例。

## 7.10 本章小结

经典模形式给出 `GL(2)` 自守表示的一个具体模型，但 Langlands 纲领使用的是 adelic 表示语言。Adelic 提升把经典变换律变成左 $G(\mathbb Q)$-不变、右开紧变换和无穷处 $K$-type；尖点条件变成沿 unipotent radical 的常数项积分为零；Hecke 本征值变成非分歧局部分量的 Satake 参数。由此，经典的 $L(f,s)$ 在 algebraic normalization 下是同变量的标准 L 函数；对本书默认的 unitary $\pi_f$，精确关系是 $L(s,\pi_f)=L(f,s+(k-1)/2)$。

## 练习

**练习 7.1.** 证明 $K_0(N)$ 是 $G(\mathbb A_{\mathbb Q,f})$ 的开紧子群。

**练习 7.2.** 对 $N=1$，用 strong approximation 说明
$$
G(\mathbb A_\mathbb Q)=G(\mathbb Q)G(\mathbb R)^+\operatorname{GL}_2(\widehat{\mathbb Z})
$$
的意义，并解释为什么它把 adelic 函数限制到上半平面。

**练习 7.3.** 设 $\Phi$ 是 adelic 模形式。说明右 $K_0(N)$-不变性如何对应经典的 $\Gamma_0(N)$-变换律。

**练习 7.4.** 证明定义 7.18 和命题 7.19 的等价性。

**练习 7.5.** 解释为什么 oldform 可能对应同一个自守表示中的不同 level vectors，而不是不同的全局表示。

**练习 7.6.** 设 $f$ 为权 $k$ 的归一化 Hecke eigenform，好素数 $p$ 处 classical Satake roots 为 $\alpha_p,\beta_p$。证明 unitary automorphic normalization 下
$$
L(s,\pi_f,\operatorname{Std})=L(f,s+(k-1)/2)
$$
在所有好素数 Euler factors 上相容。
