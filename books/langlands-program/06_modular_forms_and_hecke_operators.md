# 第六章：上半平面上的模形式与 Hecke 算子

## 本章目标

本章建立 `GL(2)/\mathbb Q` 的经典入口：上半平面、同余子群、模形式、尖点形式、Hecke 算子、Hecke 特征形式和由 Fourier 系数定义的 L 函数。下一章将把这些经典对象改写为 adelic automorphic representations。

## 依赖前置知识

需要复分析、Riemann 曲面、群作用和线性代数。模曲线紧化、有限维性、Hecke 算子保持模形式空间等事实在本章作为外部输入或外部输入的证明路线处理。附录 W 给出模曲线代数化、Hecke correspondences、old/new 分解和 Atkin-Lehner-Li 理论的接口。

收口归一化回指：本章采用 classical modular form normalization；与第七、九、十四章比较时，Hecke roots、Galois Frobenius 和自守 L 函数变量按 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 第 5、6、8 节转换。

## 6.1 上半平面和 slash 算子

**定义 6.1.** 上半平面定义为
$$
\mathfrak H=\{z\in\mathbb C:\operatorname{Im}(z)>0\}.
$$
群 $\operatorname{GL}_2^+(\mathbb R)$ 由行列式为正的实 $2\times2$ 可逆矩阵组成，并通过分式线性变换作用在 $\mathfrak H$ 上：
$$
\gamma z=\frac{az+b}{cz+d},
\qquad
\gamma=\begin{pmatrix}a&b\\ c&d\end{pmatrix}.
$$

**定义 6.2.** 设 $k\in\mathbb Z$。对 $\gamma\in\operatorname{GL}_2^+(\mathbb R)$ 和函数 $f:\mathfrak H\to\mathbb C$，定义权 $k$ slash 算子
$$
(f|_k\gamma)(z)
=
\det(\gamma)^{k/2}(cz+d)^{-k}f(\gamma z).
$$
这里 $\det(\gamma)^{k/2}$ 取正实数的实幂。

**命题 6.3.** Slash 算子给出右作用：
$$
(f|_k\gamma)|_k\delta=f|_k(\gamma\delta)
$$
对所有 $\gamma,\delta\in\operatorname{GL}_2^+(\mathbb R)$ 成立。

**证明.** 写
$$
j(\gamma,z)=cz+d.
$$
分式线性变换满足
$$
j(\gamma\delta,z)=j(\gamma,\delta z)j(\delta,z).
$$
于是
$$
\begin{aligned}
((f|_k\gamma)|_k\delta)(z)
&=\det(\delta)^{k/2}j(\delta,z)^{-k}
\det(\gamma)^{k/2}j(\gamma,\delta z)^{-k}f(\gamma\delta z)\\
&=\det(\gamma\delta)^{k/2}j(\gamma\delta,z)^{-k}f(\gamma\delta z)\\
&=(f|_k(\gamma\delta))(z).
\end{aligned}
$$
$\square$

## 6.2 同余子群和模形式

**定义 6.4.** 对整数 $N\ge 1$，定义同余子群
$$
\Gamma(N)=
\left\{
\gamma\in\operatorname{SL}_2(\mathbb Z):
\gamma\equiv I\pmod N
\right\},
$$
$$
\Gamma_0(N)=
\left\{
\begin{pmatrix}a&b\\ c&d\end{pmatrix}\in\operatorname{SL}_2(\mathbb Z):
c\equiv0\pmod N
\right\},
$$
$$
\Gamma_1(N)=
\left\{
\begin{pmatrix}a&b\\ c&d\end{pmatrix}\in\Gamma_0(N):
a\equiv d\equiv1\pmod N
\right\}.
$$

**定义 6.5.** 设 $\varepsilon:(\mathbb Z/N\mathbb Z)^\times\to\mathbb C^\times$ 为 Dirichlet 特征，并要求
$$
\varepsilon(-1)=(-1)^k
$$
以避免 $-I$ 作用给出矛盾。权 $k$、级 $\Gamma_0(N)$、nebentypus $\varepsilon$ 的模形式是全纯函数 $f:\mathfrak H\to\mathbb C$，满足：

1. 对所有
   $$
   \gamma=\begin{pmatrix}a&b\\ c&d\end{pmatrix}\in\Gamma_0(N),
   $$
   有
   $$
   f|_k\gamma=\varepsilon(d)f.
   $$
2. $f$ 在 $\Gamma_0(N)$ 的所有尖点处全纯。

所有此类模形式构成复向量空间，记为
$$
M_k(\Gamma_0(N),\varepsilon).
$$
若 $\varepsilon$ 平凡，则简写为 $M_k(\Gamma_0(N))$。

**定义 6.6.** 若模形式 $f$ 在所有尖点处消失，则称为尖点形式（cusp form）。尖点形式空间记为
$$
S_k(\Gamma_0(N),\varepsilon)\subset M_k(\Gamma_0(N),\varepsilon).
$$

## 6.3 尖点和 Fourier 展开

**定义 6.7.** 尖点集合定义为
$$
\Gamma_0(N)\backslash\mathbb P^1(\mathbb Q).
$$
给定尖点 $\mathfrak a$，选择 $\sigma\in\operatorname{SL}_2(\mathbb Z)$ 使 $\sigma\infty=\mathfrak a$。若 $h$ 是 $\sigma^{-1}\Gamma_0(N)\sigma$ 中稳定 $\infty$ 的平移宽度，则 $f|_k\sigma$ 满足
$$
(f|_k\sigma)(z+h)=(f|_k\sigma)(z)
$$
并有 Fourier 展开
$$
(f|_k\sigma)(z)=\sum_{n\ge 0}a_{\mathfrak a}(n) e^{2\pi inz/h}
$$
称为 $f$ 在尖点 $\mathfrak a$ 处的展开。若所有展开中常数项 $a_{\mathfrak a}(0)$ 都为 $0$，则 $f$ 为尖点形式。

**命题 6.8.** 若 $f\in M_k(\Gamma_0(N),\varepsilon)$，则在无穷尖点处有 Fourier 展开
$$
f(z)=\sum_{n\ge0}a_nq^n,\qquad q=e^{2\pi iz}.
$$
若 $f$ 为尖点形式，则 $a_0=0$。

**证明.** 矩阵
$$
T=\begin{pmatrix}1&1\\0&1\end{pmatrix}
$$
属于 $\Gamma_0(N)$ 且 $\varepsilon(1)=1$，所以 $f(z+1)=f(z)$。因此 $f$ 在水平条带上有 Fourier 展开
$$
f(z)=\sum_{n\in\mathbb Z}a_ne^{2\pi inz}.
$$
在无穷尖点处全纯等价于负指数项全部为 $0$；在无穷尖点处消失等价于常数项为 $0$。$\square$

**外部输入定理 6.9（有限维性）.** 对任意 $k,N,\varepsilon$，空间 $M_k(\Gamma_0(N),\varepsilon)$ 和 $S_k(\Gamma_0(N),\varepsilon)$ 都是有限维复向量空间。

该定理可由模曲线 $X_0(N)$ 的紧 Riemann 曲面结构和相应线丛的截面空间有限维性推出。

## 6.4 Hecke 算子

本节固定 $k,N,\varepsilon$，并记
$$
S_k(N,\varepsilon)=S_k(\Gamma_0(N),\varepsilon).
$$

**定义 6.10.** 对素数 $\ell\nmid N$，Hecke 算子 $T_\ell$ 定义为双陪集算子
$$
T_\ell f
=
\ell^{k/2-1}
\sum_{\alpha\in\Gamma_0(N)\backslash
\Gamma_0(N)
\begin{pmatrix}1&0\\0&\ell\end{pmatrix}
\Gamma_0(N)}
f|_k\alpha,
$$
其中求和取左陪集代表。对 $\ell\mid N$，定义
$$
U_\ell f
=
\ell^{k/2-1}\sum_{b=0}^{\ell-1}
f|_k
\begin{pmatrix}1&b\\0&\ell\end{pmatrix}.
$$

**外部输入定理 6.11（Hecke 算子的良定义性）.** 算子 $T_\ell$ 在 $\ell\nmid N$ 时保持 $M_k(\Gamma_0(N),\varepsilon)$ 和 $S_k(\Gamma_0(N),\varepsilon)$；算子 $U_\ell$ 在 $\ell\mid N$ 时也保持这些空间。它们与尖点处全纯性和消失性相容。

**命题 6.12（Fourier 系数公式）.** 设
$$
f(q)=\sum_{n\ge0}a_nq^n.
$$
若 $\ell\nmid N$，则
$$
T_\ell f=\sum_{n\ge0}
\left(a_{\ell n}+\varepsilon(\ell)\ell^{k-1}a_{n/\ell}\right)q^n,
$$
其中 $a_{n/\ell}=0$ 当 $\ell\nmid n$。若 $\ell\mid N$，则
$$
U_\ell f=\sum_{n\ge0}a_{\ell n}q^n.
$$

**证明路线（外部输入）.** 对 $\ell\nmid N$，双陪集有代表
$$
\begin{pmatrix}1&b\\0&\ell\end{pmatrix}\quad(0\le b<\ell),
\qquad
\begin{pmatrix}\ell&0\\0&1\end{pmatrix},
$$
第二类代表在带 nebentypus 的情形中贡献 $\varepsilon(\ell)\ell^{k-1}a_{n/\ell}$。第一类代表给出平均
$$
\ell^{k/2-1}\sum_{b=0}^{\ell-1}
f|_k\begin{pmatrix}1&b\\0&\ell\end{pmatrix},
$$
其 Fourier 展开只保留原展开中指标可被 $\ell$ 整除的项，贡献 $a_{\ell n}$。$\ell\mid N$ 时只有 $U_\ell$ 的代表族，得到第二个公式。完整证明需要双陪集分解和 nebentypus 的追踪。$\square$

**注 6.12.1.** 附录 H 给出本命题所用的双陪集代表、Fourier 系数计算、Petersson 内积接口以及与 adelic Hecke 代数的比较。

## 6.5 Hecke 特征形式和 L 函数

**定义 6.13.** 非零尖点形式 $f\in S_k(N,\varepsilon)$ 称为 Hecke 特征形式，若它是所有 $T_\ell$（$\ell\nmid N$）和所有 $U_\ell$（$\ell\mid N$）的共同特征向量。若其 Fourier 展开
$$
f(q)=\sum_{n\ge1}a_nq^n
$$
满足 $a_1=1$，则称为归一化 Hecke 特征形式。

**定义 6.14.** 对归一化 Hecke 特征形式 $f(q)=\sum_{n\ge1}a_nq^n$，定义 Dirichlet 级数
$$
L(f,s)=\sum_{n\ge1}a_nn^{-s}
$$
在绝对收敛半平面中成立。其解析延拓和函数方程由 Mellin 变换与 newform 理论给出；在证明这些事实之前，$L(f,s)$ 只表示该半平面中的 Dirichlet 级数。

**外部输入定理 6.15（Euler 乘积）.** 若 $f\in S_k(N,\varepsilon)$ 是归一化 Hecke 特征形式，则
$$
L(f,s)=
\prod_{\ell\nmid N}
\left(1-a_\ell\ell^{-s}+\varepsilon(\ell)\ell^{k-1-2s}\right)^{-1}
\cdot
\prod_{\ell\mid N}L_\ell(f,s)
$$
在绝对收敛半平面中成立。坏素数 $\ell\mid N$ 的局部因子 $L_\ell(f,s)$ 由 $U_\ell$ 本征值和 newform 理论确定。

**证明路线（外部输入）.** Hecke 代数关系给出 Fourier 系数的乘法关系
$$
a_m a_n=
\sum_{d\mid(m,n)}
\varepsilon(d)d^{k-1}a_{mn/d^2}
$$
在 $(mn,N)=1$ 时成立。该关系等价于好素数处的 Euler 因子。坏素数处需要 Atkin-Lehner newform 理论。$\square$

## 6.6 Mellin 变换

**命题 6.16.** 设 $f(q)=\sum_{n\ge1}a_nq^n\in S_k(\Gamma_0(N),\varepsilon)$。在 $\operatorname{Re}(s)$ 足够大时，
$$
\int_0^\infty f(iy)y^s\frac{dy}{y}
=
(2\pi)^{-s}\Gamma(s)L(f,s).
$$

**证明.** 因为 $f$ 是尖点形式，$f(iy)$ 在 $y\to\infty$ 指数衰减；在 $y\to0^+$ 时，模变换性质给出至多多项式增长。结合 Fourier 系数的标准增长估计，在 $\operatorname{Re}(s)$ 足够大时可逐项积分：
$$
\int_0^\infty f(iy)y^s\frac{dy}{y}
=
\sum_{n\ge1}a_n\int_0^\infty e^{-2\pi ny}y^s\frac{dy}{y}.
$$
换元 $t=2\pi ny$，得到
$$
\int_0^\infty e^{-2\pi ny}y^s\frac{dy}{y}
=(2\pi n)^{-s}\Gamma(s).
$$
故结论成立。$\square$

**外部输入定理 6.17（函数方程，newform 形式）.** 若 $f$ 是权 $k$、级 $N$、nebentypus $\varepsilon$ 的 newform，则完成 L 函数
$$
\Lambda(f,s)=N^{s/2}(2\pi)^{-s}\Gamma(s)L(f,s)
$$
满足形如
$$
\Lambda(f,s)=\eta_f\Lambda(\overline f,k-s)
$$
的函数方程，其中 $|\eta_f|=1$。精确常数依赖 Fricke involution、nebentypus 和归一化选择。

## 6.7 与 Langlands 主线的关系

经典模形式只给出了 `GL(2)` 自守对象的一种模型。其 Langlands 含义要通过两步完成：

1. 把 $f$ 提升为 $\operatorname{GL}_2(\mathbb A_\mathbb Q)$ 的 cuspidal automorphic representation $\pi_f$。
2. 把 Hecke 本征值 $a_\ell$ 解释为非分歧局部表示 $\pi_{f,\ell}$ 的 Satake 参数，进而对应二维 Galois/Weil 参数的 Frobenius trace。

**外部输入定理 6.18（Deligne，接口形式）.** 设 $f$ 为归一化 cuspidal Hecke eigenform，权 $k\ge2$，级 $N$，nebentypus $\varepsilon$，系数域为 $E_f$。对 $E_f$ 的每个有限素位 $\lambda$，存在连续半单 Galois 表示
$$
\rho_{f,\lambda}:G_\mathbb Q\to\operatorname{GL}_2(E_{f,\lambda})
$$
使得对所有 $\ell\nmid N\operatorname{char}(\lambda)$，
$$
\operatorname{tr}\rho_{f,\lambda}(\operatorname{Frob}_\ell^{\operatorname{arith}})=a_\ell,
\qquad
\det\rho_{f,\lambda}(\operatorname{Frob}_\ell^{\operatorname{arith}})=\varepsilon(\ell)\ell^{k-1}.
$$

这里 $\operatorname{Frob}_\ell^{\operatorname{arith}}$ 表示在剩余域上诱导 $x\mapsto x^\ell$ 的算术 Frobenius。本书在类域论和局部 Langlands 参数中默认使用几何 Frobenius；与本定理比较时必须取逆、对偶或 Tate twist 的相应归一化。本定理是模形式与二维 Galois 表示之间的核心桥梁。

## 6.8 本章小结

本章定义了经典模形式和尖点形式，构造了 Hecke 算子，并从归一化 Hecke 特征形式得到 L 函数。好素数处的 Euler 因子由 Hecke 本征值控制；Deligne 定理把这些本征值解释为二维 Galois 表示的 Frobenius trace。下一章将把这些经典对象翻译为 adelic `GL(2)` 自守表示。

## 练习

**练习 6.1.** 证明命题 6.3 中 slash 算子的右作用性质。

**练习 6.2.** 设 $f\in M_k(\Gamma_0(N),\varepsilon)$。证明 $f(z+1)=f(z)$，并推出无穷尖点处的 Fourier 展开。

**练习 6.3.** 对 $\ell\nmid N$，使用双陪集代表计算 $T_\ell$ 在 Fourier 系数上的作用。

**练习 6.4.** 假设 $f$ 是归一化 Hecke 特征形式。由 Hecke 关系推出好素数处的 Euler 因子
$$
\left(1-a_\ell\ell^{-s}+\varepsilon(\ell)\ell^{k-1-2s}\right)^{-1}.
$$

**练习 6.5.** 证明命题 6.16 的 Mellin 变换公式，并指出尖点条件在收敛性中起到的作用。
