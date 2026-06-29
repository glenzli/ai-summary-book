# 全书归一化总表

本文档固定跨章节比较时使用的 convention。若某章临时采用不同 convention，必须在该章中说明转换公式，并回指本文档。

## 0. 优先级

1. [NOTATION.md](NOTATION.md) 固定符号。
2. 本文档固定归一化。
3. 具体章节可以为局部计算临时改变归一化，但必须显式标注。

任何涉及 Frobenius、reciprocity、Satake 参数、Haar 测度、Fourier 变换、L 函数变量或 Tate twist 的比较，均应先检查本文档。

## 1. 绝对值和 adeles

设 $K$ 为整体域，$v$ 为非 Archimedean 位置，$K_v$ 的剩余域基数为 $q_v$。

| 对象 | 本书 convention |
|---|---|
| 一致化元 | $\varpi_v$ |
| 绝对值 | $|\varpi_v|_v=q_v^{-1}$ |
| 乘积公式 | $\prod_v |x|_v=1$ for $x\in K^\times$ |
| Adele norm | $|a|_{\mathbb A}=\prod_v|a_v|_v$ |
| Idele class group | $C_K=K^\times\backslash\mathbb A_K^\times$ |

所有整体 L 函数的 Euler 乘积变量均与该绝对值归一化相容。

## 2. Frobenius 和 reciprocity

非 Archimedean 局部域 $F$ 的剩余域基数记为 $q$。

| 记号 | 含义 |
|---|---|
| $\operatorname{Frob}_F^{\operatorname{arith}}$ | 算术 Frobenius，在剩余域上诱导 $x\mapsto x^q$ |
| $\operatorname{Fr}_F$ | 几何 Frobenius，定义为 $(\operatorname{Frob}_F^{\operatorname{arith}})^{-1}$ |
| $W_F$ | 采用几何 Frobenius 归一化的 Weil 群 |
| $\operatorname{rec}_F$ | 局部 reciprocity map，满足 $\operatorname{rec}_F(\varpi)=\operatorname{Fr}_F$ |

因此，若 $\chi:F^\times\to\mathbb C^\times$ 为非分歧特征，则对应的一维 Weil 参数 $\phi_\chi$ 满足
$$
\phi_\chi(\operatorname{Fr}_F)=\chi(\varpi).
$$

若文献采用 $\operatorname{rec}_F(\varpi)=\operatorname{Frob}_F^{\operatorname{arith}}$，则与本书相比需要把 Frobenius 元取逆。第七至十章中模形式和椭圆曲线的 Galois 表示常用算术 Frobenius，因此跨到第五、十二、十四章的局部参数时必须进行该转换。

## 3. Haar 测度和 Fourier 变换

### 3.1 加法测度

Fourier 分析和 Tate thesis 中，局部加法 Haar 测度 $dx_v$ 取为相对于所选加法特征 $\psi_v$ 的自对偶测度。整体测度为 restricted product：
$$
dx=\prod_v dx_v.
$$

若 $v$ 非 Archimedean 且 $\psi_v$ 的 conductor 为 $\mathcal O_v$，则
$$
\operatorname{vol}(\mathcal O_v,dx_v)=1.
$$

### 3.2 乘法测度

非 Archimedean 局部域上，在 $\operatorname{vol}(\mathcal O_v,dx_v)=1$ 时，本书默认
$$
d^\times x_v=(1-q_v^{-1})^{-1}\frac{dx_v}{|x_v|_v},
$$
从而
$$
\operatorname{vol}(\mathcal O_v^\times,d^\times x_v)=1.
$$

局部表示论章节若只讨论 Hecke 代数，则对开紧子群 $K_v$ 通常取
$$
\operatorname{vol}(K_v)=1.
$$
涉及 trace formula 或 Plancherel 时必须重新声明测度。

### 3.3 Fourier 变换

局部和整体 Fourier 变换均写作
$$
\widehat f(y)=\int f(x)\psi(xy)\,dx.
$$

Poisson summation、Tate zeta integral 和 functional equation 只在自对偶测度下直接采用本书公式。

## 4. Satake 参数

设 $F$ 为非 Archimedean 局部域，$G/F$ 非分歧，$K=G(\mathcal O_F)$ 为 hyperspecial subgroup。

| 对象 | 本书 convention |
|---|---|
| 球 Hecke 代数 | $\mathcal H(G(F),K)$，卷积测度满足 $\operatorname{vol}(K)=1$ |
| Satake 变换 | 使用 $\delta_B^{1/2}$ 的归一化 Satake 变换 |
| Satake 参数 | $\widehat G$ 中的半单共轭类 $s_\pi$ |
| 非分歧局部参数 | $\varphi_\pi(\operatorname{Fr}_F)=s_\pi$ |
| 标准局部因子 | $L(s,\pi,r)=\det(1-r(s_\pi)q^{-s})^{-1}$ |

对 $G=\operatorname{GL}_n$，若
$$
\pi=\operatorname{Ind}_{B(F)}^{G(F)}(\chi_1\otimes\cdots\otimes\chi_n)
$$
为归一化非分歧主级数，则
$$
s_\pi=\operatorname{diag}(\chi_1(\varpi),\ldots,\chi_n(\varpi)).
$$

未归一化诱导会把 Satake 参数乘上 modulus character 的因子；本书默认使用归一化抛物诱导。

## 5. 经典模形式与自守归一化

设 $f(q)=\sum_{n\ge1}a_nq^n$ 是权 $k$、nebentypus $\varepsilon$、级 $N$ 的 normalized cuspidal Hecke eigenform，且 $p\nmid N$。

经典 Satake roots $\alpha_p,\beta_p$ 由
$$
X^2-a_pX+\varepsilon(p)p^{k-1}=(X-\alpha_p)(X-\beta_p)
$$
定义，并给出 classical L 函数
$$
L(f,s)=\prod_{p\nmid N}(1-\alpha_pp^{-s})^{-1}(1-\beta_pp^{-s})^{-1}\cdot(\text{bad factors}).
$$

若 $\pi_f$ 表示对应的 unitary automorphic normalization，则好素数处的 unitary Satake 参数为
$$
\left(\alpha_p p^{-(k-1)/2},\ \beta_p p^{-(k-1)/2}\right),
$$
并有
$$
L(s,\pi_f,\operatorname{Std})=L(f,s+(k-1)/2)
$$
在好素数 Euler factors 上相容。

本书第六至十章在模形式和椭圆曲线应用中保留 classical normalization；第十三、十四章讨论一般自守表示时使用 automorphic normalization。比较两者必须加入上述平移。

## 6. Galois 表示和局部因子

设 $\rho:G_K\to\operatorname{GL}(V)$ 为 $\ell$-adic Galois 表示，$v\nmid\ell$ 且 $\rho$ 在 $v$ 非分歧。

若使用算术 Frobenius，则局部 Euler factor 写作
$$
L_v(s,\rho)=\det(1-\rho(\operatorname{Frob}_v^{\operatorname{arith}})q_v^{-s}\mid V)^{-1}.
$$

若转换到本书局部 Langlands 默认的几何 Frobenius convention，则同一个半单共轭类必须取逆：
$$
\rho(\operatorname{Fr}_v)=\rho(\operatorname{Frob}_v^{\operatorname{arith}})^{-1}.
$$

当比较 Galois 表示与自守 Satake 参数时，可能还需要 Tate twist、对偶或变量平移。第十四章中所有此类比较必须说明采用哪一种 convention。

## 7. Tate twist

本书采用 $\ell$-adic cyclotomic character $\chi_\ell$，并令
$$
\chi_\ell(\operatorname{Frob}_v^{\operatorname{arith}})=q_v
$$
在 $v\nmid\ell$ 时成立。于是对整数 $m$，
$$
V(m)=V\otimes\chi_\ell^m
$$
使算术 Frobenius eigenvalues 乘以 $q_v^m$，使几何 Frobenius eigenvalues 乘以 $q_v^{-m}$。

因此，若一个比较式从 arithmetic Frobenius 改写为 geometric Frobenius，同时又改变 Tate twist，必须同时记录“取逆”和“乘以 $q_v^{\pm m}$”两个操作。

## 8. L 函数变量

| 语境 | 本书 convention |
|---|---|
| Hecke character | $L(s,\chi)=\prod_v L(s,\chi_v)$ |
| Tate thesis | zeta integral 变量为 $s$，函数方程比较 $s$ 与 $1-s$ |
| `GL(n)` standard L function | $L(s,\pi,\operatorname{Std})=\prod_v L(s,\pi_v,\operatorname{Std})$ |
| Rankin-Selberg | $L(s,\pi\times\pi')$ |
| Classical modular form | $L(f,s)$ 使用 classical normalization |
| Automorphic representation from $f$ | $L(s,\pi_f,\operatorname{Std})=L(f,s+(k-1)/2)$ |

已完成 L 函数的 Archimedean gamma factors 必须说明来源：Tate thesis、Godement-Jacquet、Rankin-Selberg、Langlands-Shahidi 或外部输入。

## 9. Geometric Langlands 中的归一化

几何 Langlands 部分使用与局部 Langlands 相容的几何 Frobenius 口径。有限域上从 sheaf 到 function 时，trace function 必须声明使用几何 Frobenius 还是算术 Frobenius。

几何 Satake 中，perverse sheaves 的 cohomological shifts 和 Tate twists 不能省略。本文默认采用使
$$
\operatorname{Sat}_G:\operatorname{Perv}_{G(\mathcal O)}(\operatorname{Gr}_G)\simeq\operatorname{Rep}(\widehat G)
$$
成为 tensor equivalence 的归一化。若需要比较函数迹和球 Hecke 代数，则必须同时追踪 $q^{\langle\rho,\lambda\rangle}$ 因子。

## 10. 快速检查表

跨章节使用公式前，逐项检查：

1. 使用的是几何 Frobenius 还是算术 Frobenius。
2. Reciprocity map 把一致化元送到哪一个 Frobenius。
3. 抛物诱导是归一化还是未归一化。
4. Satake 参数是否已经包含 $\delta_B^{1/2}$。
5. Haar 测度是自对偶、开紧体积为 $1$，还是 trace formula convention。
6. L 函数是 classical normalization 还是 unitary automorphic normalization。
7. Tate twist 是否改变了 Frobenius eigenvalues。
8. 外部输入定理采用的 convention 是否与本书相同。

若任一项不同，正文必须写出转换，而不是只引用同名定理。
