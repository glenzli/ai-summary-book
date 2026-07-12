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

局部 Artin 映射的拓扑同构目标是 $W_F^{\operatorname{ab}}$：
$$
F^\times\xrightarrow{\sim}W_F^{\operatorname{ab}}.
$$
它与到 profinite 群 $G_F^{\operatorname{ab}}$ 的映射不是同一个拓扑同构；后者由前者复合稠密映射
$W_F^{\operatorname{ab}}\to G_F^{\operatorname{ab}}$ 得到，并在 profinite 完备化后成为同构。全局映射
$C_K\to G_K^{\operatorname{ab}}$ 也只在有限 Abel 商上按本书需要使用；数域情形必须除去连通分量并取 profinite 完备化，不能把 $C_K$ 本身写成 $G_K^{\operatorname{ab}}$。

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

这里 additive conductor 记为
$$
\mathfrak c(\psi_v)=\{x\in F:\psi_v(x\mathcal O_v)=1\};
$$
它是一个分式理想。$\mathcal O_v$ 在配对 $(x,y)\mapsto\psi_v(xy)$ 下的 annihilator 为
$\mathfrak c(\psi_v)$，自对偶测度满足
$$
\operatorname{vol}(\mathcal O_v,dx_v)\,
\operatorname{vol}(\mathfrak c(\psi_v),dx_v)=1.
$$
因此只有在 $\mathfrak c(\psi_v)=\mathcal O_v$ 时才能同时断言 $dx_v$ 自对偶且
$\operatorname{vol}(\mathcal O_v)=1$。对整体特征 $\psi=\prod_v\psi_v$，几乎所有
$\psi_v$ 的 conductor 为 $\mathcal O_v$，故局部自对偶测度组成 restricted product；采用本书 Fourier 变换时，相应商测度满足
$\operatorname{vol}(K\backslash\mathbb A_K)=1$。

### 3.2 乘法测度

非 Archimedean 局部域上，若 $dx_v$ 已固定，本书取
$$
d^\times x_v=
\frac{1}{(1-q_v^{-1})\operatorname{vol}(\mathcal O_v,dx_v)}
\frac{dx_v}{|x_v|_v},
$$
从而
$$
\operatorname{vol}(\mathcal O_v^\times,d^\times x_v)=1.
$$

这个乘法测度一般不是由“自对偶”条件决定的；自对偶只适用于加法群。整体乘法测度
$d^\times x=\prod_vd^\times x_v$ 还需要逐一固定 Archimedean 因子。改变任一局部乘法测度会按同一标量改变 zeta integral，但不会改变由归一化积分商定义的局部 $L$ 因子；epsilon 因子的公式则必须同步记录 $dx_v$ 与 $\psi_v$。

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

本节前述 $s_\pi\in\widehat G$ 的写法直接适用于 split $G$。若 $G/F$ 仅为 unramified，参数自然是
$$
s_\pi\rtimes\operatorname{Fr}_F
\in\widehat G\rtimes\langle\operatorname{Fr}_F\rangle
$$
的 $\widehat G$-共轭类；除非 $G$ split 或 Weil 分量已吸收到 $r$ 中，不得删去
$\operatorname{Fr}_F$。Hyperspecial subgroup 的选择、$\operatorname{vol}(K)=1$ 和归一化
Satake 变换三项共同决定这里的参数。

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

为避免同名参数混淆，正文把 $(\alpha_p,\beta_p)$ 称为 **classical Hecke roots**，并写
$$
\alpha_p^{\mathrm u}=\alpha_pp^{-(k-1)/2},\qquad
\beta_p^{\mathrm u}=\beta_pp^{-(k-1)/2}
$$
表示 $\pi_f$ 的 unitary Satake roots。若确需使标准 L 函数在同一变量等于 $L(f,s)$，则使用非酉算术归一化
$$
\pi_f^{\mathrm{alg}}=\pi_f\otimes|\det|^{-(k-1)/2},\qquad
L(s,\pi_f^{\mathrm{alg}},\operatorname{Std})=L(f,s).
$$
未加上标的 $\pi_f$ 在第七章以后默认指 unitary normalization。

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

对权 $k\ge2$ newform 的 Deligne 表示 $\rho_{f,\lambda}$，本书令
$\operatorname{WD}_v(\rho)$ 使用几何 Frobenius。若 $v\nmid\ell$，并通过
$\iota:\overline{\mathbb Q}_\ell\simeq\mathbb C$ 比较系数，则 unitary automorphic normalization 的精确局部-整体相容式为
$$
\operatorname{rec}_{v,2}(\pi_{f,v})
\cong
\iota\,\operatorname{WD}_v(\rho_{f,\lambda}^{\vee})^{\mathrm{F\text{-}ss}}
\otimes |\cdot|^{(k-1)/2}.
$$
在好位置，右侧几何 Frobenius 的特征值正是
$\alpha_v^{\mathrm u},\beta_v^{\mathrm u}$。等价的局部 L 因子关系是
$$
L_v(s,\pi_f,\operatorname{Std})
=L_v(s+(k-1)/2,\rho_{f,\lambda})
=L_v(f,s+(k-1)/2).
$$
若改用 $\pi_f^{\mathrm{alg}}$，对偶仍负责把几何 Frobenius 的逆特征值变回算术特征值，而
$|\cdot|^{(k-1)/2}$ 的 unitary twist 不再出现。

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

Hodge-Tate 权采用算术编号：$\operatorname{HT}(\chi_\ell)=\{1\}$，因而
$\operatorname{HT}(V(m))=\operatorname{HT}(V)+m$。使用
$\operatorname{gr}^iD_{\operatorname{dR}}$ 给权编号的文献常令 $\mathbb Q_\ell(1)$ 权为 $-1$；引用其公式时必须整体变号。按本书 convention，权 $k$ newform 的 Deligne 表示有 Hodge-Tate 多重集 $\{0,k-1\}$。

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

参数侧函数方程把线性参数 $r\circ\varphi$ 换成其对偶
$(r\circ\varphi)^\vee=r^\vee\circ\varphi$。因此一般写作
$$
\Lambda(s,\pi,r)=\varepsilon(s,\pi,r)\Lambda(1-s,\pi,r^\vee).
$$
若另有 $r\circ\varphi_{\pi^\vee}\cong(r\circ\varphi_\pi)^\vee$，也可等价写成
$\Lambda(1-s,\pi^\vee,r)$；不得同时把 $\pi$ 和 $r$ 都换成对偶，除非另有明确同构。

## 9. Geometric Langlands 中的归一化

几何 Langlands 部分使用与局部 Langlands 相容的几何 Frobenius 口径。有限域上从 sheaf 到 function 时，trace function 必须声明使用几何 Frobenius 还是算术 Frobenius。

本书固定 trace function 为 stalk 上几何 Frobenius 的交错迹：
$$
t_{\mathcal F}(x)=
\sum_i(-1)^i\operatorname{tr}
\left(\operatorname{Fr}_x\mid H^i(\mathcal F_{\bar x})\right).
$$
若资料使用算术 Frobenius，则需在 Weil structure 中取逆。只有定义在有限域上并配备 Weil structure 的
sheaf 才有该 trace function；代数闭域上的裸 sheaf 不能直接取 Frobenius trace。

几何 Satake 中，perverse sheaves 的 cohomological shifts 和 Tate twists 不能省略。本文默认采用使
$$
\operatorname{Sat}_G:\operatorname{Perv}_{G(\mathcal O)}(\operatorname{Gr}_G)\simeq\operatorname{Rep}(\widehat G)
$$
成为 tensor equivalence 的归一化。若需要比较函数迹和球 Hecke 代数，则必须同时追踪 $q^{\langle\rho,\lambda\rangle}$ 因子。

特征零 categorical geometric Langlands 的自动侧固定为 determinant line 平方根 $\mu_2$-gerbe 上的 half-twisted category
$$
\operatorname{DMod}_{1/2}(\operatorname{Bun}_G).
$$
选择 $\omega_X^{1/2}$ 可给出它与 $\operatorname{DMod}(\operatorname{Bun}_G)$ 的识别，但 Hecke normalization 中不得省略下标 $1/2$，否则通常的 $\operatorname{Rep}(\widehat G)$ action 会出现 canonical central-gerbe twist。

Gaitsgory-Raskin proof series 的函子方向固定为
$$
\mathbb L_G:
\operatorname{DMod}_{1/2}(\operatorname{Bun}_G)
\longrightarrow
\operatorname{IndCoh}_{\mathcal N}(\operatorname{LocSys}_{\widehat G}(X)),
$$
即 automorphic $\to$ spectral。只有在调用已证明的等价后，谱侧对象到自动侧对象才写作 $\mathbb L_G^{-1}$；不得把反向函子仍标成 $\mathbb L_G$。

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
9. Categorical geometric Langlands 是否保留 $\operatorname{DMod}_{1/2}$，并按所用方向区分 $\mathbb L_G$ 与 $\mathbb L_G^{-1}$。

若任一项不同，正文必须写出转换，而不是只引用同名定理。
