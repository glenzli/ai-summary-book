# 附录 A：代数数论复习

收口归一化回指：本附录固定赋值、剩余域大小、Frobenius、分歧、Artin 导子和类域论前置语言；与 adeles、reciprocity、Galois 表示和 L 因子比较时使用 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 第 1、2、3、6、8 节。

## A.1 整体域与位置

**定义 A.1.** 整体域是数域或有限域上的一变量函数域。其位置集合记为 $V_K$。对 $v\in V_K$，完备化为 $K_v$。

**定义 A.2.** 非 Archimedean 位置 $v$ 的整数环、极大理想和剩余域分别为
$$
\mathcal O_v,\qquad \mathfrak p_v,\qquad k_v.
$$
剩余域大小为 $q_v=\#k_v$。

**命题 A.3（乘积公式）.** 归一化绝对值可取使得
$$
\prod_{v\in V_K}|x|_v=1,\qquad x\in K^\times.
$$

**证明草图.** 数域情形由素理想分解和 Archimedean 绝对值的标准归一化得到；函数域情形由主除子的次数为零得到。$\square$

## A.2 分解群、惯性群和 Frobenius

设 $L/K$ 为 Galois 扩张，$w\mid v$。

**定义 A.4.** 设 $L/K$ 为有限扩张，$v$ 为 $K$ 的非 Archimedean 位置。若
$$
\mathfrak p_v\mathcal O_L=\prod_{w\mid v}\mathfrak p_w^{e(w/v)},
$$
则 $e(w/v)$ 称为 ramification index。剩余域扩张次数
$$
f(w/v)=[k_w:k_v]
$$
称为 residue degree。若 $L/K$ 为 Galois，则所有 $w\mid v$ 的 $e(w/v)$ 相同，所有 $f(w/v)$ 相同，并记为 $e_v$、$f_v$；$w\mid v$ 的个数记为 $g_v$。

**外部输入 A.4.1（素理想分解公式）.** 对有限可分扩张 $L/K$ 和非 Archimedean 位置 $v$，有
$$
\sum_{w\mid v}e(w/v)f(w/v)=[L:K].
$$
若 $L/K$ 为 Galois，则
$$
e_vf_vg_v=[L:K].
$$

该定理是 Dedekind 域整数闭包和局部化理论的标准结果。

**定义 A.4.2.** 若 $L/K$ 为 Galois，分解群和惯性群分别为
$$
D_w=\{\sigma\in\operatorname{Gal}(L/K):\sigma w=w\},
$$
$$
I_w=\ker(D_w\to\operatorname{Gal}(k_w/k_v)).
$$

**命题 A.4.3.** 设 $L/K$ 为有限 Galois 扩张。则 $D_w$ 自然同构于局部扩张 $L_w/K_v$ 的 Galois 群：
$$
D_w\simeq\operatorname{Gal}(L_w/K_v).
$$
并且有精确列
$$
1\to I_w\to D_w\to\operatorname{Gal}(k_w/k_v)\to1.
$$

**证明草图.** 元素 $\sigma\in D_w$ 保持 $w$，故连续作用于 $L$ 在 $w$ 处的完备化 $L_w$，并固定 $K_v$，得到 $D_w\to\operatorname{Gal}(L_w/K_v)$。反向由 $L$ 在 $L_w$ 中稠密和 $K$-嵌入延拓唯一性给出。对整数环取作用，$D_w$ 保持 $\mathcal O_w$ 和 $\mathfrak p_w$，因此作用在 $k_w$ 上；核正是按定义在剩余域上平凡的元素。剩余域有限域扩张为 Galois，且局部 Galois 理论给出满射。$\square$

**推论 A.4.4.** 在有限 Galois 情形，
$$
|D_w|=e_vf_v,\qquad |I_w|=e_v.
$$

**证明.** Galois 群 $\operatorname{Gal}(L/K)$ 传递作用在 $w\mid v$ 的集合上，稳定子为 $D_w$。由 orbit-stabilizer，
$$
|\operatorname{Gal}(L/K)|=g_v|D_w|.
$$
结合 A.4.1 的 $e_vf_vg_v=[L:K]$ 得 $|D_w|=e_vf_v$。精确列 A.4.3 的商为 $\operatorname{Gal}(k_w/k_v)$，其阶为 $f_v$，故 $|I_w|=e_v$。$\square$

**定义 A.5.** 若 $w/v$ 非分歧，算术 Frobenius 为
$$
\operatorname{Frob}_w^{\operatorname{arith}}(x)=x^{q_v}
$$
在剩余域上的作用。几何 Frobenius 是其逆。

**命题 A.5.1.** $w/v$ 非分歧当且仅当 $I_w=1$。此时 $D_w$ 由 Frobenius 元生成，并有
$$
D_w\simeq\operatorname{Gal}(k_w/k_v).
$$

**证明.** 非分歧按定义等价于 $e_v=1$ 且剩余域扩张可分；有限域扩张自动可分。由推论 A.4.4，$|I_w|=e_v$，故非分歧等价于 $I_w=1$。此时 A.4.3 给出 $D_w\simeq\operatorname{Gal}(k_w/k_v)$，后者由算术 Frobenius 生成；采用几何 Frobenius convention 时生成元取其逆。$\square$

**注 A.6.** 本书局部类域论和局部 L 因子默认几何 Frobenius；模形式和 $\ell$-adic Galois 表示章节常用算术 Frobenius。比较时必须取逆或调整 Tate twist。

**定义 A.6.1（lower ramification groups）.** 对有限 Galois 局部扩张 $E/F$，令 $G=\operatorname{Gal}(E/F)$，整数环为 $\mathcal O_E$，极大理想为 $\mathfrak p_E$。对 $i\ge -1$ 定义
$$
G_i=\{\sigma\in G:\sigma(x)-x\in\mathfrak p_E^{i+1}\text{ for all }x\in\mathcal O_E\}.
$$
于是
$$
G_{-1}=G,\qquad G_0=I_{E/F}.
$$
$G_1$ 称为 wild inertia subgroup。

**命题 A.6.2.** $G_i$ 构成递降正规子群列
$$
G=G_{-1}\supset G_0\supset G_1\supset\cdots
$$
且当 $i$ 足够大时 $G_i=1$。

**证明草图.** 若 $\sigma(x)-x\in\mathfrak p_E^{i+2}$，则也在 $\mathfrak p_E^{i+1}$ 中，故递降。正规性来自 valuation 被 Galois 作用保持：
$$
\tau\sigma\tau^{-1}(x)-x
=\tau\left(\sigma(\tau^{-1}x)-\tau^{-1}x\right),
$$
右侧的 valuation 条件与 $\sigma$ 的条件相同。有限扩张中非平凡 automorphism 在某个有限 jet 上可被检测，因此高阶同余最终只剩恒等元。$\square$

**外部输入 A.6.3（ramification quotient structure）.** 对有限 Galois 局部扩张，$G_0/G_1$ 是阶与 residue characteristic 互素的 cyclic group；对 $i\ge1$，$G_i/G_{i+1}$ 是 elementary Abelian $p$-group，其中 $p=\operatorname{char}k_F$。Upper numbering filtration 由 Herbrand 函数从 lower numbering 得到，并与取商相容。

本书在 Artin conductor、Swan conductor、Ribet 降层和局部 Langlands 分歧条件中使用该结构，但不重证 Herbrand 理论。

## A.3 Artin 导子

**定义 A.7.** 设 $F$ 为非 Archimedean 局部域，$\rho:G_F\to\operatorname{GL}(V)$ 为有限像复表示。其 Artin 导子指数定义为
$$
a(\rho)=\sum_{i\ge0}\frac{1}{[G_0:G_i]}\operatorname{codim}V^{G_i},
$$
其中 $G_i$ 为 lower numbering ramification groups。

**注 A.8.** 对 $\ell$-adic 表示，Artin 导子通过有限惯性商和 Swan conductor 定义。正文中 residual conductor 和椭圆曲线 conductor 均依赖该局部不变量。

## A.4 类域论接口

**外部输入定理 A.9（局部类域论）.** 对局部域 $F$，存在 reciprocity map
$$
\operatorname{rec}_F:F^\times\to W_F^{\operatorname{ab}}
$$
使一致化元对应几何 Frobenius，并诱导有限 Abel 扩张与开有限指数子群之间的对应。

**外部输入定理 A.10（全局类域论）.** 对整体域 $K$，存在 reciprocity map
$$
\operatorname{rec}_K:K^\times\backslash\mathbb A_K^\times\to G_K^{\operatorname{ab}}
$$
满足局部-整体相容性。其核的连通或闭包部分依赖数域/函数域情形和归一化。

**注 A.11.** 第三章把这些定理作为 `GL(1)` Langlands 的基础。完整证明需要 ideles、ray class groups、norm subgroup theorem 和 cohomological class formation。

## A.5 Chebotarev 密度

**外部输入定理 A.12（Chebotarev）.** 设 $L/K$ 为有限 Galois 扩张。每个共轭类 $C\subset\operatorname{Gal}(L/K)$ 在非分歧素点的 Frobenius 共轭类中以密度
$$
\frac{\#C}{\#\operatorname{Gal}(L/K)}
$$
出现。

**推论 A.13.** 两个连续半单 $\ell$-adic 表示若在几乎所有非分歧位置有相同 Frobenius characteristic polynomial，则它们半单同构。

**证明草图.** Frobenius 共轭类在 Galois 群中稠密。Brauer-Nesbitt 定理说明半单表示由 character 决定。$\square$

## A.6 非 Archimedean 赋值和分式理想

**定义 A.14.** 设 $K$ 为数域，$\mathcal O_K$ 为其整数环。对非零素理想 $\mathfrak p\subset\mathcal O_K$，离散赋值
$$
v_\mathfrak p:K^\times\to\mathbb Z
$$
由分式理想分解
$$
(x)=\prod_{\mathfrak p}\mathfrak p^{v_\mathfrak p(x)}
$$
定义。归一化非 Archimedean 绝对值为
$$
|x|_\mathfrak p=N(\mathfrak p)^{-v_\mathfrak p(x)}.
$$

**命题 A.15.** 对任意 $x\in K^\times$，只有有限多个 $\mathfrak p$ 满足 $v_\mathfrak p(x)\ne0$。

**证明.** 分式理想 $(x)$ 是有限生成 $\mathcal O_K$-模的秩一可逆分式理想。Dedekind 域中每个非零分式理想唯一分解为有限个非零素理想的整数次幂乘积。因此除有限多个 $\mathfrak p$ 外指数为 $0$。$\square$

**命题 A.16（数域乘积公式的理想部分）.** 对 $x\in K^\times$，
$$
\prod_{\mathfrak p\nmid\infty}|x|_\mathfrak p
=|N_{K/\mathbb Q}(x)|^{-1}.
$$

**证明.** 由定义，
$$
\prod_{\mathfrak p\nmid\infty}|x|_\mathfrak p
=\prod_{\mathfrak p}N(\mathfrak p)^{-v_\mathfrak p(x)}.
$$
分式理想范数满足
$$
N((x))=\prod_{\mathfrak p}N(\mathfrak p)^{v_\mathfrak p(x)}=|N_{K/\mathbb Q}(x)|.
$$
取倒数得到公式。$\square$

**推论 A.17（数域乘积公式）.** 若 Archimedean 绝对值按
$$
|x|_v=
\begin{cases}
|\sigma_v(x)|,&K_v\simeq\mathbb R,\\
|\sigma_v(x)|^2,&K_v\simeq\mathbb C
\end{cases}
$$
归一化，则
$$
\prod_{v\in V_K}|x|_v=1.
$$

**证明.** Archimedean 部分满足
$$
\prod_{v\mid\infty}|x|_v=|N_{K/\mathbb Q}(x)|.
$$
与命题 A.16 相乘即得结论。$\square$

## A.7 Ray Class Groups 和 Idele Class Group

**定义 A.18.** 设 $K$ 为数域，模数（modulus）$\mathfrak m$ 由有限部分 $\mathfrak m_f$ 和实嵌入集合 $\mathfrak m_\infty$ 组成。记 $I^\mathfrak m_K$ 为与 $\mathfrak m_f$ 互素的分式理想群。定义
$$
P^\mathfrak m_K=\{(a):a\in K^\times,\ a\equiv1\pmod{\mathfrak m_f},\ \sigma(a)>0\text{ for }\sigma\in\mathfrak m_\infty\}.
$$
Ray class group 为
$$
\operatorname{Cl}_\mathfrak m(K)=I^\mathfrak m_K/P^\mathfrak m_K.
$$

**命题 A.19.** $\operatorname{Cl}_\mathfrak m(K)$ 是有限 Abel 群。

**证明草图.** 理想类群有限。自然映射 $I_K^\mathfrak m\to\operatorname{Cl}(K)$ 的核由主理想控制。模 $\mathfrak m_f$ 的同余条件只引入有限商 $(\mathcal O_K/\mathfrak m_f)^\times$ 和有限个实符号条件。因此 ray class group 是有限群。完整证明需要 Dirichlet 单位定理和理想类群有限性。$\square$

**定义 A.20.** 与模数 $\mathfrak m$ 对应的开子群 $U(\mathfrak m)\subset\mathbb A_{K,f}^\times$ 定义为
$$
U(\mathfrak m)=
\prod_{\mathfrak p\nmid\mathfrak m_f}\mathcal O_\mathfrak p^\times
\times
\prod_{\mathfrak p^n\Vert\mathfrak m_f}(1+\mathfrak p^n\mathcal O_\mathfrak p).
$$
若含实符号条件，则再要求相应实分量为正。

**命题 A.21.** Ray class group 可由 idele class group 的开紧商描述：
$$
\operatorname{Cl}_\mathfrak m(K)\simeq
K^\times\backslash\mathbb A_K^\times/
\left(K_\infty^{\mathfrak m,+}\cdot U(\mathfrak m)\right),
$$
其中 $K_\infty^{\mathfrak m,+}$ 表示满足 $\mathfrak m_\infty$ 符号条件的 Archimedean 连通部分。

**证明草图.** 给 idele $x=(x_v)_v$ 关联分式理想
$$
\mathfrak a(x)=\prod_{\mathfrak p}\mathfrak p^{v_\mathfrak p(x_\mathfrak p)}.
$$
有限 idele 的 restricted product 条件保证该乘积有限。右乘 $U(\mathfrak m)$ 不改变与 $\mathfrak m_f$ 互素部分的 ray class；左乘 $K^\times$ 正对应主理想和同余条件。核与像的检查给出同构。$\square$

**注 A.22.** 全局类域论可视为当 $\mathfrak m$ 变化时，ray class groups 的逆极限与 $G_K^{\operatorname{ab}}$ 的有限 Abel 商之间的相容对应。第三章使用的是该理论的 idelic 总结形式。

## A.8 Norm Subgroups 和 Reciprocity 的唯一性口径

**定义 A.23.** 若 $L/K$ 为有限扩张，idele norm
$$
N_{L/K}:\mathbb A_L^\times\to\mathbb A_K^\times
$$
逐位置由局部 norm 映射拼合而成。它诱导
$$
N_{L/K}:C_L\to C_K.
$$

**外部输入定理 A.24（Norm subgroup theorem）.** 若 $L/K$ 为有限 Abel 扩张，则
$$
C_K/N_{L/K}C_L\simeq\operatorname{Gal}(L/K)
$$
由全局 reciprocity map 诱导。并且有限指数开子群中恰好出现这些 norm subgroups。

**注 A.25.** 该定理是全局类域论“存在定理”的核心内容之一。本书不证明它，但第三章中“有限阶 Hecke 特征等价于有限像一维 Galois 表示”的双射依赖此结构。

## A.9 Artin 导子的基本性质

**命题 A.26.** 设 $\rho_1,\rho_2$ 为有限像复表示，则
$$
a(\rho_1\oplus\rho_2)=a(\rho_1)+a(\rho_2).
$$

**证明.** 对每个 ramification group $G_i$，
$$
(\rho_1\oplus\rho_2)^{G_i}=\rho_1^{G_i}\oplus\rho_2^{G_i}.
$$
因此
$$
\operatorname{codim}(V_1\oplus V_2)^{G_i}
=\operatorname{codim}V_1^{G_i}+\operatorname{codim}V_2^{G_i}.
$$
代入定义 A.7 并逐项相加。$\square$

**命题 A.27.** 若 $\rho$ 非分歧，即 $I_F$ 在 $V$ 上平凡，则 $a(\rho)=0$。

**证明.** 对 $i\ge0$，lower numbering ramification groups $G_i$ 均包含在惯性群或其子群中。若惯性作用平凡，则 $V^{G_i}=V$，每一项 codimension 均为 $0$。$\square$

**命题 A.28.** 若 $\chi:F^\times\to\mathbb C^\times$ 为一维分歧特征，且 $n$ 是使 $\chi$ 在 $1+\mathfrak p_F^n$ 上平凡的最小非负整数，则在局部类域论对应下，一维 Artin 导子指数等于该乘法导子指数。

**证明草图.** 局部 reciprocity map 把单位滤过 $\mathcal O_F^\times\supset1+\mathfrak p_F^n$ 与 Weil 群 Abel 化中的惯性和高阶分歧滤过相匹配。于是 $\chi$ 在单位滤过中的首次平凡层与对应 Weil 特征在 ramification filtration 中的首次平凡层一致。一维 Artin 导子定义正记录这一层数。完整证明属于局部类域论的导子相容定理，即正文外部输入定理 3.9。$\square$
