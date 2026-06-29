# 附录 K：Galois 变形、Selmer 群和 Taylor-Wiles 接口

本附录补充第九、十和九十章中“模性提升”“$R=T$”和 “Taylor-Wiles 方法” 的接口。完整证明是大型专题；本附录只建立定义、精确定理形状和逻辑角色，避免把模性提升写成无结构黑箱。

**收口归一化回指。** 本附录涉及残余 Galois 表示、Tate twist、局部变形条件、Hecke 特征值和 $R=T$ 比较；与模性提升和 Fermat 应用比较时使用 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 第 5、6、7、8 节。

## K.1 残余表示和变形问题

设 $p$ 为素数，$k$ 为有限域，$W(k)$ 为 Witt 向量环。设
$$
\overline\rho:G_{\mathbb Q,S}\to\operatorname{GL}_2(k)
$$
为连续表示，其中 $G_{\mathbb Q,S}$ 是最大在有限集合 $S$ 外非分歧的 Galois 群。

**定义 K.1.** 对 Artin local $W(k)$-algebra $A$，其 residue field 为 $k$。$\overline\rho$ 到 $A$ 的一个 lift 是连续表示
$$
\rho_A:G_{\mathbb Q,S}\to\operatorname{GL}_2(A)
$$
使得模 maximal ideal 约化后等于 $\overline\rho$。

**定义 K.2.** 两个 lifts $\rho_A,\rho_A'$ 称为 strict equivalent，若存在
$$
g\in1+M_2(\mathfrak m_A)
$$
使
$$
\rho_A'=g\rho_Ag^{-1}.
$$

**定义 K.3.** 变形函子
$$
D_{\overline\rho}:\operatorname{Art}_{W(k)}\to\operatorname{Sets}
$$
把 $A$ 送到 lifts 的 strict equivalence classes。

**外部输入定理 K.4（Mazur representability）.** 若 $\overline\rho$ 绝对不可约，则 $D_{\overline\rho}$ 可由 complete Noetherian local $W(k)$-algebra $R_{\overline\rho}$ pro-represent。也就是说存在 universal deformation
$$
\rho^{\operatorname{univ}}:G_{\mathbb Q,S}\to\operatorname{GL}_2(R_{\overline\rho})
$$
使任意 lift 由唯一局部同态 $R_{\overline\rho}\to A$ 诱导。

**命题 K.5.** 若 $\overline\rho$ 绝对不可约，则任意 lift 的 automorphism group 在 strict equivalence 口径下不产生额外标量模糊。

**证明草图.** 绝对不可约性和 Schur 引理说明与 $\overline\rho$ 交换的矩阵只有标量。Strict equivalence 中只允许约化为 $1$ 的共轭；标量自同构在固定残余表示和 determinant 条件后不产生非平凡变形同构。该刚性是 Mazur representability 的关键假设之一。$\square$

## K.2 局部变形条件

设 $v\in S$。

**定义 K.6.** 一个局部变形条件 $\mathcal D_v$ 是子函子
$$
\mathcal D_v\subset D_{\overline\rho|_{G_{\mathbb Q_v}}}
$$
满足 Schlessinger 型可表示性和稳定性条件。它规定允许的局部 lifts 类型，例如：

- unramified；
- fixed determinant；
- finite flat；
- ordinary；
- semistable；
- prescribed inertial type；
- minimal ramification。

**定义 K.7.** 全局带局部条件的变形函子
$$
D_{\mathcal S}
$$
由那些全局 lifts 组成，它们在每个 $v\in S$ 的限制属于 $\mathcal D_v$，并满足固定 determinant 条件。若可表示，其 universal ring 记为
$$
R_{\mathcal S}.
$$

**外部输入定理 K.8（局部变形环）.** 在 Taylor-Wiles 方法使用的标准局部条件下，局部变形条件由 complete Noetherian local rings 表示，并具有可计算 tangent space。许多 minimal 条件给出形式光滑或 complete intersection 型局部环。

**注 K.9.** “模性提升定理”的假设并不是一句“局部条件适当”。它需要逐个 $v$ 控制局部变形环的维数、平坦性、分支和与 Hecke 侧局部类型的匹配。

## K.3 Selmer 群和 Tangent Space

设
$$
\operatorname{ad}^0\overline\rho
$$
为 trace-zero adjoint representation。

**定义 K.10.** 给定局部条件 $\mathcal L_v\subset H^1(G_{\mathbb Q_v},\operatorname{ad}^0\overline\rho)$，Selmer 群定义为
$$
H^1_{\mathcal L}(G_{\mathbb Q,S},\operatorname{ad}^0\overline\rho)
=
\ker\left(
H^1(G_{\mathbb Q,S},\operatorname{ad}^0\overline\rho)
\to
\bigoplus_{v\in S}
\frac{H^1(G_{\mathbb Q_v},\operatorname{ad}^0\overline\rho)}{\mathcal L_v}
\right).
$$

**定义 K.11.** Dual Selmer group 使用 Tate dual
$$
(\operatorname{ad}^0\overline\rho)^*(1)
$$
和局部正交补 $\mathcal L_v^\perp$ 定义：
$$
H^1_{\mathcal L^\perp}(G_{\mathbb Q,S},(\operatorname{ad}^0\overline\rho)^*(1)).
$$

**外部输入定理 K.12（Poitou-Tate duality）.** Selmer 群与 dual Selmer 群满足全局 Euler characteristic 公式。该公式控制 $R_{\mathcal S}$ 的生成元数和关系数，并解释 Taylor-Wiles auxiliary primes 的数量。

**命题 K.13.** 变形函子 $D_{\mathcal S}$ 的 tangent space 可嵌入 Selmer 群
$$
H^1_{\mathcal L}(G_{\mathbb Q,S},\operatorname{ad}^0\overline\rho).
$$
在标准可表示局部条件下二者相等。

**证明草图.** 对双数环 $k[\epsilon]/(\epsilon^2)$ 的 lift 可写为
$$
\rho_\epsilon(g)=(1+\epsilon c(g))\overline\rho(g).
$$
表示条件等价于 $c$ 为 $1$-cocycle；strict equivalence 对应加上 coboundary。局部变形条件把 $c|_{G_{\mathbb Q_v}}$ 限制到 $\mathcal L_v$。故 tangent classes 正是满足局部条件的 cohomology classes。$\square$

## K.4 Hecke Algebra 和 Galois 表示

设 $N$ 为级，$p\nmid N$，考虑模形式空间中的 Hecke algebra $T$。令 $\mathfrak m$ 为由残余表示 $\overline\rho$ 对应的 maximal ideal。

**定义 K.14.** 局部化 Hecke algebra
$$
T_\mathfrak m
$$
是由 $T_\ell$、diamond operators 和必要的 $U_\ell$ 在相应模形式或同调空间上作用生成的有限 $W(k)$-algebra 的 $\mathfrak m$-adic 局部化。

**外部输入定理 K.15（Hecke 侧 Galois 表示）.** 对由模形式或模曲线同调构造、并按残余 maximal ideal $\mathfrak m$ 局部化的 Hecke algebra，且在残余不可约性和局部类型已固定的条件下，存在 Galois 表示
$$
\rho_T:G_{\mathbb Q,S}\to\operatorname{GL}_2(T_\mathfrak m)
$$
使得对 $\ell\notin S$，
$$
\operatorname{tr}\rho_T(\operatorname{Frob}_\ell^{\operatorname{arith}})=T_\ell,
\qquad
\det\rho_T(\operatorname{Frob}_\ell^{\operatorname{arith}})=\ell\langle\ell\rangle
$$
在权 $2$ convention 下成立。

**命题 K.16.** 若 $\rho_T$ 满足变形问题 $\mathcal S$ 的局部条件，则 universal property 给出自然同态
$$
R_{\mathcal S}\to T_\mathfrak m.
$$

**证明.** $\rho_T$ 是 $\overline\rho$ 到 $T_\mathfrak m$ 的 lift，并满足 $\mathcal S$ 的局部条件。由 $R_{\mathcal S}$ 表示该变形函子，$\rho_T$ 对应唯一局部 $W(k)$-algebra homomorphism
$$
R_{\mathcal S}\to T_\mathfrak m.
$$
$\square$

## K.5 `R=T` 和模性提升

**外部输入定理 K.17（Taylor-Wiles minimal `R=T`，接口形式）.** 在 residual representation $\overline\rho$ 奇、绝对不可约、残余模且满足 Taylor-Wiles 条件的情形，minimal deformation ring $R_{\min}$ 与 minimal Hecke algebra $T_{\min}$ 之间的自然映射
$$
R_{\min}\to T_{\min}
$$
是同构，并且二者为 complete intersection。

**外部输入定理 K.18（Taylor-Wiles patching，接口形式）.** 通过选择 auxiliary Taylor-Wiles primes $Q_n$，构造带有额外 level 的 rings $R_{Q_n}$ 和 Hecke modules $M_{Q_n}$。取逆极限或 patching 后得到大环 $R_\infty$ 和模 $M_\infty$，其深度、维数和 complete intersection 性质推出 $R=T$。

**命题 K.19（`R=T` 推出模性提升）.** 设
$$
\rho:G_{\mathbb Q,S}\to\operatorname{GL}_2(\mathcal O)
$$
是满足变形问题 $\mathcal S$ 的 lift，并由同态 $x_\rho:R_{\mathcal S}\to\mathcal O$ 给出。若
$$
R_{\mathcal S}\simeq T_\mathfrak m,
$$
则 $\rho$ 是模的，即来自相应 Hecke eigenform 或其 $p$-adic 系。

**证明.** 同构把 $x_\rho$ 视为 Hecke algebra 的 $\mathcal O$-值 system of eigenvalues。Hecke algebra 作用在模形式或相应同调空间上；局部化后该 system of eigenvalues 给出 Hecke eigenclass。由 Eichler-Shimura、Deligne 表示或相应 Galois 表示构造，所得 eigenclass 的 Galois 表示与 $\rho$ 在几乎所有 Frobenius trace 上相同。Chebotarev 和半单性给出同构。$\square$

## K.6 半稳定模性和费马应用

**外部输入定理 K.20（半稳定模性定理的变形论形状）.** 若 $E/\mathbb Q$ 为半稳定椭圆曲线，则可选取 Taylor-Wiles 论证允许的素数 $p$，使残余表示 $\overline\rho_{E,p}$ 进入 minimal 或 nearly minimal 变形问题。结合残余模性、必要时的 $3$-$5$ switch 和 `R=T`，得到 $p$-adic 表示 $\rho_{E,p}$ 模，进而 $E$ 模。

**命题 K.21.** 第九十章费马应用只使用 K.20 的结论，而不使用其证明内部结构。

**证明.** 第九十章逻辑链为：Frey 曲线半稳定；半稳定模性给出其模性；Ribet 降层给出级 $2$ newform；附录 D 给出该空间为零。该链只调用“半稳定曲线模”这一命题，不调用 deformation rings、Selmer groups 或 patching 的内部证明。$\square$

## K.7 本附录小结

本附录给出模性提升的逻辑骨架：

1. 残余表示 $\overline\rho$ 定义变形函子。
2. 局部条件切出 deformation problem $\mathcal S$。
3. Selmer 群控制 tangent space，dual Selmer 群控制关系。
4. Hecke algebra 上的 Galois 表示给出 $R_{\mathcal S}\to T_\mathfrak m$。
5. Taylor-Wiles patching 证明 $R=T$。
6. `R=T` 把 Galois lift 的点转化为 Hecke eigenform，从而给出模性提升。

## 练习

**练习 K.1.** 对 $k[\epsilon]/(\epsilon^2)$，验证 lift 条件等价于 $1$-cocycle 条件。

**练习 K.2.** 解释为什么绝对不可约假设在 Mazur representability 中不可随意去掉。

**练习 K.3.** 写出 fixed determinant 变形问题中 tangent representation 为什么是 $\operatorname{ad}^0\overline\rho$。

**练习 K.4.** 证明命题 K.16。

**练习 K.5.** 用命题 K.19 解释“$R=T$ 推出模性提升”的逻辑，而不引用 patching 细节。
