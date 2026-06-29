# 附录 T：模曲线上同调、Eichler-Shimura 和 Deligne 表示

收口归一化回指：本附录比较 Hecke correspondence、Deligne 表示、Frobenius trace 和 classical modular form normalization；见 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 第 5、6、7 节。

## T.1 模曲线和局部系统

设 $N\ge1$，令 $Y_1(N)$ 为参数化带 $\Gamma_1(N)$-level structure 的椭圆曲线的模曲线，$X_1(N)$ 为其紧化。令
$$
\pi:\mathcal E\to Y_1(N)
$$
为 universal elliptic curve，在适当精细 level 或 stack 口径下理解。

**定义 T.1.** 对 $k\ge2$，令
$$
\mathcal V_{k-2,\ell}=\operatorname{Sym}^{k-2}R^1\pi_*\mathbb Q_\ell
$$
为 $Y_1(N)$ 上的 $\ell$-adic local system。Betti 口径下相应 local system 记为
$$
\mathcal V_{k-2,\mathbb C}=\operatorname{Sym}^{k-2}R^1\pi_*\mathbb C.
$$

**外部输入定理 T.2（模曲线与局部系统）.** 模曲线 $Y_1(N)$、$X_1(N)$ 及 local systems $\mathcal V_{k-2}$ 可在 $\mathbb Z[1/N]$ 上构造，并带有 Hecke correspondences 的自然作用。

**注 T.3.** 本附录只固定 cohomological construction 的接口。Stack、fine moduli 和 cusp 处延拓的技术细节作为外部输入。

## T.2 Hecke correspondences on cohomology

**定义 T.4.** 对素数 $p\nmid N$，Hecke correspondence $T_p$ 由 degree $p$ cyclic isogenies 给出：
$$
X_1(N)\xleftarrow{q_1}C_p\xrightarrow{q_2}X_1(N).
$$
它在 cohomology 上的作用定义为
$$
T_p=q_{2,*}q_1^*.
$$
Diamond operator $\langle d\rangle$ 来自 level structure 的自同构。

**命题 T.5.** Hecke correspondences 在 cohomology 上两两交换，并与 Galois 作用交换。

**证明草图.** Hecke correspondences 由模问题中的 isogeny diagrams 定义。不同素数的 cyclic isogenies 可按有限 subgroup 的直和交换次序，从而给出 correspondences 的交换关系。它们定义在 $\mathbb Q$ 或 $\mathbb Z[1/Np]$ 上，因此在 $\ell$-adic cohomology 上与绝对 Galois 作用交换。完整证明需使用 correspondences 的 proper pushforward/base-change formalism。$\square$

## T.3 Eichler-Shimura 同构

令 $S_k(\Gamma_1(N),\varepsilon)$ 为权 $k$、Nebentypus $\varepsilon$ 的 cusp forms 空间。

**外部输入定理 T.6（Eichler-Shimura）.** Betti cohomology 中存在 Hecke-equivariant 分解
$$
H^1_c(Y_1(N)(\mathbb C),\mathcal V_{k-2,\mathbb C})
\otimes_{\mathbb Q}\mathbb C
\supset
S_k(\Gamma_1(N))\oplus\overline{S_k(\Gamma_1(N))}
$$
并在 parabolic cohomology 中给出精确对应。Hecke 算子 $T_p$ 在 cusp form 侧的本征值与 cohomology 侧一致。

**命题 T.7.** 若 $f$ 为归一化 Hecke eigenform，则 $f$ 给出 Hecke algebra 在相应 cohomology 子商上的 character。

**证明.** Eichler-Shimura 同构把 $f$ 所在的一维 simultaneous eigenspace 嵌入 parabolic cohomology 的复化。Hecke 算子在该 eigenspace 上按 $a_p(f)$ 和 diamond eigenvalues 作用。因此 Hecke algebra 通过
$$
T_p\mapsto a_p(f),\qquad \langle d\rangle\mapsto\varepsilon(d)
$$
给出 character。$\square$

## T.4 Deligne 的 Galois 表示

**外部输入定理 T.8（Deligne 表示）.** 设 $f=\sum a_nq^n$ 为权 $k\ge2$、level $N$、Nebentypus $\varepsilon$ 的归一化 cuspidal Hecke eigenform。对每个素数 $\ell$ 和嵌入 coefficient field 到 $\overline{\mathbb Q}_\ell$，存在连续半单表示
$$
\rho_{f,\ell}:G_\mathbb Q\to\operatorname{GL}_2(\overline{\mathbb Q}_\ell)
$$
使得对 $p\nmid N\ell$，
$$
\operatorname{tr}\rho_{f,\ell}(\operatorname{Frob}^{\operatorname{arith}}_p)=a_p,
\qquad
\det\rho_{f,\ell}(\operatorname{Frob}^{\operatorname{arith}}_p)=\varepsilon(p)p^{k-1}.
$$

**命题 T.9.** 定理 T.8 给出局部 Euler 因子相容：
$$
L_p(s,\rho_{f,\ell})
=
\left(1-a_pp^{-s}+\varepsilon(p)p^{k-1}p^{-2s}\right)^{-1}
$$
对 $p\nmid N\ell$ 成立。

**证明.** 对二维表示，
$$
L_p(s,\rho)=\det(1-\rho(\operatorname{Frob}^{\operatorname{arith}}_p)p^{-s})^{-1}.
$$
将 trace 和 determinant 代入 characteristic polynomial
$$
X^2-a_pX+\varepsilon(p)p^{k-1}
$$
得到公式。$\square$

**注 T.10.** 第七章的 automorphic normalization 常使用几何 Frobenius 或 unitary twist。与定理 T.8 比较时必须检查取逆、对偶和 Tate twist。

## T.5 Weight two 和椭圆曲线

**外部输入定理 T.11（Eichler-Shimura construction for weight two）.** 对权 $2$ newform $f$，存在 Abelian variety $A_f/\mathbb Q$，其 Tate module 的一个二维因子给出 $\rho_{f,\ell}$。当 $f$ 有有理 Fourier coefficients 且对应椭圆曲线时，$A_f$ 为椭圆曲线 $E_f$。

**命题 T.12.** 若椭圆曲线 $E/\mathbb Q$ 对应权 $2$ newform $f$，则对好素数 $p\nmid N_E\ell$，
$$
a_p(f)=p+1-\#E(\mathbb F_p).
$$

**证明.** 模性给出 $V_\ell(E)$ 与 $\rho_{f,\ell}$ 同构或半单同构。椭圆曲线的 Grothendieck-Lefschetz trace formula 给出
$$
\#E(\mathbb F_p)=p+1-\operatorname{tr}(\operatorname{Frob}^{\operatorname{arith}}_p\mid V_\ell(E)).
$$
由定理 T.8，后一个 trace 为 $a_p(f)$。$\square$

## T.6 Residual representations

设 $\mathcal O$ 为 coefficient field 的整数环，$\lambda\mid\ell$。

**定义 T.13.** 取 $\rho_{f,\lambda}$ 的 Galois 稳定格，约化得到
$$
\overline\rho_{f,\lambda}:G_\mathbb Q\to\operatorname{GL}_2(\overline{\mathbb F}_\ell)
$$
的半单化，称为 $f$ 的 residual representation。

**命题 T.14.** 对 $p\nmid N\ell$，
$$
\operatorname{tr}\overline\rho_{f,\lambda}(\operatorname{Frob}^{\operatorname{arith}}_p)\equiv a_p\pmod\lambda,
\qquad
\det\overline\rho_{f,\lambda}(\operatorname{Frob}^{\operatorname{arith}}_p)\equiv\varepsilon(p)p^{k-1}\pmod\lambda.
$$

**证明.** 选择稳定格后，$\rho_{f,\lambda}(\operatorname{Frob}_p)$ 的 characteristic polynomial 有 $\mathcal O_\lambda$ 系数。模 $\lambda$ 约化 characteristic polynomial，trace 和 determinant 分别约化为上述同余。半单化不改变 characteristic polynomial。$\square$

## T.7 费马应用中的位置

**命题 T.15.** 费马应用章使用 Deligne 表示的方式只需要 Hecke-Frobenius trace 相容和 residual representation。

**证明.** 第九十章需要把椭圆曲线或模形式产生的 mod $\ell$ Galois 表示放入 Ribet 降层和模性提升框架。该过程使用：

1. 模形式给出 $\ell$-adic 表示；
2. 好素数处 trace 等于 Hecke eigenvalue；
3. 约化后得到 residual representation；
4. level lowering 比较 residual representation 的导子和 Hecke eigenvalues。

这些均由定理 T.8 和定义 T.13 提供。它不需要重新证明 Eichler-Shimura 或 Deligne 的完整构造。$\square$

## 练习

**练习 T.1.** 从 trace 和 determinant 推导定理 T.8 的 Euler 因子公式。

**练习 T.2.** 解释 Eichler-Shimura 中为什么出现 $S_k$ 和其复共轭两份。

**练习 T.3.** 说明 residual representation 为什么只在半单化意义下与稳定格无关。

**练习 T.4.** 对权 $2$ newform，解释 $a_p=p+1-\#E(\mathbb F_p)$ 的来源。

**练习 T.5.** 说明 Deligne 表示在费马应用链中的最小使用范围。
