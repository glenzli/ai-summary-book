# 附录 P：球 Hecke 代数、Cartan 分解和 Satake 同构

收口归一化回指：本附录是 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 第 4 节的主要支撑，所有球 Hecke 代数、Cartan 分解、Satake 变换和非分歧 L 因子计算均采用该 convention。

## P.1 设定和归一化

设 $F$ 为非 Archimedean 局部域，剩余域大小为 $q$。设 $G/F$ 为 split connected reductive group，$T\subset B=TN$ 为 split maximal torus 和 Borel subgroup，$K=G(\mathcal O_F)$ 为 hyperspecial maximal compact subgroup。设
$$
X_*(T)^+=\{\lambda\in X_*(T):\langle\alpha,\lambda\rangle\ge0\text{ for all simple roots }\alpha\}
$$
为 dominant coweights。对 $\lambda\in X_*(T)$，记
$$
\varpi^\lambda=\lambda(\varpi)\in T(F).
$$

**定义 P.1.** 球 Hecke 代数为
$$
\mathcal H(G,K)=C_c(K\backslash G(F)/K,\mathbb C)
$$
配备卷积乘法
$$
(f_1*f_2)(g)=\int_{G(F)}f_1(x)f_2(x^{-1}g)\,dx,
$$
其中 Haar 测度归一化为 $\operatorname{vol}(K)=1$。

**命题 P.2.** $\mathcal H(G,K)$ 是结合含幺代数，单位元为 $\mathbf 1_K$。

**证明.** 卷积结合律由附录 B 的 locally profinite group 卷积结合律给出。若 $f$ 为双 $K$ 不变函数，则 $\mathbf 1_K*f=f*\mathbf 1_K=f$，因为积分只在 $K$ 上取平均且 $f$ 左右 $K$ 不变。$\square$

## P.2 Cartan 分解

**外部输入定理 P.3（Cartan 分解）.** 在上述 split hyperspecial 设定下，
$$
G(F)=\bigsqcup_{\lambda\in X_*(T)^+}K\varpi^\lambda K.
$$
因此 $\{\mathbf 1_{K\varpi^\lambda K}\}_{\lambda\in X_*(T)^+}$ 构成 $\mathcal H(G,K)$ 的向量空间基。

**证明说明.** 一般情形依赖 Bruhat-Tits theory。对 $G=\operatorname{GL}_n$ 可用 Smith normal form 证明：任意 $g\in\operatorname{GL}_n(F)$ 可由左右 $\operatorname{GL}_n(\mathcal O_F)$ 初等变换化为
$$
\operatorname{diag}(\varpi^{a_1},\ldots,\varpi^{a_n}),\qquad a_1\ge\cdots\ge a_n,
$$
指数列唯一。$\square$

**定义 P.4.** 记
$$
T_\lambda=\mathbf 1_{K\varpi^\lambda K}\in\mathcal H(G,K).
$$
Cartan 分解使得每个 $f\in\mathcal H(G,K)$ 可唯一写成有限和
$$
f=\sum_{\lambda\in X_*(T)^+}a_\lambda T_\lambda.
$$

## P.3 Satake 变换

设 $\delta_B:B(F)\to\mathbb R_{>0}$ 为模 character：
$$
\delta_B(t)=|\det(\operatorname{Ad}(t)|_{\operatorname{Lie}N})|_F.
$$

**定义 P.5.** 归一化 Satake 变换为
$$
\mathcal S:\mathcal H(G,K)\to\mathbb C[X_*(T)]
$$
给定为
$$
\mathcal S(f)(t)=\delta_B(t)^{-1/2}\int_{N(F)}f(tn)\,dn,
$$
并通过 $T(F)/T(\mathcal O_F)\simeq X_*(T)$ 视为群代数上的函数。这里 $N(F)$ 的 Haar 测度取使 $N(\mathcal O_F)$ 体积为 $1$ 的归一化。

**命题 P.6.** $\mathcal S(f)$ 支撑有限。

**证明.** $f$ 紧支撑，故 $\operatorname{supp}(f)\cap T(F)N(F)$ 在 $T(F)/T(\mathcal O_F)$ 的投影只含有限多个 $T(\mathcal O_F)$-陪集。积分只可能在这些陪集上非零。$\square$

**外部输入定理 P.7（Satake 同构）.** Satake 变换给出代数同构
$$
\mathcal S:\mathcal H(G,K)\xrightarrow{\sim}\mathbb C[X_*(T)]^W,
$$
其中 $W$ 为 Weyl group。该同构与归一化非分歧主级数的 spherical vector eigencharacter 相容。

**证明说明.** 乘法性来自 Iwasawa 分解 $G(F)=B(F)K$ 和卷积积分的 Fubini 重排；像落在 $W$-不变量中使用 rank-one intertwining operators；满射和单射使用 Cartan 分解下的三角性：
$$
\mathcal S(T_\lambda)=q^{\langle\rho,\lambda\rangle}e^\lambda+\sum_{\mu<\lambda}c_{\lambda\mu}e^\mu
$$
在适当 dominance order 下成立。完整证明需处理根子群积分和 Weyl 群归一化。$\square$

## P.4 非分歧表示和 Satake 参数

**定义 P.8.** 不可约 smooth representation $\pi$ of $G(F)$ 称为 spherical，若
$$
\pi^K\ne0.
$$

**外部输入定理 P.9（一维球向量）.** 若 $\pi$ 不可约且 spherical，则
$$
\dim_\mathbb C\pi^K=1.
$$

**命题 P.10.** 不可约 spherical representation $\pi$ 给出 algebra homomorphism
$$
\chi_\pi:\mathcal H(G,K)\to\mathbb C.
$$

**证明.** 由定理 P.9，$\pi^K$ 为一维。Hecke 代数通过卷积算子保持 $\pi^K$，于是每个 $h\in\mathcal H(G,K)$ 在 $\pi^K$ 上为标量。卷积作用满足 $\pi(h_1*h_2)=\pi(h_1)\pi(h_2)$，故标量函数为代数同态。$\square$

**定义 P.11.** 由 Satake 同构，
$$
\operatorname{Hom}_{\operatorname{alg}}(\mathcal H(G,K),\mathbb C)
\simeq
\widehat T(\mathbb C)//W.
$$
不可约 spherical representation $\pi$ 的像称为它的 Satake parameter，记为
$$
s(\pi)\in\widehat T(\mathbb C)//W.
$$

**命题 P.12.** 若 $\pi$ 是归一化非分歧主级数
$$
\pi=\operatorname{Ind}_{B(F)}^{G(F)}(\chi)
$$
的 spherical constituent，则 $s(\pi)$ 等于非分歧 character $\chi:T(F)\to\mathbb C^\times$ 对应的 $\widehat T$ 半单共轭类。

**证明草图.** 归一化诱导中的 spherical vector $v_K$ 由 $v_K(1)=1$ 决定。对 $h\in\mathcal H(G,K)$，卷积作用在 $v_K(1)$ 处给出
$$
\int_{G(F)}h(g)v_K(g)\,dg.
$$
用 Iwasawa 分解 $g=tnk$ 重写积分，归一化因子正好产生定义 P.5 中的 $\delta_B^{-1/2}$。因此 Hecke eigencharacter 等于把 $\mathcal S(h)$ 在 $\chi$ 对应点求值。$\square$

## P.5 `GL(n)` 的显式公式

令 $G=\operatorname{GL}_n$，$K=\operatorname{GL}_n(\mathcal O_F)$，$T$ 为 diagonal torus。此时
$$
X_*(T)\simeq\mathbb Z^n,\qquad W=S_n.
$$
设非分歧主级数由 characters $\chi_i:F^\times\to\mathbb C^\times$ 给出，令
$$
\alpha_i=\chi_i(\varpi).
$$

**命题 P.13.** `GL(n)` 的 spherical Satake parameter 可写为无序 $n$ 元组
$$
\{\alpha_1,\ldots,\alpha_n\}\in(\mathbb C^\times)^n/S_n.
$$

**证明.** 对角 torus 的对偶为 $\widehat T=(\mathbb C^\times)^n$，Weyl 群 $S_n$ 置换坐标。命题 P.12 把非分歧 character 的 $\varpi$ 值送到 $\widehat T/W$，即上述无序元组。$\square$

**命题 P.14.** 对 `GL(n)` 标准表示，非分歧局部 L 因子为
$$
L(s,\pi,\operatorname{Std})=\prod_{i=1}^n(1-\alpha_iq^{-s})^{-1}.
$$

**证明.** 标准表示把 Satake parameter $\operatorname{diag}(\alpha_1,\ldots,\alpha_n)$ 作用在 $\mathbb C^n$ 上，其特征多项式为
$$
\prod_{i=1}^n(X-\alpha_i).
$$
按 Langlands 局部因子的非分歧定义，
$$
\det(1-\operatorname{Std}(s(\pi))q^{-s})^{-1}
=\prod_{i=1}^n(1-\alpha_iq^{-s})^{-1}.
$$
$\square$

## P.6 与几何 Satake 的关系

**命题 P.15.** 几何 Satake 的 sheaf-function trace 在有限域曲线的非分歧点处恢复 classical Satake 同构。

**证明草图.** 几何 Satake 把 $\operatorname{Rep}(\widehat G)$ 与 affine Grassmannian 上的 $G(\mathcal O)$-equivariant perverse sheaves 的卷积范畴等价。对定义在有限域上的对象取 Frobenius trace 函数，卷积对应函数卷积；Schubert cell 的 IC sheaf trace 给出球 Hecke 代数中的标准基元素的归一化版本。因此张量积规则在函数侧变为球 Hecke 代数乘法，得到 Satake 同构。完整证明依赖 sheaf-function dictionary 和 decomposition theorem。$\square$

## 练习

**练习 P.1.** 对 `GL(2)` 写出 Cartan 分解中的双陪集代表。

**练习 P.2.** 证明 $\mathbf 1_K$ 是 $\mathcal H(G,K)$ 的单位元。

**练习 P.3.** 对 `GL(2)` 的非分歧主级数，写出标准 L 因子的 Satake 参数公式。

**练习 P.4.** 说明归一化 Satake 变换中 $\delta_B^{-1/2}$ 的作用。

**练习 P.5.** 解释为什么几何 Satake 是 Satake 同构的范畴化。
