# 附录 C：Smooth Admissible Representations

收口归一化回指：本附录涉及 smooth vectors、Hecke idempotents、compact induction、Jacquet modules 和归一化抛物诱导；与局部 Langlands、Satake 和谱分解比较时使用 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 第 3、4、8 节。

## C.1 光滑表示

设 $G$ 为 locally profinite group。

**定义 C.1.** 复表示 $(\pi,V)$ 称为 smooth，若每个 $v\in V$ 的 stabilizer
$$
G_v=\{g\in G:\pi(g)v=v\}
$$
为开子群。

**定义 C.2.** 若 $J\subset G$ 为开紧子群，$J$-不变量为
$$
V^J=\{v\in V:\pi(j)v=v,\ j\in J\}.
$$

**定义 C.3.** Smooth representation 称为 admissible，若对每个开紧 $J\subset G$，
$$
\dim_\mathbb C V^J<\infty.
$$

## C.2 Hecke 代数作用

**命题 C.4.** 若 $(\pi,V)$ 为 smooth representation，$f\in C_c^\infty(G)$，则
$$
\pi(f)v=\int_G f(g)\pi(g)v\,dg
$$
良定义。

**证明草图.** 因 $f$ 紧支撑且局部常值，支撑可分解为有限多个开紧陪集；对 smooth vector，积分化为有限和。$\square$

**命题 C.5.** 若 $v\in V^J$ 且 $f\in\mathcal H(G,J)$，则 $\pi(f)v\in V^J$。

**证明.** $f$ 为 $J$-双不变。对 $j\in J$，
$$
\pi(j)\pi(f)v=\int f(g)\pi(jg)v\,dg
=\int f(j^{-1}g')\pi(g')v\,dg'
=\pi(f)v.
$$
$\square$

## C.3 归一化抛物诱导

设 $G$ 为 reductive $p$-adic group，$P=MN$ 为 parabolic subgroup。

**定义 C.6.** 对 $M$ 的 smooth representation $\sigma$，归一化抛物诱导定义为
$$
\operatorname{Ind}_{P}^{G}(\sigma)
$$
其空间由 smooth functions $f:G\to V_\sigma$ 构成，满足
$$
f(mng)=\delta_P(m)^{1/2}\sigma(m)f(g).
$$

**注 C.7.** $\delta_P^{1/2}$ 的归一化使 unitary 表示诱导后仍有良好 unitary 性质，并与局部 Langlands 的 tempered convention 相容。

## C.4 Jacquet 模

**定义 C.8.** 对 smooth $G$-表示 $V$，其 Jacquet module 为
$$
V_N=V/\langle \pi(n)v-v:n\in N,\ v\in V\rangle.
$$
它自然是 $M$ 的表示。

**外部输入定理 C.9（Jacquet functor 的基本性质）.** Jacquet functor 与抛物诱导满足 Frobenius reciprocity，并在 Bernstein-Zelevinsky 分类、Langlands quotient theorem 和 constant term 计算中起核心作用。

## C.5 Matrix Coefficients

**定义 C.10.** 若 $(\pi,V)$ 为 smooth representation，$\ell\in V^\vee$，$v\in V$，matrix coefficient 为
$$
g\mapsto \ell(\pi(g)v).
$$

**定义 C.11.** 表示称为 tempered，若其 matrix coefficients 满足 Harish-Chandra 的 tempered 增长条件。称为 square-integrable modulo center，若适当 matrix coefficients 在 $G/Z_G$ 上平方可积。

**注 C.12.** 本书正文把 tempered 与 discrete series 的精确定理作为外部输入；附录只固定术语接口。

## C.6 Schur 引理和中心特征

**命题 C.13（Schur 引理，复向量空间形式）.** 设 $(\pi,V)$ 为群 $G$ 的不可约复表示。若 $\dim_\mathbb C V$ 有限，则
$$
\operatorname{End}_G(V)=\mathbb C.
$$

**证明.** 任取 $T\in\operatorname{End}_G(V)$。因 $\mathbb C$ 代数闭且 $V$ 有限维，$T$ 有特征值 $\lambda$。于是 $\ker(T-\lambda)$ 是非零 $G$-稳定子空间。不可约性给出 $\ker(T-\lambda)=V$，即 $T=\lambda\operatorname{id}_V$。$\square$

**外部输入定理 C.14（Schur 引理，admissible 光滑形式）.** 若 $G$ 为 reductive $p$-adic group，$(\pi,V)$ 为不可约 admissible smooth complex representation，则
$$
\operatorname{End}_G(V)=\mathbb C.
$$

**注 C.15.** 无限维情形需要 admissibility 或更一般的 Schur 性条件。正文第十二章中心特征命题在该定理的口径下使用。

**命题 C.16（中心特征）.** 设 $G$ 为 reductive $p$-adic group，$Z$ 为其中心。若 $(\pi,V)$ 为不可约 admissible smooth complex representation，则存在唯一 smooth character
$$
\omega_\pi:Z\to\mathbb C^\times
$$
使得
$$
\pi(z)=\omega_\pi(z)\operatorname{id}_V,\qquad z\in Z.
$$

**证明.** 对每个 $z\in Z$，算子 $\pi(z)$ 与所有 $\pi(g)$ 交换，因为 $zg=gz$。由 C.14，$\pi(z)$ 为某个标量 $\omega_\pi(z)$。表示性质给出 $\omega_\pi(zz')=\omega_\pi(z)\omega_\pi(z')$。光滑性来自 $V$ 的 smooth 性：取非零 $v\in V$，其 stabilizer 在 $G$ 中开，交 $Z$ 后得到 $\omega_\pi$ 在单位元邻域上平凡。唯一性由 $V\ne0$ 得到。$\square$

## C.7 Smooth Dual 和 Contragredient

**定义 C.17.** 设 $(\pi,V)$ 为 smooth representation。代数对偶 $V^\vee=\operatorname{Hom}_\mathbb C(V,\mathbb C)$ 上有 contragredient 作用
$$
(\pi^\vee(g)\lambda)(v)=\lambda(\pi(g^{-1})v).
$$
smooth dual 定义为
$$
V^\vee_{\operatorname{sm}}=\{\lambda\in V^\vee:\operatorname{Stab}_G(\lambda)\text{ 开}\}.
$$
相应表示记为 $\pi^\vee_{\operatorname{sm}}$。

**命题 C.18.** $V^\vee_{\operatorname{sm}}$ 是 $V^\vee$ 的 $G$-稳定子空间，且 $\pi^\vee_{\operatorname{sm}}$ 是 smooth representation。

**证明.** 若 $\lambda$ 的 stabilizer 为开子群 $J$，则对 $g\in G$，$\pi^\vee(g)\lambda$ 的 stabilizer 包含 $gJg^{-1}$，仍为开子群。因此 $V^\vee_{\operatorname{sm}}$ 稳定。定义本身说明每个向量有开 stabilizer，所以表示 smooth。$\square$

**命题 C.19.** 若 $J\subset G$ 为开紧子群，则有自然配对
$$
V^J\times (V^\vee_{\operatorname{sm}})^J\to\mathbb C.
$$
若 $V$ admissible，则 $(V^\vee_{\operatorname{sm}})^J$ 可识别为 $(V^J)^\vee$。

**证明草图.** 限制泛函给出映射
$$
(V^\vee_{\operatorname{sm}})^J\to(V^J)^\vee.
$$
反向构造可用平均幂等元 $e_J$：给定 $\ell\in(V^J)^\vee$，定义
$$
\widetilde\ell(v)=\ell(\pi(e_J)v).
$$
命题 B.11 给出 $\pi(e_J)v\in V^J$，且 $\widetilde\ell$ 为 $J$-不变 smooth functional。admissibility 保证有限维对偶操作不引入拓扑完备化问题。$\square$

## C.8 有限长度和可容许性

**定义 C.20.** Smooth representation $V$ 称为有限长度，若存在有限 filtration
$$
0=V_0\subset V_1\subset\cdots\subset V_r=V
$$
使每个 $V_i/V_{i-1}$ 不可约。

**命题 C.21.** 若 $0\to V_1\to V\to V_2\to0$ 为 smooth representations 的短正合列，且 $V_1,V_2$ admissible，则 $V$ admissible。

**证明.** 对任意开紧 $J\subset G$，取 $J$-不变量得到左正合列
$$
0\to V_1^J\to V^J\to V_2^J.
$$
因此
$$
\dim V^J\le \dim V_1^J+\dim V_2^J<\infty.
$$
故 $V$ admissible。$\square$

**推论 C.22.** 若 $V$ 有有限长度且每个不可约 subquotient admissible，则 $V$ admissible。

**证明.** 对长度归纳。长度 $1$ 为假设。长度大于 $1$ 时取短正合列
$$
0\to V_{r-1}\to V\to V/V_{r-1}\to0,
$$
由归纳假设和命题 C.21 得结论。$\square$

**外部输入定理 C.23（不可约表示的可容许性，典型情形）.** 对 reductive $p$-adic group $G(F)$，在本书涉及的标准范畴中，不可约 smooth complex representations 是 admissible。

**注 C.24.** 该结论属于 Bernstein 理论的基础部分。正文在讨论 $\operatorname{Irr}(G(F))$、L-packets 和局部 L 函数时默认工作在这种可容许范畴中。
