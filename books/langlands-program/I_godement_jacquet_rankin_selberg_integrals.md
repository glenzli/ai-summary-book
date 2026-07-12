# 附录 I：Godement-Jacquet 与 Rankin-Selberg 积分

本附录补充第十三、十四章中 `GL(n)` L 函数解析性质的来源。Euler 乘积只给出局部因子的形式定义；解析延拓和函数方程来自全局积分、局部函数方程和 Fourier 分析。

收口归一化回指：本附录的 Schwartz-Bruhat 测度、Godement-Jacquet 因子、Rankin-Selberg 因子和 converse theorem 检测均按 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 第 3、4、8 节解释。

## I.1 Godement-Jacquet 局部积分

设 $F$ 为局部域，$G=\operatorname{GL}_n(F)$，$M_n(F)$ 为 $n\times n$ 矩阵空间。令 $\mathcal S(M_n(F))$ 为 Schwartz-Bruhat 空间。

**定义 I.1.** 设 $\pi$ 为 $G$ 的不可约可容许表示，$v\in V_\pi$，$\lambda\in V_{\pi^\vee}$。Matrix coefficient 为
$$
\xi_{v,\lambda}(g)=\lambda(\pi(g)v).
$$
对 $\Phi\in\mathcal S(M_n(F))$，Godement-Jacquet 局部 zeta integral 定义为
$$
Z(s,\Phi,\xi)=\int_{\operatorname{GL}_n(F)}
\Phi(g)\xi(g)|\det g|_F^{s+(n-1)/2}\,dg
$$
在绝对收敛半平面中成立。这里的 $s+(n-1)/2$ 是本附录采用的自守归一化；其他文献可能把该平移吸收到 $s$ 中。

**外部输入定理 I.2（局部 Godement-Jacquet 理论）.** 当 $\Phi$ 和 $\xi$ 变化时，局部积分 $Z(s,\Phi,\xi)$ 生成 $\mathbb C(q^{-s})$ 中的一个主分式理想，其标准生成元定义为
$$
L(s,\pi,\operatorname{Std}).
$$
并存在局部 $\gamma$ 因子使 Fourier 变换满足局部函数方程
$$
Z(1-s,\widehat\Phi,\xi^\vee)
=\gamma(s,\pi,\operatorname{Std},\psi)Z(s,\Phi,\xi)
$$
在相应归一化下成立。

**外部输入定理 I.3（非分歧计算）.** 若 $F$ 非 Archimedean，$\pi$ 为 spherical representation，Satake 参数为
$$
\operatorname{diag}(\alpha_1,\ldots,\alpha_n),
$$
取 $\Phi=\mathbf 1_{M_n(\mathcal O_F)}$，并取归一化 spherical matrix coefficient $\xi(1)=1$，则
$$
Z(s,\Phi,\xi)=L(s,\pi,\operatorname{Std})
=\prod_{i=1}^n(1-\alpha_iq^{-s})^{-1}
$$
至多差由测度归一化和本附录 $s$ 平移约定决定的标准常数；按 Godement-Jacquet 标准归一化可使等号严格成立。

**注 I.4.** 本附录不重新证明 I.2 和 I.3。它们的证明需要 Iwasawa 分解、Cartan 分解、球函数计算和局部 Fourier 变换。

## I.2 Godement-Jacquet 全局积分

设 $K$ 为整体域，$\pi$ 为 cuspidal automorphic representation of $\operatorname{GL}_n(\mathbb A_K)$，$\varphi\in V_\pi$，$\varphi^\vee\in V_{\pi^\vee}$，$\Phi\in\mathcal S(M_n(\mathbb A_K))$。

**定义 I.5.** 全局 Godement-Jacquet zeta integral 定义为
$$
Z(s,\Phi,\varphi,\varphi^\vee)
=
\int_{\operatorname{GL}_n(\mathbb A_K)}
\Phi(g)\varphi(g)\varphi^\vee(g)|\det g|_{\mathbb A}^{s+(n-1)/2}\,dg
$$
在绝对收敛半平面中成立。严格地说，$\varphi(g)\varphi^\vee(g)$ 表示由 $\pi$ 与 $\pi^\vee$ 向量形成的 automorphic matrix coefficient。

**外部输入定理 I.6（Euler 分解）.** 若数据可分解为纯张量
$$
\Phi=\otimes_v\Phi_v,\qquad
\varphi=\otimes_v\varphi_v,\qquad
\varphi^\vee=\otimes_v\varphi_v^\vee,
$$
且测度按 restricted product 归一化，则在绝对收敛半平面中
$$
Z(s,\Phi,\varphi,\varphi^\vee)
=\prod_v Z(s,\Phi_v,\xi_v),
$$
其中 $\xi_v$ 为局部 matrix coefficient。

**证明路线（外部输入）.** 自守表示的 restricted tensor product 分解给出 matrix coefficient 的张量分解。Schwartz-Bruhat 函数也为 restricted tensor product。绝对收敛保证可应用 Fubini；附录 B 的 restricted product 积分公式把整体积分写成局部积分乘积。$\square$

**外部输入定理 I.7（Godement-Jacquet 全局定理）.** 取几乎所有非分歧位置的 spherical vector 与标准 Schwartz 函数，并在有限多个 ramified 或 Archimedean 位置固定局部测试数据，则 $Z(s,\Phi,\varphi,\varphi^\vee)$ 表示 $L(s,\pi,\operatorname{Std})$ 乘以有限多个局部修正因子。Fourier 变换 $\Phi\mapsto\widehat\Phi$ 和 Poisson summation on $M_n(K)\subset M_n(\mathbb A_K)$ 给出 $L(s,\pi,\operatorname{Std})$ 的 meromorphic continuation 和函数方程。

**注 I.8.** 对 cuspidal $\pi$ 且 $n>1$，标准 L 函数为 entire；$n=1$ 时退化为 Tate thesis，平凡 character 有标准极点。

## I.3 Whittaker 模型和 Rankin-Selberg 局部积分

令 $N_n\subset\operatorname{GL}_n$ 为 upper triangular unipotent subgroup。固定非平凡加法特征 $\psi:F\to\mathbb C^\times$，并定义
$$
\psi_{N_n}(u)=\psi(u_{1,2}+u_{2,3}+\cdots+u_{n-1,n}).
$$

**定义 I.9.** 表示 $\pi$ 称为 generic，若存在非零连续线性泛函 $\Lambda:V_\pi\to\mathbb C$ 满足
$$
\Lambda(\pi(u)v)=\psi_{N_n}(u)\Lambda(v),
\qquad u\in N_n(F).
$$
相应 Whittaker function 为
$$
W_v(g)=\Lambda(\pi(g)v).
$$

**外部输入定理 I.10（Whittaker uniqueness for `GL(n)`）.** 对 $\operatorname{GL}_n(F)$ 的 irreducible admissible generic representation，Whittaker functional 在标量倍下唯一。

设 $\pi$ 和 $\pi'$ 分别为 $\operatorname{GL}_n(F)$ 与 $\operatorname{GL}_m(F)$ 的 generic representations。

**定义 I.11.** Rankin-Selberg 局部 zeta integrals 是由 Whittaker functions $W\in\mathcal W(\pi,\psi)$、$W'\in\mathcal W(\pi',\psi^{-1})$ 和必要的 Schwartz 函数构造的积分。典型情形 $m=n-1$ 中，可写成沿 $N_{n-1}(F)\backslash\operatorname{GL}_{n-1}(F)$ 的积分
$$
Z(s,W,W')
=
\int_{N_{n-1}(F)\backslash\operatorname{GL}_{n-1}(F)}
W\begin{pmatrix}g&0\\0&1\end{pmatrix}
W'(g)|\det g|^{s-1/2}\,dg
$$
的变体。不同 $n,m$ 需采用 Jacquet-Piatetski-Shapiro-Shalika 的精确模型。

**外部输入定理 I.12（局部 Rankin-Selberg 理论）.** 局部 Rankin-Selberg 积分生成局部因子
$$
L(s,\pi\times\pi')
$$
并满足局部函数方程
$$
Z(1-s,\widetilde W,\widetilde W')
=\gamma(s,\pi\times\pi',\psi)Z(s,W,W')
$$
在 Jacquet-Piatetski-Shapiro-Shalika 的 Whittaker、Haar 测度和 contragredient normalization 下成立。若 $\pi,\pi'$ 非分歧，Satake 参数分别为 $(\alpha_i)$ 和 $(\beta_j)$，则
$$
L(s,\pi\times\pi')
=
\prod_{i,j}(1-\alpha_i\beta_jq^{-s})^{-1}.
$$

## I.4 Rankin-Selberg 全局积分

设 $K$ 为整体域，$\pi$ 和 $\pi'$ 为 cuspidal automorphic representations of $\operatorname{GL}_n(\mathbb A_K)$ 和 $\operatorname{GL}_m(\mathbb A_K)$，并假设它们 generic。

**外部输入定理 I.13（全局 Rankin-Selberg unfolding）.** 对 cusp forms $\varphi\in\pi$、$\varphi'\in\pi'$ 和所选 Rankin-Selberg 模型中的 Eisenstein series 或 kernel，Rankin-Selberg 全局积分可 unfolding 为 Whittaker functions 的积分，并在可分解数据下分解为局部 zeta integrals：
$$
Z(s,\varphi,\varphi')
=\prod_v Z(s,W_v,W_v').
$$
非分歧位置的局部计算给出
$$
L^S(s,\pi\times\pi')=\prod_{v\notin S}L(s,\pi_v\times\pi_v').
$$

**外部输入定理 I.14（Rankin-Selberg 解析性质）.** 对 cuspidal $\pi,\pi'$，完成的 Rankin-Selberg L 函数 $L(s,\pi\times\pi')$ 有 meromorphic continuation 和函数方程。其可能极点由 $\pi'$ 是否为 $\pi^\vee$ 的 twist 控制；精确极点位置依赖中心特征和归一化。

**注 I.15.** 第十四章定理 14.11 是 I.14 的正文接口版本。该理论支撑强重数一、converse theorem 和许多 functoriality 证明。

## I.5 Converse Theorem 的分析输入

**定义 I.16.** 设 $\Pi=\otimes_v'\Pi_v$ 是 $\operatorname{GL}_N(\mathbb A_K)$ 的 admissible representation。称 $\Pi$ 满足 Rankin-Selberg 测试条件，若对一族足够大的 cuspidal automorphic representations $\tau$ of $\operatorname{GL}_m(\mathbb A_K)$，$1\le m\le N-1$，函数
$$
L(s,\Pi\times\tau)
$$
及其 contragredient 版本具有 meromorphic continuation、函数方程和竖带有界性。

**外部输入定理 I.17（Converse theorem 的积分表示背景）.** Cogdell-Piatetski-Shapiro converse theorem 证明：在中心特征、可容许性、有限 ramification 集、unitary normalization after twist 和竖带有界性等标准假设下，Rankin-Selberg 测试条件推出 $\Pi$ 是 automorphic。

**注 I.18.** 这解释了为什么第十五章的函子性证明常被转化为 L 函数解析性质证明：若 L 群同态给出候选局部参数族，构造 `GL(N)` 上的候选 $\Pi$ 后，只要能证明所有必要 twist 的 Rankin-Selberg L 函数满足 converse theorem 条件，就可推出 $\Pi$ 自守。

## I.6 与 Langlands 函子性的关系

**命题 I.19.** 设 $\xi:{}^LH\to{}^L\operatorname{GL}_N$ 为 L 同态，$\sigma$ 为 $H(\mathbb A_K)$ 的自守表示。若存在 $\operatorname{GL}_N(\mathbb A_K)$ 的 automorphic representation $\Pi$，使几乎所有非分歧位置满足
$$
s_v(\Pi)=\xi(s_v(\sigma)),
$$
则对每个有限维表示 $r$ of $\operatorname{GL}_N(\mathbb C)$，非分歧部分 L 函数满足
$$
L^S(s,\Pi,r)=L^S(s,\sigma,r\circ\xi).
$$

**证明.** 对 $v\notin S$，局部 Satake 参数满足 $s_v(\Pi)=\xi(s_v(\sigma))$。于是
$$
r(s_v(\Pi))=(r\circ\xi)(s_v(\sigma)).
$$
两侧局部 Euler 因子 characteristic polynomial 相同。对所有 $v\notin S$ 相乘得到部分 L 函数相等。$\square$

**注 I.20.** 反方向不是形式命题。由足够多 L 函数相等推出自守转移，需要强重数一或 converse theorem；这正是 `GL(N)` 作为函子性目标的特殊优势。

## I.7 本附录小结

本附录提供以下接口：

1. Godement-Jacquet 积分构造 `GL(n)` 标准 L 函数。
2. Rankin-Selberg 积分构造 `GL(n)\times GL(m)` L 函数。
3. 局部非分歧计算给出 Satake 参数的 Euler 因子。
4. 全局 unfolding 和 Poisson/Fourier 方法给出解析延拓与函数方程。
5. Converse theorem 把 L 函数解析性质转化为 automorphy。

## 练习

**练习 I.1.** 对 $n=1$，说明 Godement-Jacquet 局部积分退化为 Tate thesis 的局部 zeta integral。

**练习 I.2.** 若 `GL(n)` 和 `GL(m)` 的非分歧 Satake 参数分别为 $(\alpha_i)$ 和 $(\beta_j)$，推导 Rankin-Selberg 局部因子
$$
\prod_{i,j}(1-\alpha_i\beta_jq^{-s})^{-1}.
$$

**练习 I.3.** 解释 global unfolding 为什么通常需要 cusp form 的尖点条件来消去非开轨道或边界项。

**练习 I.4.** 说明 converse theorem 中为什么需要对足够多 $\tau$ 的 twists，而不是只检查 $L(s,\Pi)$。

**练习 I.5.** 设 symmetric square lift $\operatorname{Sym}^2\pi$ 已存在。用命题 I.19 写出其标准 L 函数与 $\pi$ 的 symmetric square L 函数的非分歧相容公式。
