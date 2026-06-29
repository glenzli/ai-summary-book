# 附录 M：Langlands-Shahidi 方法、局部系数和 Gamma 因子

本附录补充第十三、十五和附录 L 中的 L 函数构造接口。Godement-Jacquet 和 Rankin-Selberg 积分主要覆盖 `GL(n)` 的标准与卷积 L 函数；Langlands-Shahidi 方法从 Eisenstein series 的常数项和 intertwining operators 出发，构造更一般还原群上的一大类 L 函数。

收口归一化回指：本附录的 local coefficient、$\gamma$ 因子、Shahidi 局部 L 因子和全局函数方程均应与 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 第 4、8 节的 Satake 和 L 函数变量 convention 相容。

## M.1 抛物子群和 Adjoint 表示

设 $F$ 为局部域，$G/F$ 为 quasi-split connected reductive group。设 $P=MN$ 为 maximal parabolic subgroup，$\widehat P=\widehat M\widehat N$ 为对偶群中的对应 parabolic。

**定义 M.1.** 令
$$
{}^L M=\widehat M\rtimes W_F.
$$
对偶 Lie algebra
$$
{}^L\mathfrak n=\operatorname{Lie}(\widehat N)
$$
上有 ${}^LM$ 的 adjoint action。它可分解为不可约表示
$$
r=\bigoplus_i r_i,\qquad
r_i:{}^LM\to\operatorname{GL}(V_i).
$$

**例 M.2.** 对 $G=\operatorname{GL}_{n+m}$，$M\simeq\operatorname{GL}_n\times\operatorname{GL}_m$ 的标准 maximal parabolic，$\operatorname{Lie}(\widehat N)$ 可识别为
$$
\operatorname{Hom}(\mathbb C^m,\mathbb C^n)
\simeq \operatorname{Std}_n\otimes\operatorname{Std}_m^\vee.
$$
相应 L 函数就是 Rankin-Selberg L 函数 $L(s,\pi_n\times\pi_m^\vee)$ 的一个归一化版本。

**命题 M.3.** 若 $\pi$ 是 $M(F)$ 的局部 Langlands 参数为 $\varphi_\pi$ 的表示，则 Langlands-Shahidi 方法预期构造的局部因子为
$$
L(s,\pi,r_i)=L(s,r_i\circ\varphi_\pi),
$$
并同时构造 $\gamma(s,\pi,r_i,\psi)$ 与 $\varepsilon(s,\pi,r_i,\psi)$。

**证明说明.** 这是 compatibility statement，不是证明。Langlands-Shahidi 方法在许多 generic 表示情形中通过局部系数定义这些因子，并证明它们满足与 LLC 预期一致的性质。完整相容性依赖局部 Langlands 和 Shahidi 理论。$\square$

## M.2 Generic 表示和 Whittaker 泛函

固定 $G$ 的 Borel subgroup $B=TU$ 和非退化 character
$$
\psi_U:U(F)\to\mathbb C^\times.
$$

**定义 M.4.** $G(F)$ 的 irreducible admissible representation $\pi$ 称为 $\psi_U$-generic，若
$$
\operatorname{Hom}_{U(F)}(\pi,\psi_U)\ne0.
$$
其中 $\operatorname{Hom}_{U(F)}(\pi,\psi_U)$ 的元素称为 Whittaker functional。

**外部输入定理 M.5（Whittaker uniqueness）.** 对 quasi-split reductive group 的合适 irreducible admissible generic representations，Whittaker functional 在标量倍下唯一：
$$
\dim\operatorname{Hom}_{U(F)}(\pi,\psi_U)=1.
$$
对 `GL(n)` 这是经典唯一性定理；一般 quasi-split groups 也有相应版本。

**注 M.6.** Shahidi 局部系数依赖 generic 性。非 generic packet 成员的 L 因子通常通过 packet、Langlands 分类或局部 LLC 延拓定义。

## M.3 Local Coefficient

设 $\tau$ 为 $M(F)$ 的 irreducible admissible generic representation。考虑归一化诱导
$$
I(s,\tau)=\operatorname{Ind}_{P(F)}^{G(F)}(\tau\otimes e^{s\widetilde\alpha}),
$$
其中 $\widetilde\alpha$ 是由 $P$ 决定的一维实参数。

**定义 M.7.** Standard intertwining operator
$$
A(s,\tau,w):I(s,\tau)\to I(-s,w\tau)
$$
在收敛区域由沿 $N$ 的积分定义，并有 meromorphic continuation。

**定义 M.8.** 设 $\lambda_{s,\tau}$ 为诱导表示上的 Whittaker functional。Local coefficient $C(s,\tau,w,\psi)$ 由函数方程
$$
\lambda_{-s,w\tau}\circ A(s,\tau,w)
=C(s,\tau,w,\psi)\lambda_{s,\tau}
$$
定义。

**外部输入定理 M.9（Shahidi local coefficient）.** Local coefficient $C(s,\tau,w,\psi)$ 是 $q^{-s}$ 的有理函数，并可分解为由 adjoint action 的不可约分量 $r_i$ 给出的局部 $\gamma$ 因子乘积：
$$
C(s,\tau,w,\psi)
=\prod_i \gamma(a_i s,\tau,r_i,\psi)
$$
在适当归一化和整数 $a_i$ 下成立。

**注 M.10.** 精确的 $a_i$、对偶、中心 character 和 $\rho$-shift 依赖 parabolic 和归一化。正文中使用该方法时必须声明采用的 $s$ convention。

## M.4 局部 L、epsilon 和 gamma 因子

**定义 M.11.** Shahidi 局部 $\gamma$ 因子满足形式关系
$$
\gamma(s,\tau,r,\psi)
=
\varepsilon(s,\tau,r,\psi)
\frac{L(1-s,\tau^\vee,r^\vee)}{L(s,\tau,r)}.
$$
在该式中，若先构造 $\gamma$ 因子，则 $L$ 因子可由归一化和分母多项式条件反向定义。

**外部输入定理 M.12（局部因子的基本性质）.** Langlands-Shahidi 局部因子满足：

1. Multiplicativity with respect to parabolic induction；
2. 对非分歧 spherical data 与 Satake 参数给出的 Euler 因子一致；
3. 对 character twist 有可控变换；
4. 与局部函数方程相容；
5. 在已知 LLC 情形中与 Artin-Deligne 局部因子相容。

**命题 M.13（非分歧归一化）.** 若 $F$ 非 Archimedean，$\tau$ spherical，Satake 参数为 $s(\tau)\in{}^LM$，则
$$
L(s,\tau,r_i)=
\det(1-q^{-s}r_i(s(\tau)))^{-1}.
$$

**证明.** 这是非分歧相容性。Spherical vector 上的归一化 intertwining operator 由 Gindikin-Karpelevich 公式给出，其标量正是若干 Euler 因子的商。把 numerator 和 denominator 按定义 M.11 分离，得到 Satake determinant 公式。完整计算属于 Gindikin-Karpelevich-Shahidi 理论。$\square$

## M.5 全局 Eisenstein Series 和函数方程

设 $K$ 为整体域，$G/K$ quasi-split，$P=MN$ maximal parabolic。令 $\tau=\otimes_v'\tau_v$ 为 $M(\mathbb A_K)$ 的 cuspidal generic automorphic representation。

**定义 M.14.** 由 $\tau$ 和 section $f_s$ 构造 Eisenstein series
$$
E(g,s,f)=\sum_{\gamma\in P(K)\backslash G(K)}f_s(\gamma g).
$$

**外部输入定理 M.15（全局 Shahidi 方法）.** Eisenstein series 的常数项和全局 intertwining operator 的函数方程给出部分 L 函数
$$
L^S(s,\tau,r_i)
$$
的 meromorphic continuation 和函数方程，前提是相应 Eisenstein series 和 intertwining operators 的解析性质已经建立。

**命题 M.16（Euler 乘积和局部系数的拼合）.** 在可分解数据下，全局 intertwining operator 分解为局部 intertwining operators 的 restricted tensor product；其 Whittaker coefficient 的全局函数方程分解为局部 local coefficients 的乘积。

**证明草图.** 全局 Whittaker coefficient 沿 $U(K)\backslash U(\mathbb A_K)$ 积分。对纯张量 section，unfolding 后得到局部 Whittaker integrals 的乘积。全局 standard intertwining operator 也按 restricted tensor product 分解。Whittaker uniqueness 使每个局部函数方程由一个标量 local coefficient 控制；全局标量为局部标量乘积。$\square$

## M.6 例子

**例 M.17（`GL(n)×GL(m)` Rankin-Selberg）.** 取 $G=\operatorname{GL}_{n+m}$，$M=\operatorname{GL}_n\times\operatorname{GL}_m$。Langlands-Shahidi 方法给出的 L 函数与 Rankin-Selberg L 函数相容：
$$
L(s,\pi_n\boxtimes\pi_m^\vee,r)
=L(s,\pi_n\times\pi_m^\vee).
$$

**例 M.18（Symmetric square for `GL(2)`）.** 在适当 classical group 或 $\operatorname{GSpin}$ 群的 Shahidi 设置中，adjoint action 的某个分量给出 $\operatorname{Sym}^2$ 或 exterior square L 函数。Gelbart-Jacquet 的 symmetric square lift 可与这些 L 函数解析性质相互校验。

**例 M.19（Adjoint L 函数）.** 对许多 groups，adjoint representation 出现在某个 parabolic 的 $\operatorname{Lie}(\widehat N)$ 中。相应 adjoint L 函数的 poles 与残余谱、Plancherel measure 和 functorial transfer 密切相关。

## M.7 与函子性的关系

**命题 M.20.** 若 Langlands-Shahidi 方法证明了 $L(s,\sigma,r\circ\xi)$ 对足够多 twists 的解析性质，而 $\xi:{}^LH\to{}^L\operatorname{GL}_N$ 给出候选转移，则这些解析性质可作为 converse theorem 的输入。

**证明.** Converse theorem 要求候选 $\operatorname{GL}_N$ 表示与足够多 $\operatorname{GL}_m$ cuspidal twists 的 Rankin-Selberg L 函数有解析延拓、函数方程和有界性。若这些 twisted L 函数能通过 Shahidi 方法识别为 $H$ 侧某些 $r\circ\xi$ 的 L 函数，并已满足所需解析性质，则 converse theorem 推出候选表示自守。$\square$

**注 M.21.** 这就是 Langlands-Shahidi 方法在许多低阶 functorial lifts 中的作用：它不直接构造转移表示，而是提供足够强的 L 函数解析性质以触发 converse theorem。

## M.8 本附录小结

本附录建立：

1. $r_i$ 来自 ${}^LM$ 在 $\operatorname{Lie}(\widehat N)$ 上的 adjoint action。
2. Local coefficient 是 normalized intertwining operator 与 Whittaker functional 的比较标量。
3. Local coefficient 分解出 $\gamma(s,\tau,r_i,\psi)$。
4. $\gamma$ 因子与 $L$、$\varepsilon$ 因子满足标准关系。
5. 全局 Eisenstein series 函数方程给出 L 函数函数方程。
6. 这些解析性质可作为 converse theorem 和 functoriality 的输入。

## 练习

**练习 M.1.** 对 $G=\operatorname{GL}_{n+m}$ 的标准 parabolic，计算 $\operatorname{Lie}(\widehat N)$ 作为 $\operatorname{GL}_n(\mathbb C)\times\operatorname{GL}_m(\mathbb C)$ 表示。

**练习 M.2.** 解释 Whittaker uniqueness 为什么使 local coefficient 成为一个标量。

**练习 M.3.** 写出 $\gamma=\varepsilon L(1-s)/L(s)$ 关系中若改变 $s$ 归一化会发生什么。

**练习 M.4.** 说明 Gindikin-Karpelevich 公式为什么能在 spherical 情形给出非分歧 Euler 因子。

**练习 M.5.** 比较 Godement-Jacquet、Rankin-Selberg 和 Langlands-Shahidi 三种方法分别构造哪些 L 函数。
