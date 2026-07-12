# 符号约定

本文档记录《Langlands 纲领》的固定符号。后续章节不得随意更改。

归一化 convention 见 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md)；本文只固定记号。

## 集合论与范畴

- 固定 Grothendieck universes
  $$
  \mathcal U\in\mathcal V\in\mathcal W.
  $$
  若不特别说明，“集合”指 $\mathcal U$-小集合。
- 范畴通常记为 $\mathcal C,\mathcal D$；函子写作 $F:\mathcal C\to\mathcal D$。
- 表示范畴写作 $\operatorname{Rep}(G)$，具体光滑、连续、代数或 $\ell$-adic 条件必须在上下文中声明。
- 对象的同构类集合写作 $\pi_0(\mathcal C)$，但只在该集合确实小的情况下使用。

## 域、赋值和完备化

- 整体域写作 $K$。若不特别说明，$K$ 可以是数域或有限域上的一变量函数域。
- $V_K$：$K$ 的所有位置集合；$V_K^\infty$：Archimedean 位置；$V_K^f$：非 Archimedean 位置。
- 对 $v\in V_K$，$K_v$ 表示 $K$ 在 $v$ 处的完备化。
- 非 Archimedean 局部域 $K_v$ 的整数环写作 $\mathcal O_v$，极大理想写作 $\mathfrak p_v$，剩余域写作 $k_v$，基数写作 $q_v=\#k_v$。
- 归一化绝对值写作 $|\cdot|_v$，并固定使乘积公式
  $$
  \prod_{v\in V_K}|x|_v=1,\qquad x\in K^\times
  $$
  成立。

## Adeles、ideles 和 Haar 测度

- Adele 环写作
  $$
  \mathbb A_K=\prod_{v\in V_K}'K_v
  $$
  其中 restricted product 相对于 $\mathcal O_v\subset K_v$ 取。
- Finite adeles 写作 $\mathbb A_{K,f}$。
- Idele 群写作
  $$
  \mathbb A_K^\times=\prod_{v\in V_K}'K_v^\times
  $$
  其中 restricted product 相对于 $\mathcal O_v^\times\subset K_v^\times$ 取。
- Idele class group 写作
  $$
  C_K=K^\times\backslash\mathbb A_K^\times.
  $$
- Norm-one idele class group 写作
  $$
  C_K^1=\ker\bigl(|\cdot|_{\mathbb A}:C_K\to\mathbb R_{>0}\bigr).
  $$
- 对导子恰为 $N$ 的 primitive Dirichlet character $\chi$，其 Euler-compatible finite-order idèle class character 写作
  $$
  \omega_\chi:C_\mathbb Q\to\mathbb C^\times.
  $$
  其 finite-unit restriction 为 $\widehat\chi^{-1}$，无穷分量为
  $\operatorname{sgn}^{a_\chi}$，其中 $\chi(-1)=(-1)^{a_\chi}$；见归一化总表 1.1。
- Adele norm 写作
  $$
  |x|_{\mathbb A}=\prod_v |x_v|_v,\qquad x=(x_v)_v\in\mathbb A_K^\times.
  $$
- 加法 Haar 测度写作 $dx_v,dx$；乘法 Haar 测度写作 $d^\times x_v,d^\times x$。测度归一化必须在涉及积分时声明。

## 特征、Fourier 变换和 L 函数

- 局部加法特征写作 $\psi_v:K_v\to\mathbb C^\times$；其 additive conductor ideal 写作
  $$
  \mathfrak c(\psi_v)=\{x\in K_v:\psi_v(x\mathcal O_v)=1\}.
  $$
  整体加法特征写作 $\psi:\mathbb A_K/K\to\mathbb C^\times$。
- 局部 reciprocity map 写作
  $$
  \operatorname{rec}_F:F^\times\to W_F^{\operatorname{ab}}
  $$
  或在有限 Abel 商中写作 $F^\times\to G_F^{\operatorname{ab}}$；本书采用几何 Frobenius 归一化。
- 全局 reciprocity map 写作
  $$
  \operatorname{rec}_K:C_K\to G_K^{\operatorname{ab}}.
  $$
  这里到 $G_K^{\operatorname{ab}}$ 的像一般只稠密；对有限 Abel 商使用该映射，对完整拓扑同构使用 $C_K$ 的 profinite completion。
- Hecke 特征写作
  $$
  \chi:C_K\to\mathbb C^\times
  $$
  或等价地写作 $\chi:\mathbb A_K^\times\to\mathbb C^\times$ 且 $\chi|_{K^\times}=1$。
- Hecke quasi-character 的唯一酉化写作
  $$
  \chi=\chi_0|\cdot|_{\mathbb A}^{\sigma(\chi)},
  $$
  其中 $\chi_0$ unitary，$\sigma(\chi)\in\mathbb R$。
- 局部分量写作 $\chi_v:K_v^\times\to\mathbb C^\times$。
- Schwartz-Bruhat 空间写作 $\mathcal S(K_v)$ 和 $\mathcal S(\mathbb A_K)$。
- Fourier 变换写作
  $$
  \widehat\Phi(y)=\int_{\mathbb A_K}\Phi(x)\psi(xy)\,dx.
  $$
- 局部 L 因子写作 $L(s,\chi_v)$；整体 L 函数写作
  $$
  L(s,\chi)=\prod_v L(s,\chi_v)
  $$
  并必须声明收敛半平面和解析延拓来源。

## 群、表示和自守对象

- 代数群通常记为 $G$，其对偶群写作 $\widehat G$，L 群写作 ${}^LG$。
- $G$ 的 Borel subgroup 通常记为 $B$，maximal torus 通常记为 $T$。
- Torus $T$ 的特征格和余特征格写作
  $$
  X^*(T)=\operatorname{Hom}(T_{\overline F},\mathbb G_m),\qquad
  X_*(T)=\operatorname{Hom}(\mathbb G_m,T_{\overline F}).
  $$
- 特征与余特征的自然配对写作
  $$
  \langle\chi,\lambda\rangle,\qquad \chi\in X^*(T),\ \lambda\in X_*(T).
  $$
- 根集合写作 $\Phi(G,T)\subset X^*(T)$，余根集合写作 $\Phi^\vee(G,T)\subset X_*(T)$。
- Simple roots 集合写作 $\Delta$，相应 simple coroots 集合写作 $\Delta^\vee$。
- Weyl group 写作
  $$
  W(G,T)=N_G(T)/T.
  $$
- 绝对 Galois 群在 L 群语境中也写作
  $$
  \Gamma_F=\operatorname{Gal}(F^{\operatorname{sep}}/F).
  $$
- 局部 L 群默认写作
  $$
  {}^LG=\widehat G\rtimes W_F,
  $$
  其中 $W_F$ 对 $\widehat G$ 的作用由 $G$ 的 pinned root datum 上的 Galois 作用给出；split 情形该作用平凡。
- 对局部域 $F$，$G(F)$ 的光滑表示写作 $(\pi,V_\pi)$。
- 不可约可容许表示同构类集合写作
  $$
  \operatorname{Irr}(G(F)).
  $$
- 开紧子群通常记为 $J$ 或 $K$；$J$-不变量写作
  $$
  V^J=\{v\in V:\pi(j)v=v\text{ for all }j\in J\}.
  $$
- Hecke 代数写作
  $$
  \mathcal H(G,J)=e_J*C_c^\infty(G)*e_J.
  $$
- 紧诱导写作 $\operatorname{c-Ind}_H^G\sigma$；归一化抛物诱导写作 $\operatorname{Ind}_{P(F)}^{G(F)}(\sigma)$。
- 对整体域 $K$，$G(\mathbb A_K)$ 的自守表示写作 $\pi=\otimes_v'\pi_v$。
- 自守商写作
  $$
  [G]=G(K)\backslash G(\mathbb A_K).
  $$
- 中心特征写作 $\omega_\pi$。
- `GL(2)` 的中心写作 $Z$；右正则作用写作
  $$
  (R(h)\Phi)(g)=\Phi(gh).
  $$
- Cuspidal automorphic forms 空间写作 $\mathcal A_0(G,\omega)$。
- 一般自守形式空间写作 $\mathcal A(G,\omega)$。
- 适合某个 Euler 乘积的有限坏位置集合通常写作 $S$，并至少包含 Archimedean 位置、群的 ramified 位置、表示的 ramified 位置和 L 群表示的 ramified 位置。
- 标准 L 函数按表示 $r:{}^LG\to\operatorname{GL}(V)$ 写作
  $$
  L(s,\pi,r)=\prod_v L(s,\pi_v,r).
  $$
- 部分 L 函数写作
  $$
  L^S(s,\pi,r)=\prod_{v\notin S}L(s,\pi_v,r).
  $$

## Galois、Weil 和 Langlands 参数

- 绝对 Galois 群写作
  $$
  G_K=\operatorname{Gal}(\overline K/K).
  $$
- 二维 $\ell$-adic Galois 表示通常写作
  $$
  \rho:G_\mathbb Q\to\operatorname{GL}_2(E)
  $$
  其中 $E/\mathbb Q_\ell$ 为有限扩张。
- 残余表示写作 $\overline\rho$，默认指选取稳定格后约化并取半单化所得的同构类。
- $\ell$-adic cyclotomic character 写作
  $$
  \chi_\ell:G_\mathbb Q\to\mathbb Z_\ell^\times.
  $$
- 复共轭元写作 $c\in G_\mathbb Q$；二维表示称为奇的，若 $\det\rho(c)=-1$。
- 对局部域 $F$，绝对 Galois 群写作 $G_F$，惯性群写作 $I_F$。
- 局部 Weil 群写作 $W_F$；非 Archimedean 情形有
  $$
  1\to I_F\to W_F\to\mathbb Z\to 1.
  $$
- 几何 Frobenius 写作 $\operatorname{Fr}_F$ 或 $\operatorname{Fr}_v$；算术 Frobenius 写作 $\operatorname{Frob}_v^{\operatorname{arith}}$。
- Weil-Deligne 数据写作 $(V,r,N)$，其中 $r:W_F\to\operatorname{GL}(V)$ 且 $N$ nilpotent。
- $n$ 维 Frobenius-semisimple Weil-Deligne 表示的同构类集合写作
  $$
  \operatorname{WDRep}_n(F).
  $$
- 记号 $W_F'$ 表示 $W_F\times\operatorname{SL}_2(\mathbb C)$ 或等价的 Weil-Deligne 参数域；具体模型按章节声明。
- Frobenius 必须说明采用几何 Frobenius 还是算术 Frobenius；本书默认局部 L 因子使用几何 Frobenius 归一化。
- Langlands 参数写作
  $$
  \varphi_v:W_{K_v}'\to{}^LG
  $$
  其中 $W_{K_v}'$ 的含义依局部域类型声明。
- 局部参数集合写作
  $$
  \Phi(G/F).
  $$
- 参数 $\varphi$ 对应的 L-packet 写作
  $$
  \Pi_\varphi(G)\subset\operatorname{Irr}(G(F)).
  $$
- 参数的 centralizer 和 component group 写作
  $$
  S_\varphi=\operatorname{Cent}_{\widehat G}(\operatorname{im}\varphi),\qquad
  \mathcal S_\varphi=\pi_0(S_\varphi/Z(\widehat G)^{W_F}).
  $$
- 增强参数写作 $(\varphi,\rho)$，其中 $\rho\in\operatorname{Irr}(\mathcal S_\varphi)$，具体 relevance 条件按群和内形式声明。
- L 同态通常写作
  $$
  \xi:{}^LH\to{}^LG.
  $$
- 参数沿 L 同态的推前写作
  $$
  \xi_*\varphi=\xi\circ\varphi.
  $$
- 表示的弱或强函子性转移在明确存在时写作
  $$
  \Pi=\xi_*\sigma
  $$
  或在需要避免唯一性假设时写作“$\Pi$ 是 $\sigma$ 沿 $\xi$ 的转移”。
- Base change 和 automorphic induction 分别写作
  $$
  \operatorname{BC}_{E/K}(\pi),\qquad \operatorname{AI}_{E/K}(\sigma).
  $$
- `GL(n)` 的 isobaric sum 在需要时写作
  $$
  \pi_1\boxplus\cdots\boxplus\pi_r.
  $$
- Arthur 参数通常写作 $\psi$；局部 Arthur packet 写作 $\Pi_\psi(G/F)$，全局 Arthur component group 写作 $\mathcal S_\psi$。
- `GL(n)` 的局部 Langlands 对应写作
  $$
  \operatorname{rec}_{F,n}:\operatorname{Irr}(\operatorname{GL}_n(F))\to\operatorname{WDRep}_n(F).
  $$
- `GL(n)\times GL(m)` 的 Rankin-Selberg L 函数写作
  $$
  L(s,\pi\times\pi').
  $$
- Regular algebraic automorphic representation $\pi$ 关联的 $\ell$-adic Galois 表示写作
  $$
  \rho_{\pi,\ell}:G_K\to\operatorname{GL}_n(\overline{\mathbb Q}_\ell).
  $$

## 几何 Langlands 符号

- 几何部分的曲线通常写作 $X/k$，函数域写作 $K_X=k(X)$。
- 闭点 $x\in X$ 的形式圆盘和穿孔形式圆盘写作
  $$
  D_x=\operatorname{Spec}\mathcal O_x,\qquad D_x^\times=\operatorname{Spec}K_x.
  $$
- $G$-bundle 模栈写作
  $$
  \operatorname{Bun}_G(X)
  $$
  或简写为 $\operatorname{Bun}_G$。
- Hecke 栈写作 $\operatorname{Hecke}_G$，两条投影写作
  $$
  h^\leftarrow,\ h^\rightarrow:\operatorname{Hecke}_G\to\operatorname{Bun}_G.
  $$
- Affine Grassmannian 写作
  $$
  \operatorname{Gr}_G=G((t))/G[[t]].
  $$
- Dominant coweight $\lambda\in X_*(T)^+$ 对应的 Schubert cell 和 Schubert variety 写作
  $$
  \operatorname{Gr}_G^\lambda,\qquad \overline{\operatorname{Gr}}_G^\lambda.
  $$
- Satake 范畴写作
  $$
  \operatorname{Sat}_G=\operatorname{Perv}_{G[[t]]}(\operatorname{Gr}_G).
  $$
- 几何 Satake 下 $V\in\operatorname{Rep}(\widehat G)$ 对应的 sheaf 写作 $\mathcal S_V$；Hecke 函子写作 $\mathsf H_V$。
- $\widehat G$-local systems 模栈写作
  $$
  \operatorname{LocSys}_{\widehat G}(X).
  $$
- 特征零 de Rham categorical geometric Langlands 的普通 half-twisted 自动侧写作
  $$
  \operatorname{DMod}_{1/2}(\operatorname{Bun}_G),
  $$
  表示 determinant line 平方根 gerbe 上的 ordinary half-twisted D-modules。其 renormalized companion 始终另记为
  $$
  \operatorname{DMod}_{1/2}(\operatorname{Bun}_G)_{\operatorname{ren}};
  $$
  不得省略下标 $\operatorname{ren}$ 后与普通范畴混同。来源构造的主 Langlands functor 写作
  $$
  \mathbb L_G:
  \operatorname{DMod}_{1/2}(\operatorname{Bun}_G)
  \longrightarrow
  \operatorname{IndCoh}_{\mathcal N}(\operatorname{LocSys}_{\widehat G}(X));
  $$
  方向固定为 automorphic $\to$ spectral，反向等价写作 $\mathbb L_G^{-1}$。另有比较函子
  $$
  \operatorname{ren}:
  \operatorname{DMod}_{1/2}(\operatorname{Bun}_G)
  \rightleftarrows
  \operatorname{DMod}_{1/2}(\operatorname{Bun}_G)_{\operatorname{ren}}
  :\operatorname{un\text{-}ren}
  $$
  及 companion functor
  $$
  \mathbb L_{G,\operatorname{ren}}:
  \operatorname{DMod}_{1/2}(\operatorname{Bun}_G)_{\operatorname{ren}}
  \longrightarrow
  \operatorname{IndCoh}(\operatorname{LocSys}_{\widehat G}(X)).
  $$
  本书不用该 companion 公式改写主等价的两端。
- Hecke eigensheaf 的本征值 $\widehat G$-local system 写作 $\mathcal E$，关联局部系统写作 $V_{\mathcal E}$。
- 对 local Shimura datum $(G,b,\mu)$，本书固定
  $$
  b\in B(G,\mu),
  \qquad
  \mathcal E_b\dashrightarrow\mathcal E_1.
  $$
  $E_\mu$ 表示 $\mu$ 的 reflex field，$\breve E_\mu$ 表示其最大非分歧扩张的完备化，相应 Weil group 写作 $W_{E_\mu}$。Local shtuka diamond 写作
  $$
  \operatorname{Sht}_{G,b,\mu}/\operatorname{Spd}\breve E_\mu.
  $$

## 模形式、椭圆曲线和费马应用

- 上半平面写作 $\mathfrak H$。
- 权 $k$ slash 算子写作 $f|_k\gamma$。
- 同余子群写作 $\Gamma(N),\Gamma_0(N),\Gamma_1(N)$。
- 权 $k$、级 $\Gamma_0(N)$ 的 cusp forms 空间写作 $S_k(\Gamma_0(N))$。
- 带 nebentypus $\varepsilon$ 的模形式和尖点形式空间写作
  $$
  M_k(\Gamma_0(N),\varepsilon),\qquad S_k(\Gamma_0(N),\varepsilon).
  $$
- Fourier 展开变量写作 $q=e^{2\pi iz}$。
- 好素数 $\ell\nmid N$ 处的 Hecke 算子写作 $T_\ell$；坏素数 $\ell\mid N$ 处的 Atkin operator 写作 $U_\ell$。
- Adelic 级结构写作
  $$
  K_0(N),\qquad K_1(N)\subset\operatorname{GL}_2(\mathbb A_{\mathbb Q,f}).
  $$
- 经典 eigenform $f$ 的 adelic 提升写作 $\Phi_f$，生成的自守表示写作 $\pi_f$。
- 好素数 $p\nmid N$ 处经典 Hecke 多项式
  $$
  X^2-a_p(f)X+\varepsilon(p)p^{k-1}
  $$
  的根默认写作 $(\alpha_p,\beta_p)$；需要在同一公式中强调归一化时也可写作
  $(\alpha_p^{\mathrm{cl}},\beta_p^{\mathrm{cl}})$。Unitary automorphic normalization 的 Satake 根写作
  $$
  (\alpha_p^{\mathrm{u}},\beta_p^{\mathrm{u}})
  =p^{-(k-1)/2}(\alpha_p,\beta_p).
  $$
- 默认 $\pi_f$ 表示 unitary automorphic normalization；algebraic normalization 写作
  $$
  \pi_f^{\mathrm{alg}}=\pi_f\otimes|\det|^{-(k-1)/2}.
  $$
- 椭圆曲线写作 $E/\mathbb Q$，导子写作 $N_E$。
- 椭圆曲线的最小判别式写作 $\Delta_E$；局部约化 trace 写作
  $$
  a_p(E)=p+1-\#\widetilde E(\mathbb F_p)
  $$
  在好约化处使用。
- Hasse-Weil L 函数写作
  $$
  L(E,s)=\prod_pL_p(E,s).
  $$
- $\ell$-adic Tate module 写作 $T_\ell(E)$，相关表示写作
  $$
  \rho_{E,\ell}:G_{\mathbb Q}\to\operatorname{GL}_2(\mathbb Z_\ell).
  $$
- 模 $p$ 表示写作
  $$
  \overline\rho_{E,p}:G_{\mathbb Q}\to\operatorname{GL}_2(\mathbb F_p)
  $$
  并默认取半单化时会明确说明。
- 模性提升中的 universal deformation ring 写作 $R$，相应 Hecke algebra 写作 $T$；`$R=T$` 只在具体局部变形条件下使用。
- 残余表示的 prime-to-$p$ Serre conductor 写作
  $$
  N(\overline\rho)=\prod_{q\ne p}q^{n_q(\overline\rho)}.
  $$
- Frey 曲线写作
  $$
  E_{a,b,p}:y^2=x(x-a^p)(x+b^p).
  $$
