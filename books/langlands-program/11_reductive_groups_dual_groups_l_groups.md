# 第十一章：还原群、对偶群和 L 群

## 本章目标

本章建立一般 Langlands 纲领所需的代数群语言：connected reductive group、根资料、对偶群、Galois 作用和 L 群。前十章主要围绕 `GL(1)` 与 `GL(2)` 展开；从本章开始，自守表示的对象不再固定为 `GL(n)`，而是任意局部域或整体域上的还原群 $G$。Langlands 参数的靶不应写成普通复群 $\widehat G$，而应写成含有 Galois 或 Weil 作用的 L 群 ${}^LG$。

## 依赖前置知识

需要第一章的局部域和整体域，第四章的局部紧群与光滑表示，第五章的 Weil 群与局部参数。需要基本代数几何中的 affine group scheme、smoothness、base change 和 Lie algebra 权空间分解。本章把 connected reductive groups 的结构定理、根资料分类定理和 pinning 的 Galois 作用作为外部输入。

收口归一化回指：本章 L 群、L 同态、非分歧参数、Satake 参数和 L 群表示给出的局部因子按 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 第 2、4、8 节解释。

## 11.1 代数群与还原群

本节设 $F$ 为任意域，固定代数闭包 $\overline F$。当涉及绝对 Galois 群时，在 $\overline F$ 中固定可分闭包 $F^{\operatorname{sep}}$，并写
$$
\Gamma_F=\operatorname{Gal}(F^{\operatorname{sep}}/F).
$$

**定义 11.1.** 一个 $F$-代数群是一个群对象
$$
G:\operatorname{Alg}_F\to\operatorname{Groups}
$$
且其底层函子由一个有限型 $F$-scheme 表示。若表示 scheme 为 affine，则称 $G$ 为 affine algebraic group。

若 $E/F$ 为域扩张，记
$$
G_E=G\times_{\operatorname{Spec}F}\operatorname{Spec}E.
$$
这是 $G$ 的 base change。

**定义 11.2.** 一个 $F$-torus 是一个 $F$-代数群 $T$，使得存在有限可分扩张 $E/F$ 和整数 $r\ge0$ 满足
$$
T_E\cong\mathbb G_{m,E}^r.
$$
若可以取 $E=F$，则称 $T$ 为 split torus。整数 $r$ 称为 $T$ 的维数。

**例 11.3.** 乘法群 $\mathbb G_m$ 是一维 split torus。若 $E/F$ 为有限可分扩张，则 Weil restriction
$$
\operatorname{Res}_{E/F}\mathbb G_m
$$
是 $F$-torus；它在 $E$ 上分裂，但通常不在 $F$ 上同构于 $\mathbb G_m^{[E:F]}$。

**定义 11.4.** 设 $G$ 为 smooth connected affine algebraic group。其几何 unipotent radical 是 $G_{\overline F}$ 中最大的 connected normal unipotent subgroup，记为
$$
R_u(G_{\overline F}).
$$
若
$$
R_u(G_{\overline F})=1,
$$
则称 $G$ 为 connected reductive group，简称还原群（reductive group）。

**注 11.5.** 这里“还原”不是表示论中的 completely reducible representation，也不是环论中的 reduced scheme。它是对 smooth connected affine algebraic group 的结构限制：在代数闭包上没有非平凡连通正规 unipotent 部分。

**例 11.6.** 下列群是 $F$ 上的 connected reductive groups：

1. $\operatorname{GL}_{n,F}$。
2. $\operatorname{SL}_{n,F}$。
3. $\operatorname{PGL}_{n,F}$。
4. 任意 $F$-torus。
5. split symplectic group $\operatorname{Sp}_{2n,F}$。

**例 11.7.** Borel subgroup $B\subset\operatorname{GL}_2$，即上三角可逆矩阵群，不是 reductive。它有非平凡 unipotent radical
$$
R_u(B)=
\left\{
\begin{pmatrix}
1&x\\
0&1
\end{pmatrix}:x\in\mathbb G_a
\right\}.
$$

**定义 11.8.** 设 $G$ 为 connected reductive group。一个 maximal torus 是 $G$ 的 torus 子群 $T\subset G$，使得在 $G_{\overline F}$ 中不存在严格包含 $T_{\overline F}$ 的 torus 子群。若 $G$ 含有一个 split maximal torus，则称 $G$ 为 split reductive group。

**外部输入定理 11.9（极大环面和 Borel 子群）.** 对代数闭域 $k$ 上的 connected reductive group $G$：

1. $G$ 含有 maximal torus。
2. 任意两个 maximal tori 在 $G(k)$ 中共轭。
3. $G$ 含有 Borel subgroup，即 maximal connected solvable subgroup。
4. 任意两个 Borel subgroups 在 $G(k)$ 中共轭。

本章后续关于根系和 based root datum 的构造依赖该结构定理。

## 11.2 特征格、余特征格和自然配对

设 $T$ 为 $F$-torus。为了得到与分裂域无关的格，先在 $\overline F$ 上定义 character 和 cocharacter。

**定义 11.10.** $T$ 的特征格和余特征格分别为
$$
X^*(T)=\operatorname{Hom}_{\overline F\text{-grp}}(T_{\overline F},\mathbb G_{m,\overline F}),
$$
$$
X_*(T)=\operatorname{Hom}_{\overline F\text{-grp}}(\mathbb G_{m,\overline F},T_{\overline F}).
$$
它们是有限生成自由 Abel 群。若 $\dim T=r$，则
$$
X^*(T)\cong\mathbb Z^r,\qquad X_*(T)\cong\mathbb Z^r,
$$
但该同构依赖于分裂坐标的选择。

**定义 11.11.** 对 $\chi\in X^*(T)$ 和 $\lambda\in X_*(T)$，复合
$$
\chi\circ\lambda:\mathbb G_m\to\mathbb G_m
$$
唯一写成
$$
t\mapsto t^{\langle\chi,\lambda\rangle}
$$
其中 $\langle\chi,\lambda\rangle\in\mathbb Z$。由此得到双线性配对
$$
\langle\cdot,\cdot\rangle:X^*(T)\times X_*(T)\to\mathbb Z.
$$

**命题 11.12.** 配对 $\langle\cdot,\cdot\rangle$ 是完美配对，即诱导同构
$$
X^*(T)\cong\operatorname{Hom}_{\mathbb Z}(X_*(T),\mathbb Z),
\qquad
X_*(T)\cong\operatorname{Hom}_{\mathbb Z}(X^*(T),\mathbb Z).
$$

**证明.** 经有限可分扩张后可设 $T$ 分裂。若
$$
T\cong\mathbb G_m^r,
$$
则 $X^*(T)$ 有基 $e_1,\ldots,e_r$，其中
$$
e_i(t_1,\ldots,t_r)=t_i,
$$
而 $X_*(T)$ 有对偶基 $e_1^\vee,\ldots,e_r^\vee$，其中
$$
e_i^\vee(t)=(1,\ldots,1,t,1,\ldots,1).
$$
此时
$$
e_i\circ e_j^\vee(t)=t^{\delta_{ij}},
$$
故配对矩阵为单位矩阵。由于 character 和 cocharacter 的定义在 $\overline F$ 上进行，分裂坐标证明了抽象 Abel 群上的完美性。$\square$

**注 11.13.** 若 $T$ 未在 $F$ 上分裂，则 $\Gamma_F$ 作用在 $X^*(T)$ 和 $X_*(T)$ 上。这个 Galois 模结构携带了 $T$ 的下降数据；对 split torus，该作用平凡。

## 11.3 根、余根与根资料

本节设 $G$ 为 $F$ 上的 split connected reductive group，并固定 split maximal torus $T\subset G$。设 $\mathfrak g=\operatorname{Lie}(G)$，$\mathfrak t=\operatorname{Lie}(T)$。

由于 $T$ 通过 adjoint action 作用在 $\mathfrak g$ 上，$\mathfrak g$ 分解为 $T$ 的权空间：
$$
\mathfrak g
=
\mathfrak t
\oplus
\bigoplus_{\alpha\in\Phi(G,T)}\mathfrak g_\alpha.
$$
这里
$$
\mathfrak g_\alpha=\{X\in\mathfrak g:\operatorname{Ad}(t)X=\alpha(t)X\text{ for all }t\in T\},
$$
而 $\Phi(G,T)\subset X^*(T)$ 是非零权的集合。

**定义 11.14.** 集合 $\Phi(G,T)$ 称为 $G$ 关于 $T$ 的根系。其元素称为 roots。

**外部输入定理 11.15（根子群与余根）.** 对每个根 $\alpha\in\Phi(G,T)$，存在对应的根子群 $U_\alpha\subset G$，并存在唯一余根
$$
\alpha^\vee\in X_*(T)
$$
使得由 $U_\alpha$ 和 $U_{-\alpha}$ 生成的 rank-one subgroup 给出与 $\operatorname{SL}_2$ 或 $\operatorname{PGL}_2$ 同源的标准三元结构。该余根满足
$$
\langle\alpha,\alpha^\vee\rangle=2.
$$

**定义 11.16.** $G$ 关于 $T$ 的根资料（root datum）是四元组
$$
\Psi(G,T)=\left(X^*(T),\Phi(G,T),X_*(T),\Phi^\vee(G,T)\right),
$$
其中
$$
\Phi^\vee(G,T)=\{\alpha^\vee:\alpha\in\Phi(G,T)\}\subset X_*(T).
$$

**命题 11.17.** 对每个 $\alpha\in\Phi(G,T)$，定义反射
$$
s_\alpha:X^*(T)\to X^*(T),\qquad
s_\alpha(\chi)=\chi-\langle\chi,\alpha^\vee\rangle\alpha.
$$
则 $s_\alpha$ 保持 $\Phi(G,T)$。对偶地，反射
$$
s_{\alpha^\vee}:X_*(T)\to X_*(T),\qquad
s_{\alpha^\vee}(\lambda)=\lambda-\langle\alpha,\lambda\rangle\alpha^\vee
$$
保持 $\Phi^\vee(G,T)$。

**证明草图.** 根 $\alpha$ 对应的 rank-one subgroup 给出 Weyl group 中的元素 $n_\alpha$，它归一化 $T$。共轭作用
$$
t\mapsto n_\alpha t n_\alpha^{-1}
$$
诱导 $X^*(T)$ 上的反射 $s_\alpha$，并把 $\mathfrak g_\beta$ 送到 $\mathfrak g_{s_\alpha(\beta)}$。因此 $s_\alpha$ 保持根集合。余根侧由对偶配对和同一个 rank-one subgroup 得到。完整证明依赖 reductive group 的结构理论。$\square$

**定义 11.18.** Weyl group 定义为
$$
W(G,T)=N_G(T)/T.
$$
它作用在 $X^*(T)$ 和 $X_*(T)$ 上。

**命题 11.19.** 群 $W(G,T)$ 由根反射 $s_\alpha$ 生成。

**证明草图.** 由 Bruhat decomposition 或 Borel 子群结构定理，$N_G(T)/T$ 的元素由 simple root 对应的 rank-one subgroup 的正规化元生成。每个这样的正规化元在 $X^*(T)$ 上的作用是相应根反射。$\square$

为了得到分类所需的不带选择歧义的数据，还要选择 Borel subgroup。

**定义 11.20.** 设 $B\subset G$ 为包含 $T$ 的 Borel subgroup。由 $B$ 确定的正根集合记为 $\Phi^+$，其 simple roots 集合记为 $\Delta\subset\Phi^+$。Based root datum 是六元组
$$
\Psi(G,B,T)=
\left(X^*(T),\Delta,\Phi(G,T),X_*(T),\Delta^\vee,\Phi^\vee(G,T)\right).
$$

**外部输入定理 11.21（split reductive groups 的根资料分类）.** 设 $F$ 为域。split connected reductive $F$-groups 连同 pinned splitting 的同构类，与基于根资料的同构类等价。特别地，split connected reductive group 的同构类型由其 based root datum 决定。

**注 11.22.** 本书使用定理 11.21 的方向是：从 $G$ 提取根资料，再由对偶根资料构造复对偶群。定理的证明属于代数群结构理论，不在本章展开。

## 11.4 对偶群

Langlands 纲领中出现的对偶群是一个复 reductive group。它由根资料的互换构造，而不是由 $G(F)$ 的 Pontryagin dual 构造。

**定义 11.23.** 设 $G$ 为 split connected reductive group，固定 split maximal torus $T$。其复对偶群 $\widehat G$ 是满足如下根资料条件的 connected reductive group over $\mathbb C$：
$$
\Psi(\widehat G,\widehat T)
=
\left(X_*(T),\Phi^\vee(G,T),X^*(T),\Phi(G,T)\right).
$$
也就是说，$\widehat T$ 的 character lattice 是 $X_*(T)$，而 $\widehat G$ 的根是 $G$ 的余根。

**注 11.24.** 对偶群 $\widehat G$ 的定义依赖根资料分类定理。不同的 $T$ 或 $B$ 选择给出同构的 $\widehat G$，但没有指定 pinning 时不应把这些同构视为完全无选择。

**例 11.25.** 对 $G=\operatorname{GL}_{n,F}$，取 diagonal torus $T$。有
$$
X^*(T)=\bigoplus_{i=1}^n\mathbb Z e_i,\qquad
X_*(T)=\bigoplus_{i=1}^n\mathbb Z e_i^\vee.
$$
根为
$$
\Phi=\{e_i-e_j:i\ne j\}.
$$
余根为
$$
\Phi^\vee=\{e_i^\vee-e_j^\vee:i\ne j\}.
$$
于是对偶根资料与原根资料同型，因此
$$
\widehat{\operatorname{GL}_n}\cong\operatorname{GL}_n(\mathbb C).
$$

**例 11.26.** 对 split groups 有
$$
\widehat{\operatorname{SL}_n}\cong\operatorname{PGL}_n(\mathbb C),
\qquad
\widehat{\operatorname{PGL}_n}\cong\operatorname{SL}_n(\mathbb C).
$$
原因是 $\operatorname{SL}_n$ 与 $\operatorname{PGL}_n$ 的根系统相同，但 character lattice 和 cocharacter lattice 互为对偶位置：simply connected root datum 与 adjoint root datum 在对偶操作下交换。

**例 11.27.** 对 split symplectic 和 odd orthogonal groups，有
$$
\widehat{\operatorname{Sp}_{2n}}\cong\operatorname{SO}_{2n+1}(\mathbb C),
\qquad
\widehat{\operatorname{SO}_{2n+1}}\cong\operatorname{Sp}_{2n}(\mathbb C).
$$
这是因为 $C_n$ 与 $B_n$ 根系统在取余根后互换。

**例 11.28.** 若 $T$ 是 $F$-torus，则 $\Phi=\varnothing$。其对偶群 $\widehat T$ 是复 torus，并由
$$
X^*(\widehat T)=X_*(T)
$$
确定。对 $T=\mathbb G_m$，有
$$
\widehat T=\mathbb C^\times.
$$

**命题 11.29.** 若 $G_1,G_2$ 为 split connected reductive groups，则
$$
\widehat{G_1\times G_2}\cong \widehat G_1\times\widehat G_2.
$$

**证明.** 取 split maximal tori $T_i\subset G_i$。则 $T_1\times T_2$ 是 $G_1\times G_2$ 的 split maximal torus，并且
$$
X^*(T_1\times T_2)=X^*(T_1)\oplus X^*(T_2),
$$
$$
X_*(T_1\times T_2)=X_*(T_1)\oplus X_*(T_2).
$$
根集合为两部分的不交并：
$$
\Phi(G_1\times G_2,T_1\times T_2)
=
(\Phi(G_1,T_1)\times 0)\sqcup(0\times\Phi(G_2,T_2)).
$$
对偶根资料因此为两个对偶根资料的直和。由根资料分类定理，对应复 reductive group 是 $\widehat G_1\times\widehat G_2$。$\square$

## 11.5 非 split 群的 Galois 作用

设 $G$ 为 $F$ 上的 connected reductive group，不要求 split。在 $\overline F$ 上，$G_{\overline F}$ 是 split reductive group。因此可以取 $\overline F$ 上的 pinning。

**定义 11.30.** $G_{\overline F}$ 的 pinning 是四元组
$$
(B,T,\{X_\alpha\}_{\alpha\in\Delta}),
$$
其中 $T\subset B\subset G_{\overline F}$，$B$ 为 Borel subgroup，$T$ 为 maximal torus，$\Delta$ 为 corresponding simple roots，且每个 $X_\alpha$ 是 simple root space $\mathfrak g_\alpha$ 中的非零向量。

**外部输入定理 11.31（pinned automorphisms 与 Galois 作用）.** 固定 pinning 后，$\Gamma_F$ 在 $G_{\overline F}$ 上的 semilinear 作用可与唯一的内自同构校正相复合，使给定 pinning 被送回自身。由此得到 based root datum 的作用
$$
\Gamma_F\to\operatorname{Aut}\Psi(G_{\overline F},B,T).
$$
换言之，$\Gamma_F$ 通过保持根资料结构的 automorphisms 作用在 $X^*(T)$、$X_*(T)$、$\Phi$、$\Phi^\vee$ 和 $\Delta$ 上。改变 pinning 会把该作用替换为同构的 pinned root datum 作用；在 L 群中这至多改变半直积的内共轭实现。

**定义 11.32.** 设 $\widehat G$ 为 $G_{\overline F}$ 的复对偶群。由定理 11.31 和对偶根资料得到的作用
$$
\Gamma_F\to\operatorname{Aut}(\widehat G)
$$
称为 $G$ 的 dual Galois action。更精确地说，该作用取值于保持 $\widehat G$ 中固定 pinning 的 automorphism group。

**注 11.33.** 若 $G$ 在有限 Galois 扩张 $E/F$ 上分裂，则上述作用通过有限商
$$
\operatorname{Gal}(E/F)
$$
分解。若 $G$ 已在 $F$ 上 split，则作用平凡。

**例 11.34.** 设 $T=\operatorname{Res}_{E/F}\mathbb G_m$，其中 $E/F$ 为有限可分扩张。则
$$
X^*(T)\cong\mathbb Z[\operatorname{Hom}_F(E,\overline F)]
$$
并带有 $\Gamma_F$ 对嵌入集合的置换作用。其对偶 torus $\widehat T$ 的 character lattice 是 $X_*(T)$，也带相应的 Galois 作用。这个例子说明：非 split torus 的 Langlands 参数不能只看复 torus 本身，还必须记录 Galois 作用。

## 11.6 L 群

L 群把复对偶群和 Galois/Weil 作用放在同一个群中。它是一般 Langlands 参数的靶。

**定义 11.35（Galois 型 L 群）.** 设 $G$ 为 $F$ 上的 connected reductive group。其 Galois 型 L 群定义为半直积
$$
{}^LG_{\Gamma}=\widehat G\rtimes\Gamma_F,
$$
其中 $\Gamma_F$ 对 $\widehat G$ 的作用为定义 11.32 中的 dual Galois action。

乘法由
$$
(g_1,\gamma_1)(g_2,\gamma_2)
=
(g_1\cdot \gamma_1(g_2),\gamma_1\gamma_2)
$$
给出。

**定义 11.36（局部 Weil 型 L 群）.** 若 $F$ 为局部域，则局部 L 群定义为
$$
{}^LG=\widehat G\rtimes W_F,
$$
其中 $W_F\to\Gamma_F$ 后接定义 11.32 的作用。非 Archimedean 情形的 $W_F$ 采用第五章的几何 Frobenius 归一化；Archimedean 情形采用第五章的 $W_\mathbb R$、$W_\mathbb C$。

**注 11.37.** 当 $G$ 在 $F$ 上 split 时，$W_F$ 对 $\widehat G$ 的作用平凡，因此
$$
{}^LG\cong \widehat G\times W_F.
$$
这解释了第五章在 split 情形中把局部参数写成 $W_F'\to\widehat G$ 的简化口径。

**注 11.38.** 对整体域 $K$，可以写形式上的半直积
$$
\widehat G\rtimes\Gamma_K,
$$
但这不是全局 Langlands 纲领中全部期望的“全局 Langlands 群”。后者应同时编码所有局部参数、Arthur $\operatorname{SL}_2$、motivic 或 Galois 信息以及局部化映射；其完整对象在数域情形仍属于纲领性结构。本书在全局自守 L 函数中先使用各处局部 L 群和非分歧 Satake 参数，不把一个未经定义的全局 Langlands 群作为定理前提。

**定义 11.39.** 设 $F$ 为非 Archimedean 局部域。Langlands 参数域记为
$$
W_F'=W_F\times\operatorname{SL}_2(\mathbb C)
$$
或等价的 Weil-Deligne 版本。给定 $G/F$，一个局部 Langlands 参数是连续同态
$$
\varphi:W_F'\to{}^LG
$$
满足：

1. 复合
   $$
   W_F'\xrightarrow{\varphi}{}^LG\to W_F
   $$
   等于 $W_F'$ 到 $W_F$ 的自然投影。
2. $\varphi|_{\operatorname{SL}_2(\mathbb C)}$ 是代数同态到 $\widehat G$。
3. 对每个 $w\in W_F$，$\varphi(w)$ 在 $\widehat G\rtimes W_F$ 中的 $\widehat G$-部分满足通常的半单性条件。

参数只按 $\widehat G$-共轭类计入局部 Langlands 对应。

**注 11.40.** 在非 split 情形，参数不是同态 $W_F'\to\widehat G$。它必须落在 ${}^LG$ 中并覆盖 $W_F$。这正是 L 群区别于单纯对偶群的地方。

## 11.7 L 同态与函子性接口

Langlands 函子性把群之间的关系放在 L 群侧，而不是直接放在原代数群侧。

**定义 11.41.** 设 $H$ 和 $G$ 为同一局部域 $F$ 上的 connected reductive groups。一个 L 同态是连续同态
$$
{}^LH\longrightarrow{}^LG
$$
满足：

1. 它与到 $W_F$ 的投影相容，即交换图
   $$
   \begin{matrix}
   {}^LH &\longrightarrow& {}^LG\\
   \downarrow && \downarrow\\
   W_F&=&W_F
   \end{matrix}
   $$
   交换。
2. 限制在 $\widehat H$ 上给出复代数群同态
   $$
   \widehat H\to\widehat G.
   $$
3. 它与 $W_F$ 对两侧对偶群的作用相容，允许按标准定义作 $\widehat G$-共轭调整。

**定义 11.42.** 设
$$
\xi:{}^LH\to{}^LG
$$
为 L 同态。若 $\varphi_H:W_F'\to{}^LH$ 是 $H$ 的局部参数，则
$$
\xi\circ\varphi_H:W_F'\to{}^LG
$$
是 $G$ 的局部参数。该操作称为参数的 functorial pushforward。

**命题 11.43.** 定义 11.42 中的复合确实覆盖 $W_F$。

**证明.** 由 $\varphi_H$ 的定义，复合
$$
W_F'\xrightarrow{\varphi_H}{}^LH\to W_F
$$
等于自然投影。由 L 同态定义，${}^LH\to{}^LG$ 与两侧到 $W_F$ 的投影相容。因此
$$
W_F'\xrightarrow{\xi\circ\varphi_H}{}^LG\to W_F
$$
仍等于自然投影。$\square$

**猜想 11.44（Langlands 函子性，局部接口）.** 对与 Weil 投影相容并保持相关内形式 relevance 条件的 L 同态
$$
\xi:{}^LH\to{}^LG,
$$
局部 Langlands 对应应把 $H(F)$ 的不可约可容许表示的 L-packets 转移到 $G(F)$ 的 L-packets，使得参数由
$$
\varphi_H\mapsto \xi\circ\varphi_H
$$
给出。

**注 11.45.** 猜想 11.44 是接口陈述，不是完整局部函子性定理。完整版本必须处理 L-packet 内部参数化、central character、enhanced parameters、endoscopy 和归一化的 transfer factors。

## 11.8 表示给出的 L 函数

一般自守 L 函数不是只由 $\widehat G$ 决定，还要选择一个有限维表示。

**定义 11.46.** 设 $F$ 为局部域，$G/F$ 为 connected reductive group。一个 L 群表示是有限维复向量空间 $V$ 上的同态
$$
r:{}^LG\to\operatorname{GL}(V)
$$
其限制
$$
r|_{\widehat G}:\widehat G\to\operatorname{GL}(V)
$$
为复代数表示，并且 $W_F$ 的作用满足相应的连续性条件。

若 $\varphi:W_F'\to{}^LG$ 是局部参数，则
$$
r\circ\varphi:W_F'\to\operatorname{GL}(V)
$$
给出一个线性 Weil-Deligne 型参数，从而可定义局部因子
$$
L(s,\varphi,r),\qquad
\varepsilon(s,\varphi,r,\psi),\qquad
\gamma(s,\varphi,r,\psi).
$$

**例 11.47.** 对 $G=\operatorname{GL}_n$，有
$$
{}^LG=\operatorname{GL}_n(\mathbb C)\times W_F.
$$
标准表示
$$
\operatorname{Std}:\operatorname{GL}_n(\mathbb C)\to\operatorname{GL}_n(\mathbb C)
$$
给出 `GL(n)` 的标准 L 函数。若参数 $\varphi$ 对应 Weil-Deligne 表示 $(V,r,N)$，则
$$
L(s,\varphi,\operatorname{Std})
=
\det\left(1-q^{-s}r(\operatorname{Fr}_F)\mid(\ker N)^{I_F}\right)^{-1}
$$
与第五章定义一致。

**例 11.48.** 即使对 $G=\operatorname{GL}_n$，同一个自守表示也可配上不同的 L 群表示：
$$
\operatorname{Std},\qquad
\operatorname{Sym}^2\operatorname{Std},\qquad
\wedge^2\operatorname{Std},\qquad
\operatorname{Ad}.
$$
它们分别给出 standard、symmetric square、exterior square 和 adjoint L 函数。由此可见，“$G$ 的 L 函数”不是单个对象；必须说明所用的
$$
r:{}^LG\to\operatorname{GL}(V).
$$

**注 11.49.** 对 classical groups，还会出现 standard、spin、similitude 和 adjoint 等不同表示。后续章节必须先固定根资料、中心和 similitude convention，再写相应 L 函数。

## 11.9 非分歧参数与 Satake

本节把第四章的 Satake 同构与本章的 L 群语言连接起来。设 $F$ 为非 Archimedean 局部域，$\mathcal O_F$ 为整数环，剩余域大小为 $q$。

附录 P 给出 split hyperspecial 情形下的 Satake 变换和 `GL(n)` 显式计算，附录 AA 给出 hyperspecial subgroup、Cartan decomposition 和 unramified reductive group 的 Bruhat-Tits 来源。本节只把这些结果翻译成 L 群中的半单共轭类。

**定义 11.50.** $G/F$ 称为 unramified reductive group，若 $G$ 为 quasi-split，且在某个非分歧有限扩张上 split，并存在 reductive $\mathcal O_F$-model $\mathcal G$ 使
$$
K=\mathcal G(\mathcal O_F)
$$
为 hyperspecial maximal compact subgroup。

**外部输入定理 11.51（非分歧 Satake 参数）.** 设 $G/F$ 为 unramified reductive group，$K\subset G(F)$ 为 hyperspecial maximal compact subgroup。不可约 spherical representation $\pi$，即
$$
\pi^K\ne0,
$$
给出 L 群中形如
$$
s_\pi\rtimes\operatorname{Fr}_F\in \widehat G\rtimes W_F
$$
的半单共轭类。该共轭类称为 $\pi$ 的 Satake parameter。反过来，$\widehat G\rtimes\operatorname{Fr}_F$ 中的 semisimple $\widehat G$-twisted conjugacy class 给出 spherical Hecke algebra 的 character。

**注 11.52.** 若 $G$ split，则 $W_F$ 对 $\widehat G$ 作用平凡，非分歧 Satake 参数可写成 $\widehat G$ 中的半单共轭类 $s_\pi$。若 $G$ 非 split 但 unramified，则参数自然位于连通分量
$$
\widehat G\rtimes\operatorname{Fr}_F
$$
中，并按 $\widehat G$-twisted conjugacy 取等价类。

**命题 11.53.** 对 split $G=\operatorname{GL}_n$，定义 11.51 与第七章的好素数 Satake 参数一致。

**证明.** split 情形有
$$
{}^LG=\operatorname{GL}_n(\mathbb C)\times W_F.
$$
球表示 $\pi$ 的 spherical Hecke eigenvalues 由 diagonal semisimple conjugacy class
$$
\operatorname{diag}(\alpha_1,\ldots,\alpha_n)\subset\operatorname{GL}_n(\mathbb C)
$$
给出。第四章和第七章的 Satake 同构把 Hecke 代数的 character 识别为这些 $\alpha_i$ 的对称多项式取值。本章的 L 群参数为
$$
\operatorname{Fr}_F\mapsto(\operatorname{diag}(\alpha_1,\ldots,\alpha_n),\operatorname{Fr}_F),
$$
因此两种记法记录同一组半单共轭数据。$\square$

## 11.10 本章小结

一般 Langlands 纲领的基本对象是 connected reductive group $G$，但参数的靶不是 $G$ 本身。先从 $G_{\overline F}$ 的根资料构造复对偶群 $\widehat G$，再把 $F$-结构给出的 Galois 或 Weil 作用加入，得到 L 群
$$
{}^LG=\widehat G\rtimes W_F
$$
或相应的 Galois 型版本。L 同态是函子性的结构载体；表示
$$
r:{}^LG\to\operatorname{GL}(V)
$$
是构造局部和全局 L 函数的额外数据。非分歧局部表示通过 Satake 同构给出 ${}^LG$ 中的半单共轭类，这正是一般局部 Langlands 参数在未分歧处的基本模型。

## 练习

**练习 11.1.** 设 $T=\mathbb G_m^r$。直接计算 $X^*(T)$、$X_*(T)$ 和配对 $\langle\cdot,\cdot\rangle$。

**练习 11.2.** 对 $G=\operatorname{GL}_2$ 的 diagonal torus，写出根、余根、Weyl group 和 simple root。

**练习 11.3.** 证明 $\operatorname{SL}_2$ 的对偶群为 $\operatorname{PGL}_2(\mathbb C)$，并指出 character lattice 与 cocharacter lattice 在证明中的角色。

**练习 11.4.** 设 $E/F$ 为二次 Galois 扩张，$T=\operatorname{Res}_{E/F}\mathbb G_m$。描述 $\Gamma_F$ 在 $X^*(T)$ 上的作用。

**练习 11.5.** 若 $G/F$ split，说明为什么 L 参数 $W_F'\to{}^LG$ 等价于同态 $W_F'\to\widehat G$ 加上到 $W_F$ 的自然投影。

**练习 11.6.** 对 split $G=\operatorname{PGL}_2$，写出 $\widehat G$，并说明一个 L 参数进入 $\operatorname{SL}_2(\mathbb C)$ 与进入 $\operatorname{GL}_2(\mathbb C)$ 的差异。

**练习 11.7.** 设 $\xi:{}^LH\to{}^LG$ 为 L 同态。验证若两个 $H$-参数 $\varphi_1,\varphi_2$ 互为 $\widehat H$-共轭，则 $\xi\circ\varphi_1$ 与 $\xi\circ\varphi_2$ 互为 $\widehat G$-共轭。
