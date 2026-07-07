# 第一章：Reductive groups、flag varieties 与权格

## 本章目标

本章建立几何表示论的代数群基础：reductive group、Borel subgroup、maximal torus、root datum、flag variety、Schubert cell 和权格。重点是固定全书 convention，并证明若干可内部验证的结构命题。

## 依赖前置知识

需要附录 A 的代数群作用和商栈语言，以及基本交换代数和代数簇概念。

## 1.1 Reductive group 和 Borel 数据

**定义 1.1.** 令 $k$ 为代数闭域。一个线性代数群 $G$ 称为 reductive，若其 identity component 平滑仿射，且 unipotent radical 平凡。本书默认 $G$ 连通 reductive。

**定义 1.2.** $G$ 的 Borel subgroup 是极大连通可解闭子群。若 $B\subset G$ 为 Borel subgroup，$T\subset B$ 为 maximal torus，则称 $(G,B,T)$ 为 pinned 之前的 Borel 数据。这里不选择 pinning root vectors。

**定义 1.3.** 令
$$
X^\ast(T)=\operatorname{Hom}(T,\mathbb G_m),\qquad
X_\ast(T)=\operatorname{Hom}(\mathbb G_m,T).
$$
前者称为 character lattice，后者称为 cocharacter lattice。二者有自然配对
$$
\langle \chi,\lambda\rangle\in\mathbb Z,\qquad
\chi\circ\lambda(t)=t^{\langle\chi,\lambda\rangle}.
$$

**定义 1.4.** $T$ 对 $\mathfrak g=\operatorname{Lie}(G)$ 的 adjoint action 给出权空间分解
$$
\mathfrak g=\mathfrak t\oplus\bigoplus_{\alpha\in\Phi}\mathfrak g_\alpha,
$$
其中非零权 $\alpha$ 构成根系 $\Phi=\Phi(G,T)\subset X^\ast(T)$。由 $B$ 决定正根集合 $\Phi^+$，其 simple roots 记为 $\Delta$。

**外部输入定理 1.5.** 对连通 reductive group $G$，Borel subgroup 和 maximal torus 存在；任意两个 Borel subgroups 共轭，任意两个 maximal tori 共轭；$(X^\ast(T),\Phi,X_\ast(T),\Phi^\vee)$ 构成 reduced root datum。  
用途：全书根数据基础。来源需在附录 D 定位到 Borel 或 Springer。

## 1.2 Weyl group 和 flag variety

**定义 1.6.** Weyl group 定义为
$$
W=N_G(T)/T.
$$
若 $s_\alpha$ 是 simple root $\alpha\in\Delta$ 对应的反射，则 $(W,\{s_\alpha\}_{\alpha\in\Delta})$ 是 Coxeter system。长度函数记为 $\ell$，Bruhat order 记为 $\le$。

**定义 1.7.** 完全旗簇定义为
$$
\mathcal B=G/B.
$$
若 $P\supset B$ 为 parabolic subgroup，则 partial flag variety 为 $G/P$。

**命题 1.8.** $G$ 左乘作用在 $\mathcal B=G/B$ 上是传递的，点 $gB\in\mathcal B$ 的稳定子为 $gBg^{-1}$。

**证明.** 传递性来自商 $G/B$ 的定义：任意点 $gB$ 是 $g$ 作用于基点 $B$ 的结果。稳定子计算如下：$h\in G$ 固定 $gB$ 当且仅当 $hgB=gB$，等价于 $g^{-1}hg\in B$，即 $h\in gBg^{-1}$。对任意测试代数的点同样成立，因此稳定子 subgroup scheme 为 $gBg^{-1}$。$\square$

**外部输入定理 1.9.** $\mathcal B=G/B$ 是光滑 projective variety，且
$$
\dim \mathcal B=|\Phi^+|.
$$
用途：Schubert geometry、D-modules、perverse sheaves。来源需定位到 Borel 或 Springer。

**例 1.10.** 令 $G=SL_2(k)$，取 $B$ 为上三角矩阵，$T$ 为对角矩阵。则
$$
SL_2/B\simeq \mathbb P^1.
$$
同构把矩阵 $g$ 的陪集 $gB$ 送到 $g$ 作用于标准直线 $ke_1$ 得到的直线。稳定子为保持 $ke_1$ 的矩阵，即 $B$。因此该映射是 $SL_2$-equivariant 的齐性空间同构。

## 1.3 Bruhat decomposition 和 Schubert varieties

**定义 1.11.** 对 $w\in W$，选取代表 $\dot w\in N_G(T)$，定义 Schubert cell
$$
X_w=B\dot wB/B\subset G/B.
$$
其 Zariski 闭包
$$
\overline X_w=\overline{B\dot wB/B}
$$
称为 Schubert variety。

**命题 1.12.** $X_w$ 的定义与代表 $\dot w$ 的选择无关。

**证明.** 若 $\dot w'$ 是另一代表，则 $\dot w'=\dot wt$ 对某个 $t\in T$ 成立。因为 $T\subset B$，
$$
B\dot w'B=B\dot wtB=B\dot wB.
$$
故商到 $G/B$ 后得到相同 locally closed subset。$\square$

**外部输入定理 1.13.** Bruhat decomposition 给出不交并
$$
G=\coprod_{w\in W}B\dot wB,\qquad
G/B=\coprod_{w\in W}X_w,
$$
且 $X_w\simeq\mathbb A^{\ell(w)}$。闭包关系为
$$
\overline X_w=\coprod_{v\le w}X_v.
$$
用途：Schubert stratification、Hecke algebra、Kazhdan-Lusztig theory。来源需定位。

**推论 1.14.** $\mathcal B$ 有有限分层
$$
\mathcal B=\coprod_{w\in W}X_w
$$
且每个 stratum 是 affine space。

**证明.** 直接由外部输入定理 1.13 得到。该推论在本书中不增加新外部假设，只是重新包装 Bruhat decomposition。$\square$

## 1.4 权、线丛和 equivariant vector bundles

**定义 1.15.** 对 $\lambda\in X^\ast(T)$，令 $k_\lambda$ 为一维 $B$-表示，其中 $U=R_u(B)$ 平凡作用，$T$ 通过 character $\lambda$ 作用。定义 $G/B$ 上的 $G$-equivariant line bundle
$$
\mathcal L_\lambda=G\times^B k_{-\lambda}.
$$
这里取 $-\lambda$ 是为了与 Borel-Weil convention 相容；若后续章节使用相反 convention，必须显式说明。

**命题 1.16.** $G$-equivariant vector bundles on $G/B$ 与有限维 algebraic $B$-representations 等价。

**证明.** 给定 $B$-表示 $V$，构造 associated bundle
$$
G\times^B V=(G\times V)/B,
$$
其中右作用为 $(g,v)\cdot b=(gb,b^{-1}v)$。左 $G$ 作用在第一因子上，因此得到 $G$-equivariant vector bundle。反向地，给定 $G$-equivariant vector bundle $\mathcal E$，取基点 $eB$ 处纤维 $\mathcal E_{eB}$。稳定子为 $B$，由 equivariance 得到 $B$ 在该纤维上的表示。两个构造互逆：associated bundle 在基点处纤维为 $V$；而任意 $G$-equivariant bundle 由 $G$ 对基点轨道的传递性从基点纤维诱导出来。态射层面同理，$G$-equivariant bundle morphism 由基点纤维上的 $B$-linear map 唯一决定。$\square$

**外部输入定理 1.17.** Borel-Weil-Bott theorem 描述 $\mathcal L_\lambda$ 的 cohomology 为不可约 $G$-表示或零，具体由 dot action 和 Weyl group 长度决定。  
用途：第九章。当前不在第一章证明链中使用。

## 1.5 低秩检查：$SL_2$

**例 1.18.** 对 $G=SL_2$，$W=\{e,s\}$，$\ell(e)=0$，$\ell(s)=1$。Bruhat decomposition 给出
$$
\mathbb P^1=X_e\sqcup X_s,
$$
其中 $X_e$ 是点 $[1:0]$，$X_s\simeq\mathbb A^1$ 是其补。

**证明.** $B$ 作用在 $\mathbb P^1$ 上保持点 $[1:0]$。其余点可写为 $[x:1]$，上三角矩阵
$$
\begin{pmatrix}a&b\\0&a^{-1}\end{pmatrix}
$$
把 $[x:1]$ 送到 $[a x+b:a^{-1}]$，即仿射坐标变为 $a^2x+ab$。取 $a=1$、$b$ 可任意平移，所以补集为单一 $B$-轨道，且同构于 $\mathbb A^1$。$\square$

## 本章小结

本章固定了 $G,B,T,W,\mathcal B$、根数据和 Schubert stratification 的基本符号。内部证明覆盖了稳定子、代表无关性、equivariant vector bundle 与 $B$-表示的对应以及 $SL_2$ 的显式轨道分解。大型结构定理，如 Bruhat decomposition、$G/B$ 的 projectivity 和 Borel-Weil-Bott，已明确标为外部输入。

## 练习

**练习 1.1.** 对 $G=GL_n$，说明 $G/B$ 与完整旗标
$$
0=V_0\subset V_1\subset\cdots\subset V_n=k^n,\qquad \dim V_i=i
$$
的 variety 同构。

**练习 1.2.** 证明 $G$-equivariant line bundles on $G/B$ 与 $X^\ast(B)=X^\ast(T)$ 中的 characters 对应，并指出本章 convention 中 $\lambda$ 与 $\mathcal L_\lambda$ 的符号关系。

**练习 1.3.** 对 $G=SL_3$ 写出 $W\simeq S_3$ 的六个元素、长度和 Bruhat order 的 Hasse diagram。

