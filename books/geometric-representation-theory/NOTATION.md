# 符号约定

本文件登记《Geometric Representation Theory》全书符号。若正文引入新符号，应先在本文件登记。

## 0. 大小、底域和系数

- $\mathcal U\in\mathcal V\in\mathcal W$：固定 Grothendieck universes。
- $k$：代数闭底域。若涉及 D-modules、Riemann-Hilbert、category $\mathcal O$ 或复解析拓扑，默认 $k=\mathbb C$。
- $E$：sheaf 的系数域，默认特征 $0$。若使用 $\ell$-adic sheaves，则 $E$ 是有限扩张的 $\mathbb Q_\ell$，且 $\ell\ne\operatorname{char} k$。
- $\mathbf{Vect}_E$、$\mathbf{Vect}^{\mathrm{fd}}_E$：全部 $E$-向量空间、有限维 $E$-向量空间的范畴。
- $D^b_c(X,E)$：$X$ 上 $E$-系数 constructible complexes 的有界导出范畴。具体拓扑依章节声明。
- $\operatorname{Perv}(X,E)$：middle perversity 下的 perverse sheaves。
- $\mathcal D_X$：光滑复代数簇 $X$ 上的微分算子层。默认使用 left $\mathcal D_X$-modules，除非另行说明。

## 1. 群、根数据和 flag varieties

- $G$：连通 reductive algebraic group。
- $\mathfrak g=\operatorname{Lie}(G)$：$G$ 的 Lie 代数。
- $B\subset G$：Borel subgroup。
- $U=R_u(B)$：$B$ 的 unipotent radical。
- $T\subset B$：maximal torus。
- $N=N_G(T)$：$T$ 的 normalizer。
- $W=N/T$：Weyl group。
- $\Phi=\Phi(G,T)$：根系。
- $\Phi^+$：由 $B$ 决定的正根集合。
- $\Delta\subset\Phi^+$：simple roots。
- $X^\ast(T)=\operatorname{Hom}(T,\mathbb G_m)$：character lattice。
- $X_\ast(T)=\operatorname{Hom}(\mathbb G_m,T)$：cocharacter lattice。
- $\rho=\frac12\sum_{\alpha\in\Phi^+}\alpha$：半和。
- $\mathcal B=G/B$：完全旗簇。
- $P\supset B$：parabolic subgroup。
- $\mathcal P=G/P$：partial flag variety。
- $X_w=BwB/B\subset\mathcal B$：Schubert cell。
- $\overline X_w$：Schubert variety。
- $\ell:W\to\mathbb Z_{\ge0}$：Coxeter length。
- $\le$：Bruhat order。

## 2. 代数和表示

- $U(\mathfrak g)$：universal enveloping algebra。
- $Z(\mathfrak g)=Z(U(\mathfrak g))$：中心。
- $\mathfrak b=\operatorname{Lie}(B)$、$\mathfrak t=\operatorname{Lie}(T)$、$\mathfrak n=\operatorname{Lie}(U)$。
- $\mathfrak g=\mathfrak n^-\oplus\mathfrak t\oplus\mathfrak n$：由 $B$ 诱导的 triangular decomposition。
- $M(\lambda)$：最高权 $\lambda$ 的 Verma module。
- $L(\lambda)$：$M(\lambda)$ 的唯一简单商。
- $\mathcal O$：BGG category $\mathcal O$。
- $\mathcal O_\chi$：中心 character 为 $\chi$ 的 block。
- $w\cdot\lambda=w(\lambda+\rho)-\rho$：dot action。

## 3. Sheaves、stacks 和卷积

- $[X/H]$：代数群 $H$ 作用于 $X$ 的 quotient stack。
- $D^b_H(X,E)$：$H$-equivariant constructible derived category，定义为 $D^b_c([X/H],E)$ 的适当模型。
- $\operatorname{For}_H:D^b_H(X,E)\to D^b_c(X,E)$：沿 atlas 的 forgetful functor；$\operatorname{Perv}_H$ 按它 t-exact 的 convention 定义。
- $f^\ast,f_\ast,f_!,f^!$：六函子中的拉回、推前、紧支推前和 extraordinary pullback。
- $\mathbb D_X$：Verdier duality functor。
- $\operatorname{IC}(\overline S,\mathcal L)$：stratum $S$ 上 local system $\mathcal L$ 的 middle extension intersection complex。
- $\widetilde\boxtimes$：沿 contracted-product torsor descent 得到的 twisted external product。
- $\star_!,\star_\ast$：分别使用目标 map 的 $!$-pushforward、$\ast$-pushforward 的 convolution；在实际支撑上 proper 时共同简记为 $\star$。
- $\mathcal H_W$：Hecke algebra。
- $\mathcal H_B=D^b_B(G/B,E)$ 或等价的 $D^b(B\backslash G/B,E)$，按章节上下文指定。

## 4. 前沿和高级对象

- $\mathscr O=\mathbb C[[z]]$、$\mathscr K=\mathbb C((z))$：第十二、十三章的 formal disk ring 与 punctured-disk field；不要与 BGG category $\mathcal O$ 混同。
- $LG(R)=G(R((z)))$、$L^+G(R)=G(R[[z]])$：loop-group 与 arc-group functors。
- $\operatorname{Gr}_G=LG/L^+G$：fpqc affine Grassmannian；第十二、十三章形成 Betti sheaves 时取 reduction。
- $\operatorname{Gr}^\lambda$、$\overline{\operatorname{Gr}}^\lambda$：dominant coweight $\lambda$ 的 $L^+G$-orbit 及 Schubert closure。
- $D^b_{L^+G,\mathrm{fs}}(\operatorname{Gr}_G,E)$：finite-dimensional-support equivariant Betti derived category。
- $\operatorname{Sat}_G=\operatorname{Perv}_{L^+G,\mathrm{fs}}(\operatorname{Gr}_G,E)$：characteristic-zero Betti Satake category。
- $\operatorname{IC}_\lambda$：$\overline{\operatorname{Gr}}^\lambda$ 上按 open orbit 常值 local system 归一化的 IC sheaf。
- $\omega(\mathcal F)=\bigoplus_i\mathbb H^i(\operatorname{Gr}_G,\mathcal F)$：忘掉 cohomological grading 的 Satake fiber functor。
- $S_\mu=N(\mathscr K)z^\mu L^+G/L^+G$、$F_\mu$：semi-infinite orbit 与相应 weight functor。
- $T^\vee$：满足 $X^\ast(T^\vee)=X_\ast(T)$ 的 split dual torus。
- $\operatorname{Fl}_G=G((z))/I$：affine flag variety，其中 $I$ 是 Iwahori subgroup。
- $G^\vee$：Langlands dual group。
- $\operatorname{Bun}_G(C)$：曲线 $C$ 上 $G$-bundles 的模栈。
- $\operatorname{LocSys}_{G^\vee}(C)$：$G^\vee$-local systems 的导出栈或相应 Betti/de Rham 版本。
- $\mathcal M_C(G,N)$：BFN 口径下由 $(G,N)$ 构造的 Coulomb branch。
- $\mathcal A_\hbar(G,N)$：quantized Coulomb branch algebra。
- $I\subset L^+G$：Iwahori subgroup。
- $\widehat{\mathfrak g}$：affine Kac-Moody algebra，通常为 $\mathfrak g((z))$ 的中心扩张。
- $\mathsf H_I$：Iwahori-Hecke category。
- $\mathfrak M_\theta(v,w)$：Nakajima quiver variety。
- $R(\nu)$：Khovanov-Lauda-Rouquier algebra。
- $R^\Lambda$：cyclotomic KLR algebra。
- $U_q(\mathfrak g)$：quantum group。
- $U_q^-(\mathfrak g)$：quantum group 的负半部。
- $B(\Lambda)$：最高权 $\Lambda$ 的 crystal。
- $\operatorname{CoHA}(Q)$：quiver $Q$ 的 cohomological Hall algebra，具体版本依章节声明。
