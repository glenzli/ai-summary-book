# 附录 O：几何 Langlands 的 D-Modules、IndCoh 和 Singular Support

本附录补充第十八至二十一章的技术背景。几何 Langlands 的现代范畴形式不能只用普通 sheaves 表述；它需要代数栈上的 D-modules、derived algebraic geometry、IndCoh、singular support 和 nilpotent singular support 条件。

收口归一化回指：本附录只固定几何技术接口；当 D-modules、IndCoh 或 kernel formalism 与 Hecke eigensheaf 的函数迹比较时，使用 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 第 9 节。

## O.1 D-Modules 的基本口径

设 $k$ 为 characteristic $0$ 的域，$X/k$ 为 smooth algebraic variety。

**定义 O.1.** $\mathcal D_X$ 表示 $X$ 上的 differential operators sheaf。左 D-module 是 $\mathcal D_X$-module。其 derived category 记为
$$
D(\mathcal D_X).
$$

**定义 O.2.** 若 $X$ smooth，de Rham stack $X_{\operatorname{dR}}$ 是使
$$
X_{\operatorname{dR}}(R)=X(R_{\operatorname{red}})
$$
的 prestack。D-modules 可视为 $X_{\operatorname{dR}}$ 上的 quasi-coherent sheaves：
$$
D\operatorname{-mod}(X)\simeq\operatorname{QCoh}(X_{\operatorname{dR}})
$$
在 de Rham prestack 与 derived algebraic geometry 的标准口径下成立。

**外部输入定理 O.3（D-modules as sheaves on de Rham stack）.** 对 smooth schemes 和 D-module formalism 已建立的 QCA 或 derived stacks，上述等价成立，并与 pullback、pushforward、tensor product 和 Verdier duality 相容。

**注 O.4.** 第二十章中 Hecke eigensheaf 的 Betti、de Rham 和 $\ell$-adic 版本取决于基域和 coefficient theory。本附录采用 characteristic $0$ 的 D-module 口径。

## O.2 Six Functors 和 Kernels

**外部输入定理 O.5（six functor formalism）.** 对固定的 schemes/stacks 范围和 sheaf theory，并在相应 properness、smoothness、constructibility 或 QCA hypotheses 下，存在函子
$$
f^*,\quad f_*,\quad f^!,\quad f_!,\quad \otimes,\quad \mathcal H om
$$
满足 base change、projection formula、adjunction 和 Verdier duality。

**定义 O.6.** 若
$$
X\xleftarrow{p}Z\xrightarrow{q}Y
$$
为 correspondence，且 $\mathcal K$ 为 $Z$ 上的 kernel，则它定义积分变换
$$
\Phi_{\mathcal K}:D(X)\to D(Y),
\qquad
\Phi_{\mathcal K}(\mathcal F)=q_!(p^*\mathcal F\otimes\mathcal K)
$$
或使用 $q_*$、$p^!$ 的变体，依 sheaf theory 和 properness 条件决定。

**命题 O.7.** Hecke functor 是由 Hecke correspondence 给出的 kernel transform。

**证明.** 第十八章定义 Hecke stack
$$
\operatorname{Bun}_G\xleftarrow{h^\leftarrow}\operatorname{Hecke}_G\xrightarrow{h^\rightarrow}\operatorname{Bun}_G\times X.
$$
给定几何 Satake sheaf $\mathcal S_V$ 作为 Hecke stack 相对位置方向上的 kernel，按 O.6 得到
$$
\mathsf H_V(\mathcal F)=h^\rightarrow_!(h^{\leftarrow,*}\mathcal F\otimes\mathcal S_V)
$$
的 $!$ 或 $*$ 版本。这正是第十九、二十章使用的 Hecke functor 形式。$\square$

## O.3 QCoh、IndCoh 和 Perfect Stacks

**定义 O.8.** 对 derived stack $Y$，$\operatorname{QCoh}(Y)$ 为 quasi-coherent complexes 的 stable category。$\operatorname{IndCoh}(Y)$ 是 coherent sheaves 的 ind-completion，适合处理 singular derived stacks 上的 duality。

**外部输入定理 O.9（QCoh 与 IndCoh 的比较）.** 若 $Y$ smooth 或 perfect 且无奇异问题，则 $\operatorname{QCoh}(Y)$ 与 $\operatorname{IndCoh}(Y)$ 有自然比较；若 $Y$ singular derived stack，则 $\operatorname{IndCoh}(Y)$ 更适合表达 duality 和 functorial operations。

**注 O.10.** 几何 Langlands 的谱侧 $\operatorname{LocSys}_{\widehat G}(X)$ 往往是 derived 且 singular。只用 $\operatorname{QCoh}$ 会丢失 nilpotent singular support 修正。

## O.4 Singular Support

设 $Y$ 为 quasi-smooth derived stack。

**定义 O.11.** $Y$ 的 classical singularity stack 记为
$$
\operatorname{Sing}(Y).
$$
它可理解为 $H^{-1}$ cotangent complex 的相对 spectrum，带自然 $\mathbb G_m$-作用。

**外部输入定理 O.12（singular support theory）.** 对 $\mathcal F\in\operatorname{IndCoh}(Y)$，可定义 closed conical subset
$$
\operatorname{SS}(\mathcal F)\subset\operatorname{Sing}(Y)
$$
称为 singular support。对 closed conical subset $\mathcal N\subset\operatorname{Sing}(Y)$，可定义 full subcategory
$$
\operatorname{IndCoh}_{\mathcal N}(Y)\subset\operatorname{IndCoh}(Y).
$$

**定义 O.13.** 对 $Y=\operatorname{LocSys}_{\widehat G}(X)$，nilpotent cone
$$
\mathcal N_{\operatorname{glob}}\subset\operatorname{Sing}(Y)
$$
由 Higgs field nilpotent 条件定义。几何 Langlands 谱侧常写为
$$
\operatorname{IndCoh}_{\mathcal N_{\operatorname{glob}}}
(\operatorname{LocSys}_{\widehat G}(X)).
$$

**注 O.14.** 这解释了第二十一章中“nilpotent singular support 修正”的动机。若忽略该条件，谱侧范畴通常过大，不能与自动侧 half-twisted D-modules on $\operatorname{Bun}_G$ 等价。

## O.5 Categorical Geometric Langlands

**外部输入定理 O.15（几何 Langlands，IndCoh 形式；特征零 preprint theorem）.** 设 $X/\mathbb C$ 为 smooth projective connected curve，$G/\mathbb C$ 为 connected reductive group。按 Gaitsgory-Raskin proof series 固定第二十一章 21.5 的 half-twisting gerbe、中心、renormalized automorphic category 和 global nilpotent cone 后，来源构造从自动侧到谱侧的函子并证明范畴等价
$$
\mathbb L_G:
\operatorname{DMod}_{1/2}(\operatorname{Bun}_G)
\xrightarrow{\sim}
\operatorname{IndCoh}_{\mathcal N_{\operatorname{glob}}}
(\operatorname{LocSys}_{\widehat G}(X))
$$
并有相应 Betti 版本；等价与来源中构造的 spectral action、Hecke action 和 parabolic induction 相容。

**外部输入定理 O.16（技术来源与定理边界）.** O.15 依赖的外部技术链包括：

1. $G$ 为 torus 的几何类域论；
2. Beilinson-Drinfeld 的 opers 和 Hecke eigensheaves 构造；
3. Gaitsgory-Rozenblyum 关于 derived stacks、D-modules on $\operatorname{Bun}_G$ 和 renormalization 的理论；half-twist 另按 21.5 与 Gaitsgory-Raskin proof series 固定；
4. Arinkin-Gaitsgory 的 singular support 和 $\operatorname{IndCoh}_{\mathcal N}$ 框架；
5. Gaitsgory-Raskin 合作项目五篇 2024 proof-series preprints 中 Langlands functor、fundamental local equivalence、parabolic induction、ambidexterity 和 multiplicity one 的组合。

**外部输入定理 O.16.1（正特征中的部分 $\ell$-adic 结果；preprint theorem）.** Gaitsgory-Raskin 2025 预印本在其精确定义的正特征 unramified $\ell$-adic 设定中，证明带 nilpotent singular support 的 automorphic sheaves 范畴等价于 Langlands 参数栈的某些连通分支之并上的适当 $\operatorname{IndCoh}$ 范畴。该陈述既不把“某些分支”改成“全部分支”，也不把来源中的范畴替换成 naive constructible derived category。

**研究边界 O.16.2.** O.15 与 O.16.1 合在一起仍不陈述正特征整个参数栈上的 unrestricted $\ell$-adic、ramified、integral 或 quantum categorical geometric Langlands，也不陈述 Fargues-Scholze local geometrization。Preprint theorem 的来源状态、部分正特征定理和这些未覆盖版本必须分别记录。

**推论 O.17.** 设 local system $\mathcal E$ 的 residual gerbe 点对象 $\delta_{\mathcal E}$ 属于 O.15 的谱侧范畴。则
$$
\mathcal F_{\mathcal E}:=\mathbb L_G^{-1}(\delta_{\mathcal E})
\in\operatorname{DMod}_{1/2}(\operatorname{Bun}_G)
$$
满足 Hecke eigensheaf 条件。

**证明.** O.15 的 spectral/Hecke compatibility 说明逆等价 $\mathbb L_G^{-1}$ 把谱侧 tautological tensor action 送回自动侧 Hecke functors。Residual gerbe 点对象在 $V$ 的作用下产生 $V_{\mathcal E}$；逆等价传到自动侧后给出
$$
\mathsf H_V(\mathcal F_{\mathcal E})
\simeq
\mathcal F_{\mathcal E}\boxtimes V_{\mathcal E},
$$
并继承对 $V$ 的 tensor compatibility，即 Hecke eigensheaf 条件。$\square$

## O.6 Betti、de Rham 和 de Rham Betti 比较

**定义 O.18.** 若 $k=\mathbb C$，$\widehat G$-local systems 可有 Betti 描述
$$
\operatorname{LocSys}_{\widehat G}^{\operatorname{Betti}}(X)
\simeq
\operatorname{Map}(X_{\operatorname{top}},B\widehat G)
$$
和 de Rham 描述，即 flat $\widehat G$-bundles。

**外部输入定理 O.19（Riemann-Hilbert correspondence）.** 对复光滑代数簇，regular holonomic D-modules 与 perverse sheaves/constructible sheaves 之间有 Riemann-Hilbert correspondence。用于 Betti 与 de Rham 几何 Langlands 比较的 stacky 和 derived 推广需要分别固定 holonomicity、constructibility 和 finiteness hypotheses。

## O.7 本附录小结

本附录给出几何 Langlands 范畴形式的技术口径：

1. D-modules 可视为 de Rham stack 上的 sheaves。
2. Hecke functors 是 Hecke correspondence 的 kernel transforms。
3. 谱侧 local systems stack 是 derived 且 singular。
4. IndCoh 比 QCoh 更适合 singular spectral side。
5. Nilpotent singular support 条件定义正确大小的谱侧范畴。
6. 特征零 unramified categorical geometric Langlands 按 O.15 登记为外部 preprint theorem；其 $\mathbb L_G$ 为 half-twisted automorphic $\to$ spectral，反向调用写 $\mathbb L_G^{-1}$。正特征只按 O.16.1 调用某些参数连通分支之并上的部分结果，其他分支、系数、ramification 与 local variants 仍是独立边界。

## 练习

**练习 O.1.** 对 smooth variety $X$，解释 flat connection 与 D-module 结构的关系。

**练习 O.2.** 写出 Hecke correspondence 如何定义 Hecke functor 的 kernel transform。

**练习 O.3.** 说明为什么 $\operatorname{LocSys}_{\widehat G}(X)$ 应视为 derived stack。

**练习 O.4.** 解释 nilpotent singular support 条件为什么是谱侧的限制，而不是自动侧的限制。

**练习 O.5.** 在 $G=\mathbb G_m$ 情形，比较 O.15 与几何类域论。
