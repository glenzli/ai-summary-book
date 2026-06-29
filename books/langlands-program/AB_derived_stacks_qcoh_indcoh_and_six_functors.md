# 附录 AB：Derived Stacks、QCoh/IndCoh 和 Six Functors 技术接口

收口归一化回指：本附录的 derived、IndCoh、six functors 和 spectral action 仅作为几何 Langlands 技术接口；与 Hecke 函数迹或 sheaf-function 比较时按 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 第 9 节处理。

## AB.1 Derived affine schemes

几何 Langlands 的现代形式不能只使用 classical algebraic stacks。本附录固定 derived algebraic geometry 的最小接口。

**定义 AB.1.** Derived affine scheme 是形如
$$
\operatorname{Spec}A
$$
的对象，其中 $A$ 为 connective commutative differential graded algebra 或 connective $E_\infty$-algebra。其 classical truncation 为
$$
(\operatorname{Spec}A)_{\operatorname{cl}}=\operatorname{Spec}H^0(A).
$$

**定义 AB.2.** Derived stack 是 derived affines 上满足 etale、smooth 或 fppf Grothendieck topology descent 的 functor
$$
X:\operatorname{CAlg}^{\operatorname{cn}}\to\mathcal S
$$
并满足代数性、几何性或 Artin 条件。其 cotangent complex 记为
$$
\mathbb L_X.
$$

**外部输入定理 AB.3（cotangent complex formalism）.** 对 locally almost of finite presentation 的 derived Artin stacks，cotangent complex 存在并满足 transitivity triangle、base change、smooth/etale 判别和 deformation theory 控制性质。

**命题 AB.4.** Derived structure 记录 classical truncation 丢失的 obstruction data。

**证明.** Classical truncation 只保留 $H^0(A)$。若 $A$ 有非零负上同调或同伦群，则 maps into $A$ 的 deformation theory 由 $\mathbb L_X$ 的高阶信息控制。Obstruction classes 位于 cotangent complex 的 Ext groups 中，而这些 groups 不由 $H^0(A)$ 单独决定。因此 derived structure 是 obstruction theory 的必要载体。$\square$

## AB.2 Perfect、quasi-smooth 和 singularity stack

**定义 AB.5.** Derived stack $X$ 称为 perfect，若 $\operatorname{QCoh}(X)$ 紧生成且 compact objects 与 perfect complexes 相容。称 $X$ quasi-smooth，若 cotangent complex $\mathbb L_X$ 的 Tor-amplitude 在 $[-1,0]$ 中。

**定义 AB.6.** 若 $X$ quasi-smooth，其 scheme of singularities 或 singularity stack 形式上由
$$
\operatorname{Sing}(X)=\operatorname{Spec}_{X_{\operatorname{cl}}}\operatorname{Sym}H^1(\mathbb T_X)
$$
给出，其中 $\mathbb T_X=\mathbb L_X^\vee$。

**外部输入定理 AB.7（singular support theory）.** 对 quasi-smooth derived stacks，$\operatorname{IndCoh}(X)$ 的对象有 singular support，定义为 $\operatorname{Sing}(X)$ 中的 conical closed subset。给定 closed conical subset $\mathcal N$，可定义 full subcategory
$$
\operatorname{IndCoh}_{\mathcal N}(X)\subset\operatorname{IndCoh}(X).
$$

**命题 AB.8.** 若 $X$ smooth，则 singular support 条件退化。

**证明.** Smooth 时 $\mathbb L_X$ Tor-amplitude 在 $0$，故 $\mathbb T_X$ 无 $H^1$。于是
$$
\operatorname{Sing}(X)=X
$$
的零方向。所有 coherent/ind-coherent objects 的 singular support 只能落在零截面，因此 $\operatorname{IndCoh}_{\mathcal N}(X)$ 在 $\mathcal N$ 含零截面时与通常的 $\operatorname{IndCoh}(X)$ 无差别。$\square$

## AB.3 QCoh 与 IndCoh

**定义 AB.9.** 对 derived stack $X$，$\operatorname{QCoh}(X)$ 是 quasi-coherent sheaves 的 stable presentable category。$\operatorname{IndCoh}(X)$ 是 coherent sheaves 的 ind-completion，适合处理 singular stacks 上的 !-pullback 和 duality。

**外部输入定理 AB.10（QCoh/IndCoh comparison）.** 若 $X$ 为 eventually coconnective 且满足 Gaitsgory-Rozenblyum 框架中 finiteness hypotheses 的 derived stack，存在 functor
$$
\Upsilon_X:\operatorname{QCoh}(X)\to\operatorname{IndCoh}(X).
$$
当 $X$ smooth 时，该 functor 是等价；当 $X$ singular 时，$\operatorname{IndCoh}(X)$ 通常严格大于 $\operatorname{QCoh}(X)$。

**命题 AB.11.** 几何 Langlands 谱侧使用 $\operatorname{IndCoh}_{\mathcal N}$ 是为了保留 singular directions 但限制其大小。

**证明.** $\operatorname{LocSys}_{\widehat G}(X)$ 通常是 derived singular stack。若只用 $\operatorname{QCoh}$，会丢失某些 !-functorial 和 duality 行为；若用全部 $\operatorname{IndCoh}$，谱侧过大。Nilpotent singular support 条件 $\mathcal N$ 在 $\operatorname{Sing}(\operatorname{LocSys}_{\widehat G})$ 中切出受 Langlands dual nilpotent cone 控制的部分。因此 $\operatorname{IndCoh}_{\mathcal N}$ 同时保留 singular theory 和正确大小。$\square$

## AB.4 Six functors and kernels

**外部输入定理 AB.12（six functor formalism for sheaf theories）.** 对固定的 sheaf theory，包括 D-modules、constructible sheaves、IndCoh 或 variants，并在相应 finiteness、properness、smoothness 或 constructibility hypotheses 下，存在 functors
$$
f^*,\quad f_*,\quad f^!,\quad f_!,\quad \otimes,\quad \mathcal Hom
$$
并满足 projection formula、base change、proper pushforward、smooth pullback 和 Verdier/Serre duality。

**定义 AB.13.** 若
$$
X\xleftarrow{p}K\xrightarrow{q}Y
$$
为 correspondence，且 $\mathcal K$ 为 $K$ 上的 kernel，则对应 kernel functor 为
$$
\Phi_{\mathcal K}(\mathcal F)=q_!(p^!\mathcal F\otimes\mathcal K)
$$
或按 sheaf theory 使用 $*$/$!$ 的相应版本。

**命题 AB.14.** Hecke functor 是 kernel functor 的特例。

**证明.** Hecke stack 给出 correspondence
$$
\operatorname{Bun}_G\xleftarrow{h_1}\operatorname{Hecke}_G\xrightarrow{h_2}\operatorname{Bun}_G\times X.
$$
几何 Satake 给出 kernel $\mathcal S_V$。代入定义 AB.13 得
$$
\mathsf H_V(\mathcal F)=h_{2,!}(h_1^!\mathcal F\otimes\mathcal S_V),
$$
即 Hecke functor 的形式。$\square$

## AB.5 Renormalization and non-proper stacks

**定义 AB.15.** 若 stack $X$ 非 quasi-compact 或非 finite type，naive sheaf category 可能不满足 compact generation 或 !-extension 的良好性质。Renormalized category 是通过 compact objects、safe maps 或 co-truncative substacks 重新定义的 stable category，记为
$$
\operatorname{DMod}(X)_{\operatorname{ren}}
$$
或相应符号。

**外部输入定理 AB.16（renormalized D-modules on Bun）.** 对 $\operatorname{Bun}_G$，存在适合几何 Langlands 的 renormalized D-module category，支持 Hecke functors、Eisenstein functors、constant term functors 和 Verdier duality 的相容形式。

**命题 AB.17.** $\operatorname{Bun}_G$ 的非紧性迫使 renormalization 进入范畴化几何 Langlands。

**证明.** $\operatorname{Bun}_G$ 按 Harder-Narasimhan strata 分解，通常非 quasi-compact。Naive D-module category 的 compact objects 和 !-pushforward 行为不足以使 Eisenstein/constant term adjunctions 同时良好。Renormalization 通过改变生成对象或连续性条件修正这些 functorial properties。因此范畴化几何 Langlands 的自动侧不能只写成未修正的 naive category。$\square$

## AB.6 Spectral action

**外部输入定理 AB.18（spectral action）.** 在几何 Langlands 中，谱侧 quasi-coherent 或 ind-coherent categories 作用于自动侧 $\operatorname{DMod}(\operatorname{Bun}_G)$，并与 Hecke action、Eisenstein functors 和 singular support 条件相容。

**命题 AB.19.** Hecke eigensheaf 是 spectral action 的点支撑特例。

**证明草图.** 谱侧点 $\mathcal E\in\operatorname{LocSys}_{\widehat G}$ 给出 skyscraper 或 residual gerbe 上的对象。若谱侧作用与 Hecke action 相容，则该点对象作用在自动侧对象上产生 Hecke eigenvalue 为 $\mathcal E$ 的对象。一般 spectral action 允许谱侧 sheaves 不只支撑在单点，因此是 Hecke eigensheaf 概念的范畴化扩展。$\square$

## 练习

**练习 AB.1.** 说明 derived structure 如何记录 obstruction data。

**练习 AB.2.** 解释 smooth stack 情形下 QCoh 与 IndCoh 的关系。

**练习 AB.3.** 把 Hecke functor 写成 correspondence kernel functor。

**练习 AB.4.** 说明 $\operatorname{Bun}_G$ 非 quasi-compact 对 D-module category 的影响。

**练习 AB.5.** 解释 nilpotent singular support 在谱侧的作用。
