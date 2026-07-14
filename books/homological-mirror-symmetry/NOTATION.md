# Homological Mirror Symmetry 符号约定

## 集合论与线性约定

- 固定 Grothendieck universes
  $$
  \mathcal U\in\mathcal V\in\mathcal W.
  $$
  未特别说明时，“集合”指 $\mathcal U$-小集合，“范畴”指对象类属于 $\mathcal V$ 的范畴。
- 固定基域 $k$。未特别说明时，线性范畴、复形、dg category 和 $A_\infty$-category 均为 $k$-线性。
- 非 exact Floer 章节另固定带继承全序的加法子群
  $\Gamma\subset\mathbb R$，并记
  $\Lambda=\Lambda_{k,\Gamma}$；只有 $\Gamma=\mathbb R$ 时称其为 universal
  Novikov field。乘法与 valuation 采用定义 5.1。
- `Ch_k` 表示 cohomological grading 的链复形范畴；微分次数为 $+1$。
- 对分次向量空间 $V$，平移 $V[1]$ 采用 $V[1]^i=V^{i+1}$。Suspension
  记为 $sV=V[1]$，故 $|sa|=|a|-1$。$A_\infty$ 张量按
  $sa_d\otimes\cdots\otimes sa_1$ 排列，coderivation 符号以附录 B 的
  (B.1)--(B.3) 为唯一约定。

## 范畴与增强

- $\operatorname{Cat}_{\mathrm{dg},k}$：小 $k$-线性 dg categories 的范畴。
- $\operatorname{Cat}_{A_\infty,k}$：小严格含单位 $k$-线性 $A_\infty$-categories 与 $A_\infty$-functors 的语境。
- $\operatorname{Cat}^{\mathrm{perf}}_k$：小、幂等完备、$k$-线性 stable
  categories 的 Morita 局部化；dg/$A_\infty$ 模型先取 perfect-module
  stable category 再进入此记号。
- 对 dg 或 $A_\infty$ category $\mathcal A$，$H^0(\mathcal A)$ 表示同对象、morphism 为 $H^0\operatorname{hom}_{\mathcal A}(X,Y)$ 的普通范畴。
- $\operatorname{Tw}(\mathcal A)$ 表示 twisted complexes 构成的预三角化包；$\operatorname{Perf}(\mathcal A)$ 表示 perfect right $\mathcal A$-modules 的 dg 或 $A_\infty$ category。
- “quasi-equivalence” 指 morphism complexes 上 quasi-isomorphism 且 $H^0$ 上本质满的 dg/$A_\infty$ functor。
- “Morita equivalence” 指诱导 perfect module categories 的 quasi-equivalence。
- $\mathcal G$ split-generates $\mathcal A$ 指 $Y_G$ 的厚闭包等于
  $H^0\operatorname{Perf}(\mathcal A)$；除非 $\mathcal A$ 已 Morita-complete，
  不把目标简写为 $H^0(\mathcal A)$。
- “Morita-complete” 指 Yoneda functor
  $\mathcal A\to\operatorname{Perf}(\mathcal A)$ 是 quasi-equivalence。

## B-side 几何

- 对 qcqs $k$-scheme $X$，$\operatorname{Perf}_{\mathrm{dg}}(X)$ 表示约定
  2.10 的 h-injective dg model 中 perfect objects 的小 skeleton；省略下标
  $\mathrm{dg}$ 时必须已在局部语境固定该 model。
- $\mathrm D^b\operatorname{Coh}(X)$ 表示 ordinary triangulated category；
  $\mathrm D^b_{\mathrm{dg}}\operatorname{Coh}(X)$ 表示约定 2.10 的 dg
  enhancement。Regular noetherian 情形二者与
  $\operatorname{Perf}_{\mathrm{dg}}(X)$ 的关系见定理 2.7、2.11。
- 对 $X=\operatorname{Spec}R$、$w\in R$，
  $\operatorname{MF}^{\mathrm{fr}}_{\mathrm{dg}}(R,w)$ 是 finite-projective
  affine dg model；$\operatorname{MF}(X,w)$ 默认表示其 pretriangulated、
  idempotent-complete Morita envelope。Nonaffine/graded/equivariant 版本必须
  另行声明。
- Fourier-Mukai kernel 记为 $K\in \operatorname{Perf}(X\times Y)$，相应函子记为
  $$
  \Phi_K(-)=\mathbf R p_{Y*}(p_X^*(-)\otimes^{\mathbf L} K).
  $$

## A-side 几何

- $(M,\omega)$ 表示辛流形；若 $\omega=d\lambda$ 且带 Liouville 向量场，则写作 $(M,\lambda)$。
- $L\subset M$ 表示 Lagrangian submanifold；compact exact brane 写为
  $\mathbb L=(L,f_L,\alpha_L,\mathfrak p_L,E_L)$，依次记录 chosen primitive、
  grading、relative Pin/spin data 与 finite-rank local system。
- $\mathcal F^c_{\mathrm{ex}}(\widehat M;\mathscr L,k)$ 表示第四章固定的
  compact exact Hamiltonian-chord model；$\mathcal F(M)$ 只在 scope 已明确时
  简写。$\mathcal W(M)$ 表示 wrapped Fukaya category。
- 对两个横截 branes $L_0,L_1$，$CF^\ast(L_0,L_1)$ 表示 Floer cochain complex。
- $\mathcal X(L_0,L_1)$ 表示固定 pair Floer datum 的非退化 Hamiltonian
  chords；$o_x$ 表示 chord 的 rank-one orientation module。
- $CW^\ast(L_0,L_1)$ 默认指定义 6.7 的 continuation mapping telescope，
  不是 raw finite-slope complexes 的普通 colimit。
- $HW^\ast(L_0,L_1)$ 表示 wrapped Floer cohomology。
- $QH^\ast(M)$ 表示 quantum cohomology；$SH^\ast(M)$ 表示 symplectic cohomology。
- 对 $\dim M=2n$，$\mathcal{OC}$ 采用
  $HH_\bullet(\mathcal W(M))\to SH^{\bullet+n}(M)$ 的 degree-$n$ 约定；
  $\mathcal{CO}:SH^\bullet(M)\to HH^\bullet(\mathcal W(M))$ 表示
  closed-open map。
- $\mathcal F\mathcal S(W)$ 表示 Landau-Ginzburg potential $W$ 的 Fukaya-Seidel category。
- $\mathcal W_{\mathrm{fib}}(Y,W)$ 表示来源指定的 fiberwise wrapped Fukaya
  category；它不与普通 $\mathcal W(Y)$ 自动等同。

## Landau-Ginzburg 与奇点

- $W:X\to\mathbb A^1$ 或 $W:Y\to\mathbb C$ 表示 Landau-Ginzburg potential。
- $\operatorname{Jac}(W)$ 表示 Jacobian ring；在 Laurent 情况使用 $z_i\partial W/\partial z_i$ 生成的理想。
- $\mathrm D_{\mathrm{sg}}(X)$ 表示 singularity category。
- $\mathcal R\mathcal F(F)$ 表示 Rabinowitz Fukaya category；具体模型必须在章内声明。

## Sheaf 与 microlocal

- $\operatorname{Sh}_c(Q)$ 表示 constructible sheaves 的增强范畴。
- $SS(\mathcal F)$ 表示 sheaf $\mathcal F$ 的 microsupport。
- $\operatorname{Sh}_{\Lambda}(Q)$ 表示 microsupport 包含于 $\Lambda\subset T^\ast Q$ 的 sheaf category。
- 对光滑 $S\subset Q$，$T_S^\ast Q$ 表示其 conormal bundle。

## 稳定性与函子化数据

- $\Gamma_{\mathrm{ch}}$ 表示有限秩自由 charge lattice，避免与第五章的
  Novikov 值群 $\Gamma\subset\mathbb R$ 混用；
  $\Gamma_{\mathrm{ch},\mathbb R}=\Gamma_{\mathrm{ch}}\otimes_\mathbb Z\mathbb R$。
- $v:K_0(\mathcal C)\to\Gamma_{\mathrm{ch}}$ 是类映射，
  $Z:\Gamma_{\mathrm{ch}}\to\mathbb C$ 是 central charge，
  $\mathcal P(\phi)$ 是 Bridgeland slicing 的相位子范畴。
- 对 spherical object $S$，$T_S$ 表示由 evaluation triangle 定义的
  spherical twist；它与泛称的 wall-crossing transformation 不作同义使用。

## HMS 断言

- `HMS(A,B)` 表示一条包含如下数据的断言：A-side 几何对象、B-side 几何对象、
  系数、增强 models、raw/pretriangulated/perfect completion、等价类型和候选
  等价函子。
- 本书不把 HMS 写成裸等式。允许的标准形式包括：
  $$
  \mathcal F(A)\simeq \operatorname{Perf}(B),\qquad
  \mathcal W(A)\simeq \operatorname{MF}(B,W),\qquad
  \mathcal F\mathcal S(W_A)\simeq \operatorname{Perf}(B),
  $$
  其中 $\simeq$ 必须在章内解释为 quasi-equivalence、Morita equivalence 或 stable $\infty$-category equivalence。
