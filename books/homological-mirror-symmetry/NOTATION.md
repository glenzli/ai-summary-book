# Homological Mirror Symmetry 符号约定

## 集合论与线性约定

- 固定 Grothendieck universes
  $$
  \mathcal U\in\mathcal V\in\mathcal W.
  $$
  未特别说明时，“集合”指 $\mathcal U$-小集合，“范畴”指对象类属于 $\mathcal V$ 的范畴。
- 固定基域 $k$。未特别说明时，线性范畴、复形、dg category 和 $A_\infty$-category 均为 $k$-线性。
- `Ch_k` 表示 cohomological grading 的链复形范畴；微分次数为 $+1$。
- 对分次向量空间 $V$，平移 $V[1]$ 采用 $V[1]^i=V^{i+1}$。suspension 记为 $sV=V[1]$。

## 范畴与增强

- $\operatorname{Cat}_{\mathrm{dg},k}$：小 $k$-线性 dg categories 的范畴。
- $\operatorname{Cat}_{A_\infty,k}$：小严格含单位 $k$-线性 $A_\infty$-categories 与 $A_\infty$-functors 的语境。
- 对 dg 或 $A_\infty$ category $\mathcal A$，$H^0(\mathcal A)$ 表示同对象、morphism 为 $H^0\operatorname{hom}_{\mathcal A}(X,Y)$ 的普通范畴。
- $\operatorname{Tw}(\mathcal A)$ 表示 twisted complexes 构成的预三角化包；$\operatorname{Perf}(\mathcal A)$ 表示 perfect right $\mathcal A$-modules 的 dg 或 $A_\infty$ category。
- “quasi-equivalence” 指 morphism complexes 上 quasi-isomorphism 且 $H^0$ 上本质满的 dg/$A_\infty$ functor。
- “Morita equivalence” 指诱导 perfect module categories 的 quasi-equivalence。

## B-side 几何

- 对 $k$-scheme 或 $k$-variety $X$，$\operatorname{Perf}(X)$ 表示 perfect complexes 的 dg/stable enhancement。
- $\mathrm D^b\operatorname{Coh}(X)$ 表示 bounded derived category of coherent sheaves。若 $X$ 光滑且适当，则正文会说明何时可与 $\operatorname{Perf}(X)$ 的同伦范畴比较。
- 若 $W:X\to\mathbb A^1$ 是 Landau-Ginzburg potential，$\operatorname{MF}(X,W)$ 表示 matrix factorizations 的增强范畴。
- Fourier-Mukai kernel 记为 $K\in \operatorname{Perf}(X\times Y)$，相应函子记为
  $$
  \Phi_K(-)=\mathbf R p_{Y*}(p_X^*(-)\otimes^{\mathbf L} K).
  $$

## A-side 几何

- $(M,\omega)$ 表示辛流形；若 $\omega=d\lambda$ 且带 Liouville 向量场，则写作 $(M,\lambda)$。
- $L\subset M$ 表示 Lagrangian submanifold；带 grading、orientation、spin 或 Pin 结构、局部系统和必要的 bounding cochain 后写作 Lagrangian brane。
- $\mathcal F(M)$ 表示 compact Fukaya category；$\mathcal W(M)$ 表示 wrapped Fukaya category。具体章内必须说明 exact、monotone、Novikov、obstructed 或 curved 口径。
- 对两个横截 branes $L_0,L_1$，$CF^\ast(L_0,L_1)$ 表示 Floer cochain complex。
- $HW^\ast(L_0,L_1)$ 表示 wrapped Floer cohomology。
- $QH^\ast(M)$ 表示 quantum cohomology；$SH^\ast(M)$ 表示 symplectic cohomology。
- $\mathcal{OC}$ 表示 open-closed map，$\mathcal{CO}$ 表示 closed-open map。
- $\mathcal F\mathcal S(W)$ 表示 Landau-Ginzburg potential $W$ 的 Fukaya-Seidel category。

## Landau-Ginzburg 与奇点

- $W:X\to\mathbb A^1$ 或 $W:Y\to\mathbb C$ 表示 Landau-Ginzburg potential。
- $\operatorname{Jac}(W)$ 表示 Jacobian ring；在 Laurent 情况使用 $z_i\partial W/\partial z_i$ 生成的理想。
- $\mathrm D_{\mathrm{sg}}(X)$ 表示 singularity category。
- $\mathcal R\mathcal F(F)$ 表示 Rabinowitz Fukaya category；具体模型必须在章内声明。

## Sheaf 与 microlocal

- $\operatorname{Sh}_c(Q)$ 表示 constructible sheaves 的增强范畴。
- $SS(\mathcal F)$ 表示 sheaf $\mathcal F$ 的 microsupport。
- $\operatorname{Sh}_{\Lambda}(Q)$ 表示 microsupport 包含于 $\Lambda\subset T^\ast Q$ 的 sheaf category。

## HMS 断言

- `HMS(A,B)` 表示一条包含如下数据的断言：A-side 几何对象、B-side 几何对象、系数、增强类别、等价类型和候选等价函子。
- 本书不把 HMS 写成裸等式。允许的标准形式包括：
  $$
  \mathcal F(A)\simeq \operatorname{Perf}(B),\qquad
  \mathcal W(A)\simeq \operatorname{MF}(B,W),\qquad
  \mathcal F\mathcal S(W_A)\simeq \operatorname{Perf}(B),
  $$
  其中 $\simeq$ 必须在章内解释为 quasi-equivalence、Morita equivalence 或 stable $\infty$-category equivalence。
