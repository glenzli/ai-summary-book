# 第十五章：Kac-Moody localization、chiral algebras 与 factorization categories

Loop group 的 Lie algebra $\mathfrak g((z))$ 不能只用逐点 bracket 描述其表示论：invariant form 与 residue 产生一个中心扩张，中心的标量作用就是 level。Fourier modes 的计算说明只有次数相加为零时才出现中心项，而 $\mathfrak g[[z]]$ 在 cocycle 下 isotropic，因此可以从它诱导真空模。有限维 Beilinson--Bernstein localization 在这里变成 affine flag 或 affine Grassmannian 上的 Kac--Moody localization；当插入点在曲线上移动时，多个局部对象还必须满足 factorization。以下只在 residue cocycle 与诱导模层面完成书内证明，更深的 critical-level、chiral 与 factorization 等价保持为精确的外部输入边界。

## 15.1 Affine Kac-Moody algebra

**定义 15.1.** 令 $\mathfrak g$ 为 simple Lie algebra。loop algebra 为
$$
\mathfrak g((z))=\mathfrak g\otimes_\mathbb C\mathbb C((z)).
$$
其标准中心扩张为
$$
0\to\mathbb C\mathbf 1\to\widehat{\mathfrak g}\to\mathfrak g((z))\to0,
$$
括号由 invariant form $\kappa$ 和 residue cocycle
$$
c(x\otimes f,y\otimes g)=\kappa(x,y)\operatorname{Res}_{z=0}(f\,dg)
$$
给出。

**命题 15.2.** residue cocycle 是 Lie algebra 2-cocycle。

**证明.** 反对称性来自
$$
\operatorname{Res}(f\,dg)+\operatorname{Res}(g\,df)=\operatorname{Res}d(fg)=0.
$$
Jacobi cocycle 条件化为 invariant form 的不变性
$$
\kappa([x,y],z)=\kappa(x,[y,z])
$$
和 residue 对全微分为零的事实。逐项展开三循环和后，各项可配对成 $\operatorname{Res}d(fgh)$ 类型或由 $\kappa$ 的不变性相消。$\square$

**定义 15.3.** level 是中心元素 $\mathbf 1$ 的标量作用。critical level 由 invariant form 的临界值给出，常记为 $\kappa_c$。

**例 15.4.** 若 $x,y\in\mathfrak g$ 且 $m,n\in\mathbb Z$，则
$$
c(xz^m,yz^n)=\kappa(x,y)\operatorname{Res}(z^m\,d z^n)
=n\kappa(x,y)\operatorname{Res}(z^{m+n-1}dz).
$$
因此
$$
c(xz^m,yz^n)=
\begin{cases}
n\kappa(x,y),&m+n=0,\\
0,&m+n\ne0.
\end{cases}
$$
这给出通常的 affine Kac-Moody bracket
$$
[xz^m,yz^n]=[x,y]z^{m+n}+n\delta_{m+n,0}\kappa(x,y)\mathbf 1.
$$

**命题 15.5.** 子代数 $\mathfrak g[[z]]$ 对 residue cocycle isotropic。

**证明.** 若 $f,g\in\mathbb C[[z]]$，则 $f\,dg$ 是形式幂级数乘以 $dz$，不含 $z^{-1}dz$ 项。因此
$$
\operatorname{Res}(f\,dg)=0.
$$
所以 $c$ 限制到 $\mathfrak g[[z]]$ 上为零。$\square$

Isotropic 性使中心扩张在 $\mathfrak g[[z]]$ 上分裂，因而“正 loop 平凡作用、中心按 level 作用”的一维模确实可以拿来作 induction。这个局部代数对象正是几何 localization 的基本测试对象。

## 15.2 Localization 的仿射版本

**外部输入定理 15.6.** Kac-Moody localization 把适当 level 的 $\widehat{\mathfrak g}$-modules 与 affine flag variety 或 affine Grassmannian 上的 twisted D-modules 联系起来。critical level 情形与 Feigin-Frenkel center、opers 和 local geometric Langlands 密切相关。

**警告 15.7.** Kac-Moody localization 不是第八章 Beilinson-Bernstein localization 的形式替换。它涉及 ind-schemes、completed enveloping algebras、level、monodromy、renormalized categories 和 compact generation 等额外假设。

**定义 15.8.** 给定 level $\kappa$，真空模可形式写为
$$
\mathbb V_\kappa=\operatorname{Ind}_{\mathfrak g[[z]]\oplus\mathbb C\mathbf 1}^{\widehat{\mathfrak g}} \mathbb C_\kappa,
$$
其中 $\mathfrak g[[z]]$ 平凡作用，$\mathbf 1$ 以 $\kappa$ 作用。该对象是 affine vertex algebra 和 localization 的基本输入之一。

**命题 15.9.** $\mathbb V_\kappa$ 具有诱导模泛性质。

**证明.** 对任一 $\widehat{\mathfrak g}$-module $M$，给出 $\widehat{\mathfrak g}$-module map $\mathbb V_\kappa\to M$ 等价于给出向量 $m\in M$，满足
$$
\mathfrak g[[z]]m=0,\qquad \mathbf 1\cdot m=\kappa m.
$$
这正是诱导模
$$
U(\widehat{\mathfrak g})\otimes_{U(\mathfrak g[[z]]\oplus\mathbb C\mathbf 1)}\mathbb C_\kappa
$$
的 tensor-Hom adjunction。$\square$

## 15.3 Chiral 和 factorization 语言

**定义 15.10.** 本书暂把 chiral category 视为曲线 Ran space 或 configuration spaces 上满足 factorization 约束的 sheaf/category 数据。严格模型可采用 Beilinson-Drinfeld chiral algebras、factorization algebras 或 factorization categories。

**定义 15.11.** factorization category 的骨架包括：

1. 对每个有限集合 $I$，给出 $X^I$ 或 Ran space 上的 category；
2. 对 disjoint configurations 给出外张量分解；
3. 对 diagonals 给出相容的 fusion 或 specialization functors；
4. 满足 associativity、unit 和 descent 条件。

**研究边界 15.12.** Fundamental Local Equivalence 的现代表述使用 critical-level Kac--Moody localization、factorization categories 和 ind-coherent sheaves 的深层 formalism。这里只记录这些对象之间的接口，不把该等价用于任何书内证明。

Residue cocycle 把 loop modes 的局部信息凝聚为中心项，$\mathfrak g[[z]]$ 的 isotropic 性则允许构造真空模；两者共同说明 level 怎样进入 affine localization。Factorization 要求这些局部构造随曲线上多个互异点张量分解、沿对角线融合。下一章把这种局部 Hecke/fusion 数据放到整条曲线上的 $G$-bundles，形成 geometric Langlands 的 automorphic 与 spectral 两侧。

## 练习

**练习 15.1.** 验证 residue cocycle 对 $f=z^m$、$g=z^n$ 的值。

**练习 15.2.** 解释为什么 $\mathfrak g[[z]]$ 在标准 residue cocycle 下给出 isotropic 子代数。

**练习 15.3.** 列出 Kac-Moody localization 相比有限维 localization 多出的五项技术条件。

**练习 15.4.** 对 $m+n=0$ 的情形，比较 $c(xz^m,yz^n)$ 和 $c(yz^n,xz^m)$，验证反对称性。
