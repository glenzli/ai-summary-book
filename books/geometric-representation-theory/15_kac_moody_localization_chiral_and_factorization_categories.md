# 第十五章：Kac-Moody localization、chiral algebras 与 factorization categories

## 本章目标

本章登记 Kac-Moody algebras、affine flag/Grassmannian 上的 localization、critical level、chiral algebras 和 factorization categories 的基本接口。

## 依赖前置知识

需要第十二、十四章的 loop group 几何，第七章的 D-modules，以及基本 Kac-Moody algebra 知识。

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

**外部输入定理 15.12.** Geometric Langlands proof series 中的 Fundamental Local Equivalence 使用 critical level Kac-Moody localization、factorization categories 和 ind-coherent sheaves 的深层 formalism。当前只作为研究边界。

## 本章小结

本章给出 affine Kac-Moody algebra 的基本定义、residue cocycle 计算、$\mathfrak g[[z]]$ isotropic 检查和真空模泛性质，并把 Kac-Moody localization、critical level、chiral/factorization categories 标为外部输入。

## 练习

**练习 15.1.** 验证 residue cocycle 对 $f=z^m$、$g=z^n$ 的值。

**练习 15.2.** 解释为什么 $\mathfrak g[[z]]$ 在标准 residue cocycle 下给出 isotropic 子代数。

**练习 15.3.** 列出 Kac-Moody localization 相比有限维 localization 多出的五项技术条件。

**练习 15.4.** 对 $m+n=0$ 的情形，比较 $c(xz^m,yz^n)$ 和 $c(yz^n,xz^m)$，验证反对称性。
