# 第十三章：Geometric Satake 等价与 Tannakian reconstruction

## 本章目标

本章陈述 geometric Satake equivalence，并解释它如何从 affine Grassmannian 上的 convolution perverse sheaves 重构 Langlands dual group 的表示范畴。该章是本书从经典几何表示论进入 Langlands 方向的关键节点。

## 依赖前置知识

需要第十二章的 affine Grassmannian 和 convolution，第三章的 perverse sheaves，以及 Tannakian categories 的基本语言。

## 13.1 Satake category 的张量结构

**定义 13.1.** Satake category 记为
$$
\operatorname{Sat}_G=\operatorname{Perv}_{L^+G}(\operatorname{Gr}_G,E).
$$
其对象要求支撑在有限个 Schubert varieties 的并上。

**外部输入定理 13.2.** Convolution product 保持 perverse sheaves，即若 $\mathcal F,\mathcal G\in\operatorname{Sat}_G$，则
$$
\mathcal F\star\mathcal G\in\operatorname{Sat}_G.
$$
此外 $\operatorname{Sat}_G$ 在 convolution 下成为 rigid symmetric monoidal category。  
来源：Mirkovic-Vilonen、Ginzburg、Beilinson-Drinfeld、Zhu 等。

**定义 13.3.** global cohomology functor 定义为
$$
\omega:\operatorname{Sat}_G\to\mathbf{Vect}_E,\qquad
\omega(\mathcal F)=H^\ast(\operatorname{Gr}_G,\mathcal F).
$$
在 geometric Satake theorem 中，它是 tensor functor。

**外部输入定理 13.4.** Global cohomology functor $\omega$ 是 exact faithful tensor functor。  
这是 Tannakian reconstruction 的输入之一。

## 13.2 Langlands dual group

**定义 13.5.** 若 $G$ 的 root datum 为
$$
(X^\ast(T),\Phi,X_\ast(T),\Phi^\vee),
$$
则 Langlands dual group $G^\vee$ 是 root datum
$$
(X_\ast(T),\Phi^\vee,X^\ast(T),\Phi)
$$
对应的 connected reductive group over $E$。

**外部输入定理 13.6.** Geometric Satake equivalence：存在 tensor equivalence
$$
\operatorname{Sat}_G\simeq \operatorname{Rep}_E(G^\vee)
$$
把 $\operatorname{IC}_\lambda$ 对应到 $G^\vee$ 的最高权为 $\lambda$ 的 irreducible representation，其中 $\lambda\in X_\ast(T)^+$ 被视为 $G^\vee$ 的 dominant weight。

资料入口：Mirkovic-Vilonen arXiv:math/0401222，Zhu arXiv:1603.05593。

**命题 13.7.** 若接受定理 13.6，则 simple objects of $\operatorname{Sat}_G$ 由 dominant coweights 参数化。

**证明.** $\operatorname{Rep}_E(G^\vee)$ 的 simple finite-dimensional representations 由 dominant weights 参数化。$G^\vee$ 的 dominant weights 按定义是 $G$ 的 dominant coweights。tensor equivalence 保持 simple objects 和同构类，因此 $\operatorname{Sat}_G$ 的 simple objects 由 $X_\ast(T)^+$ 参数化。$\square$

## 13.3 Tannakian reconstruction

**外部输入定理 13.8.** Tannakian reconstruction：若 $(\mathcal C,\otimes,\mathbf 1)$ 是合适的 rigid abelian symmetric monoidal $E$-linear category，且有 exact faithful tensor functor
$$
\omega:\mathcal C\to\mathbf{Vect}_E,
$$
则
$$
\mathcal C\simeq\operatorname{Rep}_E(\operatorname{Aut}^\otimes(\omega)).
$$

**依赖说明 13.9.** Geometric Satake 的证明需要先证明 $\operatorname{Sat}_G$ 是 Tannakian category，再识别
$$
\operatorname{Aut}^\otimes(\omega)\simeq G^\vee.
$$
后一步使用 weight functors、MV cycles、root datum 计算和 commutativity constraint。

## 13.4 Weight functors 和 root datum 识别

**定义 13.10.** 对 coweight $\mu\in X_\ast(T)$，令 $S_\mu$ 表示 affine Grassmannian 中的 semi-infinite orbit
$$
S_\mu=N((z))\cdot z^\mu.
$$
weight functor 的基本形式为
$$
F_\mu(\mathcal F)=H_c^{\langle2\rho,\mu\rangle}(S_\mu,\mathcal F),
$$
其中 degree shift 依赖 $\rho$ 和 perversity convention。

**外部输入定理 13.11.** 对 $\mathcal F\in\operatorname{Sat}_G$，semi-infinite orbit cohomology 集中在预期次数，并且
$$
\omega(\mathcal F)\simeq\bigoplus_{\mu\in X_\ast(T)}F_\mu(\mathcal F).
$$
该分解给出 Tannakian group 的 maximal torus 和 weight lattice。

**命题 13.12.** 若接受定理 13.11，则 $\operatorname{Aut}^\otimes(\omega)$ 的 character lattice 可由 $X_\ast(T)$ 识别。

**证明.** Tannakian group $H=\operatorname{Aut}^\otimes(\omega)$ 作用在每个 fiber $\omega(\mathcal F)$ 上。分解
$$
\omega(\mathcal F)=\bigoplus_\mu F_\mu(\mathcal F)
$$
对 tensor product 相容时，给出一个 diagonalizable subgroup $T^\vee\subset H$，其 characters 正由指标 $\mu\in X_\ast(T)$ 记录。tensor compatibility 保证卷积下权相加，故 character lattice 为 $X_\ast(T)$。剩余的根和 coroot 识别需要 MV cycles 与简单根方向的几何，属于 geometric Satake 的外部输入。$\square$

## 13.5 最小例子：$GL_1$

**例 13.13.** 对 $G=GL_1$，
$$
LG/L^+G=\mathbb C((z))^\times/\mathbb C[[z]]^\times\simeq\mathbb Z.
$$
每个连通分支是一个点，记为 $\operatorname{Gr}^n$。Satake category 等价于有限支撑的 $\mathbb Z$-graded vector spaces：
$$
\operatorname{Sat}_{GL_1}\simeq \mathbf{Vect}_E^{(\mathbb Z)}.
$$
卷积由整数加法给出：
$$
E_m\star E_n\simeq E_{m+n}.
$$
另一方面，$G^\vee=GL_1$ 的有限维表示分解为 characters
$$
V=\bigoplus_{n\in\mathbb Z}V_n.
$$
因此 geometric Satake 在该情形中退化为 $\mathbb Z$-grading 与 $GL_1$ characters 的等价。

**命题 13.14.** 上述 $GL_1$ 等价是 tensor equivalence。

**证明.** 定义 functor 把支撑在分支 $n$ 上的一维 skyscraper perverse sheaf $E_n$ 送到 character $t\mapsto t^n$。任一对象是有限直和，所以 functor 在对象上完全由分解决定。卷积满足 $E_m\star E_n\simeq E_{m+n}$，而 characters 张量满足 $\chi_m\otimes\chi_n=\chi_{m+n}$。Hom 空间在两侧均只在同一整数标号之间非零，且等于 $E$。故为 tensor equivalence。$\square$

## 13.6 与 classical Satake 的关系

**边界说明 13.15.** Classical Satake isomorphism 描述 $p$-adic group 的 spherical Hecke algebra。Geometric Satake 是其范畴化和几何化版本，但二者不在同一范畴中。要从 geometric Satake 推出函数层面的 Satake，需要 sheaf-function dictionary、Frobenius trace 和有限域模型。

## 本章小结

本章陈述 Satake category 的 tensor structure、Langlands dual group 和 geometric Satake equivalence，补充了 weight functors 的 root datum 识别角色和 $GL_1$ 的完整检验例。卷积 t-exactness、对称约束、Tannakian reconstruction 和一般 root datum 识别仍为外部输入。

## 练习

**练习 13.1.** 对 $G=GL_1$，说明 $\operatorname{Gr}_G$ 的连通分支与 $G^\vee=GL_1$ 的 characters 对应。

**练习 13.2.** 写出 $G=SL_2$ 时 dominant coweights 和 $G^\vee=PGL_2$ dominant weights 的对应。

**练习 13.3.** 列出证明 geometric Satake 时必须检查的五个结构：perversity、convolution、commutativity、fiber functor、root datum。

**练习 13.4.** 对 $G=GL_1$，直接验证 Verdier duality 对应表示的 dual character。

**练习 13.5.** 说明为什么 $SL_2$ 的 geometric Satake 不能只由 $\operatorname{Gr}_G$ 的连通分支给出。
