# 附录 I：Geometric Satake 的 Tannakian 细节

## 本章目标

本附录记录 geometric Satake 中 Tannakian reconstruction 的检查清单。精确模型沿用约定 13.0：complex Betti sheaves、reduced affine Grassmannian、代数闭 characteristic-zero 系数域和 finite-dimensional support。

## I.1 Tannakian 输入

**检查表 I.1.** 要把 $\operatorname{Sat}_G$ 识别为 $\operatorname{Rep}(G^\vee)$，需要：

1. $\operatorname{Sat}_G$ 是 $E$-linear abelian category；
2. convolution 是 exact bifunctor；
3. 有 associativity、unit、commutativity constraints；
4. 每个对象有 dual；
5. global cohomology 是 exact faithful tensor functor；
6. automorphism group scheme 的 root datum 可识别为 dual root datum。

**外部输入定理 I.2.** Mirkovic--Vilonen theorem 验证上述结构并给出 geometric Satake equivalence。各项 locator 分别为 `GSAT-CONV-1`、`GSAT-FIBER-1`、`GSAT-WEIGHT-1` 和 `GSAT-1`；不能只用主等价的名称替代 convolution、fusion 与 root-datum identification 的独立假设。

**推论 I.2.1.** 若 $\operatorname{Sat}_G$ 连同 $\omega:\operatorname{Sat}_G\to\mathbf{Vect}^{\mathrm{fd}}_E$ 满足定理 13.8 的全部 neutral Tannakian 假设，则
$$
\underline{\operatorname{Aut}}^\otimes(\omega)
$$
是 affine group scheme，并且 $\operatorname{Sat}_G$ 与其 finite-dimensional representation category symmetric monoidally equivalent。

**证明.** 逐项代入外部输入定理 13.8。该推论只构造某个 affine group scheme；finite type、reductivity 和 dual root datum 都不由这一步推出。$\square$

## I.2 Weight functors

**定义 I.3.** 对 $\mu\in X_\ast(T)$，令
$S_\mu=N(\mathscr K)z^\mu L^+G/L^+G$，并对 finite-support $\mathcal F$ 定义
$$
F_\mu(\mathcal F)
=\mathbb H_c^{\langle2\rho,\mu\rangle}(S_\mu,\mathcal F).
$$
Compact support 在 $S_\mu\cap\operatorname{supp}\mathcal F$ 的 finite-dimensional stage 中计算；次数沿用定义 13.10，不再保留“依 convention”的未定 shift。

**外部输入定理 I.4.** 若 $\mathcal F\in\operatorname{Sat}_G$，则 semi-infinite cohomology 只在定义 I.3 的次数非零，且 global cohomology 有 natural finite direct-sum decomposition
$$
\omega(\mathcal F)
\simeq\bigoplus_{\mu\in X_\ast(T)}F_\mu(\mathcal F),
$$
并与 convolution 下的 weight addition 相容。来源定位：`GSAT-WEIGHT-1`；Mirkovic--Vilonen Theorem 3.6、Proposition 6.4。

**推论 I.4.1.** 定理 I.4 只先给出 split torus morphism
$$
T^\vee\longrightarrow
\underline{\operatorname{Aut}}^\otimes(\omega),
$$
maximality 和 root datum identification 仍为外部输入。

**证明.** 对每个 $t\in T^\vee(R)$，让它在 $\mu$-weight summand 上乘以 $\mu(t)$；定理 I.4 的 convolution compatibility 使这些 automorphisms 成为 tensor automorphism。完整的 base-change 与 Yoneda 检查见命题 13.12。$\square$

**例 I.5.** 对 $G=GL_1$，reduced Betti ind-scheme $\operatorname{Gr}_G\simeq\mathbb Z$，每个 $L^+G$-orbit 是一个点。命题 13.14 直接构造 Satake category 与 $X_\ast(G)$-graded finite-dimensional vector spaces、再与 $\operatorname{Rep}(\mathbb G_m)$ 的 symmetric monoidal equivalence，因而在这个已完全计算的例子中得到
$$
G^\vee\simeq \mathbb G_m.
$$

## I.3 Root datum 识别

**检查表 I.6.** 从 $\operatorname{Sat}_G$ 恢复 $G^\vee$ 的 root datum 时必须检查：

1. simple objects 由 dominant coweights $\lambda\in X_\ast(T)^+$ 参数化；
2. convolution 的最高权项满足
   $$
   \operatorname{IC}_\lambda\star\operatorname{IC}_\mu
   \text{ contains }\operatorname{IC}_{\lambda+\mu}
   \text{ with multiplicity }1;
   $$
3. weight functors 的非零 weights 落在相应 representation 的 weight polytope 中；
4. simple coroot 数据由 codimension-one MV cycles 或 root operators 识别；
5. commutativity constraint 与 dual root datum 的 signs 相容。

**边界说明 I.7.** Tannakian reconstruction 本身只给出某个 affine group scheme。把它识别为 split reductive group $G^\vee$ 是 geometric Satake 的核心内容，不能只由 formal Tannakian theorem 推出。

**边界说明 I.8.** Integral、modular 与 mixed-characteristic Satake 不共享本附录的 characteristic-zero semisimplicity。若改变 coefficient ring 或 affine-Grassmannian model，必须重做 exactness、duals、fiber functor target 和 group-scheme finiteness 检查。

## 本章小结

本附录把 geometric Satake 的证明需求拆为可审查清单：Tannakian 输入、weight-functor 外部定理、$GL_1$ 完整计算和 root-datum boundary。正式使用 Satake 等价时必须同时说明 convolution/fusion、fiber functor 与 dual group 识别，不能让形式 Tannaka 承担后两者的证明责任。
