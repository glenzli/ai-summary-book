# 附录 I：Geometric Satake 的 Tannakian 细节

## 本章目标

本附录记录 geometric Satake 中 Tannakian reconstruction 的检查清单。

## I.1 Tannakian 输入

**检查表 I.1.** 要把 $\operatorname{Sat}_G$ 识别为 $\operatorname{Rep}(G^\vee)$，需要：

1. $\operatorname{Sat}_G$ 是 $E$-linear abelian category；
2. convolution 是 exact bifunctor；
3. 有 associativity、unit、commutativity constraints；
4. 每个对象有 dual；
5. global cohomology 是 exact faithful tensor functor；
6. automorphism group scheme 的 root datum 可识别为 dual root datum。

**外部输入定理 I.2.** Mirkovic-Vilonen theorem 验证上述结构并给出 geometric Satake equivalence。

**命题 I.2.1.** 若 $\operatorname{Sat}_G$ 连同 $\omega:\operatorname{Sat}_G\to\operatorname{Vect}_E$ 满足 neutral Tannakian category 的全部假设，且 $\omega$ 是 exact faithful tensor functor，则
$$
\underline{\operatorname{Aut}}^\otimes(\omega)
$$
是 affine group scheme，并且 $\operatorname{Sat}_G$ 等价于其有限维表示范畴。

**证明.** 这是 neutral Tannakian reconstruction theorem 的直接应用。rigid abelian tensor category、有限性条件和 exact faithful fiber functor 共同保证 tensor automorphisms 形成 affine group scheme；Tannakian theorem 识别原范畴为该 group scheme 的表示范畴。该定理作为外部 Tannakian 输入使用，本书内部只检查 geometric Satake 提供这些条件的方式。$\square$

## I.2 Weight functors

**定义 I.3.** 对 coweight $\mu$，weight functor 常通过 semi-infinite orbit $S_\mu$ 上的 compactly supported cohomology定义：
$$
F_\mu(\mathcal F)=H_c^{\langle 2\rho,\mu\rangle}(S_\mu,\mathcal F).
$$
精确 shift 和支持条件依文献 convention。

**命题 I.4.** 若 $\mathcal F\in\operatorname{Sat}_G$，则 global cohomology 分解为 weight functors 的直和：
$$
H^\ast(\operatorname{Gr}_G,\mathcal F)\simeq \bigoplus_{\mu\in X_\ast(T)}F_\mu(\mathcal F).
$$

**证明.** semi-infinite orbits $S_\mu$ 给出与 $G(\mathcal O)$-orbits 横截的分层。Mirkovic-Vilonen 理论证明相应 compactly supported cohomology 集中在指定 degree，并且这些 pieces 组装为 global cohomology。这里的集中性和有限性属于 geometric Satake 外部输入；分解公式是 weight functor construction 的输出。$\square$

**例 I.5.** 对 $G=GL_1$，$\operatorname{Gr}_G\simeq\mathbb Z$，每个 $G(\mathcal O)$-orbit 是一个点。Satake category 等价于 $X_\ast(G)$-graded finite-dimensional vector spaces。卷积由整数加法给出，因此 Tannakian group 的 character lattice 是 $X_\ast(G)$，得到
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

## 本章小结

本附录把 geometric Satake 的证明需求拆为可审查清单：Tannakian 输入、weight functor 分解、$GL_1$ 计算和 root datum 识别。正式使用 Satake 等价时必须同时说明 fiber functor 与 dual group 识别。
