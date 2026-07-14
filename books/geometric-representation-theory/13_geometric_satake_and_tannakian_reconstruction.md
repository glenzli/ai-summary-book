# 第十三章：Geometric Satake 等价与 Tannakian reconstruction

第十二章已经得到一个可结合的卷积，但单凭 proper pushforward，两个 perverse sheaves 的卷积未必仍 perverse；即使存在 fiber functor，形式 Tannaka 也只产生未知的 affine group scheme $H$，不会自动把它识别为 Langlands dual group。Geometric Satake 的实质正分布在这两个缺口之间：semismall 几何保证 t-exactness，Beilinson--Drinfeld fusion 提供对称约束，semi-infinite orbits 把全上同调分解为 coweight spaces，最后由 MV 几何识别 dual root datum。$GL_1$ 可完整重建 $\mathbb G_m$，而 $GL_2$ 的 minuscule 二重卷积将逐项对应 $V^{\otimes2}=\operatorname{Sym}^2V\oplus\det V$。

**约定 13.0.** 沿用约定 12.0：$G$ 是连通 reductive complex algebraic group，$E$ 是代数闭 characteristic-zero field，$\operatorname{Gr}_G$ 在形成 Betti sheaves 前取 reduction，所有对象有 finite-dimensional support。本章的 $\operatorname{Rep}_E(H)$ 只含 finite-dimensional algebraic $H$-representations。

## 13.1 Satake category、fusion 和 fiber functor

**定义 13.1.** Satake category 是
$$
\operatorname{Sat}_G
=\operatorname{Perv}_{L^+G,\mathrm{fs}}
(\operatorname{Gr}_G,E),
$$
其 convolution、associator 和 unit $\mathbf1=E_e$ 分别由定义 12.13、命题 12.14 和命题 12.15 给出。

**外部输入定理 13.2（t-exactness、rigidity 与 fusion）.** 对任意
$\mathcal F,\mathcal G\in\operatorname{Sat}_G$：

1. $\mathcal F\star\mathcal G$ 仍是 perverse，且 $\star$ 在两个变量上都是 exact；
2. inversion on $LG$ 与 Verdier duality 构造每个对象的 left/right dual；
3. Beilinson--Drinfeld Grassmannian 上的 fusion construction 给出 commutativity constraint；经过与 cohomological parity 相容的标准修正后，
   $(\operatorname{Sat}_G,\star,\mathbf1)$ 是 rigid symmetric monoidal $E$-linear abelian category。

第 1 项使用 convolution map 的 stratified-semismall dimension estimate，不是 properness 的形式推论；第 3 项不来自 $LG$ 的交换性，因为一般 $LG$ 非交换。这里采用 Mirkovic--Vilonen Proposition 4.2、Lemma 4.4、Proposition 4.6、§5 及 §6 的 parity modification；fusion construction 不在本书重证。

**定义 13.3.** 定义 total global cohomology functor
$$
\omega:\operatorname{Sat}_G\longrightarrow\mathbf{Vect}^{\mathrm{fd}}_E,
\qquad
\omega(\mathcal F)
=\bigoplus_{i\in\mathbb Z}\mathbb H^i
(\operatorname{Gr}_G,\mathcal F).
$$
因为 $\mathcal F$ 支撑在 projective finite-dimensional Schubert union 上，右侧只有有限多个非零 finite-dimensional summands。Tannakian fiber functor 忘掉 cohomological grading；其 parity 已用于定理 13.2 的 commutativity correction。

**外部输入定理 13.4（fiber functor）.** $\omega$ 是 exact、faithful、symmetric monoidal functor，并有与 associator 和 commutativity constraints 相容的自然同构
$$
\omega(\mathcal F\star\mathcal G)
\simeq\omega(\mathcal F)\otimes_E\omega(\mathcal G),
\qquad
\omega(\mathbf1)\simeq E.
$$
这一结论见 Mirkovic--Vilonen Theorem 3.6、Corollary 3.7 与 Proposition 6.3。Exactness 和 faithfulness 依赖 semi-infinite orbit cohomology，不是 ordinary global sections 的一般性质。

## 13.2 Langlands dual group 与等价的精确版本

**定义 13.5.** 若 $G$ 的 based root datum 为
$$
(X^\ast(T),\Phi,\Delta,
X_\ast(T),\Phi^\vee,\Delta^\vee),
$$
则 $G^\vee$ 表示 $E$ 上 split connected reductive group，其 based root datum 为
$$
(X_\ast(T),\Phi^\vee,\Delta^\vee,
X^\ast(T),\Phi,\Delta).
$$
因此 $X^\ast(T^\vee)=X_\ast(T)$，且 $G^\vee$ 的 dominant weights 正是 $G$ 的 dominant coweights。这里的 $G^\vee$ 是 algebraic group over $E$，不是 complex Lie group、Langlands $L$-group 或 derived group。

**外部输入定理 13.6（geometric Satake）.** 存在 $E$-linear symmetric monoidal equivalence
$$
\mathsf{Sat}:\operatorname{Sat}_G
\xrightarrow{\ \sim\ }
\operatorname{Rep}_E(G^\vee)
$$
以及 fiber functor 的 monoidal identification
$$
\omega\simeq
\operatorname{Forget}\circ\mathsf{Sat}.
$$
对每个 $\lambda\in X_\ast(T)^+$，
$$
\mathsf{Sat}(\operatorname{IC}_\lambda)
\simeq V_\lambda,
$$
其中 $V_\lambda$ 是最高权 $\lambda$ 的 irreducible $G^\vee$-representation。这里采用 Mirkovic--Vilonen 主等价 (1.1) 与 Theorem 12.1 的 complex Betti、field、characteristic-zero specialization。

等价把 Schubert IC 层的 orbit 标号变成最高权标号，因此 semisimplicity 不再是 affine Grassmannian 的单独几何猜测，而是特征零 reductive group 表示范畴的直接结果。

**命题 13.7.** 接受定理 13.6 后，$\operatorname{Sat}_G$ 是 semisimple，且 simple objects 的同构类恰为
$$
\{\operatorname{IC}_\lambda\mid
\lambda\in X_\ast(T)^+\}.
$$

**证明.** characteristic-zero reductive group $G^\vee$ 的 finite-dimensional representation category semisimple，其 irreducibles 由 dominant weights 唯一参数化。定义 13.5 把这些 weights 识别为 $X_\ast(T)^+$；定理 13.6 的 equivalence 保持 simple objects、finite direct sums 和 isomorphism classes，并把 $\operatorname{IC}_\lambda$ 送到 $V_\lambda$。因此得到两项结论。$\square$

## 13.3 Neutral Tannaka 能推出的范围

**外部输入定理 13.8（neutral Tannakian reconstruction）.** 令 $\mathcal C$ 为 essentially small、rigid、symmetric monoidal、$E$-linear abelian category，满足
$$
\operatorname{End}_{\mathcal C}(\mathbf1)=E,
$$
且 $\otimes$ 对每个变量 exact。若有 exact faithful symmetric monoidal functor
$$
\omega:\mathcal C\to\mathbf{Vect}^{\mathrm{fd}}_E,
$$
则 functor of tensor automorphisms
$$
H:=\underline{\operatorname{Aut}}^\otimes(\omega)
$$
由 affine group scheme over $E$ 表示，并存在与 fiber functors 相容的 symmetric monoidal equivalence
$$
\mathcal C\simeq\operatorname{Rep}_E(H).
$$
这里采用 Saavedra Rivano、Deligne--Milne 的 neutral Tannakian theorem，并把它作为外部范畴论输入。

**机制边界 13.9.** 把定理 13.8 用于
$(\operatorname{Sat}_G,\omega)$ 只构造
$$
H=\underline{\operatorname{Aut}}^\otimes(\omega).
$$
形式 Tannaka 本身不证明 $H$ finite type、connected、reductive，也不识别其 roots 与 coroots。Geometric Satake 还必须用 weight functors、MV cycles、rank-one slices 和 fusion commutativity 证明
$$
H\simeq G^\vee.
$$
因此不能把“存在 fiber functor”直接改写为“dual group 已识别”。

## 13.4 Weight functors 与 torus morphism

**定义 13.10.** 令 $N$ 是 $B$ 的 unipotent radical。对 $\mu\in X_\ast(T)$，定义 semi-infinite orbit
$$
S_\mu=N(\mathscr K)\cdot z^\mu L^+G/L^+G
\subset\operatorname{Gr}_G.
$$
对 finite-support $\mathcal F$，compactly supported cohomology 通过
$S_\mu\cap\operatorname{supp}\mathcal F$ 在 finite-dimensional stage 中定义，并置
$$
F_\mu(\mathcal F)
:=\mathbb H_c^{\langle2\rho,\mu\rangle}
(S_\mu,\mathcal F)\in\mathbf{Vect}^{\mathrm{fd}}_E.
$$
次数 $\langle2\rho,\mu\rangle$ 可以为负；它是 cohomological degree，不是 $S_\mu$ 的有限维 dimension。

**外部输入定理 13.11（weight decomposition）.** 对每个
$\mathcal F\in\operatorname{Sat}_G$：

1. $\mathbb H_c^j(S_\mu,\mathcal F)=0$，除非
   $j=\langle2\rho,\mu\rangle$；
2. 只有有限多个 $F_\mu(\mathcal F)$ 非零，且存在 natural decomposition
   $$
   \omega(\mathcal F)
   \simeq\bigoplus_{\mu\in X_\ast(T)}F_\mu(\mathcal F);
   $$
3. 该分解与 convolution 相容：
   $$
   F_\nu(\mathcal F\star\mathcal G)
   \simeq
   \bigoplus_{\mu+\eta=\nu}
   F_\mu(\mathcal F)\otimes_EF_\eta(\mathcal G).
   $$

集中性、有限性和 tensor compatibility 采用 Mirkovic--Vilonen Theorem 3.6 与 Proposition 6.4，均为外部输入。

**命题 13.12.** 令 $T^\vee$ 为 split torus with
$X^\ast(T^\vee)=X_\ast(T)$，并令
$H=\underline{\operatorname{Aut}}^\otimes(\omega)$。定理 13.11 的 graded tensor decomposition 自然定义一个 group-scheme morphism
$$
\iota:T^\vee\longrightarrow H.
$$

**证明.** 对任意 commutative $E$-algebra $R$ 和
$t\in T^\vee(R)=\operatorname{Hom}(X_\ast(T),R^\times)$，在
$$
\omega(\mathcal F)_R
=\bigoplus_\mu F_\mu(\mathcal F)\otimes_E R
$$
上定义 $t_\mathcal F$：它在 $\mu$-summand 上乘以标量 $\mu(t)\in R^\times$。Weight decomposition 的 naturality 说明 $t_\mathcal F$ 对 $\mathcal F$ functorial；定理 13.11(3) 和
$(\mu+\eta)(t)=\mu(t)\eta(t)$ 说明
$$
t_{\mathcal F\star\mathcal G}
=t_\mathcal F\otimes t_\mathcal G,
$$
且 $t_\mathbf1=1$。故 $(t_\mathcal F)_\mathcal F$ 是
$\omega_R$ 的 tensor automorphism，给出
$T^\vee(R)\to H(R)$。该构造与 $R$ 的基变换相容，因而由 Yoneda lemma 得到 group-scheme morphism $\iota$。$\square$

**边界说明 13.12.1.** 命题 13.12 只构造 $\iota$；它没有证明 $\iota$ 是 closed immersion，也没有证明其像是 maximal torus。更不能仅凭 $X^\ast(T^\vee)=X_\ast(T)$ 恢复 $H$ 的 roots。把 $\iota(T^\vee)$ 识别为 maximal torus，并证明 $H$ 的 root datum 是 dual root datum，属于定理 13.6 的外部输入部分。

## 13.5 完整的 torus 检验：$GL_1$

一般的 root-datum 识别依赖 MV cycles；对 torus 没有根，所有 Schubert strata 都是点，Tannaka group 可以不借助该大型输入直接算出。这也检验了卷积、rigid dual 与 character grading 的每一项。

**例 13.13.** 对 $G=GL_1$，命题 12.5 的 reduced Betti Grassmannian 是由 $\mathbb Z$ 标号的离散 points，且
$$
\operatorname{Sat}_{GL_1}
\simeq\mathbf{Vect}^{(\mathbb Z)}_E,
$$
其中右侧对象是 finite-support $\mathbb Z$-graded finite-dimensional vector spaces。事实上，每个 point 上的 perverse sheaf 是 degree-$0$ finite-dimensional vector space；$L^+GL_1$ 在该 point 上的 stabilizer connected，且 equivariance 在 finite jet quotient 上定义，所以推论 A.9 排除额外的不可约 monodromy。Finite support 随即给出上述 grading。命题 12.16 给出 grading convolution
$$
(V\star W)_r
=\bigoplus_{a+b=r}V_a\otimes_EW_b.
$$

**命题 13.14.** Functor
$$
\Phi:\mathbf{Vect}^{(\mathbb Z)}_E
\longrightarrow\operatorname{Rep}_E(\mathbb G_m),
\qquad
(V_n)_n\longmapsto
\bigoplus_n V_n\otimes_E\chi_n,
$$
其中 $\chi_n(t)=t^n$，是 symmetric monoidal equivalence。

**证明.** 每个 finite-dimensional algebraic $\mathbb G_m$-representation 唯一分解成有限多个 weight spaces，所以 $\Phi$ essentially surjective。对 graded objects $V,W$，
$$
\operatorname{Hom}(V,W)
=\bigoplus_{n\in\mathbb Z}\operatorname{Hom}_E(V_n,W_n),
$$
而 equivariant linear maps 恰保持 $\mathbb G_m$-weights，故 $\Phi$ fully faithful。最后，
$$
\chi_a\otimes\chi_b\simeq\chi_{a+b}
$$
把 tensor-product weight decomposition 与命题 12.16 的 convolution formula 逐项识别；交换约束两侧都是 ordinary flip，因为相关 points 维数为 $0$。因此 $\Phi$ 是 symmetric monoidal equivalence。$\square$

## 13.6 Rank two 检验：$GL_2$ 的 tensor square

**命题 13.15.** 令 $G=GL_2$，并令 $V=E^2$ 为 dual group
$G^\vee=GL_2$ 的 standard representation。定理 13.6 把第十二章的几何分解
$$
\operatorname{IC}_{(1,0)}\star\operatorname{IC}_{(1,0)}
\simeq
\operatorname{IC}_{(2,0)}\oplus\operatorname{IC}_{(1,1)}
$$
逐项送到
$$
V\otimes_E V
\simeq\operatorname{Sym}^2V\oplus\det(V).
$$

**证明.** 定理 13.6 给出
$$
\mathsf{Sat}(\operatorname{IC}_{(1,0)})=V,
$$
并把 convolution 送到 tensor product。最高权 $(2,0)$ 的 irreducible 是
$\operatorname{Sym}^2V$，最高权 $(1,1)$ 的 irreducible 是
$\bigwedge^2V=\det(V)$。由于 $\operatorname{char}E=0$，endomorphisms
$$
\frac{1+\tau}{2},\qquad\frac{1-\tau}{2}
$$
是 $V\otimes V$ 上互补的 $GL_2$-equivariant idempotents，其 images 分别为
$\operatorname{Sym}^2V$ 和 $\bigwedge^2V$。故表示侧确有上述 direct-sum decomposition；推论 12.18 已在几何侧独立算出相同的两个 multiplicity-one summands。$\square$

## 13.7 Classical Satake 与系数边界

**边界说明 13.16.** Classical Satake isomorphism 描述 nonarchimedean local field 上 $p$-adic group 的 spherical Hecke algebra。要从 geometric Satake 得到函数层结论，必须另选 finite-field/$\ell$-adic model、Frobenius structure、sheaf--function dictionary 和 $q^{\langle\rho,\lambda\rangle}$ normalization。本章的 complex Betti category 没有 Frobenius trace，因此不能单独推出 classical Satake formula。

**边界说明 13.17.** Mirkovic--Vilonen 的 theorem 可处理比本章更一般的 coefficient rings，但 modular 或 integral coefficients 下 $\operatorname{Sat}_G$ 不再由 characteristic-zero semisimplicity 描述，推论 12.18 的 decomposition-theorem proof 也不能原样搬用。Mixed-characteristic affine Grassmannian 和 Satake equivalence 是另一套几何模型，必须单独引用 Zhu 等来源；本章不把这些版本合并成一个无条件陈述。

Weight functors 先在 Tannaka group 中构造 torus，MV 几何再识别 roots 与 coroots；两步不可合并。$GL_1$ 直接给出有限支撑分次向量空间与 $\operatorname{Rep}(\mathbb G_m)$ 的对称张量等价，$GL_2$ 则把 closed convolution fiber 的一个重数转成 determinant summand。若把 $L^+G$ 换成 Iwahori subgroup，orbit 标号会从 dominant coweights 细化为 affine Weyl group；下一章研究相应的非对称 affine Hecke category。

## 练习

**练习 13.1.** 在命题 13.12 中逐项验证 $t_{\mathcal F}$ 与 morphisms、unit 和 associator 相容，并说明为何这仍不证明 $\iota$ injective。

**练习 13.2.** 对 $G=SL_2$，列出 dominant coweights，并与 $G^\vee=PGL_2$ 的 dominant weights 对齐；注意不是每个 $SL_2$ highest weight 都出现。

**练习 13.3.** 对 $GL_1$ 证明 rigid dual 把 degree $n$ 送到 degree $-n$，并与 character dual $\chi_n^\vee=\chi_{-n}$ 比较。

**练习 13.4.** 计算 $\dim\operatorname{Sym}^2E^2$ 和 $\dim\det(E^2)$，与命题 12.17 中 open fiber 和 relevant closed fiber 的两个 multiplicities 对照。
