# 第二十三章：研究边界与开放障碍

前二十二章反复出现同一种推进方式：先由几何 correspondence 构造作用，再用分解、Tannaka 或 Grothendieck group 识别表示。然而走到 geometric Langlands、Coulomb branches、modular coefficients 与多种范畴化的交界处，仅有对象名称和相似的 decategorification 已经不够。真正未解决或需要更强外部输入的地方，可以归结为若干稳定的数学障碍：局部作用如何粘成全局等价，Poisson 代数的谱何时具有辛分辨率，特征零分解在模系数下如何改变，以及两个共享同一基或 crystal 的范畴是否真的等价。以下按这些障碍组织边界；每一节先给可验证的最低秩模型，再说明一般问题缺少的结构。

## 23.1 局部 Hecke 作用与全局 Langlands 等价之间的距离

**约定 23.1.** 令 $C$ 为光滑 projective complex curve，$G$ 为 connected reductive complex group。Automorphic side 取第十六章的 $G$-bundle stack $\operatorname{Bun}_G(C)$，spectral side 取 derived stack $\operatorname{LocSys}_{G^\vee}(C)$。本节只比较这两个对象所需的结构，不预设一个未注明 sheaf theory 和 singular-support 条件的范畴等价。

Geometric Satake 已经给出 $\operatorname{Rep}(G^\vee)$ 对 automorphic category 的逐点 Hecke 作用。这个局部 tensor action 是任何 global equivalence 必须保持的数据，却不包含 essential surjectivity，也没有自行构造 spectral category 上的全部对象。

**命题 23.2（$GL_1$ 的交换 Hecke 平移）.** 对 $G=GL_1$ 和 divisor $D$，令
$$
T_D:\operatorname{Pic}(C)\longrightarrow\operatorname{Pic}(C),
\qquad L\longmapsto L(D).
$$
则对任意 divisors $D,D'$ 有规范同构
$$
T_DT_{D'}\simeq T_{D'}T_D\simeq T_{D+D'},
$$
并且这些同构满足三个 divisors 上的 associativity 与 symmetry coherence。

**证明.** Cartier divisors 的加法给出规范线丛同构
$$
(L(D'))(D)=L\otimes\mathcal O_C(D')\otimes\mathcal O_C(D)
\simeq L\otimes\mathcal O_C(D+D').
$$
交换同构来自线丛 tensor product 的 symmetry，结合同构来自其 associator。四个 divisors 上的两种重排都由 symmetric monoidal category of line bundles 的 coherence 给出同一同构，因此满足所述相容性。$\square$

这个 abelian 模型已经含有 Hecke tensor compatibility，但它仍不是一般 geometric Langlands 的证明。非交换群至少引入以下彼此独立的困难：

1. $\operatorname{Bun}_G$ 通常不是 quasi-compact，D-module category 需要选择 ordinary、renormalized 或 compactly generated 模型；
2. $\operatorname{LocSys}_{G^\vee}$ 必须保留 derived deformation complex，并常需施加 nilpotent singular support；
3. 一点上的 Satake action 必须在 Ran space 上 factorize，才可能控制多点修改与全局 gluing；
4. 构造一个保持 Hecke action 的 functor 不等于证明它 fully faithful，更不等于证明 essential surjectivity。

**研究边界 23.3.** Geometric Langlands 的确定陈述必须同时固定两侧范畴、群的假设、曲线的版本和 singular-support 条件。局部 Satake、Kac--Moody localization 与 factorization 是构造 global functor 的输入；它们中的任何一项都不单独推出 global equivalence。

**开放问题 23.4.** 能否为一类足够广的 reductive groups 给出模块化的 local-to-global 证明，使 compact generation、factorization gluing、fully faithfulness 与 essential surjectivity 分成可独立验证的定理，同时在 $GL_1$ 与低秩 Hecke 修改上退化为可计算模型？

## 23.2 从卷积代数的谱到辛分辨率

BFN 构造首先给出一个交换卷积代数 $\mathcal A$。把 $\operatorname{Spec}\mathcal A$ 称为 Coulomb branch，并不意味着它已经是第十九章意义下的 symplectic singularity，更不意味着存在 symplectic resolution。

**定义 23.5.** 对交换 Poisson $\mathbb C$-algebra $A$，从代数到可量子化辛几何通常要依次验证：

1. $A$ finitely generated，使 $X=\operatorname{Spec}A$ 为 finite-type affine scheme；
2. $A$ reduced and normal，使奇点与函数域行为可控；
3. $X_{\mathrm{sm}}$ 上的 Poisson tensor 非退化，给出 symplectic form；
4. 该 symplectic form 满足 symplectic-singularity 的延拓条件；
5. 存在 proper birational symplectic resolution $\widetilde X\to X$；
6. $A_\hbar$ 对 $\mathbb C[\hbar]$ flat，且 $A_\hbar/(\hbar)\simeq A$，从而确为 quantization。

这些条件不是定义上的同义反复。有限生成代数 $\mathbb C[\varepsilon]/(\varepsilon^2)$ 不 reduced；cusp ring $\mathbb C[t^2,t^3]$ reduced 但不 normal；带零 Poisson bracket 的 $\mathbb A^3$ 光滑且 normal，却因维数为奇数而不可能在 smooth locus 上 symplectic。

**例 23.6（$A$-型 Poisson 曲面）.** 对整数 $m\ge1$，令
$$
X_m=\{(x,y,z)\in\mathbb A^3\mid xy=z^m\}.
$$
当 $m=1$ 时，可消去 $z$ 并以 $(x,y)$ 为坐标，故 $X_1\simeq\mathbb A^2$，其全局 symplectic form 可取 $dx\wedge dy$。当 $m\ge2$ 时，Jacobian
$$
(y,x,-mz^{m-1})
$$
只在原点同时为零，所以奇点是孤立点；此时 smooth locus 由 $x\ne0$ 与 $y\ne0$ 两个开集覆盖。在这两个开集上分别定义
$$
\omega_x=\frac{dx\wedge dz}{x},
\qquad
\omega_y=-\frac{dy\wedge dz}{y}.
$$
对关系 $xy=z^m$ 微分并与 $dz$ 作外积，得到
$$
x\,dy\wedge dz+y\,dx\wedge dz=0,
$$
所以 $\omega_x=\omega_y$ 于重叠处。该二形式在每个开集上非退化，因而当 $m\ge2$ 时给出 $X_m^{\mathrm{sm}}$ 的 symplectic form。$m=1$ 时，同一公式在相应开集上都等于前述 $dx\wedge dy$。

**外部输入定理 23.6.1.** 当 $m\ge2$ 时，$X_m$ 是 type $A_{m-1}$ rational double point；其最小分辨率的 exceptional divisor 是 $m-1$ 条 $\mathbb P^1$ 组成的链，交叉矩阵为负的 type $A_{m-1}$ Cartan matrix。这个经典 ADE 分辨率定理不参与前面的 smooth-locus 计算，只用于识别奇点的全局分辨率。

这个曲面模型说明“显式 Poisson 方程”“symplectic smooth locus”与“存在且识别一个 resolution”是三件不同的事。对一般 BFN pair $(G,N)$，还要先把非有限型同调构造与 finite-type spectrum 联系起来。

**研究边界 23.7.** Coulomb branch 与 affine Grassmannian slice、shifted Yangian 或 quiver variety 的同构一旦成立，往往能转移 normality、symplectic leaves 与 resolution 数据；但每个同构都依赖具体 $(G,N)$，不能从卷积代数的定义形式推出。

**开放问题 23.8.** 哪些 BFN inputs 产生 symplectic singularities，哪些承认 projective symplectic resolutions？若 resolution 不存在，quantized category $\mathcal O$、symplectic leaves 与 representation-theoretic actions 应以何种奇异模型取代？

## 23.3 模系数下的分解与 torsion

特征零中，decomposition theorem、semisimplicity 与 IC sheaves 支撑了 KL 基和 Satake 分解。把系数换成特征 $p$ 的域后，这条链可能在 integral stalk/costalk 出现 torsion 的位置断裂；parity sheaves 提供另一组对系数更稳定的对象。

**定义 23.9.** 固定一个分层 $X=\coprod_sX_s$，各 stratum 上只允许有限秩 local systems。若 $\mathcal F\in D^b_c(X,E)$ 的所有 $j_s^*\mathcal F$ 与 $j_s^!\mathcal F$ 只在偶次数有 cohomology，称 $\mathcal F$ even；若只在奇次数有 cohomology，称其 odd。Even object 与 odd object 的直和称为 parity complex。对 stratum $X_s$ 的 indecomposable parity extension $\mathcal E_s$，要求其支撑为 $\overline X_s$，并在 $X_s$ 上限制为 $E_{X_s}[\dim X_s]$，差一个整体 parity shift。

**命题 23.10（$A_1$ 中没有 modular 修正）.** 对 $SL_2/B\simeq\mathbb P^1$ 的 Schubert 分层 $\{pt\}\sqcup\mathbb A^1$，两个 normalized indecomposable parity sheaves 分别为
$$
E_{\{pt\}},
\qquad
E_{\mathbb P^1}[1].
$$
它们也分别是两个 Schubert IC sheaves。因此在 type $A_1$ 中，任意域系数下由 parity sheaves 定义的 $p$-canonical basis 与通常 KL basis 一致。

**证明.** 点上的常值 sheaf 集中在次数 $0$，是 even。$E_{\mathbb P^1}[1]$ 在开 stratum 上限制为 $E_{\mathbb A^1}[1]$，只在奇次数非零；在闭点处
$$
i^*E_{\mathbb P^1}[1]\simeq E[1],
\qquad
i^!E_{\mathbb P^1}[1]\simeq E[-1],
$$
也都只在奇次数非零，故它是 odd parity complex。两个 Schubert closures 都光滑，所以第三章命题 3.17 给出相同的 IC sheaves。相应 Hecke classes 因而没有系数依赖。$\square$

最低秩的平凡性不能外推到高秩奇异 Schubert varieties。Integral intersection forms 的行列式若被 $p$ 整除，decomposition multiplicities 与 parity extensions 会随系数改变，进而产生 $p$-canonical basis。

**研究边界 23.11.** 模表示论的核心障碍不是把 $E$ 的 characteristic 从 $0$ 改成 $p$，而是控制 integral stalks、costalks 与 intersection forms 中的 torsion。统一的结果需要同时解释 parity sheaves、tilting characters、modular category $\mathcal O$ 和 $p$-canonical combinatorics 的适用范围。

**开放问题 23.12.** 能否从 Schubert singularities 的可计算局部模型预测 torsion primes，并把这种预测转化为 modular characters 或 tilting multiplicities 的有效界？

## 23.4 相同的 decategorification 不给出范畴等价

第十七至二十二章出现了多种共享同一 highest-weight representation、canonical basis 或 crystal 的模型。共同的 $K_0$ 是比较的必要条件，却没有记录 extension、grading、monoidal constraint 或 higher morphisms。

**命题 23.13.** 两个有限长度 abelian categories 的 Grothendieck groups 同构，不推出 categories 等价。

**证明.** 令 $\mathcal C=\mathbf{Vect}^{\mathrm{fd}}_E$，令 $\mathcal D$ 为 dual-number algebra $E[\varepsilon]/(\varepsilon^2)$ 的有限维 modules。两者都只有一个 simple object 的同构类，所以
$$
K_0(\mathcal C)\simeq\mathbb Z\simeq K_0(\mathcal D).
$$
但 $\mathcal C$ semisimple，而 $\mathcal D$ 中存在非分裂正合列
$$
0\longrightarrow E\varepsilon
\longrightarrow E[\varepsilon]/(\varepsilon^2)
\longrightarrow E\longrightarrow0.
$$
等价保持正合列是否分裂，因此 $\mathcal C$ 与 $\mathcal D$ 不等价。$\square$

所以，仅仅证明 quiver-variety homology、cyclotomic KLR projectives 与某个 CoHA module 有同构的 Grothendieck group，还不能识别其范畴化。一个真正的比较至少要保留：

1. grading shift 与 $q$-参数；
2. convolution/induction 的 monoidal structure；
3. simple、projective、standard 与 costandard objects 的对应；
4. Verdier、bar 或 graded duality；
5. 若比较 2-representations，还要保留 generators、2-morphisms 和 Kac--Moody relations 的 coherent action。

**研究边界 23.14.** $A_1$ 中，Grassmannian fibers 给出 crystal chain，nilHecke projectives 给出 divided powers，两者由 Nakajima 与 KLR 外部定理映到同一 $\mathfrak{sl}_2$-表示。高秩时，要把这种一致性提升为 monoidal、dg 或 2-categorical equivalence，必须构造明确 functor；“都产生 canonical basis”不能代替它。

**开放问题 23.15.** 对哪些 quiver 或 Coulomb-branch families，可以在 quiver sheaves、KLR modules、CoHA modules 与 quantized category $\mathcal O$ 之间建立保持 generators、gradings 和 dualities 的统一 2-representation？

## 23.5 经典点不足以决定几何模型

许多前沿构造发生在 ind-schemes、derived stacks 或 quotient prestacks 上。只计算代数闭域上的点，会丢失 nilpotent families、derived intersections 与 stabilizer directions；而这些方向恰好控制 tangent complexes、singular support 和卷积中的 excess intersection。

**反例 23.16.** 第十二章的 $GL_1$ affine Grassmannian 在 reduced geometric points 上由 $\mathbb Z$ 标号，但对
$$
R=\mathbb C[\varepsilon]/(\varepsilon^2)
$$
，loop $1+\varepsilon z^{-1}\in R((z))^\times$ 不属于 $R[[z]]^\times$，却在 $R/(\varepsilon)$ 上退化为单位。故 pointwise valuation 不能识别未约化 fpqc quotient。类似地，把 derived fiber product 替换成经典交集会删除 Tor directions，可能改变 CoHA、Steinberg 或 spectral-side singular support。

**研究边界 23.17.** Geometric Langlands 的 spectral stack、BFN 的非有限型空间、CoHA 的短正合列 stack 和 higher Morita theory 需要不同但相容的 categorical models。困难不只在定义对象，还在证明六函子、compact generation、base change 与 monoidal actions 在所选模型中同时存在。

**开放问题 23.18.** 能否建立一套对 ind-geometric convolution、derived singular support 与 factorization 都稳定的比较 formalism，使 Betti、de Rham、mixed 和 modular realizations 之间的变换有明确假设，而不把它们压成同一个无类型的“层范畴”？

这些障碍给出了全书结尾处的稳定边界。局部 $GL_1$ 平移、$A$-型 Poisson 曲面、$SL_2$ parity sheaves 和 dual-number module 都能完整计算；它们分别说明 coherence、辛性、系数依赖和 extension data 中哪些信息必须保留。更一般的研究问题并不是把这些例子换成更大的符号，而是构造能跨过相应障碍的定理，同时在低秩模型上退化到已经验证的公式。

## 练习

**练习 23.1.** 对有效除子 $D=\sum_i n_ix_i$，定义 $T_D(L)=L(D)$。证明 $T_DT_{D'}\simeq T_{D+D'}$，并说明该作用如何由 divisor monoid 延拓到 $\operatorname{Pic}(C)$ 的群作用。

**练习 23.2.** 对 $X_m=\{xy=z^m\}$，验证 $\omega_x$ 与 $\omega_y$ 在重叠处相等，并求 $m=1,2,3$ 时的 singular locus。

**练习 23.3.** 逐项计算 $E_{\mathbb P^1}[1]$ 在开 cell 与闭点上的 stalk/costalk parity，解释为什么 type $A_1$ 看不见 $p$-canonical 修正。

**练习 23.4.** 在命题 23.13 的 dual-number category 中计算 $\operatorname{Ext}^1(E,E)$，并与 $\mathbf{Vect}^{\mathrm{fd}}_E$ 比较。

**练习 23.5.** 任选第十七至二十二章中的两个模型，列出除 $K_0$ 或 crystal 外还必须比较的三个 categorical structures，并说明缺少其中一项会丢失什么信息。
