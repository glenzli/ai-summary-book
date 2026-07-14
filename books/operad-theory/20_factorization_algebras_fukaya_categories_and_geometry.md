# 第二十章：Factorization algebra、Fukaya categories 与几何应用

两个互不相交的小圆盘嵌入一个大圆盘，会给局部对象之间的张量乘法；一族彼此交叠的开集覆盖大开集，则要求局部数据通过 descent 重构整体。这两种局部到整体机制不能混为同一个 colimit：前者是 prefactorization algebra 的多输入结构，后者是底层一元 precosheaf 对 Weiss cover 的余层条件。Factorization homology 又把 disk algebra 沿流形作 infinity-categorical colimit。辛几何中的 Fukaya category 提供另一类树形复合，但其定义还依赖伪全纯曲线的紧性、横截性和定向。本章先把这些接口逐项定型，再只在假设明确时连接 $E_n$-代数、Hochschild 同调与几何 gluing。

## 20.1 Disk categories

设 $\mathcal C^\otimes$ 是 presentable symmetric monoidal infinity-category，并且 tensor product separately preserves colimits。

**定义 20.1.** 本节固定 framing。令 $\mathbf{Disk}^{fr}_n$ 为 framed $n$-disks 的 symmetric monoidal infinity-category：

1. 对象是有限个 $\mathbb R^n$ 的不交并
   $$
   \coprod_{i=1}^r\mathbb R^n,\qquad r\ge0;
   $$
2. morphism spaces 是与所选 framings 相容的 embeddings 及其高阶同伦；
3. symmetric monoidal structure 由 disjoint union 给出。

若 $M$ 是 framed $n$-维拓扑或光滑流形，定义 overcategory
$$
\mathbf{Disk}^{fr}_{n/M}=\mathbf{Disk}^{fr}_n\times_{\mathbf{Mfld}^{fr}_n}\mathbf{Mfld}^{fr}_{n/M}.
$$
其对象可理解为带嵌入 $U\hookrightarrow M$ 的有限 disk 并。

若改用 oriented、unoriented 或一般 tangential structure $\xi:B\to BO(n)$，必须把上式全部替换为 $\mathbf{Disk}^{\xi}_n$ 与 $\mathbf{Mfld}^{\xi}_n$；同一个裸 $E_n$-algebra 不自动提供这些扩张数据。

**说明 20.2.** $\mathbf{Disk}^{fr}_{n/M}$ 不是 ordinary poset of opens；morphisms 保留 embeddings 的同伦信息。若只取开集包含关系，会丢失同伦相干自同构。

## 20.2 Factorization algebras

**定义 20.3.** 令 $\mathbf{Open}^{\otimes}(M)$ 为如下对称多范畴：颜色是 $M$ 的开子集；从 $(U_1,\ldots,U_r)$ 到 $V$ 有唯一 multimorphism，当且仅当 $U_i$ 两两不交且 $\bigcup_iU_i\subset V$。$M$ 上取值于 $\mathcal C^\otimes$ 的 prefactorization algebra 是一个对称 multifunctor
$$
\mathcal F:\mathbf{Open}^{\otimes}(M)\longrightarrow\mathcal C^\otimes.
$$
展开后，它给出：

1. 对每个开集 $U\subset M$，一个对象 $\mathcal F(U)\in\mathcal C$；
2. 对任意两两不交开集 $U_1,\ldots,U_r\subset V$，一条乘法结构映射
   $$
   m_{U_1,\ldots,U_r;V}:
   \mathcal F(U_1)\otimes\cdots\otimes\mathcal F(U_r)
   \longrightarrow\mathcal F(V);
   $$
3. 对 $r=0$，给出单位映射 $\mathbb 1_\mathcal C\to\mathcal F(V)$；
4. 对嵌套的不交开集配置，上述映射满足结合律、单位律和对称群等变性，或在 infinity-categorical 模型中满足相应的全部相干条件。

取 $r=1$ 得到包含 $U\subset V$ 所诱导的 unary map $\mathcal F(U)\to\mathcal F(V)$；因此 $\mathcal F$ 有一个底层协变函子
$$
\mathcal F_{\mathrm{un}}:\mathbf{Open}(M)\longrightarrow\mathcal C,
$$
称为 underlying precosheaf。

**定义 20.4（Weiss cover、Čech 索引与 descent）.** 设 $\varnothing\ne V\subset M$ 为开集，$I$ 为 $\mathcal U$-小集合。一族开集 $\mathcal U=\{U_i\subset V\}_{i\in I}$ 称为 $V$ 的 Weiss cover，若每个非空有限子集 $S\subset V$ 都包含在某个 $U_i$ 中。取单点集可见 $\bigcup_iU_i=V$；普通开覆盖未必是 Weiss cover。

定义 Čech 索引范畴 $\mathsf{Cech}(\mathcal U)$。它的对象是二元组
$$
([q],\mathbf i),\qquad
q\ge0,\quad
\mathbf i=(i_0,\ldots,i_q)\in I^{q+1},\quad
U_{\mathbf i}:=\bigcap_{a=0}^qU_{i_a}\ne\varnothing.
$$
从 $([q],\mathbf i)$ 到 $([p],\mathbf j)$ 的态射是一条保序映射
$$
\theta:[p]\longrightarrow[q]
$$
使 $j_a=i_{\theta(a)}$ 对所有 $0\le a\le p$ 成立。此时
$U_{\mathbf i}\subset U_{\mathbf j}$，故底层 precosheaf 的一元结构映射给出函子
$$
D_{\mathcal U}:\mathsf{Cech}(\mathcal U)\longrightarrow\mathcal C,
\qquad
([q],\mathbf i)\longmapsto\mathcal F(U_{\mathbf i}).
$$
这个索引保留重复指标及不同的保序映射。更具体地，$\mathsf{Cech}(\mathcal U)$ 是 simplicial index set $I^{\mathcal U}_\bullet:\Delta^{\operatorname{op}}\to\mathbf{Set}$
$$
I^{\mathcal U}_q=
\{\mathbf i\in I^{q+1}:U_{\mathbf i}\ne\varnothing\}
$$
的 Grothendieck construction；把每个指标 $\mathbf i$ 送到 $\mathcal F(U_{\mathbf i})$，所得 simplicial Čech object 为
$$
\check C_q(\mathcal U;\mathcal F)
=
\coprod_{\substack{\mathbf i\in I^{q+1}\\U_{\mathbf i}\ne\varnothing}}
\mathcal F(U_{\mathbf i})
$$
删除或重复指标所诱导的面映射与退化映射都来自相应开集包含。

各包含 $U_{\mathbf i}\subset V$ 构成 cocone。称 $\mathcal F$ 满足关于 $\mathcal U$ 的 Weiss descent，若典范映射
$$
\operatorname*{colim}_{\mathsf{Cech}(\mathcal U)}D_{\mathcal U}
\simeq
\left|\check C_\bullet(\mathcal U;\mathcal F)\right|
\longrightarrow
\mathcal F(V)
$$
是 $\mathcal C$ 中的 equivalence。这里的 colimit 是 infinity-categorical colimit。Factorization algebra 是对每个非空开集及其每个 Weiss cover 都满足该条件的 prefactorization algebra。本书把空开集的值与定义 20.3 的 nullary 单位作为额外约定；若采用 $\mathcal F(\varnothing)$ 为初始对象的 reduced 口径，也可把空交集加入 Čech 图而不改变 colimit。

**说明 20.4.1（乘法不是 descent 映射）.** Weiss descent 的图只使用 $\mathcal F_{\mathrm{un}}$ 的一元包含映射，不含张量积。反之，定义 20.3 的
$$
\mathcal F(U_1)\otimes\cdots\otimes\mathcal F(U_r)\to\mathcal F(V)
$$
要求 $U_i$ 两两不交，是 prefactorization 乘法。若 $V=V_1\amalg V_2$ 且两部分均非空，普通覆盖 $\{V_1,V_2\}$ 不是 Weiss cover，因为分别取一点得到的二点集不包含在任一覆盖元中；所以不能用 descent 把这条乘法误写成 Čech colimit 映射。定义 20.4 本身也不额外假设
$$
\mathcal F(V_1)\otimes\mathcal F(V_2)\longrightarrow
\mathcal F(V_1\amalg V_2)
$$
必为 equivalence；这种性质另称 multiplicativity。

**定义 20.5.** Factorization algebra $\mathcal F$ 称为 locally constant，若对任意 disks $D\subset D'\subset M$，包含诱导映射
$$
\mathcal F(D)\to\mathcal F(D')
$$
是 equivalence。

称 $\mathcal F$ 为 multiplicative，若对任意有限族两两不交的非空开集 $U_1,\ldots,U_r$，$r\ge1$，prefactorization 结构映射
$$
\mathcal F(U_1)\otimes\cdots\otimes\mathcal F(U_r)
\longrightarrow
\mathcal F(U_1\amalg\cdots\amalg U_r)
$$
是 equivalence。这里要求的是一条已经存在的乘法映射成为 equivalence；它既不是该映射的定义，也不是 Weiss descent 映射。Unital 版本还要求这些等价与定义 20.3 的 nullary 单位相容。

**例 20.5.1（直线上的有序乘法）.** 设 $A$ 是 $\mathbf{Ch}_k$ 中的含单位结合 dg 代数。在 $\mathbb R$ 中有限个开区间的不交并所成的 factorizing basis 上，令
$$
\mathcal F_A(I_1\amalg\cdots\amalg I_r)=A^{\otimes r}.
$$
若干源区间落入同一个目标区间时，就按直线从左到右的次序在 $A$ 中相乘；没有源区间落入某个目标分量时插入单位。特别地，对
$$
I_1<I_2<J
$$
有
$$
m_{I_1,I_2;J}:A\otimes A\longrightarrow A,
\qquad a\otimes b\longmapsto ab.
$$
三个区间的两种嵌套结构映射分别给出 $(ab)c$ 与 $a(bc)$，所以 prefactorization 结合律在此恰为 $A$ 的结合律。所有单区间包含诱导恒等 quasi-isomorphism，故该 basis 数据局部常值；不交并的定义等式还表明它在该 basis 上 multiplicative。

把这份 basis 数据延拓到全部开集并证明 Weiss descent，需要同伦左 Kan extension 或等价的 cosheafification 定理；这是外部比较 20.6 的构造部分。需要注意，上式 $A\otimes A\to A$ 来自两个不交区间的乘法，而一个 Weiss cover 的 descent 图由交集及一元包含组成，两种映射的类型已经不同。

**外部边界 20.6（locally constant comparison）.** 在 Costello--Gwilliam/Lurie 型的适当模型中，$\mathbb R^n$ 上取值于 $\mathcal C$ 的 locally constant multiplicative factorization algebras 与 $E_n$-algebras 之间存在 equivalence of infinity-categories：
$$
\operatorname{Fact}^{lc,\otimes}_{\mathbb R^n}(\mathcal C)\simeq
\operatorname{Alg}_{E_n}(\mathcal C).
$$

一个 locally constant multiplicative factorization algebra 的 disk 值给出对象 $A=\mathcal F(D)$；多个小 disk 嵌入大 disk 的结构映射给出 little-disks 运算，而 multiplicativity 把有限 disk 并的值识别为各分量值的张量积。反向构造使用沿 disk embeddings 的同伦左 Kan extension。完整证明需要固定 factorization-algebra 模型、Weiss descent、multiplicativity 与 isotopy invariance；这里只把定理 20.6 作为具有这些假设的外部比较，不用 manifolds 上同调理论的分类结果替代这条局部等价。

## 20.3 Factorization homology

**定义 20.7.** 设 $A$ 是 $E_n$-algebra in $\mathcal C$，$M$ 是 $n$-manifold。$M$ 上以 $A$ 为系数的 factorization homology 定义为 colimit
$$
\int_M A
=
\operatorname{colim}_{(U\hookrightarrow M)\in\mathbf{Disk}^{fr}_{n/M}} A(U),
$$
其中 $A$ 被视为 symmetric monoidal functor
$$
\mathbf{Disk}^{fr}_n\to\mathcal C,
$$
且 $M$ 为 framed $n$-manifold。

若 $U\simeq\coprod_{i=1}^r\mathbb R^n$，则
$$
A(U)=A^{\otimes r}.
$$

**命题 20.8.** 对 $M=\mathbb R^n$，有 canonical equivalence
$$
\int_{\mathbb R^n}A\simeq A.
$$

**证明.** 因为标准 framed $\mathbb R^n$ 本身是 $\mathbf{Disk}^{fr}_n$ 的对象，
$$
\mathbf{Disk}^{fr}_{n/\mathbb R^n}
\simeq
(\mathbf{Disk}^{fr}_n)_{/\mathbb R^n}.
$$
任意 infinity-category 的 slice $\mathcal D_{/d}$ 都以 $\operatorname{id}_d$ 为 final object；这里该对象正是恒等嵌入 $\mathbb R^n\hookrightarrow\mathbb R^n$。因此
$$
\int_{\mathbb R^n}A\simeq A(\mathbb R^n)=A.
$$
该证明是 slice 的形式性质，不需要额外声称所有 embedding spaces 可缩。$\square$

**外部输入定理 20.9（excision；AF-1）.** 若 $M$ 沿 collar 分解为
$$
M=M_-\cup_{N\times\mathbb R}M_+,
$$
其中 $N$ 是 $(n-1)$-manifold，则
$$
\int_M A
\simeq
\int_{M_-}A
\otimes_{\int_{N\times\mathbb R}A}
\int_{M_+}A.
$$
本书引用 Ayala--Francis, arXiv:1206.5522v6, Lemma 3.18 作为该 topological manifolds 版本的外部来源；分层或 Fukaya 版本需另行定位。

**说明 20.10.** Excision 是 factorization homology 的核心计算定理。它是 ordinary homology 的 Mayer-Vietoris 性质在 $E_n$-algebra 系数下的非交换版本。

**例 20.11.** 若 $n=1$ 且 $A$ 是 associative algebra，则
$$
\int_{S^1}A
$$
与 Hochschild homology $HH_\*(A)$ 对应。这是 factorization homology 与第十一、十二章 Hochschild 理论的连接。

该例的完整链级识别依赖 cyclic bar construction 和 $E_1$-algebra 的 factorization homology 计算，作为外部输入；本书引用 Ayala--Francis, arXiv:1206.5522v6, Theorem 3.19。

**说明 20.11.1.** 外部输入定理 N.18 和说明 N.19 给出本例的严格使用边界：圆周计算应写成
$$
\int_{S^1}A\simeq A\otimes^{\mathbf L}_{A\otimes A^{op}}A
$$
或等价的 cyclic bar construction。若没有指定 derived relative tensor product、边界版本和链级模型，不能把该式当作普通张量积公式。

## 20.4 Dunn additivity 与迭代代数

**外部输入定理 20.12（Dunn additivity；DUNN-1）.** 对 $m,n\ge0$，Lurie *Higher Algebra* Theorem 5.1.2.2 断言 Construction 5.1.2.1 的典范 bifunctor
$$
E_m^\otimes\times E_n^\otimes\longrightarrow E_{m+n}^\otimes
$$
把 $E_{m+n}^\otimes$ 展示为 infinity-operads 的 tensor product，即
$$
E_m^\otimes\otimes E_n^\otimes\simeq E_{m+n}^\otimes.
$$
因此，对任意 symmetric monoidal infinity-category $\mathcal C$，tensor product 的泛性质给出
$$
\operatorname{Alg}_{E_{m+n}}(\mathcal C)
\simeq
\operatorname{Alg}_{E_m}\big(\operatorname{Alg}_{E_n}(\mathcal C)\big).
$$
该输入是 infinity-operadic 定理；它不声称任意 strict topological operad 模型的 Boardman--Vogt tensor product 在无 cofibrancy 假设下即有同一结论。

**说明 20.13.** 该定理说明 $E_{m+n}$-algebra 可看作 $E_m$-algebra object in $E_n$-algebras。它是许多“higher center”和 iterated Hochschild constructions 的 operadic 根源。

**命题 20.14.** 若 $A$ 是 $E_{m+n}$-algebra，则 $A$ canonically determines an $E_m$-algebra object in $\operatorname{Alg}_{E_n}(\mathcal C)$。

**证明.** 由外部输入定理 20.12，$E_{m+n}$-algebra 的结构等价于 $E_m\otimes E_n$-algebra 的结构。Tensor product of infinity-operads 的 universal property 把后者识别为 $E_m$-algebra object in $E_n$-algebras。$\square$

## 20.5 Fukaya categories as $A_\infty$-categories

**定义 20.15.** $A_\infty$-category $\mathcal A$ 由以下数据组成：

1. 对象类 $\operatorname{Ob}(\mathcal A)$；
2. 对每对对象 $X,Y$，给出链复形 $\operatorname{Hom}_\mathcal A(X,Y)$；
3. 对每个 $r\ge1$，给出次数 $r-2$ 的 composition maps
   $$
   m_r:\operatorname{Hom}(X_{r-1},X_r)\otimes\cdots\otimes
   \operatorname{Hom}(X_0,X_1)
   \to
   \operatorname{Hom}(X_0,X_r);
   $$
4. maps $m_r$ 满足 $A_\infty$ relations。

**定义 20.16.** Symplectic manifold $(X,\omega)$ 的 Fukaya category $\mathcal F(X)$ 是以合适 Lagrangian submanifolds 为对象、Floer complexes 为 morphism complexes、holomorphic polygons counts 为 $A_\infty$ compositions 的 $A_\infty$-category。

**警告 20.17.** 定义 20.16 是结构性描述，不是完整构造。完整 Fukaya category 需要选择 brane structures、gradings、spin structures、Novikov coefficients、transversality theory、compactness、bounding cochains 或 wrapped conditions。不同几何情形有不同模型。

**外部边界 20.18（Fukaya $A_\infty$ 构造）.** 在一个已经固定并验证 compactness、transversality、orientation 与 brane data 的 symplectic setting 中，holomorphic polygon counts 可定义 $A_\infty$-category，并且一维模空间的边界退化给出 $A_\infty$ relations。本书未固定单一几何模型，故该项只作接口，不能作为无条件定理调用。

**说明 20.19.** $A_\infty$ relations 的来源是 $1$-维 compactified moduli spaces 的边界。边界 strata 对应把一个 polygon 分裂成两个 polygons；代数上正是
$$
\sum m(\ldots,m(\ldots),\ldots)=0.
$$
这就是 operad 的 associahedra 与 Floer theory 的几何连接。

## 20.6 Operadic structures on Fukaya categories

Fukaya categories 不只形成单个 $A_\infty$-category。在不同几何操作下，它们还带有更高 operadic 结构。

**例 20.20（pair-of-pants product）.** 在某些 wrapped 或 exact 设置中，pair-of-pants 型曲面给出 Floer theory 上的乘法、coproduct 或 module operations。这些 operations 的 gluing 对应曲面 moduli 的 operadic composition。

**例 20.21（Swiss-cheese 型结构）.** 同时含 closed strings 与 open strings 的理论常由 Swiss-cheese operad 或其变体组织。Closed sector 通常带 $E_2$ 或 BV 型结构，open sector 带 $A_\infty$ 型结构，二者之间有兼容 action。

**研究边界 20.22（Fukaya 的高阶 operadic 结构）.** 在特定几何假设下，Fukaya categories、wrapped Fukaya categories 或其 Hochschild invariants 可能组织为由 surfaces、disks、stratified spaces 或 higher operads 控制的代数对象。

这不是单一通用结论。不同版本依赖不同的 compactness、gluing、orientation 和 transversality 定理。本书只记录 operadic 组织方式；具体几何定理必须在相应模型中单独引用。

## 20.7 Factorization homology 与 Fukaya 理论的接口

Factorization homology 把 $E_n$-algebra 沿 $n$-manifold 积分。Fukaya 理论把 symplectic 或 Liouville geometry 赋给 $A_\infty$-categories。二者的交汇点包括：

1. factorization homology of categories；
2. topological Fukaya categories；
3. wrapped Fukaya categories as cosheaves on skeleta；
4. Hochschild invariants and centers of Fukaya categories；
5. extended topological field theories。

**研究边界 20.23（Fukaya gluing 接口）.** 在若干已验证设置中，Fukaya 型范畴可由局部模型通过 cosheaf/factorization-homology 型 gluing 得到；其 Hochschild 或 center 型不变量可能由 factorization homology 计算。没有指定几何类别、系数和 gluing theorem 时，这只是一组研究方向，不能作为一般 Fukaya 范畴的定理使用。

**说明 20.24.** 这类定理是当前研究活跃区域。除非指定具体几何类别、系数、局部模型和 gluing 定理，本书不把它作为全局定理使用。

**说明 20.25.** 本章的计算性补充见定义 N.3、外部输入定理 N.15、外部输入定理 N.18 和研究边界 N.30；常见错误命题和不可混用约定见错误命题 O.23--正确边界 O.28。特别地，factorization homology 不等于普通同调，Fukaya category 的构造不由 operad 公理单独推出。

## 20.8 两种局部到整体机制的边界

Prefactorization 乘法沿不交嵌入合并若干局部输入，Weiss descent 则沿交叠的 Čech 图重构一个开集的值；factorization algebra 同时携带二者，但二者的索引和映射类型不同。Locally constant 条件使 disk 值只依赖局部同胚型，在指定模型与外部比较定理下由此得到 $E_n$-代数。Factorization homology 是另一项构造，它在 $\mathbf{Disk}^{fr}_{n/M}$ 上取 colimit，并以 excision 而非普通开覆盖的 Mayer--Vietoris 公式完成计算。Fukaya 型结构可以提供 $A_\infty$ 运算和某些 gluing 接口，但伪全纯曲线的分析与 sectorial descent 仍是独立几何输入，不能由 operad 公理或 Weiss descent 的形式定义推出。

## 练习

**练习 20.1.** 写出 prefactorization algebra 的二重嵌套复合公理。

**练习 20.2.** 说明 locally constant factorization algebra on $\mathbb R$ 如何给出 associative algebra。

**练习 20.3.** 对 associative algebra $A$，解释 $\int_{S^1}A$ 与 cyclic bar construction 的关系。

**练习 20.4.** 写出 $A_\infty$-category 中 $m_1,m_2,m_3$ 参与的一条低阶关系。

**练习 20.5.** 说明为什么 Fukaya category 的构造不能只用形式 operad 公理完成，还需要分析定理。
