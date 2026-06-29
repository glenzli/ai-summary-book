# 附录 N：Factorization homology、excision 与几何计算

本附录把第二十章的几何接口改写成可检查的计算规则。核心原则如下：

1. factorization homology 是一个由 disk category 定义的 colimit；
2. 计算能力来自 excision，而不是来自把开覆盖当作普通 Mayer-Vietoris 序列；
3. 进入非平凡几何应用时，必须同时记录切结构、系数范畴、边界条件和外部分析定理。

本附录不替代 Ayala-Francis、Costello-Gwilliam、Lurie 或 Fukaya 理论文献中的证明。凡涉及 isotopy invariance、Weiss descent、collar-gluing、sectorial descent、transversality 或 compactness 的结论均标为外部输入。

## N.1 基本假设

**约定 N.1.** 本附录默认：

1. $\mathcal C^\otimes$ 是 presentable symmetric monoidal infinity-category；
2. tensor product 在每个变量中保持 small colimits；
3. $M$ 是 $\mathcal U$-小 atlas 表示的光滑 $n$-manifold；
4. 除非额外说明，$M$ 带 framing，并使用 framed disk category。

第 4 条不是装饰性假设。Framed $n$-disks 的自同构同伦类型与 $O(n)$-数据有关；若只写 $\mathbf{Disk}_n$ 而不说明切结构，就无法判定系数对象应为 framed $E_n$-algebra、oriented $E_n$-algebra 还是带一般 tangential structure 的代数。

**定义 N.2（framed disk category）.** 令 $\mathbf{Disk}^{fr}_n$ 为 symmetric monoidal infinity-category：

1. 对象为有限个带标准 framing 的 $\mathbb R^n$ 的不交并；
2. morphism spaces 为保 framing 的光滑 embeddings 的空间；
3. monoidal structure 为 disjoint union。

若 $M$ 是 framed $n$-manifold，定义 overcategory
$$
\mathbf{Disk}^{fr}_{n/M}
=
\mathbf{Disk}^{fr}_n\times_{\mathbf{Mfld}^{fr}_n}
\mathbf{Mfld}^{fr}_{n/M}.
$$
其对象是保 framing 的嵌入 $U\hookrightarrow M$，其中 $U$ 是有限个标准 disks 的不交并。

**定义 N.3（factorization homology）.** 对 $A\in\operatorname{Alg}_{E_n}(\mathcal C)$，把 $A$ 看作 symmetric monoidal functor
$$
A:\mathbf{Disk}^{fr}_n\longrightarrow \mathcal C.
$$
定义
$$
\int_M A
=
\operatorname*{colim}_{(U\hookrightarrow M)\in\mathbf{Disk}^{fr}_{n/M}} A(U).
$$
若 $U=\coprod_{i=1}^r\mathbb R^n$，则
$$
A(U)\simeq A^{\otimes r}.
$$

**说明 N.4.** 该定义中的 colimit 是 infinity-categorical colimit。若 $\mathcal C=\mathbf{Ch}_k$，实际计算常需先选择 cofibrant chain-level $E_n$-代数模型，再用 derived colimit 或等价的 bar 型模型计算。

## N.2 切结构版本

**定义 N.5.** 设 $\xi:B\to BO(n)$ 是 tangential structure。一个 $\xi$-framed $n$-manifold 是配备切丛 classifying map 到 $B$ 的 lift 的 $n$-manifold。记相应 disk category 为
$$
\mathbf{Disk}^{\xi}_n.
$$

**定义 N.6.** $\xi$-structured $E_n$-algebra 是 symmetric monoidal functor
$$
A:\mathbf{Disk}^{\xi}_n\to\mathcal C.
$$
对 $\xi$-structured $M$，定义
$$
\int_M A
=
\operatorname*{colim}_{\mathbf{Disk}^{\xi}_{n/M}}A(U).
$$

**例 N.7.**

1. 若 $\xi:*\to BO(n)$ 是一个 framing，则得到 framed factorization homology。
2. 若 $\xi:BSO(n)\to BO(n)$，则得到 oriented 版本。
3. 若 $\xi=\operatorname{id}_{BO(n)}$，则得到 unoriented 版本；此时系数对象必须带相应 $O(n)$-相干作用。

**警告 N.8.** 同一个底层 $E_n$-algebra 不自动定义 unoriented manifold 上的 factorization homology。必须给出旋转群或结构群作用的 homotopy coherent extension。

## N.3 函子性与不交并

**命题 N.9（嵌入函子性）.** 若 $j:M\hookrightarrow N$ 是 framed embedding，则有自然态射
$$
j_*:\int_M A\longrightarrow\int_N A.
$$

**证明.** 嵌入 $j$ 诱导函子
$$
\mathbf{Disk}^{fr}_{n/M}\longrightarrow \mathbf{Disk}^{fr}_{n/N},
\qquad
(U\hookrightarrow M)\mapsto(U\hookrightarrow M\xrightarrow{j}N).
$$
对定义 N.3 的图取 colimit，得到从较小索引范畴 colimit 到较大索引范畴 colimit 的 canonical map。自然性来自 overcategory composition 的结合律。$\square$

**命题 N.10（不交并公式）.** 对 framed $n$-manifolds $M,N$，有 canonical equivalence
$$
\int_{M\amalg N}A\simeq
\left(\int_M A\right)\otimes
\left(\int_N A\right).
$$

**证明.** 任一 disk 嵌入
$$
U=\coprod_{i\in I}\mathbb R^n\hookrightarrow M\amalg N
$$
按像落入 $M$ 或 $N$ 唯一分解为
$$
U_M\hookrightarrow M,\qquad U_N\hookrightarrow N.
$$
这给出索引范畴的相干等价
$$
\mathbf{Disk}^{fr}_{n/(M\amalg N)}
\simeq
\mathbf{Disk}^{fr}_{n/M}\times\mathbf{Disk}^{fr}_{n/N}
$$
与 disjoint union monoidal structure 相容。于是
$$
\begin{aligned}
\int_{M\amalg N}A
&\simeq
\operatorname*{colim}_{(U_M,U_N)}
A(U_M\amalg U_N)\\
&\simeq
\operatorname*{colim}_{(U_M,U_N)}
A(U_M)\otimes A(U_N)\\
&\simeq
\left(\operatorname*{colim}_{U_M}A(U_M)\right)
\otimes
\left(\operatorname*{colim}_{U_N}A(U_N)\right),
\end{aligned}
$$
最后一步使用 tensor product 分变量保持 colimits。$\square$

**推论 N.11.** 对有限不交并 $M=\coprod_{i=1}^rM_i$，
$$
\int_M A\simeq\bigotimes_{i=1}^r\int_{M_i}A.
$$
当 $r=0$ 时，空不交并给出 monoidal unit：
$$
\int_\varnothing A\simeq\mathbb 1_{\mathcal C}.
$$

## N.4 欧氏空间与点的计算

**命题 N.12（disk 归一化）.** 对标准 framed disk $\mathbb R^n$，
$$
\int_{\mathbb R^n}A\simeq A.
$$

**证明.** 在 $\mathbf{Disk}^{fr}_{n/\mathbb R^n}$ 中，恒等嵌入
$$
\mathbb R^n\hookrightarrow\mathbb R^n
$$
是 final object up to contractible choice：对任一 $U\hookrightarrow\mathbb R^n$，到恒等对象的 over-morphism 正是该嵌入，且相干自同伦空间由 framed isotopy contractibility 控制。Colimit over a category with final object 等于 final object 上的值，因此
$$
\int_{\mathbb R^n}A\simeq A(\mathbb R^n)\simeq A.
$$
其中 finality 的空间级陈述属于 embeddings 的标准同伦事实。$\square$

**例 N.13（零维情形）.** 当 $n=0$ 时，$\mathbb R^0=*$，一个 $0$-manifold 是离散有限集 $S$。若 $A$ 是 $E_0$-algebra，则
$$
\int_S A\simeq A^{\otimes S}.
$$
特别地，
$$
\int_\varnothing A\simeq\mathbb 1_{\mathcal C}.
$$

该例说明 factorization homology 同时推广了“把系数赋给点并对点集取张量积”的规则。

## N.5 Excision

**定义 N.14（collar gluing）.** 设 $N$ 是 $(n-1)$-manifold。一个 collar gluing 数据是
$$
M=M_-\cup_{N\times\mathbb R}M_+
$$
其中 $N\times\mathbb R$ 以开 collar 的形式嵌入 $M_-$ 与 $M_+$，并且两侧切结构与 $M$ 的切结构相容。

**外部输入定理 N.15（factorization homology excision；AF-1）.** 在约定 N.1 下，若 $M=M_-\cup_{N\times\mathbb R}M_+$ 是 collar gluing，则有自然等价
$$
\int_M A
\simeq
\left(\int_{M_-}A\right)
\otimes_{\int_{N\times\mathbb R}A}
\left(\int_{M_+}A\right).
$$
这里 $\int_{N\times\mathbb R}A$ 是由 collar 方向给出的 $E_1$-algebra object，$\int_{M_-}A$ 与 $\int_{M_+}A$ 分别是右、左 module object。
定位来源为 Ayala--Francis, arXiv:1206.5522v6, Lemma 3.18。

**说明 N.16.** 公式 N.15 中的 tensor product 是 derived relative tensor product。若 $\mathcal C=\mathbf{Ch}_k$，链级模型通常由 two-sided bar construction 表示：
$$
B\left(
\int_{M_-}A,\,
\int_{N\times\mathbb R}A,\,
\int_{M_+}A
\right).
$$
若底层代数或模对象未作 cofibrant replacement，普通相对张量积可能给出错误结果。

**命题 N.17（excision 的迭代使用规则）.** 设 $M$ 由有限个 collar gluing 构成。若每一步 gluing 的中间 collar $N_i\times\mathbb R$ 和两侧模块结构均已在同一 $\mathcal C^\otimes$ 中构造，则 $\int_MA$ 可由有限次 derived relative tensor product 计算。

**证明.** 对 gluing 次数归纳。一次 gluing 是定理 N.15。若 $M$ 先由 $M'$ 与 $P$ 沿 $N\times\mathbb R$ glue 得到，而 $M'$ 已由 $r-1$ 次 gluing 构成，则归纳假设给出 $\int_{M'}A$ 的表达式。把该表达式代入定理 N.15，即得到 $r$ 次 gluing 的表达式。结合性不是普通等式，而是 bar construction 或 infinity-category 中 colimit 的 canonical associativity equivalence。$\square$

## N.6 圆周与 Hochschild 同调

本节采用 $\mathcal C=\mathbf{Ch}_k$ 或更一般的稳定 presentable symmetric monoidal infinity-category，且 $A$ 是 $E_1$-algebra。

**外部输入定理 N.18（圆周计算；AF-2）.** 对 associative 或 $E_1$-algebra $A$，有自然等价
$$
\int_{S^1}A\simeq HH_\*(A),
$$
右侧为 $A$ 的 Hochschild homology object。

在 dg 代数模型中，
$$
HH_\*(A)\simeq
A\otimes^{\mathbf L}_{A\otimes A^{op}}A.
$$
定位来源为 AF-2，即 Ayala--Francis, arXiv:1206.5522v6, Theorem 3.19；Hochschild chain model 的符号仍需与定义 E.18--定义 E.23 和检查 W.1--检查 W.11 分开核对。

**证明边界.** 把 $S^1$ 沿两个区间 glue，可把计算化为一维 excision。端点形成的 $0$-manifold 的 collar algebra 对应 $A\otimes A^{op}$，两个区间给出相应双模，derived tensor product 得到 cyclic bar construction。严格证明需要带边界或 stratified 版本的 factorization homology 以及 cyclic bar 与 gluing 模型的识别，因此作为外部输入。$\square$

**说明 N.19.** 公式
$$
\int_{S^1}A\simeq HH_\*(A)
$$
不表示 $\int_{S^1}A$ 是普通 singular homology $H_\*(S^1;A)$。当 $A$ 非交换时，乘法顺序和双模结构参与计算。

**命题 N.20（交换系数的退化）.** 若 $B$ 是 $E_\infty$-algebra，并且其 $E_1$-结构由交换结构限制而来，则 $HH_\*(B)$ 可识别为 $B$ 上的 higher Hochschild chains over $S^1$。在适当线性情形中，这常写作
$$
B\otimes S^1.
$$

**证明边界.** 该识别依赖 $\mathcal C$ 是否 tensored over spaces 以及 $E_\infty$-algebra 的具体模型。不能在一般 $\mathcal C^\otimes$ 中把 $B\otimes S^1$ 当作已定义对象。$\square$

## N.7 高维圆柱与 Hochschild 对象

**外部输入定理 N.21.** 若 $A$ 是 $E_n$-algebra，则
$$
\int_{S^1\times\mathbb R^{n-1}}A
$$
携带自然 $E_{n-1}$-algebra 结构，并可解释为 $E_{n-1}$-Hochschild object of $A$。

**说明 N.22.** 该结论把 Dunn additivity 与一维圆周计算结合起来。直观上，$\mathbb R^{n-1}$ 方向留下 $E_{n-1}$-结构，$S^1$ 方向执行 trace 或 Hochschild 型循环化。严格陈述需要指定 $E_n$-operad 模型和 additivity equivalence。

## N.8 球面的 excision 表达式

设 $S^n$ 写成两个半球沿赤道 collar 的 gluing：
$$
S^n\simeq D^n_-\cup_{S^{n-1}\times\mathbb R}D^n_+.
$$

**命题 N.23.** 在带边界版本可用且半球系数对象已定义的情形下，
$$
\int_{S^n}A
\simeq
\left(\int_{D^n_-}A\right)
\otimes_{\int_{S^{n-1}\times\mathbb R}A}
\left(\int_{D^n_+}A\right).
$$

**证明.** 这是外部输入定理 N.15 应用于本节给出的赤道 collar gluing 的直接实例。$\square$

**警告 N.24.** 不应把 $\int_{D^n_\pm}A$ 自动替换成 $A$，除非已经固定带边界版本中半球的边界条件，并证明该半球对象与所需 module object 的识别。无边界开 disk $\mathbb R^n$ 的归一化命题 N.12 不能直接替代带边界半球计算。

## N.9 Locally constant factorization algebra 的重建

**定义 N.25.** 给定 $A\in\operatorname{Alg}_{E_n}(\mathcal C)$，定义
$$
\mathcal F_A(U)=\int_UA
$$
其中 $U\subset M$ 是 open submanifold。

若 $U_1,\ldots,U_r\subset V$ 两两不交，由不交并公式和嵌入函子性得到结构映射
$$
\mathcal F_A(U_1)\otimes\cdots\otimes\mathcal F_A(U_r)
\simeq
\mathcal F_A\left(\coprod_iU_i\right)
\longrightarrow
\mathcal F_A(V).
$$

**外部输入定理 N.26.** $\mathcal F_A$ 是 locally constant factorization algebra，并且在 $\mathbb R^n$ 上该构造给出 equivalence
$$
\operatorname{Alg}_{E_n}(\mathcal C)
\simeq
\operatorname{Fact}^{lc}_{\mathbb R^n}(\mathcal C).
$$

**说明 N.27.** 该定理不是形式 Kan extension 的普通范畴论推论。关键点是 Weiss descent、isotopy invariance 和 disk embeddings 的同伦性质。Ayala--Francis, arXiv:1206.5522v6, Theorem 3.24 已定位 homology theories for manifolds 与 Disk$_n$-algebras 的刻画；Costello--Gwilliam/Lurie 语境中的 locally constant factorization algebra 完整等价仍按 REFERENCE_LOCATOR_LEDGER 另行定位。

## N.10 Factorization homology 与普通同调的关系

**外部输入定理 N.28（交换系数）.** 若 $B$ 是 sufficiently commutative 的 $E_\infty$-algebra，且 $\mathcal C$ 支持 tensoring over spaces，则
$$
\int_M B\simeq B\otimes M
$$
其中 $B\otimes M$ 表示由空间 $M$ 对 $B$ 的 tensor。

在 $\mathbf{Ch}_k$ 的良好情形中，右侧可视为 higher Hochschild chains。

**说明 N.29.** 当 $B=k$ 是域上的常值交换代数时，上式与 chains on $M$ 的关系依赖所选模型。不能从 $\int_MB$ 的定义直接推出一个未加条件的等式
$$
\int_MB=H_\*(M;k).
$$
factorization homology 是链级或 infinity-categorical 对象；取同调是后续操作。

## N.11 与 Fukaya 理论的接口

Fukaya theory 中出现 factorization homology 的方式通常不是“给任意 symplectic manifold 一个 $E_n$-algebra 然后积分”。常见路径如下：

1. 局部模型：给某类 basic pieces 分配 dg category 或 $A_\infty$-category；
2. cosheaf 或 factorization algebra：证明这些局部范畴满足适当 descent；
3. gluing theorem：用 sectorial descent、skeletal descent 或 stratified factorization homology 重构全局 Fukaya category；
4. trace/center：用 Hochschild invariants、centers 或 factorization homology 计算全局不变量。

**外部输入定理 N.30（Fukaya 型 gluing 模式）.** 在 exact、wrapped、Liouville sector 或其他指定几何设置中，若已经建立 transversality、compactness、orientation、brane data 和 sectorial descent，则相应 Fukaya 型范畴可由局部模型通过 cosheaf 或 factorization-homology 型 gluing 得到。

**证明边界.** 该命题是模式陈述，不是单一数学定理。每个具体版本必须指定：

1. 几何对象类别；
2. Lagrangian 对象和 morphism complexes；
3. coefficient ring 或 Novikov field；
4. wrapped/exact/monotone/obstructed 条件；
5. gluing 的覆盖类型；
6. 对应的解析紧性与横截性定理。

缺少任一项时，只能把结论作为研究动机，不能纳入证明链。$\square$

## N.12 使用检查表

在正文中使用 factorization homology 结论前，必须检查：

1. **切结构。** $M$ 是 framed、oriented、unoriented，还是带一般 $\xi$-structure。
2. **系数结构。** $A$ 是 $E_n$、$E_n^\xi$、$E_\infty$，还是 enriched/categorical 版本。
3. **底层范畴。** $\mathcal C^\otimes$ 是否 presentable，tensor 是否分变量保持 colimits。
4. **边界。** 若 manifold 有边界或 corners，是否使用了带边界或 stratified 版本。
5. **excision。** 中间 collar algebra 和左右 module structure 是否已构造。
6. **derivedness。** 相对张量积是否为 derived relative tensor product。
7. **模型转换。** 若从 strict dg 模型切换到 infinity-categorical 模型，是否引用 rectification/localization 定理。
8. **几何分析。** 若涉及 Fukaya theory，是否有独立的 transversality、compactness、orientation 和 gluing 输入。

## N.13 与正文的依赖关系

本附录依赖：

1. 第十章的 $E_n$-operad 和 additivity 背景；
2. 第十二章的 Hochschild cochains 与 brace 背景；
3. 第十四章的模型范畴中 operad 和 rectification 边界；
4. 第十八、十九章的 infinity-operad 与 localization 语言；
5. 第二十章的 factorization algebra 与 Fukaya 接口。

反向地，第二十章的圆周计算、excision 说明、几何 gluing 边界应引用本附录，而不是在正文中重复技术条件。

## 练习

**练习 N.1.** 证明命题 N.10 中空不交并情形给出 monoidal unit。

**练习 N.2.** 对 $n=0$，把 $\mathbf{Disk}_{0/S}$ 写成有限子集嵌入 $S$ 的范畴，并验证例 N.13。

**练习 N.3.** 在 $\mathbf{Ch}_k$ 中写出 two-sided bar construction
$$
B(M,B,N)
$$
的 simplicial object，并说明其 geometric realization 计算 $M\otimes_B^{\mathbf L}N$。

**练习 N.4.** 对 associative algebra $A$，解释为什么 $A\otimes A^{op}$ 出现在 $HH_\*(A)$ 的相对张量积公式中。

**练习 N.5.** 给出一个 framed $E_n$-algebra 不能直接在 unoriented manifolds 上积分的原因。

**练习 N.6.** 列出一个 Fukaya category gluing 定理需要的几何假设，并说明哪些假设不是 operad theory 本身能证明的。
