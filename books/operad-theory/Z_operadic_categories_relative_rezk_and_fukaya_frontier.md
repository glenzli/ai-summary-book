# 附录 Z：Operadic categories、relative Rezk nerve 与 Fukaya 前沿接口

## Z.0 目的和边界

本附录处理版本表 21.3 和流程 21.16 中另外三类前沿方向：

1. operadic categories、wreath product 和 operadic nerve；
2. relative dendroidal Rezk nerve 与 operadic localization；
3. Fukaya categories 的高阶 operadic 或多范畴结构。

这些方向目前均保持研究边界状态。本附录只给出本书可以内部验证的接口定义、低阶检查和条件性命题；凡涉及近期预印本的主定理、Rezk nerve 完整模型结构、Fukaya category 的分析构造或 gluing theorem，均仍为外部输入或研究边界。

## Z.1 Operadic category 的数据包

**定义 Z.1.1.** 一个 pre-operadic category datum 由以下数据组成：

1. 一个小范畴 $\mathcal O$；
2. 一个 cardinality functor
   $$
   |-|:\mathcal O\longrightarrow \mathbf{Fin};
   $$
3. 对每个对象 $T\in\mathcal O$ 和每个 $i\in |T|$，一个 local color 或 local component 记号 $T_i$；
4. 对每个 morphism $f:T\to S$ 和每个 $i\in |S|$，一个 fiber object
   $$
   f^{-1}(i)\in\mathcal O
   $$
   满足
   $$
   |f^{-1}(i)|=|f|^{-1}(i)
   $$
   作为有限集；
5. 对每个 connected component 选择一个 local terminal object。

若这些数据还满足 Batanin--Markl 型的 identity、composition 和 local terminal axioms，则称为 operadic category。本书不在此证明这些公理的完整理论。

**例 Z.1.2.** $\mathbf{Fin}$ 自身给出 pre-operadic category datum：令 $|S|=S$，令 $f^{-1}(i)$ 为普通集合论 fiber，local terminal 为 singleton。恒等映射的 fiber 是 singleton；复合
$$
T\xrightarrow{f}S\xrightarrow{g}R
$$
满足
$$
(g f)^{-1}(r)=\coprod_{s\in g^{-1}(r)} f^{-1}(s)
$$
作为有限集。

**命题 Z.1.3.** 在例 Z.1.2 中，fiber 对复合的公式与定义 1.5 的代入乘积中的有限集分块拉平相同。

**证明.** 定义 1.5 的代入乘积使用有限集 $I$ 的分块 $\pi:I\twoheadrightarrow J$，再对每个块 $I_j$ 继续分块。把 $\pi$ 记为函数 $f:I\to J$，第二层分块记为 $g:J\to K$，则 $K$ 的一个元素 $k$ 的总 fiber 是
$$
(g f)^{-1}(k)=\coprod_{j\in g^{-1}(k)}f^{-1}(j).
$$
这正是例 Z.1.2 中 fiber 对复合的公式。代入乘积的结合律来自同一个 disjoint-union associativity。$\square$

**定义 Z.1.4.** 若 $\mathcal O$ 是 operadic category，一个 $\mathcal O$-operad 应理解为“以 $\mathcal O$ 的对象为 operation profiles，以 fiber functor 控制 substitution”的广义 operad。完整定义依赖 operadic category 的公理包，本书只在研究边界中使用该语言。

**警告 Z.1.5.** Colored operad 与 operadic category 不是同一个定义。Colored operad 以颜色集和输入有限集为基础；operadic category 把 fiber、cardinality 和 local terminal 放入一个独立范畴结构中。任何把二者等同的说法都必须经过 Grothendieck construction 或 nerve comparison。

## Z.2 Boardman--Vogt tensor product 与 wreath product 的接口

**定义 Z.2.1.** 设 $\mathcal P,\mathcal Q$ 是同一对称幺半底范畴中的 operads。若存在 operad $\mathcal P\otimes_{BV}\mathcal Q$，使得对任意 $\mathcal C$ 中对象 $A$，
$$
(\mathcal P\otimes_{BV}\mathcal Q)\text{-Alg}(A)
$$
等价于“$\mathcal P$-algebra object in $\mathcal Q$-algebras”或等价地“$\mathcal P$ 与 $\mathcal Q$ 的 operations 在 $A$ 上满足 interchange laws”，则称它为 Boardman--Vogt tensor product。

**命题 Z.2.2（interchange 的低阶形式）.** 若 $\mu\in\mathcal P(2)$，$\nu\in\mathcal Q(2)$，则 $\mathcal P\otimes_{BV}\mathcal Q$-algebra 上的二元运算必须满足
$$
\nu(\mu(a_{11},a_{12}),\mu(a_{21},a_{22}))
=
\mu(\nu(a_{11},a_{21}),\nu(a_{12},a_{22}))
$$
的适当对称重标号版本。

**证明.** Boardman--Vogt tensor product 的 universal property 要求 $\mathcal P$-operations 是 $\mathcal Q$-algebra morphisms，等价地 $\mathcal Q$-operations 是 $\mathcal P$-algebra morphisms。把 $\nu$ 视为 $\mathcal P$-algebra morphism，它必须保持 $\mu$，于是得到公式左侧等于右侧。带对称群作用时，需要把四个输入按矩阵转置置换；这正是 interchange law 的重标号部分。$\square$

**警告 Z.2.3.** Pavlova 的 wreath product 与 Boardman--Vogt tensor product 的关系属于研究边界。命题 Z.2.2 只说明 classical tensor product 的低阶 interchange，不证明 operadic categories 上的近期定理。

## Z.3 Operadic nerve 的接口

**定义 Z.3.1.** 一个 operadic nerve interface 是把 operadic category datum $\mathcal O$ 送到某种 higher nerve object $N_{\mathrm{op}}(\mathcal O)$ 的构造候选，并要求：

1. $0$-simplices 或 colors 记录 local terminal/local color data；
2. $1$-simplices 记录 morphisms in $\mathcal O$；
3. higher simplices 记录 fiber 对复合的相干性；
4. Segal 型条件恢复“复合由 fiber 拉平控制”的事实。

本定义只是本书的接口，不等同于 Batanin--Kock--Weber 预印本中的 operadic nerve。

**命题 Z.3.2.** 任何 operadic nerve 若满足定义 Z.3.1 的四条要求，则其一维截断至少恢复 $\mathcal O$ 的 underlying category。

**证明.** 一维截断包含 $0$-simplices 和 $1$-simplices。由要求 1，$0$-simplices 至少记录对象的 local component 数据；由要求 2，$1$-simplices 记录 $\mathcal O$ 中 morphisms。要求 3 在 $2$-simplices 层记录 composable morphisms 的复合相干性，因此一维截断连同 $2$-simplex 的合成边恢复 ordinary nerve 中的 category composition。故 underlying category 被恢复到 nerve 所允许的等价精度。$\square$

**警告 Z.3.3.** Dendroidal nerve $N_d(\mathcal P)$、category of operators nerve 和 operadic category 的 operadic nerve 是三种不同构造。它们可以比较，但不能在定义层面互相替换。

## Z.4 Relative dendroidal object 与 Rezk nerve 接口

**定义 Z.4.1.** 一个 relative dendroidal object 是二元组 $(X,W)$，其中 $X$ 是 dendroidal set，$W$ 是 $i^\*X$ 中一类 $1$-simplices，满足：

1. $i:\Delta\hookrightarrow\Omega$ 是线性树嵌入；
2. $i^\*X$ 是 $X$ 的线性限制；
3. $W$ 包含所有 degenerate $1$-simplices；
4. 若 $i^\*X$ 中存在两个 $W$-morphisms 的复合，则复合仍在 $W$ 中。

把 $W$ 中的 $1$-simplices 称为 weak unary equivalences。

**例 Z.4.2.** 设 $\mathcal P$ 是 strict colored operad。选择其 unary part $\mathcal P(1)$ 中一类 morphisms $W$，包含单位并对复合封闭。则
$$
(N_d(\mathcal P),W)
$$
是 relative dendroidal object。

**证明.** 线性限制 $i^\*N_d(\mathcal P)$ 是由 $\mathcal P$ 的 unary operations 构成的 simplicial nerve 型对象。单位 unary operations 给出 degenerate $1$-simplices。若 $W$ 对 operad 的 unary composition 封闭，则在 $i^\*N_d(\mathcal P)$ 中可复合的 $W$-边复合仍在 $W$ 中。故满足定义 Z.4.1。$\square$

**定义 Z.4.3.** 一个 dendroidal Rezk nerve construction 应把 relative dendroidal object $(X,W)$ 送到某种 complete Segal / Rezk 型对象，使得：

1. dendroidal Segal 条件编码多输入复合；
2. Rezk completeness 条件编码 $W$ 中的一元弱等价；
3. 对线性树的限制退化为 ordinary relative category 的 Rezk nerve；
4. localization 的 universal property 只在指定模型结构中陈述。

**命题 Z.4.4.** 若 $X$ 的所有非线性树值被忽略，则 relative dendroidal object $(X,W)$ 的线性部分给出 ordinary relative simplicial object。

**证明.** 线性嵌入 $i:\Delta\hookrightarrow\Omega$ 把 $X$ 限制为 simplicial set $i^\*X$。定义 Z.4.1 中的 $W$ 正是该 simplicial set 的一类 $1$-simplices，包含退化边并对可定义复合封闭。因此线性部分只记录 ordinary categorical direction 上的 weak morphisms，而不记录多输入顶点。$\square$

**警告 Z.4.5.** Relative dendroidal Rezk nerve 的完整理论不能由定义 Z.4.3 推出；它需要模型结构、fibrancy、localization universal property 和与已有 infinity-operad 模型的比较。

## Z.5 Fukaya 高阶结构的 operadic 接口

**定义 Z.5.1.** 一个 Fukaya operadic interface 由以下数据组成：

1. 一类几何对象，记为 Lagrangian labels；
2. 对每对 labels $L_0,L_1$，一个同调分次链复形 $CF_\*(L_0,L_1)$；
3. 对每个 $n\ge0$ 和 label 序列 $L_0,\ldots,L_n$，一个运算
   $$
   \mu^n:
   CF_\*(L_{n-1},L_n)\otimes\cdots\otimes CF_\*(L_0,L_1)
   \longrightarrow
   CF_\*(L_0,L_n)
   $$
   其同调次数为 $n-2$；
4. 对 module、bimodule 或 higher slot 的变体，允许部分输入位置被标记为 module-type inputs；
5. 一个几何 gluing rule，说明一维紧化模空间的边界由两个低维 operation 的复合组成。

本定义只抽取代数接口；真正的 $CF^\*$、$\mu^n$ 和 gluing rule 依赖辛几何分析。

**命题 Z.5.2（条件性 $A_\infty$ 关系）.** 假设定义 Z.5.1 的运算来自带定向的一维紧化模空间计数，并且每个一维模空间的边界带符号等于所有两层 broken configurations 的和。则 $\{\mu^n\}$ 满足 $A_\infty$ relations。

**证明.** 固定输入 $x_1,\ldots,x_n$。一维紧化模空间的有向边界计数为 $0$。按假设，边界点被分解为所有两层 broken configurations；代数上每个 broken configuration 对应一个复合项
$$
\mu^{r+1+t}(x_n,\ldots,x_{s+r+1},\mu^r(x_{s+r},\ldots,x_{s+1}),x_s,\ldots,x_1)
$$
乘以 orientation 和 Koszul 符号。把所有边界点求和等于 $0$，正是 $A_\infty$ relation。符号是否与附录 E 的 convention 相同需要单独核对；若已核对，则得到本书 convention 下的关系。$\square$

**警告 Z.5.3.** 命题 Z.5.2 是条件性代数命题。它不证明 pseudo-holomorphic curve moduli spaces 存在、紧化良好、横截性成立或 orientation system 已构造。这些是 Fukaya theory 的外部输入。

**定义 Z.5.4.** 一个 dg $\mathbf{fc}$-multicategory interface 应至少包含：

1. object slots，对应 Fukaya categories 或 Lagrangian labels；
2. morphism slots，对应 morphism complexes、modules 或 bimodules；
3. multimorphisms，由带多个输入/输出角色的曲面或多边形类型索引；
4. differential，由一维边界退化给出；
5. composition，由 gluing of domains 给出。

**命题 Z.5.5.** 若一个 dg $\mathbf{fc}$-multicategory interface 的 gluing operation 满足严格结合，且 differential 的边界分解只含一次 gluing，则其 multimorphism complexes 形成 dg multicategory 的候选结构。

**证明.** Dg multicategory 需要三类数据：multimorphism complexes、composition maps、units，并要求 differential 与 composition 满足 Leibniz rule。前两类由定义给出；units 对应 trivial strips 或 identity operations。严格结合来自 gluing operation 的严格结合假设。Leibniz rule 来自一维边界分解：一个 glued configuration 的边界等于先取第一因子边界再 glue，加上按 Koszul sign 先取第二因子边界再 glue。故在这些假设下得到 dg multicategory 的候选结构。是否为完整 Fukaya 型对象仍依赖外部几何输入。$\square$

## Z.6 三类前沿的共同失败模式

**失败模式 Z.6.1.** 把 operadic category 的 nerve 当作 dendroidal nerve 使用。
修正：必须写出比较函子，并说明它保持 colors、operations、fiber 和 Segal 条件。

**失败模式 Z.6.2.** 把 relative dendroidal Rezk nerve 当作 ordinary localization。
修正：必须保留多输入树方向；只看线性树会丢失 operadic composition。

**失败模式 Z.6.3.** 把 Fukaya gluing 当作纯 operad 公理。
修正：operad 公理只表达 gluing 的代数后果；gluing 的几何成立依赖 compactness、transversality 和 orientation。

**失败模式 Z.6.4.** 把 2026 预印本作为已验证教材定理。
修正：必须先通过流程 21.16 和定义 D.0.2 的引用包登记。

## Z.7 进入正文的检查表

### Operadic categories

1. 给出完整 operadic category 公理。
2. 固定 cardinality functor、fiber functor 和 local terminal convention。
3. 说明与 colored operad 或 multicategory 的关系。
4. 对 operadic nerve，给出 simplicial 或 pseudo-simplicial 相干性数据。
5. 标记 Boardman--Vogt tensor product 与 wreath product 的精确定理来源。

### Relative dendroidal Rezk nerve

1. 定义 relative infinity-operad 的模型。
2. 指定 weak unary equivalences 的位置。
3. 给出 dendroidal Rezk nerve 的源、靶和 fibrant objects。
4. 说明其 localization universal property。
5. 与定义 19.3、外部输入定理 19.4、定义 19.24 和外部输入定理 19.25 中的 Dwyer--Kan localization 和 operadic localization 比较。

### Fukaya 高阶结构

1. 固定几何设置：exact、monotone、wrapped、stopped 或其他。
2. 记录 brane data、grading、Pin/spin、orientation 和 coefficient choices。
3. 说明 transversality 和 compactness 输入。
4. 写出 operations 的 degree 和 signs，并与定义 E.18--定义 E.23 和检查 W.1--检查 W.11 对齐。
5. 把 gluing theorem、sectorial descent 或 higher operad structure 标为外部输入。

## 练习

**练习 Z.1.** 在 $\mathbf{Fin}$ 中验证
$$
(g f)^{-1}(r)=\coprod_{s\in g^{-1}(r)}f^{-1}(s)
$$
与分块 refinement 的拉平公式相同。

**练习 Z.2.** 写出 Boardman--Vogt interchange law 在一个一元运算和一个二元运算之间的形式。

**练习 Z.3.** 对 strict colored operad $\mathcal P$，选择 unary isomorphisms 作为 $W$，证明 $(N_d(\mathcal P),W)$ 满足定义 Z.4.1。

**练习 Z.4.** 说明为什么只取 relative dendroidal object 的线性限制会丢失二元 operadic composition。

**练习 Z.5.** 在命题 Z.5.2 中，令 $n=3$，写出一维模空间边界对应的 $A_\infty$ 关系项。
