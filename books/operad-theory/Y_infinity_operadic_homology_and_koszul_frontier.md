# 附录 Y：Infinity-operadic homology 与 Koszul 对偶的前沿接口

## Y.0 目的和边界

本附录为开放问题 21.7--21.9 建立 strict specialization 接口。它不把 infinity-operadic homology 或 Koszul 对偶的外部结果并入定义 8.16 和定理 9.20 所属的 classical Koszul duality，而是完成三件可在书内验证的工作：

1. 给出 strict operad 到树指标线性对象的可证明嵌入。
2. 说明 classical operad algebra 与 tree-indexed algebra 描述之间的严格关系。
3. 写出任何 infinity-operadic Koszul 扩张若要兼容经典理论必须满足的特化条件。

除非另有说明，本附录固定交换环 $k$，链复形采用同调分次，$\Omega$ 表示 Moerdijk--Weiss 树范畴。

## Y.1 树指标线性化

**定义 Y.1.1.** 一个 $k$-linear dendroidal module 是函子
$$
X:\Omega^{op}\longrightarrow \mathbf{Ch}_k .
$$
若 $X$ 取值于非负度链复形且每个 $X_T$ 由集合线性生成，则称 $X$ 为集合型线性化对象。

**定义 Y.1.2.** 设 $\mathcal P$ 是集合值 colored operad。它的 dendroidal nerve 是
$$
N_d(\mathcal P)(T)=\operatorname{Hom}_{\operatorname{Op}_{col}}(\Omega(T),\mathcal P).
$$
它的线性 nerve 是
$$
kN_d(\mathcal P)(T)=k[N_d(\mathcal P)(T)]
$$
视为集中在次数 $0$ 的链复形。

**定义 Y.1.3.** 若 $\mathcal P$ 是 $k$-线性 colored operad，定义它的树装饰链复形
$$
C_{\mathcal P}(T)
=
\bigoplus_{\phi:E(T)\to \operatorname{Col}(\mathcal P)}
\bigotimes_{v\in V(T)}
\mathcal P\bigl((\phi(e))_{e\in \operatorname{in}(v)};\phi(\operatorname{out}(v))\bigr).
$$
张量积按 $V(T)$ 的任一全序书写；换序由 $\mathbf{Ch}_k$ 的对称 braiding 给出。若所有顶点装饰集中在次数 $0$，则换序不产生符号。

**命题 Y.1.4.** 若 $\mathcal P$ 是集合值 colored operad，则有自然同构
$$
kN_d(\mathcal P)(T)\cong C_{k[\mathcal P]}(T).
$$

**证明.** 由命题 16.7，给出 $\Omega(T)\to\mathcal P$ 等价于给出每条边的颜色 $\phi:E(T)\to\operatorname{Col}(\mathcal P)$，并对每个顶点 $v$ 给出一个运算
$$
\theta_v\in
\mathcal P\bigl((\phi(e))_{e\in\operatorname{in}(v)};\phi(\operatorname{out}(v))\bigr).
$$
因此 $N_d(\mathcal P)(T)$ 是这些边着色和顶点装饰数据的集合。对集合取自由 $k$-模，把 Cartesian product 的自由模识别为相应张量积，把 disjoint union 识别为直和，即得到右侧。自然性来自 $\Omega(T)$ 的泛性质：树态射先把源树中的顶点生成元送到由目标树生成元复合得到的运算；线性化后正是把顶点装饰按该树态射复合。$\square$

**定义 Y.1.5.** 设 $e$ 是树 $T$ 的 inner edge，$T/e$ 为收缩 $e$ 后的树。对应的 inner face 态射 $T/e\to T$ 在 presheaf 方向给出
$$
C_{\mathcal P}(T)\longrightarrow C_{\mathcal P}(T/e).
$$
该映射在 $e$ 相邻的两个顶点处使用 $\mathcal P$ 的 operadic composition，在其他顶点处取恒等。

**命题 Y.1.6.** 对 strict $k$-线性 operad $\mathcal P$，任意两条可收缩 inner edge $e_1,e_2$ 的先后收缩诱导相同的映射
$$
C_{\mathcal P}(T)\longrightarrow C_{\mathcal P}(T/(e_1,e_2))
$$
只要两条收缩得到同一个商树。

**证明.** 若 $e_1,e_2$ 不相邻，则两次映射作用在不相交的顶点装饰因子上；由张量积函子性和 braiding 自然性，先后顺序无关。若两条边相邻，则相关部分是一棵三顶点子树。两个收缩顺序分别对应
$$
(\alpha\circ_i\beta)\circ_j\gamma
\quad\text{和}\quad
\alpha\circ_i(\beta\circ_j\gamma)
$$
的某个重标号版本。operad 的结合律公理断言这两个复合相同，且对称重标号由等变性处理。因此两条路径给出同一链映射。$\square$

## Y.2 Segal 型线性 dendroidal 对象

**定义 Y.2.1.** 一个 Segal-linear dendroidal object 是 $k$-linear dendroidal module $X$ 连同对每棵树 $T$ 的 Segal 映射
$$
X(T)\longrightarrow
\operatorname*{holim}_{v\in V(T)} X(C_{\operatorname{in}(v)})
$$
的选择，使得：

1. 对 corolla $C_S$，Segal 映射为恒等；
2. 对单位树 $\eta$，$X(\eta)$ 记录颜色对象；
3. 对 inner edge 收缩，诱导映射与顶点复合相容；
4. 若所有 Segal 映射都是 weak equivalences，则称 $X$ 满足同伦 Segal 条件。

本定义是本书的接口定义，不声称等同于任一预印本中的 linear infinity-operad 定义。

**命题 Y.2.2.** 每个 strict $k$-线性 colored operad $\mathcal P$ 给出满足严格 Segal 条件的 Segal-linear dendroidal object $C_{\mathcal P}$。

**证明.** 对树 $T$，定义 Y.1.3 已把 $C_{\mathcal P}(T)$ 写成所有边颜色选择和顶点运算张量的直和。右侧
$$
\operatorname*{holim}_{v\in V(T)} C_{\mathcal P}(C_{\operatorname{in}(v)})
$$
在 strict 集合型情形退化为对共同边颜色相容条件的 ordinary limit，再线性化后正是同一个直和-张量表达式。复合相容性由命题 Y.1.6 给出。因此 Segal 映射是同构，特别是 weak equivalence。$\square$

**警告 Y.2.3.** 命题 Y.2.2 的逆命题不成立。一个满足同伦 Segal 条件的 dendroidal object 只在同伦意义上具有多输入复合；它不自动给出 strict operad。若需要 strict operad，必须额外选择 fibrant/cofibrant replacement、rectification 或严格化定理。

## Y.3 Algebra over an infinity-operad 的严格特化

**定义 Y.3.1.** 对链复形 $A$，其 endomorphism operad $\operatorname{End}_A$ 定义为
$$
\operatorname{End}_A(S)=\operatorname{Hom}_{\mathbf{Ch}_k}(A^{\otimes S},A).
$$
复合由 multilinear maps 的代入给出，符号由约定 E.1 和约定 E.3 的 Koszul rule 决定。

**命题 Y.3.2.** 对 strict $k$-线性 operad $\mathcal P$，以下数据等价：

1. $\mathcal P$-algebra structure on $A$；
2. operad morphism $\mathcal P\to\operatorname{End}_A$；
3. dendroidal natural transformation
   $$
   C_{\mathcal P}\longrightarrow C_{\operatorname{End}_A}
   $$
   whose value on corollas is compatible with operad composition.

**证明.** 第一项和第二项的等价是定义 1.14、命题 1.15 和定义 6.6 的线性版本。第二项推出第三项：对树 $T$ 的每个顶点装饰 $\theta_v$ 应用 operad morphism，得到 $\operatorname{End}_A$ 中的顶点装饰；对边颜色无色情形无额外数据，有色情形按颜色函数处理。自然性来自 operad morphism 保持复合和单位。第三项推出第二项：取 $T=C_S$，得到每个 arity 的映射
$$
\mathcal P(S)\longrightarrow \operatorname{End}_A(S).
$$
它与 inner face 自然性相容，正是保持 operadic composition；与 degeneracy/unit 自然性相容，正是保持单位。故得到 operad morphism。$\square$

**定义 Y.3.3.** 若 $X$ 是某个模型中的 infinity-operad，且 $\mathcal E_A$ 是同一模型中的 endomorphism object，则一个 $X$-algebra structure on $A$ 应为该模型中的映射
$$
X\longrightarrow \mathcal E_A.
$$
本定义只有在指定模型结构、fibrancy/cofibrancy 和 mapping object 后才有数学含义。

**警告 Y.3.4.** 不能把命题 Y.3.2 直接推广为“任意 infinity-operad 的代数就是 strict operad map”。对 infinity-operad，映射空间通常不是集合，composition 只在相干同伦意义下存在。

## Y.4 Infinity-operadic Koszul 对偶的特化检验

**定义 Y.4.1.** 一个 infinity-operadic Koszul extension claim 是下列形式的断言：

> 对某类 linear infinity-operads $X$，存在 cooperadic dual object $X^\ash$ 和 bar-cobar 型构造，使得 $X$-algebras 与 $X^\ash$-coalgebras 之间满足某种 Koszul duality。

本书不把这样的断言视作定理，除非补齐定义 D.0.2 的引用包。

**定义 Y.4.2.** 设 $\mathcal P$ 是 classical quadratic dg operad。称某个 extension claim 通过 strict specialization test，若满足：

1. strict operad $\mathcal P$ 经过嵌入 $j$ 变成 linear infinity-operad $j\mathcal P$；
2. $j\mathcal P$ 的 dual object 在 strict 子情形中等价于 classical Koszul dual cooperad $\mathcal P^¡$；
3. extension claim 的 bar-cobar comparison 在 $j\mathcal P$ 上退化为
   $$
   \Omega\mathcal P^¡\longrightarrow \mathcal P;
   $$
4. $j\mathcal P$-algebras 与 ordinary $\mathcal P$-algebras 的 homotopy theories 相容。

**命题 Y.4.3.** 若某个 infinity-operadic Koszul extension claim 未通过 strict specialization test，则它不能替代定义 8.16 和定理 9.20 所属的 classical Koszul theorem。

**证明.** 定义 8.16 和定理 9.20 之后用于 $A_\infty$、$L_\infty$ 和 bar-cobar 的结论依赖 classical statement
$$
\Omega\mathcal P^¡\to\mathcal P
$$
在 $\mathcal P$ Koszul 时是 quasi-isomorphism。若 extension claim 不能在 strict 子情形中恢复此映射，则它与定义 8.16 和定理 9.20 所使用的对象不是同一断言；用它替代 classical theorem 会改变结论的源、靶或代数范畴。若代数 homotopy theory 不相容，则即使源、靶形式相似，也不能推出同一个 category of algebras 的结果。因此未通过该检验的 extension claim 不能作为 classical theorem 的替代品。$\square$

**命题 Y.4.4.** 若 extension claim 通过 strict specialization test，也仍然只能作为 classical theorem 的扩展候选；要把它用作外部输入，还必须给出模型结构、完整假设和精确定理来源。

**证明.** strict specialization test 只检查该 claim 在已知子情形中不矛盾。它不证明一般 linear infinity-operad 上的 dual object 存在，也不证明 bar-cobar adjunction、model structure、fibrancy/cofibrancy 或 derived equivalence。上述内容都是新的外部输入。因此通过测试只是必要条件，不是充分证明。$\square$

## Y.5 低权重检查：非对称结合 operad

**例 Y.5.1.** 设 $\operatorname{Ass}_{ns}$ 是非对称结合 operad，$T$ 为两顶点平面树。每个顶点只能由唯一二元乘法 $\mu$ 装饰，因此
$$
C_{\operatorname{Ass}_{ns}}(T)\cong k.
$$
收缩唯一 inner edge 得到三叶 corolla $C_3$，其装饰仍由三元乘法的唯一元素表示，故
$$
C_{\operatorname{Ass}_{ns}}(C_3)\cong k.
$$
收缩映射为恒等 $k\to k$。

**解释 Y.5.2.** 若改用自由非结合 operad，则两种两顶点平面树代表 $(xy)z$ 与 $x(yz)$ 两个不同基元；结合关系正是把它们在商 operad 中等同。由此可见：树指标对象记录的不只是 arity，也记录复合形状；Koszul 或 bar-cobar 理论中的权重滤过必须保留树形信息。

## Y.6 Koszul 扩张问题的充分数据

一个 infinity-operadic Koszul duality 命题只有在下列数据均已定义时才具有确定的真假条件：

1. linear infinity-operad 的正式定义及其与本附录 Segal-linear object 的关系；
2. strict operad 嵌入该模型的函子 $j$；
3. 该模型中的 weak equivalence、fibration、cofibration 或至少 mapping space；
4. dual object 的构造及其在 strict quadratic 情形中的特化；
5. bar-cobar adjunction 的源、靶和单位/余单位；
6. algebras over $X$ 与 coalgebras over $X^\ash$ 的 precise homotopy theory；
7. 与定义 8.16 和定理 9.20 所属 classical Koszul duality 的 commuting comparison diagram；
8. 若结论来自外部资料，给出可定位的定理版本和 proof dependencies。

## 练习

**练习 Y.1.** 对一棵三顶点线性树，写出命题 Y.1.6 中两种 inner edge 收缩顺序对应的 operad 结合律公式。

**练习 Y.2.** 设 $\mathcal P$ 是单色集合值 operad。证明 $N_d(\mathcal P)(\eta)$ 是单点集，并解释有色情形中它如何记录颜色集合。

**练习 Y.3.** 对 $\operatorname{Com}$ 计算 $C_{\operatorname{Com}}(T)$，其中 $T$ 是任意有限有根树。说明为什么所有顶点复合收缩后都得到同一个 arity 运算。

**练习 Y.4.** 选择一个 classical Koszul operad $\mathcal P$，写出 strict specialization test 的四个条件在该例中的具体形式。
