# 附录 D：pro-etale 与凝聚数学的比较细节

## D.1 两个站点

凝聚数学的基本测试站点可取为 compact Hausdorff 空间：

$$
\mathbf{CHaus}.
$$

对象是紧 Hausdorff 空间，覆盖通常由有限 jointly surjective 族给出。

pro-etale 理论中，对 scheme $X$ 考虑

$$
X_{\operatorname{proet}}.
$$

按 Bhatt--Scholze 的原始口径，对象是 $X$ 上的 weakly étale scheme
$U\to X$，即该态射及其对角态射均平坦；覆盖取 fpqc covering。pro-etale 逆极限
描述定义同一个 Grothendieck topology，但这项比较是外部输入。

因此二者不是同一个站点。比较只能发生在证明模式、局部对象和同调控制策略层面。

## D.2 对照表

| 主题 | 凝聚数学 | pro-etale 理论 |
| --- | --- | --- |
| 基本站点 | compact Hausdorff/profinite/ED 测试空间 | $X$ 上的 weakly étale 对象与 fpqc 覆盖 |
| 典型对象 | $S\in\mathbf{CHaus}$ | $U\to X$ weakly étale |
| 关键局部对象 | extremally disconnected compact Hausdorff spaces | w-contractible 或类似对象 |
| 主要用途 | 拓扑代数、函数分析、solid/analytic/liquid 结构 | etale cohomology 的局部与几何控制 |
| 共同技术 | sheaf、覆盖、共同细化、投射测试对象 | sheaf、覆盖、共同细化、投射测试对象 |

## D.3 抽象共同模式

设 $\mathcal C$ 是站点，$\mathcal P\subset\mathcal C$ 是一族对象。若满足：

1. 每个 $U\in\mathcal C$ 可由 $\mathcal P$ 中对象覆盖；
2. $\mathcal P$ 对纤维积和共同细化足够稳定；
3. $\mathcal P$ 中对象对覆盖提升问题表现为投射；

则 sheaf 计算常可化到 $\mathcal P$ 上进行。

**命题 D.3.1。** 在上述假设下，若 $F$ 是 sheaf，且对每个 $P\in\mathcal P$ 有 $F(P)=0$，则在 $\mathcal P$ 覆盖能检测截面的条件下，$F=0$。

**证明。** 对任意 $U$，选 $\mathcal P$-覆盖 $\{P_a\to U\}$。sheaf 条件给出单射

$$
F(U)\to\prod_aF(P_a).
$$

右侧为 $0$，故 $F(U)=0$。由于 $U$ 任意，$F=0$。证毕。

这个命题很简单，但体现了共同思想：足够多的好测试对象可以检测 sheaf。

## D.4 高阶同调的谨慎说法

如果 $\mathcal P$ 中对象不仅检测截面，还对 sheaf cohomology 有消失性质，那么可进一步用它们计算高阶同调。凝聚数学中，极不连通对象的投射性使许多导出计算变得可控；pro-etale 理论中，w-contractible 对象也服务于类似目标。

谨慎表述：

1. 可以说“二者都使用投射型局部对象降低 sheaf cohomology 难度”。
2. 不应说“凝聚数学的 ED 空间就是 pro-etale 的 w-contractible 对象”。
3. 不应把 $\mathbf{CHaus}$ 上的 sheaf 直接当作 $X_{\operatorname{proet}}$ 上的 sheaf。

## D.5 与第三卷复几何的关系

第三卷讨论复几何时，凝聚语言主要用于组织拓扑向量空间、解析模、Dolbeault 复形和相干层同调。pro-etale 理论则更多出现在算术几何和 etale cohomology 背景中。

二者在“现代几何如何选择更好的站点”这一层面相互启发。共同经验是：若原始站点上对象不够投射、覆盖不够细、同调不够可控，就扩大或改变站点，使局部对象更适合计算。

这里还应记录一个比“证明模式相似”更强、但仍不等同两个站点的已知结果。对 coherent
scheme $X$，Wolf 证明 hypercomplete pro-étale $\infty$-topos 等价于
$\operatorname{Gal}(X)$ 在 pyknotic spaces 中的连续表示范畴。比较的中介是 Galois
category，不是一个把裸 compactum 直接视为 $X$-scheme 的函子；coherent 与
hypercomplete 假设也不能从陈述中删去。正文第七章 7.6 节给出精确调用口径。

## D.6 典型错误

错误 1：把 profinite set 当作 scheme 上的 pro-etale cover。

修正：profinite set 可作为凝聚测试对象；pro-etale 站点中的覆盖对象必须带有到 $X$ 的
weakly étale 几何结构，并满足该站点的 fpqc 覆盖条件。

错误 2：把 condensed sheaf 的值 $F(S)$ 与 etale sheaf 的值 $G(U)$ 直接比较。

修正：需要一个明确的函子或几何构造把 $S$ 与 $U$ 联系起来，否则二者只是形式相似。

错误 3：把“极不连通”当作所有站点中的同一个性质。

修正：不同站点中的投射型对象定义不同；可以比较性质，不可省略定义。
