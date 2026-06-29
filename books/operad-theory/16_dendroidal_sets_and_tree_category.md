# 第十六章：Dendroidal sets 与树范畴 $\Omega$

Simplicial set 是定义在 simplex category $\Delta$ 上的 presheaf。Dendroidal set 是定义在树范畴 $\Omega$ 上的 presheaf。若 $\Delta$ 编码 composable strings of unary arrows，则 $\Omega$ 编码 rooted trees of many-input operations。本章只建立组合定义与严格 operad nerve；模型结构和 inner Kan 条件留到下一章。

## 16.1 Rooted trees

**定义 16.1.** 一个有限 rooted tree $T$ 由以下数据组成：

1. 有限边集 $E(T)$；
2. 有限顶点集 $V(T)$；
3. 对每个顶点 $v\in V(T)$，一个有限输入边集合 $\operatorname{in}(v)\subset E(T)$ 和一个输出边 $\operatorname{out}(v)\in E(T)$；
4. 一条根边 $r_T\in E(T)$；

满足：

1. 每条边至多是一个顶点的输出边；
2. 每条边至多是一个顶点的输入边；
3. 根边不是任何顶点的输入边；
4. 若从一条边 $e$ 开始，反复把 $e$ 替换为使 $e$ 成为输入的顶点的输出边，则过程有限并最终到达 $r_T$；
5. 除根边外，每条不是某个顶点输出边的边称为叶边。

条件 4 排除有向环，并保证所有边沿输出方向流向根。

**定义 16.2.** 若 $e\in E(T)$ 既是某个顶点的输出边，又是某个顶点的输入边，则称 $e$ 为 inner edge。叶边和根边称为 outer edges。

**例 16.3.** 单位树 $\eta$ 有一条边且无顶点。它编码 operad 的颜色而非运算。

**例 16.4.** $n$-corolla $C_n$ 有一个顶点 $v$，输入边为 $n$ 条叶边，输出边为根边。它编码一个 $n$ 元运算。特别地，$C_0$ 编码 nullary operation。

**定义 16.5.** 线性树 $L_n$ 有 $n$ 个 unary 顶点和 $n+1$ 条边，形状为
$$
e_n\to e_{n-1}\to\cdots\to e_0,
$$
其中 $e_0$ 是根边。线性树给出 $\Delta$ 到 $\Omega$ 的嵌入来源。

## 16.2 由树生成的 colored operad

**定义 16.6.** 给定 rooted tree $T$，定义集合值 colored operad $\Omega(T)$ 如下：

1. 颜色集为边集 $E(T)$；
2. 对每个顶点 $v$，加入一个生成运算
   $$
   \theta_v:\operatorname{in}(v)\to \operatorname{out}(v);
   $$
3. $\Omega(T)$ 是由这些生成运算自由生成的 colored operad。

这里 $\operatorname{in}(v)$ 是一个有限输入边集合，而非有序列表；对称性由 colored symmetric operad 的双射重标号给出。

**命题 16.7.** 对任意 colored operad $\mathcal P$，给出 operad morphism
$$
F:\Omega(T)\to\mathcal P
$$
等价于给出：

1. 对每条边 $e\in E(T)$ 的颜色选择 $F(e)\in\operatorname{Col}(\mathcal P)$；
2. 对每个顶点 $v\in V(T)$ 的运算
   $$
   F(\theta_v)\in\mathcal P\big((F(e))_{e\in\operatorname{in}(v)};F(\operatorname{out}(v))\big).
   $$

**证明.** $\Omega(T)$ 按定义是由颜色 $E(T)$ 和顶点生成运算 $\theta_v$ 自由生成的 colored operad。自由 colored operad 的泛性质说明：从 $\Omega(T)$ 到 $\mathcal P$ 的 operad morphism 唯一地由颜色函数和生成运算的像决定，条件仅是每个生成运算的输入输出颜色匹配。反向地，任何满足颜色匹配的数据由泛性质唯一延拓为 operad morphism。$\square$

**定义 16.8.** 树范畴 $\Omega$ 的对象是有限 rooted trees。态射定义为
$$
\Omega(S,T)=\operatorname{Operad}_{\mathrm{col}}\big(\Omega(S),\Omega(T)\big),
$$
即相应自由 colored operads 之间的 morphisms。

**说明 16.9.** 这个定义避免把 $\Omega$ 的态射误解为朴素图嵌入。一个态射可把源树的顶点送到目标树中的复合运算，即送到目标树的一个子树所表示的运算。

## 16.3 Dendroidal sets

**定义 16.10.** Dendroidal set 是 presheaf
$$
X:\Omega^{\operatorname{op}}\to\mathbf{Set}_{\mathcal U}.
$$
Dendroidal sets 的范畴记为
$$
\mathbf{dSet}=\operatorname{Fun}(\Omega^{\operatorname{op}},\mathbf{Set}_{\mathcal U}).
$$

若 $T$ 是树，$X_T$ 表示 $X(T)$，称为 $T$-dendrexes 集合。

**定义 16.11.** 对树 $T$，representable dendroidal set 记为
$$
\Omega[T]=\Omega(-,T).
$$

Yoneda 引理给出自然双射
$$
\mathbf{dSet}(\Omega[T],X)\cong X_T.
$$

**命题 16.12.** $\mathbf{dSet}$ 有所有小极限和小余极限，并且它们逐树计算。

**证明.** $\mathbf{dSet}$ 是小范畴 $\Omega$ 上的 presheaf 范畴。集合值 presheaf 范畴的极限和余极限逐对象计算。因此对任意 diagram $D:I\to\mathbf{dSet}$，
$$
(\lim_I D)_T=\lim_I(D_i(T)),\qquad
(\operatorname{colim}_I D)_T=\operatorname{colim}_I(D_i(T)).
$$
$\square$

**命题 16.13.** Representables $\Omega[T]$ 生成 $\mathbf{dSet}$：每个 dendroidal set $X$ 都是 representables over $X$ 的 colimit：
$$
X\cong \operatorname{colim}_{(\Omega[T]\to X)}\Omega[T].
$$

**证明.** 这是 presheaf 范畴的 category of elements 表示。由 Yoneda 引理，$\Omega[T]\to X$ 等价于元素 $x\in X_T$。把所有这样的元素组成 category of elements $\int_\Omega X$，其投影到 $\Omega$ 给出 representables diagram。对每个树 $S$，该 colimit 在 $S$ 上的值为所有三元组 $(T,x\in X_T,\alpha:S\to T)$ 对 Yoneda 关系取商。映射到 $X_S$ 定义为 $(T,x,\alpha)\mapsto X(\alpha)(x)$。Yoneda 关系正是保证该映射双射的关系。$\square$

## 16.4 Dendroidal nerve

**定义 16.14.** 设 $\mathcal P$ 是 small colored operad。它的 dendroidal nerve 是 dendroidal set
$$
N_d(\mathcal P)_T
=\operatorname{Operad}_{\mathrm{col}}\big(\Omega(T),\mathcal P\big).
$$
对态射 $\alpha:S\to T$，限制映射
$$
N_d(\mathcal P)_T\to N_d(\mathcal P)_S
$$
由预复合 $\Omega(S)\to\Omega(T)\to\mathcal P$ 给出。

**命题 16.15.** 若 $\mathcal P$ 是 colored operad，则
$$
N_d(\mathcal P)_\eta=\operatorname{Col}(\mathcal P).
$$

**证明.** $\Omega(\eta)$ 是只有一个颜色且无非恒等生成运算的自由 colored operad。从 $\Omega(\eta)$ 到 $\mathcal P$ 的 operad morphism 只需选择该唯一颜色的像。因此这类 morphisms 与 $\operatorname{Col}(\mathcal P)$ 的元素一一对应。$\square$

**命题 16.16.** 若 $C_n$ 是 $n$-corolla，则
$$
N_d(\mathcal P)_{C_n}
$$
是 $\mathcal P$ 中所有 $n$ 元运算连同其输入输出颜色的集合。若 $\mathcal P$ 是 one-colored operad，则
$$
N_d(\mathcal P)_{C_n}\cong \mathcal P(n).
$$

**证明.** $\Omega(C_n)$ 的颜色是 $n$ 条输入边和一条输出边，并有一个生成 $n$ 元运算。由命题 16.7，给出 $\Omega(C_n)\to\mathcal P$ 等价于选择这些边的颜色，并选择一个具有相应输入输出颜色的 $n$ 元运算。若 $\mathcal P$ 只有一个颜色，颜色选择唯一，剩余数据正是 $\mathcal P(n)$ 的元素。$\square$

**定义 16.17.** 树 $T$ 的 Segal core $\operatorname{Sc}[T]\subset\Omega[T]$ 是由所有顶点 corollas
$$
C_v\to T,\qquad v\in V(T),
$$
的像生成的 sub-dendroidal set。

**命题 16.18（严格 Segal 性）.** 对任意 colored operad $\mathcal P$ 和树 $T$，限制映射
$$
N_d(\mathcal P)_T\to \mathbf{dSet}(\operatorname{Sc}[T],N_d(\mathcal P))
$$
是双射。

**证明.** 左端是 operad morphisms $\Omega(T)\to\mathcal P$。由命题 16.7，这等价于对每条边选择颜色，并对每个顶点选择颜色匹配的运算。

右端是从所有顶点 corollas 的并所给出的图形到 $N_d(\mathcal P)$ 的自然变换。给出这样的自然变换等价于：

1. 对每个顶点 corolla $C_v$ 给出一个元素 $N_d(\mathcal P)_{C_v}$；
2. 当两个 corollas 共享一条边时，它们在对应 $\eta$-face 上给出的颜色相同。

由命题 16.15 和 16.16，这正是为 $T$ 的每条边选择一个颜色，并为每个顶点选择一个输入输出颜色匹配的运算。两边数据相同，构造互逆。$\square$

**外部输入定理 16.19.** Dendroidal nerve
$$
N_d:\operatorname{Operad}_{\mathrm{col}}\to\mathbf{dSet}
$$
是 fully faithful。其本质像可由严格 Segal 条件刻画。

**说明 16.20.** 命题 16.18 证明了 nerve 满足严格 Segal 性。Fully faithfulness 还需证明 dendroidal natural transformations 由 $\eta$ 与 corollas 上的数据唯一确定，并且自然性恰好等价于 operad 的单位、复合和等变性保持。完整证明属于 Moerdijk-Weiss dendroidal nerve 的标准定理。

## 16.5 Faces、degeneracies 与 horns

**定义 16.21.** 设 $T$ 是树。

1. 若 $e$ 是 inner edge，收缩 $e$ 得到树 $T/e$；对应态射
   $$
   T/e\to T
   $$
   称为 inner face map。
2. 若 $v$ 是可从外侧删除的 outer vertex，删除 $v$ 及相应 outer edges 得到树 $T\setminus v$；对应态射
   $$
   T\setminus v\to T
   $$
   称为 outer face map。
3. 若 $v$ 是 unary vertex，删除 $v$ 并合并其输入边和输出边得到树 $T/v$；对应态射
   $$
   T\to T/v
   $$
   称为 degeneracy map。

这里“可从外侧删除”指 $v$ 至多连接一条 inner edge，其余相邻外边全是 leaves 或 root，使删除后仍是 rooted tree。对 corolla $C_n$，还允许每条边给出的颜色嵌入
$$
\eta\to C_n
$$
作为 elementary outer face。

**说明 16.22.** Faces 是 $\Omega$ 中注入型的基本态射，degeneracies 是满射型的基本态射。完整的唯一分解定理需要树范畴的 generalized Reedy 结构，本章不使用其全强度。

**定义 16.23.** Representable $\Omega[T]$ 的边界 $\partial\Omega[T]$ 是所有 proper face images 的并：
$$
\partial\Omega[T]=\bigcup_{\partial:T'\to T}\operatorname{im}\big(\Omega[T']\to\Omega[T]\big),
$$
其中 $\partial$ 遍历 elementary face maps。

**定义 16.24.** 若 $e$ 是 $T$ 的 inner edge，则 inner horn $\Lambda^e[T]$ 是除去对应 inner face $T/e\to T$ 之外所有 elementary faces 的并：
$$
\Lambda^e[T]=
\bigcup_{\partial:T'\to T,\ \partial\ne \partial_e}
\operatorname{im}\big(\Omega[T']\to\Omega[T]\big).
$$

**说明 16.25.** 下一章的 dendroidal inner Kan 条件就是要求每个 horn inclusion
$$
\Lambda^e[T]\hookrightarrow\Omega[T]
$$
在给定 dendroidal set 中有 fillers。这个条件是 operadic analogue of inner horn filling for quasi-categories。

## 16.6 $\Delta$ 作为线性树子范畴

**定义 16.26.** 定义函子
$$
i:\Delta\to\Omega
$$
把 $[n]$ 送到线性树 $L_n$。

在态射上，$\Delta$ 中的 coface maps 对应删除或收缩线性树中的相应外/内部分，codegeneracy maps 对应删除 unary vertex。

**外部输入定理 16.27.** 上述函子 $i:\Delta\to\Omega$ 是 fully faithful。

由此得到 restriction functor
$$
i^\*:\mathbf{dSet}\to\mathbf{sSet},\qquad
i^\*(X)_n=X_{L_n}.
$$

**命题 16.28.** 若 $\mathcal A$ 是 small category，把它视为只有 unary operations 的 colored operad，并采用
$$
N(\mathcal A)_n=\{x_n\to x_{n-1}\to\cdots\to x_0\}
$$
作为 ordinary nerve 的根向约定，则
$$
i^\*N_d(\mathcal A)
$$
同构于 ordinary simplicial nerve $N(\mathcal A)$。

**证明.** $i^\*N_d(\mathcal A)_n=N_d(\mathcal A)_{L_n}$。按定义这等于
$$
\operatorname{Operad}_{\mathrm{col}}(\Omega(L_n),\mathcal A).
$$
树 $L_n$ 有 $n$ 个 unary 顶点和 $n+1$ 条边。由命题 16.7，给出上述 operad morphism 等价于选择一串对象
$$
x_0,x_1,\ldots,x_n
$$
和一串 morphisms
$$
x_n\to x_{n-1}\to\cdots\to x_0
$$
这正是本命题所采用的 $N(\mathcal A)_n$。Face maps 复合相邻 morphisms 或删除端点，degeneracy maps 插入 identity；这与该根向 nerve 的 simplicial operators 一致。$\square$

**说明 16.29.** 文献中 ordinary nerve 常写成 $x_0\to\cdots\to x_n$。这与本章根向约定相差一个方向选择；必要时可通过取 opposite category 或改变线性树嵌入约定转换。

## 16.7 本章小结

Dendroidal set 是树范畴 $\Omega$ 上的 presheaf。树 $T$ 通过自由 colored operad $\Omega(T)$ 嵌入 operad 理论，dendroidal nerve 由
$$
N_d(\mathcal P)_T=\operatorname{Operad}_{\mathrm{col}}(\Omega(T),\mathcal P)
$$
定义。Corollas 读取具体 operations，单位树读取 colors，Segal core 记录一棵树上每个顶点的局部运算。Strict operads 的 nerve 满足严格 Segal 性；放松边界和 horn 的填充条件，将得到 homotopy coherent operads。

## 练习

**练习 16.1.** 写出 $C_0,C_1,C_2$ 的边、顶点、根边和叶边。

**练习 16.2.** 对有两个顶点和一条 inner edge 的树 $T$，显式描述 $\Omega(T)$ 的颜色和生成运算。

**练习 16.3.** 证明 $N_d(\mathcal P)_{C_0}$ 是 $\mathcal P$ 的 nullary operations 集合。

**练习 16.4.** 画出一个 inner face map 和一个 degeneracy map，并写出其在自由 colored operad 上的作用。

**练习 16.5.** 对一棵三顶点树 $T$，描述 $\operatorname{Sc}[T]$ 并验证命题 16.18 的数据对应。
