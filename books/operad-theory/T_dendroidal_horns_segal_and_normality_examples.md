# 附录 T：Dendroidal horns、Segal core 与 normality 样例

本附录补充第十六、十七章。目标是把 dendroidal set 中的 face、horn、Segal core 和 normality 条件落实到小树上。Dendroidal nerve fully faithfulness 和 strict nerve unique fillers 已由 MW-2、MW-4 定位；完整 Cisinski-Moerdijk 模型结构、normal monomorphism 引理和 erratum 影响仍作为外部输入。

## T.1 两顶点树

令 $T$ 是由两个 corollas 沿一条 inner edge $e$ 粘合得到的树。记下方顶点为 $v$，上方顶点为 $w$，$v$ 的输出边为 $e$，$e$ 是 $w$ 的一个输入边。

若 $v$ 有输入集合 $A$，$w$ 的其他输入集合为 $B$，则 $T$ 的外部输入为
$$
A\amalg B
$$
输出为 $w$ 的输出边。

**inner face.** 收缩 $e$ 得到一个 corolla，其输入集合为 $A\amalg B$。该 face 对应 operad 中的复合：
$$
\theta_w\circ_e\theta_v.
$$

**outer faces.** 删除 $v$ 或删除 $w$ 的外侧部分给出只保留另一个顶点或相应边颜色的 faces。具体可删性取决于 $v,w$ 是否位于树的外侧；两顶点树中它们都是外侧顶点，故有两个主要 outer face。

**命题 T.1.** 对 strict colored operad $\mathcal P$，一个 inner horn
$$
\Lambda^e[T]\to N_d(\mathcal P)
$$
唯一确定两个顶点运算以及共享边颜色。

**证明.** Horn 包含除收缩 $e$ 之外的所有 elementary faces，特别包含两个顶点 corolla 的数据和它们在共享边 $\eta$ 上的限制。由命题 16.15 和 16.16，corolla 数据正是顶点运算，$\eta$-restriction 正是颜色。兼容性要求 $v$ 的输出颜色等于 $w$ 对应输入颜色。$\square$

**命题 T.2.** 命题 T.1 的 horn 在 $N_d(\mathcal P)$ 中的唯一 filler 的缺失 inner face 是复合运算。

**证明.** 由 strict Segal 性，给出 $\Omega[T]\to N_d(\mathcal P)$ 等价于给出每条边颜色和每个顶点运算。命题 T.1 已给出这些数据，因此存在唯一 filler。将 filler 限制到 inner face $T/e$，得到由 $\Omega(T/e)$ 的唯一顶点表示的运算；该运算按 $\Omega(T)$ 的自由 colored operad 复合定义，正是 $\theta_w\circ_e\theta_v$ 在 $\mathcal P$ 中的像。$\square$

## T.2 三顶点线性树与 simplicial inner horn

令 $L_3$ 为三顶点线性树：
$$
x_3\to x_2\to x_1\to x_0.
$$
其两条 inner edges 对应 ordinary simplex $\Delta[3]$ 中的 inner horns $\Lambda^1[3]$ 与 $\Lambda^2[3]$，方向依赖定义 16.1--定义 16.8 的根向 convention。

**命题 T.3.** 若 $X$ 是 dendroidal inner Kan，则 $i^\*X$ 中的 $\Lambda^1[3]$ 和 $\Lambda^2[3]$ horns 有 fillers。

**证明.** 线性树嵌入 $i:\Delta\to\Omega$ 把 simplicial inner horn 识别为线性树的 dendroidal inner horn。$X$ inner Kan 给出 dendroidal filler；限制到 $\Delta$ 后得到 simplicial filler。$\square$

**说明 T.4.** 当 $X=N_d(\mathcal C)$ 来自 ordinary category $\mathcal C$ 时，filler 唯一，表示三箭头串中缺失复合由范畴复合唯一确定。一般 dendroidal inner Kan object 只保证存在。

## T.3 Segal core 的两顶点计算

对 T.1 的两顶点树，Segal core
$$
\operatorname{Sc}[T]\subset\Omega[T]
$$
由两个顶点 corollas 的像生成，并沿共享 inner edge 的颜色 face 粘合。

**命题 T.5.** 对 strict operad nerve $N_d(\mathcal P)$，
$$
\mathbf{dSet}(\operatorname{Sc}[T],N_d(\mathcal P))
$$
等于所有可复合二元顶点数据的集合。

**证明.** 一个从 Segal core 到 nerve 的映射等价于给每个顶点 corolla 一个 $\mathcal P$-运算，并要求它们在共同边的 $\eta$-restriction 上给出同一颜色。该条件正是两个运算可复合。$\square$

**推论 T.6.** 限制映射
$$
N_d(\mathcal P)_T\to
\mathbf{dSet}(\operatorname{Sc}[T],N_d(\mathcal P))
$$
是双射。

**证明.** 这是命题 16.18 在两顶点树上的实例。也可直接由 $\Omega(T)$ 的自由性证明：一个 morphism $\Omega(T)\to\mathcal P$ 由边颜色和顶点运算唯一决定。$\square$

## T.4 边界与 inner horn 的差异

对有 inner edge $e$ 的树 $T$，边界
$$
\partial\Omega[T]
$$
包含所有 elementary faces；inner horn
$$
\Lambda^e[T]
$$
缺少收缩 $e$ 的 face。

**命题 T.7.** 对两顶点树，给出边界映射
$$
\partial\Omega[T]\to N_d(\mathcal P)
$$
不仅给出两个可复合顶点运算，还给出它们的复合并要求兼容。

**证明.** 边界包含 inner face $T/e\to T$。映射到 $N_d(\mathcal P)$ 后，该 face 给出一个单顶点 corolla 上的运算，即候选复合。边界还包含两个顶点 corollas；自然性要求候选复合与两个顶点运算在 $\Omega$ 的 face 关系下相容。对 strict nerve，这正是候选复合等于 operad 复合。$\square$

**说明 T.8.** Inner horn 的作用是“给定可复合局部数据，要求存在复合”；边界的作用是“给定局部数据和复合，要求它们兼容”。二者不可混用。

## T.5 Corolla automorphisms 与 normality

令 $C_n$ 为 $n$-corolla。若不标号输入边，则
$$
\operatorname{Aut}_\Omega(C_n)\cong\Sigma_n.
$$

**例 T.9.** Representable $\Omega[C_n]$ 的顶维 dendrex $\operatorname{id}_{C_n}$ 有 $\Sigma_n$-作用。若一个 dendroidal set $X$ 中存在非退化 $C_n$-dendrex $x$，且某个非平凡 $\sigma\in\Sigma_n$ 固定 $x$，则 $X$ 不是 normal。

**说明 T.10.** Normality 不是说树没有自同构，而是说自同构不能固定新增的非退化 dendrex。这个条件用于让 cofibrations 类似“带自由对称群作用的胞腔附加”。

**外部输入命题 T.11.** Inner horn inclusion
$$
\Lambda^e[T]\hookrightarrow\Omega[T]
$$
是 normal monomorphism。

**证明路线（外部输入）.** Monomorphism 部分逐树可由 horn 是 representable 的 subpresheaf 检查。Normality 需要分析未包含 face 的非退化 dendrexes 的 automorphism stabilizers；该步骤依赖 $\Omega$ 的 elementary face/degeneracy 分解以及 CM-1--CM-2，本书不重证。

## T.6 Degeneracy 的低阶例子

令 $L_1$ 是一个 unary 顶点的线性树。Degeneracy map
$$
\sigma:L_1\to\eta
$$
在 presheaf 方向给出
$$
X(\sigma):X_\eta\to X_{L_1}
$$
并把一个 $\eta$-dendrex 送到相应的 degenerate unary dendrex。

**说明 T.12.** 对 category nerve，这正是 simplicial degeneracy $s_0:X_0\to X_1$，即把对象送到它的 identity arrow。对 operad nerve，unary identity operation 是 degeneracy 的来源。

**命题 T.13.** 在 strict operad nerve $N_d(\mathcal P)$ 中，degenerate unary dendrex 对应 identity unary operation。

**证明.** $\eta$-dendrex 只记录一个颜色，即 colored operad morphism $\Omega(\eta)\to\mathcal P$。树箭头 $\sigma:L_1\to\eta$ 按定义对应 operad morphism
$$
\Omega(L_1)\longrightarrow\Omega(\eta),
$$
它把 $L_1$ 的 unary 生成运算送到 $\Omega(\eta)$ 唯一颜色上的 identity operation。预复合给出
$$
\Omega(L_1)\to\Omega(\eta)\to\mathcal P,
$$
所以所得 $L_1$-dendrex 的 unary 运算正是颜色 $c$ 上的单位运算。$\square$

## T.7 使用检查表

使用 dendroidal horns 时必须说明：

1. 树 $T$ 的 inner edge 是哪一条；
2. horn 缺少的是哪个 face；
3. 目标是 strict nerve、inner Kan dendroidal set，还是某个 fibrant replacement；
4. filler 是唯一、存在，还是 contractible choice；
5. 是否使用 normal monomorphism 或 operadic model structure；
6. 是否通过线性树限制回 quasi-category。

## T.8 小结

两顶点树的 inner horn 是 operadic composition 的最小模型。Segal core 记录顶点级数据，inner face 记录复合，boundary 同时记录局部数据和复合。Strict operad nerve 中 fillers 唯一；一般 dendroidal infinity-operad 中只要求 fillers 存在。Normality 则控制树自同构对非退化 dendrexes 的稳定子。
