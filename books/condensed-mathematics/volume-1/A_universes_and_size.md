# 附录 A：集合论层级、基数截断与小性

## 本附录目标

“所有 profinite 空间上的 sheaf”不是一个不加限制便良定义的定义：测试范畴不是小范畴，
而 solid/analytic 定义中的“对所有测试对象取 cone”也必须是集合指标，而不能是 proper
class 指标。本附录固定本书实际使用的层级，并区分固定层级的教材范畴与不依赖截断的
大范畴。

## A.1 Universe 与基数数据

**定义 A.1.** Grothendieck universe \(\mathcal U\) 是满足下列条件的集合：

1. 若 \(x\in y\in\mathcal U\)，则 \(x\in\mathcal U\)；
2. 若 \(x,y\in\mathcal U\)，则 \(\{x,y\}\in\mathcal U\)；
3. 若 \(x\in\mathcal U\)，则 \(\mathcal P(x)\in\mathcal U\)；
4. 若 \(I\in\mathcal U\) 且 \(x_i\in\mathcal U\) 对每个 \(i\in I\) 成立，
   则 \(\bigcup_{i\in I}x_i\in\mathcal U\)。

元素属于 \(\mathcal U\) 的集合称为 \(\mathcal U\)-小集合。

**约定 A.2（本书的工作层级）.** 固定以下数据：

1. 一个不可数强极限基数 \(\kappa\)，即对每个 \(\lambda<\kappa\) 都有
   \(2^\lambda<\kappa\)；
2. 两个 Grothendieck universes \(\mathcal U\in\mathcal V\)，并要求
   \(\kappa\in\mathcal U\)；
3. \(\mathcal V\) 中选定的 \(\kappa\)-小测试范畴骨架。

记 \(\mathbf{CHaus}_\kappa\)、\(\mathbf{ProFin}_\kappa\) 和
\(\mathbf{ED}_\kappa\) 分别为底层集合基数严格小于 \(\kappa\) 的 compact
Hausdorff、profinite 和 extremally disconnected 空间的选定骨架。旧章节中的
下标 \(\mathcal U\) 与本附录中的下标 \(\kappa\) 指同一固定工作层级；没有下标时也
默认这个层级。

sheaf 取值于 \(\mathbf{Set}_{\mathcal U}\) 或
\(\mathbf{Ab}_{\mathcal U}\)，而预层范畴、sheaf 范畴及其函子在
\(\mathcal V\) 中讨论。于是：

| 表达式 | 类型与大小 |
| --- | --- |
| \(S\in\mathbf{ProFin}_\kappa\) | \(S\) 的底层集合基数 \(<\kappa\) |
| \(X(S)\) | \(\mathcal U\)-小集合 |
| “对所有 \(S\) 取直和或 cone” | 对选定骨架的对象集取集合指标构造 |
| \(\mathbf{CondSet}_\kappa\) | \(\operatorname{Sh}(\mathbf{CHaus}_\kappa;\mathbf{Set}_{\mathcal U})\) |

这些约定只控制大小，不把不同 \(\kappa\) 的范畴直接宣称为相等。

## A.2 为什么要求强极限

**命题 A.3.** 若 \(X\) 是底层集合基数 \(<\kappa\) 的 compact Hausdorff 空间，
则存在 \(E\in\mathbf{ED}_\kappa\) 及连续满射 \(E\twoheadrightarrow X\)。

**证明.** 把 \(|X|\) 赋予离散拓扑并取 Stone--Čech 紧化
\(E=\beta(|X|_{\mathrm{disc}})\)。离散集上的恒等集合映射连续地延拓为
\(E\to X\)，且因其像包含 \(X\) 的每个点而满射。

为补足 extremal disconnectedness，令 \(D=|X|\)。空间 \(\beta D\) 是完备 Boolean
代数 \(\mathcal P(D)\) 的 Stone 空间；其 clopen 基记为
\(\widehat A\)、\(A\subseteq D\)。任一开集可写成

$$
U=\bigcup_{j\in J}\widehat{A_j}.
$$

令 \(A=\bigcup_jA_j\)，则 \(\overline U=\widehat A\)。确实，左到右的包含由
\(U\subseteq\widehat A\) 得到；反之，若 ultrafilter \(x\in\widehat A\)，则其任一
clopen 邻域 \(\widehat B\) 满足 \(A\cap B\ne\varnothing\)。由于

$$
A\cap B=\bigcup_j(A_j\cap B),
$$

某个 \(A_j\cap B\ne\varnothing\)，故 \(\widehat B\cap U\ne\varnothing\)。所以
\(x\in\overline U\)。于是每个开集的闭包仍 clopen，\(E\) 极不连通。

最后，\(E\) 的点可由 \(D\) 上的 ultrafilter 表示，所以

$$
|E|\le 2^{2^{|X|}}<\kappa,
$$

最后一个不等式连续两次使用强极限条件。故 \(E\) 仍在固定测试层级内。证毕。

这个基数估计是 CHaus、ProFin 与 ED 三种站点比较保持在同一层级内的关键；只写
“提升 universe”不能替代它。

## A.3 骨架与集合指标构造

**命题 A.4.** \(\kappa\)-小 compact Hausdorff 空间的同构类有一个
\(\mathcal U\)-小代表集；同样结论适用于 profinite 与 extremally disconnected
子类。

**证明.** 每个基数小于 \(\kappa\) 的集合都与某个序数 \(\alpha<\kappa\) 等势。
在固定底层集合 \(\alpha\) 上，一个拓扑是 \(\mathcal P(\mathcal P(\alpha))\)
的元素。由 \(\kappa\in\mathcal U\) 以及 \(\mathcal U\) 对幂集和
\(\mathcal U\)-小并封闭，所有这些拓扑的总集合属于 \(\mathcal U\)。取满足
compact Hausdorff 条件的子集，并在 ambient ZFC 的选择公理下从每个同构类选一个
代表。ProFin 与 ED 由性质截取子集即可。证毕。

因此第七章的生成元直和、solid 的 Dirac-to-measure cones 以及 analytic ring 的
测试对象都由集合而非 proper class 指标。

## A.4 固定层级与大凝聚范畴

**外部输入定理 A.5（改变截断；Scholze）.** 设
\(\kappa<\kappa'\) 都是不可数强极限基数。限制函子

$$
\mathbf{CondSet}_{\kappa'}\longrightarrow\mathbf{CondSet}_\kappa
$$

有由左 Kan 延拓后 sheafification 给出的左伴随
\(i_{\kappa,\kappa'}\)。该左伴随全忠实，保持所有余极限，并保持所有指标范畴
基数小于 \(\operatorname{cf}(\kappa')\) 的极限，特别保持有限极限。不依赖单个
截断的“大凝聚集合范畴”定义为这些全忠实过渡函子的滤过 2-余极限。

**来源与边界.** 这是 S26 Proposition 2.9 与 Definition 2.11 的本书版本。本书不重证
左 Kan 延拓的滤过性估计，也不把“改变 \(\kappa\)”当作恒等操作。四卷正文在固定
\(\kappa\) 层级工作；只有明确引用 A.5 时，结论才被送入更大的层级或大凝聚范畴。

**反例边界 A.6.** 从 \(\kappa'\)-层级忘到 \(\kappa\)-层级会丢失大测试对象上的
原始取值。A.5 断言的是存在全忠实的反向嵌入及其保持性质，不是两个 sheaf 范畴逐对象
相等，也不是任意大小极限都与该嵌入交换。

## A.5 本附录小结

本书所有无下标的 condensed、solid、analytic 与 liquid 范畴均在固定
\(\kappa\) 层级解释。测试对象已取小骨架，sheaf 值和 Hom 集有明确 universe，
跨层级使用则单独引用输入定理 A.5。

## 练习

**练习 A.1.** 证明所有 \(\kappa\)-小集合构成的范畴通常不属于它自身所对应的
小性层级。

**练习 A.2.** 检查第七章生成元直和的指标集如何由命题 A.4 控制。

**练习 A.3.** 在命题 A.3 中指出强极限条件被使用两次的位置。

**练习 A.4.** 说明 A.5 只保证小于 \(\operatorname{cf}(\kappa')\) 的极限相容，
为什么这不足以无条件交换任意乘积。
