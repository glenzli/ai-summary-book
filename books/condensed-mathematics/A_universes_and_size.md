# 附录 A：集合论宇宙与小性约定

## 本附录目标

凝聚数学很容易遇到大小问题：所有紧 Hausdorff 空间构成的范畴不是普通意义下的小范畴，所有 sheaf 的集合也需要 universe 控制。本附录给出本书采用的处理方式。

## A.1 Grothendieck universe

**定义 A.1.** 一个 Grothendieck universe $\mathcal U$ 是集合论中的一个集合，满足以下封闭性：

1. 若 $x\in y$ 且 $y\in\mathcal U$，则 $x\in\mathcal U$。
2. 若 $x,y\in\mathcal U$，则 $\{x,y\}\in\mathcal U$。
3. 若 $x\in\mathcal U$，则 $\mathcal P(x)\in\mathcal U$。
4. 若 $I\in\mathcal U$ 且 $x_i\in\mathcal U$ 对所有 $i\in I$ 成立，则 $\bigcup_{i\in I}x_i\in\mathcal U$。

属于 $\mathcal U$ 的集合称为 $\mathcal U$-小集合。

本书固定一个 Grothendieck universe $\mathcal U$，并默认所有集合、拓扑空间、范畴对象都在 $\mathcal U$ 内，除非特别说明。

## A.2 小范畴与大范畴

**定义 A.2.** 一个范畴 $\mathcal C$ 称为 $\mathcal U$-小，如果其对象集和所有态射集都属于 $\mathcal U$。

若对象总体不属于 $\mathcal U$，但每个 Hom 是 $\mathcal U$-小集合，则称其为局部小范畴。

严格地说，$\mathbf{CHaus}$ 作为所有 $\mathcal U$-小紧 Hausdorff 空间构成的范畴，其对象类可能不属于 $\mathcal U$。为了避免每章处理这件事，本书采用以下约定：

- 固定一个足够大的 universe $\mathcal V\supset\mathcal U$。
- $\mathbf{CHaus}$ 的对象为 $\mathcal U$-小紧 Hausdorff 空间。
- sheaf 范畴在 $\mathcal V$ 中形成。

这样，本书中的 Hom 集、极限、余极限和 sheafification 都在一个更大的 universe 内存在。

## A.3 为什么这不改变数学内容

凝聚数学常有 $\kappa$-small 版本：选取某个强不可达基数 $\kappa$，只允许基数小于 $\kappa$ 的测试对象。改变 $\kappa$ 会影响技术细节，但在许多应用中可通过提升 universe 或比较定理控制。

本书第一卷不研究这些 set-theoretic 细节，只采用固定 universe 约定。每当证明中出现“取所有极不连通空间的代表集合”这类说法，严格含义都是：在选定 universe 内取一组同构代表。

## A.4 代表集合

**命题 A.3.** 在固定 universe 约定下，$\mathcal U$-小紧 Hausdorff 空间的同构类可由某个 $\mathcal V$-小集合表示。

**证明说明.** 每个 $\mathcal U$-小拓扑空间由一个 $\mathcal U$-小底层集合及其拓扑给出。底层集合和其幂集都在 $\mathcal U$ 或 $\mathcal V$ 控制下，因此所有拓扑结构构成 $\mathcal V$-小集合。取满足紧 Hausdorff 条件的子集合，再按同构关系选代表。证毕。

这保证第七章和第十章中“对所有 $(E,a)$ 取直和”的说法可以在固定 universe 下解释。

## A.5 本附录小结

本书默认使用 Grothendieck universe 控制大小问题。读者在第一次阅读时可以把这当作技术背景；但在证明“有足够多投射对象”或“sheaf 范畴存在所有余极限”时，必须知道这些构造发生在哪个 universe 中。

## 练习

**练习 A.1.** 说明为什么所有 $\mathcal U$-小集合构成的“范畴”通常不是 $\mathcal U$-小范畴。

**练习 A.2.** 检查第七章定理 7.7 中取直和的指标集如何依赖 universe 约定。

**练习 A.3.** 查阅 $\kappa$-condensed sets 的定义，比较它与本书固定 universe 口径的差异。
