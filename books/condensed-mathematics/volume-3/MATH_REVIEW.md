# 第三卷数学审查记录

## 当前状态

第三卷已经完成应用导读草稿，内容包括：

1. 复几何应用的目标和边界。
2. 复解析空间的凝聚语言。
3. 相干层与导出范畴。
4. Dolbeault 复形与 liquid 模。
5. 相干上同调有限性。
6. Serre 对偶。
7. GAGA。
8. Grothendieck-Hirzebruch-Riemann-Roch。
9. 六函子形式与后续方向。
10. 复几何定理证明路线。
11. 经典语言与凝聚语言对照。
12. Stein、Cech 与相干分解。
13. Dolbeault 与 Serre 对偶计算模型。
14. GAGA 与 Riemann-Roch 基础例子。
15. 经典复几何输入定理的精确形式。
16. 复几何主定理的依赖链。
17. $\mathbb P^1$ 上线丛上同调的 Čech 计算。
18. Čech-to-derived 谱序列、acyclic 覆盖定理和超上同调计算。
19. Serre 对偶的链级配对、perfect pairing 与 trace/counit 形式证明层。
20. GAGA 与 Riemann-Roch 的导出范畴、Euler characteristic 和 $K$-理论形式推论。
21. Fredholm-Hodge 有限性的 Hilbert 复形、Hodge decomposition 和 harmonic forms 形式证明层。
22. 有限分解、有限过滤、谱序列有限性传播，以及 Stein-Cech 不能单独推出有限维性的边界说明。
23. Fine sheaf、Cech 同伦、acyclic resolution 和 Dolbeault cohomology 计算 sheaf cohomology 的形式证明。
24. 有限局部自由 resolution 条件下，从向量丛 Serre 对偶推出相干层 Ext-Serre 形式的同调代数证明。
25. Chern character、Todd class、splitting principle、$K$-理论可加性和 HRR 右侧形式代数。
26. GAGA properness 的非 proper 反例、exact equivalence 到 derived equivalence、上同调比较到 $R\Gamma$ 比较。
27. Dolbeault 局部正合的 Cauchy-Green 算子、polydisc 同伦和向量丛系数推导。
28. $\mathbb P^n$ 上 $\mathcal O(d)$ 的 Čech 单项式分解、基础 Bott 公式、Serre 对偶配对和 Euler characteristic。
29. $\mathbb P^n$ 上线丛情形的 canonical bundle、Čech residue trace 和 Serre 对偶完美配对。
30. $\mathbb P^n$ 上线丛情形的 Chern character、Todd class、residue 系数计算和 HRR 公式证明。

## 外部输入定理

当前作为输入使用的定理：

1. Clausen-Scholze 对 compact complex manifolds 的 condensed/analytic 建模。
2. coherent cohomology finite-dimensionality。
3. Serre duality。
4. GAGA。
5. Grothendieck-Hirzebruch-Riemann-Roch。
6. 复几何场景下的 six functor formalism。

## 当前数学口径

本卷不是完整证明版复几何教材。它提供经典表述、凝聚表述、证明策略、依赖关系、术语对照、局部计算模型和例子；深层证明仍以 Clausen-Scholze 讲义和经典复几何教材为准。附录 F-G 已把输入定理和依赖链拆细，附录 I 已把 Čech 谱序列和 acyclic 覆盖计算写成书内证明，附录 J-K 已把 Serre 对偶、GAGA 和 Riemann-Roch 的形式推论写成证明，以便读者精确区分“本书证明的推论”和“外部输入”。

## 教材性不足

若目标是“严格写完一本教材”，第三卷还需要补：

1. Dolbeault lemma 的局部解析骨架已补；Cartan A/B 和相干层有限分解仍需完整预备证明或精确引用。
2. coherent cohomology finite-dimensionality 的 elliptic/Fredholm 证明细节。
3. Serre duality 在线丛射影空间模型中已完整证明；一般相干层中的配对非退化和完美性仍需完整证明。
4. GAGA 中代数化与上同调比较的完整证明链。
5. Riemann-Roch 在线丛射影空间模型中已完整证明；一般情形中 Chern character、Todd class、trace 和 pushforward 相容性仍需证明。
6. 每章练习的完整解答；目前总答案见 [../SOLUTIONS.md](../SOLUTIONS.md)。

## 本轮严格化进展

1. 附录 F 精确列出 Dolbeault resolution、Cartan A/B、有限性、Serre duality、GAGA、HRR 和 Clausen-Scholze 建模的输入形式。
2. 附录 F 明确证明了从 Dolbeault resolution 到 sheaf cohomology 计算、从 Stein acyclicity 到 Čech 计算、以及 $\mathbb P^1$ 上 Riemann-Roch 公式。
3. 附录 G 将 Dolbeault、有限性、Serre duality、GAGA、Riemann-Roch 和六函子展望拆成依赖链。
4. 附录 H 完整计算 $H^0(\mathbb P^1,\mathcal O(d))$、$H^1(\mathbb P^1,\mathcal O(d))$ 和 Euler characteristic。
5. 附录 I 完整证明 Čech-to-derived 谱序列、acyclic 覆盖定理、超上同调计算和 Stein 覆盖到 $R\Gamma$ 的形式推论。
6. 附录 J 完整证明 Serre 对偶中链级配对到上同调配对、perfect pairing 到导出同构、trace/counit 到六函子配对的形式步骤。
7. 附录 K 完整证明 GAGA exact equivalence 到 bounded derived equivalence、$R\Gamma$ 比较、Euler characteristic 和 $\mathbb P^1$ Riemann-Roch 检验的形式步骤。
8. 附录 L 完整证明 Fredholm-Hodge 输入推出 Dolbeault cohomology 有限维的形式步骤。
9. 附录 M 完整证明有限维复形、有限过滤、谱序列和有限 acyclic 分解传播有限性的形式步骤，并明确指出 Stein-Cech 复形项通常无限维，不能单独证明相干上同调有限性。
10. 附录 N 完整证明 fine sheaf 的 Cech 消没、acyclic resolution 定理和 Dolbeault lemma 推出 sheaf cohomology 计算的形式步骤。
11. 附录 O 完整证明在有限局部自由 resolution 假设下，向量丛 Serre duality 如何推出相干层 Ext-Serre duality。
12. 附录 P 完整证明 Chern character 和 Todd class 在接受 Chern 类与 splitting principle 后的加法、乘法和 $\mathbb P^1$ 计算。
13. 附录 Q 完整证明 properness 缺失时 $\mathbb A^1$ 的全局函数比较失败，并补 GAGA 到导出比较的形式步骤。
14. 附录 R 展开 Dolbeault 局部正合的解析骨架：Cauchy-Green 算子、逐变量同伦、带系数局部正合和拓扑连续性边界。
15. 附录 S 完整计算 $\mathbb P^n$ 上 $\mathcal O(d)$ 的 Čech 上同调，并给出基础 Bott 公式、Serre 对偶单项式配对和 Euler characteristic 检验。
16. 附录 T 完整证明 $\mathbb P^n$ 上线丛的 Serre 对偶，包括 canonical bundle、Čech residue trace 和单项式配对矩阵。
17. 附录 U 完整证明 $\mathbb P^n$ 上线丛的 HRR 公式，包括 Todd class 和 residue 系数计算。
18. 仍未完成的是 Cartan A/B、Grauert finiteness、一般 Serre perfectness、GAGA 代数化和一般 HRR 深层输入定理本身的完整证明；这些需要单独的复几何预备卷或大量经典教材内容。
