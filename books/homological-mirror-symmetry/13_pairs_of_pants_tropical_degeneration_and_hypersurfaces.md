# 第十三章：pairs of pants、tropical degeneration 与 hypersurfaces in $(\mathbb C^\ast)^n$

高维 hypersurface 的复杂性往往集中在不同局部模型如何粘合，而不是每个局部模型本身。大复结构极限把 hypersurface 的 amoeba 压向 tropical 骨架；骨架的顶点邻域由高维 pair of pants 描述，边与更高余维面记录这些局部块的交叠。若这些块能提升为 Weinstein sectorial cover，第六、七章的 wrapped 与 stopped 范畴便可沿协变图胶合。本章先把 tropicalization 的组合数据算清，再说明哪些额外辛几何假设才允许把组合覆盖升级为范畴 descent。

## 13.1 Pair of pants

**定义 13.1.** $n$-dimensional pair of pants 是
$$
P^n=\{(z_1,\ldots,z_{n+1})\in(\mathbb C^\ast)^{n+1}\mid z_1+\cdots+z_{n+1}=1\}
$$
或其等价的 toric/open hypersurface 模型。

**例 13.1A（低维 pair of pants）.** 当 $n=1$ 时，投影到第一坐标给出
$$
P^1=\{(x,y)\in(\mathbb C^\ast)^2:x+y=1\}
\xrightarrow{\sim}\mathbb P^1\setminus\{0,1,\infty\},
\qquad (x,y)\longmapsto x.
\tag{13.1}
$$
逆映射为 $x\mapsto(x,1-x)$。条件 $x\ne0$、$1-x\ne0$ 分别删去
$0,1$，而仿射坐标本身删去 $\infty$。因此一维 pair of pants 就是三端
punctured sphere；高维定义保留了“若干圆柱端在一个局部块中相遇”的特征。

**解释 13.2.** 高维 hypersurface 的 large complex structure degeneration 可局部分解为 pairs of pants。A-side 上对应 wrapped 或 partially wrapped pieces；B-side 上对应 toric 或 singular pieces。

## 13.2 Tropical degeneration

**定义 13.3.** 一个 Laurent polynomial family
$$
f_t(z)=\sum_{m\in A} c_m(t)z^m
$$
的 tropicalization 是函数
$$
\operatorname{Trop}(f)(u)=\min_{m\in A}\{v(c_m)+\langle m,u\rangle\}.
$$
其非光滑 locus 给出 tropical hypersurface。

**例 13.3A（tropical line）.** 取平凡赋值且
$f(x,y)=1+x+y$。则
$$
\operatorname{Trop}(f)(u,v)=\min\{0,u,v\}.
$$
非光滑 locus 是最小值至少由两项同时取得的集合，因而等于三条射线
$$
\{u=0\le v\}\ \cup\ \{v=0\le u\}\ \cup\ \{u=v\le0\}.
\tag{13.2}
$$
三条射线交于原点；原点邻域正是 pair-of-pants 顶点模型，射线则记录与
相邻局部块粘合的圆柱端。这个计算也说明 tropicalization 给出组合骨架，
并不自行给出辛形式或 sectorial boundary data。

**解释 13.4.** Tropical hypersurface 编码 degeneration 的组合骨架。Pair-of-pants decomposition 可理解为 tropical hypersurface 的局部顶点模型。

**命题 13.5.** 若一个 degeneration 的 tropical hypersurface 可由标准局部
模型覆盖，该覆盖在 A-side 提升为定义 15.1 的 Weinstein sectorial cover，
且局部 Fukaya models 与这些 inclusions 相容，则全局 category 可由局部
pair-of-pants categories 胶合。

**证明.** 这是定理 15.3 的形式结论。Weinstein sectorial descent 把全局
wrapped Fukaya category 识别为局部 categories 沿交叠的 homotopy colimit。
每个局部 piece 为 pair-of-pants 模型，故全局由它们胶合。证毕。

**警告 13.6.** 命题 13.5 的关键假设是 Weinstein sectorial descent，这是
外部输入；只有有限交存在或 tropical cover 与 Liouville sector 结构兼容，
都还不足以替代定义 15.1 的 Weinstein hypotheses。

## 13.3 Hypersurfaces in $(\mathbb C^\ast)^n$

**外部输入定理 13.7（Abouzaid--Auroux 的 quasi-embedding）.** 对
$(\mathbb C^\ast)^n$ 中 maximally degenerating hypersurface families 及其
mirror toric Landau--Ginzburg A-models，Abouzaid--Auroux 构造 fiberwise
wrapped Fukaya category 和一个 fibered admissible Lagrangian；其 fiberwise
wrapped Floer cohomology 同构于 hypersurface 的正则函数环。由此得到
$$
\mathrm D^b\operatorname{Coh}(H)\longrightarrow
\mathcal W_{\mathrm{fib}}(Y,W)
\tag{13.3}
$$
的 quasi-embedding。该来源的此项结论本身不声称 (13.3) 本质满；若要升级为
Morita equivalence，还须另证目标侧生成性。
来源：Abouzaid--Auroux, *Homological mirror symmetry for hypersurfaces in
$(\mathbb C^\ast)^n$*, arXiv:2111.06543。

**解释 13.8.** 该结果不是 compact Fukaya category 的等价，而是落在适合
该 fibration 的 fiberwise wrapped 范畴中的 quasi-embedding。把目标范畴替换为
普通 wrapped 或 compact Fukaya category，或者删去 maximal degeneration 假设，
都不在定理 13.7 的结论范围内。

## 13.4 Categorical resolution

**外部输入定理 13.9（higher-dimensional pants）.** 设 $P^n$ 为
$\mathbb P^n$ 中 $n+2$ 个一般位置超平面的补，并取 Lekili--Polishchuk
指定的 stops。其 partially wrapped Fukaya category 等价于奇异仿射簇
$$
Z=\{x_1\cdots x_{n+1}=0\}\subset\mathbb A^{n+1}
$$
的 $\mathrm D^b\operatorname{Coh}(Z)$ 的一个 categorical resolution；移除
相应 stops 后，fully wrapped Fukaya category 等价于
$\mathrm D^b\operatorname{Coh}(Z)$。该结果也有有限 Abelian covers 的版本。
来源：Lekili--Polishchuk, *Homological mirror symmetry for higher
dimensional pairs of pants*, arXiv:1811.04264。

**解释 13.10.** Pair-of-pants 模型把 singular B-side 和 stopped/wrapped A-side 直接联系起来。Stop removal 对应 categorical resolution 到 singular category 的 localization。

## 13.5 相容局部镜像数据

**定义 13.11（相容局部镜像数据）.** 对一个 tropical/sectorial 分解，称
下列六项为一组相容局部镜像数据：

1. 选取 hypersurface degeneration 和 tropical skeleton。
2. 把 A-side Liouville space 切成 sectorial pieces。
3. 对每个 piece 识别其 wrapped category。
4. 在 B-side 构造相应 affine/toric cover 或 categorical resolution。
5. 比较两边 descent diagrams。
6. 取 homotopy colimit 或 Morita colimit 得到全局 HMS。

**命题 13.12.** 若定义 13.11 中每个局部比较为 Morita equivalence，且两边 descent diagrams 在 Morita homotopy category 中相容，则全局 categories Morita equivalent。

**证明.** Morita equivalences 在 homotopy colimit 下保持：若两个 diagrams 由逐点等价和相容自然变换连接，则它们的 homotopy colimits 等价。全局 categories 由 descent 识别为这些 colimits，因此得到结论。证毕。

Tropical 骨架只告诉我们局部块如何相遇；从它到全局 Fukaya 范畴还必须验证 Weinstein sectorial 条件和所有 inclusion functors 的相容性。满足这些条件后，pair-of-pants categories 组成的 Cech 型图才可通过 homotopy colimit 重建全局对象。第十五章将把这里调用的 descent 从证明策略提升为精确的外部输入定理。

## 练习

**练习 13.1.** 写出 $P^1$ 与三点去除球面的同构关系。

**练习 13.2.** 计算二项式和三项式 Laurent polynomial 的 tropical hypersurface。

**练习 13.3.** 解释为什么 sectorial descent 是命题 13.5 的非形式输入。

**练习 13.4.** 为两个 pair-of-pants pieces 沿一个圆柱端的粘合写出定义
13.11 的两个范畴图，并标出命题 13.12 所需的自然相容性。
