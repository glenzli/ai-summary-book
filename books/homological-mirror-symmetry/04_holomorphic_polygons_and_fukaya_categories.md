# 第四章：holomorphic polygon、$A_\infty$ 结构与 Fukaya category

## 本章目标

本章在 exact、横截、regular 的入口假设下定义 Fukaya category 的高阶复合，并说明 $A_\infty$ 方程来自一维 holomorphic polygon 模空间的边界分解。一般 compact 或 obstructed Fukaya category 的构造不在本章内部完成。

## 依赖前置知识

需要第一章的 $A_\infty$ 范畴语言和第三章的 exact Floer cochains。需要知道 Riemann surface、almost complex structure 和 Cauchy-Riemann 方程的基本形式。

## 4.1 带边界标记点的圆盘

**定义 4.1.** 对 $d\ge1$，记 $\mathcal R^{d+1}$ 为带有 $d+1$ 个按逆时针顺序排列的边界 punctures
$$
\zeta_0,\zeta_1,\ldots,\zeta_d
$$
的圆盘复结构模空间，其中 $\zeta_0$ 作为输出点，$\zeta_1,\ldots,\zeta_d$ 作为输入点。每两个相邻 punctures 之间的边界弧记为 $I_0,\ldots,I_d$。

**事实 4.2.** $\mathcal R^{d+1}$ 的实维数为 $d-2$。当 $d=1$ 时，它对应 strip；当 $d=2$ 时，它对应三角形，模空间为零维。

**解释 4.3.** 维数 $d-2$ 是圆盘边界上 $d+1$ 个有序点的参数数目减去 $\operatorname{PSL}(2,\mathbb R)$ 的三维自同构群。严格处理需要稳定曲线紧化；本章只使用其边界分解形式。

## 4.2 Polygon 模空间

**定义 4.4.** 设 $\mathbb L_0,\ldots,\mathbb L_d$ 是 exact Lagrangian branes，且相邻对横截。给定输入交点
$$
x_i\in L_{i-1}\cap L_i,\qquad 1\le i\le d,
$$
和输出交点 $x_0\in L_0\cap L_d$，一个 holomorphic polygon 是映射
$$
u:S\to M,\qquad S\in\mathcal R^{d+1},
$$
满足：

1. Cauchy-Riemann 方程 $(du)^{0,1}=0$，或扰动版本；
2. $u(I_i)\subset L_i$；
3. 在 puncture $\zeta_i$ 附近渐近于 $x_i$。

相应模空间记为
$$
\mathcal M(x_0;x_d,\ldots,x_1).
$$

**定义 4.5.** 若 regularity 已成立，模空间的零维部分带有由 brane data 诱导的 orientation。其带符号计数记为
$$
n(x_0;x_d,\ldots,x_1)\in k.
$$

**警告 4.6.** regularity、orientation、compactness 和 gluing 不是形式代数事实。本章后续所有关于模空间计数的结论都依赖外部分析输入，来源主要为 Seidel exact 口径和 FOOO 一般口径。

## 4.3 高阶复合

**定义 4.7.** 在 exact regular 假设下，Fukaya category 的高阶复合在交点生成元上定义为
$$
\mu^d(x_d,\ldots,x_1)
=\sum_{x_0} n(x_0;x_d,\ldots,x_1)\,x_0,
$$
并按局部系统、orientation lines 和 Koszul 符号线性延拓到
$$
\mu^d:
CF^\ast(\mathbb L_{d-1},\mathbb L_d)\otimes\cdots\otimes
CF^\ast(\mathbb L_0,\mathbb L_1)
\to
CF^\ast(\mathbb L_0,\mathbb L_d)[2-d].
$$

**例 4.8.** 当 $d=1$ 时，$\mu^1$ 是第三章的 Floer 微分。当 $d=2$ 时，$\mu^2$ 由 holomorphic triangles 计数给出，它在 Floer cohomology 上诱导乘法。

**外部输入定理 4.9（Fukaya $A_\infty$ 方程，exact 入口）.** 在 exact、横截、regular、compactness 和 orientation 假设下，运算 $\{\mu^d\}_{d\ge1}$ 满足第一章定义的 $A_\infty$ 方程。

**证明草图.** 固定输入和输出，使相应 polygon 模空间的虚维数为 $1$。其紧化边界由 broken configurations 组成：一个 polygon 的某段连续输入先由 $\mu^s$ 合成，再作为另一个 polygon 的输入。这些边界分量与 $A_\infty$ 方程中的各项一一对应。带符号边界计数为零，因此这些项的代数和为零。exactness 用于排除 disk bubbling；regularity 和 gluing 保证紧化边界没有未计入的额外分量。证毕。

**推论 4.10.** 在定理 4.9 的假设下，exact Lagrangian branes 与 Floer cochains 构成一个 $A_\infty$ category。

**证明.** 对象取 branes，morphism spaces 取 $CF^\ast$，高阶复合取定义 4.7。$A_\infty$ 方程由定理 4.9 给出。证毕。

## 4.4 Exact Fukaya category

**定义 4.11.** 在固定 exact symplectic manifold $(M,\lambda)$ 上，exact Fukaya category $\mathcal F(M)$ 是如下 $A_\infty$ category：

1. 对象是满足给定 admissibility 条件的 exact Lagrangian branes；
2. morphism spaces 是 Floer cochain complexes $CF^\ast(\mathbb L_0,\mathbb L_1)$；
3. 高阶复合由 holomorphic polygons 的带符号计数给出；
4. 所有辅助 perturbation data 按 coherent choices 选择，使 $A_\infty$ 方程成立。

**外部输入定理 4.12（选择独立性）.** 在 exact 口径下，不同 coherent perturbation data 得到 quasi-equivalent 的 Fukaya categories。

**解释 4.13.** 定义 4.11 不是单纯的集合论定义：它压缩了大量分析选择。定理 4.12 保证压缩后的对象可以作为几何不变量。若没有选择独立性，HMS 的 A-side 会依赖非几何辅助数据。

## 4.5 单位与三角闭包

**定义 4.14.** Fukaya category 通常先得到 cohomologically unital $A_\infty$ category：对每个对象 $\mathbb L$，$HF^\ast(\mathbb L,\mathbb L)$ 中有单位元，使 $H^\ast\mathcal F(M)$ 成为含单位范畴。若通过模型替换得到严格单位，则称为 strictly unital Fukaya model。

**外部输入定理 4.15（strictification）.** 在标准 $A_\infty$ 同伦代数假设下，cohomologically unital $A_\infty$ category 可替换为 quasi-equivalent 的 strictly unital model。  
来源：$A_\infty$ homological algebra，Lefevre-Hasegawa/Keller 口径。

**定义 4.16.** Fukaya category 的 split-closed derived category 记为
$$
D^\pi\mathcal F(M):=H^0\operatorname{Perf}(\mathcal F(M)).
$$
这是许多早期 HMS 文献中“derived Fukaya category”的精确版本之一。

**警告 4.17.** 记号 $D^\pi\mathcal F(M)$ 在文献中有变体。有些作者使用 twisted complexes 的 split-closure，有些作者使用 perfect modules。本书在每个定理中说明采用哪一种，但 Morita 口径下这些模型通常表达同一类 idempotent-complete triangulated envelope。

## 4.6 与 wrapped 版本的边界

**定义 4.18.** 若 $M$ 非紧并带 Liouville 结构，wrapped Fukaya category $\mathcal W(M)$ 允许非紧 Lagrangians，并用 Hamiltonian wrapping 产生 morphisms。其 morphism cochains 由 Hamiltonian chords 而非单纯交点生成。

**警告 4.19.** $\mathcal W(M)$ 不是 $\mathcal F(M)$ 的形式小修改。wrapped 版本需要控制无穷远处的 Reeb dynamics、Hamiltonian 增长、sectorial boundary、stops 和 continuation maps。后续章节将使用 Ganatra-Pardon-Shende 的 Liouville sectors 与 descent 作为外部输入。

## 本章小结

Fukaya category 的 $A_\infty$ 运算来自 holomorphic polygons 的计数。$A_\infty$ 方程是 compactified one-dimensional moduli spaces 的边界计数为零。exact 假设使 bubbling 受到控制，但 regularity、orientation、gluing 和选择独立性仍是外部分析输入。HMS 中常用的是 $\mathcal F(M)$ 的预三角化和 split-closed/Morita 完备化。

## 练习

**练习 4.1.** 对 $d=1,2,3$，分别说明 $\mathcal R^{d+1}$ 的几何含义。

**练习 4.2.** 将 $d=2$ 的 $\mu^2$ 与 Floer cohomology 上的乘法联系起来，并说明为什么链级乘法只按 $A_\infty$ 意义结合。

**练习 4.3.** 写出一维 polygon 模空间的 broken boundary 如何对应 $A_\infty$ 方程中的一项。

**练习 4.4.** 解释 wrapped Fukaya category 中为什么 morphism 生成元应改为 Hamiltonian chords。
