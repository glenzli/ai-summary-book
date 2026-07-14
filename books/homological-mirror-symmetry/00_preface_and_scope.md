# 序章：范围、严格性标准和 HMS 的数学形态

“镜像的两个空间具有等价的导出范畴”是一句有力量却不够完整的话：它没有说明导出范畴采用哪个增强，也没有说明 A-side 的伪全纯曲线计数在哪个系数和紧致性口径下成立。遗漏这些数据之后，Hochschild 不变量、Fourier--Mukai 核与高阶复合甚至未必还有可比较的类型。本序章从这个缺口出发，固定全书的宇宙、基域、增强层级与外部输入责任，并解释为何后文必须先建立 dg/$A_\infty$ 语言，再谈具体镜像等价。读者只需预先熟悉复形、基本范畴论、微分形式和光滑流形；辛几何与导出几何中真正使用的结构会在相应章节引入。

## 0.1 本书的对象

**约定 0.1.** 本书固定 Grothendieck universes
$$
\mathcal U\in\mathcal V\in\mathcal W
$$
和一个基域 $k$。未特别说明时，所有线性范畴、dg category、$A_\infty$-category、复形和张量积均取在 $k$ 上。

**约定 0.2.** “HMS 断言”不是一个裸等式，而是一组数据和一个等价命题。数据至少包括：

1. A-side 几何对象，例如 exact symplectic manifold、Liouville manifold、Liouville sector、Landau-Ginzburg 模型或带 stop 的空间；
2. B-side 几何对象，例如光滑适当代数簇、stack、Landau-Ginzburg potential 或奇点；
3. 系数和分次约定，例如 $k$、Novikov field、grading、brane data；
4. A-side 的增强范畴，例如 $\mathcal F(M)$、$\mathcal W(M)$、partially wrapped category 或 Fukaya-Seidel category；
5. B-side 的增强范畴，例如 $\operatorname{Perf}(X)$、$\mathrm D^b\operatorname{Coh}(X)$ 的 dg enhancement 或 $\operatorname{MF}(X,W)$；
6. 等价类型，例如 quasi-equivalence、Morita equivalence 或 stable $\infty$-category equivalence。

**定义 0.3.** 一个 HMS 命题称为“增强明确的”，若它写成如下形式之一：
$$
\mathcal A\simeq \mathcal B,\qquad
\operatorname{Perf}(\mathcal A)\simeq \operatorname{Perf}(\mathcal B),\qquad
\mathcal A_{\infty}\simeq \mathcal B_{\infty},
$$
其中 $\mathcal A,\mathcal B$ 是 dg 或 $A_\infty$ category，或相应的 stable $\infty$-categories，并且正文说明 $\simeq$ 的含义。若只写
$$
H^0(\mathcal A)\cong H^0(\mathcal B)
$$
或两个三角范畴等价，则称为“三角影子层面的 HMS 陈述”。

**约定 0.4.** 本书把增强明确的 HMS 命题作为标准形态。三角影子层面的陈述可以作为必要结果或历史表述，但不作为最终版本。

## 0.2 为什么必须使用增强

Fukaya 范畴天然带有高阶复合。若 $L_0,\ldots,L_d$ 是合适的 Lagrangian branes，则 holomorphic polygons 的计数给出映射
$$
\mu^d:
CF^\ast(L_{d-1},L_d)\otimes\cdots\otimes CF^\ast(L_0,L_1)
\longrightarrow
CF^\ast(L_0,L_d)[2-d].
$$
这些 $\mu^d$ 满足 $A_\infty$ 方程，而一般不只满足普通范畴的结合律。另一方面，B-side 的导出范畴通常有 dg enhancement 或 stable $\infty$ enhancement；Fourier-Mukai kernel、Hochschild invariants 和 Morita 理论都依赖增强，而不是只依赖三角范畴的同构类。

**命题 0.5.** 若一个 HMS 等价为 dg 或 $A_\infty$ quasi-equivalence
$$
F:\mathcal A\to \mathcal B,
$$
则它诱导普通范畴等价
$$
H^0(F):H^0(\mathcal A)\xrightarrow{\sim}H^0(\mathcal B).
$$

**证明.** quasi-equivalence 的定义包含两部分。第一，任意对象 $X,Y\in\mathcal A$，映射复形
$$
\operatorname{hom}_{\mathcal A}(X,Y)\to
\operatorname{hom}_{\mathcal B}(FX,FY)
$$
是 quasi-isomorphism。因此取零次上同调后得到 morphism 集合的双射。第二，$H^0(F)$ 在对象上本质满。普通范畴等价的判别准则说，一个函子若全忠实且本质满，则为范畴等价；这里全忠实正由零次上同调的双射给出。证毕。

**警告 0.6.** 命题 0.5 的逆命题不成立。两个增强范畴的 $H^0$ 可以等价，但增强结构、Hochschild invariants 或 Morita 类型不同。因此本书不会用三角范畴等价替代增强等价，除非已经证明所需增强唯一性或 Morita 不变量一致。

## 0.3 A-side 与 B-side 的最低数据

**定义 0.7.** A-side 的基础对象是带有足够结构的辛几何数据。最简单的 exact 版本包括：

- 一个 exact symplectic manifold $(M,\omega=d\lambda)$；
- 一类 exact Lagrangian submanifolds $L\subset M$；
- 每个 $L$ 的 brane data，包括 orientation、grading、spin 或 Pin 结构和局部系统；
- 用于定义 Floer cochains 和 holomorphic polygon 计数的 almost complex structures 与扰动数据；
- compactness、transversality、orientation 和 gluing 的可用定理。

**定义 0.8.** B-side 的基础对象是带有导出增强的代数几何数据。常见版本包括：

- 光滑适当 $k$-variety $X$ 的 $\operatorname{Perf}(X)$ 或 $\mathrm D^b\operatorname{Coh}(X)$ 的 dg enhancement；
- Landau-Ginzburg 模型 $(X,W)$ 的 matrix factorization category $\operatorname{MF}(X,W)$；
- 奇点或栈上的 singularity category、derived category 或 category of coherent sheaves。

**例 0.9.** “Calabi-Yau $X$ 的镜像是 $Y$”不是 HMS 命题。可检查的陈述必须指定类似
$$
\mathcal F(Y,\omega_Y)\simeq \operatorname{Perf}(X)
$$
的范畴等价，并解释 $Y$ 上的 branes、$X$ 的代数几何假设、系数域、分次和等价类型。

## 0.4 外部输入与本书内部证明

**约定 0.10.** 本书把以下内容默认列为外部输入，除非某章明确给出完整证明：

- pseudoholomorphic curve moduli spaces 的 compactness、transversality、orientation 和 gluing；
- Kuranishi structures、polyfolds、implicit atlases 或 virtual fundamental chains 的构造；
- 一般 Fukaya category 的构造与不变量性；
- Orlov representability、Bondal-Orlov reconstruction 等导出代数几何深定理；
- sectorial descent、stop removal、microlocal Morse theorem、wrapped Fukaya categories 的生成定理；
- 近期预印本中的新 HMS 例子。

**定义 0.11.** 本书称一个命题为“内部命题”，若其证明只依赖本书已经建立的定义、引理、定理和明确列出的基础代数事实。称一个命题为“外部输入定理”，若其证明依赖本书未展开的大型定理，并且来源登记在 `SOURCES.md`。

## 0.5 本书的研究边界

HMS 是仍在发展的研究计划。基础链条包括 $A_\infty$ 范畴、Fukaya categories、导出范畴、Fourier-Mukai transforms、Landau-Ginzburg models 和标准例子；研究边界则包括 functorial HMS、sectorial descent、microlocal sheaf models、Rabinowitz Fukaya categories、BPS categories、wall-crossing 与高维 hypersurfaces 的新证明。

本书的策略是：

1. 先建立可检查的增强范畴语言；
2. 再把 A-side 与 B-side 的必要结构分开构造；
3. 然后把 HMS 写成带假设的范畴等价命题；
4. 最后把近期结果纳入研究边界，并逐条登记来源和假设。

因此，本书所称的 HMS 从一开始就是一条有类型的增强范畴等价：系数、分次、brane data、增强模型和等价强度都是命题的一部分。第一章从 dg 与 $A_\infty$ 范畴开始，不是绕开几何，而是在建立一套能同时容纳两侧几何、并能区分“证据”与“等价”的语法。

## 练习

**练习 0.1.** 给出一个只写成三角范畴等价的 HMS 陈述，并指出其中缺少哪些增强数据。

**练习 0.2.** 设 $F:\mathcal A\to\mathcal B$ 是 dg quasi-equivalence。逐项写出 $H^0(F)$ 全忠实和本质满的证明。

**练习 0.3.** 解释为什么 holomorphic polygon 的高阶复合不能一般简化为普通范畴中的二元结合复合。
