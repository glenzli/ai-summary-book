# 外部输入定理登记表

作者：Dr. Stochastic Parrot

## 目标

本表列出四卷中仍作为外部输入使用的定理。每条记录包含四项：

1. 精确用途；
2. 本书已经证明的部分；
3. 尚未书内证明的部分；
4. 依赖该输入的章节。

这份登记表用于防止两种错误：把深定理伪装成已证结论，或在后文无标记地使用外部输入。

## A. 拓扑与 Stone-Gleason 输入

### A.1 Boolean prime ideal theorem

**用途。** 构造 Boolean algebra 的 Stone 空间点，即 ultrafilter。

**书内部分。** 附录 N 证明 Stone 空间的拓扑、开闭基、紧 Hausdorff 性和 Boolean algebra 恢复。

**外部部分。** 每个 proper Boolean ideal 包含于 prime ideal。

**来源定位。** Johnstone, *Stone Spaces* 中 Stone duality/ultrafilter 背景；当前为 `REFERENCE_LOCATOR_LEDGER.md` 中 L1，出版前需补 theorem/page。

**依赖位置。** 第一卷附录 N；第一卷附录 D、J、O 间接依赖。

### A.2 Sikorski extension theorem

**用途。** 完备 Boolean algebra 对 Boolean algebra 嵌入具有延拓性质。

**书内部分。** 第一卷附录 O 证明该定理推出 Stone 空间范畴中的投射性。

**外部部分。** Sikorski extension theorem 本身。

**来源定位。** Sikorski, *Boolean Algebras* 中 extension theorem；当前为 `REFERENCE_LOCATOR_LEDGER.md` 中 L1，出版前需补 edition theorem/page。

**依赖位置。** 第一卷附录 O。

### A.3 Gleason lifting theorem

**用途。** 极不连通紧 Hausdorff 空间对任意 compact Hausdorff 满射具有提升。

**书内部分。** 第一卷附录 J 构造 Gleason cover；附录 O 证明 Boolean algebra 端的投射性模块和投射推出极不连通的方向。

**外部部分。** 从 regular open/Stone 端提升下降到一般 compact Hausdorff 空间的拓扑选择步骤。

**来源定位。** Gleason, *Projective topological spaces*, Illinois J. Math. 2 (1958), Theorem 2.5；Johnstone, *Stone Spaces* 中 Stonean/projective compact Hausdorff 背景。

**依赖位置。** 第一卷第六至八章、附录 K、附录 O；第二卷和第三卷通过投射生成元间接依赖。

## B. Nöbeling 与 solid 输入

### B.1 Nöbeling theorem

**用途。** 证明 \(C(S,\mathbb Z)\) 对 profinite \(S\) 自由，从而控制 \(\mathbb Z^\square[S]\) 的乘积型模型。

**书内部分。** 第一卷附录 F 和 P 证明有限、可数情形，并给出超限过滤到自由性的代数引理。

**外部部分。** 任意 profinite \(S\) 的 Nöbeling-Asgeirsson 超限过滤构造。

**来源定位。** A23 Section `sec:theorem`，Theorem label `Nobeling`；预备结果在 `sec:preliminaries`，形式化说明在 `sec:formalisation`。S26 Lecture V theorem `thm:specker` 也使用该结构结果。

**依赖位置。** 第一卷第十二章、附录 F、附录 P；第二卷 solid localization；第四卷 solid 计算例子。

### B.2 Solidification existence

**用途。** 构造 \(D_\square(\mathbb Z)\) 作为 \(D(\mathbf{CondAb})\) 的反射局部子范畴。

**书内部分。** 第二卷附录 C、E、K、O、V 证明反射局部化、局部等价、kernel、幺半下降、集合生成局部化和 \(K_S\)-正交判别的形式部分。

**外部部分。** Scholze 的 solid kernel 生成、solid 反射存在性和张量理想性。

**来源定位。** S26 Lecture V definition `def:solid` and theorem `thm:solid`；Lecture VI gives the proof of `thm:solid` and corollary `cor:solidproperties`。

**依赖位置。** 第一卷第十二至十三章；第二卷第一、二章和附录 M、Q、V；第四卷 solid 例子。

### B.3 Solid tensor product

**用途。** 定义

$$
M\otimes_{\mathbb Z}^{L,\square}N
$$

并证明其对称幺半性质。

**书内部分。** 第二卷附录 K、L、O、W 证明张量理想 kernel 推出幺半下降、闭结构比较、张量理想生成元判别和归约到 profinite 测度张量计算。

**外部部分。** solid kernel 对张量封闭；solid tensor 与测度对象相容。

**来源定位。** S26 Lecture VI theorem `thm:solidtensor`；tensor compatibility is proved using `cor:solidproperties` and appears again in proposition `prop:tensorinfproducts`。

**依赖位置。** 第一卷第十三章；第二卷第二章、附录 Q、W；第四卷第四章。

### B.4 Profinite measure tensor formula

**用途。** 计算 profinite 测度对象的 solid 张量积：

$$
\mathbb Z^\square[S]\otimes_{\mathbb Z}^{L,\square}\mathbb Z^\square[T]
\simeq
\mathbb Z^\square[S\times T].
$$

**书内部分。** 第二卷附录 Q、W 证明该公式一旦作为输入，就推出 solid 生成元计算、solid kernel 张量理想性归约、solid 环/模的类型检查和第四卷 solid tensor 例子的使用边界。

**外部部分。** Scholze solid theory 中 profinite 测度对象与 solid tensor 的相容性。

**来源定位。** S26 Lecture VI proposition `prop:tensorinfproducts`；A23 theorem `Nobeling` supplies the profinite \(C(S,\mathbb Z)\) freeness used in the measure-object calculation.

**依赖位置。** 第二卷第二章、附录 Q、附录 T、附录 W；第四卷第四章。

## C. Analytic 与 liquid 输入

### C.1 Analytic ring localization

**用途。** 对 analytic ring \((A,\mathcal M)\)，构造 \(D(A,\mathcal M)\) 和 analyticization。

**书内部分。** 第二卷第三、四章和附录 I、N、X 给出 cone 判别、失败模式、rational descent 证明义务、analyticization 泛性质和 analytic tensor 的形式下降。

**外部部分。** Scholze analytic ring 公理推出反射局部化、张量结构和 descent。

**来源定位。** S26 Lecture VII definition `def:analytic`, proposition `prop:analyticnice`, functoriality proposition `prop:functoriality`, examples proposition `prop:exanalytic`。

**依赖位置。** 第一卷第十四章；第二卷第三至六章、附录 R、X；第三卷 analytic/liquid 复几何接口。

### C.2 Liquid realization

**用途。** 把 Banach、Fréchet、Dolbeault、分布和核函数对象放入 analytic/liquid 范畴。

**书内部分。** 第二卷附录 J、P、Z 证明拓扑向量空间凝聚化、Fréchet 复形闭值域、Fredholm cohomology 和 realization 后的类型检查。

**外部部分。** Scholze/Clausen-Scholze 的 liquid realization 与经典连续线性算子相容。

**来源定位。** CS26 Lectures II-III: definition of \(p\)-liquid objects, theorem `thm:qspliquid`, theorem `thm:liquidmain`；Lecture IV 给出 tensor calculations。精确 realization 子范畴在本书中作为输入范围保留。

**依赖位置。** 第二卷第五章、附录 S、Z；第三卷第三、四、五章；第四卷第六章。

### C.3 \(p\)-liquid analytic ring

**用途。** 构造 \(p\)-liquid 实向量空间所在的 analytic 模范畴。

**书内部分。** 第二卷第五章给出定义入口；附录 S、Z 证明接受该输入后，finite-dimensional cohomology、Fredholm 复形和 Dolbeault 类型检查在 liquid 范畴中闭合。

**外部部分。** \((\mathbb R,\mathcal M_{<p})\) 满足 analytic ring 条件，以及 \(p\)-liquid 测度理论的构造。

**来源定位。** S26 Lecture VII theorem asserting \((\mathbb R,\mathcal M_{<p})\) is analytic for \(0<p\le 1\)；CS26 Lectures II-III develop \(p\)-liquid vector spaces and the main theorem `thm:liquidmain`。

**依赖位置。** 第二卷第五章、附录 S、附录 T；第三卷 Dolbeault/liquid 接口。

### C.4 Rational Čech descent

**用途。** Huber pair rational localization 的对象和态射 gluing。

**书内部分。** 第二卷附录 G、N、Y 证明 Čech nerve、totalization、mapping-space descent、对象 gluing 和接受 descent 后的形式推论。

**外部部分。** Scholze rational localization 满足 analytic descent。

**来源定位。** S26 Lectures IX-X: discrete Huber pair, rational subsets, proposition `prop:locfullyfaithful`, proposition `prop:locbasechange`, theorem `thm:globalization`。

**依赖位置。** 第二卷第六章、附录 N、附录 R、附录 T、附录 Y；第三卷复几何 analytic localization 接口。

### C.5 Liquid exactness for closed-range complexes

**用途。** 将闭值域 Fréchet 复形的 cohomology 与 liquid realization 后的 cohomology 比较。

**书内部分。** 第二卷附录 P、S、Z 证明闭值域给 Hausdorff Fréchet cohomology，Fredholm 有限性给 perfect liquid cohomology，并说明 Dolbeault 复形使用时必须记录的类型数据。

**外部部分。** realization 保持相应闭值域短正合列，以及椭圆 Fréchet 复形的 Hodge/Fredholm 定理。

**来源定位。** CS26 liquid realization and tensor framework in Lectures II-IV；classical Hodge/Fredholm part remains INPUT D.4 and is not a condensed-mathematics theorem.

**依赖位置。** 第二卷附录 P、S、T、Z；第三卷 Dolbeault、有限性和 Serre duality 章节。

## D. 经典复几何输入

### D.1 Dolbeault lemma

**用途。** Dolbeault complex 解析相干层的 sheaf cohomology。

**书内部分。** 第三卷附录 R 给出 Cauchy-Green 和 polydisc 同伦骨架；附录 N 证明 acyclic resolution 的形式层。

**外部部分。** 带估计的局部 \(\bar\partial\)-Poincare lemma。

**来源定位。** Wells, *Differential Analysis on Complex Manifolds*, Dolbeault theorem chapter；Huybrechts, *Complex Geometry*, Dolbeault cohomology section。CS26 uses an analysis-free analytic-ring route for many results, but this book's Dolbeault comparison keeps the classical lemma as separate input.

**依赖位置。** 第三卷第三章、附录 N、R。

### D.2 Cartan A/B

**用途。** Stein 空间相干层全局生成和高阶上同调消没。

**书内部分。** 第三卷附录 V、AB、AG、AH 给出 Cartan A/B 的形式后果、Runge-Cousin 路线和 Hörmander 路线模块。

**外部部分。** Oka coherence、Runge approximation、Cousin theorem 或 Hörmander estimate 的完整证明。

**来源定位。** Cartan, *Varietes analytiques complexes et cohomologie*, Bruxelles colloquium；Grauert-Remmert, *Coherent Analytic Sheaves*, Cartan theorems chapter。当前为 L2，仍需补 edition theorem/page。

**依赖位置。** 第三卷第四章、附录 C、V、AB、AG、AH。

### D.3 Grauert direct image theorem

**用途。** proper holomorphic map 下相干直接像相干；推出紧复空间相干上同调有限维。

**书内部分。** 第三卷附录 AC、AN 证明从 Grauert 到有限性、Banach 复形、半连续性和 base change 的形式模块。

**外部部分。** privileged covering 的存在和 finite generation step。

**来源定位。** Grauert, *Ein Theorem der analytischen Garbentheorie und Modulraeume komplexer Strukturen*, Publ. Math. IHES 5 (1960), main theorem/Hauptsatz；CS26 Lecture XII theorem `thm:grauert` is the condensed/analytic package used in this book's complex-geometry atlas.

**依赖位置。** 第三卷第四章、附录 AC、AN。

### D.4 Hodge-Fredholm theorem

**用途。** 椭圆复形的 harmonic representatives、闭值域和有限维 cohomology。

**书内部分。** 第三卷附录 Z、AA 和第二卷附录 P 证明 Hilbert/Fréchet 复形到有限性和 Serre 配对的形式部分。

**外部部分。** parametrix、椭圆估计、Hodge decomposition。

**来源定位。** Wells, *Differential Analysis on Complex Manifolds*, elliptic/Hodge chapters；Huybrechts, *Complex Geometry*, Hodge theorem section。该输入有意与 CS26 的 condensed 证明路线分开登记。

**依赖位置。** 第三卷第四、五章；附录 Z、AA。

### D.5 Serre duality

**用途。** 相干层上同调与 Ext 的完美配对。

**书内部分。** 第三卷附录 J、O、AA、AD 给出链级配对、有限 resolution 推广、Hodge 向量丛情形和 dualizing complex 形式。

**外部部分。** 一般相干层完美性、dualizing complex 存在性和 trace theorem。

**来源定位。** Serre, *Un theoreme de dualite*, Comment. Math. Helv. 29 (1955), 9-26；Hartshorne, *Algebraic Geometry*, Theorem III.7.6；Huybrechts, *Complex Geometry*, Theorem 3.12。CS26 Lecture XIII proposition `prop:serreduality0` and theorem “Serre Duality”；S26 Lecture XI theorem `thm:openduality` for solid coherent duality analogue.

**依赖位置。** 第三卷第五章、附录 J、O、AA、AD。

### D.6 GAGA

**用途。** proper/projective 代数几何与解析几何相干层、上同调和导出范畴比较。

**书内部分。** 第三卷附录 Q、Y、AI、AO 给出 properness 反例、graded module 路线、形式函数路线和 derived comparison 形式层。

**外部部分。** Serre analytic finite generation、Grothendieck existence 和解析形式代数化。

**来源定位。** Serre, *Geometrie algebrique et geometrie analytique*, Ann. Inst. Fourier 6 (1956), 1-42；Hartshorne, Appendix B, Theorems 3.1-3.2。CS26 Lectures VI-VII theorem `thm:GAGAabstract` and Lecture XIII theorem “GAGA” give the condensed/analytic route.

**依赖位置。** 第三卷第六章、附录 Q、Y、AI、AO。

### D.7 Grothendieck-Riemann-Roch

**用途。** Chern character、Todd class 和 proper pushforward 的相容公式。

**书内部分。** 第三卷附录 P、U、AE、AK、AP 证明特征类形式代数、射影空间线丛 HRR、一般 GRR 后果、deformation-to-normal-cone 模块和局部化推前相容。

**外部部分。** localized Chern character、deformation specialization、excess intersection 和 GRR 基本因子定理。

**来源定位。** Borel-Serre, *Le theoreme de Riemann-Roch*, Bull. Soc. Math. France 86 (1958), 97-136；SGA 6, LNM 225, Expose III；Fulton, *Intersection Theory*, Riemann-Roch chapter。CS26 Lecture XIV theorem `GRR` and Lecture XV theorem `thm:GHRRfinal` with propositions `prop:GHRRformal`, `prop:GHRRtodd` give the condensed/analytic route.

**依赖位置。** 第三卷第七章、附录 P、U、AE、AK、AP。

## E. 同伦与谱值输入

### E.1 Pyknotic objects

**用途。** 与 condensed objects 比较，进入凝聚同伦类型和谱值 sheaf。

**书内部分。** 第四卷附录 E、G 给出 sheaf/hypersheaf 口径、谱值 sheaf 稳定性和 heart 层比较。

**外部部分。** Barwick-Haine 的 pyknotic \(\infty\)-topos 理论。

**依赖位置。** 第四卷附录 E、G。

### E.2 Spectral solid/analytic localization

**用途。** 将 solid/analytic localization 升级到谱值 sheaf 或稳定 \(\infty\)-范畴。

**书内部分。** 第二卷附录 O 和第四卷附录 G 给出稳定局部化和谱值 sheaf 的形式模块。

**外部部分。** 谱值 Dirac-to-measure cone 的 compact generation 和 monoidal compatibility。

**依赖位置。** 第四卷附录 G；后续同伦化方向。
