# 第三卷资料源

## 核心来源

1. Dustin Clausen and Peter Scholze, *Condensed Mathematics and Complex Geometry*, arXiv:2605.11731.  
   链接：<https://arxiv.org/abs/2605.11731>  
   用途：第三卷主资料源，包含复几何应用、相干上同调、Serre duality、GAGA 和 Riemann-Roch 的 condensed/analytic 处理。

2. Peter Scholze, *Lectures on Condensed Mathematics*, arXiv:2605.03658.  
   链接：<https://arxiv.org/abs/2605.03658>  
   用途：solid、analytic、liquid、$f_!$ 与相干对偶的技术背景。

3. 第一卷与第二卷。  
   用途：本卷不重建凝聚基础和 analytic/liquid 范畴，只引用前两卷。

## 经典复几何来源

1. Henri Cartan, *Varietes analytiques complexes et cohomologie*.
   用途：Cartan 定理 A/B、Stein acyclicity 和 Cousin/Runge 证明路线的原始来源之一。

2. Hans Grauert, *Ein Theorem der analytischen Garbentheorie und Modulraeume komplexer Strukturen*, Publ. Math. IHES 5 (1960).
   用途：Grauert direct image theorem 和 coherent finiteness 输入。

3. Jean-Pierre Serre, *Un theoreme de dualite*, Comment. Math. Helv. 29 (1955), 9-26.
   用途：Serre duality 的经典来源。

4. Jean-Pierre Serre, *Geometrie algebrique et geometrie analytique*, Ann. Inst. Fourier 6 (1956), 1-42.
   用途：GAGA 的经典来源。

5. Armand Borel and Jean-Pierre Serre, *Le theoreme de Riemann-Roch*, Bull. Soc. Math. France 86 (1958), 97-136；SGA 6, LNM 225, Expose III.
   用途：GRR/HRR 的经典输入来源。

6. Raymond O. Wells, *Differential Analysis on Complex Manifolds*；Daniel Huybrechts, *Complex Geometry*.
   用途：Dolbeault lemma、elliptic/Hodge theory、Serre duality 和 Hodge-Fredholm 输入的教材定位。

## 引用纪律

- 第三卷的深层复几何定理均标为输入定理，除非正文真的给出完整证明。
- classical theorem 与 condensed theorem 分开写。
- 凝聚/analytic 复几何 locator 统一见总目录 [REFERENCE_LOCATOR_LEDGER.md](../REFERENCE_LOCATOR_LEDGER.md) 第 3 节；classical theorem 的定位状态见总目录 [INPUT_THEOREM_REGISTER.md](../INPUT_THEOREM_REGISTER.md) D 类输入和 locator 台账第 4 节。
- 若某节只是说明范畴语言，必须明确不声称完成证明。

## 章节依赖映射

- 第 1-3 章依赖第三卷 AR.1-AR.2 与输入 C.2-C.5：Fréchet 项的 liquid membership
  精确引用 CS26 Theorem 2.14、Lemma 2.16 与 Theorem 3.11；cohomology 比较使用
  第二卷命题 5.9 的局部提升及 classical Hodge/Green splitting。Clausen-Scholze
  提供复解析对象、holomorphic functions 与 coherent sheaves 的建模，classical
  Dolbeault lemma 仍是 D.1。
- 第 4-7 章依赖第三卷 AR.3-AR.6 与输入 D.3-D.7：coherent cohomology finite-dimensionality、Serre duality、GAGA 和 HRR/GRR；CS26 locator 已登记，经典证明来源已经分层定位，部分仍需最终 theorem/page locator。
- 第 8 章整理 six functor formalism 的位置。
- 附录 A-B 用于说明证明路线和术语翻译，不引入新的外部来源。
- 附录 C-E 使用经典复几何标准事实，包括 Cartan A/B、Dolbeault lemma、Serre duality 和 $\mathbb P^1$ 上线丛上同调计算。
- 附录 F-G 不引入新来源；它们把第三卷已经使用的经典输入和 Clausen-Scholze 输入拆成精确陈述与依赖链。
- 附录 H 使用经典代数几何/复几何中 $\mathbb P^1$ 上 $\mathcal O(d)$ 的标准 Čech 计算。
- 附录 I 使用 sheaf cohomology、injective/flasque resolution、Cech-to-derived spectral sequence 和 hypercohomology 的标准同调代数。它不新增复几何输入，只证明从 acyclic 覆盖到 $R\Gamma$ 计算的形式部分。
- 附录 J 使用有限维复形对偶、链级配对、trace/counit 和闭幺半范畴的一般形式理论；Serre perfectness 仍作为输入。
- 附录 K 使用 exact equivalence、bounded derived category、Grothendieck group、Euler characteristic 和 characteristic class 的形式性质；GAGA 与 Riemann-Roch 本身仍作为输入。
- 附录 L 使用 Hilbert complex、Hodge decomposition、Fredholm operator 和 Dolbeault Laplacian 的标准形式理论；椭圆正则性和 Fredholm 性本身仍作为分析输入。
- 附录 M 使用有限维线性代数、有限过滤、谱序列收敛和超上同调的标准形式理论；它明确说明 Stein-Cech 计算本身不推出有限维性，有限性仍需 Grauert、Fredholm-Hodge 或 Clausen-Scholze 输入。
- 附录 N 使用 fine sheaf、partition of unity、Cech 同伦、paracompact Cech-sheaf 比较和 acyclic resolution 的标准形式理论；局部 $\bar\partial$-Poincare lemma 与 liquid/analytic 提升仍作为输入。
- 附录 O 使用有限局部自由 resolution、派生 sheaf Hom、向量丛 Serre duality 和有限复形同调代数；一般相干层全局有限 resolution 与 dualizing complex 理论仍作为外部输入。
- 附录 P 使用 Chern 类、splitting principle、Chern character、Todd class 和 $K$-理论的标准形式代数；特征类的几何构造和 HRR 本身仍作为输入。
- 附录 Q 使用 Serre GAGA 的 classical 输入、非 proper 仿射直线反例、exact equivalence 到 derived equivalence 的形式理论；GAGA 本身仍作为输入。
- 附录 R 使用 Cauchy-Green 算子、$\bar\partial$ 基本解、polydisc 局部同伦和向量丛平凡化；一变量奇核估计作为经典复分析输入。
- 附录 S 使用 $\mathbb P^n$ 的标准仿射覆盖、齐次 Laurent 单项式 Čech 复形、Cartan B 和基础 Serre duality 配对；不调用 Borel-Weil-Bott。
- 附录 T 使用 Euler sequence、canonical bundle、Čech residue 和附录 S 的单项式基，证明 $\mathbb P^n$ 线丛情形的 Serre 对偶。
- 附录 U 使用 $\mathbb P^n$ 的 cohomology 环、Euler sequence、Chern character、Todd class 和 residue 系数计算，证明线丛情形的 HRR。
- 附录 V 使用 Stein 空间、Cartan A/B、相干层有限表示、acyclic 覆盖和 Čech-to-derived 谱序列；Cartan A/B 仍作为经典输入。
- 附录 W 使用正则局部环、收敛幂级数环、Auslander-Buchsbaum/有限整体维数和相干层 stalk-to-sheaf 论证；交换代数定理作为输入。
- 附录 X 使用 hypercohomology spectral sequence 和有限复形同调代数，证明向量丛有限性在有全局有限局部自由 resolution 时传播到相干层。
- 附录 Y 使用 Serre twisting、projective GAGA 的 full faithfulness/essential surjectivity/cohomology comparison 模块和 finite presentation；完整代数化仍作为 GAGA 输入。
- 附录 Z 使用 Hilbert complex、椭圆 Fredholm-Hodge 输入、harmonic forms 和 Dolbeault Laplacian；parametrix 与椭圆估计作为分析输入。
- 附录 AA 使用 Dolbeault 配对、Hodge star、harmonic representatives 和有限维线性代数，证明向量丛 Serre 对偶的完美性；Hodge star 与 Laplacian 相容作为输入。
- 附录 AB 使用 Weierstrass preparation/division、Oka coherence、Cousin 问题和 Cartan 定理的经典证明模块；这些多复变定理作为输入。
- 附录 AC 使用 Grauert direct image theorem、semi-continuity 和 base change 形式；Grauert 定理作为输入，有限性作为推论证明。
- 附录 AD 使用 dualizing complex 和 Grothendieck-Serre duality；dualizing complex 存在性和 global duality 作为输入。
- 附录 AE 使用 Chern character、Todd class 和 Grothendieck-Riemann-Roch；GRR 本身作为输入，HRR 和函子性后果在书内证明。
- 附录 AF 使用 Weierstrass division、distinguished polynomial、Oka coherence 的归纳骨架和收敛性边界；Weierstrass 定理作为输入。
- 附录 AG 使用 Runge approximation、Cousin 分裂、Stein covering refinement 和 Čech direct limit，补 Cartan B 的证明机制。
- 附录 AH 使用 Stein exhaustion、Hörmander $L^2$ estimates、椭圆正则性和 Dolbeault resolution，补 $\bar\partial$ 方法推出 Stein 消没的模块。
- 附录 AI 使用 Serre correspondence、graded modules、twisting 和 analytic finite generation，补 projective GAGA 的代数化细节。
- 附录 AJ 使用 smooth/closed immersion 的 $f^!$、Koszul duality、trace 和 Grothendieck duality，补 duality 构造义务。
- 附录 AK 使用 projective bundle formula、regular immersion、deformation to the normal cone 和 graph factorization，补 GRR 证明模块。
- 附录 AL 使用 Banach 幂级数范数、Cauchy 估计、Neumann 级数和 Weierstrass division 的估计形式，补 Oka coherence 所需的收敛控制。
- 附录 AM 使用 Bochner-Kodaira-Nakano 恒等式、Hilbert 复形、闭值域判别和 Hahn-Banach 解算子，补 Hörmander 方法从基本估计到解方程的步骤。
- 附录 AN 使用 Grauert privileged covering、有限 Banach 复形、半连续性和 base change 判别，补 direct image coherence 的证明模块。
- 附录 AO 使用 theorem on formal functions、Grothendieck existence、解析形式邻域比较和形式 GAGA，补 projective GAGA 的代数化路线。
- 附录 AP 使用 $K$-理论局部化、Chow/cohomology 局部化、Chern character 边界相容和 proper pushforward 复合，补 GRR 的局部化证明组织。
- 附录 AQ 使用本卷所有主输入定理和第二卷 analytic/liquid 类型检查，补复几何主定理包在 condensed/analytic 语言中的闭包证明。
- 附录 AR 使用 Clausen-Scholze 复几何讲义、第二卷 analytic/liquid 接口和第三卷 AQ 主定理包，把复几何核心定理整理成建模、Dolbeault、有限性、对偶、GAGA、HRR/GRR 和 six functor 接口的图谱；它不新增输入。
