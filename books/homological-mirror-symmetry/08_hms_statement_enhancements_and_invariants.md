# 第八章：HMS 断言、增强等价与必要不变量

## 本章目标

本章给出 HMS 命题的标准写法，并列出任何候选 HMS 等价必须通过的范畴不变量检查。虽然前面若干技术章节尚待补写，本章提前固定“什么才算一个严格 HMS 陈述”。

## 依赖前置知识

需要第一章的 dg/$A_\infty$ 增强语言、第二章的 B-side 术语和第三章的 A-side 入口。

## 8.1 HMS 数据包

**定义 8.1.** 一个 HMS 数据包由
$$
\mathfrak H=(A,B,\mathcal A_A,\mathcal B_B,\kappa,\tau)
$$
组成，其中：

1. $A$ 是 A-side 几何数据，例如 $(M,\omega)$、Liouville sector、stopped sector 或 Landau-Ginzburg 模型；
2. $B$ 是 B-side 几何数据，例如 $k$-variety、stack、奇点或 Landau-Ginzburg 模型；
3. $\mathcal A_A$ 是由 $A$ 构造的 dg 或 $A_\infty$ category；
4. $\mathcal B_B$ 是由 $B$ 构造的 dg、$A_\infty$ 或 stable $\infty$ enhancement；
5. $\kappa$ 是系数、Novikov、grading、orientation、brane 和 spin/Pin 数据；
6. $\tau$ 指定等价类型：quasi-equivalence、Morita equivalence 或 stable $\infty$-equivalence。

**定义 8.2.** 数据包 $\mathfrak H$ 的 HMS 断言是命题
$$
\mathcal A_A\simeq_\tau \mathcal B_B.
$$
若 $\tau$ 是 Morita equivalence，则该断言等价于
$$
\operatorname{Perf}(\mathcal A_A)\simeq\operatorname{Perf}(\mathcal B_B).
$$

**例 8.3.** 对一个 compact Calabi-Yau mirror pair 的理想化陈述形如
$$
\mathcal F(Y,\omega_Y)\simeq \operatorname{Perf}(X).
$$
这不是完整陈述，除非同时说明 $\mathcal F$ 的构造口径、$X$ 的几何假设、系数域、分次、brane data、是否 idempotent complete，以及 $\simeq$ 的含义。

## 8.2 三个强度层级

**定义 8.4.** HMS 断言有三个常见强度层级。

1. **Cohomological shadow.** 只比较 $H^\ast$ 层面的 morphism spaces、Euler pairing 或 Grothendieck groups。
2. **Triangulated equivalence.** 给出三角范畴等价
   $$
   H^0\operatorname{Tw}(\mathcal A_A)\simeq H^0(\mathcal B_B).
   $$
3. **Enhanced/Morita equivalence.** 给出 dg/$A_\infty$/stable $\infty$ 增强层面的等价。

**约定 8.5.** 本书称第三层级为“完整 HMS 形态”。第一、第二层级只能作为证据、推论或历史版本。

**命题 8.6.** enhanced/Morita equivalence 蕴含 triangulated equivalence；triangulated equivalence 蕴含 Grothendieck group 同构和 Euler pairing 保持。

**证明.** enhanced equivalence 取 $H^0$ 得到三角范畴等价，因为 pretriangulated 结构保证 shifts 和 cones 在 $H^0$ 中存在并被保持。三角范畴等价保持 distinguished triangles，因此诱导 Grothendieck groups 的同构。Euler pairing
$$
\chi(E,F)=\sum_i(-1)^i\dim_k\operatorname{Hom}(E,F[i])
$$
在有限维和有界假设下由 Hom 空间维数定义，三角等价保持 Hom 和 shift，所以保持 $\chi$。证毕。

**警告 8.7.** 命题 8.6 不能反向使用。Grothendieck group、Hodge number 或 Hochschild homology 匹配只是必要条件，不是 HMS 等价的证明。

## 8.3 生成元法

许多 HMS 证明不直接构造所有对象上的函子，而是比较生成对象的 endomorphism algebras。

**定义 8.8.** 设 $\mathcal A$ 是 pretriangulated $A_\infty$ category。对象集合 $\mathcal G\subset\operatorname{Ob}\mathcal A$ split-generates $\mathcal A$，若 $\mathcal A$ 中每个对象都属于由 $\mathcal G$ 在 shifts、cones 和 direct summands 下生成的厚子范畴。

**命题 8.9（生成元比较原则）.** 设 $\mathcal G\subset\mathcal A$、$\mathcal H\subset\mathcal B$ 分别 split-generate。若 full $A_\infty$ subcategories $\mathcal A_{\mathcal G}$ 与 $\mathcal B_{\mathcal H}$ quasi-equivalent，则 $\mathcal A$ 与 $\mathcal B$ Morita equivalent。

**证明.** 这正是命题 1.24。生成对象决定 perfect module category 的厚闭包；full subcategories 的 quasi-equivalence 诱导 representable modules 的 quasi-equivalence，并延拓到 shifts、cones 和 direct summands 闭包。证毕。

**解释 8.10.** 在实际 HMS 证明中，A-side 的 $\mathcal G$ 往往是 vanishing cycles、thimbles、cocores 或特定 Lagrangian tori；B-side 的 $\mathcal H$ 往往是 line bundles、exceptional collection、tilting bundle 或 skyscraper/structure sheaves。

## 8.4 必要不变量

**定义 8.11.** 对 proper 增强范畴 $\mathcal A$，Euler pairing 定义为
$$
\chi_{\mathcal A}(X,Y)=
\sum_i(-1)^i\dim_k H^i\operatorname{hom}_{\mathcal A}(X,Y).
$$

**命题 8.12.** 若 $F:\mathcal A\to\mathcal B$ 是 proper $A_\infty$ categories 的 quasi-equivalence，则
$$
\chi_{\mathcal A}(X,Y)=\chi_{\mathcal B}(FX,FY).
$$

**证明.** quasi-equivalence 给出 morphism complexes 的 quasi-isomorphism
$$
\operatorname{hom}_{\mathcal A}(X,Y)\to
\operatorname{hom}_{\mathcal B}(FX,FY).
$$
quasi-isomorphism 保持上同调维数。proper 假设保证上式中的交错和有限。因此 Euler pairing 被保持。证毕。

**外部输入定理 8.13（Hochschild Morita invariance）.** Hochschild homology 和 Hochschild cohomology 在 dg 或 $A_\infty$ Morita equivalence 下不变。  
来源：Keller 的 dg categories 与 derived Morita theory。

**推论 8.14.** 一个 Morita 版本 HMS 等价给出
$$
HH_\ast(\mathcal A_A)\cong HH_\ast(\mathcal B_B).
$$
在光滑适当 B-side 情况下，Hochschild-Kostant-Rosenberg 型定理可进一步把右边与 Hodge cohomology 联系起来；这一步是外部输入，且需要特征、光滑性和适当性假设。

## 8.5 HMS 陈述模板

一个严格 HMS 章节应采用如下模板。

**模板 8.15.**

1. **镜像数据。** 明确 A-side 和 B-side 的几何对象。
2. **系数与分次。** 指定 $k$、Novikov field、grading、brane data 和 orientation conventions。
3. **两边类别。** 写出 $\mathcal A_A$ 和 $\mathcal B_B$ 的增强模型。
4. **生成对象。** 列出候选生成元，并说明 split-generation 的证明来源。
5. **endomorphism algebra。** 计算两边生成元的 $A_\infty$ endomorphism algebra。
6. **等价函子。** 说明函子、kernel 或 module 如何构造。
7. **等价强度。** 标明 quasi-equivalence、Morita equivalence 或 stable $\infty$-equivalence。
8. **不变量检查。** 检查 Euler pairing、Grothendieck group、Hochschild invariants、Serre functor 或 Calabi-Yau dimension。
9. **外部输入。** 登记 analytic、algebraic 或 recent-preprint 输入。

**例 8.16（非陈述）。** “$X$ 与 $Y$ 是镜像，所以 Hodge diamonds 旋转”不构成 HMS 证明。它最多是 closed-string 层面的必要数值证据，尚未构造 open-string category 等价。

**例 8.17（可接受陈述形态）。** 设 $A$ 是某 Landau-Ginzburg A-model 的 Fukaya-Seidel category，$B$ 是某 Fano variety。一个可接受的 HMS 陈述应写为：
$$
\mathcal F\mathcal S(A)\simeq_\mathrm{Morita}\operatorname{Perf}(B),
$$
并给出 vanishing thimbles 与 B-side exceptional collection 的 endomorphism $A_\infty$ algebras 的 quasi-isomorphism。

## 8.6 当前研究边界

截至 2026-07-08，HMS 的活跃方向包括：

- wrapped/partially wrapped Fukaya categories 的 functoriality、descent 与 localization；
- microlocal sheaves 与 Fukaya categories 的等价；
- hypersurfaces in algebraic tori、pair-of-pants decompositions 和 tropical degenerations；
- Rabinowitz Fukaya categories 与 singularity/matrix factorization 版本；
- functorial HMS、wall-crossing、BPS categories 和物理启发的高阶结构。

这些方向会进入后续专题章节。除非完成 theorem locator 和证明依赖审查，本书不会把近期预印本中的新结论写成基础定理。

## 本章小结

严格 HMS 命题是一条增强范畴等价命题，而不是镜像直觉或数值匹配。生成元法是实际证明中的核心机制：先证明两边生成，再比较 endomorphism $A_\infty$ algebras。Euler pairing、Hochschild invariants 和 Grothendieck groups 是必要检查，但不能替代等价证明。

## 练习

**练习 8.1.** 将一个口语化 HMS 陈述改写成定义 8.1 的数据包。

**练习 8.2.** 证明 quasi-equivalence 保持 Euler pairing，并指出 proper 假设在哪里使用。

**练习 8.3.** 给出一个只通过 Grothendieck group 同构但不应推出范畴等价的理由。

**练习 8.4.** 选一个已知 HMS 例子，按模板 8.15 列出需要检查的九项数据。
