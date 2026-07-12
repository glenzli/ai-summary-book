# 第八章：HMS 断言、增强等价与必要不变量

## 本章目标

本章给出 HMS 命题的标准写法，并列出任何候选 HMS 等价必须通过的范畴不变量检查。前面各技术章节已经固定两侧增强范畴的对象类型与外部输入边界；本章据此收束“什么才算一个严格 HMS 陈述”。

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
6. $\tau$ 指定等价类型：quasi-equivalence、Morita equivalence 或 stable $\infty$-equivalence；
7. 对每一边说明比较 raw category、pretriangulated envelope 还是
   idempotent-complete perfect-module category。

**定义 8.2.** 数据包 $\mathfrak H$ 的 HMS 断言是命题
$$
\mathcal A_A\simeq_\tau \mathcal B_B.
$$
若 $\tau$ 是 Morita equivalence，则该断言的精确含义是存在
quasi-equivalence
$$
\operatorname{Perf}(\mathcal A_A)
\simeq_{\mathrm{qe}}
\operatorname{Perf}(\mathcal B_B),
$$
或等价的 invertible bimodule。右式不是原始 categories 的
quasi-equivalence 断言。

**例 8.3.** 对一个 compact Calabi-Yau mirror pair 的理想化陈述形如
$$
\mathcal F(Y,\omega_Y)\simeq \operatorname{Perf}(X).
$$
这不是完整陈述，除非同时说明 $\mathcal F$ 的构造口径、$X$ 的几何假设、系数域、分次、brane data、是否 idempotent complete，以及 $\simeq$ 的含义。

## 8.2 等价层级与允许的推出

**定义 8.4.** HMS 断言必须从下列层级中明确选择。

1. **Cohomological shadow.** 只比较 $H^\ast$ 层面的 morphism spaces、Euler pairing 或 Grothendieck groups。
2. **Triangulated shadow.** 给出三角范畴等价，并标明是 raw 还是 perfect
   completion，例如
   $$
   H^0\operatorname{Perf}(\mathcal A_A)
   \simeq H^0\operatorname{Perf}(\mathcal B_B).
   $$
3. **Quasi-equivalence.** 给出具体 dg/$A_\infty$ functor，在全部 morphism
   complexes 上 quasi-isomorphic 且在 $H^0$ 上 essentially surjective。
4. **Morita equivalence.** 比较 perfect module categories；stable
   $\infty$ 口径则比较相应 idempotent-complete small stable categories。

**约定 8.5.** 本书称第三或第四层级为“增强明确的 HMS 形态”。第一、
第二层级只能作为证据、推论或历史版本。Quasi-equivalence 蕴含 Morita
equivalence，但 Morita equivalence 不是原始对象层面的 quasi-equivalence。

**命题 8.6（层级推出的精确版本）.** 设 $\mathcal A,\mathcal B$ 是小、
严格含单位 dg 或 $A_\infty$ categories。

1. Quasi-equivalence $F:\mathcal A\to\mathcal B$ 诱导
   $H^0(\mathcal A)\simeq H^0(\mathcal B)$，并且是 Morita equivalence。
2. Morita equivalence 诱导三角等价
   $$
   H^0\operatorname{Perf}(\mathcal A)
   \simeq H^0\operatorname{Perf}(\mathcal B).
   $$
3. 只有再假设 Yoneda maps
   $\mathcal A\to\operatorname{Perf}(\mathcal A)$、
   $\mathcal B\to\operatorname{Perf}(\mathcal B)$ 均为 quasi-equivalences
   时，第 2 项才推出原始 $H^0(\mathcal A)\simeq H^0(\mathcal B)$。
4. 第 2 项诱导 Grothendieck groups 同构；若两边 proper，使每对对象的
   total morphism cohomology 有限维，则还保持 Euler pairing。

**证明.** 第 1 项的 $H^0$ 结论是引理 1.12；quasi-equivalence 诱导
representable modules 的 quasi-equivalence，并在 shifts、cones 与
retracts 下延拓，故诱导 perfect modules 的 quasi-equivalence。第 2 项
正是 Morita equivalence 的定义。第 3 项在两边分别用所假设的 Yoneda
quasi-equivalence 与第 2 项复合。三角等价保持 distinguished triangles，
所以诱导 Grothendieck groups 同构。Euler pairing
$$
\chi(E,F)=\sum_i(-1)^i\dim_k\operatorname{Hom}(E,F[i])
$$
在 proper 假设下是有限和；三角等价保持 Hom 与 shifts，所以保持
$\chi$。证毕。

**反例 8.7（Morita 不推出 raw $H^0$ 等价）.** 令 $\mathcal A$ 为只有
一个对象、endomorphism dg algebra 为 $k$ 的 dg category。Yoneda inclusion
$$
\mathcal A\longrightarrow\operatorname{Perf}_{\mathrm{dg}}(k)
$$
是 Morita equivalence，但 $H^0(\mathcal A)$ 只有一个对象同构类，而
$H^0\operatorname{Perf}_{\mathrm{dg}}(k)$ 还含不与 $k$ 同构的 $k[1]$。故两
个 raw $H^0$ categories 不等价。Morita 断言的三角影子必须取 perfect
completion。

**警告 8.7A.** 命题 8.6 的其他箭头也不能反向使用。Grothendieck group、
Hodge number 或 Hochschild homology 匹配只是必要条件，不是 HMS 等价的
证明。

## 8.3 生成元法

许多 HMS 证明不直接构造所有对象上的函子，而是比较生成对象的 endomorphism algebras。

**定义 8.8.** 设 $\mathcal A$ 是小、严格含单位 $A_\infty$ category。
对象集合 $\mathcal G$ split-generates $\mathcal A$，若 representables
$Y_G$ 的厚闭包等于
$H^0\operatorname{Perf}(\mathcal A)$；这就是定义 1.23A。只有当
$\mathcal A$ 已 pretriangulated、idempotent-complete 且 Morita-complete
时，才可把目标简写成 $H^0(\mathcal A)$。这里 Morita-complete 是指
Yoneda functor $\mathcal A\to\operatorname{Perf}(\mathcal A)$ 为
quasi-equivalence；仅有前两个形容词并不在任意模型中自动给出这个结论。

**命题 8.9（生成元比较原则）.** 设 $\mathcal A,\mathcal B$ 是小、严格
含单位 dg/$A_\infty$ categories，$\mathcal G\subset\mathcal A$、
$\mathcal H\subset\mathcal B$ 分别 split-generate。若存在 full
subcategories 的 strictly unital quasi-equivalence
$$
F:\mathcal A_{\mathcal G}\xrightarrow{\simeq_{\mathrm{qe}}}
\mathcal B_{\mathcal H},
$$
则 $\mathcal A$ 与 $\mathcal B$ Morita equivalent。

**证明.** 这正是命题 1.24。生成对象决定 perfect module category 的厚闭包；full subcategories 的 quasi-equivalence 诱导 representable modules 的 quasi-equivalence，并延拓到 shifts、cones 和 direct summands 闭包。证毕。

**解释 8.10.** 在实际 HMS 证明中，A-side 的 $\mathcal G$ 往往是 vanishing cycles、thimbles、cocores 或特定 Lagrangian tori；B-side 的 $\mathcal H$ 往往是 line bundles、exceptional collection、tilting bundle 或 skyscraper/structure sheaves。

若 $\mathcal G,\mathcal H$ 有限且两边已添加有限直和，可令
$G=\bigoplus G_i$、$H=\bigoplus H_i$，把 $F$ 改写为带对象幂等元的
endomorphism $A_\infty$ algebras quasi-isomorphism。只比较
$H^\ast\operatorname{End}(G)$ 与
$H^\ast\operatorname{End}(H)$ 的 graded algebras 不够：不同 $\mu^3$ 或
Massey products 可以有相同 cohomology algebra 而非 quasi-isomorphic 的
$A_\infty$ structures。

## 8.4 必要不变量

**定义 8.11.** 小 $k$-线性增强范畴 $\mathcal A$ 称为 proper，若对任意
$X,Y$，$\bigoplus_iH^i\operatorname{hom}_{\mathcal A}(X,Y)$ 为有限维
$k$-向量空间。此时 Euler pairing 定义为
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

**证明.** HMS 等价在本推论中按 Morita 等价解释。由外部输入定理
8.13，Hochschild homology 在 Morita 等价下不变，故得到显示的同构。
HKR 识别不是该推论的一部分，只在另行满足所列几何假设时使用。证毕。

## 8.5 HMS 陈述模板

一个严格 HMS 章节应采用如下模板。

**模板 8.15.**

1. **镜像数据。** 明确 A-side 和 B-side 的几何对象。
2. **系数与分次。** 指定 $k$、Novikov field、grading、brane data 和 orientation conventions。
3. **两边类别。** 写出 $\mathcal A_A$ 和 $\mathcal B_B$ 的增强模型。
4. **生成对象。** 列出候选生成元，并说明 split-generation 的证明来源。
5. **endomorphism category。** 计算两边生成 full subcategories；只有在
   finite-direct-sum 与对象幂等元已固定时才压缩为一个 endomorphism
   $A_\infty$ algebra。
6. **等价函子。** 说明函子、kernel 或 module 如何构造。
7. **等价强度。** 标明 quasi-equivalence、Morita equivalence 或 stable $\infty$-equivalence。
8. **不变量检查。** 检查 Euler pairing、Grothendieck group、Hochschild invariants、Serre functor 或 Calabi-Yau dimension。
9. **外部输入。** 登记 analytic、algebraic 或 recent-preprint 输入。

**例 8.16（非陈述）。** “$X$ 与 $Y$ 是镜像，所以 Hodge diamonds 旋转”不构成 HMS 证明。它最多是 closed-string 层面的必要数值证据，尚未构造 open-string category 等价。

**例 8.17（可接受陈述形态）。** 设 $A$ 是某 Landau-Ginzburg A-model 的 Fukaya-Seidel category，$B$ 是某 Fano variety。一个可接受的 HMS 陈述应写为：
$$
\mathcal F\mathcal S(A)\simeq_\mathrm{Morita}\operatorname{Perf}(B),
$$
并分别证明两组对象 split-generate，再给出其 full subcategories 的
strictly unital $A_\infty$ quasi-equivalence；有限直和口径下，这可由保持
对象幂等元的 endomorphism $A_\infty$ algebras quasi-isomorphism 实现。

## 8.6 当前研究边界

截至 2026-07-11，HMS 的活跃方向包括：

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
