# 第十四章：split-generation、open-closed map 与 Abouzaid criterion

逐个描述 wrapped Fukaya category 的全部对象通常不现实；一组候选 Lagrangians 即使有可计算的 endomorphism algebra，也还可能只生成一个真子范畴。Abouzaid 的判据把这个缺口转化为 closed-string 端的单位问题：若候选子范畴的 Hochschild 类经完整 open-closed 复合命中 $SH^0(M)$ 的全局单位，它便 split-generate。为准确使用这句话，本章先在第一章的 Morita 语言中固定 split-generation，再区分 open-closed 与 closed-open 的方向和次数，最后分离命中全局单位、命中幂等分量与只得到非零像三种不同结论。曲线计数和 Cardy 同伦采用外部分析输入，形式范畴推论在正文证明。

## 14.1 Split-generation 的类型

**定义 14.1.** 设 $\mathcal A$ 是小、严格含单位 $A_\infty$ category，
$\mathcal G\subset\operatorname{Ob}\mathcal A$，并记
$\mathcal A_{\mathcal G}$ 为它们张成的 full subcategory。称
$\mathcal G$ split-generates $\mathcal A$，若
$$
\operatorname{thick}\{Y_G:G\in\mathcal G\}
=H^0\operatorname{Perf}(\mathcal A).
\tag{14.1}
$$
这里 $\operatorname{thick}$ 表示对 shifts、cones、有限直和、同构和
retracts 闭合。定义的目标是 perfect-module category；只有
$\mathcal A$ 已 Morita-complete 时才可把右端换成 $H^0(\mathcal A)$。

**命题 14.2.** 若 $\mathcal G$ split-generates $\mathcal A$，则 inclusion
$i:\mathcal A_{\mathcal G}\hookrightarrow\mathcal A$ 是 Morita equivalence。
更具体地，derived restriction 与 extension
$$
i^\ast:\operatorname{Perf}(\mathcal A)
\rightleftarrows
\operatorname{Perf}(\mathcal A_{\mathcal G}):
-\otimes^{\mathbf L}_{\mathcal A_{\mathcal G}}\mathcal A
\tag{14.2}
$$
互为 quasi-inverse。

**证明.** 对 $G\in\mathcal G$，restriction 把 $\mathcal A$-Yoneda module
$Y_G$ 送到 $\mathcal A_{\mathcal G}$-Yoneda module，extension 再把后者送回
$Y_G$；Yoneda fully-faithfulness 使 adjunction unit/counit 在这些 generators
上为 quasi-isomorphisms。两个 functors 都保持 shifts、cones、有限直和与
retracts。左边由 (14.1) 中的 $Y_G$ 厚生成，右边按 perfect module 的定义
由其全部 representables 厚生成，而这些 representables 正是
$G\in\mathcal G$ 的 Yoneda modules。因此 unit/counit 在两边所有 perfect
objects 上均为 quasi-isomorphisms，(14.2) 为 quasi-equivalence。证毕。

## 14.2 Hochschild chains 与 string maps

**定义 14.3.** 对小 $A_\infty$ category $\mathcal A$，Hochschild chain
complex $CC_\bullet(\mathcal A)$ 是 composable cyclic tensors 的直和，
differential 由各 $\mu^d$ 的内部与 cyclic insertions 给出，符号由附录 B
的 suspension convention 决定。其 homology 记为 $HH_\bullet(\mathcal A)$。
Full inclusion $\mathcal B\hookrightarrow\mathcal A$ 诱导
$HH_\bullet(\mathcal B)\to HH_\bullet(\mathcal A)$。

**定义 14.4.** 设 $M^{2n}$ 是 Liouville manifold，$\mathcal W(M)$ 是已构造
的 wrapped Fukaya category。本书采用 degree-$n$ 的 open-closed map 约定
$$
\mathcal{OC}_{\mathcal W}:
HH_\bullet(\mathcal W(M))\longrightarrow SH^{\bullet+n}(M).
\tag{14.3}
$$
若 $\mathcal B\subset\mathcal W(M)$ 为 full subcategory，记限制映射为复合
$$
\mathcal{OC}_{\mathcal B}:
HH_\bullet(\mathcal B)\longrightarrow HH_\bullet(\mathcal W(M))
\xrightarrow{\mathcal{OC}_{\mathcal W}}SH^{\bullet+n}(M).
\tag{14.4}
$$
于是“命中单位”准确地指存在
$\alpha\in HH_{-n}(\mathcal B)$ 使
$\mathcal{OC}_{\mathcal B}(\alpha)=1_{SH}\in SH^0(M)$。若文献整体移动
Hochschild grading，必须连同 (14.3) 一起移动，不能只删去 $+n$。

**定义 14.5.** Closed-open map 的方向为
$$
\mathcal{CO}:SH^\bullet(M)\longrightarrow
HH^\bullet(\mathcal W(M)).
\tag{14.5}
$$
在 compact monotone 口径中，闭弦端通常改为 quantum cohomology；该版本
需要另行指定 Novikov coefficients、$c_1$-eigensummand、curvature 与 virtual
perturbation package。

**外部输入定理 14.6（exact wrapped string-map package）.** 设 $M$ 为
Liouville manifold，并固定使 $\mathcal W(M)$、$SH^\ast(M)$ 及其单位良定义
的 coefficient、grading、brane、orientation、Hamiltonian cofinality 和
compactness/transversality data。则 (14.3)、(14.5) 可由带 interior marked
points 的 punctured disks 构造，并与 products、module structures 及
Abouzaid generation proof 所需的 Cardy relation 相容。

来源：Abouzaid 的 exact wrapped construction 与 Cardy argument；Ganatra
关于 wrapped duality/string maps 的工作；Liouville sectors 口径见 GPS。
本书不重建这些非紧模空间的分析或 chain-level Cardy homotopy。

## 14.3 Exact wrapped generation theorem

**外部输入定理 14.7（Abouzaid generation criterion，精确使用版）.** 在
定理 14.6 的假设下，令 $\mathcal B\subset\mathcal W(M)$ 是 full
$A_\infty$ subcategory。若
$$
1_{SH}\in\operatorname{im}\left(
HH_\bullet(\mathcal B)\longrightarrow
HH_\bullet(\mathcal W(M))\xrightarrow{\mathcal{OC}_{\mathcal W}}
SH^{\bullet+n}(M)
\right),
\tag{14.6}
$$
则 $\mathcal B$ 的对象 split-generate $\mathcal W(M)$；等价地，inclusion
$\mathcal B\hookrightarrow\mathcal W(M)$ 是 Morita equivalence。

**证明路线（外部输入）.** Cardy relation 把 (14.6) 中代表单位的
Hochschild cycle 与 diagonal bimodule 的一个 resolution 联系起来，从而
证明每个 representable $\mathcal W(M)$-module 属于 $\mathcal B$-
representables 的厚闭包。构造 two-output disk operations、证明 Cardy
homotopy 并控制 wrapped ends 是该定理的主体，未在本书内部建立。精确来源
为 Abouzaid, *A geometric criterion for generating the Fukaya category*,
Theorem 1.1 与公式 (1.2)。

**解释 14.8.** 定理 14.7 的结论是 split-generation/Morita equivalence，
不是 raw categories 的 quasi-equivalence。它也要求 (14.6) 的 composite
命中 global unit；只验证 open-closed map 非零不够。

**反例 14.8A（非零不等于命中单位）.** 若 closed-string degree-zero
algebra 为 $k\times k$，某 subcategory 的 open-closed image 可能只含
$k(1,0)$。该 image 非零且命中 idempotent $(1,0)$，却不含 global unit
$(1,1)$；最多能支持第一个 summand 的生成结论，不能推出全范畴生成。

**命题 14.9（generation 加 enhanced comparison 推出 wrapped HMS）.** 设
$\mathcal C$ 是小、严格含单位 B-side dg/$A_\infty$ category。假设：

1. $\mathcal G\subset\mathcal W(M)$ 张成 full subcategory
   $\mathcal W_{\mathcal G}$，并满足定理 14.7 的 unit-image 条件；
2. $\mathcal H\subset\mathcal C$ split-generates $\mathcal C$；
3. 存在 strictly unital quasi-equivalence
   $$
   F:\mathcal W_{\mathcal G}
   \xrightarrow{\simeq_{\mathrm{qe}}}\mathcal C_{\mathcal H}.
   $$

则
$$
\mathcal W(M)\simeq_{\mathrm{Morita}}\mathcal C.
$$

**证明.** 由定理 14.7，$\mathcal G$ split-generates $\mathcal W(M)$；第 2
项给出 B-side split-generation；第 3 项比较两边的 full generating
subcategories。命题 8.9 立即给出所示 Morita equivalence。证毕。

**边界 14.9A.** 若两组 generators 有限且两边已添加 finite direct sums，
可把第 3 项改写为
$\operatorname{End}(\bigoplus G_i)$ 与
$\operatorname{End}(\bigoplus H_i)$ 的 strictly unital $A_\infty$
quasi-isomorphism，但必须保持标记各对象的 orthogonal idempotents。仅有
cohomology endomorphism algebras 同构，或只比较 $\mu^2$，不满足第 3 项。

## 14.4 Idempotent summands 与 curvature values

**定义 14.10.** 设外部输入 (14.5) 已构造，且
$e\in SH^0(M)$ 是 central idempotent。其像
$\mathcal{CO}(e)\in HH^0(\mathcal W(M))$ 给出 identity functor 的
idempotent natural endomorphism。对
$H^0\operatorname{Perf}(\mathcal W(M))$ 中每个对象 $X$，记相应 projector
的 image 为 $X_e$；由所有 $X_e$ 厚生成的 full subcategory 记为
$\operatorname{Perf}(\mathcal W(M))_e$。

**命题 14.11（idempotent 分块的形式正交性）.** 在定义 14.10 的假设下，
$$
H^0\operatorname{Perf}(\mathcal W(M))
\simeq
H^0\operatorname{Perf}(\mathcal W(M))_e
\oplus
H^0\operatorname{Perf}(\mathcal W(M))_{1-e},
$$
且两个 summands 之间所有 morphisms 为零。

**证明.** Perfect category idempotent-complete，所以每个 projector 分裂为
$X\simeq X_e\oplus X_{1-e}$。若
$f:X_e\to Y_{1-e}$，naturality 给出
$$
f=f\,\mathcal{CO}(e)_{X_e}
=\mathcal{CO}(e)_{Y_{1-e}}f=0;
$$
反方向相同。每个对象均按两 projector 分解，得到所示直和。证毕。

**警告 14.12.** 若 $\mathcal{OC}_{\mathcal B}$ 命中 $e$ 而非
$1_{SH}$，要推出 $\mathcal B$ split-generates $e$-summand，仍需引用相应
summand version of the generation theorem；命题 14.11 本身只给形式分块。
在 compact monotone Fukaya theory 中，weak curvature value
$\mu_b^0=\lambda e_L$ 通常先标记 quantum multiplication by $c_1$ 的
$\lambda$-eigensummand；一个 eigensummand 还可能含多个 central
idempotents。故“potential value $\lambda$”与“primitive idempotent $e$”
不是同一种索引。不同 curvature values 的 morphism 类型由命题 5.14
处理，不能代替 closed-open projector 的分块定理。

命中 $1_{SH}$ 把局部的 endomorphism 计算提升为全 wrapped category 的 split-generation，但它只给 Morita 结论；HMS 仍需 B-side 的生成性和两组生成对象之间的增强比较。若 open-closed 像只含 central idempotent，结论至多落在相应 summand，且该分块与第五章的 curvature value 分解不是同一索引。下一章将说明这些局部生成对象如何随 sectorial cover 一同胶合。

## 练习

**练习 14.1.** 补全命题 14.2 中 unit/counit 的厚闭包论证。

**练习 14.2.** 在 degree convention (14.3) 下说明命中 $SH^0$ 的
Hochschild class 位于哪个次数。

**练习 14.3.** 用命题 14.9 写出一个 wrapped HMS 证明的完整类型骨架。

**练习 14.4.** 对 algebra $k\times k$ 验证反例 14.8A，并说明命中
$(1,0)$ 与命中 $(1,1)$ 的生成结论为何不同。
