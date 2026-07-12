# 附录 K：生成、descent 与 localization 证明模板

## K.1 生成元比较模板

**模板 K.1.** 要证明
$$
\mathcal A\simeq_{\mathrm{Morita}}\mathcal B,
$$
按以下步骤：

1. 选 $\mathcal G\subset\mathcal A$；
2. 选 $\mathcal H\subset\mathcal B$；
3. 证明 $\mathcal G$ split-generates $\mathcal A$；
4. 证明 $\mathcal H$ split-generates $\mathcal B$；
5. 构造 full subcategories 的 $A_\infty$ quasi-equivalence
   $$
   \mathcal A_{\mathcal G}\simeq\mathcal B_{\mathcal H};
   $$
6. 推出 Morita equivalence。

**命题 K.2.** 若模板 K.1 的第 3--5 步成立，则
$\mathcal A\simeq_{\mathrm{Morita}}\mathcal B$。

**证明.** 第 3、4 步把 $\operatorname{Perf}(\mathcal A)$ 与
$\operatorname{Perf}(\mathcal B)$ 分别识别为由 $\mathcal G,\mathcal H$ 的
representables 生成的厚闭包。第 5 步使两边 full generating
subcategories 的 perfect module categories quasi-equivalent，并保持每个
对象的 representable。该 quasi-equivalence 对 shifts、cones 与 retracts
延拓，因而给出两个厚闭包的 quasi-equivalence。证毕。

**边界 K.2A.** 只有在 generators 有限、finite direct sums 已存在且对象
idempotents 被保持时，才可把第 5 步压缩为一个 endomorphism
$A_\infty$ algebra quasi-isomorphism。Cohomology algebras 同构不够。

## K.2 Open-closed 生成模板

**模板 K.3.** 要用 Abouzaid criterion 证明 A-side 生成：

1. 明确使用 exact wrapped theorem 还是另有来源的 compact/monotone
   summand theorem；
2. 在 exact wrapped 情形定位 $1_{SH}\in SH^0(M)$；
3. 对 full subcategory $\mathcal W_{\mathcal G}$ 构造
   $\alpha\in HH_{-n}(\mathcal W_{\mathcal G})$；
4. 计算完整 composite
   $$
   HH_\bullet(\mathcal W_{\mathcal G})\to
   HH_\bullet(\mathcal W(M))\xrightarrow{\mathcal{OC}}SH^{\bullet+n}(M);
   $$
5. 证明 $\mathcal{OC}(\alpha)=1_{SH}$，再引用定理 14.7。

**边界 K.4.** 若只知道 $\mathcal{OC}$ 非零，不足以推出生成。命中 central
idempotent $e\ne1$ 也不能用 exact theorem 14.7 推出全范畴生成；要生成
$e$-summand，必须另引 summand generation theorem 并先用 closed-open
projector 定义该 summand。

## K.3 Sectorial descent 模板

**模板 K.5.** 要用 sectorial descent：

1. 给出定义 15.1 意义下的 Weinstein sectorial cover $\{X_i\}$；
2. 检查 sectorial hypersurface 条件，并验证所有 strata 的 symplectic
   reductions 在 convexification 后 Weinstein；这同时保证有限交仍为
   Liouville sectors；
3. 写出 Cech diagram $J\mapsto\mathcal W(X_J)$；
4. 证明局部 categories 的生成或局部 HMS；
5. 引用 GPS descent 得到
   $$
   \operatorname*{hocolim}_J\mathcal W(X_J)\simeq\mathcal W(X).
   $$

**命题 K.6.** 若两个 Cech diagrams 逐点 Morita equivalent，且自然变换相容，则其 homotopy colimits Morita equivalent。

**证明.** Morita homotopy category 中逐点等价给出 diagrams 的等价。homotopy colimit 是该 homotopy category 中的导出余极限，保持 diagram 等价。证毕。

## K.4 Stop removal 模板

**模板 K.7.** 要证明 stop removal 对应 localization：

1. 给出 stops $\mathfrak f\subset\mathfrak g$；
2. 找出 $\mathfrak g\setminus\mathfrak f$ 的 linking disks 集合 $\mathcal D$；
3. 证明或引用 linking disks 生成 kernel；
4. 写出 quotient
   $$
   \mathcal W(M,\mathfrak g)/\langle\mathcal D\rangle
   \simeq \mathcal W(M,\mathfrak f).
   $$

**命题 K.8.** 在模板 K.7 中，若对象 $X$ 属于 $\langle\mathcal D\rangle$，则它在 $\mathcal W(M,\mathfrak f)$ 中为零。

**证明.** quotient functor 把 $\langle\mathcal D\rangle$ 中对象送零；该子范畴在 shifts、cones 和 direct summands 下闭合，所以 $X$ 也被送零。证毕。

## 本附录小结

正式 HMS 证明大多是这三类模板的组合：生成元比较、open-closed 生成、descent/localization。把模板写清楚可以防止把数值匹配或对象字典误认为范畴等价。
