# 附录 R：模型范畴、admissibility 与 rectification 案例

本附录把第十四章的模型范畴框架和附录 G 中定义 G.3--定义 G.6、外部输入定理 G.11--外部输入定理 G.13 的检查表应用到常见底范畴。它不是文献定理的替代品；作用是训练如何判断一个 operad-theoretic homotopy 断言是否具备足够假设。

## R.1 案例书写格式

每个案例按以下格式记录：

1. 底范畴 $\mathcal M$；
2. 模型结构；
3. monoidal compatibility；
4. operad/代数 admissibility 状态；
5. rectification 状态；
6. Bousfield localization 是否保持 operad/colored-operad algebras；
7. 禁止推出的结论。

## R.2 Simplicial sets

令
$$
\mathcal M=\mathbf{sSet}
$$
并采用 Kan-Quillen 模型结构，monoidal product 为 cartesian product。

**事实 R.1.** $\mathbf{sSet}$ 是 cofibrantly generated cartesian monoidal model category，所有对象 cofibrant。

**证明边界.** Kan-Quillen 模型结构和 cartesian monoidal compatibility 是标准外部基础事实。$\square$

**案例 R.2.** 对 simplicial operads，常见 transferred 模型结构逐 arity 检测 weak equivalences 和 fibrations。该结论依赖 Berger-Moerdijk 型外部输入。

**可用结论.** 在引用相应模型结构后，可以讨论 simplicial operad 的 weak equivalence、fibrant replacement 和 derived mapping space。

**不可用结论.** 不能仅由 $\mathbf{sSet}$ 良好推出：

1. 每个 simplicial operad 的 algebra category 都 admissible；
2. 任意 weak equivalent operads 的代数范畴都 Quillen equivalent；
3. simplicial operad 自动等同于 Lurie-style infinity-operad。

第三点必须经过外部输入定理 18.20、定义 19.24--外部输入定理 19.25 和规则 M.18 的比较路径。

## R.3 Compactly generated topological spaces

令
$$
\mathcal M=\mathbf{Top}_{cgwh}
$$
为 compactly generated weak Hausdorff spaces，采用 weak homotopy equivalences。

**事实 R.3.** 在合适模型结构下，$\mathbf{Top}_{cgwh}$ 与 $\mathbf{sSet}$ 由
$$
|-|:\mathbf{sSet}\rightleftarrows\mathbf{Top}_{cgwh}:\operatorname{Sing}
$$
构成 Quillen equivalence。

**外部输入.** 该事实属于 Quillen/Goerss-Jardine/May 传统基础。点集范畴若换成所有拓扑空间，结论形式可能改变。

**案例 R.4.** Topological operads 常需 well-pointed 或 $\Sigma$-cofibrant 条件来保证 Boardman-Vogt resolution、homotopy invariant algebra structures 和 transferred structures 表现良好。

**禁止简化.** 不应把“空间值 operad 逐 arity weak equivalence”直接替换为“其代数范畴同伦等价”。若 algebra objects 是 topological spaces、spectra 或 categories，仍需相应 admissibility 和 rectification 定理。

## R.4 Chain complexes over a characteristic zero field

令
$$
\mathcal M=\mathbf{Ch}_k
$$
其中 $k$ 是特征 $0$ 的域，采用 projective 模型结构：weak equivalences 为 quasi-isomorphisms，fibrations 为 degreewise surjections。

**事实 R.5.** 有限群代数 $k[G]$ 半单。因此 invariants 与 coinvariants exact。

**证明.** 这是 Maschke 定理。对有限群 $G$，$|G|$ 在 $k$ 中可逆，平均算子
$$
\frac1{|G|}\sum_{g\in G}g
$$
给出投影到 invariants 的 idempotent；半单性推出 coinvariants/invariants 的 exactness 性质。$\square$

**案例 R.6.** 在 $\mathbf{Ch}_k$、$\operatorname{char}k=0$ 中，许多 dg-operad 的 admissibility 和 rectification 定理可用。特别是 $\Sigma$-cofibrancy 与对称幂 exactness 风险显著降低。

**外部输入.** “许多”必须替换为具体定理：Hinich、Berger-Moerdijk、Fresse 或 Pavlov-Scholbach 的版本。不能把本案例当作全称定理。

**允许说法.** 若引用适当外部定理，$E_\infty$-dg algebras 与 commutative dg algebras 在特征 $0$ 中常可 rectification。

**禁止说法.** 即使在特征 $0$，也不能不说明模型结构、cofibrancy 和 operad map，就直接声明任意 $E_\infty$-algebra 等于 strict commutative dg algebra。

## R.5 Chain complexes over a general ring

令 $R$ 为一般交换环，$\mathcal M=\mathbf{Ch}_R$。

**问题 R.7.** 对有限群 $G$，$R[G]$ 未必半单，coinvariants functor
$$
(-)_G:\mathbf{Mod}_{R[G]}\to\mathbf{Mod}_R
$$
未必 exact。

**例 R.8.** 取 $R=\mathbb F_p$，$G=C_p$。平凡模 $\mathbb F_p$ 上 norm element
$$
N=\sum_{g\in C_p}g
$$
在特征 $p$ 中退化，平均投影不存在。于是 invariants/coinvariants 的特征 $0$ 证明失效。

**结论 R.9.** 在 $\mathbf{Ch}_R$ 上，涉及 symmetric powers 的自由 commutative algebra functor
$$
\operatorname{Sym}_R(X)=\bigoplus_{n\ge0}(X^{\otimes n})_{\Sigma_n}
$$
可能不保持 quasi-isomorphism。故 $\operatorname{Com}$-algebra 的 transferred model structure 或 $E_\infty\to\operatorname{Com}$ rectification 不能无条件使用。

**说明 R.10.** Associative dg algebras 的模型结构通常比 commutative dg algebras 更稳定，因为自由 associative algebra 使用
$$
T(X)=\bigoplus_{n\ge0}X^{\otimes n}
$$
不含 $\Sigma_n$ coinvariants。但完整转移仍需 monoid axiom、小性和生成 cofibration 假设。

## R.6 Nonnegative chain complexes

令
$$
\mathbf{Ch}^{\ge0}_R
$$
为非负同调次数链复形。

**说明 R.11.** 非负链复形中的 commutative dg algebras 与 simplicial commutative rings 有 Dold-Kan/monoidal refinement 背景，但普通 Dold-Kan correspondence 不是强对称幺半等价。涉及 commutative algebra 或 $E_\infty$-algebra 时，必须引用专门的 monoidal Dold-Kan 或 divided power 结果。

**禁止说法 R.12.** 不得写：
$$
\text{simplicial commutative }R\text{-algebras}
\quad=\quad
\text{nonnegative commutative dg }R\text{-algebras}
$$
除非说明底环、特征和所采用的修正结构。

## R.7 Symmetric spectra 与 structured ring spectra

设 $\mathcal M$ 是 symmetric spectra、orthogonal spectra 或类似 stable monoidal model category。

**案例 R.13.** Structured ring spectra 的 operad 代数理论通常需要 positive stable model structure、flat model structure 或其他避免单位对象 cofibrancy 问题的技术设置。

**外部输入.** 该案例依赖 symmetric spectra/orthogonal spectra 的专门文献。本书只把它作为 operadic admissibility 的典型警告，不在主体证明链中使用。

**禁止说法.** 不能把 $\mathbf{Ch}_k$ 中的 rectification 结论迁移到 spectra，而不检查 positive model structure 和 commutative monoid axiom。

## R.8 Colored operads 与 enriched categories

设 $C$ 是颜色集，$\mathcal O$ 是 $C$-colored operad。

**案例 R.14.** 即使单色 operads 在 $\mathcal M$ 中有良好模型结构，colored operads 的 admissibility 仍可能需要更强条件，因为自由 colored algebra 同时对颜色轮廓和对称群取 coproducts、coinvariants 与 pushouts。

**检查点.**

1. 颜色集 $C$ 是否 small；
2. 每个 profile 的 $\Sigma$-作用是否受控；
3. $\mathcal M^C$ 的 projective 模型结构是否存在；
4. 自由 colored algebra 是否保持生成 trivial cofibrations 所需的弱等价；
5. enriched Hom objects 是否在 $\mathcal M$ 中满足 fibrancy/cofibrancy 要求。

**结论 R.15.** 定义 K.16 的 enriched colored operad 语言不是外部输入定理 K.20 的 admissibility 定理。若要把 enriched categories 或 modules 的同伦理论模型化，必须引用 colored admissibility 结果。

## R.9 Rectification 的正例格式

一个 rectification 正例必须写成如下形式。

**模板 R.16.** 设 $\mathcal M$ 是满足条件 $(H)$ 的对称幺半模型范畴。设
$$
\varphi:\mathcal O\to\mathcal P
$$
是满足条件 $(C)$ 的 operad weak equivalence，且 $\mathcal O,\mathcal P$ admissible。则 extension-restriction adjunction
$$
\varphi_!:\operatorname{Alg}_{\mathcal O}(\mathcal M)
\rightleftarrows
\operatorname{Alg}_{\mathcal P}(\mathcal M):\varphi^\*
$$
是 Quillen equivalence。

其中 $(H)$ 和 $(C)$ 必须由具体文献定理给出，不能写成“良好条件”。

**例 R.17（特征零示意）.** 若 $k$ 是特征 $0$ 域，$\mathcal M=\mathbf{Ch}_k$，并引用 Hinich 或 Berger-Moerdijk/Fresse 的相应 rectification 定理，则某些 cofibrant $E_\infty$-operad 到 $\operatorname{Com}$ 的 weak equivalence 可诱导 $E_\infty$-algebras 与 commutative dg algebras 的 Quillen equivalence。

**证明边界.** 这里的关键工作全部在外部定理中：admissibility、symmetric flatness、cofibrancy 和 operad weak equivalence 的兼容性。本书不能用本例替代引用。$\square$

## R.10 Rectification 的反例格式

一个 rectification 失败或不可用案例应写成如下形式。

**模板 R.18.** 若缺少以下任一项：

1. $\mathcal O,\mathcal P$ admissible；
2. $\varphi$ 是适当类型的 operad weak equivalence；
3. 底范畴满足 symmetric flatness 或替代条件；
4. 自由代数中的对称幂保持 weak equivalences；
5. 代数对象 cofibrancy 假设；

则不得声明 rectification。

**例 R.19（正特征警告）.** 在 $\mathbf{Ch}_{\mathbb F_p}$ 中，即使存在 operad weak equivalence
$$
E_\infty\to\operatorname{Com}
$$
也不能由逐 arity quasi-isomorphism 单独推出
$$
\operatorname{Alg}_{E_\infty}(\mathbf{Ch}_{\mathbb F_p})
\simeq
\operatorname{Alg}_{\operatorname{Com}}(\mathbf{Ch}_{\mathbb F_p})
$$
的 Quillen equivalence。对称幂不 exact 和 power operations 是核心风险。
命题 X.15 和推论 X.16 给出 $\mathbb F_p$ 上 acyclic complex $C$ 但 $\operatorname{Sym}^p(C)$ 非 acyclic 的显式计算。

## R.11 与正文的使用关系

本附录提供案例判定，不新增全称定理。正文使用时应采取以下形式：

1. 若只需说明风险，引用本附录即可。
2. 若需证明模型结构存在，引用附录 D 中的外部输入定理。
3. 若需证明 rectification，必须给出具体文献定理。
4. 若需证明 Bousfield localization preserves operad/colored-operad algebras，应引用 WHT-1--WHT-4 或 WY-1--WY-3，并逐项检查 localized model structure、monoid axiom、cofibrancy 与颜色集假设。
5. 若底范畴不在本附录中，应新建案例，不得套用相邻情形。

## R.12 小结

模型范畴中的 operad 理论依赖底范畴。最常见错误是把特征 $0$ 链复形中的 rectification 直觉迁移到一般环、正特征、spectra、colored operads 或 enriched categories，或把 localization preservation 误当成 infinity-categorical algebra comparison。正确做法是逐案例检查 monoidal model structure、对称群作用、自由代数、admissibility、rectification 定理和 localization preservation 定理。
