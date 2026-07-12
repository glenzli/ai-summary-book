# 附录 O：失败模式、反例边界与不可混用约定

本附录记录 operad theory 教材中最常见的错误命题。每条都给出失败原因和可用替代说法。目的不是收集病态例子，而是把正文证明链中不能省略的假设固定下来。

## O.1 Arity $0$ 与单位

**错误命题 O.1.** “operad 是否允许 arity $0$ 不影响代数概念。”

**失败原因.** 若 $\mathcal O(0)$ 非空，则 $\mathcal O$-代数含 nullary operations。对 $\operatorname{Com}$，arity $0$ 元素给出交换代数的单位；若去掉 arity $0$，得到的是非含单位交换代数。自由代数公式也随之改变：
$$
\bigoplus_{n\ge0}\mathcal O(n)\otimes_{\Sigma_n}V^{\otimes n}
$$
中的 $n=0$ 项正是常量项来源。

**正确边界 O.2.** 本书基础 operad 默认允许 arity $0$。二次 Koszul 理论中若采用 reduced 非含单位口径，必须显式写出 reduced/augmented convention。

## O.2 左右对称群作用

**错误命题 O.3.** “把右 $\Sigma_n$-作用改成左 $\Sigma_n$-作用只是记号差异，不会影响公式。”

**失败原因.** 代入乘积的 arity 公式使用 coinvariants：
$$
(M\circ N)(n)\cong
\bigoplus_{r\ge0}
M(r)\otimes_{\Sigma_r}
\left(
\bigoplus_{n_1+\cdots+n_r=n}
\operatorname{Ind}_{\Sigma_{n_1}\times\cdots\times\Sigma_{n_r}}^{\Sigma_n}
N(n_1)\otimes\cdots\otimes N(n_r)
\right).
$$
若左右作用未转换，诱导表示、coinvariants 和复合中的置换方向会不匹配。

**正确边界 O.4.** 本书以有限集群胚 $\mathbf B_{\mathcal U}$ 上的函子为默认定义；arity 公式只作为派生公式使用。若转到 $[n]$ 和 $\Sigma_n$，必须使用命题 A.9 和约定 A.10 的左右作用转换。

## O.3 Coinvariants 与 invariants

**错误命题 O.5.** “$(V^{\otimes n})_{\Sigma_n}$ 与 $(V^{\otimes n})^{\Sigma_n}$ 可以互换。”

**失败原因.** 只有在 $|\Sigma_n|$ 可逆等条件下，averaging idempotent 给出 invariants 与 coinvariants 的同构。若底环为正特征或一般交换环，该同构可能失败，且 coinvariants 不保持 exactness。

**正确边界 O.6.** 在线性 operad、Schur functor、commutative algebra 和 rectification 中，必须记录底环特征。特征 $0$ 上的直觉不能自动迁移到正特征。

## O.4 Strict commutative 与 $E_\infty$

**错误命题 O.7.** “$E_\infty$-algebra 总能替换成 strict commutative dg algebra。”

**失败原因.** Rectification 需要强假设，例如合适的模型结构、cofibrancy、symmetric flatness 或特征 $0$ 条件。在正特征中，$E_\infty$-algebras 可携带 power operations 等严格交换 dg algebra 不记录的信息。

**正确边界 O.8.** 只有在定义 G.3--定义 G.6 和外部输入定理 G.11--外部输入定理 G.13 的 rectification 检查表通过后，才能使用 strict commutative 模型替代 $E_\infty$-模型。否则必须保持 $E_\infty$ 结构。

## O.5 $\mathcal P_\infty$ 的含义

**错误命题 O.9.** “$\mathcal P_\infty$ 表示任意 cofibrant replacement。”

**失败原因.** 文献中 $\mathcal P_\infty$ 常有两种含义：

1. Koszul 情形下的 $\Omega\mathcal P^¡$；
2. 某个未指定的 cofibrant resolution $Q\mathcal P\to\mathcal P$。

二者不一定逐项相同，也不携带同一套生成元、符号和树公式。

**正确边界 O.10.** 本书中
$$
\mathcal P_\infty=\Omega\mathcal P^¡
$$
只在 Koszul 语境下使用。任意 cofibrant replacement 写作 $Q\mathcal P$。

## O.6 Chains on spaces

**错误命题 O.11.** “对拓扑 operad 逐项取 chains 自动得到严格对称幺半 functor，因此不需要额外相干性。”

**失败原因.** Singular chains 通常是 lax symmetric monoidal 或 $E_\infty$-monoidal 结构；Eilenberg-Zilber 与 Alexander-Whitney maps 有方向和相干性选择。若忽略这些选择，operad composition 的链级相干性无法检查。

**正确边界 O.12.** 从 topological operad 到 dg-operad 时，必须指定 chains functor 的 monoidal model，并说明得到的是 dg-operad、$E_\infty$-coalgebra enriched 对象还是只在 homotopy category 中定义的结构。

## O.7 $E_n$ 形式性

**错误命题 O.13.** “$E_n$-operad 都形式。”

**失败原因.** 形式性结论依赖系数域、特征、维数和所选链模型。经典 Kontsevich/Tamarkin 型形式性在特征 $0$ 的特定设置中成立；正特征中存在额外同调操作，不能套用同一结论。

**正确边界 O.14.** 本书把 $E_n$ 形式性全部标为外部输入，并在使用处写明系数和链模型。

## O.8 Dendroidal inner Kan 与 strict operad

**错误命题 O.15.** “dendroidal inner Kan object 就是 strict operad 的 nerve。”

**失败原因.** Strict operad 的 dendroidal nerve 具有唯一 inner horn fillers。一般 dendroidal inner Kan object 只要求 fillers 存在；不同 fillers 记录同伦相干选择。

**正确边界 O.16.** Strict operads 嵌入 dendroidal sets，但 homotopy operads 是更大的对象类。唯一填充与存在填充必须分开。

## O.9 Dendroidal 模型与 Lurie 模型

**错误命题 O.17.** “dendroidal infinity-operad 与 Lurie-style infinity-operad 是同一个定义。”

**失败原因.** 前者是 dendroidal set 上的 inner Kan/operadic model structure 语言；后者是映到 $N(\mathbf{Fin}_*)$ 的 simplicial set，带 inert/coCartesian 条件。二者通过比较定理相连，而不是字面相同。

**正确边界 O.18.** 从 dendroidal 模型移动到 Lurie 模型，必须引用 Heuts--Hinich--Moerdijk 或相应比较定理，并记录模型结构版本。HHM-1--HHM-5 的本书 locator 只覆盖 open/no-constants 路径；含 arity $0$ 的对象必须另行处理。

## O.10 Localization 与取代数

**错误命题 O.19.** “先取 $\mathcal O$-代数再 localization，与先 localization 再取 $\mathcal O$-代数总是等价。”

**失败原因.** 代数范畴模型结构的存在依赖 admissibility；localization 是否保持 monoidal structure 也需条件。即使两边都存在，比较 functor 成为 equivalence 仍需 cofibrancy、rectification 或 operadic localization 定理。

**正确边界 O.20.** 必须使用外部输入定理 19.25 和规则 M.18 的 algebra localization comparison 路径，并避开警告 M.19 的禁止捷径。没有这些条件时，只能得到一个候选比较函子，不能声明等价。

## O.11 Properad 与 trace

**错误命题 O.21.** “endomorphism properad 自动有 wheeled contraction。”

**失败原因.** Wheeled contraction 要把输出与输入相连并取 trace。在线性范畴中，trace 通常要求有限生成投射、dualizable 或 compact 条件。无限维链复形或一般对象上 trace 未必定义。

**正确边界 O.22.** 例 7.19 的 wheeled endomorphism 结构只在 trace 已定义的对象上成立；警告 7.20 说明一般对象上不能自动使用 trace。

## O.12 Factorization homology 与普通同调

**错误命题 O.23.** “$\int_MA$ 就是 $M$ 的普通同调，系数为 $A$。”

**失败原因.** Factorization homology 的系数是 $E_n$-algebra，而不是 abelian group local system。对 $E_1$-algebra，
$$
\int_{S^1}A\simeq HH_\*(A),
$$
一般不是 $H_\*(S^1;A)$。

**正确边界 O.24.** 只有在交换系数和额外 tensoring over spaces 条件下，factorization homology 才退化到 higher Hochschild chains 或类似普通同调的表达式。

## O.13 带边界流形

**错误命题 O.25.** “$\int_{D^n}A\simeq A$ 可直接用于闭半球、带边界圆盘和所有 disk-like pieces。”

**失败原因.** $\mathbb R^n$ 是无边界开 disk；带边界圆盘需要 disk-stratified、manifold with boundary 或 module boundary condition 的版本。边界 strata 会引入 module 或 algebra action 数据。

**正确边界 O.26.** 使用带边界块时必须指定边界条件。无边界 disk 归一化不能替代带边界理论。

## O.14 Fukaya category 的形式来源

**错误命题 O.27.** “Fukaya category 的 $A_\infty$ relations 可由 operad 公理形式推出，因此不需要分析。”

**失败原因.** Operad 公理可组织 operations，但 operations 本身来自 holomorphic polygons 的计数。计数可定义性、紧化边界、符号、orientation 和 transversality 都是辛几何分析内容。

**正确边界 O.28.** Operad theory 负责表达结构；Fukaya theory 的构造和 gluing 定理必须作为外部输入。

## O.15 预印本与研究边界

**错误命题 O.29.** “最新 arXiv 结果可以直接写入教材主定理链。”

**失败原因.** 预印本版本可能变化，定理假设可能与本书模型约定不同，且证明依赖未必与前文兼容。若没有核查定理编号、版本、模型和假设，主定理链无法审校。

**正确边界 O.30.** 2025-2026 文献先进入第二十一章的流程 21.16 和 `FRONTIER_SOURCE_AUDIT_2026_06_30.md`。进入正文定理链前，必须补齐版本、定理编号、模型约定和依赖路径。

## O.16 小性与宇宙

**错误命题 O.31.** “所有 operad、代数范畴和 functor categories 都可以放在同一个集合宇宙中。”

**失败原因.** 对 $\mathcal U$-小颜色集、$\mathcal V$-小范畴、presentable infinity-categories 和 functor categories 的大小控制不同。若不升宇宙，某些“所有对象构成的范畴”不是 $\mathcal U$-小。

**正确边界 O.32.** 本书固定
$$
\mathcal U\in\mathcal V\in\mathcal W.
$$
构造对象时必须说明所在宇宙；把大范畴当作小范畴使用前必须作骨架化或升宇宙。

## O.17 “自然”等于“无选择”

**错误命题 O.33.** “一个构造自然，因此不依赖任何选择。”

**失败原因.** 在 homotopy 或 infinity 语境中，构造常依赖 cofibrant replacement、fibrant replacement、cellular decomposition 或 perturbation data。正确说法通常是选择空间可缩、不同选择由 contractible space of equivalences 连接，或在 localization 后给出同一对象。

**正确边界 O.34.** 本书中“唯一”必须区分严格唯一、up to unique isomorphism、up to contractible choice 和在 homotopy category 中唯一。

## O.18 Boardman-Vogt tensor product

**错误命题 O.35.** “$E_m\otimes E_n=E_{m+n}$ 是逐项同构。”

**失败原因.** Dunn additivity 给出的是合适模型中的 weak equivalence 或 infinity-operad equivalence。不同 $E_n$ 模型之间通常只由 zigzag of weak equivalences 连接。

**正确边界 O.36.** 使用 additivity 时写作
$$
E_m\otimes E_n\simeq E_{m+n}
$$
并说明等价所在模型。若需要点集级等式，必须选择专门模型并证明兼容性。

## O.19 Operadic suspension signs

**错误命题 O.37.** “$A_\infty$、$L_\infty$ 的符号只差一个整体正负号。”

**失败原因.** 符号来自 suspension、Koszul braiding、shuffle permutation、operadic suspension 和同调/上同调分次约定的组合。换一个 convention 会改变每一项的指数。

**正确边界 O.38.** 本书采用同调分次，$|m_n|=|\ell_n|=n-2$。具体展开必须由定义 E.18--定义 E.23、定义 J.5--外部输入定理 J.17 和定义 L.4--定义 L.7 的 suspended coderivation 口径推出。

## O.20 章节使用规则

每当正文出现如下跨模型语句，必须回到本附录检查：

1. strict operad $\leftrightarrow$ infinity-operad；
2. topological operad $\leftrightarrow$ dg-operad；
3. $E_\infty$ $\leftrightarrow$ strict commutative；
4. factorization algebra $\leftrightarrow$ $E_n$-algebra；
5. Fukaya gluing $\leftrightarrow$ operadic composition；
6. model category algebra $\leftrightarrow$ infinity-category algebra；
7. coinvariants $\leftrightarrow$ invariants；
8. arity $0$ included $\leftrightarrow$ reduced nonunital convention。

若没有相应比较定理、模型假设或外部输入，相关句子只能作为说明，不能作为证明步骤。

## 练习

**练习 O.1.** 在正特征域上给出 coinvariants functor 不 exact 的一个群表示例子。

**练习 O.2.** 说明为什么 strict commutative dg algebra 不能无条件表达所有 $E_\infty$-algebra 信息。

**练习 O.3.** 比较 strict operad nerve 的唯一 inner horn filler 与一般 dendroidal inner Kan object 的存在性 filler。

**练习 O.4.** 对 $A$ 为 associative algebra，解释 $\int_{S^1}A$ 与 $H_\*(S^1;A)$ 的差异。

**练习 O.5.** 找出第十章到第二十章中任意一个跨模型语句，并列出使其成为定理所需的假设。
