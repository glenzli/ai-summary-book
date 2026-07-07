# 第二十三章：2024-2026 研究边界与开放问题目录

## 本章目标

本章给出截至 2026-07-08 的研究边界目录。它不把近期预印本变成教材核心定理，而是记录哪些方向可以后续进入正文，进入前需要哪些验证。

## 依赖前置知识

需要本书第一至第二十二章的主线结构和 `FRONTIER_SOURCE_AUDIT_2026_07_08.md`。

## 23.1 成熟度分级

**定义 23.1.** 本书把研究前沿按教材可用性分为三类。

1. **核心定理级。** statement、假设、证明源和本书符号翻译均已锁定，可以作为后续命题的输入。
2. **结构接口级。** 与本书主线有清楚函子、范畴或几何对象接口，但关键证明仍作为外部输入或需要专门模型。
3. **边界方向级。** 结果代表活跃前沿，但本书只记录问题、术语和进入正文的条件，不把它用于证明链。

**原则 23.2.** 第三类结果不得出现在任何证明的“因为某前沿理论”步骤中。若必须使用，先把它提升到第一类或第二类，并在 `THEOREM_LEDGER.md` 和 `D_source_theorem_index.md` 登记。

## 23.2 当前前沿方向

**方向 23.3.** Geometric Langlands proof series。  
本方向连接第十三章 geometric Satake、第十五章 critical Kac-Moody localization 和第十六章 Hecke eigensheaves。教材入口是 automorphic side 与 spectral side 的范畴对应：
$$
\mathsf D(\operatorname{Bun}_G)\longrightarrow \operatorname{IndCoh}_{\mathcal N}(\operatorname{LocSys}_{G^\vee}).
$$
当前可作为结构接口使用的内容是 Hecke 作用、spectral category、singular support 和 factorization/localization 的术语网络；完整 equivalence 不作为本书内部定理。

**方向 23.4.** Coulomb branches 和 symplectic duality。  
本方向连接第十九章 conical symplectic resolutions、第二十章 BFN convolution algebra 和第十八章 KLR/Yangian 型范畴化。教材入口是
$$
\mathcal M_C(G,N)=\operatorname{Spec}H_\ast^{G(\mathcal O)}(\mathcal R)
$$
及其量子化 $\mathcal A_\hbar$。可在正文中使用的是 BFN 构造的定义、torus/pure gauge 计算和与 affine Grassmannian slices 的已验证同构；3d mirror symmetry 只作为组织性猜想或外部解释。

**方向 23.5.** Modular representation theory、parity sheaves 和 torsion phenomena。  
本方向修正第四章中基于 characteristic zero IC sheaves 的 KL 图像。若系数域特征为 $p>0$，IC sheaves、parity sheaves 和 tilting characters 的关系不能由 characteristic zero 的 decomposition theorem 自动推出。教材入口应包括 parity sheaf 的定义、even stratification、intersection form 和 $p$-canonical basis。

**方向 23.6.** CoHA、DT theory 和 Yangian。  
本方向连接第二十一章 Hall/CoHA 接口、第二十二章 canonical bases 和 quiver variety 表示。进入正文时必须先固定 vanishing cycles、orientation data、critical CoHA 与 BPS Lie algebra 的模型。没有这些数据时，只能使用普通 Hall algebra 或非 critical CoHA 的简化版本。

**方向 23.7.** Categorical and spectral representation theory。  
本方向把 Soergel bimodules、KLR 2-representations、monoidal dg categories、spectral categories 和 Morita theory 放入统一框架。教材入口是“作用”而非“元素”：一个 monoidal category $\mathcal C$ 作用在 category $\mathcal M$ 上，是 monoidal functor
$$
\mathcal C\longrightarrow \operatorname{End}(\mathcal M).
$$
本书已在 Hecke category、Satake category 和 KLR categories 中使用该模式，但高阶 Morita 理论只作为边界工具。

## 23.3 进入正文的验证流程

**流程 23.8.** 任一前沿结果进入正文定理链前，必须完成：

1. 确认版本、作者、日期、发表状态或 arXiv 版本；
2. 抽取准确 theorem statement；
3. 翻译所有假设到本书 notation；
4. 检查是否依赖未引入的模型；
5. 在 `D_source_theorem_index.md` 加 locator；
6. 在 `THEOREM_LEDGER.md` 标明外部输入；
7. 若只作边界说明，不得用于证明后续命题。

**例 23.9.** 若要把某个 geometric Langlands equivalence statement 放入第十六章，至少要登记：

1. automorphic category 使用 ordinary D-modules、renormalized D-modules 还是 ind-completion；
2. spectral side 是 $\operatorname{QCoh}$、$\operatorname{IndCoh}$ 还是带 nilpotent singular support 的子范畴；
3. 曲线 $X$ 的底域、光滑性、完备性和 characteristic；
4. 群 $G$ 是否 reductive、connected、split；
5. Hecke action 的 normalization；
6. 该 statement 是 fully faithful、essential surjectivity、equivalence 还是 compatibility theorem。

**命题 23.10.** 若一个前沿结果缺少 23.8 中任一项，则它不能作为本书内部证明的输入。

**证明.** 本书内部证明链要求每个外部输入可替换为明确的 theorem statement。缺少版本会造成 statement 漂移；缺少假设翻译会造成对象不在同一范畴；缺少 locator 会使读者无法验证引用；缺少 ledger 标记会把外部定理误认为内部证明。因此任一缺项都会破坏证明链的可审查性。$\square$

## 23.4 开放问题地图

**问题 23.11.** 如何给出 geometric Langlands proof series 的教材化路径，使其在不牺牲严格性的前提下进入研究生教材？

**问题 23.12.** 哪些 Coulomb branches 承认 symplectic resolutions，哪些只应作为 singular affine Poisson varieties 处理？

**问题 23.13.** parity sheaves 和 torsion phenomena 如何系统修正特征 $0$ 的 IC/KL 叙述？

**问题 23.14.** CoHA、KLR、Yangian 和 Coulomb branch 的 canonical basis 结构能否在统一的 monoidal category 中表达？

**问题 23.15.** 能否给出一个覆盖 Hecke categories、Satake categories、KLR 2-categories 和 spectral categories 的统一 Morita-theoretic 教材框架，同时保持初学者可进入的计算例子？

## 本章小结

本章完成主体目录的研究边界收口：前沿材料被分为核心定理级、结构接口级和边界方向级。正文只允许前两类进入证明链，第三类只作为开放问题和后续扩展入口。

## 练习

**练习 23.1.** 任选一个前沿结果，按流程 23.8 写出进入正文前的验证表。

**练习 23.2.** 说明为什么 2024 geometric Langlands proof series 不能直接作为第十六章定理使用。

**练习 23.3.** 选择 Coulomb branch 或 symplectic duality 的一个例子，列出需要核查的几何假设。

**练习 23.4.** 对 parity sheaves，说明为什么 characteristic zero 的 IC-sheaf argument 不能直接推出 $p$-canonical basis statement。

**练习 23.5.** 把一个 monoidal category action 写成 $\mathcal C\to\operatorname{End}(\mathcal M)$，并分别指出 Hecke category 和 KLR category 中的 $\mathcal C$ 与 $\mathcal M$。
