# 数学审查记录

本文档记录《范畴论》教材的审查清单、当前风险和后续校订任务。

## 全书审查清单

- [ ] 每章是否声明依赖前置知识。
- [ ] 所有新增符号是否进入 `NOTATION.md`。
- [ ] 每个定义是否说明所在宇宙或范畴。
- [ ] 每个非平凡命题是否带完整证明或外部输入来源。
- [ ] 每个泛性质是否写成自然同构、终对象/始对象或表示性陈述。
- [ ] 是否避免把“同构”“等价”“同伦”等词混用。
- [ ] $\infty$-范畴章节是否说明模型口径。
- [ ] 外部输入定理是否进入 `SOURCES.md` 或章节末尾来源表。
- [ ] 每个练习是否在 `SOLUTIONS.md` 中有对应答案。
- [ ] 综合题是否在 `COMPREHENSIVE_SOLUTIONS.md` 中有对应答案。
- [ ] 新术语是否进入 `TERM_INDEX.md`。
- [ ] 新外部输入定理是否进入 `THEOREM_DEPENDENCIES.md`。
- [ ] 章节资料源是否进入 `CHAPTER_SOURCE_NOTES.md`。

## 当前初稿风险

- 第一章的“完全忠实且本质满推出存在拟逆”使用选择原则；附录 A 已记录，但后续若形式化需明确选择强度。
- 第五章预层密度定理、第六章 Kan 点态公式、第十一章 co-Yoneda 已补为正文证明；后续仍可增加更多例子和变体。
- 第八章 Mac Lane 相干性、第十二章局部可表现范畴伴随函子定理、第十四章 sheaf 化/Giraud、第十六章模型范畴局部化、第十九章 straightening、第二十章稳定同伦范畴三角结构、第二十一章高阶 Giraud、第二十二章高阶代数存在性定理均为外部输入。
- 第十二至第十四章已补预层范畴局部可表现性、强生成子、image/coimage、正合函子、模范畴 Grothendieck 性、separated 预层、plus 构造和几何态射；plus 构造完整证明与 Giraud 定理仍作为外部输入。
- 第十七至第十九章已补入标准单纯形计算、左右映射空间模型、correspondence 表示性口径、adjunction data 低维展开、walking adjunction、scaled nerve 低维口径、ordinary pullback 恢复、普通 Grothendieck construction、基为 $[1]$ 和 $[2]$ 的 straightening 低维模型、Cartesian 传输函子与 Cartesian sections；完整 scaled model structure 仍作为外部输入。
- 第二十章已补 sequential prespectrum、$\Omega$-谱、映射谱、smash product、悬挂-环路互逆、正合函子、t-结构 heart 核余核、heart 加性、cohomology 长正合列、exact couple、有限滤过谱序列收敛、完备滤过和条件收敛入口；heart 阿贝尔性和一般收敛理论仍作为外部输入。
- 第二十一章已补超覆盖、超下降、effective epimorphism、groupoid object、Postnikov tower、hypercompletion、$\infty$-几何态射和点；hyperdescent 与不同 sheaf 条件的等价、groupoid objects 有效性和 hypercompletion 理论仍作为外部输入。
- 第二十二章已补 Segal 条件、多重映射空间、模 $\infty$-范畴、bar 构造、相对张量积、Morita、单位双模、矩阵代数 Morita 等价、smooth/proper 可对偶性判别、Frobenius 代数二维 TFT 影子、中心、因子化同调、fully dualizable objects 和 cobordism hypothesis 入口；更高维 fully extended TFT 的具体计算仍未展开。
- 第二十三章已补 presentable $\infty$-category、$\operatorname{Ind}_\kappa$、accessible localization、Bousfield localization、left exact/exact localization 与 $\operatorname{Pr}^L$；Ind 刻画、伴随函子定理和 $\operatorname{Pr}^L$ 幺半结构仍为外部输入。
- 第二十四章已补 profunctor、coend 复合、Cauchy completion、加权余极限和 $\infty$-correspondence；$\mathbf{Prof}$ 双范畴相干性和高阶 correspondence 的 $(\infty,2)$-结构仍为外部输入。
- 第二十五章已补富 profunctor、equipment、companion/conjoint、Beck-Chevalley 条件、indexed category 与 fibration 比较；高阶 equipment/framed bicategory 模型仍为外部输入。
- 第二十六章已补 compact generation、Brown representability、localizing subcategory、Verdier quotient、Bousfield localization、smashing localization 和 Neeman-Thomason 型定理；Brown 表示性和紧对象商定理仍为外部输入。
- 第二十七章已补 dg category、dg modules、dg Yoneda、pretriangulated enhancement、Morita equivalence、dg bimodules、perfect modules 和 Hochschild chains；dg 模模型结构、dg nerve 稳定性、导出 Morita 定理和 Hochschild 型 Morita 不变性仍为外部输入。
- 第二十八章已补六操作形式主义、基变换态射、投影公式、proper compatibility、recollement、Verdier duality 和 equipment 比较；具体几何理论中的六操作存在性、基变换定理、投影公式、purity 和 Verdier 对偶仍为外部输入。
- 第二十九章已补 relative category、$\infty$-categorical localization、saturation、simplicial category、Dwyer-Kan equivalence、underlying $\infty$-category、coherent nerve、complete Segal space 和模型选择原则；simplicial localization、hammock localization、Bergner-Joyal 比较和 Rezk CSS 模型比较仍为外部输入。
- 术语索引、章节来源注释和外部输入依赖图已经建立；后续新增章节必须同步维护这些文件。

## 下一轮建议

1. 继续扩充第十七至第十九章：给出更多 slice 计算、mapping space 之间的模型比较、HTT adjunction data 的完整定义和 scaled model structure 的技术附录。
2. 扩充每章例题，并把 `SOLUTIONS.md` 从答案要点升级为逐步解答。
3. 为每章增加“本章术语”和“来源注释”内嵌小节，把当前独立索引回填到正文。
4. 继续扩写第二十至第二十二章：一般 t-结构 heart 完整证明、非有限滤过谱序列收敛定理、Morita $(\infty,2)$-范畴的高阶复合细节和具体 fully extended TFT 例子。
5. 扩写第十二至第十四章的例子：locally presentable categories 的更多代数实例、Gabriel-Popescu 的证明路线、具体站点上的 sheaf 化计算。
6. 扩写第二十三章：加入 compactly generated presentable categories、稳定 presentable 范畴的 t-结构可达性和具体 Bousfield localization 计算。
7. 扩写第二十四至第二十五章：加入 exact square 的更多例子、six functor formalism、富 Cauchy completion 和与 $(\infty,2)$-Morita 范畴的系统比较。
8. 扩写第二十六章：加入 chromatic homotopy 中的 Bousfield lattice、telescope conjecture 背景和 derived algebraic geometry 中的 compact generation 例子。
9. 扩写第二十七章：加入 dg quotient、Drinfeld quotient、唯一增强定理的适用条件、非交换 motives 和 localizing invariants 的系统比较。
10. 扩写第二十八章：加入具体 sheaf 理论中的 proper base change 证明路线、étale/motivic 六操作的假设表、nearby cycles 和 vanishing cycles。
11. 扩写第二十九章：加入 marked simplicial categories、relative nerve 的显式公式、Barwick-Kan 模型结构和 Rezk completion 的具体计算。
