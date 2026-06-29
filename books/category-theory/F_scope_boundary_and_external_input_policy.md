# 附录 F：范围边界与外部输入政策

本附录规定后续扩写的范围边界。它是写作约束的一部分。

## F.1 本书的本体范围

本书的本体内容是范畴论及其内部高阶化，包括：

- 普通范畴、函子、自然变换、极限、伴随、Kan 延拓、单子和可表性。
- 幺半、闭、富、indexed、fibered、2-范畴、双范畴、profunctor、equipment。
- 可表现、可达、正合、topos、sketch、doctrine、逻辑语义。
- 模型无关同伦范畴论、quasi-category、Cartesian fibration、稳定 $\infty$-范畴、高阶 topos、高阶代数。
- 范畴论内部的 completion、localization、factorization system、Morita、derivator、$\infty$-cosmos。

这些主题应尽量给出定义、核心命题、证明、例子和练习。

## F.2 外部接口的处理

若某章涉及代数几何、稳定同伦论、$D$-module、motivic homotopy、chromatic homotopy、Langlands、解析几何或其他外部数学理论，本书只处理其中的范畴论接口：

1. 说明所用范畴、函子、自然变换、伴随、局部化、six operations 或高阶结构。
2. 把该外部领域的深定理标为“外部输入定理”。
3. 在 `SOURCES.md`、`D_theorem_source_index.md` 和 `THEOREM_DEPENDENCIES.md` 中记录来源。
4. 不在本书内部补外部领域定理的证明。

例如，Riemann-Hilbert correspondence、Hopkins-Smith thick subcategory theorem、motivic six functor formalism 和 Dundas-Goodwillie-McCarthy theorem 都不是本书内部需闭合的定理。

## F.3 后续扩写优先级

后续扩写优先级按如下顺序：

1. 补范畴论本体缺口。
2. 强化已有核心章节的证明、例题、反例和练习答案。
3. 梳理内部依赖链和术语统一。
4. 仅在必要时补外部接口说明，不展开外部理论。

## F.4 审查规则

新增内容若满足以下任一条件，应推迟或只作外部输入记录：

- 主要证明依赖具体几何、数论、分析或稳定同伦论技术，而非范畴论结构。
- 章节目标是某外部理论本身，而非其范畴论形式主义。
- 需要大量领域专门计算才能成立。

新增内容若属于以下情形，应优先加入：

- 给出范畴论内部构造的泛性质。
- 建立不同范畴论结构之间的等价或伴随。
- 澄清模型、大小、相干性、自然性或闭包条件。
- 增加核心概念的例子、反例和边界条件。
