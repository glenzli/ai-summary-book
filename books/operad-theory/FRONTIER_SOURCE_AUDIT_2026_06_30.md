# 前沿文献版本核查记录：2026-06-30

本文件补充 [FRONTIER_SOURCE_AUDIT_2026_06_29.md](FRONTIER_SOURCE_AUDIT_2026_06_29.md)。核查来源为官方 arXiv 摘要页，核查日为 2026-06-30。本文档只记录版本、对象、逻辑位置和进入正文前的验证义务；它不把任何前沿预印本中的新结果提升为本书定理。

## 0. 使用规则

**规则 0.1.** 本文件中的条目只能在第二十一章和附录 D 中作为“研究边界”或“已登记的边界 locator”出现。若正文第八至二十章需要使用其中某个结论，必须额外记录：

1. arXiv 版本号或出版版本。
2. 精确的定理、命题、定义或构造编号。
3. 该结论使用的模型：strict operad、colored operad、dg operad、dendroidal set、Lurie-style infinity-operad、relative infinity-operad 或 operadic category。
4. 该模型与本书模型之间的比较函子或等价定理。
5. 结论是否依赖未被本书证明的模型结构、localization、analysis 或 transversality 输入。

**规则 0.2.** 如果某条目只有预印本版本而没有出版版本，本书不得把其中新结果作为基础章节的无条件定理。允许的用法只有：

- 研究边界说明；
- 开放问题入口；
- 与既有经典定理的定义比较；
- 附录 D 中新建的外部输入包。

## 1. 版本表

| 条目 | 官方 arXiv 状态 | 本书逻辑位置 | 当前使用状态 |
|---|---:|---|---|
| Eric Hoffbeck and Ieke Moerdijk, “Homology of infinity-operads” | arXiv:2105.11943，v1 submitted 2021-05-25 | 第十八、十九章之后；infinity-operad 的同调、bar-cobar 和 Koszul 型背景 | 研究边界与背景；不得替代第九章经典 bar-cobar/Koszul 定理 |
| Eric Hoffbeck and Ieke Moerdijk, “Koszul duality for algebras over infinity-operads” | arXiv:2602.08851，v1 submitted 2026-02-09 | 第八、九、十八、十九章之后；linear infinity-operad 上代数与余代数的 Koszul 对偶 | 研究边界；进入正文前必须比较 classical operad algebra 与 linear infinity-operad algebra |
| Daria Pavlova, “Boardman-Vogt tensor product and wreath product of operadic categories” | arXiv:2601.03985，v1 submitted 2026-01-07，v2 revised 2026-05-28 | 第五、七、十四、二十一章；operadic categories 与 Boardman-Vogt tensor product | 研究边界；可用于指出 operadic Grothendieck construction 的近期方向 |
| Hang Yuan, “Higher operad structure for Fukaya categories” | arXiv:2603.08039，v1 submitted 2026-03-09 | 第二十、二十一章；Fukaya categories 的高阶 operadic structure | 研究边界；任何几何结论必须另行引用 Fukaya category 的分析输入 |
| Kensuke Arakawa, Victor Carmona, and Francesca Pratali, “Relative dendroidal Rezk nerve and applications” | arXiv:2606.11895，v1 submitted 2026-06-10 | 第十七、十八、十九、二十、二十一章；relative infinity-operads 与 operadic localization | 研究边界；不得直接并入第十九章 localization 定理链 |
| Michael Batanin, Joachim Kock, and Mark Weber, “Operadic categories as (pseudo)-simplicial groupoids” | arXiv:2606.15671，v1 submitted 2026-06-14 | 第五、十六、十八、二十一章；operadic category 的 operadic nerve | 研究边界；进入正文前必须补 2-范畴/伪 simplicial 相干性背景 |

## 2. 与本书定理链的关系

### 2.1 Infinity-operad 的同调与 Koszul 对偶

Hoffbeck--Moerdijk 的 2021 条目与 2026 条目属于本书 classical Koszul duality 之后的前沿扩展。第八、九章当前的定理链只依赖 classical dg operad、cooperad、twisting morphism 和 bar-cobar 构造。若未来把 infinity-operadic Koszul duality 写入正文，需要新增一个独立章节，而不是把它压入第九章的 classical bar-cobar 证明。

进入正文前的最小检查清单为：

1. 明确 linear infinity-operad 是树范畴上的何种预层或线性化对象。
2. 写出 classical operad 进入 linear infinity-operad 语境的全忠或弱等价模型。
3. 比较该文的 bar/cobar 构造与本书第九章 $\mathrm B,\Omega$ 的权重滤过。
4. 检查代数范畴上的 Koszul 对偶是否给出 Quillen equivalence、derived equivalence，还是更一般的 adjunction/duality。

### 2.2 Operadic categories 与 tensor product

Pavlova 的 2026 条目和 Batanin--Kock--Weber 的 2026 条目都属于 operadic categories 方向。它们不能直接替换本书第五章的 colored operad 定义，因为本书第五章使用的是颜色集、输入有限集、colored substitution coend 和 multicategory 的严格一范畴口径。

进入正文前的最小检查清单为：

1. 写出 operadic category 的 chosen local terminals、fiber functor 和 cardinality functor。
2. 说明 operadic Grothendieck construction 如何从 colored operad 产生 operadic category。
3. 检查 Boardman--Vogt tensor product 在 `Set`-valued colored operads 中的适用范围。
4. 对 operadic nerve 的 pseudo-simplicial groupoid 结构给出独立的相干性约定。

### 2.3 Relative dendroidal Rezk nerve

Arakawa--Carmona--Pratali 的 2026 条目处在第十七至十九章之间：它同时触及 dendroidal objects、relative infinity-operads 和 localization。第十九章目前采用 relative categories、Dwyer--Kan localization、straightening/unstraightening 和 operadic localization 的分层叙述；该预印本不能被用来替代这些基础输入。

进入正文前的最小检查清单为：

1. 定义 relative infinity-operad，并与 Lurie-style infinity-operad、dendroidal inner Kan object 分别比较。
2. 写明 dendroidal Rezk nerve 的源、靶和 fibrancy 条件。
3. 标出与 Mazel-Gee 型 localization theorem 的精确关系。
4. 分离其在 cyclic operads、operadic modules 和 factorization algebras 中的应用，不把应用结论倒用为基础定理。

### 2.4 Fukaya categories 的高阶 operadic structure

Yuan 的 2026 条目属于几何应用前沿。它可以作为第二十章之后的研究入口，但不能替代 Fukaya category 的分析构造。第二十章已经把 brane data、transversality、compactness、orientation、obstruction theory 和 gluing 定理标为外部输入；该标记仍然必须保留。

进入正文前的最小检查清单为：

1. 定义 $\mathbf{fc}$-multicategory，并与 colored operad、double category、virtual double category 比较。
2. 明确 pseudo-holomorphic polygons 的模空间需要哪些紧性、横截性和定向输入。
3. 检查 dg $\mathbf{fc}$-multicategory 的符号是否与定义 E.18--定义 E.23 和检查 W.1--检查 W.11 的同调分次约定一致。
4. 区分 curved $A_\infty$ algebra、module、bimodule 和 category 的不同曲率项。

## 3. 当前结论

截至 2026-06-30，本书可覆盖的当代方向包括：classical operad、colored operad、PROP/properad、Koszul duality、bar-cobar、$A_\infty/L_\infty/E_n$、Deligne conjecture 路线、同伦转移、模型范畴中的 operad、dendroidal sets、Lurie-style infinity-operads、operadic localization、factorization homology 与 Fukaya category 边界。上述 2026 前沿条目表明仍有四个需要专章才能严肃处理的方向：

1. infinity-operadic homology 与 algebras over infinity-operads 的 Koszul duality；
2. operadic categories 的 tensor product、wreath product 与 operadic nerve；
3. relative dendroidal Rezk nerve 与 operadic localization；
4. Fukaya categories 的高阶 operadic/多范畴结构。

附录 Y 已把第 1 类方向转化为 strict operad 的树指标线性化、Segal-linear 接口和 Koszul extension 特化检验。附录 Z 已把第 2--4 类方向转化为 operadic category 数据包、relative dendroidal object、Rezk nerve 接口和 Fukaya 条件性代数命题。这些附录减少了研究边界与正文之间的断裂，但仍未把近期预印本的新定理纳入核心证明链。

因此，本书当前已经达到 operad theory 数学收口态，但不是 camera-ready 出版态。它已经具备严格教材的主体骨架、依赖账本、边界审计和前沿接口附录；若未来目标是把上述前沿预印本的新定理纳入核心正文，需要另开专题并补齐正式模型、定理编号、证明依赖和文献定位。
