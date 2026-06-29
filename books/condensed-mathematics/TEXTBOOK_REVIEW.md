# 凝聚数学讲义教材性审查

作者：Dr. Stochastic Parrot
审查日期：2026-06-29

## 总结

四卷已经形成一套可读的中文凝聚数学教材草稿，但还不能全部称为“完整证明版教材”。更准确的判断是：

| 分卷 | 当前定位 | 教材性判断 | 主要缺口 |
| --- | --- | --- | --- |
| 卷一 | 凝聚基础 | 最接近教材级 | 部分深层定理依赖外部输入，练习答案需系统化 |
| 卷二 | solid/analytic/liquid 结构 | 严格讲义，未达完整教材 | Bousfield localization、analytic ring、liquid 的证明细节不足 |
| 卷三 | 复几何应用 | 应用导读与证明路线，不是完整证明版教材 | Dolbeault、有限性、Serre duality、GAGA、Riemann-Roch 多为输入定理 |
| 卷四 | 计算与形式化补充 | 作为工具卷较合格 | 形式化仍是路线图，pro-etale 只作比较，不是专题教材 |

因此，若对外称呼，应使用“严格教材草稿”或“四卷讲义草稿”，而不是“完整证明教材”。如果目标是严格教材，下一轮优先级应是：卷三证明细化、卷二核心结构证明细化、统一练习答案。

本轮严格化已经完成两项结构性补强：

1. 第二卷新增附录 C-D，把 Bousfield localization、solidification、analyticization 和输入定理登记表写成较精确的形式定理。
2. 第三卷新增附录 F-G，把 Dolbeault、Cartan A/B、有限性、Serre duality、GAGA、HRR 和 Clausen-Scholze 建模拆成精确输入定理与依赖链。
3. 第三卷新增附录 H，完整计算 $\mathbb P^1$ 上 $\mathcal O(d)$ 的 Čech 上同调。
4. 第四卷新增附录 E，补入 pyknotic 对象、凝聚同伦类型和凝聚谱的定义入口。
5. 第一卷新增附录 H，补正合 sheafification、Grothendieck 阿贝尔范畴和 K-flat 派生张量的证明链。
6. 第二卷新增附录 E，补局部化核、张量理想、幺半下降和相对张量积的完整范畴论证明。
7. 第三卷新增附录 I，补 Čech-to-derived 谱序列、acyclic 覆盖和超上同调计算。
8. 总目录新增 [THEOREM_INDEX.md](THEOREM_INDEX.md) 与 [DEPENDENCY_GRAPH.md](DEPENDENCY_GRAPH.md)，用于追踪主定理、输入定理和跨卷依赖。
9. 第二卷新增附录 F，补 $f_!$ 右伴随、投影公式检验和 $f^!$ 内部 Hom 公式的形式证明。
10. 第三卷新增附录 J-K，补 Serre 对偶、GAGA 和 Riemann-Roch 在接受输入定理后的链级、导出范畴和 $K$-理论形式证明。
11. 第一卷新增附录 I，补 horseshoe lemma、投射分解比较、Ext/Tor 长正合列和维数平移的书内证明。
12. 第三卷新增附录 L，补 Fredholm-Hodge 输入推出 Dolbeault cohomology 有限维的形式证明。
13. 第一卷新增附录 J，补 regular open 代数与 Gleason cover 连续满射构造；Gleason 投射性仍作为外部输入。
14. 第一卷新增附录 K，补 ED 覆盖检测 sheaf 单射、满射、同构和阿贝尔 sheaf 正合性的完整形式证明。
15. 第二卷新增附录 G，补 Cech nerve、totalization、稳定范畴值 descent 和 rational Cech 下降的形式推论。
16. 第三卷新增附录 M，补有限分解、谱序列和有限过滤传播有限性的证明，并明确 Stein-Cech 计算本身不推出有限维性。
17. 第三卷新增附录 N，补 fine sheaf、Cech 同伦、acyclic resolution 和 Dolbeault cohomology 计算 sheaf cohomology 的形式证明。
18. 第二卷新增附录 H，补紧生成、localizing subcategory、自然变换和全忠实的生成元检验。
19. 第三卷新增附录 O，补有限局部自由 resolution 假设下，从向量丛 Serre 对偶到相干层 Ext-Serre 形式的条件性证明。
20. 第二卷新增附录 I，补 analytic ring 公理检查表、Dirac cone 判别、有限测试对象空洞性和失败模式。
21. 第三卷新增附录 P，补 Chern character、Todd class、splitting principle 和 Riemann-Roch 右侧的形式代数。
22. 第一卷新增附录 L，补 sheaf 满射、separated presheaf、基子站点、普通张量积和拓扑阿贝尔群的边界反例。
23. 第一卷新增附录 M，补 Ext/Tor 工作例题，包括有限离散对象、两项投射分解、$\mathbb Z_{\operatorname{cond}}/n$ 和 Tor 的计算。
24. 第二卷新增附录 J，补 liquid 与 Banach/Fréchet 的边界、Banach 非闭像风险和 Dolbeault Fréchet 类型检查。
25. 第三卷新增附录 Q，补 GAGA properness 的 $\mathbb A^1$ 反例和导出比较形式证明。
26. 第一卷新增附录 N，补超滤子、Stone 空间紧性、开闭代数同构和 profinite 逆极限表示。
27. 第二卷新增附录 K，补幺半 Bousfield 局部化、张量理想判别、交换代数和相对张量积下降公式。
28. 第三卷新增附录 R，补 Dolbeault 局部正合的 Cauchy-Green 算子和 polydisc 同伦骨架。
29. 第三卷新增附录 S，补 $\mathbb P^n$ 上 $\mathcal O(d)$ 的 Čech 单项式计算、基础 Bott 公式和 Euler characteristic。
30. 第二卷新增附录 L，补闭幺半局部化、内部 Hom、dualizable 对象和右伴随 Hom 比较的类型边界。
31. 第三卷新增附录 T-U，补 $\mathbb P^n$ 上线丛的 Serre 对偶和 HRR 公式证明。

这些补强提高了严谨性，但仍不等于完整证明所有深层定理。

## 风格与严格性判断

从“致密的数学严格性”标准看，当前版本还不达标。它已经比普通综述更严格：定义、定理、输入定理、证明路线和风险边界基本分开；但还没有达到成熟数学教材中“每个定理的量词、假设、证明依赖和例外都完全闭合”的程度。

主要风格问题如下：

1. **软表述偏多。** 文中仍有“适当”“通常”“良好”“可视为”“类似”“证明说明”等表述。它们在导读中可以接受，但在教材定理中应替换成明确假设、定义或引用。
2. **证明说明多于完整证明。** 卷一基础部分有较多完整证明；卷二、卷三和部分卷四仍依赖“证明路线”或“标准事实”。这不是错误，但不够致密。
3. **输入定理颗粒度过粗。** 例如“Clausen-Scholze 复几何建模”“Scholze analytic ring 结构定理”范围很大。严格教材应拆成多个编号定理，并说明每处使用哪一个结论。
4. **范畴层级需要更干净。** 普通阿贝尔群、凝聚阿贝尔群、solid 对象、analytic 模、liquid 向量空间、派生范畴之间的切换已经有类型检查，但正文仍应更频繁标注对象所在范畴。
5. **例子和反例不足。** 严格教材需要通过反例说明假设不可删；当前反例主要集中在卷四，卷一至卷三还不够。
6. **练习答案还偏提示化。** 新增答案文档覆盖题目，但许多证明题仍是答案要点，不是逐行证明。

这意味着：当前版本可以作为“严格讲义草稿”继续维护；若要达到“致密数学教材”，需要把所有“证明说明”分流为两类：一类改写成完整证明，另一类明确降格为“外部输入定理”。

## 可能的数学漏洞类型

当前审查没有发现一个可以简单判定为“定理结论明显错误”的核心漏洞；更主要的问题是严格性缺口。这些缺口如果不标注清楚，会在教材使用中变成数学漏洞：

1. **站点比较定理的假设容易不足。** 需要明确基子站点对拉回、共同细化和诱导拓扑的稳定性；卷四已采用“稳定基”版本，但卷一相关章节仍可继续统一。
2. **solid 张量积公式不能从普通张量积推出。** 凡涉及无限乘积或 profinite 测度对象，都必须声明使用 Scholze 的 solid 输入定理。
3. **analytic ring 的 localization 不能类比普通完备化。** 解析化、Bousfield localization 和 Huber pair rational localization 需要分别陈述。
4. **liquid 结构不能等同于 Banach 或 Fréchet 拓扑。** 需要把 Hom 判别和测度测试对象写清楚。
5. **卷三经典复几何定理不能压缩为 condensed 翻译。** Dolbeault、Cartan A/B、Serre duality、GAGA、Riemann-Roch 的经典证明或精确引用必须补齐，否则只能作为应用导读。

## 表述清洁度要求

下一轮编辑应采用以下规则：

1. 定理中禁用“适当”“良好”“通常”；改为具体假设。
2. “可视为”“直观上”只能放在定理之后的解释段，不得承担证明功能。
3. “标准事实”必须带来源或在附录证明。
4. 每个输入定理必须说明来源、精确使用形式和本书不证明的部分。
5. 每个跨范畴公式必须标明所在范畴和张量/极限的意义。
6. 每章至少给出一个反例或边界例子，说明关键假设为何必要。

## 教材标准

本审查按以下标准判断：

1. 定义是否可独立理解。
2. 定理是否明确区分“本书证明”和“输入定理”。
3. 证明是否足以让读者逐步复核。
4. 是否有典型例子、反例和类型检查。
5. 练习是否有答案或提示。
6. 术语、符号和依赖是否跨卷一致。

## 卷一：凝聚数学基础

卷一内容最扎实。它已经覆盖站点、sheaf、紧 Hausdorff/profinite 空间、凝聚集合、凝聚阿贝尔群、投射对象、Ext/Tor、solid 和 analytic 入口。附录也补了 universe、站点比较、sheaf 阿贝尔群、Stone-Gleason、Stone 对偶完整证明链、Gleason cover 构造、Nöbeling、基本 Ext/Tor 计算、正合 sheafification、派生工具、horseshoe/导出函子形式、ED 覆盖检测正合性和基础反例。

仍不足之处：

1. Gleason 定理、Nöbeling 定理和 Scholze 的 solid 核心计算仍是外部输入。
2. Stone 对偶已由附录 N 补齐；若作为完全自足教材，还需要为 Gleason 定理和 Nöbeling 定理提供更完整证明或明确前置章节。
3. 练习数量充足，但统一答案文档仍偏答案要点，不是逐行教师手册。
4. Ext 与 Tor 的基础工作例题已由附录 M 补充；更深的 solid/analytic Ext 计算仍依赖 Scholze 输入。

判断：卷一可以作为“基础教材草稿”，但不是完全自足的拓扑/范畴/同调代数教材。

## 卷二：Solid、Analytic 与 Liquid 结构

卷二的主题本身技术重，当前正文能给出清晰定义、输入定理、类型检查和例子，但证明密度低于卷一。

主要缺口：

1. solid 派生范畴的构造仍依赖 Bousfield localization 的一般理论。
2. analytic ring 条件多以 Hom 判别和 cone 条件呈现，但缺少完整的模型范畴或稳定范畴证明。
3. liquid 向量空间只给出判别口径和风险点，不足以作为 liquid theory 的自足教材。
4. Huber pair 和 $f_!$ 的章节更像通往复几何的入口，不是完整几何教材。
5. 练习多为类型检查题，缺少完整计算题和反例题。

判断：卷二是严格结构草稿，适合接续卷一。本轮已补 Bousfield localization 的形式骨架、输入定理登记表、局部化技术引理、幺半与闭幺半 Bousfield 局部化、伴随函子、投影公式形式骨架、Cech descent/totalization 形式层、紧生成生成元检验、analytic ring 检查表和 liquid/Banach/Fréchet 边界；若要完全教材级，还需证明或完整引用 presentable stable localization、solid 核张量理想性、analytic ring 条件推出 localization 的全部细节。

## 卷三：复几何与相干对偶

你的怀疑是对的：卷三的证明确实不够详细。当前卷三的价值在于把复几何定理放进 condensed/analytic 语言，给出证明路线、术语对照和局部计算模型；但它不是完整证明版复几何教材。

主要缺口：

1. Dolbeault lemma、Cartan A/B、相干层有限分解等经典复几何输入没有重证。
2. coherent cohomology finite-dimensionality 只给路线，没有完整 elliptic/Fredholm 证明。
3. Serre duality 的积分配对和符号相容有局部计算，但完美性证明仍依赖有限性和深层输入。
4. GAGA 只写了 classical/condensed 比较路线，没有完整代数化证明。
5. Riemann-Roch 只给公式、$\mathbb P^1$ 例子和 trace/Chern character 路线，没有完整 GRR 证明。
6. 六函子形式是展望，不是已建立理论。

判断：卷三应标为“复几何应用导读与证明路线”。本轮已补经典输入定理的精确形式、依赖链、$\mathbb P^1$ 和 $\mathbb P^n$ 线丛上同调计算、射影空间线丛 Serre 对偶和 HRR 公式证明、Čech 谱序列证明、Dolbeault 局部正合骨架、Serre 对偶形式证明层、GAGA/RR 形式推论、Fredholm-Hodge 有限性形式层、有限分解和谱序列有限性传播边界、fine sheaf 与 Dolbeault resolution 形式证明、有限 resolution 下的 Ext-Serre 条件性证明、Chern/Todd/RR 形式代数、GAGA properness 反例和导出比较细节；要达到完整证明教材级，还需要至少新增三类内容：

1. 经典复几何预备章：Dolbeault lemma 的局部骨架已补；Stein、Cartan A/B、elliptic regularity 的证明或可引用版本仍需补齐。
2. 凝聚/analytic 翻译章：逐步证明经典对象进入 analytic/liquid 范畴后保持同调结构。
3. 定理级证明章：有限性、Serre duality、GAGA、Riemann-Roch 各自给出完整证明或严格引用边界。

## 卷四：形式化、计算与例子

卷四已补强为工具卷，包含 sheaf 等化子、可表 sheaf、基子站点比较、Ext/Tor、solid 反例、analytic/liquid 类型检查和 pro-etale 对照。作为“前三卷的计算与形式化补充”已经比较合理。

仍不足之处：

1. 形式化内容仍是 Lean 风格路线图，不含实际形式化代码。
2. pro-etale 只提供比较框架，不是 pro-etale topology 教材。
3. Ext/Tor 的证明停留在一般阿贝尔范畴层面，凝聚范畴的存在性仍作为输入。
4. solid/analytic/liquid 例子仍需更多非平凡计算。

判断：卷四不是主线理论卷，但作为工具卷基本合格。

## 练习答案状态

新增 [SOLUTIONS.md](SOLUTIONS.md) 作为统一答案/提示文档。它覆盖四卷全部显式练习的答案要点，但不是完整教师手册。后续若继续打磨，应把答案分拆为：

1. `volume-1/SOLUTIONS.md`
2. `volume-2/SOLUTIONS.md`
3. `volume-3/SOLUTIONS.md`
4. `volume-4/SOLUTIONS.md`

其中卷三答案最需要扩成完整版本，因为卷三练习常要求读者连接经典复几何与 condensed/analytic 表述。

## 下一轮建议

若继续提升教材质量，优先顺序应为：

1. 补卷三：把 Dolbeault、Serre duality、GAGA、Riemann-Roch 的证明边界逐节细化。
2. 补卷二：增加 Bousfield localization、analytic ring 和 liquid 的详细证明或标准引用。
3. 扩展当代方向：把 pyknotic、condensed spectra、Galois/exodromy 和 pro-etale 接口写成独立专题。
4. 拆分答案：把总答案文档分到各卷，并给难题完整证明。
5. 增加索引：术语索引、定理索引、输入定理索引。
6. 增加依赖图：每个主定理列出前置定义、输入定理和证明使用位置。
