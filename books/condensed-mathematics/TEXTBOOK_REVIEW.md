# 凝聚数学讲义教材性审查

作者：Dr. Stochastic Parrot
审查日期：2026-06-30

## 总结

四卷已经形成一套中文凝聚数学输入定理型最终收口版教材草稿。更准确的判断需要区分两个标准：

1. **完全自足证明版教材。** 没有达到，也不宜作为当前仓库目标；那会要求重证 Gleason、Nöbeling、solid/analytic 核心结构定理、Cartan/Grauert、Hodge-Fredholm、GAGA、GRR 和高阶范畴论预备。
2. **主线输入定理型严格教材草稿。** 已经达到 condensed/solid/analytic/liquid 主线输入定理型最终收口状态；书内证明基础命题和接受输入定理后的形式推论，书外深层定理集中登记为输入定理。

| 分卷 | 当前定位 | 教材性判断 | 出版级增强方向 |
| --- | --- | --- | --- |
| 卷一 | 凝聚基础 | 基础教材草稿，基本完本 | Gleason lifting 与 Nöbeling 深层输入可另写预备教材 |
| 卷二 | solid/analytic/liquid 主线 | 主线输入定理型最终收口版 | solidification、analytic localization、liquid realization 已拆成证明模块；后续可继续补逐条文献定位和深层证明 |
| 卷三 | 复几何与相干对偶 | 输入定理型严格教材草稿，基本完本 | 经典复几何深定理可另写完全证明卷或精细引用手册 |
| 卷四 | 计算与形式化补充 | 工具卷基本完本 | Lean 代码、pro-etale 专题和谱值接口可继续扩展 |

因此，对外称呼应使用“四卷主线输入定理型严格教材草稿”或“condensed/solid/analytic/liquid 主线输入闭合版”，而不是“完整自足证明教材”。本轮出版级校对已经把凝聚主线输入提升到讲次、章节和源码 label 级 locator，并把经典输入从 L0 清到 L1/L2/L3 分层状态；最终收口说明见 [FINAL_PUBLICATION_STATUS.md](FINAL_PUBLICATION_STATUS.md)。引用定位和校对优先级见 [REFERENCE_LOCATOR_LEDGER.md](REFERENCE_LOCATOR_LEDGER.md) 与 [PUBLICATION_PROOFREADING_AUDIT.md](PUBLICATION_PROOFREADING_AUDIT.md)。

本轮严格化已经完成多项结构性补强：

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
32. 第三卷新增附录 V-Y，补 Stein/Cartan 工具、正则局部环与相干层有限分解、有限性传播和 projective GAGA 证明结构。
33. 第三卷新增附录 Z-AA，补椭圆复形 Hodge 定理接口和向量丛 Serre 对偶的 Hodge 证明。
34. 第三卷新增附录 AB-AE，补 Cartan A/B 证明模块、Grauert 直接像、dualizing complex 版 Serre 对偶和一般 GRR 形式。
35. 第二卷新增附录 M-N，补 solid localization 的生成核和 analytic rational descent 的证明义务。
36. 第三卷新增附录 AF-AG，补 Weierstrass-Oka coherence 的局部代数和 Runge-Cousin 推出 Cartan B 的机制。
37. 第三卷新增附录 AH-AK，补 Hörmander L2 到 Stein 消没、projective GAGA graded module 细节、Grothendieck duality 构造义务和 GRR deformation-to-normal-cone 证明模块。
38. 第三卷新增附录 AL-AP，补 Weierstrass 除法估计、Hörmander 基本估计到闭值域、Grauert Banach 复形、形式函数/GAGA 代数化和 GRR 局部化推前相容。
39. 第二卷新增附录 O-P，补可展示稳定局部化正合性和 Fréchet/liquid 闭值域类型检查。
40. 第一卷新增附录 O-P，补 Gleason 投射性和 Nöbeling 定理的证明模块。
41. 第四卷新增附录 F-G，补凝聚基础形式化证明义务和凝聚谱/pyknotic 接口。
42. 总目录新增 [INPUT_THEOREM_REGISTER.md](INPUT_THEOREM_REGISTER.md) 和 [GLOSSARY.md](GLOSSARY.md)，把外部输入与术语集中登记。
43. 总目录新增 [COMPLETION_CRITERIA.md](COMPLETION_CRITERIA.md)，并在第三卷新增附录 AQ，把复几何主定理包合成为一个输入定理型闭包。

44. 第二卷新增附录 V-Z，把 solidification 反射存在性、solid 核张量理想性、analytic localization、rational descent 和 liquid realization 拆成证明模块。
45. 第二卷新增附录 AA、第三卷新增附录 AR，把 Scholze 与 Clausen-Scholze 的核心定理作为凝聚数学主线图谱集中列出。
46. 总目录新增 [REFERENCE_LOCATOR_LEDGER.md](REFERENCE_LOCATOR_LEDGER.md) 和 [PUBLICATION_PROOFREADING_AUDIT.md](PUBLICATION_PROOFREADING_AUDIT.md)，进入出版级引用定位和软表述校对阶段。
47. 出版级校对第一轮补固定 universe 下的站点比较表述，把第一卷 ProFin/CHaus 比较改为稳定基定理，并为 S26、CS26、A23、ABKMT24 主线输入回填讲次、章节和 TeX label 级 locator。
48. 出版级校对第二轮补经典输入 locator，将 Gleason、Serre duality、GAGA 升到 L3，将 Dolbeault、Cartan A/B、Grauert、Hodge-Fredholm、GRR 升到 L2，并在第三卷资料源中增补经典复几何来源。

这些补强把项目推进到“主线输入定理型最终收口版”。第二卷附录 Q-AA 将 solid、analytic、liquid 和三者统一接口收束为主定理包，补出版级闭包审查，把核心输入拆成可复核的证明模块，并把 Scholze/Clausen-Scholze 的核心定理列入主线图谱；第三卷附录 AR 对复几何核心定理做同样处理。本轮校对又把主线引用定位、经典输入定位和站点比较入口向出版口径推进了一步。它仍不等于完整证明 solid/analytic/liquid 核心结构定理和所有应用深层定理的自足教材。

## 风格与严格性判断

从“致密的数学严格性”标准看，需要分层判断。若按完全自足或逐题教师手册标准，当前版本仍可继续增强；若按主线输入定理型严格教材标准，当前版本已经最终收口。它已经比普通综述更严格：定义、定理、输入定理、证明路线和风险边界基本分开；非阻塞维护项主要是少量 theorem/page locator、逐题答案扩写和排版级统一。

主要风格问题如下：

1. **软表述偏多。** 文中仍有“适当”“通常”“良好”“可视为”“类似”“证明说明”等表述。它们在导读中可以接受，但在教材定理中应替换成明确假设、定义或引用。
2. **证明说明多于完整证明。** 卷一基础部分有较多完整证明；卷二、卷三和部分卷四仍依赖证明路线或输入定理。这不是错误，但不够致密。
3. **输入定理颗粒度仍需继续细化。** 新增输入定理登记表已经把主要外部输入集中列出，但“Clausen-Scholze 复几何建模”“Scholze analytic ring 结构定理”等仍可继续拆成更小编号定理。
4. **范畴层级需要更干净。** 普通阿贝尔群、凝聚阿贝尔群、solid 对象、analytic 模、liquid 向量空间、派生范畴之间的切换已经有类型检查，但正文仍应更频繁标注对象所在范畴。
5. **例子和反例不足。** 严格教材需要通过反例说明假设不可删；当前反例主要集中在卷四，卷一至卷三还不够。
6. **练习答案仍可扩成教师手册。** 新增统一答案和分卷答案覆盖核心题目，但若按出版标准，许多证明题仍应扩为逐行证明。

这意味着：当前版本可以作为“主线输入定理型最终收口版”维护；若要达到完全自足或逐题教师手册标准，需要把剩余“证明说明”继续分流为两类：一类改写成完整证明，另一类明确降格为更细编号的“外部输入定理”。其中 solid、analytic 和 liquid 的核心结构定理应优先处理，因为它们属于本书主线。

## 可能的数学漏洞类型

当前审查没有发现一个可以简单判定为“定理结论明显错误”的核心漏洞；更主要的问题是严格性缺口。这些缺口如果不标注清楚，会在教材使用中变成数学漏洞：

1. **站点比较定理的假设容易不足。** 需要明确基子站点对拉回、共同细化和诱导拓扑的稳定性；本轮已把卷一相关章节统一到固定 universe 与稳定基版本。
2. **solid 张量积公式不能从普通张量积推出。** 凡涉及无限乘积或 profinite 测度对象，都必须声明使用 Scholze 的 solid 输入定理。
3. **analytic ring 的 localization 不能类比普通完备化。** 解析化、Bousfield localization 和 Huber pair rational localization 需要分别陈述。
4. **liquid 结构不能等同于 Banach 或 Fréchet 拓扑。** 需要把 Hom 判别和测度测试对象写清楚。
5. **卷三经典复几何定理不能压缩为 condensed 翻译。** Dolbeault、Cartan A/B、Serre duality、GAGA、Riemann-Roch 的经典证明或精确引用必须补齐；未书内证明的部分必须登记为外部输入定理。

## 表述清洁度要求

出版级增强编辑应采用以下规则：

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

1. Gleason lifting、Nöbeling 一般 profinite 情形和 Scholze 的 solid 核心计算仍是外部输入。
2. Stone 对偶已由附录 N 补齐；Gleason 和 Nöbeling 已由附录 O-P 拆成证明模块；若作为完全自足教材，还需要重证 Gleason lifting 和 Nöbeling-Asgeirsson 超限过滤。
3. 练习数量充足，已有统一答案和分卷答案入口；若按出版标准，还需把所有证明题扩为逐行教师手册。
4. Ext 与 Tor 的基础工作例题已由附录 M 补充；更深的 solid/analytic Ext 计算仍依赖 Scholze 输入。

判断：卷一可以作为“基础教材草稿”，并且基础证明闭合度最高；但它仍不是完全自足的拓扑/范畴/同调代数教材。

## 卷二：Solid、Analytic 与 Liquid 结构

卷二的主题本身技术重，当前正文能给出清晰定义、输入定理、类型检查和例子，但证明密度低于卷一。

出版级增强点：

1. solid 派生范畴的构造仍依赖 Bousfield localization 的一般理论。
2. analytic ring 条件多以 Hom 判别和 cone 条件呈现，但缺少完整的模型范畴或稳定范畴证明。
3. liquid 向量空间只给出判别口径和风险点，不足以作为 liquid theory 的自足教材。
4. Huber pair 和 $f_!$ 的章节更像通往复几何的入口，不是完整几何教材。
5. 练习多为类型检查题，缺少完整计算题和反例题。

判断：卷二是 solid/analytic/liquid 主线输入定理型最终收口版，适合接续卷一，并承担全书核心理论职责。本轮已补 Bousfield localization 的形式骨架、输入定理登记表、solid localization 生成核、analytic rational descent 证明义务、presentable stable localization 的正合形式推论、局部化技术引理、幺半与闭幺半 Bousfield 局部化、伴随函子、投影公式形式骨架、Cech descent/totalization 形式层、紧生成生成元检验、analytic ring 检查表、liquid/Banach/Fréchet 边界、Fréchet/liquid 闭值域检查，Q-T 的 solid、analytic、liquid 主定理包和统一闭包定理，U 的出版级闭包审查，V-Z 的核心证明模块，以及 AA 的 Scholze/Clausen-Scholze 核心定理图谱；若要提升为完全自足主线教材，还需证明或逐条引用 presentable stable localization 存在性、Scholze solidification 识别、profinite 测度张量公式、analytic ring 条件推出 localization、rational acyclicity、liquid realization 与经典分析对象相容的全部细节。

## 卷三：复几何与相干对偶

你的怀疑曾经是对的：卷三早期证明密度不足。现在它已经补入经典输入定理、依赖链、局部计算、形式推论和主定理闭包章，因此应定位为输入定理型严格教材草稿；但它仍不是完全自足证明版复几何教材。

出版级增强点：

1. Dolbeault lemma、Cartan A/B、相干层有限分解已经有证明模块；完整经典证明仍需复几何预备卷或精确文献定位。
2. coherent cohomology finite-dimensionality 已有 Fredholm-Hodge、Grauert 和谱序列形式层；完整 elliptic/Fredholm 估计仍作为输入。
3. Serre duality 已有射影空间、向量丛和 dualizing complex 三层形式；完美性和 dualizing complex 存在性仍依赖深层输入。
4. GAGA 已有 projective graded module 与形式函数路线；完整代数化证明仍作为外部输入。
5. Riemann-Roch 已有射影空间线丛证明、Chern/Todd 形式代数和一般 GRR 形式后果；GRR 基本因子定理仍作为输入。
6. 六函子形式仍是后续专题接口，不是本卷已建立的完整理论。

判断：卷三现在应标为“输入定理型严格教材草稿，基本完本草稿”。本轮已补经典输入定理的精确形式、依赖链、Stein/Cartan 工具和 Cartan A/B 证明模块、Weierstrass-Oka 局部相干性、Weierstrass 除法估计、Runge-Cousin-Cartan 与 Hörmander \(L^2\) 机制、Hörmander 基本估计到闭值域步骤、正则局部环与相干层有限分解、Grauert 直接像与有限性、Grauert Banach 复形证明模块、椭圆复形 Hodge 接口、向量丛和 dualizing complex 版 Serre 对偶、$\mathbb P^1$ 和 $\mathbb P^n$ 线丛上同调计算、射影空间线丛 Serre 对偶和 HRR 公式证明、Čech 谱序列证明、Dolbeault 局部正合骨架、GAGA/RR 形式推论、形式函数/GAGA 代数化路线、Fredholm-Hodge 有限性形式层、有限分解和谱序列有限性传播边界、fine sheaf 与 Dolbeault resolution 形式证明、Chern/Todd/RR 形式代数、GAGA properness 反例、projective GAGA graded module 结构、Grothendieck duality 构造义务、一般 GRR 形式、deformation-to-normal-cone 证明模块、GRR 局部化推前相容、主定理闭包章和 Clausen-Scholze 复几何核心定理图谱；若要提升为完全自足证明版复几何教材，还需要至少新增三类内容：

1. 经典复几何预备章：Dolbeault lemma 的局部骨架已补；Stein/Cartan 的形式推论和证明模块已补；Hodge theorem 和 Grauert 的形式结论已补；Weierstrass/Oka/Cousin、parametrix/elliptic estimates 和 Grauert 定理本身的证明或精确引用仍需补齐。
2. 凝聚/analytic 翻译章：逐步证明经典对象进入 analytic/liquid 范畴后保持同调结构。
3. 定理级证明章：有限性、Serre duality、GAGA、Riemann-Roch 各自给出完整证明或严格引用边界。

## 卷四：形式化、计算与例子

卷四已补强为工具卷，包含 sheaf 等化子、可表 sheaf、基子站点比较、Ext/Tor、solid 反例、analytic/liquid 类型检查和 pro-etale 对照。作为“前三卷的计算与形式化补充”已经比较合理。

出版级增强点：

1. 形式化内容仍是 Lean 风格路线图，不含实际形式化代码。
2. pro-etale 只提供比较框架，不是 pro-etale topology 教材。
3. Ext/Tor 的证明停留在一般阿贝尔范畴层面，凝聚范畴的存在性仍作为输入。
4. solid/analytic/liquid 例子仍需更多非平凡计算。

判断：卷四不是主线理论卷，但作为工具卷已经达到基本完本草稿状态。本轮新增形式化证明义务和凝聚谱接口后，它更适合作为“如何检查前三卷证明和如何走向形式化/同伦化”的工具卷。

## 练习答案状态

新增 [SOLUTIONS.md](SOLUTIONS.md) 作为统一答案/提示文档，并新增分卷答案入口：

1. [volume-1/SOLUTIONS.md](volume-1/SOLUTIONS.md)
2. [volume-2/SOLUTIONS.md](volume-2/SOLUTIONS.md)
3. [volume-3/SOLUTIONS.md](volume-3/SOLUTIONS.md)
4. [volume-4/SOLUTIONS.md](volume-4/SOLUTIONS.md)

当前达到“答案要点 + 核心难题详解”的状态。若按出版教师手册标准，还需把每道证明题扩为逐行解答。

## 出版级增强建议

若继续提升教材质量，优先顺序应为：

1. 补卷二：增加 solidification、analytic ring 和 liquid 的详细证明链或标准引用。
2. 补卷三：把 Dolbeault、Serre duality、GAGA、Riemann-Roch 的证明边界逐节细化。
3. 扩展当代方向：把 pyknotic、condensed spectra、Galois/exodromy 和 pro-etale 接口写成独立专题。
4. 扩展答案：分卷答案已建立；后续应把全部证明题扩为逐行教师手册。
5. 精化索引：术语索引、定理索引、输入定理索引已经建立，后续可继续补页内锚点和正文回链。
6. 细化依赖图：依赖图已经建立，后续可把每个主定理的前置定义、输入定理和证明使用位置补到更细粒度。
