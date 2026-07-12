# 出版校对账本

本文件服务于最终出版态。它不新增数学内容；它记录出版校对中已经检查、已经修正和仍属于 production/copy-editing 的项目。Operad theory 数学收口判定见 [FINAL_OPERAD_THEORY_CLOSURE.md](FINAL_OPERAD_THEORY_CLOSURE.md)。

## 0. 校对口径

出版校对不再横向增加 operad theory 主题。校对目标只有五类：

1. 把证明链外部输入精确到 theorem/proposition/lemma 编号、版本和假设；
2. 清除会误导读者的局部指称、旧锚点和自指证明；
3. 把“证明边界”改写为完整内部证明或正式外部引用；
4. 把符号约定与选定链级模型逐项对齐；
5. 建立 bibliography、索引和最终排版交叉引用。

## 1. 第一轮出版校对动作

| 项目 | 处理结果 |
| --- | --- |
| 旧锚点检索 | `附录 X.9`、`本书附录 X.9`、`B1--B12` 等旧状态短语已无有效残留 |
| 命题 X.9 | 消除证明自指；改为引用外部输入定理 N.18 和本节开头 Hochschild 计算 |
| 附录 N 圆周计算 | 把 Ayala--Francis Theorem 3.19 标为 AF-2，并把符号核对入口改为定义 E.18--定义 E.23 与检查 W.1--检查 W.11 |
| 第十四章 rectification 边界 | 标明 BM-4--BM-5 已定位；现代 Pavlov--Scholbach symmetric flatness/rectification 后续已由 PSAR-1--PSAR-6 与 PSP-1--PSP-2 定位 |
| 附录 E brace 符号说明 | 把含糊的左侧输入描述和 E.7 节泛称改为插入位置左侧输入和定义 E.18--定义 E.23 |

## 2. 已有 P0 定位批次

| 批次 | 覆盖 | 当前用途 |
| --- | --- | --- |
| `P0_REFERENCE_LOCATORS_BATCH_1.md` | Berger--Moerdijk BM-1--BM-5；Cisinski--Moerdijk CM-1--CM-4 | 支撑第十四章 transferred structure、早期 rectification 边界、第十七章 operadic model structure |
| `P0_REFERENCE_LOCATORS_BATCH_2.md` | Lurie HTT-1 | 支撑第十九章 ordinary straightening/unstraightening；不支撑 operadic straightening |
| `P0_REFERENCE_LOCATORS_BATCH_3.md` | Ayala--Francis AF-0--AF-5 | 支撑第二十章和附录 N/V 的 topological-manifold factorization homology、boundary 版本与 commutative-coefficient 公式 |
| `P0_REFERENCE_LOCATORS_BATCH_4.md` | Ginzburg--Kapranov GK-1--GK-7；Loday--Vallette LV-1--LV-3 | 支撑 classical quadratic core、connected weight-graded twisting 四项等价、modern quadratic Koszul criterion 和 nonsymmetric rewriting criterion；不替代模型范畴 cofibrancy |
| `P0_REFERENCE_LOCATORS_BATCH_5.md` | Fresse FRE-1--FRE-6；Hinich HIN-1--HIN-2 | 支撑第九、十、十三、十四章和附录 I/Q/L/R 的 modern cobar/cofibrant replacement 与 dg-operad model context；不完全支撑 Loday--Vallette 书本编号 |
| `P0_REFERENCE_LOCATORS_BATCH_6.md` | Markl MHT-1--MHT-8 | 支撑第十三章和附录 J/S/W 的 operadic homotopy transfer existence；不完全支撑 HPL 显式公式、tree sign convention 或 minimal model uniqueness |
| `P0_REFERENCE_LOCATORS_BATCH_7.md` | Moerdijk--Weiss MW-1--MW-6 | 支撑第十六、十七章和附录 M/T 的 dendroidal nerve fully faithfulness、$\Delta\subset\Omega$、strict nerve unique fillers 和 homotopy coherent nerve inner Kan 入口 |
| `P0_REFERENCE_LOCATORS_BATCH_8.md` | White WHT-1--WHT-4；White--Yau WY-1--WY-3 | 支撑第十九章、附录 D/M/R 中模型范畴 Bousfield localization preserves operad/colored-operad algebras 的 preservation 版本；不支撑完整 infinity-categorical algebra localization comparison |
| `P0_REFERENCE_LOCATORS_BATCH_9.md` | Pavlov--Scholbach PSAR-1--PSAR-6；PSP-1--PSP-2；Lurie HA-ALG-1--HA-ALG-3、HA-MON-1--HA-MON-2 | 支撑第十四、十九章和附录 G/M/R/X 的 modern colored admissibility、rectification、strict-to-infinity algebra comparison 及 underlying symmetric monoidal infinity-category；不支撑任意无假设 rectification |
| `P0_REFERENCE_LOCATORS_BATCH_10.md` | Hinich DKR-1--DKR-7；Heuts--Hinich--Moerdijk HHM-1--HHM-5；Lurie HA-OP-1--HA-OP-3；Pratali PRA-1--PRA-5 | 支撑第十八、十九章和附录 M/Z 的 localization、open/no-constants dendroidal--Lurie zig-zag、category of operators 和 operadic straightening 边界 |
| `P1_REFERENCE_LOCATORS_FINAL_SWEEP.md` | Lurie DUNN-1；McClure--Smith MS-1--MS-3；Berger--Fresse BF-1--BF-4；May/Cohen/Kontsevich/Tamarkin/Getzler/geometry boundary locators | 支撑第十至十二、二十章和附录 L/N/V/Z 的 P1 locator 收口；把拓扑与几何结论保留为外部边界 |

## 3. 出版生产级剩余包

| 等级 | 包 | 剩余动作 |
| --- | --- | --- |
| CT | Koszul/bar-cobar convention package | LV-1--LV-2 已定位现代四项判别，LV-3 已定位 nonsymmetric rewriting criterion；GK 与 FRE/HIN 分别控制 classical 和 model-category 版本。剩余只做逐符号 crosswalk，不再缺 theorem number |
| CT | HPT / homotopy transfer sign package | Markl strongly homotopy transfer 已定位；basic perturbation lemma、Kadeishvili minimal model、tree signs 和 uniqueness 不再作为 operad-theory locator 空缺，只需符号模型对照 |
| HT | 模型范畴 operad 假设翻译 | Berger--Moerdijk、Hinich、Fresse、Pavlov--Scholbach、White/White--Yau、Lurie HA 和 Hinich DK locator 均已登记；剩余为底范畴假设表和 bibliography 版本核验 |
| HT | Dendroidal/model comparison 假设翻译 | Moerdijk--Weiss、Cisinski--Moerdijk 和 HHM locator 已登记；剩余为 erratum、open/no-constants 限制和 generalized Reedy/tree-decomposition 约定核查 |
| BD | Fukaya category construction | 关闭为几何边界；若未来定理化，需要另开具体几何模型、transversality、compactness、orientation 和 gluing theorem 来源包 |
| BD | Topological operad 与几何边界 | Deligne、Dunn、category operators、dendroidal-Lurie、May/Poisson/formality/framed BV 已有 locator 或 boundary locator；剩余为 bibliography normalization、系数环/degree convention 和符号模型核对 |

## 4. 局部指称判定

两轮交叉引用替换后，剩余“上述/上式/前面”主要分为两类：

1. 紧邻公式或定义的局部语义，例如“该公式”“本节给出的 collar gluing”；
2. 元文档中的审计对象名称，例如记录曾经替换过的“上述规则”。

这两类不阻断 operad theory 数学收口，但排版终校仍应逐章人工通读。

## 5. 2026-07-11 严格性修订

本轮不是纯 production 校对。逐式复核发现并修正了会改变正文结论的数学问题：

1. 默认允许 arity $0$ 时，代入乘积必须取遍有限集映射并保留空纤维；非空分块公式只在内层 nullary 项为初对象时成立。序章、第一至第六章、第十五章及附录 A/B/F/K/P 已同步；自由代数、自由树 operad、operadic congruence、线性化和拓扑到链构造均保留空槽，并加入反例 B.10.1/P.0。
2. 第十四章和附录 G 已把 operad 模型结构、固定 operad 代数的 transferred structure、strong admissibility 与 map-specific rectification 分开；BM-1、PSAR-2、PSAR-4/PSAR-5 的假设包逐项写入正文，正特征失败由 X.15--X.16 内部计算支撑。
3. 第八、九章及附录 I/Q 已分离 $\mathcal P^!$ 与 $\mathcal P^¡$，补出 conilpotence、直和/完成化和 bar/cobar 相反滤过方向，并以 I.22.1 说明 unary 情形的收敛失败。
4. 第十七、十九章及附录 M/T 已把 MW/CM/HHM/PSAR 输入按精确范围重新分类；HHM zig-zag 的 open/no-constants 限制不得覆盖本书默认 arity $0$。
5. 第二十章及附录 N/V 已固定 framed/tangential 语境、以 slice final object 证明 disk normalization、以 open-collar pieces 写球面 excision，并把 N.30 降为研究边界；AF-5 精确定位交换系数公式。

## 6. 当前判定

上述数学缺口已经在正文、定理账本、来源索引和 label 表中同步关闭。后续仍可做 bibliography、符号模型与排版终校，但不得再以“仅剩 production work”为理由跳过假设和 arity $0$ 检查。

Koszul 四项判别的书本 locator 已在本轮进一步精确到 Loday--Vallette Theorems 6.6.2（LV-1）和 7.4.6（LV-2）；Theorem 8.1.1 及其后 $\operatorname{As}$ 例子登记为 LV-3，支撑附录 Q 的 nonsymmetric rewriting/Koszul 步骤。这些结论不替代 FRE/HIN 的模型范畴 cofibrancy 输入。
