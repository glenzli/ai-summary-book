# 第二轮严格化路线图

核查日期：2026-06-29。

本文件规定《Operad Theory》从第一轮主体草稿进入可审校教材形态所需的第二轮工作。它不是目录提纲，而是质量控制文件：每一项都必须能反映到正文、附录、资料源索引或审查记录中。

## 0. 当前状态判断

截至本文件建立时，本书已经具备序章、二十一章主体章节和五个附录。主体第一轮草稿覆盖了普通 operad、colored operad、线性 operad、Koszul 对偶、bar-cobar、同伦代数、模型范畴中的 operad、simplicial/topological operads、dendroidal sets、Lurie-style infinity-operads、模型比较、factorization algebra、Fukaya category 入口和 2025-2026 研究边界。

这仍不是最终教材。原因如下：

1. 许多大型定理仍以“外部输入定理”形式出现，尚未精确到原文定理编号、页码、版本或证明依赖链。
2. 符号系统已经固定基础版本，但 $A_\infty$、$L_\infty$、brace、operadic suspension 与 Hochschild signs 的全套展开尚未逐项交叉校验。
3. 模型范畴和 infinity-operad 部分覆盖面已经足够宽，但 transferred model structure、rectification、localization 和 straightening 的假设仍需逐语境核对。
4. 2025-2026 年预印本只应作为研究边界；任何进入正文定理链的使用都需要版本核查和独立证明路径。

## 1. 第二轮总目标

第二轮的目标不是继续横向堆叠主题，而是把已有章节压缩成致密、可检查的数学教材形态：

- 定义之间的依赖必须形成有向无环链。
- 每个非平凡命题必须有本书内证明，或在标题处标明“外部输入定理/命题”。
- 每个外部输入必须能在附录 D 中定位到具体来源。
- 每个符号必须能在 `NOTATION.md` 或相应章节首次定义处找到。
- 每个同伦或 infinity 断言必须说明所处模型：模型范畴、simplicial category、dendroidal set、quasi-category 或 Lurie-style infinity-operad。
- 每个近期研究条目必须保留版本、日期、模型约定和不进入核心证明链的边界。

## 2. 通过标准

一个章节在第二轮中称为“严格化通过”，若满足以下条件：

1. 章节开头列出精确前置定义。
2. 所有局部记号在首次使用处定义。
3. 每个定义后至少有一个结构性检查、例子或反例说明其约束。
4. 每个命题、引理、定理具有证明；若没有证明，则标题含“外部输入”并在附录 D 中有对应行。
5. 章节末尾列出本章没有证明的依赖，且这些依赖不被伪装成已证事实。
6. 若出现链复形、悬挂或 Hochschild 结构，符号与附录 E 一致。
7. 若出现模型结构、弱等价或 Quillen 等价，假设与附录 C 一致，并注明是否需要 cofibrant generation、monoid axiom、left properness、simplicial enrichment 或 combinatorial 条件。

## 3. 优先级 A：基础定义链

第一优先级是第一至第五章和附录 A-B。这里的错误会污染全书。

需要核对：

1. 有限集群胚 $\mathbf B_{\mathcal U}$、骨架 $[n]$ 与 $\Sigma_n$ 的互译。
2. 右作用/左作用约定是否在 arity 公式中一致。
3. 对称序列代入乘积的 coend 公式与 arity coinvariants 公式是否完全等价。
4. colored substitution product 是否在颜色函数、分块和 bijection 的作用上保持自然性。
5. 自由 operad 的树商是否与 operadic congruence 兼容。

通过产物：

- 附录 B 中每个自然同构补全函子性证明。
- 第二章自由代数公式改写为有限集 coend 版本，并说明 arity 骨架版本如何由附录 A 推出。
- 第四章自由 operad 的树公式增加“自同构群作用”和“等变商”的显式检查。

## 4. 优先级 B：线性与符号系统

第二优先级是第六、八、九、十、十一、十二、十三章和附录 E。

需要核对：

1. 链复形采用同调分次还是上同调分次。
2. differential 的次数、张量 differential 的 Koszul sign、Hom differential 的约定是否一致。
3. suspension/desuspension 与 operadic suspension 的 arity sign 是否匹配。
4. $A_\infty$ 恒等式是否由 $\Omega(\operatorname{coAss}^{\ash})$ 的 square-zero differential 推出，而不是单独选择一套不兼容 signs。
5. Gerstenhaber bracket、brace operations、Hochschild differential 与 cup product 的 convention 是否共享同一 reduced degree。

通过产物：

- 附录 E 增加完整 brace sign 表。
- 第十章只保留与附录 E 一致的一套 $A_\infty/L_\infty$ 展开式，其余作为 convention remark。
- 第十二章 graded Hochschild 部分从“未分次主文本”升级为“分次主文本，未分次作为特例”。

## 5. 优先级 C：模型范畴和 infinity-operad 假设

第三优先级是第十四至十九章和附录 C-D。

需要核对：

1. 每个 transferred model structure 是否指定生成 cofibration、weak equivalence 的创建方式和小对象条件。
2. 每个 rectification criterion 是否说明 operad map 是什么类型的弱等价，以及底层模型范畴需要哪些 symmetric flatness 或 admissibility 假设。
3. Dendroidal inner Kan、operadic model structure 和 Lurie-style operadic fibration 之间的比较是否只在外部输入定理中出现。
4. Straightening/unstraightening 是否在同一个模型中使用，避免把 quasi-category 版本和 dendroidal 版本无说明地互换。
5. Operadic localization 中“先代数后局部化”和“先局部化后代数”的比较是否附有必要条件。

通过产物：

- 附录 D 精确化第十四至十九章的定理来源。
- 第十四章增加 positive characteristic 下 rectification 失败或不可用的例子。
- 第十九章增加一张“模型比较只可经外部输入定理使用”的依赖图。

## 6. 优先级 D：例子与计算

第四优先级是把核心定义落实到可检查例子。

需要补充：

1. $\operatorname{Ass}$、$\operatorname{Com}$、$\operatorname{Lie}$、$\operatorname{Pois}_n$ 的逐项公理检查。
2. 自由 $\operatorname{Ass}$-代数、自由 $\operatorname{Com}$-代数、自由 Lie 代数的 universal property 对照。
3. Endomorphism operad 在集合、模、链复形中的复合和等变性。
4. Little cubes operad 的基本复合验证，以及为什么其同调 operad 结论必须作为外部输入。
5. Factorization homology 在区间、圆周和可分解流形上的基本例子。

通过产物：

- 至少一个专门的例子附录。
- 每个例子都含单位、结合律、等变性和泛性质检查中的至少两类。

## 7. 优先级 E：前沿研究边界

第五优先级是第二十一章和资料源。

需要核对：

1. 每个 2025-2026 预印本的 arXiv 编号、标题、作者、提交日期和当前版本。
2. 每个前沿条目使用的 operad 模型：strict operad、operadic category、dendroidal set、linear infinity-operad、Fukaya category 或 relative infinity-operad。
3. 哪些结果只是研究方向，哪些可在本书未来版本中作为外部输入。
4. 是否存在更晚版本、撤稿、改题或出版版本。

通过产物：

- 建立前沿文献版本核查文件。
- 第二十一章不出现未经验证的“定理化”表述。
- `SOURCES.md` 对每个近期条目保留链接和版本边界。

## 8. 章节级任务表

| 范围 | 第二轮任务 | 完成标志 |
| --- | --- | --- |
| 第零至二章 | 有限集 coend 口径统一 | 自由代数公式与附录 A-B 完全对齐 |
| 第三至四章 | 树和自由 operad 严格化 | 平面/非平面/带标号树对照表完成 |
| 第五章 | colored substitution 自然性 | 输入颜色函数和分块拉平证明补全 |
| 第六章 | 经典线性例子 | Ass/Com/Lie/Pois 公理逐项检查 |
| 第七章 | properad/PROP 边界 | 图商构造仍为外部输入，条件写全 |
| 第八至九章 | Koszul 与 bar-cobar | suspension 和 twisting signs 与附录 E 对齐 |
| 第十至十三章 | 同伦代数符号 | $A_\infty/L_\infty$/brace 展开只保留一套 convention |
| 第十四至十五章 | 模型结构假设 | transferred/admissible/rectification 条件逐项列出 |
| 第十六至十七章 | dendroidal 模型 | inner horn、normal mono、operadic weak equivalence 关系图完成 |
| 第十八至十九章 | Lurie 与模型比较 | 所有比较定理均进入附录 D |
| 第二十章 | 几何应用 | factorization 与 Fukaya 只使用已声明的外部输入 |
| 第二十一章 | 前沿边界 | arXiv 版本核查完成 |
| 附录 A-J | 基础工具链、例子验算、模型假设、树约定和同伦代数计算 | 所有正文引用可回链 |

## 9. 停止横向扩张原则

第二轮期间原则上不再新增大主题，除非满足下列至少一项：

1. 新主题是为修复现有证明缺口所必需。
2. 新主题是某个外部输入定理的前置定义。
3. 新主题是为了把 2025-2026 研究边界准确放入已有逻辑地图。

否则，新材料应进入“后续版本候选”，不得插入核心定理链。
