# 前沿文献版本核查记录：2026-06-29

本文件记录第二十一章使用的近期文献入口。它不是证明来源索引；证明来源索引仍为附录 D。本文件只回答三个问题：

1. 该条目是否有可追踪的 arXiv 或公开来源。
2. 它在本书中应进入哪一块逻辑地图。
3. 它目前是否允许进入核心定理链。

结论：以下 2025-2026 相关条目截至 2026-06-29 只作为研究边界使用，不作为本书已证明定理，也不作为基础章节的无条件外部输入定理。

## 1. 核查规则

每个近期条目进入正文前必须满足：

1. 标题、作者、arXiv 编号和提交日期已核对。
2. 若有多个版本，正文必须写明使用版本。
3. 论文使用的 operad 模型必须与本书章节模型对应：strict operad、linear operad、dendroidal set、Lurie-style infinity-operad、operadic category、Fukaya category 或 relative infinity-operad。
4. 若引用结果是主定理，必须记录定理编号和依赖的前置假设。
5. 若无法满足以上条件，该条目只能保留在第二十一章研究边界中。

## 2. 核查表

| 条目 | 公开来源 | 本书逻辑位置 | 当前使用状态 |
| --- | --- | --- | --- |
| Hoffbeck-Moerdijk, *Homology of infinity-operads* | arXiv:2105.11943, <https://arxiv.org/abs/2105.11943> | 第十七至十九章之后的 infinity-operad 同调背景 | 可作背景；定理化需核对定义和 theorem numbering |
| Hoffbeck-Moerdijk, *Koszul duality for algebras over infinity-operads* | arXiv:2602.08851, <https://arxiv.org/abs/2602.08851> | 第八至十章 Koszul/bar-cobar 与第十七章之后的 infinity-operad 交界 | 只作 2026 研究边界 |
| Pavlova, *Boardman-Vogt tensor product and wreath product of operadic categories* | arXiv:2601.03985, <https://arxiv.org/abs/2601.03985> | 第七章 Boardman-Vogt 背景与 operadic categories 后续方向 | 只作研究边界；不得替代 Boardman-Vogt 或 Batanin-Markl 基础定义 |
| Yuan, *Higher operad structure for Fukaya categories* | arXiv:2603.08039, <https://arxiv.org/abs/2603.08039> | 第二十章 Fukaya category 与几何应用 | 只作研究边界；Fukaya 构造仍依赖几何分析外部输入 |
| Arakawa-Carmona-Pratali, *Relative dendroidal Rezk nerve and applications* | arXiv:2606.11895, <https://arxiv.org/abs/2606.11895> | 第十七至十九章 dendroidal localization 与 relative infinity-operads | 只作研究边界；不能作为 Cisinski-Moerdijk 模型结构的替代证明 |
| Batanin-Kock-Weber, *Operadic categories as (pseudo)-simplicial groupoids* | arXiv:2606.15671, <https://arxiv.org/abs/2606.15671> | operadic categories、higher nerve 和未来扩展章节 | 只作研究边界；正文尚未建立 operadic category 基础 |

## 3. 逐条边界说明

### 3.1 Infinity-operad 的同调与 Koszul 对偶

Hoffbeck-Moerdijk 的 2021 条目为 infinity-operad 的同调理论和 Koszul 型结构提供背景。2026 条目把 Koszul duality 推向 algebras over infinity-operads。它们与第八章、第九章、第十七章和第十九章都有交叉，但不能直接并入本书的 classical Koszul duality 证明链。

进入正文定理链前需要补齐：

1. linear infinity-operad 的精确定义；
2. 该定义与 dendroidal set、dg-operad 或 Lurie-style infinity-operad 的比较；
3. bar-cobar 或 Koszul functor 的源、目标和 weak equivalence 概念；
4. 主定理编号、版本号和假设。

### 3.2 Operadic categories 与 Boardman-Vogt tensor product

Pavlova 的条目涉及 operadic categories、Boardman-Vogt tensor product 和 wreath product。它与第七章和未来可能加入的 operadic category 专章相关，但当前正文尚未把 operadic categories 作为基础对象建立。

当前处理原则：

1. 不把该文结果作为第七章 PROP/properad 定义的依据。
2. 不把其 Boardman-Vogt tensor product 与第十四章的 Boardman-Vogt resolution 混同。
3. 若未来加入 operadic categories，需要先定义 Batanin-Markl 体系中的 operadic category，再比较该文使用的模型。

### 3.3 Relative dendroidal Rezk nerve

Arakawa-Carmona-Pratali 的条目与 relative dendroidal Rezk nerve 和 localization 相关。它接近第十七章 dendroidal model structure 和第十九章 operadic localization 的交界，但本书当前只把 Cisinski-Moerdijk 模型结构、dendroidal-Lurie 比较和 localization 比较作为外部输入。

进入正文前需要补齐：

1. relative dendroidal object 的定义；
2. Rezk nerve 的源范畴和目标模型结构；
3. localization 前后的 weak equivalence；
4. 应用定理所需的 fibrancy、completeness 或 Segal 条件。

### 3.4 Fukaya categories 的高阶 operadic 结构

Yuan 的条目与 Fukaya categories 的高阶 operadic structure 相关。它可放在第二十章的研究边界中，但不能削弱第二十章已经声明的几何分析依赖。

进入正文前需要补齐：

1. 所用 Fukaya category 的具体版本；
2. brane data、transversality、compactness、orientation 和 obstruction theory 假设；
3. operadic structure 的模型；
4. 与既有 $A_\infty$-category、sectorial descent 或 factorization homology 叙述之间的比较。

### 3.5 Operadic categories as simplicial groupoids

Batanin-Kock-Weber 的条目把 operadic categories 与 pseudo-simplicial groupoids 联系起来。它属于高阶 operadic categories 和 nerve 技术方向。由于本书当前没有 operadic categories 的完整基础章节，不能把该条目用于证明 dendroidal set 或 Lurie-style infinity-operad 的基础结论。

未来若吸收该方向，应先加入：

1. operadic category 的定义；
2. terminal objects、fiber functor 和 cardinality functor；
3. operadic nerve 的定义；
4. 与 dendroidal nerve 和 category of operators nerve 的关系。

## 4. 对正文的约束

第二十一章可以陈述这些论文研究了什么对象、与本书哪些章节相邻、为什么需要后续验证。它不能陈述如下形式的未核查断言：

1. “由某 2026 论文可知，本书第九章定理推广到所有 infinity-operad。”
2. “Fukaya categories 自然形成某种高阶 operad 代数”，除非说明具体模型和假设。
3. “Relative dendroidal Rezk nerve 给出 operadic localization 的最终模型”，除非给出定理编号、模型结构和比较函子。
4. “Operadic categories 与 dendroidal sets 等价”，除非明确这是哪种范畴、哪种 nerve、哪种等价。

## 5. 下一轮核查任务

下一轮若继续推进前沿部分，应逐篇补全：

| 条目 | 必须补全的信息 |
| --- | --- |
| arXiv:2602.08851 | 当前版本号、主定理编号、linear infinity-operad 定义 |
| arXiv:2601.03985 | 当前版本号、Boardman-Vogt tensor product 定义、wreath product 定理 |
| arXiv:2603.08039 | Fukaya category 模型、operadic structure 类型、几何假设 |
| arXiv:2606.11895 | relative dendroidal Rezk nerve 定义、localization 定理编号 |
| arXiv:2606.15671 | operadic category 与 pseudo-simplicial groupoid 的等价形式 |

在完成这些核查以前，本书的“最新研究”覆盖只能称为“研究边界覆盖”，不能称为“完整吸收最新研究成果”。

