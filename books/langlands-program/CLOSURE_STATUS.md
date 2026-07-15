# 收口标准与当前状态

本文档给出本书进入收口阶段的判据、当前判定和后续准入规则。这里的“收口”不表示把所有相关外部理论重写成完整专著，也不表示已经完成最终出版审定；它只表示 Langlands 主线已经闭合到可阅读、可追溯、可审查的教材形态。

## 基本收口标准

本书达到基本收口，必须同时满足以下条件。

1. 主线目录固定：正文只保留 `GL(1)`、`GL(2)` 与费马应用、一般算术 Langlands、几何 Langlands 四条主线，不再新增第五条同级主线。
2. 对象链闭合：每条主线中的域、局部群、表示、参数、Hecke 作用、L 因子、导子、谱对象和几何对象均已有定义或明确外部输入来源。
3. 定理状态闭合：每个非平凡命题被标为已证、外部输入的证明路线、外部输入或猜想，并能在定理索引或外部输入索引中追踪。
4. 外部输入闭合：每个外部输入被分为核心结构、支撑接口或卫星理论；正文只使用精确定理陈述、假设、归一化和使用位置。
5. 归一化闭合：Frobenius、Haar 测度、Fourier 变换、Satake 参数、局部类域论、classical normalization 与 unitary normalization 的转换点均有交叉引用。
6. 应用链闭合：费马大定理应用章只使用“外部输入定理 + 本书已证引理 + 逻辑推出”的形式，不把 Taylor-Wiles、Ribet 或 Neron 模型理论重写为正文证明目标。
7. 阅读路径闭合：`DEPENDENCY_GRAPH.md` 给出的四条阅读路径，每条都有最短入口、支撑附录和不可替代外部输入清单。
8. 习题与索引闭合：每条主线至少有核心计算题或概念题，且答案能回指正文定义和命题。

若上述条件有一项缺失，本书不能视为基本收口版本。

## 当前判定：审定前闭合

截至第十三轮出版前维护，本书已经进入审定前闭合版。这里的“审定前闭合”表示四条主线、关键归一化、索引一致性、概念边界、局部主线小补、主体与附录接口、来源索引和应用链均已形成可审查闭环；它不表示最终出版审定版，也不表示外部输入理论已在本书内完整证明。

| 项目 | 当前状态 | 判定 |
|---|---|---|
| 主线目录 | 已覆盖四条主线，附录 A-AE 已承担主要支撑接口；不再新增第五条同级主线 | 闭合 |
| 对象链 | 四条主线的域、群、表示、参数、Hecke 作用、L 因子、导子、谱对象和几何对象均已有定义或外部输入落点；概念边界见 [CONCEPTUAL_AUDIT.md](CONCEPTUAL_AUDIT.md) | 主线闭合 |
| 定理状态 | `THEOREM_INDEX.md` 已建立，且 [INDEX_CONSISTENCY_AUDIT.md](INDEX_CONSISTENCY_AUDIT.md) 与 [NUMBERING_CROSSREF_AUDIT.md](NUMBERING_CROSSREF_AUDIT.md) 未发现阻塞收口的状态冲突 | 主线闭合 |
| 外部输入 | 已建立三分法，重点外部输入来源已在附录 E 拆细；主体和附录中被证明链引用的外部输入已收紧假设和版本选择 | 接口闭合 |
| 归一化 | 已建立 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md)，高风险主章和高风险附录均已加入回指，审定前通读已收紧残留泛称 | 高风险层闭合 |
| 费马应用链 | 逻辑链已经成立，局部导子、级数矛盾、外部输入状态和习题闭环均已补接口 | 应用闭合 |
| 一般算术 Langlands | 对象链、最短证明链、trace formula/endoscopy/Arthur/functoriality 的状态边界已集中说明 | 对象链闭合 |
| 几何 Langlands | 主线入口、几何 Satake、Hecke eigensheaf、谱侧范畴和函数域桥梁均已形成接口链 | 接口闭合 |

结论：本书不应继续横向扩张新的大方向。逐章风险清理已转写为 [CHAPTER_CLOSURE_AUDIT.md](CHAPTER_CLOSURE_AUDIT.md)，重点外部输入来源已在 [E_external_input_theorem_index.md](E_external_input_theorem_index.md) 拆细，编号与交叉引用审计已记录于 [NUMBERING_CROSSREF_AUDIT.md](NUMBERING_CROSSREF_AUDIT.md)，审定前概念边界已记录于 [CONCEPTUAL_AUDIT.md](CONCEPTUAL_AUDIT.md)，第六、七轮主线小补已补入正文，第八至十轮审定前通读已完成主体和附录接口严格化；后续工作是出版前审定维护：修正局部证明细节、措辞、来源标注、索引和排版。

## 是否还需主线扩张

不需要新的同级主线扩张。现有目录已经覆盖 Langlands 纲领作为教材必须解释的四个方向：

1. `GL(1)`：Tate thesis、Hecke 特征、idele class characters 和类域论。
2. `GL(2)` 与费马应用：模形式、椭圆曲线、Galois 表示、局部-整体相容、降层和费马大定理应用链。
3. 一般算术 Langlands：还原群、L 群、局部参数、自守表示、`GL(n)` 定理、函子性、trace formula、endoscopy 和 Arthur 参数。
4. 几何 Langlands：`G`-bundles、Hecke 修改、几何 Satake、Hecke eigensheaves、谱侧范畴和函数域桥梁。

允许的后续新增内容只应是主线小补，不应成为新附录群。

## 收口维护规则

以下任务用于维持审定前闭合状态，不属于横向扩张。

1. 维护 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 的总约定地位；新增或修订涉及 Frobenius、Artin reciprocity、Satake、Haar、Fourier、L 函数变量和 Tate twist 的段落时必须回指该表。
2. 维持 `MATH_REVIEW.md` 和 [CHAPTER_CLOSURE_AUDIT.md](CHAPTER_CLOSURE_AUDIT.md) 的收口清单，避免把背景专著项目重新提升为正文目标。
3. 维护 `THEOREM_INDEX.md` 和 `E_external_input_theorem_index.md`，保证每个主章外部输入都有来源、使用章节和状态。
4. 维护 [MAINLINE_PROOF_CHAINS.md](MAINLINE_PROOF_CHAINS.md)：每条阅读路径都应标明本书证明、外部输入的证明路线、外部输入或猜想。
5. 维护主线核心接口例子：rank-one Satake、`GL(2)` 非分歧参数、Frey 曲线导子、`SL_2` packet、几何 Satake 的最小 Hecke 作用。
6. 维护 [EXERCISE_COVERAGE.md](EXERCISE_COVERAGE.md)，确保四条主线都有已解闭环练习。

## 主线小补准入

下列内容可以继续写入本书，但只允许写到服务主线的接口深度。

| 主题 | 允许深度 | 不允许目标 |
|---|---|---|
| 一般 LCA Fourier 分析 | 固定 Poisson/Tate thesis 使用的定理、测度和归一化 | 完整调和分析专著 |
| 类域论 | reciprocity map、conductors、ray class、Artin L 函数接口 | 完整 class formation 证明 |
| Strong approximation 和 adelic 比较 | 经典模形式到 adelic 表示的良定义性和 Hecke 比较 | 一般代数群强逼近专论 |
| Neron/Tate algorithm | 导子、Kodaira 类型、Frey 曲线局部计算接口 | Neron 模型存在性完整证明 |
| Taylor-Wiles/Ribet | 定理陈述、假设、逻辑使用点 | patching 或降层技术完整证明 |
| Bruhat-Tits 和 Harish-Chandra | hyperspecial、parahoric、Satake、字符和 Plancherel 接口 | 完整建筑和局部调和分析证明 |
| Trace formula | 项级字典、稳定化使用位置、endoscopy 接口 | Arthur trace formula 完整证明 |
| 几何技术层 | D-modules、IndCoh、factorization、BD Grassmannian 的 Langlands 用法 | 派生代数几何或六运算完整构造 |
| Fargues-Scholze | 局部几何 Langlands 的对象字典和与 LLC 的关系 | perfectoid/diamond 理论完整重建 |

## 后置或另卷内容

以下内容不再作为本书基本收口前的扩写目标。

1. 完整 class formation cohomological proof。
2. 完整 Tate thesis 解析证明卷。
3. 完整代数曲线、模曲线代数化和 Atkin-Lehner-Li 理论。
4. 完整 Neron 模型存在性和 Tate algorithm 逐步证明。
5. 完整 Taylor-Wiles patching、Poitou-Tate 和 p-adic Hodge theory。
6. 完整 Harish-Chandra、Bruhat-Tits、Bernstein center 和 Plancherel 理论。
7. 完整 Arthur trace formula、稳定化、基本引理和 twisted trace formula。
8. 完整 Arthur-Mok 分类证明。
9. 完整 D-module、derived stack、IndCoh、six functors 和 factorization theory。
10. 完整 perfectoid、diamond、Fargues-Fontaine 和 local shtuka 理论。

这些理论可以被精确引用，但不再驱动本书继续扩张。

## 出版前审定维护顺序

1. 全文通读，修正数学错误、措辞不严、来源不明和归一化回指缺失。
2. 维护编号、交叉引用、外部输入索引和习题解答回指。
3. 发布前再做一次全量编号、链接、格式和术语审计。

从本状态开始，新增大块理论必须被视为另卷或新版本目标，而不是当前审定前闭合版的收口任务。
