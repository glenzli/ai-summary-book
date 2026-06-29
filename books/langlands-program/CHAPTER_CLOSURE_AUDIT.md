# 逐章收口缺口审查

本文档把 [MATH_REVIEW.md](MATH_REVIEW.md) 中原先的“当前风险”改写为可执行的逐章收口台账。判定原则是：只记录会影响 Langlands 主线阅读路径、归一化一致性、外部输入状态或应用链闭环的缺口；完整背景理论证明一律后置或另卷，不再作为本书收口阻断项。

状态标记如下。

- `闭合`：主线阅读不被阻断，仅需维护交叉引用。
- `精校`：主线已可读，但需要来源、编号、假设或局部措辞精校。
- `维护`：新增或改写相关段落时必须保持既有约定。
- `后置`：完整证明不属于本书基本收口目标。

## 总体判定

本轮逐章审查未发现需要新增同级主线的缺口，也未发现应用章把猜想当作定理使用的阻断项。第六轮已补入第 3、7、10、14、16、19、22、90 章的接口检查表或最小模型说明；第七轮已补入第 1、2、5、8、12、17 章的使用边界表，并完成附录层精校状态审稿；第八至十轮已完成主体和附录接口严格化、来源索引抽查和残留泛称清理；第十一至十二轮已完成最终收口口径统一和最终概念审定。当前进入出版前审定维护：

1. 不再保留阻断收口的 `精校` 项；除非发现新的数学错误，不再新增同级主线。
2. 维护外部输入来源索引；新增外部输入必须先进入 [E_external_input_theorem_index.md](E_external_input_theorem_index.md) 或 [SOURCES.md](SOURCES.md)。
3. 维护 [NUMBERING_CROSSREF_AUDIT.md](NUMBERING_CROSSREF_AUDIT.md) 的规则；新增主线命题或已解习题时同步检查编号。

## 正文章节台账

| 范围 | 当前判定 | 外部输入边界 | 收口动作 |
|---|---|---|---|
| 第 0 章 | 闭合 | 无新增数学输入 | 维护范围声明：本书不替代外部专著 |
| 第 1 章 | 维护 | `\mathbb A_K/K` 紧性、LCA Fourier inversion、Plancherel、Poisson summation | 已补 adelic analysis 使用边界；不补完整 LCA 证明 |
| 第 2 章 | 维护 | Tate thesis 的局部函数方程、Archimedean gamma 因子、整体解析延拓 | 已补 Tate thesis 使用边界；Poisson-theta 骨架保持附录 F 接口 |
| 第 3 章 | 维护 | 局部/全局类域论、class formation、Herbrand 理论 | 已补 class formation 使用点、ray class、conductor、有限阶 Hecke character 与 Weil/Galois 侧边界 |
| 第 4 章 | 维护 | Haar 存在唯一性、Satake、Harish-Chandra 理论 | 保持 Haar/Satake 归一化回指；不展开 Harish-Chandra 证明 |
| 第 5 章 | 维护 | `GL(n)` LLC 和一般 reductive L-packet | 已补局部参数使用表；一般 LLC 保持外部输入或猜想 |
| 第 6 章 | 维护 | 模形式有限维性、Hecke 良定义性、newform 函数方程、Deligne 表示 | 维护 classical normalization 与 Galois Frobenius 回指 |
| 第 7 章 | 维护 | strong approximation、classical-to-adelic 对应、newform 生成自守表示 | 已补 classical-to-adelic 比较检查表；strong approximation 保持外部输入 |
| 第 8 章 | 维护 | Neron 模型、Neron-Ogg-Shafarevich、Tate algorithm、导子公式、模性 | 已补椭圆曲线到 `GL(2)` 接口表；Frey 应用只用局部导子接口 |
| 第 9 章 | 维护 | Fontaine-Mazur、Deligne 表示、模性提升、`R=T` | 保持 residual/lift/elliptic modularity 的分层表述 |
| 第 10 章 | 维护 | 局部-整体相容、Serre 模性、Ribet 降层、Frey 导子 | 已补 Ribet 降层假设表、使用位置和级删除逻辑 |
| 第 11 章 | 维护 | reductive group 结构定理、root datum 分类、pinning 的 Galois 作用 | 维护 L 群、Satake 和非分歧参数的总约定 |
| 第 12 章 | 维护 | `GL(n)` LLC、tori、Archimedean LLC、enhanced LLC、endoscopy | 已补 LLC 状态边界；不补逐群分类表 |
| 第 13 章 | 维护 | 离散谱、restricted tensor product、Godement-Jacquet、Rankin-Selberg、Shahidi、一般解析性质 | 维护 Euler 因子定义与解析输入状态 |
| 第 14 章 | 维护 | BZ 分类、`GL(n)` LLC、强重数一、converse theorem、Lafforgue、数域 RAECSDC 构造 | 已补函数域定理、数域已知情形和完整猜想边界 |
| 第 15 章 | 维护 | 一般函子性仍属猜想；base change、automorphic induction、低阶 lifts 为外部输入 | 保持弱/强转移和已知特殊情形分层 |
| 第 16 章 | 维护 | Arthur trace formula、稳定化、transfer factor、fundamental lemma、twisted trace formula | 已补 trace formula 使用边界；不补完整 trace formula 证明 |
| 第 17 章 | 维护 | Arthur packets、multiplicity formula、classical groups 标准转移 | 已补 Arthur 输入边界；维护 Arthur-Mok 外部输入状态和 non-tempered 谱解释 |
| 第 18 章 | 维护 | 代数栈、Hecke stack 技术 | 保持几何对象链；深层栈技术后置 |
| 第 19 章 | 维护 | 几何 Satake | 已补最小 Hecke 作用模型；维护 Tate twist、$q$-因子和 sheaf-function convention |
| 第 20 章 | 维护 | 几何类域论、Hecke eigensheaf 存在性 | 保持定义层闭合；一般存在性不作为正文定理 |
| 第 21 章 | 维护 | categorical geometric Langlands、D-modules、IndCoh、singular support | 保持范畴接口；完整技术层后置 |
| 第 22 章 | 维护 | shtukas、Drinfeld-Lafforgue、V. Lafforgue、Ngô 支持定理 | 已补函数域桥梁与 sheaf-function trace convention |
| 第 90 章 | 闭合 | Frey 曲线性质、半稳定模性、Ribet 降层 | 应用链闭合；已补费马应用输入表，后续只维护来源和假设 |

## 附录层台账

| 范围 | 当前判定 | 收口动作 |
|---|---|---|
| 附录 A-D | 维护 | 基础对象和费马应用所需计算已补；完整类域论、Haar 存在性、模曲线代数化后置 |
| 附录 E | 维护 | 重点外部输入来源已拆细；新增外部输入必须先登记 |
| 附录 F-G | 维护 | Fourier/Poisson 与 L 群计算作为基础接口；不扩写为完整背景卷 |
| 附录 H-W | 维护 | `GL(2)`、`GL(n)`、Galois deformation、局部因子、class formation、模曲线接口已足够支撑主线；审定前接口假设已收紧 |
| 附录 X-AC | 维护 | Arthur、几何 Satake、Fargues-Scholze 重点来源已拆细；derived/IndCoh 后续只做编号、引用和版本选择维护 |
| 附录 AD-AE | 维护 | Frey 局部导子和 `GL(2)` 局部 LLC 例子已支撑应用链；后续维护假设和引用编号 |

## 非阻断后置项

下列项目不再作为本书基本收口前的缺口。

1. 完整 Tate thesis 解析证明。
2. 完整 class formation 和 Herbrand 理论。
3. 完整 Atkin-Lehner-Li、Deligne 表示构造和模曲线代数化。
4. 完整 Neron 模型存在性和 Tate algorithm 全表。
5. 完整 Taylor-Wiles patching、Poitou-Tate 和 p-adic Hodge theory。
6. 完整 Harish-Chandra、Bruhat-Tits、Bernstein center 和 Plancherel 理论。
7. 完整 Arthur trace formula、稳定化、fundamental lemma 和 Arthur-Mok 分类。
8. 完整 D-module、IndCoh、derived stack、factorization、perfectoid、diamond 和 Fargues-Fontaine 技术层。

## 出版前审定维护优先级

1. 进行出版前全文通读：只修正数学错误、措辞、来源、编号和归一化。
2. 不新增同级主线；只允许补足定义、假设、归一化或使用位置。
3. 新增结果时按 [NUMBERING_CROSSREF_AUDIT.md](NUMBERING_CROSSREF_AUDIT.md) 规则维护索引和链接。
