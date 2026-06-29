# 数学审查记录

本文件记录《同伦类型论与单值基础》在文本出版口径下的数学审查结果。它不是正文，而是后续校订时的质量控制清单。

## 当前结论

1.  **核心 HoTT 主线已收口。**  
    L0-L5 的内部语言、路径代数、等价、单值性、HIT、圆的基本群、单值范畴论、Yoneda 和 Rezk 完备化架构均已有书内证明核、精确输入规则或明确外部输入。

2.  **高级接口已分层。**  
    合成上同调、稳定同伦论、Postnikov、局部系数、构造性分析、集合层代数、directed/cohesive/2LTT、Rezk/Segal 高阶相干和模型论均保留为高级接口、外部输入或研究边界，不阻塞核心封稿。

3.  **书稿外路线已移出封稿条件。**  
    新口径不以书稿外材料作为本书封稿条件。正文中若讨论模型、计算性或对象语言扩展，只作为数学来源、语义背景或研究边界。

4.  **出版审计已建立。**  
    `PUBLICATION_CLOSURE_AUDIT.md` 记录本地链接、README 覆盖、章节结构和跳步词扫描。当前结果支持文本出版收口。

## 已关闭的核心缺口

1.  路径归纳、固定端点归纳、路径复合、逆路径、transport 和 $\Sigma$ 路径已有证明核。
2.  fiber 可收缩意义下的等价、准逆相干化、等价复合、逆等价和等价保持 h-level 已补齐。
3.  函数外延性、命题外延性、单值性和结构等同性原则的使用位置已显式化。
4.  命题截断、一般截断、集合商、圆、悬挂和 pushout 的输入规则已集中列出。
5.  整数对象、successor 等价、loop 幂、encode-decode 和圆的基本群同构已有证明核。
6.  预范畴、单值范畴、Yoneda、预层范畴、函子范畴、终对象唯一性、伴随形式和 Rezk 完备化架构已有证明核或证明架构。

## 已知风险

1.  **证明说明不可滥用。**  
    若核心链中出现证明说明，必须能降为具体书内证明或外部输入。高级接口中的证明说明必须明确未展开内容的性质。

2.  **外部模型论不可替代内部证明。**  
    Simplicial model、cubical model、canonicity 和 normalization 结果是元理论输入。它们不能在对象语言中当作普通项构造使用。

3.  **高级接口不可回流。**  
    Directed hom、cohesive modal rules、2LTT strict equality、QIIT 语法和 Rezk/Segal 高阶相干不得混入 L0-L5 的普通 identity type 证明。

4.  **构造性边界必须保留。**  
    LEM、DNE、choice、resizing、locatedness 和经典完备性原则不得默认使用。相关结论必须列出假设。

5.  **稳定同伦论计算仍是接口。**  
    Steenrod operations、Ext、Adams differentials、具体谱序列收敛和低阶球面同伦群计算不属于核心封稿条件。

## 当前外部输入

1.  HoTT 基础理论的一致性和模型背景；
2.  单值性和函数外延性之间的标准关系；
3.  一般 HIT、HIIT、QIIT 的元理论存在性和语义；
4.  EM 型塔、Blakers-Massey、Freudenthal、Hopf fibration、Postnikov 和谱序列的高级定理形态；
5.  Cubical type theory、simplicial type theory、2LTT、cohesive HoTT 和合成代数几何的模型或对象语言结果。

## 后续审查清单

- 新章节是否更新 `README.md` 阅读顺序。
- 新符号是否更新 `NOTATION.md`。
- 新来源是否更新 `SOURCES.md` 或附录 S。
- 每个定理是否标注“书内证明”“证明说明”“外部输入”或“研究边界”。
- 是否误用了函数外延性、命题外延性、单值性、高阶归纳类型、截断、排中律或选择原则。
- 是否把拓扑空间直觉、模型论语义或高级对象语言规则直接当作类型论证明。
- 是否把 L6-L9 的高级接口误写成 L0-L5 的核心定理。
