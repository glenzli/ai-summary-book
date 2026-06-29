# 基本完本判据与证明闭包矩阵

作者：Dr. Stochastic Parrot

## 0. 为什么不能直接宣称完本

本书若按“完全自足教材”标准，必须重证以下大块内容：

1. Gleason lifting theorem；
2. Nöbeling theorem 的一般 profinite 情形；
3. Scholze solid/analytic/liquid 的核心结构定理；
4. Cartan A/B、Grauert、Hodge-Fredholm、Serre duality、GAGA、GRR；
5. pyknotic 和谱值 solid/analytic localization 的高阶范畴理论。

其中 solid、analytic 和 liquid 属于本书的凝聚数学主线；但它们的核心结构定理是 Scholze/Clausen-Scholze 理论中的深层结果。若把这些结果连同拓扑、泛函分析、复几何、代数几何和高阶范畴论的全部预备一起重证，本项目会变成多套教材的合本。因此本书的合理“基本完本”标准不能是完全自足，而应是主线输入定理型闭合：

> 所有定义、主定理、证明依赖、外部输入和形式推论都闭合；书内能证明的命题给出证明；书外深定理以精确输入定理登记，不在正文中伪装成已证。

## 1. 三种完本标准

### 标准 A：完全自足完本

要求从集合论、拓扑、同调代数、solid/analytic、复几何到 GRR 全部书内证明。

**当前判断。** 不适合作为本项目目标。它会要求新增数千页内容，并且重复多部标准教材和论文。

### 标准 B：主线输入定理型严格教材草稿完本

要求：

1. condensed、solid、analytic、liquid 四层主线完整；
2. 每个外部输入定理集中登记；
3. 每个主定理有依赖链；
4. 每个形式推论有书内证明；
5. 每章有例子、反例或类型检查；
6. 练习有答案要点；
7. 不把外部输入写成已证定理。

**当前判断。** 已达到本项目合理的“主线输入定理型最终收口版”标准。solid、analytic 和 liquid 已纳入主线卷；第二卷附录 Q-Z 已将三条主线收束为主定理包、完成出版级闭包审查，并把 solidification、solid 核张量理想性、analytic localization、rational descent 和 liquid realization 拆成证明模块；第二卷附录 AA 和第三卷附录 AR 又把 Scholze/Clausen-Scholze 的核心定理集中列为主线图谱。其深层核心结构定理仍作为精确输入定理登记，接受输入后的形式推论、类型检查、例子和依赖链已经闭合。

### 标准 C：研究导读完本

要求读者能用本书理解 Scholze/Clausen-Scholze 资料的结构、术语和主定理位置，但不要求教材式证明闭合。

**当前判断。** 已达到并超过该标准。

## 2. 当前闭包矩阵

| 模块 | 书内闭合度 | 外部输入 | 当前完本判断 |
| --- | --- | --- | --- |
| 站点与 sheaf | 高 | 小性与一般 Grothendieck 拓扑背景 | 基本闭合 |
| 凝聚集合/阿贝尔群 | 高 | sheafification 一般理论局部输入 | 基本闭合 |
| Stone/profinite | 高 | Boolean prime ideal theorem | 基本闭合 |
| Gleason/ED 投射 | 中高 | Gleason lifting | 输入边界清楚 |
| Ext/Tor 基础 | 高 | K-flat/K-injective 存在性 | 基本闭合 |
| Nöbeling | 中 | 一般 profinite 超限过滤 | 输入边界清楚 |
| solid | 中高 | Scholze solidification 识别、profinite 测度张量公式 | 结构闭合，证明模块已拆，深层计算依赖外部 |
| analytic rings | 中高 | analytic localization 与 rational descent | 结构闭合，证明模块已拆，深层 descent 依赖外部 |
| liquid | 中高 | liquid realization exactness | 类型边界清楚，证明模块已拆 |
| Dolbeault | 中高 | 局部估计与正则性 | 形式闭合，分析输入外部 |
| Cartan/Grauert | 中 | Oka/Runge/Cousin/Grauert | 输入拆分充分 |
| Serre duality | 中高 | 完美性、dualizing complex | 形式闭合，深层输入外部 |
| GAGA | 中高 | analytic finite generation/Grothendieck existence | 形式闭合，深层输入外部 |
| HRR/GRR | 中 | localized Chern character/GRR basic factors | 形式闭合，深层输入外部 |
| pro-etale/pyknotic | 中 | 高阶拓扑与谱值 localization | 工具卷闭合 |

## 3. 最终收口后的非阻塞维护

按标准 B，本书现在可以称为“condensed/solid/analytic/liquid 主线输入定理型最终收口版”。仍可继续做的工作属于非阻塞出版维护：

1. **输入定理编号回填正文。** 现有登记表已集中列出外部输入；后续可把每个正文定理的引用进一步回填到具体输入编号。
2. **引用定位提升。** [REFERENCE_LOCATOR_LEDGER.md](REFERENCE_LOCATOR_LEDGER.md) 已建立 L0-L3 状态；凝聚主线输入已经提升到 L2/L3，经典输入已经从 L0 清到 L1/L2/L3 分层状态。后续重点是给 Boolean/Sikorski 补 edition locator，并给 Cartan、Grauert、Dolbeault、Hodge-Fredholm 和 GRR 补 theorem/page。
3. **教师手册级答案。** 分卷答案已经建立；出版级版本应把所有证明题扩成逐行解答。
4. **反例密度继续提高。** 重要技术假设已有边界例子；后续可为每个关键假设配置一个删除失败例子。
5. **主线核心输入自足化。** 若要重证 solidification、analytic localization、liquid realization 等主线核心定理，需要把 Scholze/Clausen-Scholze 的长证明展开为独立章节。
6. **深层应用输入自足化。** 若要重证 Cartan/Grauert、Hodge-Fredholm、GAGA 和 GRR，需要另写复几何、代数几何和泛函分析预备教材。

这些项目不影响最终收口判断；它们只会把本书从“输入定理型最终收口版”继续推向“完全自足证明版”或“逐题教师手册版”。

## 4. 为什么之前一直没有推到基本完本

此前没有推到“基本完本”的原因不是单纯内容量不足，而是完成标准一直混合了两件不同的事：

1. **完全自足证明版。** 这要求重证多个学科的深层定理，不适合作为当前仓库的一本凝聚数学教材目标。
2. **主线输入定理型严格教材草稿。** 这要求 condensed、solid、analytic、liquid 主线闭合、依赖闭合、输入定理登记、形式推论证明、练习答案和边界例子。

本轮已经补齐第二种标准所需的闭包结构：

1. 外部输入定理登记表；
2. 术语索引；
3. Gleason 与 Nöbeling 的证明模块；
4. 形式化证明义务；
5. 谱值/pyknotic 接口；
6. 卷三主定理闭包章；
7. 统一答案与分卷答案入口；
8. 第二卷 solid/analytic/liquid 主定理包与统一闭包；
9. 第二卷 solidification、solid 核张量理想性、analytic localization、rational descent 和 liquid realization 证明模块；
10. Scholze/Clausen-Scholze 核心定理图谱；
11. 基本完本判据、定理索引和依赖图。

因此，按标准 B，本书已经从“继续膨胀的讲义”推进到“主线输入定理型最终收口版”。按标准 A，它仍然不是也不应被声称为完全自足证明版教材。

## 5. 对外口径

推荐对外描述：

> 这是一套四卷中文凝聚数学严格教材草稿。condensed、solid、analytic 和 liquid 都作为主线处理；基础层给出书内证明，solid/analytic/liquid 核心结构定理和复几何深层结果以精确输入定理登记；书内证明所有接受输入后的形式推论、计算模型、例子、反例和依赖链。

不推荐描述：

> 完全自足证明了凝聚数学、solid theory、complex geometry 和 GRR。

后者不符合当前数学事实。
