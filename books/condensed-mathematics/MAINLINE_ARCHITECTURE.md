# 凝聚数学主线架构

作者：Dr. Stochastic Parrot

## 0. 修订原则

本书的“凝聚数学主线”不应只理解为 condensed sets 和 condensed abelian groups。按 Scholze 与 Clausen-Scholze 的实际理论结构，主线至少包括四层：

1. condensed：凝聚集合、凝聚阿贝尔群、凝聚模和基本同调代数；
2. solid：solidification、solid tensor、solid 环与 solid 模；
3. analytic：analytic rings、analyticization、rational localization 和解析下降；
4. liquid：liquid 向量空间、Fréchet/Banach 边界、Dolbeault 与连续线性算子的 analytic-liquid 实现。

因此，solid、analytic 和 liquid 不是应用附录，而是凝聚数学从基础范畴进入几何、分析和同调代数的主干。应用卷可以引用它们，但不能替代它们的主线位置。

## 1. 当前四卷安排

当前仓库维持四卷结构，不立即移动已有文件。修订后的职责如下。

| 分卷 | 主线职责 | 当前处理方式 | 后续增强方向 |
| --- | --- | --- | --- |
| 卷一：凝聚数学基础 | 建立 condensed 层：站点、sheaf、profinite/ED、凝聚集合、凝聚阿贝尔群、凝聚模和基础 Ext/Tor | 已基本闭合；第十二至十五章只作为 solid/analytic 入口 | 不再把 solid/analytic 主证明塞入卷一，只保留动机、定义入口和前置对象 |
| 卷二：Solid、Analytic 与 Liquid 结构 | 承担 solid/analytic/liquid 主线 | 已给出定义、输入定理、类型检查、形式推论和证明义务 | 这是后续扩写的核心卷，应继续补 solidification、analytic localization、liquid realization 的细节 |
| 卷三：复几何与相干对偶 | 应用卷：把卷二主线用于 Dolbeault、有限性、Serre duality、GAGA、HRR/GRR | 已作为输入定理型应用闭包 | 不应把 liquid 主线证明放在卷三；卷三只验证应用中需要的类型和形式后果 |
| 卷四：形式化、计算与例子 | 工具卷：形式化义务、计算样板、反例、pro-etale/pyknotic 接口 | 已作为工具卷基本闭合 | 补 Lean 代码、更多 solid/analytic/liquid 计算和谱值接口 |

## 2. 卷二内部应按三条主线重排

第二卷现在标题已经覆盖 solid、analytic 与 liquid，但后续扩写时应避免把三者混成一团。建议按以下结构维护。

### 2.1 Solid 主线

放在第二卷前半部分。核心内容包括：

1. \(D(\mathbf{CondAb})\) 中的 solid 局部对象；
2. solidification 作为反射局部化；
3. Dirac-to-measure cone 与 kernel 生成；
4. solid kernel 的张量理想性；
5. solid tensor product 与闭对称幺半结构；
6. solid 环、solid 模、相对 solid 张量积；
7. profinite 测度对象和 \(\prod_I\mathbb Z\) 型计算。

当前对应位置：

- 第二卷第一、二章；
- 附录 C、E、K、L、M、O；
- 附录 Q 给出 solid 主定理包；
- 附录 V-W 把 solidification 反射存在性和 solid 核张量理想性拆成证明模块；
- 第四卷第四章作为计算补充。

尚未书内完全证明的核心输入：

- solid 反射存在性；
- solid kernel 生成；
- solid kernel 张量理想性；
- solid tensor 与测度对象的完整相容性。

### 2.2 Analytic 主线

放在 solid 之后，因为 analytic ring 的定义和解析化依赖 solid/condensed 派生语言。核心内容包括：

1. pre-analytic ring 与 analytic ring 条件；
2. analytic module 与 analyticization；
3. Bousfield localization 口径；
4. analytic tensor 与闭结构；
5. Huber pair 的 analytic ring；
6. rational localization；
7. rational Čech descent。

当前对应位置：

- 第二卷第三、四、六章；
- 附录 I、N；
- 附录 R 给出 analytic 主定理包；
- 附录 X-Y 把 analytic localization 与 rational descent 拆成证明模块；
- 第一卷第十四、十五章只作入口；
- 第三卷只使用 analytic 语言表达复几何对象。

尚未书内完全证明的核心输入：

- analytic ring 公理推出反射局部化；
- analytic tensor 与 localization 相容；
- Huber pair 给出 analytic ring；
- rational localization 的 descent。

### 2.3 Liquid 主线

liquid 也应放在第二卷，而不是只在第三卷 Dolbeault 应用里出现。它是 analytic 主线进入实/复分析对象的桥。核心内容包括：

1. \(p\)-liquid 测度对象；
2. \(p\)-liquid 实向量空间；
3. liquid 模的 Hom 判别；
4. Banach、Fréchet、核 Fréchet 空间与 liquid 的边界；
5. 连续线性算子、闭值域、Fredholm 复形的 liquid 类型检查；
6. Dolbeault 复形、分布、核函数的 liquid realization；
7. liquid 与 analytic tensor/derived Hom 的相容性。

当前对应位置：

- 第二卷第五章；
- 第二卷附录 J、P；
- 附录 S 给出 liquid 主定理包；
- 附录 Z 把拓扑向量空间凝聚化、liquid realization 输入和 Dolbeault cohomology 比较拆成证明模块；
- 第三卷第三至五章使用这些对象；
- 第四卷第六章作为函数分析例子库。

尚未书内完全证明的核心输入：

- \(p\)-liquid 测度理论；
- \((\mathbb R,\mathcal M_{<p})\) 满足 analytic ring 条件；
- liquid realization 与经典连续线性算子的相容性；
- Dolbeault/Fréchet 对象进入 liquid 范畴的完整构造。

## 3. 是否需要拆卷

当前不必立即拆卷。四卷结构可以保持：

1. 卷一：condensed 基础；
2. 卷二：solid/analytic/liquid 主线；
3. 卷三：复几何应用；
4. 卷四：计算、形式化和同伦接口。

若后续继续扩写到出版级篇幅，最自然的拆分是五卷：

1. 卷一：Condensed 基础；
2. 卷二：Solid 理论；
3. 卷三：Analytic 与 Liquid 理论；
4. 卷四：复几何与相干对偶；
5. 卷五：计算、形式化、pro-etale、pyknotic 和谱值方向。

但在当前仓库中，先不移动文件。应先把第二卷扩成真正的主线卷，再考虑是否拆分。

## 4. 完成标准修订

在本修订下，“基本完本”只能指输入定理型闭合：

1. condensed 层的基础构造尽量书内证明；
2. solid/analytic/liquid 的核心结构定理若不书内证明，必须登记为输入定理；
3. 每个输入定理必须说明精确用途、书内已证部分、外部部分和依赖位置；
4. 接受输入定理之后，所有形式推论、类型检查、例子和反例必须书内闭合；
5. 不得把 solid/analytic/liquid 降格为应用附录或可选背景。
6. 第二卷应有统一闭包定理说明 solid、analytic、liquid 三者的层级关系；当前由附录 T 承担。
7. 第二卷应有出版级审查矩阵检查定义、输入、证明、边界、接口和练习答案状态；当前由附录 U 承担。

因此，当前状态应表述为：

> 四卷已经达到“condensed/solid/analytic/liquid 主线输入定理型闭合草稿”。凝聚基础证明闭合度较高；solid、analytic 和 liquid 主线已经纳入教材结构，并由第二卷附录 Q-T 收束为主定理包，附录 U 给出出版级闭包审查，附录 V-Z 把核心输入进一步拆成证明模块，附录 AA 与第三卷附录 AR 把 Scholze/Clausen-Scholze 定理列为核心图谱；其深层测度、张量、rational acyclicity、realization 和复几何建模定理仍以 Scholze/Clausen-Scholze 输入定理使用。

## 5. 后续写作优先级

若继续推进，优先级不应再放在新增复几何应用，而应先补第二卷主线：

1. 将附录 Q 中 Q.4-Q.6 展开为 solidification 的完整证明链或逐条文献证明；
2. 将附录 R 中 R.4-R.13 展开为 analytic ring、analyticization、rational localization 与 Čech descent 的完整证明链或逐条文献证明；
3. 将附录 S 中 S.1-S.3 展开为 \(p\)-liquid 测度对象、liquid Hom 判别和 realization 的完整证明链或逐条文献证明；
4. 给附录 Q-AA 和第三卷附录 AR 的练习补逐行教师手册；
5. 为每个主线定理增加非平凡计算例子和删除假设的反例。
