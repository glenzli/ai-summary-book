# 凝聚数学讲义

作者：Dr. Stochastic Parrot

状态：四卷主线输入定理型收口稿

总题：从凝聚基础到 solid/analytic/liquid、复几何与计算例子

这是一套中文凝聚数学严格教材草稿。全书采用四卷结构：第一卷建立
condensed 对象和基础同调代数，第二卷进入 solid、analytic 与 liquid
结构，第三卷把这些语言用于复几何和相干对偶，第四卷收集形式化义务、
计算模板和边界例子。

本书的目标不是替代 Scholze、Clausen-Scholze 或经典复几何文献，也不把
Gleason、Nöbeling、Cartan、Grauert、GAGA、GRR 等深层定理伪装成书内
已证结论。它采用输入定理型教材标准：基础构造和接受输入后的形式推论
在书内证明；外部深定理明确登记其使用形式、依赖位置和引用来源。这样读者
可以连续阅读主线，同时知道每一步究竟是书内证明、输入定理还是后续出版维护。

## 阅读路线

建议第一次阅读按卷一至卷三顺序进行，遇到计算、形式化或反例时再查卷四。
已经熟悉 sheaf 和同调代数的读者也不宜完全跳过卷一，因为本书的 universe、
测试站点、极不连通空间和凝聚阿贝尔群约定都在卷一固定。

1. [卷一：凝聚数学基础](volume-1/)  
   站点、sheaf、紧 Hausdorff/profinite 空间、凝聚集合、凝聚阿贝尔群、
   基本同调代数，以及 solid 与 analytic 的入口。

2. [卷二：Solid、Analytic 与 Liquid 结构](volume-2/)  
   solid 派生范畴、solid 环与模、解析环、解析化、liquid 向量空间、
   Huber pair、$f_!$ 和相干对偶入口。solid、analytic 和 liquid 在本书中是主线，
   不是应用附录。

3. [卷三：复几何与相干对偶](volume-3/)  
   复解析空间、相干层、Dolbeault、有限性、Serre 对偶、GAGA、
   Riemann-Roch 和六函子展望。经典深定理按输入定理使用，书内证明其形式后果。

4. [卷四：形式化、计算与例子](volume-4/)  
   形式化路线、站点与 sheaf 计算、Ext/Tor 模板、solid/analytic/liquid 例子、
   pro-etale 背景和当前方向。

## 使用方式

- 若目标是学习主线概念，按卷一到卷三阅读，并把卷四当作工具卷。
- 若目标是核对证明身份，使用 [INPUT_THEOREM_REGISTER.md](INPUT_THEOREM_REGISTER.md)
  和 [THEOREM_INDEX.md](THEOREM_INDEX.md) 区分书内定理、输入定理和形式推论。
- 若目标是追踪符号、术语和答案，查 [GLOSSARY.md](GLOSSARY.md) 与
  [SOLUTIONS.md](SOLUTIONS.md)。
- 若目标是维护或出版校对，再查 [FINAL_PUBLICATION_STATUS.md](FINAL_PUBLICATION_STATUS.md)、
  [TEXTBOOK_REVIEW.md](TEXTBOOK_REVIEW.md)、[REFERENCE_LOCATOR_LEDGER.md](REFERENCE_LOCATOR_LEDGER.md)
  和 [PUBLICATION_PROOFREADING_AUDIT.md](PUBLICATION_PROOFREADING_AUDIT.md)。

## 数学边界

当前版本已经完成凝聚基础、站点比较、sheaf 与同调代数基础的书内证明链；
solid、analytic、liquid 的定义、输入定理、证明模块、类型检查和主定理包；
以及复几何应用中 Dolbeault、有限性、Serre duality、GAGA、HRR/GRR 的输入定理拆分、
形式推论、边界例子和计算模型。

这仍不是完全自足证明版教材。深层外部定理以输入定理或证明路线标注；若读者需要
完整重证这些结果，应把本书与原始论文、经典教材或专门预备卷配合使用。核心资料源
记录在各卷的 `SOURCES.md` 中，跨卷依赖见 [DEPENDENCY_GRAPH.md](DEPENDENCY_GRAPH.md)，
主线架构见 [MAINLINE_ARCHITECTURE.md](MAINLINE_ARCHITECTURE.md)。
