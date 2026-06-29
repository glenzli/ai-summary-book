# 附录 U：第二卷出版级闭包审查

## U.0 目标

本附录对第二卷进行出版级闭包审查。它不引入新的数学定理，而是检查 solid、analytic、liquid 三条主线是否满足教材式闭合要求：

1. 定义是否有所在范畴；
2. 输入定理是否编号并登记；
3. 接受输入后的形式推论是否书内证明；
4. 关键误用是否有边界例子；
5. 第三卷使用第二卷时是否有类型接口；
6. 练习是否有答案入口。

附录 V-Z 增加后，本审查覆盖范围从主定理包 Q-T 扩展到核心证明模块 V-Z：审查对象不仅是“是否登记输入”，还包括“输入定理被拆成哪些书内可复核的形式步骤”。

## U.1 审查标准

本卷采用四级状态：

| 状态 | 含义 |
| --- | --- |
| 闭合 | 书内给出定义、定理和证明，不依赖未登记输入 |
| 输入闭合 | 深层定理作为输入登记；接受输入后的推论书内证明 |
| 检查表闭合 | 不证明主定理，但列出使用时必须验证的义务和失败模式 |
| 未闭合 | 正文使用了没有登记或没有证明的结论 |

出版级草稿允许“输入闭合”，但不允许“未闭合”。

## U.2 Solid 主线审查

| 项目 | 位置 | 状态 | 说明 |
| --- | --- | --- | --- |
| solid 局部对象定义 | 附录 M、Q、V | 闭合 | 由 \(K_S\)-正交定义 |
| localizing kernel | 附录 M、Q、V | 输入闭合 | kernel 由 \(K_S\) 生成作为 solidification 输入；一般局部化 kernel 性质书内证明 |
| solid 反射局部化 | D.1、Q.4、V | 输入闭合 | 集合生成局部化形式层已写；与 Scholze solidification 的识别仍为输入 |
| solid kernel 张量理想性 | D.2、Q.5、W | 输入闭合 | 生成元判别书内证明；profinite 测度张量计算仍为输入 |
| solid 张量积 | Q.7-Q.9、W | 输入闭合 | 接受张量理想性后闭合 |
| solid 环与 solid 模 | Q.10-Q.12 | 闭合 | 一般幺半稳定范畴中的形式推论 |
| profinite 测度张量公式 | D.3、Q.6、W | 输入闭合 | 公式本身作为 Scholze 输入；其对张量理想性的作用已书内证明 |
| 生成元检验 | H、Q.13-Q.14 | 输入闭合 | 生成族来自输入，检验形式书内证明 |

**结论 U.1.** Solid 主线达到输入闭合。完全自足版仍需重证 D.1-D.3，尤其是 solid kernel 张量理想性。

## U.3 Analytic 主线审查

| 项目 | 位置 | 状态 | 说明 |
| --- | --- | --- | --- |
| pre-analytic datum | 第三章、附录 R | 闭合 | 测度对象与 Dirac cone 明确 |
| analytic 局部对象 | R.1-R.3、X | 闭合 | 由 \(K_S^{\mathcal M}\)-正交定义 |
| analytic 反射局部化 | D.4、R.4、X | 输入闭合 | 集合生成局部化形式层已写；analytic ring 公理推出正确 localization 作为输入 |
| analytic 张量结构 | D.4、R.5-R.9、X | 输入闭合 | 核张量理想性作为输入，下降书内证明 |
| analyticization 泛性质 | R.7-R.8、X | 输入闭合 | 来自反射局部化伴随 |
| Huber pair analytic ring | D.7、R.11 | 输入闭合 | 构造测度对象不书内重证 |
| rational localization | D.7、R.12、Y | 输入闭合 | 与解析模范畴相容作为输入 |
| rational Čech descent | D.7、R.13-R.15、Y | 输入闭合 | Čech/mapping/gluing 形式后果书内证明；rational acyclicity 作为输入 |
| perfect 性局部检测 | R.16 | 检查表闭合 | 需额外假设 perfect 子范畴满足 descent |

**结论 U.2.** Analytic 主线达到输入闭合。完全自足版仍需重证 analytic ring 条件如何推出反射局部化、张量相容和 rational descent。

## U.4 Liquid 主线审查

| 项目 | 位置 | 状态 | 说明 |
| --- | --- | --- | --- |
| \(p\)-liquid analytic ring | D.5、S.1 | 输入闭合 | analytic ring 条件作为输入 |
| liquid realization | D.6、S.2-S.3、Z | 输入闭合 | 拓扑向量空间凝聚化书内证明；realization 构造与 exactness 范围作为输入 |
| finite-dimensional objects | S.6 | 输入闭合 | 接受 realization 后书内证明 perfect 性 |
| Banach/Fréchet 边界 | J、P、S.5、Z | 闭合 | 明确拓扑向量空间只先凝聚化，不自动 liquid |
| 闭值域 Fréchet cohomology | P、S.7-S.9、Z | 输入闭合 | 闭值域到 Hausdorff 商书内证明，realization exactness 输入 |
| Fredholm perfect cohomology | P、S.10-S.11、Z | 输入闭合 | 有限维性来自 Fredholm 输入 |
| Dolbeault 类型闭合 | S.12-S.13、Z | 输入闭合 | Dolbeault-Fredholm 输入后 liquid 类型书内闭合 |

**结论 U.3.** Liquid 主线达到输入闭合。完全自足版仍需重证 \(p\)-liquid 测度理论、realization 构造和与经典连续线性算子的相容性。

## U.5 三条主线接口审查

| 接口 | 位置 | 状态 | 说明 |
| --- | --- | --- | --- |
| condensed 到 solid | 卷一第十二章、第二卷 Q | 输入闭合 | 通过 Dirac-to-measure cone 局部化 |
| solid 到 analytic | 第二卷 R | 输入闭合 | analytic ring 是更一般的测度局部化结构 |
| analytic 到 liquid | 第二卷 S | 输入闭合 | liquid 是特定 analytic ring 及 realization 理论 |
| 第二卷到第三卷 | 第二卷 T.4 | 闭合 | 第三卷只作为应用卷使用第二卷类型 |
| 第二卷到第四卷 | 第二卷 T.5、第四卷例子 | 检查表闭合 | 计算例子依赖输入定理，不反推主定理 |

**结论 U.4.** 三条主线的接口已经在附录 T 中闭合。后续新增应用章节必须按 T.4 标注所用接口。

## U.6 术语和符号审查

1. \(D_\square(\mathbb Z)\)：固定表示 solid 派生范畴。
2. \(\otimes^{L,\square}\)：只表示 solid 派生张量。
3. \(D(A,\mathcal M)\)：表示 analytic ring 的 analytic 模范畴。
4. \(L_{(A,\mathcal M)}\)：表示 analyticization。
5. \(D_{\mathrm{liq},p}\)：表示固定 \(p\) 后的 liquid analytic 范畴。
6. \(\mathcal L_p\)：表示拓扑向量空间到 liquid 范畴的 realization，不能省略适用范围。

**结论 U.5.** 第二卷符号已足以支撑主线闭包；出版级版本仍应在正文首次出现处增加更多回链。

## U.7 练习与答案审查

| 章节 | 练习状态 | 答案状态 |
| --- | --- | --- |
| 附录 Q | 有 solid localization 与张量下降练习 | 第二卷 SOLUTIONS 第 6 节给出要点 |
| 附录 R | 有 analyticization 与 descent 练习 | 第二卷 SOLUTIONS 第 7 节给出要点 |
| 附录 S | 有 closed range 与 liquid realization 练习 | 第二卷 SOLUTIONS 第 8 节给出要点 |
| 附录 T | 有统一闭包与应用接口练习 | 第二卷 SOLUTIONS 第 9 节给出要点 |
| 附录 V | 有 solidification 反射存在性练习 | 第二卷 SOLUTIONS 第 10 节给出要点 |
| 附录 W | 有 solid kernel 张量理想性练习 | 第二卷 SOLUTIONS 第 11 节给出要点 |
| 附录 X | 有 analytic localization 练习 | 第二卷 SOLUTIONS 第 12 节给出要点 |
| 附录 Y | 有 rational descent 练习 | 第二卷 SOLUTIONS 第 13 节给出要点 |
| 附录 Z | 有 liquid realization 练习 | 第二卷 SOLUTIONS 第 14 节给出要点 |

**结论 U.6.** 当前达到“答案要点”标准。出版级教师手册仍需把每题扩成逐行证明。

## U.8 未闭合项清单

当前没有发现第二卷主线中“正文使用但未登记”的核心定理。附录 V-Z 已把若干原本粗粒度的核心输入拆成证明模块；仍未书内重证的内容均已作为输入定理登记：

1. solid 反射局部化；
2. solid kernel 张量理想性；
3. profinite 测度张量公式；
4. analytic ring localization；
5. analytic kernel 张量理想性；
6. Huber pair rational localization；
7. rational Čech descent；
8. \(p\)-liquid analytic ring；
9. liquid realization；
10. Fréchet Fredholm/Hodge 输入。

## U.9 后续出版级任务

按优先级排序：

1. 为 D.1-D.3 和附录 V-W 增加逐条文献定位和证明概要。
2. 为 D.4、D.7 和附录 X-Y 增加 analytic ring 与 rational descent 的逐条文献定位。
3. 为 D.5-D.6 和附录 Z 增加 liquid realization 的精确适用范围。
4. 把附录 Q-AA 的答案从要点扩成教师手册。
5. 增加每条输入定理的最小反例或失败模式。

## U.10 审查结论

第二卷现在达到：

> solid/analytic/liquid 主线输入定理型闭合。

这意味着它可以作为凝聚数学主线教材草稿使用；附录 V-Z 已经把 Scholze/Clausen-Scholze 核心结构定理拆成教材内部证明模块和外部输入边界。但它还不是这些核心结构定理的完全自足证明版。
