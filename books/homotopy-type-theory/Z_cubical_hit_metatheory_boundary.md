# 附录 Z：Cubical 与 HIT 元理论边界

本附录精确化第十六章的元理论声明。它不把模型论证明改写成对象语言中的项，而是说明本书何时使用公理化 HoTT 规则，何时引用 cubical type theory 的计算性解释，何时只登记为外部元定理。

## Z.1 三层语言

**定义 Z.1（对象语言）.** 对象语言是本书第 1-11 章使用的依赖类型论：判断、语境、类型形成、项构造、恒等类型、归纳类型、函数外延性、单值性和指定 HIT 规则。对象语言中的证明是某个类型的项。

**定义 Z.2（元语言）.** 元语言是讨论对象语言语法、模型、规范化、canonicity 和一致性的数学语言。元语言中的定理通常不是对象语言中的项。

**定义 Z.3（对象语言扩展）.** 对象语言扩展指在基础 HoTT 之外加入 cubical、directed、simplicial、cohesive 或 two-level 规则的系统。扩展规则必须说明其形成、构造、消去、计算和模型假设。

**原则 Z.4（不可混用原则）.** 若一个结论是元语言定理，例如 canonicity，则不能在对象语言中直接把它当作构造子或消去规则使用。若一个结论依赖某个对象语言扩展，也不能不经翻译地当作公理化 HoTT 中的定理。

## Z.2 公理化 HoTT 口径

**口径 Z.5（公理化单值性）.** 在公理化 HoTT 阅读下，单值性以常量或公理形式加入：
$$
\mathsf{ua}_{A,B}:(A\simeq B)\to(A=B)
$$
并满足相应的路径计算原则。此时 $\mathsf{ua}$ 不必具有 judgmental computation rule；它的计算通常是 propositional。

**口径 Z.6（公理化 HIT）.** 在公理化 HoTT 阅读下，圆、截断、集合商、悬挂、pushout 等 HIT 由附录 L 的形成、构造、递归/消去和计算规则给出。不同 HIT 的路径构造子计算可分为：

1.  point constructor 的 judgmental computation；
2.  path constructor 的 propositional computation；
3.  在 cubical 口径中可加强为更计算性的规则。

**使用边界 Z.7。** 第十一章和附录 N、V 的圆 encode-decode 证明只需要附录 L 给出的规则；若采用 path constructor 的 propositional computation，则需在文本证明中显式给出对应计算路径，例如附录 N.8 所说的基点计算路径。

## Z.3 Cubical 口径

**输入 Z.8（区间与路径）.** Cubical type theory 引入区间对象 $\mathbb I$、端点 $0,1:\mathbb I$、面条件和沿区间的路径。路径可理解为函数
$$
\mathbb I\to A
$$
并带有端点约束。

**输入 Z.9（composition / filling）.** Cubical 系统包含 composition 或 filling 操作，用于从部分盒子数据构造填充。该结构是 cubical 路径代数、Kan 操作和许多 HIT 计算规则的核心。

**输入 Z.10（Glue 与计算单值性）.** Glue 类型提供单值性的计算性解释：等价可以被编码为 universe 中的路径，并且沿该路径的 transport 计算为等价的底层函数，至少满足 cubical 系统给定的 judgmental/propositional 计算规则。

**定理 Z.11（cubical 单值性，元理论输入）.** 在支持 Glue 的 cubical type theory 中，单值性可由类型形成与计算规则构造，而不必作为外部公理加入。

**证明状态。** 这是 cubical type theory 的元理论结果。cubical 对象语言中可使用相应的 `ua`、`Glue` 和 transport 计算规则；公理化 HoTT 中只能把它登记为单值性的模型或解释。

## Z.4 Canonicity 与 normalization

**定义 Z.12（自然数 canonicity）.** 一个类型论满足自然数 canonicity，若任意闭项
$$
\vdash t:\mathbb N
$$
都计算到某个标准 numeral
$$
\mathsf{succ}^k(0).
$$

**定义 Z.13（normalization）.** Normalization 断言项可按给定归约关系化为正规形。Normalization 通常强于 canonicity，但具体关系取决于系统。

**定理 Z.14（canonicity 是元定理）.** Canonicity、normalization 和 type checking decidability 是关于类型论语法与归约系统的元语言定理，不是本书对象语言中的一般构造原则。

**使用边界。** 第十六章引用 canonicity 时，只说明某类 cubical 系统具有良好计算行为；它不允许在前文证明中把任意闭自然数项直接替换为 numeral，除非当前章节明确采用对应归约关系。

## Z.5 HIT 语义与计算规则

**输入 Z.15（HIT 语义的一般任务）.** 对一个 HIT 规范，元理论需给出：

1.  语法形成规则；
2.  点构造子、路径构造子和高阶路径构造子；
3.  依赖消去原则；
4.  递归原则；
5.  构造子计算规则；
6.  与 universe、截断和其他类型形成子的相容性；
7.  若要求计算性，还需给出归约规则和 canonicity/normalization 相容性。

**命题 Z.16（本书 HIT 使用的有限性）.** 本书基础证明只使用附录 L 登记的有限 HIT 清单：命题截断、一般截断、集合商、圆、悬挂和 pushout。

**证明.** 逐章检查：第八章使用截断和集合商；第十章使用圆、悬挂和 pushout；第十一章使用圆和整数对象；第十二章的 EM 型和谱只在附录 Y 中作为高级输入出现，不回流为前面基础章节的证明前提。$\square$

**原则 Z.17（HIT 引用纪律）.** 后续若新增 HIT，必须同步给出：

1.  形成、构造、消去和计算规则；
2.  其计算规则是 judgmental 还是 propositional；
3.  是否影响 canonicity 或需要额外元理论假设；
4.  是否回流到前文基础证明。

## Z.6 模型比较

**事实 Z.18（simplicial 模型）.** Simplicial set 模型给出单值性与同伦语义的一致性背景，但不直接提供可执行的 univalence 计算规则。

**事实 Z.19（cubical 模型族）.** Cubical 模型有多种变体，包括 de Morgan、cartesian 和其他 presheaf/cubical categories。它们支持的连接、退化、composition、Glue 和 universe 结构可能不同。

**事实 Z.20（strict universes 与 coherence）.** Universe hierarchy、strictification、local universes 和 Grothendieck topos 语义处理的是模型层面的 coherence 问题。引用这些结果时，必须说明它们服务于一致性、语义解释还是具体实现。

## Z.7 与本书章节的接口

本附录对全书的约束如下：

1.  第 0-8 章的基础类型论不依赖 cubical canonicity。
2.  第 9-11 章使用 HIT，但只使用附录 L 的规则。
3.  第 12 章的 EM 上同调使用附录 Y 的高级输入。
4.  第 13-14 章的单值范畴论使用对象语言中的单值性；Rezk 完备化泛性质的证明架构见附录 AA，剩余为文本层 transport 相容细节。
5.  第 15-16 章可以讨论模型论和对象语言扩展差异，但不得把一个系统的规则无翻译地搬到另一个系统。

**当前状态 Z.21。** 本附录关闭的是“边界不清”缺口，而不是证明全部 cubical 元理论。完整 cubical canonicity、normalization、HIT 语义和模型比较仍应引用原论文和模型论文献。
