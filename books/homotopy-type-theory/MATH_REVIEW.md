# 数学审查记录

本记录关注教材结论能否按其声明的身份使用。检查对象包括量词与 universe、截断消去、transport、模型/语法分层和外部定理的精确版本；章节是否通过机械叙事检查另见 `PUBLICATION_CLOSURE_AUDIT.md`。

## 当前结论

1. 第 0-17 章已经按连续教材叙事组织，固定目标/前置/主线/小结模板不再是章节结构。
2. 低层判断、路径代数、等价、同伦层级、截断、HIT、圆的基本群、单值一范畴和 Yoneda 构成可追踪的内部链。
3. Rezk 本质像构造与 weak-equivalence 性质在书内给出；对单值目标的限制函子泛性质采用 Ahrens--Kapulkin--Shulman Theorem 8.4。
4. 第十二、十五和十七章把高级同伦论、模型语义与扩展语言写成具有明确假设的教材导读，不把研究路线冒充已证定理。
5. 附录 BO 的 Banach 不动点定理包含 $\|X\|$，统一使用有理开球关系，并在消去 mere completeness 前证明极限总类型是命题。

## 重点风险

**截断代表元。** 命题截断只能消去到命题。Rezk 扩张、Cauchy 极限和本质存在性都必须先展示允许消去的目标层级，或明确交由外部来源定理处理。

**语法与模型。** Simplicial/cubical 模型、canonicity、normalization 和相对一致性是元语言结论，不产生基础 HoTT 对象语言中的新计算规则。

**高级接口。** EM 型、Blakers--Massey、Freudenthal、Hopf fiber sequence、谱模型和谱序列收敛均需精确输入。存在 $E_r$ 页不推出强收敛，经典空间图也不自动成为内部 fiber sequence。

**构造性分析。** 排中律、选择、resizing、locatedness 和强完备性不得默认加入。使用 mere 极限时，不得构造一个未截断的极限点，除非目标的命题性已经证明。

## 审查问题

- 定义是否给出所有自由变量、依赖与 universe 层级？
- 书内证明是否真的构造结论类型的项，而非只列证明路线？
- 截断消去目标是否具有正确 h-level？
- transport 是否沿正确的族，并与选择的代表元或对象路径相容？
- 外部输入是否有精确来源、假设、结论和未采用部分？
- 扩展语言或模型结果是否被误写成基础 HoTT 的 judgmental equality？
- 新符号、来源和依赖是否同步到 `NOTATION.md`、`SOURCES.md` 与附录 K？
