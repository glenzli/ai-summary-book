# 第二十三章：Universal six-functor formalisms 与 pullback formalisms

前几章逐个构造或引用了六操作；一个更结构性的问题是，这整套形式主义是否由少量
几何公理唯一决定。若 `\mathbf{SH}` 在某个结构化范畴中是初对象，那么任意满足相同
下降、`\mathbb A^1`-不变性、稳定化与局部化公理的系数理论，都应从它获得一个
结构保持函子。这比底层范畴等价强；但“结构保持”究竟包括哪些 mate，必须由 morphism
所在的结构化范畴明确指定，不能从“初对象”三个字中额外读出。

本章先固定 coefficient system 和其 morphism 的精确类型，再陈述
Drew--Gallauer 的 universal property。随后把只组织 `f^*` 的 pullback formalism
与完整六操作区分开：右伴随、exceptional 伴随、投影公式和 gluing 都是额外结构。
Smooth generators 说明一个结构保持函子为何可由几何生成子控制；stacky 与 analytic
推广则保留在已核查版本所允许的研究边界中。

## 23.1 Universal coefficient system

**定义 23.1（Drew--Gallauer 口径）.** 固定基概形 `B`，令
`\operatorname{Sch}_B` 为有限型 `B`-概形范畴。记
`\operatorname{CoSy}^{c}_B` 为 Drew--Gallauer Definition 7.7 的
cocomplete coefficient systems 所成的 infinity-范畴：其对象取值于 cocomplete
stable infinity-范畴并带该定义规定的 pullback、smooth pushforward、幺半、
localization 等结构；其 morphisms 保持小余极限和 coefficient-system 结构。
本章使用这一来源定义，不把较弱的“范畴值预层”也称为
`\operatorname{CoSy}^{c}_B` 的对象。

**外部输入定理 23.2（Drew--Gallauer）.** 在定义 23.1 的口径中，Morel--Voevodsky
稳定 `\mathbb A^1`-同伦理论给出的 coefficient system

$$
\mathbf{SH}\in\operatorname{CoSy}^{c}_B
$$

是初对象。

**精确来源与边界.** Brad Drew, Martin Gallauer, *The universal six-functor
formalism*, Theorem 7.14；Proposition 7.13 与 Theorem 7.3 给出其前置比较步骤，
`https://arxiv.org/abs/2009.13610`。原文特别指出，一般 coefficient-system
morphism 不自动与全部六操作交换。因此本定理不能单独替代第五、八章的几何
base-change、projection formula 或 purity 定理，也不能推出定义 23.10 的强相容性。

**命题 23.3.** 对每个
`\mathcal D\in\operatorname{CoSy}^{c}_B`，映射空间

$$
\operatorname{Map}_{\operatorname{CoSy}^{c}_B}(\mathbf{SH},\mathcal D)
$$

可缩。

**证明.** Infinity-范畴中对象 `I` 为初对象的定义，正是对任意对象 `D`，映射空间
`\operatorname{Map}(I,D)` 可缩。将定理 23.2 代入即得。可缩性给出的是
`\operatorname{CoSy}^{c}_B` 所记录结构下的唯一性；由来源边界，它不额外声称该
morphism 与定义中未要求的全部六操作相容。`\square`

## 23.2 初性所处的结构化范畴

**定义 23.4.** 本章把下列条件分成三个相互作用而非彼此割裂的层次：

1. **局部性：** Nisnevich descent 与闭开 localization；
2. **motivic 不变性：** `\mathbb A^1`-invariance 与 `T`-stability；
3. **系数结构：** symmetric monoidal structure、smooth pushforward 及来源定义中要求的
   base-change 和 projection-formula 数据。

只有当对象、1-morphisms 与 2-morphisms 都按 Drew--Gallauer Definition 7.7 组织时，
这些条件才定义 `\operatorname{CoSy}^{c}_B`。把同一组纤维范畴放进一个较弱的预层
范畴，会得到不同的初性问题。

**命题 23.5.** 定理 23.2 的初性只保证其 morphism 保持
`\operatorname{CoSy}^{c}_B` **已经编码**的结构；它既不删除这些结构，也不自动增加
定义中未编码的 operation compatibility。

**证明.** 初对象由其所在 infinity-category 的映射空间定义。更换 ambient category
会同时更换允许的 morphisms，从而改变初性命题。定理 23.2 中的可缩映射空间因此
控制 Definition 7.7 所规定的数据；对于未进入 morphism 定义的 mate transformation，
该映射空间没有可供约束的分量。`\square`

**命题 23.6.** 若一个系数理论不满足 `\mathbb A^1`-invariance，则它不能由 `\mathbf{SH}` 的 universal motivic property 直接接收。

**证明.** `\mathbf{SH}` 是从 `\mathbb A^1`-局部化后再稳定化得到的。由局部化泛性质，任何从 `\mathbf{SH}` 因子化的理论必须把 `X\times\mathbb A^1\to X` 送为等价。不满足该条件的理论不能直接由 `\mathbf{SH}` 给出，除非先做 `\mathbb A^1`-局部化。`\square`

## 23.3 Pullback formalisms

**定义 23.7.** Pullback formalism 是以 pullback functors `f^*` 为基础组织几何系数理论的抽象框架；六操作可视为在 pullback formalism 上添加 adjoints、base-change、projection formula 和 gluing 的结构。

**研究边界 23.8.** Magen 2025 的 geometric criteria 在 pullback formalisms 中给出六操作形式主义和其 morphisms 的判据，并声称加强 stacky stable motivic homotopy 的 universal property。

**依赖源.** Roy Magen, "Geometric Criteria for 6-Functor Formalisms in the Setting of Pullback Formalisms", arXiv:2511.09371。

**命题 23.9.** 若一个 pullback formalism 不含 `f_!` 或 `f^!`，则它还不是六操作形式主义。

**证明.** 六操作形式主义要求 `f^*, f_*, f_!, f^!, \otimes, \underline{\operatorname{Hom}}` 及相干公理。Pullback formalism 只组织 `f^*` 方向及其相干，缺少 exceptional operations。因此它只是六操作的输入层，不是完整六操作。`\square`

## 23.4 Morphisms compatible with six operations

**定义 23.10.** 两个六操作形式主义之间的 morphism 称为与六操作相容，若它不仅与 pullback `f^*` 相容，还与 `f_*`、`f_!`、`f^!`、张量、internal Hom、base-change 和 projection formula 的相干结构相容。

**命题 23.11.** 与 pullback 相容不自动推出与六操作相容。

**证明.** `f_*`、`f_!` 和 `f^!` 由伴随和几何构造给出。一个 functor 可以与 `f^*` 交换，但不保持右伴随、左伴随或 mate transformations。例如保持 left adjoints 需要连续性/余连续性条件，保持 `f_!` 还需要 proper/open compactification 相容。因此 pullback compatibility 只是必要条件，不是充分条件。`\square`

## 23.5 纤维中的光滑生成子

**定义 23.12.** 固定基 `S`。若 presentable stable category `\mathcal D(S)` 的最小
localizing subcategory，包含所有 smooth 态射 `p:X\to S` 所给的
`p_\sharp\mathbb 1_X` 及理论中指定的可逆 Thom twists，并且等于整个
`\mathcal D(S)`，则称该纤维由 smooth generators 生成。这是逐纤维陈述；它不把
尚未构造的 `f_!` 或 `f^!` 偷渡进“生成”的定义。

**命题 23.13.** 设 `F,G:\mathcal D(S)\to\mathcal E(S)` 保持小余极限且正合，并
给定自然变换 `\eta:F\to G`。若 `\eta` 在定义 23.12 的所有生成子上为等价，则
`\eta` 在 `\mathcal D(S)` 的每个对象上为等价。

**证明.** 令 `\mathcal L` 为使 `\eta_E` 成为等价的对象 `E` 所成的全子范畴。因为
`F`、`G` 正合并保持小余极限，`\mathcal L` 对 suspension、cofiber、任意余积和 retract
封闭，故是 localizing subcategory。它包含全部 smooth generators，定义 23.12 遂给出
`\mathcal L=\mathcal D(S)`。`\square`

**注 23.14.** 在 `\mathbf{SH}(S)` 中，这些生成子表现为
`\Sigma_T^\infty X_+` 及 Thom twists。命题 23.13 说明如何**检测两个已给函子之间的
自然变换**；它本身既不构造函子，也不证明函子与六操作相容。

## 23.6 Model-independence

**定义 23.15.** 一个 motivic theory 的模型独立性断言，是指不同构造模型，如模型范畴、stable infinity-category、derivator 或 coefficient system 口径，在保留指定结构后给出等价理论。

**命题 23.16.** 在固定的结构化 infinity-category 中，universal property 比“底层范畴
存在某个等价”包含更多信息；多出的信息恰好是该 ambient category 所编码的结构。

**证明.** 底层范畴等价不要求保持 pullback、monoidal product 或 localization。
结构化初性则要求映射属于该结构化范畴，并要求相应映射空间可缩。不过由命题 23.5，
不能把 ambient category 未编码的 `f_!`、`f^!` 或 purity compatibility 也列入结论。
所以“更强”是相对于明确结构而言，而不是对全部可能六操作的无界承诺。`\square`

**命题 23.17.** 若两个模型都满足同一 universal property，则它们在结构保持意义下唯一等价。

**证明.** 初对象或自由对象若存在，在相应结构化 infinity-category 中唯一到 contractible choice。两个满足同一 universal property 的对象互相存在唯一结构保持 morphism，复合由唯一性等于恒等，故为等价。`\square`

## 23.7 Analytic 与 stacky universal properties

**研究边界 23.18.** Magen 2025/2026 工作还声称对 complex analytic stacks 构造 motivic homotopy 的六操作形式主义，并产生与 Grothendieck 六操作相容的 Betti/analytification maps。

**注 23.19.** 该方向截至 2026-07-08 是前沿预印本边界。本书只把它放在研究边界，不纳入基础定理链。

## 23.8 初性为何强于模型等价

Universal six-functor formalism 解释了为什么 `\mathbf{SH}` 是 motivic 系数理论的
核心源头。Pullback formalisms 提供更抽象的判据，但完整六操作还需要 adjoints、
gluing、purity 和相干。Stacky/analytic universal property 只按已核查的预印本版本
使用，不承担基础六操作的证明责任。

## 练习

**练习 23.1.** 定义 coefficient system。

**练习 23.2.** 解释 universal property 在本章中的含义。

**练习 23.3.** 为什么 pullback formalism 不是完整六操作？

**练习 23.4.** 举例说明 pullback compatibility 与 six-operation compatibility 的差异。

**练习 23.5.** 说明为何 2025-2026 pullback formalism 结果暂列研究边界。

**练习 23.6.** 解释 smooth generators 如何控制保持 colimits 的 morphism。

**练习 23.7.** 说明 universal property 与具体模型构造的区别。

**练习 23.8.** 说明定义 23.4 的三个层次，并解释 ambient category 为什么影响初性。

**练习 23.9.** 解释为什么不满足 `\mathbb A^1`-invariance 的理论不能直接由 `\mathbf{SH}` 接收。

**练习 23.10.** 说明 universal property 为什么强于范畴等价。

**练习 23.11.** 证明两个满足同一初性条件的对象唯一等价。
