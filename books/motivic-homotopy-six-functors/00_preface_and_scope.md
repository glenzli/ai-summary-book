# 序章：范围、严格性标准和资料源

## 本章目标

本章固定本书的数学口径：哪些对象称为 motivic spaces，哪些对象称为 stable motivic homotopy categories，哪些六操作结论属于本书内部形式推导，哪些必须作为外部输入定理使用。本书不把动机同伦论写成代数几何与代数拓扑的意象类比，而是写成一套可检查的范畴论和几何形式主义。

## 依赖前置知识

需要基本范畴论、Grothendieck topology、sheaves、simplicial sets 或 spaces、presentable infinity-categories、Bousfield localization、稳定 infinity-categories、基础概形论中的 smooth、etale、proper、open immersion 和 closed immersion。

## 0.1 本书对象

**约定 0.1.** 除非另有说明，本书固定 Grothendieck universes
`\mathbb U\in\mathbb V`。“小”意指 `\mathbb U`-小；presentable 意指
`\mathbb U`-accessible 且具有全部 `\mathbb U`-小余极限，并作为
`\operatorname{Cat}_{\mathbb V}` 的对象讨论。所有概形取
`\mathbb U`-小，几何输入范畴替换为等价的 `\mathbb U`-小骨架。完整大小
约定见附录 A。

**约定 0.2.** 第一至第三章在单个 `\mathbb U`-小有限维 Noetherian 基概形
`S` 上工作，`\operatorname{Sm}_S` 为 smooth finite-type `S`-schemes 的
`\mathbb U`-小骨架。讨论 base change 和六操作时，另固定一个
`\mathbb U`-小有限维 Noetherian 概形 `B`，默认基范畴为有限型
`B`-概形；它对有限纤维积封闭。Exceptional functors 只对 separated
`B`-morphisms 使用。

**定义 0.3.** 本书的非稳定 motivic space 范畴定义为

$$
\mathbf H(S)=L_{\mathbb A^1}\operatorname{Shv}_{Nis}(\operatorname{Sm}_S),
$$

其中 sheaf 指 Cech/Nisnevich sheaf，不默认 hypercomplete；
`L_{\mathbb A^1}` 是关于所有投影

$$
X\times_S\mathbb A^1_S\longrightarrow X,\qquad X\in\operatorname{Sm}_S
$$

的 accessible localization。

**定义 0.4.** 本书的稳定 motivic homotopy infinity-范畴写作
`\mathbf{SH}(S)`，先定义为 `\mathbf H_*(S)` 对 Tate sphere

$$
T=\mathbb A^1/(\mathbb A^1\setminus0)
$$

的 presentable symmetric monoidal object-inversion
`\mathbf H_*(S)[T^{-1}]`。该反演的存在性、`T` 的 3-symmetry 以及与
symmetric `T`-spectra 的比较是外部输入；给定这些输入后，`T` 的两个因子
`S^{1,0}` 与 `\mathbb G_m` 分别可逆，ordinary suspension 因而可逆，所以
稳定性在第三章与附录 C 中书内推出。

**注 0.5.** 原始 Morel-Voevodsky 构造使用 simplicial sheaves 和模型范畴。本书优先采用 infinity-categorical localization，因为它能直接表达泛性质、六操作和 presentability；模型范畴版本在比较章节中处理。

## 0.2 严格性标准

**定义 0.6.** 本书中一个结论称为内部结论，若其证明只使用此前定义、一般范畴论、一般 infinity-范畴论或本书已经证明的命题。

**定义 0.7.** 一个结论称为外部输入定理，若其证明依赖 Morel-Voevodsky、Ayoub、Cisinski-Deglise、Hoyois、Drew-Gallauer 或其他资料源中的深定理，且本书不在当前位置重证。

**约定 0.8.** 外部输入定理必须满足三项记录要求：

1. 正文中标明“外部输入定理”。
2. `SOURCES.md` 中记录资料源和用途。
3. `THEOREM_LEDGER.md` 中记录标签、假设和定位状态。

**命题 0.9.** 若一个命题只依赖某个外部输入定理的形式后果，则正文应把二者分离：先陈述外部输入，再证明形式后果。

**证明.** 若不分离，则读者无法判断证明缺口是在几何存在性、相干性、还是纯范畴论推导中。六操作尤其如此：`f_!`、`f^!` 和 base-change 等价的存在是几何定理；一旦它们存在，mate calculus 和投影公式的若干推论则可由伴随和幺半结构形式推出。分离记录使每个证明步骤都有明确依赖，因此是本书严格性标准的直接后果。`\square`

## 0.3 六操作的地位

**定义 0.10.** 对默认基范畴中的态射 `f:X\to Y`，总有
`f^*:\mathbf{SH}(Y)\to\mathbf{SH}(X)` 及其右伴随
`f_*:\mathbf{SH}(X)\to\mathbf{SH}(Y)`。若 `f` separated，另有
`f_!\dashv f^!`。连同 fiberwise operations，六操作写作

$$
f^*,\quad f_*,\quad f_!,\quad f^!,\quad -\otimes_X-,\quad \underline{\operatorname{Hom}}_X(-,-).
$$

其中 `f^*\dashv f_*` 对所有 `f`，`f_!\dashv f^!` 只在上述 exceptional
态射类中。Base change、projection formula、proper compatibility、
localization、purity 和复合相干各有自己的适用假设，不是六个函子存在后
自动成立的一条总括命题。

**注 0.11.** 在 `\mathbf{SH}(-)` 中，`f^*` 通常先由拉回光滑概形诱导，`f_*` 由伴随函子定理得到；`f_!` 与 `f^!` 的存在和良好性质不能由定义自动推出，是 motivic 六操作理论的核心内容。

**外部输入定理 0.12.** 在约定 0.2 的有限型 `B`-schemes 上，
`S\mapsto\mathbf{SH}(S)` 支持第四章定义的 Grothendieck 六操作形式主义：
exceptional base change 与 `!`-projection formula 对 separated morphisms
成立；ordinary proper base change、proper compatibility 对 proper
morphisms 成立；open-closed localization 对 complementary immersions
成立；smooth purity 对 smooth separated morphisms 成立。一般 ordinary
base change 和一般 closed-immersion purity 不在无条件结论中。

**依赖源.** Hoyois, Theorems 1.1 与 6.18（平凡群情形）提供本书采用的
一套精确 package；Ayoub 是非等变构造源。Drew--Gallauer Theorem 7.14
负责 universal coefficient-system 口径，不单独替代 operation-specific
geometric theorems。第五至第八章逐项拆分。

## 0.4 研究边界的处理

**约定 0.13.** 2025-2026 年关于 pullback formalisms、complex analytic stacks、perfect schemes、stacky six operations 的预印本只在研究边界章节使用。它们可以说明方向，但不替代本书基础章节的外部输入。

**命题 0.14.** 若一个近期结果声称推广既有六操作形式主义，则本书在纳入正文定理前必须核查三类信息：基对象类别、允许态射类、相干结构强度。

**证明.** 六操作形式主义不是六个函子的裸存在，而是函子、伴随、base-change、projection formula、localization、purity 和复合相干的整体结构。基对象类别改变时，pullback 方块、proper/open/closed 分解和 descent 覆盖都会改变；允许态射类改变时，`f_!` 与 `f^!` 的定义域可能改变；相干结构强度改变时，命题能否迭代使用也会改变。因此三类信息缺一不可。`\square`

## 0.5 本章小结

本书采用 `\mathbf H(S)` 和 `\mathbf{SH}(S)` 的 infinity-categorical 口径，
把 sheafification、`\mathbb A^1`-localization 和 symmetric monoidal
`T`-inversion 写成带宇宙的泛性质。六操作在固定有限型 `B`-schemes 上按
态射类声明，再在外部输入之上证明形式后果。Hypercompletion、任意 qcqs
基、代数栈和其他几何对象只在明确声明后使用。

## 练习

**练习 0.1.** 解释为什么需要把 `\operatorname{Sm}_S` 替换为小骨架。

**练习 0.2.** 写出 `\mathbf H(S)` 的定义，并指出其中两个局部化步骤。

**练习 0.3.** 说明为什么 `f_!` 和 `f^!` 的存在不是 `f^*` 的定义形式后果。

**练习 0.4.** 给出一个外部输入定理和一个内部命题的区别。
