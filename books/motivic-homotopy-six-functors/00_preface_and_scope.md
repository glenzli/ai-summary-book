# 序章：范围、严格性标准和资料源

## 本章目标

本章固定本书的数学口径：哪些对象称为 motivic spaces，哪些对象称为 stable motivic homotopy categories，哪些六操作结论属于本书内部形式推导，哪些必须作为外部输入定理使用。本书不把动机同伦论写成代数几何与代数拓扑的意象类比，而是写成一套可检查的范畴论和几何形式主义。

## 依赖前置知识

需要基本范畴论、Grothendieck topology、sheaves、simplicial sets 或 spaces、presentable infinity-categories、Bousfield localization、稳定 infinity-categories、基础概形论中的 smooth、etale、proper、open immersion 和 closed immersion。

## 0.1 本书对象

**约定 0.1.** 除非另有说明，本书固定 Grothendieck universes `\mathbb U\in\mathbb V`，并在 `\mathbb V` 中讨论 presentable infinity-categories。所有几何小范畴都替换为等价的 `\mathbb U`-小骨架。

**约定 0.2.** 默认基概形 `S` 为有限维 Noetherian 概形。记 `\operatorname{Sm}_S` 为 `S` 上光滑有限型概形的一个小骨架，赋予 Nisnevich topology。

**定义 0.3.** 本书的非稳定 motivic space 范畴定义为

$$
\mathbf H(S)=L_{\mathbb A^1}\operatorname{Shv}_{Nis}(\operatorname{Sm}_S),
$$

其中 `L_{\mathbb A^1}` 是关于所有投影

$$
X\times_S\mathbb A^1_S\longrightarrow X,\qquad X\in\operatorname{Sm}_S
$$

的 accessible localization。

**定义 0.4.** 本书的稳定 motivic homotopy 范畴写作 `\mathbf{SH}(S)`，定义为 `\mathbf H_*(S)` 对 Tate sphere

$$
T=\mathbb A^1/(\mathbb A^1\setminus0)
$$

的稳定化。它是稳定 presentable symmetric monoidal infinity-category。该结论的存在性和与经典模型的比较作为外部输入定理处理。

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

**定义 0.10.** 对基范畴中的态射 `f:X\to Y`，六操作指

$$
f^*,\quad f_*,\quad f_!,\quad f^!,\quad -\otimes_X-,\quad \underline{\operatorname{Hom}}_X(-,-).
$$

其中 `f^*\dashv f_*`，`f_!\dashv f^!`，并要求这些函子满足 base change、projection formula、proper compatibility、localization、purity 和复合相干。

**注 0.11.** 在 `\mathbf{SH}(-)` 中，`f^*` 通常先由拉回光滑概形诱导，`f_*` 由伴随函子定理得到；`f_!` 与 `f^!` 的存在和良好性质不能由定义自动推出，是 motivic 六操作理论的核心内容。

**外部输入定理 0.12.** 在适当基范畴和有限性假设下，`S\mapsto\mathbf{SH}(S)` 支持 Grothendieck 六操作形式主义，并满足 base change、projection formula、localization、purity 等相干性质。

**依赖源.** Ayoub, Cisinski-Deglise, Hoyois, Drew-Gallauer。不同资料源覆盖的基范畴和相干强度不同，后续章节逐项拆分。

## 0.4 研究边界的处理

**约定 0.13.** 2025-2026 年关于 pullback formalisms、complex analytic stacks、perfect schemes、stacky six operations 的预印本只在研究边界章节使用。它们可以说明方向，但不替代本书基础章节的外部输入。

**命题 0.14.** 若一个近期结果声称推广既有六操作形式主义，则本书在纳入正文定理前必须核查三类信息：基对象类别、允许态射类、相干结构强度。

**证明.** 六操作形式主义不是六个函子的裸存在，而是函子、伴随、base-change、projection formula、localization、purity 和复合相干的整体结构。基对象类别改变时，pullback 方块、proper/open/closed 分解和 descent 覆盖都会改变；允许态射类改变时，`f_!` 与 `f^!` 的定义域可能改变；相干结构强度改变时，命题能否迭代使用也会改变。因此三类信息缺一不可。`\square`

## 0.5 本章小结

本书采用 `\mathbf H(S)` 和 `\mathbf{SH}(S)` 的 infinity-categorical 口径，把局部化和稳定化写成泛性质，把六操作存在性作为外部输入，再在其上证明形式后果。全书默认基概形为有限维 Noetherian 概形；其他几何对象只在专门章节中引入。

## 练习

**练习 0.1.** 解释为什么需要把 `\operatorname{Sm}_S` 替换为小骨架。

**练习 0.2.** 写出 `\mathbf H(S)` 的定义，并指出其中两个局部化步骤。

**练习 0.3.** 说明为什么 `f_!` 和 `f^!` 的存在不是 `f^*` 的定义形式后果。

**练习 0.4.** 给出一个外部输入定理和一个内部命题的区别。
