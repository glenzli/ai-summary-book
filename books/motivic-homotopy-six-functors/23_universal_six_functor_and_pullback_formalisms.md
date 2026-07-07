# 第二十三章：Universal six-functor formalisms 与 pullback formalisms

## 本章目标

本章讨论六操作形式主义的普遍性质。Drew-Gallauer 的 universal six-functor formalism 把 stable `\mathbb A^1`-homotopy theory 描述为 universal coefficient system；2025-2026 的 pullback formalism 工作进一步研究六操作判据及其在 stacks/analytic contexts 中的推广。本章严格区分已作为 P0 的基础输入和研究边界。

## 依赖前置知识

需要 six operations、coefficient systems、presentable categories、pullback formalisms、base change、projection formula、localization、purity、universal properties 和 algebraic stacks。

## 23.1 Universal coefficient system

**定义 23.1.** 一个 coefficient system 是把几何对象 `S` 赋予 stable symmetric monoidal category `\mathcal D(S)`，并对 morphisms 给出 pullback functors 的结构。若还满足六操作公理，则称为 six-functor formalism。

**外部输入定理 23.2（Drew-Gallauer）.** Morel-Voevodsky stable `\mathbb A^1`-homotopy theory 提供 universal coefficient system，并由此产生 Grothendieck 六操作。

**依赖源.** Brad Drew, Martin Gallauer, "The universal six-functor formalism"。

**命题 23.3.** Universal property 的意义是：任意满足相同几何公理的系数理论都应由 `\mathbf{SH}` 通过结构保持 functor 获得。

**证明.** “Universal” 在范畴论中表示初对象或自由对象性质。若 `\mathbf{SH}` 是满足公理的 universal coefficient system，则对任何另一个 coefficient system `\mathcal D`，给出从 `\mathbf{SH}` 到 `\mathcal D` 的结构保持 morphism 是由公理唯一控制的。具体唯一性和 morphism 类型由 Drew-Gallauer 的形式化定义给出。`\square`

## 23.2 Axioms behind universality

**定义 23.4.** 一个 motivic coefficient system 通常要求满足以下公理族：

1. Nisnevich 或相应 topology descent；
2. `\mathbb A^1`-invariance；
3. stability under `T` 或 `\mathbb P^1`;
4. localization for closed-open decompositions；
5. purity for smooth 或 lci morphisms；
6. symmetric monoidal compatibility。

**命题 23.5.** 前三项控制 motivic homotopy 的对象层，后三项控制六操作的几何层。

**证明.** Descent、`\mathbb A^1`-invariance 和 `T`-stability 已足以把 presheaf-level geometry 推到 stable motivic homotopy category。Localization、purity 和 monoidal compatibility 则涉及 exceptional functors、Thom twists 和 tensor structures，是建立六操作的关键。`\square`

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

## 23.5 Free generation and generators

**定义 23.12.** 若 coefficient system `\mathcal D` 由对象 `S` 上的单位 `\mathbb 1_S` 和 smooth pushforwards `p_\sharp\mathbb 1_X` 在 colimits、stabilization 和 six-operation closure 下生成，则称它由 smooth generators 生成。

**命题 23.13.** Universal property 若成立，则任何六操作相容 morphism 由 smooth generators 上的取值控制。

**证明.** 若 smooth generators 在允许操作和 colimits 下生成整个 coefficient system，则保持这些操作和 colimits 的 morphism 在所有对象上的值由生成子上的值决定。六操作相容性保证对生成子施加 `f^*,f_*,f_!,f^!`、张量和 localization 后的对象也被相容地送到目标中。`\square`

**注 23.14.** 这类生成性解释了为什么 `\Sigma_T^\infty X_+` 在 `\mathbf{SH}(S)` 中如此核心。它们不是任意例子，而是 universal coefficient system 的基本几何生成子。

## 23.6 Model-independence

**定义 23.15.** 一个 motivic theory 的模型独立性断言，是指不同构造模型，如模型范畴、stable infinity-category、derivator 或 coefficient system 口径，在保留指定结构后给出等价理论。

**命题 23.16.** Universal property 比单纯范畴等价更强。

**证明.** 单纯范畴等价只比较底层 categories。Universal property 要求比较对象在某个结构化范畴中满足初性或自由性，因而还包含 pullbacks、六操作、monoidal structures、localization 和 purity 的相容。两个范畴可以等价，但该等价不保持六操作相干；因此 universal property 是更强断言。`\square`

**命题 23.17.** 若两个模型都满足同一 universal property，则它们在结构保持意义下唯一等价。

**证明.** 初对象或自由对象若存在，在相应结构化 infinity-category 中唯一到 contractible choice。两个满足同一 universal property 的对象互相存在唯一结构保持 morphism，复合由唯一性等于恒等，故为等价。`\square`

## 23.7 Analytic 与 stacky universal properties

**研究边界 23.18.** Magen 2025/2026 工作还声称对 complex analytic stacks 构造 motivic homotopy 的六操作形式主义，并产生与 Grothendieck 六操作相容的 Betti/analytification maps。

**注 23.19.** 该方向截至 2026-07-08 是前沿预印本边界。本书只把它放在研究边界，不纳入基础定理链。

## 23.8 本章小结

Universal six-functor formalism 解释了为什么 `\mathbf{SH}` 是 motivic 系数理论的核心源头。Pullback formalisms 提供更抽象的判据，但完整六操作还需要 adjoints、gluing、purity 和相干。近期 stacky/analytic universal property 是重要前沿，需要后续 locator 和同行验证。

## 练习

**练习 23.1.** 定义 coefficient system。

**练习 23.2.** 解释 universal property 在本章中的含义。

**练习 23.3.** 为什么 pullback formalism 不是完整六操作？

**练习 23.4.** 举例说明 pullback compatibility 与 six-operation compatibility 的差异。

**练习 23.5.** 说明为何 2025-2026 pullback formalism 结果暂列研究边界。

**练习 23.6.** 解释 smooth generators 如何控制保持 colimits 的 morphism。

**练习 23.7.** 说明 universal property 与具体模型构造的区别。

**练习 23.8.** 列出 motivic coefficient system 的六类公理。

**练习 23.9.** 解释为什么不满足 `\mathbb A^1`-invariance 的理论不能直接由 `\mathbf{SH}` 接收。

**练习 23.10.** 说明 universal property 为什么强于范畴等价。

**练习 23.11.** 证明两个满足同一初性条件的对象唯一等价。
