# 第二十二章：Betti、etale、real etale 与 analytic realization

## 本章目标

本章讨论 realization functors。Realization 把 motivic objects 送到拓扑、pro-etale、real etale 或 analytic contexts 中，是比较定理和计算的重要工具。本章强调 realization 的构造必须同时检查 descent、`\mathbb A^1`-invariance、stabilization 和六操作相容。

## 依赖前置知识

需要 Betti topology、complex/real points、etale homotopy type、pro-spaces、real etale topology、analytic spaces、symmetric monoidal functors、six operations 和 stable homotopy theory。

## 22.1 Realization 的泛性质

**定义 22.1.** 一个 realization functor 是从 motivic homotopy category 到另一同伦论环境的 functor，例如

$$
R:\mathbf{SH}(S)\to\mathcal C,
$$

其中 `\mathcal C` 可以是 spectra、pro-spectra、derived categories 或 analytic motivic categories。

**命题 22.2.** 若 `R` 从 `\mathbf H(S)` 构造，则它必须反演 `X\times\mathbb A^1\to X`。

**证明.** `\mathbf H(S)` 是 Nisnevich sheaves 关于这些投影的局部化。由第二章命题 2.11，任何从 `\mathbf H(S)` 因子化的 functor，在局部化前都必须把这些投影送为等价。`\square`

**命题 22.3.** 若 `R` 从 `\mathbf{SH}(S)` 构造，则还必须把 Tate sphere `T` 的像变为可逆对象。

**证明.** `\mathbf{SH}(S)` 是 `T`-稳定化。由稳定化泛性质，从 `\mathbf H_*(S)` 到 stable target 的 functor 要通过 `\mathbf{SH}(S)` 因子化，必须使 `T` 的像可逆。`\square`

## 22.2 Betti realization

**外部输入定理 22.4.** 对 `S=\operatorname{Spec}\mathbb C` 或合适复基，复点 functor 诱导 Betti realization

$$
Re_B:\mathbf{SH}(S)\to\mathbf{Sp}
$$

并与对称幺半结构和若干六操作相容。

**依赖源.** Morel-Voevodsky 基础、Ayoub 的 Betti realization 和六操作相容结果、后续 infinity-categorical formulations。

**命题 22.5.** 复 Betti realization 反演 `\mathbb A^1`-投影的原因是 `\mathbb C` 的仿射直线拓扑上可缩。

**证明.** 对光滑复概形 `X`，复点满足

$$
(X\times\mathbb A^1)(\mathbb C)\simeq X(\mathbb C)\times\mathbb C.
$$

拓扑空间 `\mathbb C` 可缩，因此投影到 `X(\mathbb C)` 是弱同伦等价。故复点 functor 把 `\mathbb A^1`-投影送为等价。`\square`

## 22.3 Etale realization

**外部输入定理 22.6.** Etale homotopy type 构造诱导从 motivic homotopy theory 到 pro-spaces 或 pro-spectra 的 realization functor；适当完成或反演 residue characteristics 后与 `\mathbb A^1`-localization 相容。

**依赖源.** Artin-Mazur、Friedlander、Isaksen 等 etale realization 文献。

**注 22.7.** Etale realization 的 target 通常不是 ordinary spaces，而是 pro-objects 或 profinite/pro-`l` completions。忽略 pro-结构会丢失 Galois 和 arithmetic 信息。

## 22.4 Real etale 与实 realization

**外部输入定理 22.8.** 在实闭或有序域相关语境中，real realization 和 real etale motivic homotopy 与 classical equivariant/topological information 有比较定理。

**依赖源.** Bachmann 的 real etale stable homotopy、Morel 和后续实 realization 文献。

**命题 22.9.** Real realization 不等同于 complex Betti realization 的限制。

**证明.** 实点 functor 记录 `X(\mathbb R)` 的拓扑，而复 Betti realization 记录 `X(\mathbb C)` 的拓扑及可带有复共轭作用。两者目标和保留的信息不同；例如无实点的实概形仍可能有非空复点。因此不能把 real realization 当作 complex realization 的简单限制。`\square`

## 22.5 Analytic stacks

**研究边界 22.10.** 2025-2026 年的 pullback formalism 和 complex analytic stacks 工作声称构造 analytic motivic homotopy theory、localization theorem 和与六操作相容的 analytification/Betti realization。

**依赖源.** Roy Magen 2025/2026 预印本。按 `FRONTIER_SOURCE_AUDIT_2026_07_08.md`，本书暂列研究边界。

## 22.6 Compatibility with six operations

**定义 22.11.** Realization functor `R:\mathbf{SH}(S)\to\mathcal C` 称为与六操作相容，若对相关态射 `f:X\to Y` 有自然等价

$$
R_Xf^*\simeq f^*R_Y,\qquad
R_Yf_*\simeq f_*R_X,\qquad
R_Yf_!\simeq f_!R_X,
$$

并且与 `f^!`、张量、internal Hom、base change 和 projection formula 相容。

**命题 22.12.** 与 `f^*` 相容不自动推出与 `f_!` 相容。

**证明.** `f_!` 涉及 compact support、proper compactification 或 exceptional pushforward。一个 functor 可以保持 pullback squares，却不保持 compact support 条件或相应 mate transformations。与 `f_!` 相容需要额外的 proper/open gluing 和 base-change 控制。`\square`

**命题 22.13.** 若 realization 与 localization recollement 相容，则它把 motivic localization cofiber sequence 送到目标理论的 localization cofiber sequence。

**证明.** Motivic localization 给出

$$
j_!j^*E\to E\to i_*i^*E.
$$

对 `R` 作用，若 `R` 与 `j_!,j^*,i_*,i^*` 相容且 exact，则得到

$$
j_!j^*R(E)\to R(E)\to i_*i^*R(E),
$$

即目标理论中的 localization cofiber sequence。`\square`

## 22.7 Conservativity

**命题 22.14.** Realization functor 的存在不推出其保守。

**证明.** Functor `R:\mathbf{SH}(S)\to\mathcal C` 保守要求 `R(E)\simeq0` 蕴含 `E\simeq0`。构造 realization 只需满足 descent、`\mathbb A^1`-invariance、stabilization 和相干；这些条件不排除非零对象落入 kernel。因此保守性是额外定理。`\square`

**例子 22.15.** 复 Betti realization 通常会忘记 arithmetic Tate twist 中的部分信息。即使它在某些 cellular 或完成子范畴上有检测力，也不能在整个 `\mathbf{SH}(S)` 上默认保守。

## 22.8 本章小结

Realization 是 motivic theory 与 topology/arithmetic/analysis 的桥梁。Betti、etale、real etale 和 analytic realization 的 target、完成、保守性和六操作相容性各不相同。严格使用时必须写明 target category、局部化、系数和相容结构。

## 练习

**练习 22.1.** 用局部化泛性质证明 realization 必须反演 `\mathbb A^1`。

**练习 22.2.** 说明 Betti realization 中 `\mathbb C` 可缩的作用。

**练习 22.3.** 为什么 etale realization 通常取值于 pro-objects？

**练习 22.4.** 比较 real realization 与 complex Betti realization。

**练习 22.5.** 解释 realization 保守性为何是额外性质。

**练习 22.6.** 写出 realization 与 `f_!` 相容所需的自然等价。

**练习 22.7.** 证明命题 22.13。
