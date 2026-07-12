# 第十五章：Framed correspondences 与 motivic infinite loop spaces

## 本章目标

本章介绍 framed correspondences。它们是现代 motivic homotopy theory 中描述 `\mathbb P^1`-infinite loop spaces 的核心工具，可视为 motivic 世界中对应 `E_\infty`-spaces 的几何结构。本章只给出严格口径和主要外部输入，不重建完整 framed correspondence 模型。

## 依赖前置知识

需要 finite syntomic morphisms、cotangent complex、K-theory trivializations、motivic spaces、group completion、infinite loop spaces、`H\mathbb Z`、`MGL` 和 suspension spectra。

## 15.1 Framed correspondences 的口径

**定义 15.1.** 一个 framed correspondence 的几何核心由以下数据组成：

1. 有限 syntomic morphism `Z\to X`；
2. 从 `Z` 到目标 `Y` 的态射；
3. cotangent complex `L_{Z/X}` 在 K-theory 中的平凡化或 framing。

完整定义需要选择具体模型，例如 Voevodsky framed correspondences、Elmanto-Hoyois-Khan-Sosnilo-Yakerson 的 framed motivic spaces，或 later variants。

**注 15.2.** Framing 不是装饰性数据。它把有限 syntomic transfer 与稳定 `\mathbb P^1`-同伦中的 Thom twist 对齐，使 correspondence 能表示 infinite loop 结构。

**定义 15.3.** Motivic space with framed transfers 是一个 motivic space，连同沿 framed correspondences 的 functorial action，并满足相应 descent、`\mathbb A^1`-invariance 和加法/复合相干。

## 15.2 Recognition principle

**外部输入定理 15.4（FR-15.x，Motivic Recognition Principle）.** 设 `k`
为 perfect field。Grouplike framed motivic spaces 与 very effective motivic
spectra 之间有 infinity-categorical equivalence

$$
\mathbf H^{fr}(k)^{gp}\simeq\mathbf{SH}(k)^{veff}.
$$

相应的 `S^1`-稳定化识别 effective framed spectra 与
`\mathbf{SH}(k)^{eff}`。Very effective 子范畴本身不对负 suspension 封闭，
所以第一条等价不能称为 stable-category equivalence；这里的 `gp`、
Nisnevich localization 和 `\mathbb A^1`-localization 都是定理输入的一部分。

**精确来源与边界.** Elmanto--Hoyois--Khan--Sosnilo--Yakerson, *Motivic
infinite loop spaces*, Theorem 1.2.3 and Theorem 3.5.14，
`https://arxiv.org/abs/1711.05248`。P0 版本限于 perfect field；本章不把它
写成任意基上的 recognition theorem。

**命题 15.5.** 定理 15.4 说明 framed transfers 是稳定化信息，而不是普通 presheaf 的附加加性结构。

**证明.** 定理 15.4 的右侧由 suspension spectra 经 very-effective 操作生成，
作为 full subcategory 嵌在 `\mathbf{SH}(k)` 中；左侧则要求 framed action 与
grouplike 条件。`S^1`-稳定化后的第二条等价进一步给出 effective spectra。
因此这些 framed 数据编码了 delooping 所需的谱级信息。这个论证不声称
`\mathbf{SH}(k)^{veff}` 对负 suspension 封闭，也不把它当成三角范畴。
`\square`

## 15.3 Sphere spectrum 的几何模型

**高级外部输入 15.6（P1）.** 在域上并采用论文指定的 stabilization 与
group-completion 口径时，motivic sphere 和 algebraic cobordism 的若干
infinite loop spaces 可由 framed finite-syntomic moduli/Hilbert-scheme
模型表示。

**精确来源与边界.** Bachmann--Elmanto--Hoyois--Khan--Sosnilo--Yakerson,
*On the infinite loop spaces of algebraic cobordism and the motivic sphere*,
Theorems 1.1 and 1.4，`https://arxiv.org/abs/1911.02262`。该几何模型不参与
定理 15.4 的 P0 教学证明。

**注 15.7.** 该定理是现代 motivic homotopy 的强几何化结果。由于模型依赖精细 moduli 构造，本书在基础部分只将其作为外部输入。

## 15.4 与 fundamental classes 的联系

**高级外部输入 15.8（P1）.** 在文献指定的 tangentially framed
correspondence 和 motivic coefficient-system 假设下，framed transfers 可与
motivic fundamental classes 所产生的 finite-syntomic Gysin transfers 比较。

**精确来源与边界.** Elmanto--Hoyois--Khan--Sosnilo--Yakerson, *Framed
transfers and motivic fundamental classes*, Section 3，
`https://arxiv.org/abs/1809.10666`；fundamental-class 端见
Deglise--Jin--Khan Theorems 3.3.2 and 4.2.1，
`https://arxiv.org/abs/1805.05920`。全相容属于 P1，不作为 recognition 的
P0 前提。

**命题 15.9.** 若一个 cohomology theory 同时来自 motivic spectrum `E` 且有 framed transfers，则其 finite syntomic Gysin maps 必须与 `E` 的 fundamental class formalism 相容，才能在 framed recognition 与六操作之间一致。

**证明.** Framed transfers 沿 finite syntomic correspondences 给出映射；fundamental class formalism 也为 finite syntomic morphisms 给出 Gysin 型映射。若二者不相容，同一几何 correspondence 会在 framed 模型和六操作模型中给出不同作用，导致由同一个 spectrum `E` 表示的 cohomology theory 没有 well-defined comparison。定理 15.8 正是排除这种歧义的外部输入。`\square`

## 15.5 与 finite correspondences 的比较

**命题 15.10.** Finite correspondences、framed correspondences 和 norms 是三种不同结构。

**证明.** Finite correspondences 是 cycle-theoretic additive data；framed correspondences 使用 finite syntomic morphisms 和 cotangent-complex trivializations，以识别 stable/infinite loop information；norms 是 finite etale morphisms 上的 multiplicative symmetric monoidal transfers。三者的输入态射、结构相干和目标性质均不同，因此不能互相替代。`\square`

## 15.6 Finite syntomic 的作用

**定义 15.11.** 态射 `f:Z\to X` 称为 finite syntomic，若它 finite、flat、
locally of finite presentation，且 cotangent complex `L_{Z/X}` perfect of
cohomological Tor-amplitude `[-1,0]`；等价地，`f` 是 finite flat local
complete intersection morphism。振幅 `[0,0]` 是 smooth 型条件，不能替代
一般 lci 条件。

**命题 15.12.** Finite etale morphism 是 finite syntomic，且其 cotangent complex 为零。

**证明.** Finite etale 态射 finite、flat 且 locally of finite presentation。
Etale 条件给出 `L_{Z/X}\simeq0`。零复形 perfect，且其 Tor-amplitude 包含于
`[-1,0]`，故满足 lci 条件。因此 finite etale 是 finite syntomic。`\square`

**注 15.13.** Framed correspondences 使用 finite syntomic 而不仅是 finite etale，是因为 motivic stable homotopy 中的 Gysin/Thom 修正自然覆盖 lci 型有限映射。Finite etale 情形 cotangent complex 为零，framing 数据退化；finite syntomic 情形则需要真实记录 `L_{Z/X}` 的平凡化。

**定义 15.14.** 对 finite syntomic `Z\to X`，一个 stable framing 可理解为 `K(Z)` 中虚向量丛类 `L_{Z/X}` 的平凡化：

$$
L_{Z/X}\simeq 0\quad\text{in }K(Z).
$$

不同模型可能使用显式嵌入 `Z\hookrightarrow\mathbb A^n_X` 和函数方程给出不稳定 framing；稳定化后它们通过 K-theory 平凡化比较。

**命题 15.15.** 若 `Z\to X` finite etale，则 stable framing 的选择空间含有 canonical base point。

**证明.** Finite etale 态射的 cotangent complex 为零。零对象在 `K(Z)` 中有标准平凡化 `0\simeq0`。因此 framing 选择空间至少有由该恒等平凡化给出的基点。`\square`

## 15.7 从 framed presheaf 到谱

**定义 15.16.** 设 `F` 是带 framed transfers 的 motivic space。称 `F` grouplike，若其由 framed disjoint union 诱导的 `E_\infty`-monoid 结构在 `\pi_0` 上为群。

**命题 15.17.** Grouplike 条件是 infinite loop recognition 中不可省略的条件。

**证明.** 在拓扑和 motivic 语境中，infinite loop space 的零空间不仅是 `E_\infty`-monoid，而且其 `\pi_0` 必须为群；否则只能对应 connective spectrum 的未群完成输入。Framed recognition 的左侧若缺少 grouplike 条件，就会包含尚未完成 group completion 的对象，不能等价于稳定谱的零空间。`\square`

**定义 15.18.** 对 `X\in\operatorname{Sm}_k`，framed suspension presheaf 的原型是

$$
Y\longmapsto \operatorname{Fr}(Y,X),
$$

其中右侧表示从 `Y` 到 `X` 的 framed correspondences 空间或集合。其 sheafification、`\mathbb A^1`-localization 和 group completion 是 framed recognition 的输入步骤。

**外部输入定理 15.19.** 在定理 15.4 的 perfect-field 假设下，`X` 的
framed correspondence presheaf 经 Nisnevich sheafification、
`\mathbb A^1`-localization 和 group completion 后，对应于
`\Sigma_{\mathbb P^1}^\infty X_+` 在 very effective 范畴中的 infinite
loop object。

**精确来源.** Elmanto--Hoyois--Khan--Sosnilo--Yakerson, Theorem 1.2.3
and Theorem 3.5.14，同定理 15.4 的 URL。这里说的是 infinity-category 中
对象的对应，不是同伦范畴中仅有的群同构。

## 15.8 失败模式与边界

**命题 15.20.** 带 framed transfers 的 presheaf 若不满足 Nisnevich descent，不能直接视为 motivic infinite loop space。

**证明.** Motivic spaces 的定义先要求 Nisnevich descent，再要求 `\mathbb A^1`-locality。Framed action 只给出沿 framed correspondences 的函子性；它不自动保证覆盖粘合。因此必须先进行 sheafification 或证明 descent。`\square`

**命题 15.21.** Framed recognition 不应被读作“所有 motivic spectra 都由一个光滑概形表示”。

**证明.** Recognition 定理把某类 grouplike framed motivic spaces 与由 suspension spectra 在 colimits 下生成的谱对象联系起来。Colimits、cofibers、localizations 和 retracts 产生的谱一般不是单个 `\Sigma^\infty X_+`。因此定理说明生成和识别机制，而不是把每个谱压缩成单个光滑概形。`\square`

## 15.9 本章小结

Framed correspondences 把 finite syntomic 数据及 `L_{Z/X}` 的 K-theory
平凡化连接到 motivic infinite loop structures。P0 recognition 精确限于
perfect field，并由 Theorems 1.2.3 and 3.5.14 定位；Hilbert-scheme 模型和
与 fundamental classes 的全相容均明确留在 P1。Finite syntomic 的正确
cotangent 振幅是 `[-1,0]`。

## 练习

**练习 15.1.** 列出 framed correspondence 的三个核心数据。

**练习 15.2.** 解释 cotangent complex 的 framing 为什么与 Thom twist 相关。

**练习 15.3.** 陈述 motivic infinite loop recognition theorem 的数学含义。

**练习 15.4.** 比较 framed transfers 和 finite correspondences。

**练习 15.5.** 说明 framed transfers 与 fundamental classes 相容的必要性。

**练习 15.6.** 证明 finite etale morphism 是 finite syntomic。

**练习 15.7.** 解释 grouplike 条件在 infinite loop recognition 中的作用。

**练习 15.8.** 写出从 framed correspondence presheaf 到 motivic spectrum 的三个局部化/完成步骤。
