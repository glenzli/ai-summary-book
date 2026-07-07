# 第二十一章：Log schemes、perfect schemes 与 universal homeomorphisms

## 本章目标

本章记录 motivic homotopy theory 的三个扩展方向：log schemes、positive characteristic 的 perfect schemes，以及 universal homeomorphism invariance。这些方向均处于活跃发展中，本章把它们作为研究边界或高级外部输入处理。

## 依赖前置知识

需要 log schemes、fs log structures、perfect schemes、Frobenius、universal homeomorphisms、positive characteristic、six-functor formalisms、localization 和 `\mathbb A^1`-invariance。

## 21.1 Log motivic homotopy

**定义 21.1.** Log motivic homotopy theory 试图把 motivic homotopy 从 schemes 扩展到带 log structure 的几何对象，使边界、退化和 compactification 数据成为同伦论的一部分。

**外部输入定理 21.2.** 对 fs log schemes，可构造 `\mathbb A^1`-local stable motivic homotopy categories；对 schemes with trivial log structure，该构造与 Morel-Voevodsky 原构造相容，并在 strict morphisms 上给出六操作形式主义。

**依赖源.** Doosung Park, "A1-homotopy theory of log schemes"。

**命题 21.3.** Trivial log structure 情形应恢复普通 motivic homotopy theory。

**证明.** 定理 21.2 已包含该相容性作为外部输入。若 log structure 为 trivial，则 log smooth/site/interval 数据退化为 scheme-level 数据；因此构造限制到普通 `\mathbf{SH}`。具体等价依赖定理 21.2 的比较部分。`\square`

## 21.2 Perfect schemes

**定义 21.4.** 在特征 `p>0` 中，perfect scheme 是 Frobenius morphism 为同构的 scheme。Perfectization 把 scheme 沿 Frobenius 迭代极限化。

**外部输入定理 21.5.** 存在 positive characteristic perfect base schemes 上的 perfect motivic homotopy theory，并可通过 coefficient system 公理建立六操作形式主义；其与 universal homeomorphism localization 和 `\mathbf{SH}[1/p]` 有联系。

**依赖源.** Dahlhausen-Hekking-Wolters, "Motivic homotopy theory for perfect schemes"。截至 2026-07-08 作为研究边界处理。

**命题 21.6.** Perfect motivic homotopy theory 不能未经说明替代普通 positive characteristic motivic homotopy theory。

**证明.** Perfectization 反演或忽略了 purely inseparable/Frobenius 方向的信息。普通 motivic homotopy theory 保留原 scheme 的结构；perfect theory 则在 universal homeomorphism 或 `p` 反演语境中工作。二者比较需要定理 21.5 中的 localization statement，不能由定义自动推出。`\square`

## 21.3 Universal homeomorphisms

**定义 21.7.** 态射 `f:X\to Y` 称为 universal homeomorphism，若对任意 base change 后的底层拓扑空间映射仍为 homeomorphism，且 `f` integral、surjective、radicial。

**外部输入定理 21.8.** 在若干 motivic contexts 中，反演指数特征后，`\mathbf{SH}` 对 universal homeomorphisms 不变。

**例子 21.9.** 特征 `p` 中的 absolute Frobenius `F_X:X\to X` 在许多有限性假设下是 universal homeomorphism。若 `X` perfect，则它是同构。

**例子 21.10.** Nilpotent thickening `X_{red}\hookrightarrow X` 常给出 universal homeomorphism。Motivic theory 是否对该 thickening 不变，取决于是否处在允许 universal homeomorphism invariance 的局部化语境。

**命题 21.11.** 若 `f:X\to Y` 是 universal homeomorphism 且已知 `f^*:\mathbf{SH}(Y)[1/p]\to\mathbf{SH}(X)[1/p]` 为等价，则 `Y` 与 `X` 在该 localized motivic theory 中不可区分。

**证明.** 这是范畴等价的直接含义。若 `f^*` 为等价，则它有逆等价，所有 objects、mapping spaces、cohomology theories 和六操作可见信息都通过该等价对应。`\square`

## 21.4 Frobenius 与 perfectization

**定义 21.12.** 若 `X` 是特征 `p` 的 scheme，其 absolute Frobenius 为

$$
F_X:X\to X
$$

在拓扑空间上为恒等，在结构层上为 `p` 次幂映射。

**定义 21.13.** Perfectization `X^{perf}` 可形式地写作 Frobenius 迭代系统的极限

$$
X^{perf}=\varprojlim(\cdots\xrightarrow{F_X}X\xrightarrow{F_X}X).
$$

**命题 21.14.** 若 `X` 已 perfect，则 `X^{perf}\simeq X`。

**证明.** Perfect 表示 Frobenius `F_X` 为同构。由同构组成的逆系统的极限同构于任一项，因此 `X^{perf}\simeq X`。`\square`

**命题 21.15.** Frobenius 在 perfect scheme 上不再产生新的 universal homeomorphism 信息。

**证明.** 在 perfect scheme 上 Frobenius 是同构。同构当然是 universal homeomorphism，但它已经可逆，不再给出需要局部化反演的新态射。`\square`

## 21.5 Log intervals and boundary information

**定义 21.16.** Log motivic homotopy 中的 interval 可能不是普通 `\mathbb A^1`，而是带 log structure 的对象，例如 log affine line 或 compactified log interval。具体选择取决于所用 log motivic theory。

**命题 21.17.** Log structure 可以把开补边界变成对象的一部分，而不是只作为 localization 的闭补处理。

**证明.** 普通 motivic homotopy 中，开嵌入 `U=X\setminus D` 与闭补 `D` 通过 localization sequence 关联；边界 `D` 是另一个对象。Log geometry 则把 divisor 或边界数据编码进 `X` 的 log structure，使得同一个底层 scheme 携带额外边界信息。因此 log motivic theory 可在对象层面保留退化和边界，而不仅通过开闭分解间接记录。`\square`

**例子 21.18.** 一个带 normal crossings divisor `D\subset X` 的对数概形可视为 compactification `U=X\setminus D` 的边界增强版本。Log motivic homotopy 试图使这种边界增强在同伦论中可见。

## 21.6 边界规则

**约定 21.19.** Log、perfect 和 universal homeomorphism 扩展在本书中默认不进入基础定理依赖链。只有在明确标注对应外部输入、基和系数假设时才可使用。

**命题 21.20.** 若某定理在 `\mathbf{SH}[1/p]` 中成立，不能自动推出它在 integral `\mathbf{SH}` 中成立。

**证明.** 局部化 `\mathbf{SH}\to\mathbf{SH}[1/p]` 会杀掉 `p`-primary 信息。一个态射在局部化后成为等价，只说明其 cofiber 为 `p`-power torsion 型或被局部化杀掉；不能推出原 cofiber 为零。因此 integral statement 需要额外证明。`\square`

## 21.7 本章小结

Log schemes、perfect schemes 和 universal homeomorphism invariance 展示了 motivic homotopy theory 在边界和正特征方向的扩展。它们对退化、边界和 Frobenius 几何很重要，但都需要精确假设；本书将它们保留为高级章节和研究边界。

## 练习

**练习 21.1.** 定义 trivial log structure，并说明为什么应恢复普通 schemes。

**练习 21.2.** 定义 perfect scheme。

**练习 21.3.** 解释 universal homeomorphism 的三个条件。

**练习 21.4.** 说明为什么 `[1/p]` 局部化会丢失 integral 信息。

**练习 21.5.** 比较 log boundary 信息和 open complement localization。

**练习 21.6.** 证明 perfect scheme 的 Frobenius 为同构。

**练习 21.7.** 说明 perfectization 为什么可能改变 integral motivic 信息。

**练习 21.8.** 解释 log structure 如何记录边界 divisor。

**练习 21.9.** 比较普通 `\mathbb A^1` interval 和 log interval 的作用。

**练习 21.10.** 给出 nilpotent thickening 是 universal homeomorphism 的例子。
