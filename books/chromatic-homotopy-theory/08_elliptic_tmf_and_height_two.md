# 第八章：Elliptic cohomology、tmf 与高度二几何

## 本章目标

本章建立高度二几何的接口：椭圆曲线的形式群、ordinary/supersingular 分层、elliptic cohomology 和 topological modular forms。当前版本只写入严格入口，不把 tmf 的构造细节作为内部证明。

## 依赖前置知识

需要第二章的形式群高度、第三章的 Morava E-theory 和第六章的 $\mathbb E_\infty$ 精化。代数栈、模形式和 derived algebraic geometry 作为外部背景。

## 8.1 椭圆曲线的形式群

**定义 8.1.** 设 $R$ 是交换环，$C/R$ 是椭圆曲线，$e:\operatorname{Spec}R\to C$ 是单位截面。$C$ 在单位处的形式完备化
$$
\widehat C_e
$$
是一维交换形式群，称为椭圆曲线的形式群。

**命题 8.2.** 若 $R$ 是特征 $p$ 的域，椭圆曲线 $C/R$ 的形式群高度只能是 $1$ 或 $2$。

**证明路线（外部输入）.** 椭圆曲线的形式群是一维形式群，其高度受曲线维数和 $p$-divisible group 的高度限制。ordinary 椭圆曲线给出高度 $1$ 的形式群，supersingular 椭圆曲线给出高度 $2$ 的形式群。完整证明依赖椭圆曲线和 $p$-divisible group 理论，作为外部输入。证毕。

**定义 8.3.** 在特征 $p$ 的模空间中，高度 $1$ locus 称为 ordinary locus，高度 $2$ locus 称为 supersingular locus。

## 8.2 Elliptic cohomology

**定义 8.4.** 一个 elliptic cohomology theory 的粗略数据包括：

1. 一个 even periodic ring spectrum $E$；
2. 一个椭圆曲线 $C$ 定义在 $E_0$ 上；
3. 一个同构，把 $E$ 的形式群与 $\widehat C_e$ 识别。

**警告 8.5.** 定义 8.4 只是入口数据。实际的 elliptic spectrum 需要满足 descent、Landweber exactness 或 derived moduli 条件；不同文献中定义强弱不同。

**命题 8.6.** 若椭圆曲线在点处 ordinary，则关联 elliptic cohomology 的局部 chromatic 信息主要落在高度 $1$；若 supersingular，则出现高度 $2$ 的 Morava E-theory 局部模型。

**证明路线（外部输入）.** 由椭圆曲线形式群高度与关联复定向理论的形式群高度相同得到。ordinary/supersingular 分别对应高度 $1/2$。谱层面的严格陈述需要 elliptic cohomology 的构造和局部模型定理。证毕。

## 8.3 Topological modular forms

**外部输入 8.7 (tmf 构造).** Topological modular forms 可由椭圆曲线模栈上的结构 sheaf of $\mathbb E_\infty$-ring spectra 的 derived global sections 构造。这个构造依赖 Goerss-Hopkins-Miller 型 obstruction theory 和后续 derived algebraic geometry。

**定义 8.8.** 本书把
$$
tmf
$$
作为 connective topological modular forms spectrum，把
$$
TMF
$$
作为 periodic version。具体大小写约定必须在使用处说明，因为不同文献对 level structure 和 compactification 的记号不同。

**警告 8.9.** $tmf$ 不是“某个高度二 Morava E-theory”。它是全局模栈对象，局部在 supersingular 点附近与高度二 Morava E-theory 及 stabilizer descent 相连。

## 8.4 高度二与计算接口

**边界 8.10.** 高度二计算通常涉及：

- supersingular elliptic curves 的 automorphism groups；
- Morava stabilizer group 的有限子群；
- descent spectral sequence；
- modular forms 与 divided congruences；
- power operations 和 $E_\infty$ 结构；
- $K(2)$-local sphere 的 chromatic splitting 问题。

当前章节只建立这些主题的入口。完整计算章需独立展开。

## 8.5 Ordinary 与 supersingular 分解

**定义 8.11.** 设 $\mathcal M_{ell}$ 为椭圆曲线模栈。固定素数 $p$ 后，其特征 $p$ fiber 可按形式群高度分为 ordinary locus 和 supersingular locus：
$$
\mathcal M_{ell,\mathbb F_p}=\mathcal M_{ell}^{ord}\cup \mathcal M_{ell}^{ss}.
$$

**命题 8.12.** 对 $\mathcal M_{ell,\mathbb F_p}$ 的任意几何点 $x$，若 $x$ 位于 ordinary locus，则形式完备群 $\widehat C_x$ 的高度为 $1$；若 $x$ 位于 supersingular locus，则其高度为 $2$。因此这两个 locus 分别是椭圆曲线模栈的高度 $1$ 与高度 $2$ 分层。

**证明.** 命题 8.2（其证明依赖椭圆曲线与 $p$-divisible group 理论）说明椭圆曲线形式群的高度只能为 $1$ 或 $2$。Ordinary 的定义等价于 Hasse invariant 非零，也等价于形式群高度为 $1$；supersingular 的定义等价于 Hasse invariant 为零，在仅有的另一种可能下即高度为 $2$。逐个几何点应用此判别即得所述分层。证毕。

**解释 8.13.** 因此 $TMF$ 的 chromatic 分析在高度 $2$ 处集中于 supersingular 点的形式邻域，而这些形式邻域由 Lubin-Tate/Morava E-theory 控制。

## 8.6 tmf 的三层对象

**定义 8.14.** 本书区分三层对象：

1. 弱 elliptic cohomology datum：even periodic spectrum 加椭圆曲线和形式群同构；
2. sheaf of $\mathbb E_\infty$-rings on elliptic moduli：可 descent 的结构化谱层；
3. global sections：$tmf$ 或 $TMF$。

**警告 8.15.** 第 1 层不推出第 2 层，第 2 层的 global sections 才给第 3 层。把三层混成一个“elliptic cohomology”会导致错误的 functoriality 和 power operation 陈述。

## 8.7 $K(2)$-local tmf 的使用规则

**规则 8.16.** 使用 $K(2)$-local tmf 时必须记录：

1. 素数 $p$；
2. 是否加入 level structure；
3. supersingular 点数量；
4. 使用的 Morava E-theory $E_2$；
5. automorphism group 或 groupoid；
6. descent spectral sequence 的输入。

**警告 8.17.** 只有在 supersingular locus 由单点和相应 automorphism group 控制的特殊情形，才可能写成单个 $E_2^{hG}$。一般情形需要 groupoid descent。

## 本章小结

椭圆曲线的形式群把 height $2$ 带入稳定同伦论。ordinary 点对应高度 $1$，supersingular 点对应高度 $2$。tmf 是模栈上的全局谱对象，不是单个 Morava E-theory。它的构造属于外部输入，后续章节应补齐定理定位和计算样例。

## 练习

**练习 8.1.** 查阅椭圆曲线的 ordinary 和 supersingular 定义，说明它们如何通过 Hasse invariant 区分。

**练习 8.2.** 解释为什么高度二局部模型仍需 Morava stabilizer group action，而不仅是 $E_2$ 的系数环。

**练习 8.3.** 写出 $tmf$、$TMF$ 和 $E_2$ 三者在对象类型上的差异。
