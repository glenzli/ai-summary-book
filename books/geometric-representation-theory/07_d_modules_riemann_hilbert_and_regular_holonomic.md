# 第七章：D-modules、Riemann-Hilbert 对应与 regular holonomic 条件

## 本章目标

本章建立 D-module 语言，为 Beilinson-Bernstein localization、Kazhdan-Lusztig conjecture 的 D-module 证明和 geometric Langlands 中的 D-module category 做准备。重点是 left/right convention、特征 variety、holonomicity、regularity 和 Riemann-Hilbert 对应的边界。

## 依赖前置知识

需要基本光滑代数簇、切向量场、导出范畴和第三章的 constructible sheaves。

## 7.1 微分算子层

**约定 7.1.** 本章取 $k=\mathbb C$，$X$ 为光滑复代数簇。默认使用 left $\mathcal D_X$-modules。

**定义 7.2.** $\mathcal D_X$ 是由 $\mathcal O_X$ 和 tangent sheaf $\mathcal T_X$ 生成的 sheaf of associative algebras，满足局部关系
$$
f\cdot g=fg,\qquad
\xi f-f\xi=\xi(f),\qquad
\xi\eta-\eta\xi=[\xi,\eta],
$$
其中 $f,g\in\mathcal O_X$，$\xi,\eta\in\mathcal T_X$。

**例 7.3.** 若 $X=\mathbb A^1$，坐标为 $x$，则全局微分算子代数为 Weyl algebra
$$
\Gamma(X,\mathcal D_X)=\mathbb C\langle x,\partial_x\rangle/(\partial_xx-x\partial_x-1).
$$

**定义 7.4.** $\mathcal D_X$ 的 order filtration $F_\bullet\mathcal D_X$ 由微分算子阶数给出。其 associated graded sheaf 记为
$$
\operatorname{gr}\mathcal D_X.
$$

**外部输入定理 7.5.** 对光滑 $X$，
$$
\operatorname{gr}\mathcal D_X\simeq\operatorname{Sym}_{\mathcal O_X}\mathcal T_X,
$$
因此
$$
\operatorname{Spec}_X(\operatorname{gr}\mathcal D_X)\simeq T^\ast X.
$$
该定理可在局部坐标中验证，但全局形式需检查坐标变化。

## 7.2 Good filtrations 和 characteristic variety

**定义 7.6.** 设 $\mathcal M$ 为 coherent left $\mathcal D_X$-module。一个 good filtration 是 increasing filtration $F_\bullet\mathcal M$，使得 $F_i\mathcal D_X\cdot F_j\mathcal M\subset F_{i+j}\mathcal M$，且 $\operatorname{gr}\mathcal M$ 是 $\operatorname{gr}\mathcal D_X$ 上的 coherent module。

**定义 7.7.** characteristic variety 定义为
$$
\operatorname{Char}(\mathcal M)=\operatorname{Supp}_{T^\ast X}(\operatorname{gr}\mathcal M).
$$
good filtration 的不同选择给出相同的 reduced support。

**外部输入定理 7.8.** characteristic variety 与 good filtration 选择无关，并且对非零 coherent $\mathcal D_X$-module 有 Bernstein inequality
$$
\dim\operatorname{Char}(\mathcal M)\ge\dim X.
$$

**定义 7.9.** coherent $\mathcal D_X$-module $\mathcal M$ 称为 holonomic，若
$$
\dim\operatorname{Char}(\mathcal M)=\dim X.
$$

**例 7.10.** $\mathcal O_X$ 作为 left $\mathcal D_X$-module 是 holonomic，其 characteristic variety 是 zero section $T_X^\ast X\subset T^\ast X$。

**证明.** 取 filtration $F_0\mathcal O_X=\mathcal O_X$，$F_i\mathcal O_X=\mathcal O_X$ for $i\ge0$。切向量场作用降低到 $\operatorname{gr}$ 时为零，因此 $\operatorname{gr}\mathcal O_X$ 支撑在 cotangent fiber 坐标全为零的 zero section 上。zero section 维数为 $\dim X$，故 holonomic。$\square$

**例 7.10.1.** 令 $X=\mathbb A^1$，$\delta_0=\mathcal D_X/\mathcal D_X x$。则 $\delta_0$ 支撑在 $0$，其 characteristic variety 是 cotangent fiber $T_0^\ast X$。

**证明.** Weyl algebra $A_1=E\langle x,\partial\rangle/(\partial x-x\partial-1)$ 中，$\delta_0=A_1/A_1x$。取 order filtration。associated graded 为
$$
\operatorname{gr}\delta_0\simeq E[x,\xi]/(x),
$$
作为 $E[x,\xi]$-module 的 support 为 $x=0$，即 $T_0^\ast\mathbb A^1$。该子簇维数为 $1=\dim X$，故 $\delta_0$ holonomic。$\square$

## 7.3 Left/right 转换

**定义 7.11.** 令 $\omega_X$ 为 canonical line bundle。若 $\mathcal M$ 是 left $\mathcal D_X$-module，则
$$
\mathcal M^r=\omega_X\otimes_{\mathcal O_X}\mathcal M
$$
带自然 right $\mathcal D_X$-module 结构。反向地，right module $\mathcal N$ 对应 left module
$$
\mathcal N^\ell=\mathcal Hom_{\mathcal O_X}(\omega_X,\mathcal N).
$$

**警告 7.12.** left/right 转换会影响 de Rham functor 的公式和 pushforward convention。本书默认 left modules；引用 right module 文献时必须转换。

## 7.4 De Rham functor 和 Riemann-Hilbert

**定义 7.13.** 对 left $\mathcal D_X$-module $\mathcal M$，de Rham complex 定义为
$$
\operatorname{DR}_X(\mathcal M)
=\left[
\mathcal M\to\Omega_X^1\otimes\mathcal M\to\cdots\to\Omega_X^{\dim X}\otimes\mathcal M
\right][\dim X],
$$
微分由 connection 作用给出。这里 shift $[\dim X]$ 使 regular holonomic modules 对应 perverse sheaves。

**定义 7.14.** holonomic $\mathcal D_X$-module 称为 regular holonomic，若其奇性满足 regularity 条件。严格定义可用曲线测试、V-filtration 或 formal classification；本书后续附录 E 固定采用的版本。

**外部输入定理 7.15.** Riemann-Hilbert correspondence 给出 regular holonomic $\mathcal D_X$-modules 的 derived category 与 constructible sheaves 的 derived category 之间的反等价或等价，取决于使用 de Rham functor 还是 solution functor。按本书 left-module convention，de Rham functor 给出
$$
D^b_{rh}(\mathcal D_X)\simeq D^b_c(X^{an},\mathbb C)
$$
的相应版本，并把 regular holonomic modules 的心对应到 perverse sheaves。  
来源：Kashiwara、Mebkhout、Hotta-Takeuchi-Tanisaki、Kashiwara-Schapira。

**例 7.16.** $\operatorname{DR}_X(\mathcal O_X)$ 是常值 sheaf $\mathbb C_X[\dim X]$。

**证明.** $\operatorname{DR}_X(\mathcal O_X)$ 是代数 de Rham complex 按 $\dim X$ shift。Poincare lemma 在复解析拓扑上给出 de Rham complex 与 $\mathbb C_X$ 的准同构，因此得到 $\mathbb C_X[\dim X]$。该证明使用 analytic Poincare lemma。$\square$

## 本章小结

本章定义了 $\mathcal D_X$、good filtration、characteristic variety、holonomic 和 regular holonomic modules、left/right 转换以及 de Rham functor，并计算了 $\mathcal O_X$ 和点支撑 delta module 的 characteristic varieties。Riemann-Hilbert correspondence 和 Bernstein inequality 是外部输入。第八章将在这些 convention 上陈述 Beilinson-Bernstein localization。

## 练习

**练习 7.1.** 对 $X=\mathbb A^n$ 写出 $\Gamma(X,\mathcal D_X)$ 的生成元和关系。

**练习 7.2.** 计算 delta module $\mathcal D_{\mathbb A^1}/\mathcal D_{\mathbb A^1}x$ 的 characteristic variety。

**练习 7.3.** 比较 de Rham functor 和 solution functor 的 variance，并说明为什么 Riemann-Hilbert 有时写成反等价。
