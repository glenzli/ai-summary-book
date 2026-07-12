# 第九章：Hodge-Tate 与 de Rham specialization 的滤过结构

## 本章目标

本章把第三章中只作为比较接口出现的 Hodge-Tate 和 de Rham specialization 展开为正式教材内容。核心任务是区分三类结构：prismatic complex 本身、Hodge-Tate specialization 的 conjugate filtration、以及 de Rham specialization 的 Hodge filtration。

## 依赖前置知识

依赖第二章的 prismatic site、第三章的 comparison theorem interface、附录 A 的 derived completion 和 cotangent complex 的基本语言。需要熟悉 filtered complexes 和 exterior powers of the cotangent complex。

## 9.1 Hodge-Tate specialization

**定义 9.1.** 令 $(A,I)$ 为 bounded prism，$X$ 为 smooth $p$-adic formal scheme over $A/I$。定义 Hodge-Tate specialization 为
$$
\overline\Delta_{X/A}
=R\Gamma_\Delta(X/A)\otimes_A^L A/I.
$$
若 $X=\operatorname{Spf}(R)$，则记为 $\overline\Delta_{R/A}$。

**定义 9.2.** 若 $I/I^2$ 为 invertible $A/I$-module，对 $A/I$-module $M$ 记
$$
M\{i\}=M\otimes_{A/I}(I/I^2)^{\otimes i}.
$$
负次数 twist 用 dual invertible module 定义：
$$
M\{-i\}=M\otimes_{A/I}(I/I^2)^{\otimes(-i)}.
$$

**警告 9.3.** 文献中 Hodge-Tate twist 的正负号可能因 convention 变化。
本书固定采用定义 9.2 与附录 F 的 convention；任何外部公式进入正文前都
必须先按该 convention 转写，不能逐章临时换号。

## 9.2 Conjugate filtration

**外部输入定理 9.4（Hodge-Tate comparison refined form）.** 在定义 9.1
的假设下，$X_{\mathrm{et}}$ 上的 Hodge--Tate complex 带自然 Postnikov
filtration。本文固定递增编号
$$
\operatorname{Fil}^{\mathrm{conj}}_i
=\tau^{\le i},
$$
并对该 sheaf filtration 取 derived global sections。所得 associated graded
满足
$$
\operatorname{gr}^{\mathrm{conj}}_i\overline\Delta_{X/A}
\simeq
R\Gamma\left(X,\wedge^i\mathbb L_{X/(A/I)}\right)[-i]\{-i\}.
$$
若 $X$ smooth，则 $\wedge^i\mathbb L_{X/(A/I)}$ 可替换为 $\Omega^i_{X/(A/I)}$。
来源为 Bhatt--Scholze, Theorems 4.11、6.3（locator `BS-COMP-HT`）。

**说明 9.5.** 定理 9.4 是外部输入，不在本书中重证。书内使用它时只抽取两个形式后果：perfectness 和 Hodge numbers 的约束。

**命题 9.6.** 若 $X$ proper smooth over $A/I$，且每个 $R\Gamma(X,\Omega^i_{X/(A/I)})$ 是 perfect $A/I$-complex，则 $\overline\Delta_{X/A}$ 是 perfect $A/I$-complex。

**证明.** 由定理 9.4，conjugate filtration 的各 associated graded 是 perfect complex 的 shift 和 invertible twist，因此仍为 perfect。Smooth relative dimension 有限，故 filtration 有有限非零 graded pieces。Perfect complexes 在有限 extension 下封闭，所以 $\overline\Delta_{X/A}$ perfect。证毕。

**命题 9.7.** 在命题 9.6 的假设下，若每个 $H^j(X,\Omega^i)$ 为有限生成 $A/I$-module，则每个 $H^n(\overline\Delta_{X/A})$ 为有限生成 $A/I$-module。

**证明.** 有限 filtration 给出 spectral sequence
$$
E_1^{i,j}=H^{i+j}\left(R\Gamma(X,\Omega^i)[-i]\{-i\}\right)
\Rightarrow H^{i+j}(\overline\Delta_{X/A}).
$$
$E_1$ 页由有限生成模组成，且只有有限多列非零。每个 $E_\infty$ 项为有限生成模的 subquotient，目标 cohomology 有有限 filtration，因此有限生成。证毕。

## 9.3 De Rham specialization

**定义 9.8.** De Rham specialization 定义为
$$
\Delta^{\mathrm{dR}}_{X/A}
=\phi_A^\ast R\Gamma_\Delta(X/A)\widehat\otimes_A^L A/I,
$$
其中
$$
\phi_A^\ast R\Gamma_\Delta(X/A)=A\otimes_{A,\phi_A}^LR\Gamma_\Delta(X/A).
$$
帽号表示 tensor product 之后的 derived $p$-completion，见定义 3.2。

**外部输入定理 9.9（de Rham comparison）.** 在 smooth bounded prism
假设下，存在自然拟同构
$$
\Delta^{\mathrm{dR}}_{X/A}
\simeq
R\Gamma_{\mathrm{dR}}(X/(A/I))^{\wedge,L}_p,
$$
并且该同构与乘法结构及 pullback functoriality 相容。来源为
Bhatt--Scholze, Corollary 15.4（locator `BS-COMP-DR`）。这是 unfiltered
comparison；右侧当然带其 Hodge filtration，但把该 filtration 搬到左侧需要
relative Nygaard theorem 等额外输入，不能从本定理自动推出。

右侧的递减 Hodge filtration 是截断 forms 的 intrinsic filtration
$$
\operatorname{Fil}_H^jR\Gamma_{\mathrm{dR}}^{\wedge,L}_p
=R\Gamma\left(X,\Omega^{\ge j}_{X/(A/I)}\right)^{\wedge,L}_p.
$$
其 associated graded 为
$R\Gamma(X,\Omega^j_{X/(A/I)})[-j]$ 的 derived $p$-completion。这个
定义不需要 Nygaard theory；需要额外输入的是它与 prismatic-side
filtration 的 compatibility。

**警告 9.10.** 定理 9.9 不能由定理 9.4 形式推出。Hodge-Tate specialization 与 de Rham specialization 的差异包含 Frobenius pullback，因此二者是不同 comparison theorem。

## 9.4 两个滤过的关系

**说明 9.11.** Conjugate filtration 位于 Hodge-Tate specialization 上；
Hodge filtration 位于 $p$-completed de Rham complex 上。二者通过 prismatic
object 同源，但不是同一个 filtration；定理 9.9 也不声称二者之间有 filtered
identification。

**命题 9.12（不可混用判别）.** 若一个论证只知道 $\overline\Delta_{X/A}$ 的 conjugate filtration，则不能推出 $R\Gamma_{\mathrm{dR}}(X/(A/I))$ 上 Hodge filtration 的 strictness。

**证明.** Conjugate filtration 的对象是 $\overline\Delta_{X/A}$；Hodge filtration 的对象是 $\Delta^{\mathrm{dR}}_{X/A}$。二者由不同 specialization 得到。没有额外比较定理给出 filtered equivalence 时，从一个 filtered object 的 strictness 推出另一个 filtered object 的 strictness 属于跨对象推理。故该推论无效。证毕。

## 9.5 低维 spectral sequence 展开

**例 9.13.** 若 $X$ relative dimension 为 $1$，则 Hodge-Tate conjugate spectral sequence 的 $E_1$ 页只有两列：
$$
E_1^{0,j}=H^j(X,\mathcal O_X),\qquad
E_1^{1,j}=H^j(X,\Omega^1_{X/(A/I)})\{-1\}.
$$
总次数由 shift $[-1]$ 调整，因此第二列贡献到 $H^{j+1}$。

**命题 9.14.** 若 $X$ relative dimension $1$ 且 $H^j(X,\mathcal O_X)=H^j(X,\Omega^1)=0$ for $j>1$，则 $\overline\Delta_{X/A}$ 的 cohomology 只可能出现在次数 $0,1,2$。

**证明.** 例 9.13 给出 spectral sequence 的非零项范围。第一列 $i=0$ 贡献次数 $j=0,1$；第二列经 shift 后贡献总次数 $j+1=1,2$。因此目标 cohomology 只可能在 $0,1,2$。证毕。

## 本章小结

Hodge-Tate specialization 和 de Rham specialization 是 prismatic cohomology 的两个不同出口。前者带 conjugate filtration，其 graded pieces 由 $\Omega^i[-i]\{-i\}$ 描述；后者通过 Frobenius pullback 后 modulo $I$ 得到 de Rham complex，并带 Hodge filtration。正式使用时必须同时标注对象、滤过和 twist convention。

## 练习

**练习 9.1.** 设 $X$ relative dimension 为 $d$。说明定理 9.4 中为什么只有 $0\le i\le d$ 的 graded pieces 可能非零。

**练习 9.2.** 写出命题 9.7 中 spectral sequence 的总次数，并解释 shift $[-i]$ 如何改变 $E_1$ 页。

**练习 9.3.** 构造一个错误论证，把 conjugate filtration 当作 Hodge filtration 使用；然后指出错误发生在哪一步。
