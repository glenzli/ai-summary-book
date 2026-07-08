# 第七章：Nygaard filtration、syntomic cohomology 与 Tate twists

## 本章目标

本章固定 Nygaard filtration、syntomic complex 和 $p$-adic Tate twists 的 prismatic 口径。该部分是 prismatic cohomology 与 motivic/etale 信息交汇的技术核心，也是 indexing convention 最容易出错的地方。当前版本给出定义框架、外部输入边界和附录 F 的 convention crosswalk；Bhatt-Scholze 的 Hodge-Tate/Nygaard convention 与 BMS2 的 syntomic/Tate twist 入口已完成源码级 locator，出版前仍需转换为 L3。

## 依赖前置知识

依赖第三章的 Frobenius on prismatic cohomology，第五章的 $A_{\inf}$ 和 $\mu$，第四章的 Tate twists。需要 filtered derived category 和 Frobenius fixed constructions 的基本语言。

## 7.1 Nygaard filtration 的基本形式

**定义 7.1（naive Nygaard condition, oriented case）.** 令 $(A,d)$ 为 oriented prism，令 $M$ 为带 Frobenius-semilinear map $\varphi_M:M\to M$ 的 $A$-module。定义 naive Nygaard 子模
$$
N^{\ge i}_{\mathrm{naive}}M=\{x\in M\mid \varphi_M(x)\in d^iM\}.
$$

**警告 7.2.** 定义 7.1 只是离散、无高阶导出问题时的模型公式。对 prismatic cohomology complex，Nygaard filtration 必须在 filtered derived category 中定义，不能逐项套用 naive 子模公式。

**外部输入定义 7.3.** 对合适的 prismatic cohomology complex $R\Gamma_\Delta(X/A)$，存在 Nygaard filtration
$$
N^{\ge i}R\Gamma_\Delta(X/A),
$$
并且 Frobenius 在第 $i$ 级上可除以 $I^i$，得到 normalized Frobenius 或 divided Frobenius
$$
\varphi_i:N^{\ge i}R\Gamma_\Delta(X/A)\to R\Gamma_\Delta(X/A)\{i\}.
$$

**说明 7.4.** $\{i\}$ twist 和 $\varphi_i$ 的目标 convention 是后续严格化重点。不同文献可能把 $I^i$、$(I/I^2)^i$ 或其 dual 写入不同侧。

## 7.2 Syntomic complexes

**定义 7.5（syntomic fibre, convention form）.** 在存在 Nygaard filtration 和 divided Frobenius 的情形，weight $i$ syntomic complex 的基本形式为 homotopy fibre
$$
R\Gamma_{\mathrm{syn}}(X,\mathbf Z_p(i))
=
\operatorname{fib}\left(
N^{\ge i}R\Gamma_\Delta(X/A)
\xrightarrow{\ \varphi_i-\operatorname{can}_i\ }
R\Gamma_\Delta(X/A)\{i\}
\right).
$$
这里 $\operatorname{can}_i$ 表示从 Nygaard piece 到同一个 Tate-twisted target 的 convention-dependent canonical map；它不是未扭曲对象上的字面恒等映射。

在 BMS2 的 quasisyntomic formulation 中，令 $\widehat{\Prism}_A$ 表示由 $\pi_0TC^-(-;\mathbf Z_p)$ 经 quasisyntomic descent 得到的对象，则模 $p$ 的源码入口为
$$
\mathbf Z/p\mathbf Z(i)(A)
=
\operatorname{hofib}\left(
\varphi_i-1:
\mathcal N^{\ge i}\widehat{\Prism}_A\{i\}/p
\to
\widehat{\Prism}_A\{i\}/p
\right).
$$

**警告 7.6.** 公式 7.5 是本书正文中的 convention form；BMS2 的源码入口已定位到 `eq:TateTwist`，但实际使用时仍必须说明处在 $p$-complete、modulo $p^n$、truncated、quasisyntomic site 或 absolute prismatic site 的哪一种版本中。

**外部输入定理 7.7（syntomic-etale comparison）.** 在适当光滑性、properness、boundedness 和 torsion 假设下，syntomic complex 与 $p$-adic etale Tate twist $\mathbf Z_p(i)$ 比较。BMS2 的 `thm:main6` 给出两个基本出口：在 characteristic $p$ smooth 情形，$\mathbf Z_p(n)$ 与 logarithmic de Rham-Witt sheaves 比较；在 mixed characteristic smooth formal $\mathcal O_C$ 情形，$\mathbf Z_p(n)$ 与截断 nearby cycles $\tau^{\le n}R\psi\mathbf Z_p(n)$ 比较。

## 7.3 Tate twists 的积分问题

**说明 7.8.** Rational Tate twist $\mathbf Q_p(i)$ 在 classical theory 中相对简单；integral Tate twist $\mathbf Z_p(i)$ 需要处理 torsion、Bockstein、Frobenius divisibility 和 filtrations。Prismatic cohomology 的 Nygaard filtration 提供了统一控制这些问题的结构。

**命题 7.9（形式层必要条件）.** 若某个 complex $C(i)$ 要作为 $\mathbf Z_p(i)$ 的 integral prismatic model，则至少应满足：

1. after inverting $p$，与 $\mathbf Q_p(i)$ 的 rational comparison 相容；
2. modulo $p^n$ 后与 etale motivic 或 syntomic model 相容；
3. cup product 下有 $C(i)\otimes C(j)\to C(i+j)$；
4. Frobenius normalization 中的 twist convention 与 Tate twist convention 一致。

**证明.** 第一项确保 rational $p$-adic Hodge theory 不被改变；第二项确保积分 torsion 信息正确；第三项是 Tate twists 的张量结构要求；第四项确保 Frobenius fixed construction 得到正确的 Galois character。缺少任一项，都无法把 $C(i)$ 作为 $\mathbf Z_p(i)$ 的积分模型。证毕。

## 7.4 与 BMS 的关系

**外部输入定理 7.10.** BMS2 构造的 $p$-adic Tate twists 和 syntomic complexes 可由 THH/TC filtration、Nygaard filtration 与 prismatic cohomology 重新解释，并与 prismatic comparison theorem 相容。源码级 locator 为 `BMS2-SYN`，见附录 D 和 [P0_REFERENCE_LOCATORS_BATCH_2.md](P0_REFERENCE_LOCATORS_BATCH_2.md)。

**说明 7.11.** BMS2 源码已确认 $\varphi_i-1$ fibre convention 的入口；Bhatt-Scholze 源码已确认 $\{i\}$ 和 Nygaard graded convention。剩余工作不再是寻找核心源，而是把 $p$-complete、mod $p^r$、truncation 和 nearby cycles 的不同版本逐一分派到正文命题。

## 7.5 前沿接口

**研究边界 7.12.** 2025 年 Carmeli-Feng 使用 perfectoid geometry 与 prismatic cohomology 构造 syntomic Steenrod algebra 和 spectral syntomic cohomology，并组织为 spectral prismatic $F$-gauges。该方向说明 Nygaard/syntomic 结构已超出传统 comparison theorem，进入 operations and duality 层面。本书只记录其位置，不把新结果纳入正文定理。

## 7.6 Naive Nygaard filtration 的形式性质

**命题 7.13.** 在定义 7.1 的 naive oriented 模型中，
$$
N^{\ge i+1}_{\mathrm{naive}}M\subseteq N^{\ge i}_{\mathrm{naive}}M.
$$

**证明.** 若 $x\in N^{\ge i+1}_{\mathrm{naive}}M$，则 $\varphi(x)\in d^{i+1}M$。因为 $d^{i+1}M\subseteq d^iM$，所以 $x\in N^{\ge i}_{\mathrm{naive}}M$。证毕。

**命题 7.14.** 若 $M$、$N$ 带 Frobenius，且 $\varphi_{M\otimes N}(m\otimes n)=\varphi_M(m)\otimes\varphi_N(n)$，则
$$
N^{\ge i}_{\mathrm{naive}}M\otimes N^{\ge j}_{\mathrm{naive}}N
\to
N^{\ge i+j}_{\mathrm{naive}}(M\otimes N).
$$

**证明.** 若 $\varphi_M(m)\in d^iM$ 且 $\varphi_N(n)\in d^jN$，则
$$
\varphi_{M\otimes N}(m\otimes n)
\in d^iM\otimes d^jN
\subset d^{i+j}(M\otimes N).
$$
故张量落在 $N^{\ge i+j}$ 中。证毕。

**警告 7.15.** 命题 7.14 只证明 naive 模型中的乘法相容性。Derived Nygaard filtration 的乘法相容需要外部输入或独立 filtered derived category 论证。

## 本章小结

Nygaard filtration 是 prismatic cohomology 中控制 Frobenius 可除性和 syntomic information 的结构。Syntomic complex 通常是 divided Frobenius 与 identity 的 homotopy fibre。Bhatt-Scholze 与 BMS2 的核心 convention 已完成源码级核查；正式版仍需把各个 $p$-complete、mod $p^r$ 和 truncation 版本写成带 locator 的最终陈述。

## 练习

**练习 7.1.** 在 oriented prism $(A,d)$ 的 naive 模型中，证明 $N^{\ge i+1}_{\mathrm{naive}}M\subseteq N^{\ge i}_{\mathrm{naive}}M$。

**练习 7.2.** 解释为什么 $\varphi_i$ 不是原始 Frobenius，而是除以 $I^i$ 后的 normalized Frobenius。

**练习 7.3.** 写出公式 7.5 中每个对象的系数环，并指出 twist convention 可能出现的两个错误位置。
