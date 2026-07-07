# 第十一章：Etale comparison、Frobenius fixed points 与 syntomic tower

## 本章目标

本章展开 etale comparison 的结构：prismatic complex 本身不是 etale cohomology，必须通过 invert prism ideal、Frobenius fixed points、modulo $p^n$ 和 syntomic tower 才能与 $\mathbf Z_p(i)$ 型对象比较。

## 依赖前置知识

依赖第三章 etale comparison、第七章 Nygaard/syntomic、第四章 Tate twists 和附录 F 的符号交叉表。需要 derived fixed points 和 homotopy fibre 的基本语言。

## 11.1 Perfect prism 上的 etale comparison

**约定 11.1.** 令 $(A,I)$ 为 perfect bounded prism，$X$ 为 smooth proper $p$-adic formal scheme over $A/I$，$X_\eta$ 为 generic fibre。

**外部输入定理 11.2（etale comparison refined form）.** 在约定 11.1 下，prismatic cohomology 在 invert $I$ 后带 Frobenius，并且其 Frobenius fixed construction modulo $p^n$ 与
$$
R\Gamma_{\mathrm{et}}(X_\eta,\mathbf Z/p^n)
$$
自然比较。取 inverse limit 后得到 $\mathbf Z_p$-cohomology 的 comparison。

**警告 11.3.** “Frobenius fixed construction”在 derived category 中应理解为 homotopy fixed 或 fibre of $\varphi-1$ 型对象，不能只取 cohomology group 上的逐点不变量。

## 11.2 Frobenius fixed points

**定义 11.4.** 令 $C$ 为带 endomorphism $\varphi:C\to C$ 的 derived object。定义 derived Frobenius fixed complex 为
$$
C^{\varphi=1}:=\operatorname{fib}(C\xrightarrow{\varphi-1}C).
$$

**命题 11.5.** 若 $C$ concentrated in degree $0$，则存在短正合列
$$
0\to H^0(C^{\varphi=1})\to C\xrightarrow{\varphi-1}C\to H^1(C^{\varphi=1})\to0,
$$
且 $H^0(C^{\varphi=1})=\ker(\varphi-1)$。

**证明.** $C^{\varphi=1}$ 是两项 complex $[C\xrightarrow{\varphi-1}C]$ 的 fibre shift convention。按本书 cohomological convention 计算其长正合列，$H^0$ 为 kernel，$H^1$ 为 cokernel。证毕。

**警告 11.6.** 命题 11.5 说明普通 fixed points 只看见 $H^0$。若 $\varphi-1$ 不满，$H^1$ 记录 obstruction；因此 derived fixed points 在比较定理中不可省略。

## 11.3 Syntomic tower

**定义 11.7.** 对 $i\ge0$，syntomic complex 的 convention form 为
$$
R\Gamma_{\mathrm{syn}}(X,\mathbf Z_p(i))
=
\operatorname{fib}\left(
N^{\ge i}R\Gamma_\Delta(X/A)
\xrightarrow{\varphi_i-1}
R\Gamma_\Delta(X/A)\{i\}
\right).
$$
Modulo $p^n$ version 记作
$$
R\Gamma_{\mathrm{syn}}(X,\mathbf Z/p^n(i)).
$$
在 BMS2 的 quasisyntomic sheaf 模型中，模 $p$ 的 Tate twist 入口为
$$
\mathbf Z/p\mathbf Z(i)(A)
=
\operatorname{hofib}\left(
\varphi_i-1:
\mathcal N^{\ge i}\widehat{\Prism}_A\{i\}/p
\to
\widehat{\Prism}_A\{i\}/p
\right),
$$
因此本章的 syntomic tower 应理解为 Nygaard filtration、Tate twist 和 Frobenius fibre 的兼容系统，而不是单一裸 complex。

**外部输入定理 11.8（syntomic-etale comparison）.** 在适当范围和 finiteness 假设下，syntomic tower 与 etale motivic 或 $p$-adic Tate twist cohomology 比较：
$$
R\Gamma_{\mathrm{syn}}(X,\mathbf Z/p^n(i))
\longrightarrow
R\Gamma_{\mathrm{et}}(X_\eta,\mathbf Z/p^n(i)).
$$
BMS2 的源码级入口 `thm:main6` 给出两类标准出口：

1. 若 $X=\operatorname{Spec}A$ smooth over perfect field $k$ of characteristic $p$，则 $\mathbf Z_p(n)$ 与 $W\Omega^n_{X,\log}[-n]$ 比较。
2. 若 $\mathfrak X=\operatorname{Spf}A$ smooth formal over $\mathcal O_C$，则 $\mathbf Z_p(n)$ 与 $\tau^{\le n}R\psi\mathbf Z_p(n)$ 比较。

比较的精确截断范围依赖 $i$、$p$ 和几何假设；这些范围必须在最终 L3 locator 阶段逐条写入。

**警告 11.9.** 定理 11.8 的“范围”不能省略。许多 syntomic-etale comparison 只在 truncation 后或特定 cohomological degree 范围内是同构。

## 11.4 Cup products

**外部输入定理 11.10.** Prismatic cohomology、Nygaard filtration 和 syntomic complexes 支持与 Tate twist 加法相容的乘法结构：
$$
R\Gamma_{\mathrm{syn}}(X,\mathbf Z_p(i))
\otimes^L
R\Gamma_{\mathrm{syn}}(X,\mathbf Z_p(j))
\to
R\Gamma_{\mathrm{syn}}(X,\mathbf Z_p(i+j)).
$$

**命题 11.11（结构保真要求）.** 若一个 comparison map 声称识别 syntomic 与 etale Tate twists，则它必须与 cup products 相容，至少在 derived category 中 up to coherent homotopy。

**证明.** Tate twists 构成 graded commutative multiplicative system。若 comparison 不保乘法，则无法把 Chern classes、cycle class maps 和 Steenrod-type operations 在两侧对应。因此它不能作为 cohomology theory 的结构性比较。证毕。

## 11.5 Derived fixed points 的长正合列

**命题 11.12.** 对带 endomorphism $\varphi:C\to C$ 的 complex，定义
$$
C^{\varphi=1}=\operatorname{fib}(C\xrightarrow{\varphi-1}C).
$$
则存在长正合列
$$
\cdots\to H^n(C^{\varphi=1})\to H^n(C)
\xrightarrow{\varphi-1}
H^n(C)\to H^{n+1}(C^{\varphi=1})\to\cdots.
$$

**证明.** Fibre triangle
$$
C^{\varphi=1}\to C\xrightarrow{\varphi-1}C
$$
在 derived category 中给出 cohomology 长正合列。证毕。

**推论 11.13.** 若 $\varphi-1$ 在每个 $H^n(C)$ 上为同构，则 $C^{\varphi=1}$ acyclic。

**证明.** 长正合列中所有中间映射均为同构，kernel 和 cokernel 均为零，故所有 $H^n(C^{\varphi=1})=0$。证毕。

## 本章小结

Etale comparison 的核心不是裸同构，而是经过 invert $I$、Frobenius fixed points、Nygaard filtration 和 syntomic tower 的结构性比较。Derived fixed points、$\varphi_i-1$ fibre、mod $p^n$ tower 和 truncation range 是正式表述中不可省略的组成部分。

## 练习

**练习 11.1.** 对两项 complex $[C\xrightarrow{\varphi-1}C]$ 写出 cohomology，并解释 kernel/cokernel 的含义。

**练习 11.2.** 说明为什么 ordinary fixed points 不能替代 derived fixed points。

**练习 11.3.** 给出一个需要 cup product compatibility 的几何构造，例如 Chern class 或 cycle class map。
