# 附录 A：导出完备化、Koszul complex 与 $p^\infty$-torsion

## 本附录目标

本附录记录 prismatic theory 中反复使用的完备化和 torsion 约定。它不是完整的 derived commutative algebra 教材，而是为正文中的 bounded prism、completed flatness 和 base change 提供最小技术词典。

## A.1 Derived completion

**定义 A.1.** 令 $A$ 为环，$J\subset A$ 为有限生成理想。对 $M\in D(A)$，其 derived $J$-adic completion 定义为
$$
M^{\wedge,L}_J=R\varprojlim_n\left(M\otimes_A^L A/J^n\right).
$$
若自然映射 $M\to M^{\wedge,L}_J$ 为同构，则称 $M$ derived $J$-complete。

**警告 A.2.** 对一般非 noetherian 环，ordinary completion 与 derived completion 可以不同。Prismatic theory 中大量环非 noetherian，因此默认使用 derived completion。

**命题 A.3.** 若 $M$ 是 bounded below complex，且每个 $H^i(M)$ 在 noetherian 环 $A$ 上有限生成，则 derived $J$-completion 与逐项 ordinary completion 相容。

**证明草图.** 这是 derived completion 的标准 Artin-Rees/ML 型结果。Noetherian 与有限生成假设保证 inverse system 满足 Mittag-Leffler 条件，并控制 $\varprojlim^1$。证毕。

## A.2 Koszul complexes

**定义 A.4.** 对 $f\in A$，Koszul complex $K(A;f)$ 为
$$
[A\xrightarrow{f}A]
$$
其中左侧位于 cohomological degree $0$。对有限序列 $\mathbf f=(f_1,\ldots,f_r)$，定义
$$
K(A;\mathbf f)=K(A;f_1)\otimes_A\cdots\otimes_AK(A;f_r).
$$

**说明 A.5.** Derived $J$-completion 可用 Koszul towers 描述。正文中不使用该模型重证 prismatic base change，但它解释了为什么完备化天然属于 derived category。

## A.3 $p^\infty$-torsion

**定义 A.6.** 对环或模 $M$，定义
$$
M[p^n]=\{x\in M\mid p^nx=0\},\qquad
M[p^\infty]=\bigcup_{n\ge0}M[p^n].
$$
称 $M$ 的 $p^\infty$-torsion 有界，如果存在 $N$ 使得
$$
M[p^\infty]=M[p^N].
$$

**命题 A.7.** 若 $M$ 无 $p$-torsion，则 $M[p^\infty]=0$，因此其 $p^\infty$-torsion 有界。

**证明.** 无 $p$-torsion 意味着乘以 $p$ 单射。若 $p^nx=0$，反复使用单射性得 $x=0$。故 $M[p^\infty]=0=M[p^0]$。证毕。

**说明 A.8.** Bounded prism 的 boundedness 条件要求 $A/I$ 的 $p^\infty$-torsion 有界。该条件防止 completion 和 descent 中出现不可控的高阶 $p$-torsion。

## A.4 Completed flatness

**定义 A.9.** 令 $A$、$B$ 为 derived $J$-complete rings。称 $A\to B$ 为 derived $J$-completely flat，如果对任意 $M\in D(A)$，completed base change
$$
M\mapsto (M\otimes_A^LB)^{\wedge,L}_J
$$
在适当完备子范畴中保持 exact triangles，并在 modulo $J$ 后给出 flat base change。

**警告 A.10.** 这是当前扩展稿的工作定义。最终版本需要替换为 Bhatt-Scholze 中使用的 precisely stated completely flatness。

## 本附录小结

Prismatic theory 的完备化、flatness 和 torsion 条件必须在 derived 语境中处理。Boundedness 不是装饰性假设，而是保证 cohomology、descent 和 base change 行为可控的结构条件。

## 练习

**练习 A.1.** 对 $A=\mathbf Z_p$、$J=(p)$、$M=A$，计算 $M^{\wedge,L}_J$。

**练习 A.2.** 若 $M$ 无 $p$-torsion，证明 $M/p^nM$ 的 transition maps 形成 Mittag-Leffler 系统。

**练习 A.3.** 写出 $K(A;f,g)$ 的四项 complex，并标明 differential 的符号约定。
