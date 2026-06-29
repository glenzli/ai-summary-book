# 附录 Q：GAGA 的 properness、反例与导出比较细节

## Q.0 目标

第六章把 GAGA 作为输入定理。本附录补充三个形式层内容：

1. properness 为什么不可省略。
2. abelian coherent equivalence 如何扩展到 bounded derived coherent category。
3. 上同调比较如何等价于 $R\Gamma$ 比较。

Serre GAGA 本身仍是输入定理；本附录证明接受它之后的形式推论，并给出非 proper 反例。

## Q.1 非 proper 反例：仿射直线

设

$$
X=\mathbb A^1_\mathbb C,
\qquad
X^{an}=\mathbb C.
$$

代数全局函数为

$$
\Gamma(X,\mathcal O_X)=\mathbb C[z].
$$

解析全局函数为

$$
\Gamma(X^{an},\mathcal O_{X^{an}})=\mathcal O(\mathbb C),
$$

即整函数。

**命题 Q.1.** 自然映射

$$
\mathbb C[z]\to\mathcal O(\mathbb C)
$$

不是满射。

**证明.** 函数 $e^z$ 是整函数。若 $e^z$ 是多项式，则它在无穷远有多项式增长；但沿实轴 $e^x$ 增长快于任意多项式。故 $e^z\notin\mathbb C[z]$。证毕。

**推论 Q.2.** 非 proper 情形下，代数相干层与解析相干层的全局截面比较失败。

**证明.** 对结构层 $\mathcal O$，全局截面比较已经不是同构。证毕。

## Q.2 Exact equivalence 到导出等价

设 $\mathcal A,\mathcal B$ 是阿贝尔范畴，$F:\mathcal A\to\mathcal B$ 是 exact equivalence。

**命题 Q.3.** $F$ 诱导三角范畴等价

$$
D^b(\mathcal A)\simeq D^b(\mathcal B).
$$

**证明.** $F$ 逐项作用于有界复形，给出

$$
C^b(\mathcal A)\to C^b(\mathcal B).
$$

因为 $F$ exact，复形 $K^\bullet$ acyclic 当且仅当 $F(K^\bullet)$ acyclic；因此 $F$ 保持并反映 quasi-isomorphism。于是 $F$ 下降到 localization 后的导出范畴。取 exact quasi-inverse $G$，同理得到 $D^b(\mathcal B)\to D^b(\mathcal A)$，两侧复合自然同构于恒等。证毕。

## Q.3 上同调比较到 $R\Gamma$ 比较

设 $X$ 是 proper $\mathbb C$-scheme，$\operatorname{an}$ 为解析化函子。假设 GAGA 给出对每个 coherent sheaf $\mathcal F$ 的上同调同构

$$
H^i(X,\mathcal F)\cong H^i(X^{an},\mathcal F^{an}).
$$

**命题 Q.4.** 对每个 $\mathcal F\in\operatorname{Coh}(X)$，自然映射

$$
R\Gamma(X,\mathcal F)
\to
R\Gamma(X^{an},\mathcal F^{an})
$$

是 $D^b(\mathbb C)$ 中的同构。

**证明.** 两侧是有界复向量空间复形。一个复形态射是 quasi-isomorphism 当且仅当它在所有 cohomology 上诱导同构。假设正是该条件。证毕。

**命题 Q.5.** 命题 Q.4 的比较同构唯一延拓到所有

$$
E\in D^b(\operatorname{Coh}(X)).
$$

**证明.** 令 $\mathcal T$ 为使

$$
R\Gamma(X,E)\to R\Gamma(X^{an},E^{an})
$$

为同构的对象构成的 full subcategory。它对 shift 和 distinguished triangle 的第三项封闭，因为 $R\Gamma$ 是三角函子，且三角范畴中同构满足二出三。它包含所有 coherent sheaf 置于次数 $0$ 的对象。任意 bounded complex 可由 stupid filtration 分解为有限个 cohomological degree 的 sheaf shift，经有限次 cone 拼出。因此 $\mathcal T=D^b(\operatorname{Coh}(X))$。证毕。

## Q.4 Euler characteristic 相容

**推论 Q.6.** 若 $E\in D^b(\operatorname{Coh}(X))$，则

$$
\chi(X,E)=\chi(X^{an},E^{an}).
$$

**证明.** 由命题 Q.5，导出全局截面复形同构，因此每个上同调向量空间维数相同，交错和相同。证毕。

## Q.5 与 condensed/analytic 比较的边界

condensed/analytic 版本的 GAGA 不只是 Q.3-Q.6 的形式推论。它还需要：

1. 代数侧与解析侧对象进入共同的 analytic 派生范畴。
2. properness 或相应紧性条件保证无穷远不产生额外解析函数。
3. rational/analytic descent 能粘合局部比较。
4. $R\Gamma$、trace 和相干对象紧性与 classical GAGA 比较相容。

这些属于 Clausen-Scholze 输入。Q.1-Q.6 只解释 classical GAGA 接受后的形式后果和 properness 的必要性。

## Q.6 练习

**练习 Q.1.** 证明 $\sin z$ 也是 $\mathbb C$ 上非多项式整函数。

**练习 Q.2.** 在命题 Q.3 中说明 exact quasi-inverse 为什么存在。

**练习 Q.3.** 用 stupid filtration 证明任意 bounded complex 可由其各项的 shift 经有限 cone 构造。

**练习 Q.4.** 说明命题 Q.6 为什么需要 coherent finiteness。
