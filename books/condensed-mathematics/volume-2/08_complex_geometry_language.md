# 第八章：复几何应用的范畴语言

## 本章目标

本章说明 condensed/analytic 方法如何进入复几何。这里不重证复几何定理，而是把 Clausen-Scholze 的目标翻译成本卷已经建立的范畴语言。

## 依赖

需要第五章的 liquid 入口和第七章的相干对偶图景。

## 8.1 复几何中的分析对象

经典复几何使用：

1. 复流形上的全纯函数。
2. coherent analytic sheaves。
3. Dolbeault 复形和 $\bar\partial$ 算子。
4. Fréchet、nuclear、Banach 等拓扑向量空间。

这些对象天然带有拓扑和分析结构。凝聚数学的目标不是抛弃这些结构，而是把它们放入适合做同调代数的范畴中。

## 8.2 condensed/analytic 替换

在本卷语言中，替换步骤如下：

1. 拓扑向量空间替换为凝聚或 liquid 向量空间。
2. 函数空间替换为某个 analytic ring 上的模。
3. sheaf cohomology 替换为 analytic/solid 派生范畴中的 derived global sections。
4. 对偶性通过 $f_!$ 和 $f^!$ 组织。

## 8.3 目标定理

Clausen-Scholze 的复几何讲义以 compact complex manifolds 为核心，重新证明若干经典定理。

**输入定理 8.1（Clausen-Scholze，路线图）.** 在 condensed/analytic 框架中，可重新证明 compact complex manifolds 上的：

1. coherent cohomology finite-dimensionality。
2. Serre duality。
3. GAGA。
4. Grothendieck-Hirzebruch-Riemann-Roch。

本书第二卷不把这些定理作为已经完成的证明，而是把它们列为后续几何卷的目标。

## 8.4 Serre duality 的形式

对紧复流形 $X$，经典 Serre duality 形如

$$
H^i(X,\mathcal F)
\times
H^{n-i}(X,\mathcal F^\vee\otimes\omega_X)
\to
\mathbb C.
$$

在六函子语言中，它应来自

$$
R\operatorname{Hom}(R\Gamma(X,\mathcal F),\mathbb C)
\simeq
R\Gamma_c(X,R\mathcal Hom(\mathcal F,\omega_X[n])).
$$

凝聚数学把右侧的紧支撑和对偶对象放入 analytic 派生范畴中处理。

## 8.5 GAGA 的形式

GAGA 比较 algebraic coherent sheaves 与 analytic coherent sheaves。凝聚数学中的比较应通过同一个 analytic/solid 派生范畴解释：

$$
D_{\operatorname{coh}}(X_{\operatorname{alg}})
\longrightarrow
D_{\operatorname{coh}}(X_{\operatorname{an}})
$$

在 proper 条件下成为等价，并保持 cohomology。

这需要：

1. analytic rings 对 complex analytic spaces 的建模。
2. coherent objects 的定义。
3. proper pushforward 或 $f_!$ 的有限性。
4. 对偶性和 trace map。

## 8.6 本卷边界

本卷完成的是范畴语言：

1. solid 派生范畴。
2. analytic rings 与解析化。
3. liquid 向量空间入口。
4. Huber pair 到解析环。
5. $f_!$、投影公式和对偶的形式。

真正证明 compact complex manifolds 的 finiteness、Serre duality、GAGA 和 Riemann-Roch，需要另写几何卷。

## 8.7 本章小结

复几何应用说明 condensed mathematics 不是单纯重写拓扑空间，而是为分析对象建立可做同调代数和对偶理论的环境。

## 练习

**练习 8.1.** 写出经典 Serre duality 的陈述，并指出其中哪个部分对应 $f^!$。

**练习 8.2.** 解释为什么 coherent cohomology finite-dimensionality 是范畴紧性问题。

**练习 8.3.** 比较 GAGA 中 algebraic 与 analytic 两侧的对象。

**练习 8.4.** 说明为什么本章只给范畴语言而不直接证明 Riemann-Roch。
