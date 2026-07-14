# 第八章：复几何应用的范畴语言

复流形上的全纯截面天然组成 Fréchet 空间，Dolbeault 微分连接这些空间，而相干层的
上同调与对偶又要求核、商和紧支撑推前相互兼容。仅把每个拓扑向量空间忘成代数向量
空间会丢失分析结构；仅说它们“是 liquid”又不足以保证复形的 cokernel 与上同调比较。
第五章的 liquid 实现和局部提升条件正好给出对象层与正合层之间的区分。

另一方面，第七章的 $f_!\dashv f^!$ 把紧支撑与对偶组织成伴随。我们据此逐项翻译
相干有限性、Serre 对偶、GAGA 和 Riemann-Roch 所需的范畴对象与态射，说明哪些来自
本卷形式，哪些仍是 Clausen-Scholze 或经典复几何的深层外部输入。第三卷将把这些接口
连回具体 Dolbeault、Čech 和射影空间计算，而不是把四个定理只列成未来目标。

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

## 8.3 四项经典比较问题

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

## 8.6 范畴接口还缺少的几何输入

solid 与 analytic localization 已经提供派生代数，liquid 实现提供函数空间对象，
Huber pair 和 $f_!\dashv f^!$ 则提供局部化与对偶接口。要从这些形式得到紧复流形上的
有限维性、Serre 对偶、GAGA 和 Riemann-Roch，仍需 Dolbeault 局部正合、椭圆有限性、
相干解析层理论和特征类推前等深层几何输入。第三卷会精确登记这些输入，并把接受输入后
的 Čech、谱序列、对偶与特征类形式后果写回主线正文。

## 8.7 通往复几何主线的接口

拓扑函数空间经 liquid 实现进入可做导出运算的范畴，解析 sheaf 的上同调可由局部化与
totalization 组织，紧支撑和对偶则由 $f_!\dashv f^!$ 连接。这些接口解释了凝聚语言
能承载哪些经典构造，却不替代有限性、对偶或 GAGA 的深层证明。第三卷将从复解析空间
和相干层本身开始，把这里的类型翻译落实为可计算例子与严格的输入后推论。

## 练习

**练习 8.1.** 写出经典 Serre duality 的陈述，并指出其中哪个部分对应 $f^!$。

**练习 8.2.** 解释为什么 coherent cohomology finite-dimensionality 是范畴紧性问题。

**练习 8.3.** 比较 GAGA 中 algebraic 与 analytic 两侧的对象。

**练习 8.4.** 说明为什么本章只给范畴语言而不直接证明 Riemann-Roch。
