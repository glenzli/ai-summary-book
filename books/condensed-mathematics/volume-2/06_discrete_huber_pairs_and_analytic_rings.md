# 第六章：离散 Huber pair 与解析环

## 本章目标

本章说明离散 Huber pair 如何给出解析环，并解释 rational localization 为什么是几何化的入口。

## 依赖

需要第三、四章和第一卷第十五章。

## 6.1 离散 Huber pair

**定义 6.1.** 离散 Huber pair 是一对

$$
(A,A^+)
$$

其中 $A$ 是离散环，$A^+\subset A$ 是整闭子环，并在 Huber pair 语境中控制幂有界元素。

典型例子：

1. $(\mathbb Z,\mathbb Z)$。
2. $(A,A)$，其中 $A$ 是有限生成 $\mathbb Z$-代数。
3. $(A,A^+)$，其中 $A^+$ 指定了允许的积分元素。

## 6.2 Spa

**定义 6.2.** $\operatorname{Spa}(A,A^+)$ 是 $A$ 上满足

$$
|a|\le1,\qquad a\in A^+
$$

的 valuation 等价类集合。

rational subset 形如

$$
U\left(\frac{g_1,\ldots,g_n}{f}\right)
=
\{x\mid |g_i(x)|\le |f(x)|\ne0,\ 1\le i\le n\}.
$$

这些集合构成拓扑基。

## 6.3 解析环构造

**输入定理 6.3（Scholze）.** 离散 Huber pair $(A,A^+)$ 可函子性地给出解析环

$$
(A,A^+)^\square.
$$

当 $A$ 是有限生成 $\mathbb Z$-代数且 $A^+=A$ 时，它与第一卷中的 $A^\square$ 型测度理论相容。

## 6.4 Rational localization

设

$$
U=U\left(\frac{g_1,\ldots,g_n}{f}\right)
\subset\operatorname{Spa}(A,A^+).
$$

几何上，$U$ 对应于把 $f$ 变为可逆并要求 $g_i/f$ 有界。代数上，可构造新的 Huber pair

$$
(B,B^+)
$$

并有解析环映射

$$
(A,A^+)^\square\to(B,B^+)^\square.
$$

**输入定理 6.4（Scholze）.** rational localization 在解析模范畴上满足期望的局部化性质；特别是限制到 rational subsets 与解析化相容。

## 6.5 Cech 下降

若 $\{U_i\}$ 是 $\operatorname{Spa}(A,A^+)$ 的 rational 覆盖，则期望有

$$
D((A,A^+)^\square)
\to
\operatorname{Tot}\left(
\prod_iD(U_i)\rightrightarrows
\prod_{i,j}D(U_i\cap U_j)\triplearrows\cdots
\right)
$$

的下降描述。

**输入定理 6.5（Scholze）.** 在适当假设下，解析模满足 rational Cech 下降。

这个定理使解析环从仿射局部对象走向几何空间。

## 6.6 例子：$\mathbb Z[T]$

令 $A=\mathbb Z[T]$，$A^+=A$。则

$$
\operatorname{Spec}A\to\operatorname{Spec}\mathbb Z
$$

是仿射直线。其解析环 $A^\square$ 记录代数变量 $T$ 的同时，也记录由凝聚/solid 测度带来的完备性。

在非 proper 情形中，$T=\infty$ 的边界贡献会在 $f_!$ 中出现。这是第七章的主题。

## 6.7 本章小结

离散 Huber pair 把解析环与几何空间联系起来。rational localization 和 Cech 下降是后续构造 $f_!$、$f^!$ 与相干对偶的局部工具。

## 练习

**练习 6.1.** 写出 valuation 的乘法性和三角不等式。

**练习 6.2.** 对 $U(g/f)$，解释条件 $|g(x)|\le|f(x)|\ne0$ 的含义。

**练习 6.3.** 对 $(\mathbb Z,\mathbb Z)$，描述 $\operatorname{Spa}(\mathbb Z,\mathbb Z)$ 中有限素数点和无穷远方向的直观区别。

**练习 6.4.** 说明为什么 rational localization 是构造 sheaf-like 几何理论的必要步骤。
