# 第六章：离散 Huber pair 与解析环

单个离散环 $A$ 不能区分哪些元素在 valuation 下应当有界，也不能记录仿射空间的
rational 子域。加入整闭子环 $A^+$ 得到 Huber pair 后，条件 $|A^+|\le1$ 选出
$\operatorname{Spa}(A,A^+)$ 的点；再指定 $|g_i|\le|f|\ne0$，便得到可由分式坐标环
描述的 rational subset。几何化的关键正是让解析模随这些局部化下降。

第一卷第十五章已给出 Huber pair 与 valuation 的入口，第三、四章则提供 analytic ring
和 localization 的正式语言。我们先陈述从 $(A,A^+)$ 构造
$(A,A^+)^\square$ 的精确外部输入，再逐步写出 rational localization 的输入、坐标环
输出和 Čech nerve。rational acyclicity 与范畴下降保持为 Scholze 深层定理，接受它们
后的 totalization 后果在正文和附录 G 中完整推导。

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

当 $A$ 是有限生成 $\mathbb Z$-代数并取
$A^+=A_{\mathbb Z}^+$（$\mathbb Z$ 在 $A$ 中像的整闭包）时，它与第一卷中的
$A^\square$ 型测度理论相容。取 $A^+=A$ 则给出另一离散 Huber 图表；除两者确实相等
的特殊情形外，不应把它与相对解析结构记成同一个 $A^\square$。

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
\prod_{i,j}D(U_i\cap U_j)
\rightrightarrows\cdots
\right)
$$

的下降描述。

**输入定理 6.5（Scholze）.** 对第二卷附录 D.7 所登记的离散 Huber pair 解析化，在 rational covering 上，解析模范畴满足 rational Cech 下降。

这个定理使解析环从仿射局部对象走向几何空间。

形式层面的 Cech nerve、totalization 和局部等价检测见附录 G。特别要注意：这里的 totalization 是稳定范畴或范畴值 descent 的 totalization，不只是阿贝尔群的等化子。

## 6.6 例子：$\mathbb Z[T]$ 的两个离散 Huber pair

令 $A=\mathbb Z[T]$。这里必须区分

$$
(A,A)
\qquad\text{与}\qquad
(A,\mathbb Z).
$$

前者要求 $|T|\le1$，给出仿射 scheme 的离散 adic 图表；后者只要求整数有界，允许
$|T|>1$ 的 valuation，因而是相对 $\operatorname{Spec}\mathbb Z$ 的解析扩大。对一般
有限型 $\mathbb Z$-代数，应把第二个正子环写成 $A_{\mathbb Z}^+$，即 $\mathbb Z$ 在
$A$ 中像的整闭包；本例中 $A_{\mathbb Z}^+=\mathbb Z$。

本卷约定

$$
A^\square:=(A,A_{\mathbb Z}^+)^\square
$$

用于有限型 $\mathbb Z$-代数的相对解析结构，而
$(A,A)^\square$ 保留完整 Huber-pair 记号。二者不能仅因底层环相同而混同。

在非 proper 情形中，离散图表
$\operatorname{Spa}(A,A)\to\operatorname{Spa}(A,\mathbb Z)$ 的补集给出“无穷远”
方向；相应边界贡献会在 $f_!$ 中出现。这是第七章的主题。

## 6.7 Rational 图册上的解析模

$(A,A^+)$ 同时控制代数函数和 valuation 的有界方向，rational subset 则把不等式
$|g_i|\le|f|$ 转化为局部坐标环。外部输入的 rational Čech 下降说明，全局解析模范畴
可由这些局部范畴及其交叠的 totalization 恢复；这比阿贝尔群等化子包含更高相容数据。
在 $\mathbb Z[T]$ 的非 proper 例子中，rational 局部化仍看见无穷远边界，下一章的
$f_!$ 正要把该边界纳入推前和对偶。

## 练习

**练习 6.1.** 写出 valuation 的乘法性和三角不等式。

**练习 6.2.** 对 $U(g/f)$，解释条件 $|g(x)|\le|f(x)|\ne0$ 的含义。

**练习 6.3.** 对 $(\mathbb Z,\mathbb Z)$，比较平凡 valuation、$p$-进型 valuation 与
支撑在 $(p)$ 的 valuation，并说明通常的 Archimedean 绝对值为什么不满足
$|\mathbb Z|\le1$，因而不定义该 Spa 的点。

**练习 6.4.** 说明为什么 rational localization 是构造 sheaf-like 几何理论的必要步骤。
