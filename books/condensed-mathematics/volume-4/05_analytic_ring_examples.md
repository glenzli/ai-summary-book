# 第五章：analytic rings 的例子库

## 本章目标

本章整理第二卷中的 analytic ring 例子。

## 5.1 solid 解析环

基本例子：

$$
(\mathbb Z,\mathbb Z^\square).
$$

其解析模就是 solid 阿贝尔群。

这里 $\mathbb Z^\square[S]$ 不是额外装饰，而是解析结构：它指定在 profinite 测试对象 $S$ 上应当允许的“自由解析模”。对有限 $S$，它退化为通常的自由模；对无限 profinite $S$，它记录 solid 完备性。

**命题 5.1.1。** 若 $S$ 有限，则

$$
\mathbb Z^\square[S]\cong\mathbb Z[S].
$$

**证明。** 有限 $S$ 上的 locally constant 整值函数、有限支整系数形式和通常自由阿贝尔群三者一致。solidification 对有限直和不产生额外完备化，因此解析自由对象就是普通自由对象。证毕。

## 5.2 $p$-进例子

对 $\mathbb Z_p$，可取

$$
\mathbb Z_p^\square[S]
=
\varprojlim_i\mathbb Z_p[S_i],
\qquad
S=\varprojlim_iS_i.
$$

这给出 $p$-进完备方向的 analytic ring。

若 $S$ 有限，上式给出 $\mathbb Z_p[S]$。若 $S$ 为 Cantor 型 profinite 集合，则 $\mathbb Z_p^\square[S]$ 是有限层自由 $\mathbb Z_p$-模的逆极限。它的元素可看成与有限商相容的 $p$-进整系数测度。

**类型检查。** 该构造同时使用两个极限：

1. $S=\varprojlim_iS_i$ 是 profinite 集合的表示。
2. $\mathbb Z_p^\square[S]=\varprojlim_i\mathbb Z_p[S_i]$ 是凝聚/solid 模中的极限。

第二个极限不能替换成离散阿贝尔群中的朴素极限后再忘记拓扑结构；否则会丢失 analytic ring 需要的完备性。

## 5.3 有限生成代数

若 $A$ 是有限生成 $\mathbb Z$-代数，则

$$
A^\square[S]=\varprojlim_iA[S_i]
$$

是基本测度对象。

对有限 $S$，该公式退化为普通多份拷贝：

$$
A^\square[S]\cong A[S]\cong \bigoplus_{s\in S}A .
$$

对 profinite $S$，它是有限层 $A[S_i]$ 的相容系统。若 $A\to B$ 是有限生成代数映射，则有自然基变换映射

$$
A^\square[S]\otimes_A B\to B^\square[S].
$$

在有限 $S$ 上这是同构；在 profinite $S$ 上是否为所需意义的同构，取决于张量积和极限是否在相应 solid/analytic 范畴中计算。这正是第二卷强调 analytic localization 的原因。

## 5.4 Huber pair

离散 Huber pair

$$
(A,A^+)
$$

给出解析环

$$
(A,A^+)^\square.
$$

rational localization 是从代数例子进入几何的关键。

这里 $A^+$ 不是可有可无的子环。它控制有界元素，从而控制哪些 rational localization 被允许进入解析几何。若只记 $A$ 而忘记 $A^+$，同一个底层环可能给出不同的解析行为。

**例 5.4.1。** 对 Tate 型对象，$A^+$ 规定幂有界元素的积分模型。rational localization

$$
A\to A\left\langle \frac{f_1,\ldots,f_n}{g}\right\rangle
$$

应同时更新有界子环。凝聚/analytic ring 语言把这一点编码进解析结构，而不只是编码进底层环同态。

## 5.5 风险点

1. $A^\square$ 与 $(A,A^+)^\square$ 需要区分。
2. ordinary completion 与 analytic localization 不是同一个操作。
3. 任意测度理论不自动满足 analytic ring 条件。
4. 有限层公式通常容易证明；无限 profinite 情形需要确认极限、张量积和局部化所在范畴。

## 练习

**练习 5.1.** 写出 $\mathbb Z^\square[S]$ 的定义。

**练习 5.2.** 对有限 $S$ 计算 $A^\square[S]$。

**练习 5.3.** 解释 $A^+$ 在 Huber pair 中的作用。

**练习 5.4.** 设 $S=\varprojlim_iS_i$。说明为什么 $A^\square[S]$ 的定义应与有限商 $S_i$ 的选择无关。
