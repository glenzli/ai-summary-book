# 第四章：solid 张量积例子

## 本章目标

本章给出 solid 张量积的基本例子，并强调它和普通张量积的区别。

## 4.1 有限集合

若 $S,T$ 有限，则

$$
\mathbb Z^\square[S]=\mathbb Z[S],
\qquad
\mathbb Z^\square[T]=\mathbb Z[T].
$$

因此

$$
\mathbb Z^\square[S]\otimes^{L,\square}
\mathbb Z^\square[T]
\simeq
\mathbb Z[S\times T].
$$

**证明。** 有限离散集合同时是 compact Hausdorff 和 profinite。此时自由凝聚阿贝尔群 $\mathbb Z[\underline S]$ 已经是 solid 对象，solidification 不改变它；同理于 $T$。普通自由阿贝尔群满足

$$
\mathbb Z[S]\otimes_{\mathbb Z}\mathbb Z[T]
\cong
\mathbb Z[S\times T],
\qquad
[s]\otimes[t]\mapsto[(s,t)].
$$

因为两边都由基元素生成，且该映射把基双射到基，所以是同构。有限情形无高阶导出项，故导出 solid 张量积与普通张量积一致。证毕。

## 4.2 profinite 集合

若 $S,T$ profinite，则输入定理给出

$$
\mathbb Z^\square[S]\otimes^{L,\square}
\mathbb Z^\square[T]
\simeq
\mathbb Z^\square[S\times T].
$$

这可视为测度外积。

更准确地说，$\mathbb Z^\square[S]$ 可理解为 $S$ 上的 solid 整系数测度对象。若

$$
S=\varprojlim_iS_i,\qquad T=\varprojlim_jT_j
$$

是有限集合的逆极限，则 $S\times T=\varprojlim_{i,j}(S_i\times T_j)$，而有限层上的公式与逆极限相容。深层输入是 solid 张量积正是使这种 profinite 极限计算成立的张量结构；它不是普通阿贝尔群张量积的形式推论。

## 4.3 乘积型对象

若

$$
\mathbb Z^\square[S]\cong\prod_I\underline{\mathbb Z},
\quad
\mathbb Z^\square[T]\cong\prod_J\underline{\mathbb Z},
$$

则

$$
\prod_I\underline{\mathbb Z}
\otimes^{L,\square}
\prod_J\underline{\mathbb Z}
\simeq
\prod_{I\times J}\underline{\mathbb Z}.
$$

这是上一节在 Stone 对偶视角下的一个常见计算：若 profinite 集合由布尔代数的极限给出，对应的 solid 自由对象表现为乘积型对象。该公式应读作 solid 范畴中的公式，而不是 $\mathbf{Ab}$ 中的公式。

**普通张量积反例。** 自然映射

$$
\left(\prod_{n\ge1}\mathbb Z\right)\otimes_{\mathbb Z}\mathbb Q
\longrightarrow
\prod_{n\ge1}\mathbb Q
$$

不是满射。左边等于把 $\prod_n\mathbb Z$ 对所有非零整数局部化，因而每个元素可写成

$$
\frac{(a_n)_n}{m}
$$

其中分母 $m$ 对所有坐标相同。右边元素

$$
\left(1,\frac12,\frac13,\frac14,\ldots\right)
$$

没有统一有界分母，因此不在像中。这个例子说明普通张量积不与无限乘积按坐标交换；solid 张量积公式的内容恰恰在于修正这个缺陷。

## 4.4 风险点

该公式不是普通阿贝尔群公式。普通张量积通常不与无限乘积按这种方式交换。关键在于 $\otimes^{L,\square}$ 包含 solidification。

使用本章公式时必须记录三类信息：

1. 输入对象是否已经 solid。
2. 张量积是在普通、导出还是 solid 导出意义下进行。
3. profinite 极限是否在 solid 范畴中被正确保持。

## 练习

**练习 4.1.** 对有限集合验证张量公式。

**练习 4.2.** 解释为什么 profinite 情形可理解为测度外积。

**练习 4.3.** 举例说明普通张量积不保持无限乘积。

**练习 4.4.** 证明上面的 $\left(1,\frac12,\frac13,\ldots\right)$ 不在 $\left(\prod_n\mathbb Z\right)\otimes\mathbb Q$ 的像中。
