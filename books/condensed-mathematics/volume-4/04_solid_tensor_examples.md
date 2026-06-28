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

## 4.2 profinite 集合

若 $S,T$ profinite，则输入定理给出

$$
\mathbb Z^\square[S]\otimes^{L,\square}
\mathbb Z^\square[T]
\simeq
\mathbb Z^\square[S\times T].
$$

这可视为测度外积。

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

## 4.4 风险点

该公式不是普通阿贝尔群公式。普通张量积通常不与无限乘积按这种方式交换。关键在于 $\otimes^{L,\square}$ 包含 solidification。

## 练习

**练习 4.1.** 对有限集合验证张量公式。

**练习 4.2.** 解释为什么 profinite 情形可理解为测度外积。

**练习 4.3.** 举例说明普通张量积不保持无限乘积。
