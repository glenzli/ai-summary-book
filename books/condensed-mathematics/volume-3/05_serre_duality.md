# 第五章：Serre 对偶

## 本章目标

本章讨论 Serre duality 的 classical 和 condensed/analytic 表述。

## 5.1 经典表述

设 $X$ 是紧复流形，维数为 $n$，$\mathcal F$ 是相干解析层。经典 Serre 对偶给出配对

$$
H^i(X,\mathcal F)\times
\operatorname{Ext}^{n-i}(\mathcal F,\omega_X)
\to
\mathbb C
$$

并在有限性条件下诱导完美对偶。

若 $\mathcal F$ 是向量丛，则可写成

$$
H^i(X,\mathcal F)^\vee
\cong
H^{n-i}(X,\mathcal F^\vee\otimes\omega_X).
$$

## 5.2 六函子表述

令 $f:X\to *$。Serre 对偶可看作

$$
R\operatorname{Hom}(Rf_*\mathcal F,\mathbb C)
\simeq
Rf_*R\mathcal Hom(\mathcal F,f^!\mathbb C).
$$

在非 proper 或带支撑条件的情形中，应使用

$$
f_!
$$

和其右伴随 $f^!$。

## 5.3 凝聚表述

**输入定理 5.1（Clausen-Scholze）.** 在 condensed/analytic 框架中，compact complex manifolds 上的 Serre duality 可由 $f_!$、$f^!$ 和 trace map 统一表达。

## 5.4 证明路线

1. 用 Dolbeault 复形表示两侧；见附录 D 与附录 I。
2. 构造积分配对
   $$
   \int_X \alpha\wedge\beta.
   $$
3. 证明配对与 $\bar\partial$ 相容；见附录 J 命题 J.4。
4. 使用有限性定理和 Serre perfectness 输入得到完美性；见附录 J 命题 J.3 与推论 J.6。
5. 把该配对识别为 $f_!\dashv f^!$ 的 counit/trace；见附录 J 定义 J.7 至推论 J.9。
6. 若相干层有有限局部自由 resolution，则向量丛形式推出 Ext 形式；见附录 O。

$\mathbb P^n$ 上线丛情形的可计算模型见附录 T；它用 Čech 单项式和 residue trace 直接证明完美配对。

## 5.5 本章小结

Serre duality 是相干对偶的核心例子。condensed/analytic 方法的贡献在于把它放入统一的六函子语言中。

## 练习

**练习 5.1.** 对紧 Riemann surface 写出 Serre duality。

**练习 5.2.** 解释 $\omega_X$ 在 Serre duality 中的角色。

**练习 5.3.** 写出 $f_!\dashv f^!$ 的伴随 counit。
