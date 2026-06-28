# 附录 E：GAGA 与 Riemann-Roch 例子

## E.0 目标

本附录补充 GAGA 和 Riemann-Roch 的基础例子，帮助读者把第六、七章中的抽象表述落到计算上。

## E.1 解析化函子

设 $X$ 是 $\mathbb C$ 上有限型 scheme。其 analytification 记为

$$
X^{an}.
$$

若 $\mathcal F$ 是 $X$ 上的 algebraic coherent sheaf，则有 analytic coherent sheaf

$$
\mathcal F^{an}
$$

在 $X^{an}$ 上。

GAGA 断言 proper 情形中

$$
\mathcal F\mapsto\mathcal F^{an}
$$

给出相干层范畴等价。

## E.2 例子：射影空间上的结构层

令 $X=\mathbb P^n_\mathbb C$。则

$$
H^0(X,\mathcal O_X)=\mathbb C,
$$

且

$$
H^i(X,\mathcal O_X)=0,\qquad i>0.
$$

GAGA 比较说明 analytic projective space 上的 $\mathcal O$ 也有相同上同调。

## E.3 例子：$\mathbb P^1$ 上的 $\mathcal O(d)$

经典计算：

$$
\dim H^0(\mathbb P^1,\mathcal O(d))=
\begin{cases}
d+1,&d\ge0,\\
0,&d<0,
\end{cases}
$$

并且由 Serre duality，

$$
H^1(\mathbb P^1,\mathcal O(d))
\cong
H^0(\mathbb P^1,\mathcal O(-d-2))^\vee.
$$

因此

$$
\chi(\mathbb P^1,\mathcal O(d))=d+1.
$$

## E.4 Riemann-Roch 检查

对 $\mathbb P^1$，令 $H$ 为 hyperplane class。则

$$
\operatorname{ch}(\mathcal O(d))=e^{dH}=1+dH
$$

因为 $H^2=0$。同时

$$
\operatorname{td}(T_{\mathbb P^1})=1+H.
$$

于是

$$
\int_{\mathbb P^1}\operatorname{ch}(\mathcal O(d))
\operatorname{td}(T_{\mathbb P^1})
=
\int_{\mathbb P^1}(1+dH)(1+H)
=
d+1.
$$

这与 $\chi(\mathbb P^1,\mathcal O(d))=d+1$ 一致。

## E.5 凝聚表述中的意义

在 condensed/analytic 框架中，上述等式应理解为：

1. $\mathcal O(d)$ 给出相干对象。
2. $R\Gamma(\mathbb P^1,\mathcal O(d))$ 是有限型对象。
3. Euler characteristic 由 trace 计算。
4. trace 与 Chern character/Todd class 兼容。

## E.6 本附录小结

GAGA 和 Riemann-Roch 的抽象形式必须通过这些低维例子检查。$\mathbb P^1$ 上的 $\mathcal O(d)$ 是最小测试例。

## 练习

**练习 E.1.** 用 Cech 覆盖计算 $H^0(\mathbb P^1,\mathcal O(d))$。

**练习 E.2.** 用 Serre duality 推出 $H^1(\mathbb P^1,\mathcal O(d))$ 的维数。

**练习 E.3.** 直接展开 $\operatorname{ch}(\mathcal O(d))\operatorname{td}(T_{\mathbb P^1})$ 并计算积分。
