# 第七章：Grothendieck-Hirzebruch-Riemann-Roch

## 本章目标

本章说明 Riemann-Roch 在 condensed/analytic 框架中的位置。

## 7.1 经典形式

对紧复流形 $X$ 和向量丛 $E$，Hirzebruch-Riemann-Roch 断言

$$
\chi(X,E)=\int_X \operatorname{ch}(E)\operatorname{td}(T_X).
$$

其中

$$
\chi(X,E)=\sum_i(-1)^i\dim H^i(X,E).
$$

## 7.2 Grothendieck 形式

对 proper 映射 $f:X\to Y$，Grothendieck-Riemann-Roch 比较

$$
\operatorname{ch}(Rf_*E)\operatorname{td}(T_Y)
$$

与

$$
f_*(\operatorname{ch}(E)\operatorname{td}(T_X)).
$$

## 7.3 凝聚表述

**输入定理 7.1（Clausen-Scholze）.** 在 condensed/analytic 框架中，Riemann-Roch 可通过相干对象、trace、Chern character 和对偶形式表达，并恢复经典公式。

## 7.4 证明路线

1. 用 GAGA 或 analytic comparison 把问题放入 analytic 派生范畴。
2. 定义 K-theory 类和 Chern character。
3. 用 trace map 表达 Euler characteristic。
4. 证明 trace 与 characteristic classes 相容。

## 7.5 本章小结

Riemann-Roch 是前面有限性、对偶性和 GAGA 的综合应用。condensed/analytic 方法给出统一范畴背景，但 characteristic class 的完整构造需要更多几何输入。

## 练习

**练习 7.1.** 对 $X=\mathbb P^1$ 和 $E=\mathcal O_X$ 计算 $\chi(X,E)$。

**练习 7.2.** 写出 Chern character 的低阶项。

**练习 7.3.** 解释 trace map 如何进入 Euler characteristic。
