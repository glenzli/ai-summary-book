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

附录 K 证明 Riemann-Roch 输入定理接受之后的形式推论：Euler characteristic 是 $K$-理论上的群同态，GAGA 比较保持 Euler characteristic，并且 $\mathbb P^1$ 上 $\mathcal O(d)$ 的 characteristic class 计算给出 $d+1$。

附录 P 进一步展开 Chern character、Todd class、splitting principle 和 $K$-理论同态的形式代数，说明 HRR 右侧为什么是 $K^0(X)\to\mathbb Q$ 的群同态。

附录 S 计算 $\mathbb P^n$ 上 $\mathcal O(d)$ 的上同调和 Euler characteristic，给出 Riemann-Roch 在线丛基础情形下的可复核检验。

附录 U 进一步直接计算 $\operatorname{ch}(\mathcal O(d))\operatorname{td}(T_{\mathbb P^n})$ 的积分，证明 $\mathbb P^n$ 线丛情形的 HRR 公式。

一般 Grothendieck-Riemann-Roch 的输入形式，以及它推出 HRR、可加性和复合相容的证明，见附录 AE。

## 7.5 本章小结

Riemann-Roch 是前面有限性、对偶性和 GAGA 的综合应用。condensed/analytic 方法给出统一范畴背景，但 characteristic class 的完整构造需要更多几何输入。

## 练习

**练习 7.1.** 对 $X=\mathbb P^1$ 和 $E=\mathcal O_X$ 计算 $\chi(X,E)$。

**练习 7.2.** 写出 Chern character 的低阶项。

**练习 7.3.** 解释 trace map 如何进入 Euler characteristic。
