# 第四章：相干上同调有限性

## 本章目标

本章讨论紧复流形上相干上同调有限维定理。

## 4.1 经典表述

**定理 4.1（经典有限性）.** 若 $X$ 是紧复流形，$\mathcal F$ 是相干解析层，则

$$
H^i(X,\mathcal F)
$$

是有限维复向量空间，并且只有有限多个 $i$ 非零。

## 4.2 凝聚表述

在 condensed/analytic 范畴中，该定理可读作：导出全局截面

$$
R\Gamma(X,\mathcal F)
$$

是 $\mathbb C$ 上的 perfect 或有限型对象。

**输入定理 4.2（Clausen-Scholze）.** 对 compact complex manifold $X$ 和 coherent analytic sheaf $\mathcal F$，$R\Gamma(X,\mathcal F)$ 在 condensed/analytic 语境中是有限型对象，其同调为有限维复向量空间。

## 4.3 证明路线

1. 用 Dolbeault 复形计算 $R\Gamma(X,\mathcal F)$。
2. 将相关函数空间放入 liquid 范畴。
3. 使用 elliptic regularity 或 Fredholm 性质证明同调有限维。
4. 将分析有限性翻译为 condensed/analytic 范畴中的紧性。

## 4.4 本章小结

有限性定理是复几何应用的第一块基石。它确保后续对偶配对和 trace map 落在有限维对象上。

## 练习

**练习 4.1.** 证明若一个有界复形由有限维向量空间组成，则其同调有限维。

**练习 4.2.** 解释为什么紧性是有限性定理的必要条件。

**练习 4.3.** 写出 Dolbeault 复形如何计算 $H^i(X,\mathcal O_X)$。
