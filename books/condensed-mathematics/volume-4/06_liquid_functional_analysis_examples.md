# 第六章：liquid 函数分析例子

## 本章目标

本章整理 liquid 向量空间的基本例子和误读风险。

## 6.1 Banach 空间

Banach 空间 $V$ 可先凝聚化：

$$
S\mapsto\operatorname{Cont}(S,V).
$$

但 liquid 结构不是单纯凝聚化，而是要求对 $\mathcal M_{<p}[S]$ 的 Hom 判别。

## 6.2 Fréchet 空间

全纯函数空间 $\mathcal O(U)$ 常为 Fréchet 空间。第三卷中，Dolbeault 复形的项应放入 liquid 范畴，以保留拓扑和连续性。

## 6.3 分布空间

分布空间通常是某种对偶空间，适合用液态或解析结构处理。关键不是选择一个范数，而是控制测试对象上的测度和连续线性泛函。

## 6.4 风险点

1. liquid 不是 Banach 空间范畴。
2. 连续线性映射必须在凝聚/liquid 意义下处理。
3. 张量积需要使用 analytic 或 liquid 张量，而非普通代数张量。

## 练习

**练习 6.1.** 说明 Banach completion 与 liquid localization 的区别。

**练习 6.2.** 写出 $p$-liquid 判别式。

**练习 6.3.** 解释为什么 Dolbeault 复形需要拓扑向量空间结构。
