# 第二章：相干层与导出范畴

## 本章目标

本章把经典相干解析层放入导出范畴语言，并说明 condensed/analytic 框架如何承载这些对象。

## 2.1 相干解析层

**定义 2.1.** 复解析空间 $X$ 上的 $\mathcal O_X$-模 $\mathcal F$ 称为相干层，如果局部存在正合列

$$
\mathcal O_X^m\to\mathcal O_X^n\to\mathcal F\to0.
$$

相干层范畴记为

$$
\operatorname{Coh}(X).
$$

## 2.2 导出范畴

记

$$
D_{\operatorname{coh}}(X)
$$

为上同调层相干的导出范畴。经典上同调写作

$$
R\Gamma(X,\mathcal F).
$$

## 2.3 凝聚表述

**输入定理 2.2（Clausen-Scholze，形式）.** 对适当的 $X$，存在 faithful 的范畴解释，把 $\operatorname{Coh}(X)$ 和 $D_{\operatorname{coh}}(X)$ 放入 analytic/liquid 派生范畴中，并保持导出全局截面。

这意味着，经典的

$$
H^i(X,\mathcal F)
$$

可以看作 analytic/liquid 派生范畴中某个对象的同调。

## 2.4 紧性与有限性

在稳定范畴中，对象 $C$ 称为紧，如果

$$
\operatorname{Hom}(C,\bigoplus_i X_i)
\cong
\bigoplus_i\operatorname{Hom}(C,X_i).
$$

相干层的有限性常表现为某些对象的紧性或 perfect 性。

## 2.5 本章小结

相干解析层进入凝聚数学后，核心问题变成：这些对象是否紧、推前是否保持相干性、对偶是否可由 $f^!$ 描述。

## 练习

**练习 2.1.** 写出相干层的局部有限表示定义。

**练习 2.2.** 解释 $D_{\operatorname{coh}}(X)$ 与 $\operatorname{Coh}(X)$ 的关系。

**练习 2.3.** 说明紧对象定义与有限维上同调之间的关系。
