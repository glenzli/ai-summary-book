# 附录 AD：Dualizing complex 与一般 Serre 对偶

## AD.0 目标

附录 AA 证明了向量丛情形的 Serre 对偶。一般相干层需要 dualizing complex。本附录给出一般 Grothendieck-Serre duality 的精确形式，并证明其推出卷三使用的 Ext-Serre 配对。

Dualizing complex 的存在和 Grothendieck duality theorem 作为输入定理。

## AD.1 Dualizing complex

**定义 AD.1.** 设 $X$ 是 proper 复解析空间。对象

$$
\omega_X^\bullet\in D^b_{\operatorname{coh}}(X)
$$

称为 dualizing complex，如果函子

$$
D_X(-)=R\mathcal Hom_X(-,\omega_X^\bullet)
$$

给出 $D^b_{\operatorname{coh}}(X)$ 的反等价，并且自然映射

$$
\mathcal F\to D_XD_X\mathcal F
$$

为同构。

**输入定理 AD.2（dualizing complex 存在性）.** 对 proper 复解析空间 $X$，存在 dualizing complex $\omega_X^\bullet$。

若 $X$ 是复维数 $n$ 的复流形，则

$$
\omega_X^\bullet\simeq\omega_X[n].
$$

## AD.2 Global duality

**输入定理 AD.3（Grothendieck-Serre duality, map to point）.** 设 $X$ 是 proper 复解析空间，$\mathcal F\in D^b_{\operatorname{coh}}(X)$。存在自然同构

$$
R\operatorname{Hom}_{\mathbb C}(R\Gamma(X,\mathcal F),\mathbb C)
\simeq
R\Gamma(X,D_X\mathcal F).
$$

该同构与 trace map

$$
R\Gamma(X,\omega_X^\bullet)\to\mathbb C
$$

相容。

## AD.3 Ext-Serre 配对

**定理 AD.4.** 若 $X$ 是 $n$ 维紧复流形，$\mathcal F$ 是相干解析层，则存在自然完美配对

$$
H^i(X,\mathcal F)
\times
\operatorname{Ext}^{n-i}_X(\mathcal F,\omega_X)
\to
\mathbb C.
$$

**证明.** 对 $\mathcal F$ 应用 AD.3，并用光滑情形 $\omega_X^\bullet\simeq\omega_X[n]$。右侧为

$$
R\Gamma(X,R\mathcal Hom(\mathcal F,\omega_X[n])).
$$

其第 $-i$ 个 cohomology 是

$$
H^{-i}R\Gamma(R\mathcal Hom(\mathcal F,\omega_X[n]))
=
\operatorname{Ext}^{n-i}_X(\mathcal F,\omega_X).
$$

左侧是 $R\Gamma(X,\mathcal F)$ 的 derived dual。由 Grauert finite-dimensionality，$H^i(X,\mathcal F)$ 有限维，因此 derived dual 的 cohomology 给出 $H^i(X,\mathcal F)^\vee$。于是得到自然同构

$$
H^i(X,\mathcal F)^\vee
\cong
\operatorname{Ext}^{n-i}_X(\mathcal F,\omega_X),
$$

等价于所需完美配对。证毕。

## AD.4 向量丛情形回收

若 $\mathcal F=\mathcal O(E)$ 是向量丛截面层，则

$$
R\mathcal Hom(\mathcal O(E),\omega_X)
\simeq
\mathcal O(E^\vee)\otimes\omega_X.
$$

因此 AD.4 化为

$$
H^i(X,\mathcal O(E))^\vee
\cong
H^{n-i}(X,\mathcal O(E^\vee\otimes\omega_X)),
$$

即附录 AA 的结论。

## AD.5 与六函子形式

设 $f:X\to *$。Grothendieck-Serre duality 可写作

$$
R\operatorname{Hom}(Rf_\ast\mathcal F,\mathbb C)
\simeq
Rf_\ast R\mathcal Hom(\mathcal F,f^!\mathbb C).
$$

其中

$$
f^!\mathbb C\simeq\omega_X^\bullet.
$$

这正是 condensed/analytic 六函子语言中 $f_!\dashv f^!$ 和 trace/counit 的 classical shadow。

## 练习

1. 在光滑曲线情形，把 AD.4 写成 $H^0$ 与 $H^1$ 的两个配对。
2. 证明向量丛情形下 AD.4 等同于 AA.3。
3. 解释有限维性在 AD.4 证明中出现的位置。
4. 说明 dualizing complex 比 canonical bundle 多处理了哪些奇异情形。
