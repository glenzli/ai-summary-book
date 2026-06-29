# 附录 AH：Hörmander \(L^2\) 方法与 Stein 消没

## AH.0 目标

Cartan B 可经 Runge-Cousin 路线证明，也可经 \(\bar\partial\) 的 \(L^2\) 解法证明。本附录记录第二条路线的证明模块：

1. Stein 流形上的严格 plurisubharmonic exhaustion。
2. Hörmander \(L^2\) 估计作为输入。
3. \(\bar\partial\)-closed 形式的全局可解性。
4. Dolbeault cohomology 消没。
5. 在有全局有限自由 resolution 时推出相干层消没。

Hörmander estimate 本身作为分析输入。

## AH.1 Stein exhaustion

**输入定理 AH.1（Stein exhaustion）.** 若 \(X\) 是 Stein 流形，则存在光滑严格 plurisubharmonic exhaustion

$$
\rho:X\to\mathbb R
$$

使得子水平集

$$
X_c=\{x\in X\mid \rho(x)<c\}
$$

相对紧，并且对正则值 \(c\)，\(X_c\) 是强 pseudoconvex 域。

## AH.2 Hörmander \(L^2\) 输入

**输入定理 AH.2（Hörmander \(L^2\) estimate）.** 设 \(X\) 是 complete Kähler Stein 流形，\(\varphi\) 是严格 plurisubharmonic 权函数。若

$$
\alpha\in L^2_{(0,q)}(X,E,\varphi),\qquad q>0,
$$

满足

$$
\bar\partial\alpha=0,
$$

并且曲率正性给出估计常数 \(C_q\)，则存在

$$
u\in L^2_{(0,q-1)}(X,E,\varphi)
$$

使

$$
\bar\partial u=\alpha,\qquad
\|u\|_\varphi^2\le C_q\|\alpha\|_\varphi^2.
$$

这里 \(E\) 可取平凡向量丛；向量丛情形需 Nakano 正性或通过扭权处理。

## AH.3 从 \(L^2\) 解到光滑解

**输入定理 AH.3（椭圆正则性）.** 若 \(\alpha\) 是光滑 \((0,q)\)-形式，且 \(u\) 是 \(L^2_{\mathrm{loc}}\) 解

$$
\bar\partial u=\alpha,
$$

则存在光滑解 \(u'\) 满足同一方程。

**命题 AH.4（Stein 上 \(\bar\partial\) 高阶消没）.** 若 \(X\) 是 Stein 流形，\(q>0\)，则

$$
H^q(\Gamma(X,\mathcal A^{0,\bullet}),\bar\partial)=0.
$$

**证明.** 取 \(\bar\partial\)-closed 光滑 \((0,q)\)-形式 \(\alpha\)。用 AH.1 取 exhaustion，并选择权函数使 \(\alpha\) 在对应 \(L^2\) 空间中可处理。AH.2 给 \(L^2\) 解 \(u\)，AH.3 把它提升为光滑解。于是 \(\alpha=\bar\partial u\)，cohomology 类为零。证毕。

## AH.4 有限自由层情形的 Cartan B

**定理 AH.5.** 若 \(X\) 是 Stein 流形，则

$$
H^q(X,\mathcal O_X^{\oplus r})=0
\qquad(q>0).
$$

**证明.** Dolbeault resolution 给

$$
H^q(X,\mathcal O_X^{\oplus r})
\cong
H^q(\Gamma(X,\mathcal A^{0,\bullet})^{\oplus r},\bar\partial).
$$

右侧由 AH.4 消没。证毕。

**定理 AH.6（有全局有限自由 resolution 的相干层消没）.** 设 \(X\) 是 Stein 流形，\(\mathcal F\) 有全局有限自由 resolution

$$
0\to\mathcal O_X^{r_m}\to\cdots\to\mathcal O_X^{r_0}\to\mathcal F\to0.
$$

则

$$
H^q(X,\mathcal F)=0
\qquad(q>0).
$$

**证明.** 对 resolution 长度归纳。长度零由 AH.5。一般情形取

$$
0\to\mathcal K\to\mathcal O_X^{r_0}\to\mathcal F\to0,
$$

其中 \(\mathcal K\) 有更短的有限自由 resolution。长正合列和归纳假设给 \(H^q(X,\mathcal F)=0\) 对 \(q>0\)。证毕。

## AH.5 一般相干层的边界

一般 Stein 空间上的相干层不一定在本书中已给定全局有限自由 resolution。完整 Cartan B 需要 Oka coherence、Cartan A、局部 resolution 的拼接和 Runge/Cousin 或 \(\bar\partial\) 方法结合。

本附录证明的是：

1. \(\bar\partial\) 分析输入推出自由层消没；
2. 有全局有限自由 resolution 时推出相干层消没；
3. 一般相干层情形仍需 Oka-Cartan 理论。

## 练习

1. 用 AH.4 证明 \(H^1(\mathbb C,\mathcal O)=0\)。
2. 写出长度一 resolution 情形下 AH.6 的长正合列。
3. 解释为什么 \(L^2\) 解需要椭圆正则性才能给 sheaf cohomology 结论。
4. 比较 AH.6 与 Cartan B 的强弱。
