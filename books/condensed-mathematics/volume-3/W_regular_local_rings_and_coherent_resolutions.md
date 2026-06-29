# 附录 W：正则局部环与相干层有限分解

## W.0 目标

一般相干层的 Serre 对偶、有限性和 Riemann-Roch 都需要把相干层同向量丛或局部自由层联系起来。本附录补充所需的局部代数：

1. 复流形局部环是正则局部环。
2. 正则局部环上的有限模有有限自由分解。
3. 相干层在每点附近有有限局部自由分解。
4. sheaf Hom 和 Ext sheaf 的局部计算。

Hilbert syzygy、Auslander-Buchsbaum 和解析局部环正则性作为交换代数输入定理使用；本附录证明这些输入如何转化为 sheaf 层面的结论。

## W.1 正则局部环输入

**定义 W.1.** Noether 局部环 $(R,\mathfrak m,k)$ 称为正则局部环，如果

$$
\dim_k\mathfrak m/\mathfrak m^2=\dim R.
$$

**输入定理 W.2（复流形局部环正则性）.** 若 $X$ 是复维数 $n$ 的复流形，$x\in X$，则局部环

$$
\mathcal O_{X,x}
$$

是维数 $n$ 的正则 Noether 局部环。

在坐标邻域中，它同构于收敛幂级数环

$$
\mathbb C\{z_1,\ldots,z_n\}.
$$

**输入定理 W.3（正则局部环有限整体维数）.** 若 $R$ 是维数 $n$ 的正则局部环，则每个有限生成 $R$-模 $M$ 有长度不超过 $n$ 的有限自由分解：

$$
0\to R^{r_n}\to\cdots\to R^{r_0}\to M\to0.
$$

## W.2 Stalk 到 sheaf

**引理 W.4.** 设 $\mathcal F$ 是复空间 $X$ 上相干解析层。若在点 $x$ 的 stalk 上有有限表示

$$
\mathcal O_{X,x}^{\oplus a}
\xrightarrow{\phi_x}
\mathcal O_{X,x}^{\oplus b}
\to
\mathcal F_x
\to0,
$$

则存在 $x$ 的开邻域 $U$ 和态射

$$
\mathcal O_U^{\oplus a}
\xrightarrow{\phi}
\mathcal O_U^{\oplus b}
\to
\mathcal F|_U
\to0
$$

使其在 $x$ 的 stalk 为原表示。

**证明.** 矩阵 $\phi_x$ 的每个元素是 $\mathcal O_{X,x}$ 的 germ，可在某个邻域上选代表全纯函数。由相干层有限表示的定义，缩小邻域后 cokernel 与 $\mathcal F$ 的自然映射在 $x$ 附近为同构。证毕。

**定理 W.5（局部有限局部自由分解）.** 设 $X$ 是复维数 $n$ 的复流形，$\mathcal F$ 是相干解析层。对每个 $x\in X$，存在开邻域 $U$ 和正合列

$$
0\to\mathcal E^{-n}\to\cdots\to\mathcal E^{-1}\to
\mathcal E^0\to\mathcal F|_U\to0,
$$

其中每个 $\mathcal E^{-i}$ 是有限秩局部自由 $\mathcal O_U$-模。

**证明.** 由 W.2 和 W.3，有限 $\mathcal O_{X,x}$-模 $\mathcal F_x$ 有长度不超过 $n$ 的有限自由分解。逐步把分解中的矩阵用 germ 代表提升到某个开邻域。每一步的同调 sheaf 在 $x$ 的 stalk 为零；coherence 说明零 stalk 条件在缩小邻域后保持为零。有限多步后得到邻域上的正合复形。证毕。

**推论 W.6（局部 Ext 有限性）.** 在 W.5 的假设下，对任意相干层 $\mathcal G$，sheaf

$$
\mathcal Ext^q_{\mathcal O_X}(\mathcal F,\mathcal G)
$$

是相干层，且 $q>n$ 时为零。

**证明.** 在 W.5 的邻域上，用有限局部自由分解 $\mathcal E^\bullet\to\mathcal F$ 计算

$$
\mathcal Ext^q(\mathcal F,\mathcal G)
=
H^q(\mathcal Hom(\mathcal E^\bullet,\mathcal G)).
$$

每个 $\mathcal Hom(\mathcal E^{-i},\mathcal G)$ 相干，有限复形的 cohomology sheaf 相干。复形长度不超过 $n$，故 $q>n$ 为零。证毕。

## W.3 全局有限分解的边界

**警告 W.7.** W.5 是局部结论，不自动给出全局有限向量丛 resolution

$$
0\to E^{-n}\to\cdots\to E^0\to\mathcal F\to0.
$$

全局 resolution 需要空间有足够多向量丛或 resolution property。光滑射影代数簇有丰富线丛和向量丛；一般紧复流形不一定满足同样的全局性质。

**命题 W.8（有全局 resolution 时的 Ext 计算）.** 若 $\mathcal F$ 有有限局部自由 resolution $E^\bullet\to\mathcal F$，则

$$
R\mathcal Hom(\mathcal F,\mathcal G)
\simeq
\mathcal Hom(E^\bullet,\mathcal G)
$$

在导出范畴中成立。

**证明.** 有限局部自由 sheaf 是 $\mathcal Hom(-,\mathcal G)$ 的 derived-acyclic 输入对象：对局部自由层取 Hom 是精确地对偶张量。有限 resolution 因此可用于计算右导出 sheaf Hom。证毕。

## W.4 Serre 对偶中的使用

若 $X$ 是复维数 $n$ 的光滑 proper 空间，且 $\mathcal F$ 有有限局部自由 resolution，则一般 Serre 对偶形式可化为向量丛情形和有限复形对偶：

$$
R\Gamma(X,\mathcal F)^\vee
\simeq
R\Gamma(X,R\mathcal Hom(\mathcal F,\omega_X[n])).
$$

附录 O 已证明这个化约的同调代数部分。本附录提供局部有限 resolution 的来源，并说明全局 resolution 是额外假设。

## 练习

1. 证明 $\mathbb C\{z\}$ 是一维正则局部环，并写出有限模的长度一自由分解形式。
2. 对光滑曲线上的 skyscraper sheaf $\mathbb C_p$，写出局部自由分解。
3. 证明 W.6 中 $\mathcal Ext^q=0$ 对 $q>n$ 成立。
4. 解释为什么局部有限 resolution 不自动粘合成全局有限 resolution。
