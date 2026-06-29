# 附录 R：Dolbeault 局部正合的解析骨架

## R.0 目标

卷三多处使用 Dolbeault lemma：

$$
0\to\mathcal O(E)\to
\mathcal A^{0,0}(E)\xrightarrow{\bar\partial}
\mathcal A^{0,1}(E)\to\cdots
$$

是 resolution。附录 F 把它列为输入定理。本附录进一步展开其局部解析骨架：在 polydisc 上用 Cauchy-Green 算子构造 $\bar\partial$-同伦，从而说明局部正合性到底依赖什么。

本附录不重证分布恒等式

$$
\bar\partial\left(\frac1{\pi z}\right)=\delta_0,
$$

也不展开所有边界正则性估计；这两点作为一变量复分析输入。除此之外，复形同伦和 sheaf exactness 的推导在书内完成。

## R.1 一变量 Cauchy-Green 算子

设 $D\subset\mathbb C$ 是圆盘，$D'\Subset D$ 是相对紧圆盘。对 $f\in C_c^\infty(D)$ 定义

$$
(Tf)(z)=\frac1{2\pi i}\int_D
\frac{f(\zeta)}{\zeta-z}\,
d\zeta\wedge d\bar\zeta,
\qquad z\in D'.
$$

**输入定理 R.1（Cauchy-Green 公式）.** 对 $f\in C_c^\infty(D)$，

$$
\frac{\partial}{\partial\bar z}Tf=f
$$

在 $D'$ 上成立。并且 $T:C_c^\infty(D)\to C^\infty(D')$ 连续，其中拓扑为光滑函数的 Fréchet 拓扑。

**说明.** 该公式等价于基本解恒等式

$$
\bar\partial_z\left(\frac1{2\pi i}\frac{d\zeta\wedge d\bar\zeta}{\zeta-z}\right)
$$

表示 Dirac 核。连续性来自奇核积分的标准估计。

## R.2 Polydisc 上的同伦算子

设

$$
P=D_1\times\cdots\times D_n\subset\mathbb C^n
$$

是 polydisc，$P'\Subset P$。对第 $j$ 个变量定义算子 $T_j$：固定其余变量，在第 $j$ 个变量上应用 R.1 中的 $T$。

一个光滑 $(0,q)$-形式写为

$$
\alpha=\sum_{|J|=q}\alpha_J\,d\bar z_J,
$$

其中 $J=\{j_1<\cdots<j_q\}$，且

$$
d\bar z_J=d\bar z_{j_1}\wedge\cdots\wedge d\bar z_{j_q}.
$$

定义收缩算子 $\iota_j$：

$$
\iota_j(d\bar z_J)=
\begin{cases}
(-1)^{r-1}d\bar z_{J\setminus\{j\}},& j=j_r\in J,\\
0,& j\notin J.
\end{cases}
$$

令

$$
H_j(\alpha)=T_j(\iota_j\alpha).
$$

这里 $T_j$ 只作用在系数函数上。

**引理 R.2（单变量同伦恒等式）.** 若形式系数在 $P$ 中有紧支撑，则在 $P'$ 上

$$
\bar\partial_j H_j+H_j\bar\partial_j=\Pi_j,
$$

其中 $\bar\partial_j=d\bar z_j\,\partial/\partial\bar z_j$，$\Pi_j$ 是含有 $d\bar z_j$ 因子的部分投影。

**证明.** 对单项

$$
\alpha_J d\bar z_J
$$

分别讨论 $j\in J$ 与 $j\notin J$。若 $j\notin J$，则 $\iota_j$ 杀掉该项，右侧投影也杀掉该项；等式归结为符号为零。若 $j=j_r\in J$，则

$$
H_j(\alpha_Jd\bar z_J)
=
(-1)^{r-1}T_j(\alpha_J)d\bar z_{J\setminus\{j\}}.
$$

再施加 $\bar\partial_j$ 得

$$
d\bar z_j\wedge (-1)^{r-1}
\frac{\partial T_j(\alpha_J)}{\partial\bar z_j}
d\bar z_{J\setminus\{j\}}
=
\alpha_Jd\bar z_J,
$$

符号由 $d\bar z_j$ 移到第 $r$ 位产生的 $(-1)^{r-1}$ 抵消。其余项由 $H_j\bar\partial_j$ 处理，并与 $\bar\partial_jH_j$ 中不含第 $j$ 方向求导的项相互抵消。证毕。

## R.3 逐变量消元

**定理 R.3（polydisc Dolbeault-Poincare lemma，局部形式）.** 设 $q>0$，$\alpha\in \mathcal A^{0,q}(P)$，且

$$
\bar\partial\alpha=0.
$$

则对每个点 $x\in P$，存在邻域 $V\subset P$ 和 $\beta\in\mathcal A^{0,q-1}(V)$，使

$$
\bar\partial\beta=\alpha|_V.
$$

**证明.** 取 $x\in P$ 的较小 polydisc

$$
P''\Subset P'\Subset P.
$$

选取光滑 cutoff 函数 $\chi$，使 $\chi=1$ 于 $P'$ 附近，且 $\operatorname{supp}\chi\Subset P$。把 $\alpha$ 替换为 $\chi\alpha$ 后，可在 $P'$ 上恢复原 $\alpha$。

使用变量数归纳。对最后一个变量 $z_n$，把 $\chi\alpha$ 分解为

$$
\chi\alpha=d\bar z_n\wedge\gamma+\delta,
$$

其中 $\delta$ 不含 $d\bar z_n$。令

$$
\beta_n=H_n(\chi\alpha).
$$

由引理 R.2，$\bar\partial\beta_n$ 在 $P'$ 上消去所有含 $d\bar z_n$ 的项；剩余

$$
\alpha_1=\alpha-\bar\partial\beta_n
$$

在 $P'$ 上不含 $d\bar z_n$，且仍满足 $\bar\partial\alpha_1=0$。从 $\bar\partial\alpha_1=0$ 的 $d\bar z_n$ 分量得到 $\alpha_1$ 的系数对 $\bar z_n$ 全纯。于是可把这些系数看成取值于第 $n$ 个变量全纯函数空间的光滑系数，在前 $n-1$ 个变量上应用归纳；Cauchy-Green 算子对参数连续，保证所得解仍为光滑形式。

对 $n-1$ 个变量重复此过程，得到

$$
\beta=\beta_n+\beta_{n-1}+\cdots+\beta_1
$$

在 $P''$ 上满足 $\bar\partial\beta=\alpha$。证毕。

**注 R.4.** 上述证明隐藏的解析估计是：每次使用 $T_j$ 后仍得到光滑系数，且缩小 polydisc 后所有积分表达式均在内部点有定义。这正是 R.1 连续性和 cutoff 的作用。

## R.4 带向量丛系数

设 $E$ 是复流形 $X$ 上的全纯向量丛。局部取全纯平凡化

$$
E|_U\simeq U\times\mathbb C^r.
$$

在该平凡化下

$$
\mathcal A^{0,q}(E)|_U\simeq
(\mathcal A^{0,q}_U)^{\oplus r},
$$

且 $\bar\partial_E$ 逐分量等于普通 $\bar\partial$。

**推论 R.5（带系数局部正合）.** 对 $q>0$，

$$
\ker\bigl(\bar\partial:\mathcal A^{0,q}(E)\to\mathcal A^{0,q+1}(E)\bigr)
=
\operatorname{im}\bigl(\bar\partial:\mathcal A^{0,q-1}(E)\to\mathcal A^{0,q}(E)\bigr)
$$

作为 sheaf 等式成立。

**证明.** sheaf 等式可在每点的足够小邻域检查。取全纯平凡化后，问题化为 $r$ 个标量情形；标量情形由定理 R.3 给出。证毕。

**推论 R.6（零次 kernel）.** 有

$$
\ker\bigl(\bar\partial:\mathcal A^{0,0}(E)\to\mathcal A^{0,1}(E)\bigr)
=
\mathcal O(E).
$$

**证明.** 在全纯平凡化中，$\bar\partial f=0$ 等价于每个分量函数满足 Cauchy-Riemann 方程。由复分析基本定理，光滑函数满足 $\bar\partial f=0$ 当且仅当全纯。证毕。

## R.5 与 liquid/analytic 语言的关系

Dolbeault resolution 进入 condensed/analytic 语言时，还需要记录拓扑：

1. $\Gamma(U,\mathcal A^{0,q}(E))$ 带 Fréchet 拓扑。
2. $\bar\partial$ 是连续线性映射。
3. $T_j$ 在缩小 polydisc 后是连续线性算子。
4. 局部同伦算子与 sheaf restriction 相容。

这些事实说明 Dolbeault lemma 不只是代数复形正合；它同时提供连续线性局部解算子。第三卷只在需要 sheaf cohomology 计算时使用正合性；进入 liquid 模时，还必须额外保留连续性。

## 练习

1. 在 $n=1,q=1$ 情形下，直接用 R.1 证明 $\bar\partial u=f\,d\bar z$ 的局部可解性。
2. 写出 $n=2,q=1$ 时 $H_1,H_2$ 对形式 $a\,d\bar z_1+b\,d\bar z_2$ 的作用，并验证同伦公式中的符号。
3. 说明为什么证明中需要缩小 polydisc。给出不缩小时边界积分可能出现的项。
4. 对平凡向量丛 $E=\mathcal O^r$，把推论 R.5 写成矩阵形式。
