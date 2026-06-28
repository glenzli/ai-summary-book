# 附录 D：Dolbeault 与 Serre 对偶计算模型

## D.0 目标

本附录给出 Dolbeault 复形和 Serre 对偶中最常用的计算模型。主体章节给出定理形式，本附录补上局部公式、配对和类型检查。

## D.1 Dolbeault 微分

在复坐标

$$
z_j=x_j+iy_j
$$

下，

$$
d=\partial+\bar\partial,
$$

其中

$$
\bar\partial f=
\sum_j\frac{\partial f}{\partial\bar z_j}d\bar z_j.
$$

对 $(p,q)$-形式

$$
\alpha=\sum_I f_I dz_{i_1}\wedge\cdots\wedge dz_{i_p}
\wedge d\bar z_{j_1}\wedge\cdots\wedge d\bar z_{j_q},
$$

定义

$$
\bar\partial\alpha=
\sum_{I,k}\frac{\partial f_I}{\partial\bar z_k}
d\bar z_k\wedge dz_{i_1}\wedge\cdots\wedge d\bar z_{j_q}.
$$

**命题 D.1.** $\bar\partial^2=0$。

**证明.** 对函数 $f$，

$$
\bar\partial^2 f
=
\sum_{i,j}
\frac{\partial^2 f}{\partial\bar z_i\partial\bar z_j}
d\bar z_i\wedge d\bar z_j.
$$

二阶偏导数对 $i,j$ 对称，而 $d\bar z_i\wedge d\bar z_j$ 反对称，因此求和为零。对一般形式，由 Leibniz 规则推出。证毕。

## D.2 Dolbeault resolution

对全纯向量丛 $E$，有复形

$$
0\to\mathcal O(E)\to\mathcal A^{0,0}(E)
\xrightarrow{\bar\partial}
\mathcal A^{0,1}(E)
\to\cdots.
$$

**输入定理 D.2（Dolbeault lemma）.** 该复形是 $\mathcal O(E)$ 的 resolution。

因此

$$
R\Gamma(X,\mathcal O(E))
\simeq
\Gamma(X,\mathcal A^{0,\bullet}(E)).
$$

## D.3 Serre 配对的局部形式

设 $\dim_\mathbb C X=n$。若 $\alpha$ 是 $E$ 值 $(0,i)$-形式，$\beta$ 是 $E^\vee\otimes\omega_X$ 值 $(0,n-i)$-形式，则收缩后得到 $(n,n)$-形式

$$
\langle\alpha,\beta\rangle.
$$

定义配对

$$
(\alpha,\beta)\mapsto
\int_X\langle\alpha,\beta\rangle.
$$

**命题 D.3.** 该配对与 $\bar\partial$ 相容：

$$
\int_X\langle\bar\partial\alpha,\beta\rangle
 =
(-1)^{i+1}
\int_X\langle\alpha,\bar\partial\beta\rangle.
$$

**证明说明.** 由 Leibniz 规则，

$$
\bar\partial\langle\alpha,\beta\rangle
=
\langle\bar\partial\alpha,\beta\rangle
 +(-1)^i\langle\alpha,\bar\partial\beta\rangle.
$$

对紧无边界 $X$，Stokes 定理给出

$$
\int_X\bar\partial\langle\alpha,\beta\rangle=0.
$$

移项即得公式。

## D.4 Riemann surface 例子

若 $X$ 是紧 Riemann surface，则 $n=1$。对线丛 $L$，Serre duality 给出

$$
H^0(X,L)^\vee
\cong
H^1(X,L^\vee\otimes\omega_X),
$$

以及

$$
H^1(X,L)^\vee
\cong
H^0(X,L^\vee\otimes\omega_X).
$$

对 $L=\mathcal O_X$，

$$
H^1(X,\mathcal O_X)^\vee
\cong
H^0(X,\omega_X).
$$

这说明 genus $g$ 同时等于

$$
\dim H^1(X,\mathcal O_X)
=
\dim H^0(X,\omega_X).
$$

## D.5 Liquid 类型检查

Dolbeault 复形中的空间

$$
\Gamma(X,\mathcal A^{0,q}(E))
$$

通常是 Fréchet 空间。第三卷的 condensed/analytic 表述要求把它看作 liquid 向量空间。微分

$$
\bar\partial:\Gamma(X,\mathcal A^{0,q}(E))
\to
\Gamma(X,\mathcal A^{0,q+1}(E))
$$

必须是 liquid 范畴中的态射。

## D.6 本附录小结

Dolbeault 计算提供相干上同调的具体复形；Serre 配对通过积分和 Stokes 定理与该复形相容。condensed/analytic 语言的额外工作，是把这些拓扑向量空间和连续算子放入 liquid 派生范畴。

## 练习

**练习 D.1.** 在 $\mathbb C$ 上直接计算 $\bar\partial^2f=0$。

**练习 D.2.** 对紧 Riemann surface，写出 $\mathcal O_X$ 的 Serre duality。

**练习 D.3.** 证明命题 D.3 的符号公式。
