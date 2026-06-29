# 附录 H：$\mathbb P^1$ 上线丛上同调的 Čech 计算

## H.0 目标

本附录完整计算

$$
H^0(\mathbb P^1,\mathcal O(d)),
\qquad
H^1(\mathbb P^1,\mathcal O(d)).
$$

这是第三卷中 Riemann-Roch、GAGA 和 Serre duality 的基本检验例子。与主体章节不同，本附录不给“证明路线”，而给逐项 Čech 计算。

## H.1 覆盖与转移函数

令

$$
\mathbb P^1=\operatorname{Proj}\mathbb C[X_0,X_1].
$$

取标准开覆盖

$$
U_0=\{X_0\ne0\}\cong\mathbb A^1_z,
\qquad
U_\infty=\{X_1\ne0\}\cong\mathbb A^1_w,
$$

其中

$$
z=\frac{X_1}{X_0},
\qquad
w=\frac{X_0}{X_1}=z^{-1}
$$

在交集

$$
U_0\cap U_\infty\cong\mathbb G_m
$$

上成立。

对线丛 $\mathcal O(d)$，取局部平凡化

$$
e_0=X_0^d\quad\text{on }U_0,
\qquad
e_\infty=X_1^d\quad\text{on }U_\infty.
$$

在交集上，

$$
e_\infty=X_1^d=(zX_0)^d=z^d e_0.
$$

因此若一个截面在 $U_0$ 上写成 $f_0(z)e_0$，在 $U_\infty$ 上写成 $f_\infty(w)e_\infty$，则粘合条件为

$$
f_0(z)=z^d f_\infty(1/z).
$$

## H.2 Čech 复形

覆盖只有两个开集，因此 Čech 复形为

$$
C^0=
\Gamma(U_0,\mathcal O(d))
\oplus
\Gamma(U_\infty,\mathcal O(d)),
$$

$$
C^1=
\Gamma(U_0\cap U_\infty,\mathcal O(d)),
$$

且无更高项。用 $e_0$ 在交集上平凡化，得到

$$
C^0=
\mathbb C[z]\oplus\mathbb C[w],
\qquad
C^1=
\mathbb C[z,z^{-1}].
$$

微分

$$
\delta:C^0\to C^1
$$

为

$$
\delta(f_0(z),f_\infty(w))
=
z^d f_\infty(z^{-1})-f_0(z).
$$

因此

$$
H^0(\mathbb P^1,\mathcal O(d))=\ker\delta,
\qquad
H^1(\mathbb P^1,\mathcal O(d))=\operatorname{coker}\delta.
$$

## H.3 $H^0$ 的计算

**命题 H.3.1.** 若 $d\ge0$，则

$$
H^0(\mathbb P^1,\mathcal O(d))
\cong
\{ \text{多项式 } f_0(z)\in\mathbb C[z]\mid \deg f_0\le d\}.
$$

特别地，

$$
\dim_\mathbb C H^0(\mathbb P^1,\mathcal O(d))=d+1.
$$

若 $d<0$，则

$$
H^0(\mathbb P^1,\mathcal O(d))=0.
$$

**证明.** 元素 $(f_0,f_\infty)\in C^0$ 属于 $\ker\delta$ 当且仅当

$$
f_0(z)=z^d f_\infty(z^{-1}).
$$

先设 $d\ge0$。若

$$
f_\infty(w)=\sum_{m=0}^N a_mw^m,
$$

则

$$
z^d f_\infty(z^{-1})
=
\sum_{m=0}^N a_mz^{d-m}.
$$

该表达是 $z$ 的多项式当且仅当所有出现的指数 $d-m$ 非负，即 $m\le d$。因此 $f_0$ 正是次数不超过 $d$ 的多项式。反过来，给定

$$
f_0(z)=\sum_{k=0}^d b_kz^k,
$$

令

$$
f_\infty(w)=w^d f_0(w^{-1})
=
\sum_{k=0}^d b_kw^{d-k},
$$

这是 $w$ 的多项式，并满足粘合条件。故得到同构。

若 $d<0$，则 $z^d f_\infty(z^{-1})$ 中每一项指数都为 $d-m<0$，除非 $f_\infty=0$，否则不可能等于多项式 $f_0(z)$。于是 $f_\infty=0$，再由粘合条件得 $f_0=0$。证毕。

## H.4 $H^1$ 的计算

**命题 H.4.1.** 若 $d\ge-1$，则

$$
H^1(\mathbb P^1,\mathcal O(d))=0.
$$

若 $d\le-2$，则

$$
H^1(\mathbb P^1,\mathcal O(d))
\cong
\bigoplus_{k=d+1}^{-1}\mathbb C\cdot z^k.
$$

特别地，

$$
\dim_\mathbb C H^1(\mathbb P^1,\mathcal O(d))=-d-1.
$$

**证明.** 任意 Laurent 多项式

$$
g(z)=\sum_{k=r}^s c_kz^k
\in\mathbb C[z,z^{-1}]
$$

在 $C^1$ 中。微分的像由两类项生成：

1. $-f_0(z)$ 给出所有非负指数项 $z^k$，$k\ge0$。
2. $z^d f_\infty(z^{-1})$ 给出所有指数 $d-m$，其中 $m\ge0$，即所有 $k\le d$ 的项。

因此

$$
\operatorname{im}\delta
=
\left\langle z^k\mid k\ge0\right\rangle
+
\left\langle z^k\mid k\le d\right\rangle.
$$

若 $d\ge-1$，则整数集合满足

$$
\mathbb Z=\{k\ge0\}\cup\{k\le d\},
$$

因为当 $d=-1$ 时两部分已覆盖全部整数，$d>-1$ 时更是如此。故每个 Laurent 单项式都在 $\operatorname{im}\delta$ 中，$H^1=0$。

若 $d\le-2$，则没有被覆盖的指数正是

$$
d+1,d+2,\ldots,-1.
$$

这些单项式在商中生成 $H^1$。它们线性无关，因为 $\operatorname{im}\delta$ 由指数集合 $\{k\ge0\}\cup\{k\le d\}$ 的单项式张成，与这些指数不相交。故得到所述同构。证毕。

## H.5 Euler characteristic

由 H.3.1 和 H.4.1，

若 $d\ge0$，

$$
\chi(\mathbb P^1,\mathcal O(d))
=
(d+1)-0=d+1.
$$

若 $d=-1$，

$$
\chi(\mathbb P^1,\mathcal O(-1))=0-0=0=d+1.
$$

若 $d\le-2$，

$$
\chi(\mathbb P^1,\mathcal O(d))
=
0-(-d-1)=d+1.
$$

所以对所有 $d\in\mathbb Z$，

$$
\chi(\mathbb P^1,\mathcal O(d))=d+1.
$$

这与附录 F 中的 Riemann-Roch 计算一致。

## H.6 Serre duality 检查

因为

$$
\omega_{\mathbb P^1}\cong\mathcal O(-2),
$$

Serre duality 预测

$$
H^1(\mathbb P^1,\mathcal O(d))^\vee
\cong
H^0(\mathbb P^1,\mathcal O(-d-2)).
$$

若 $d\le-2$，右侧维数为

$$
(-d-2)+1=-d-1,
$$

与 H.4.1 相同。若 $d\ge-1$，两侧均为零。故 Čech 计算与 Serre duality 相容。

更具体地，$H^1(\mathcal O(d))$ 的基可取

$$
z^{d+1},z^{d+2},\ldots,z^{-1}.
$$

而 $H^0(\mathcal O(-d-2))$ 的基可取

$$
1,z,\ldots,z^{-d-2}.
$$

留数配对把对应指数相加为 $-1$ 的项配对起来，给出非退化配对。

## 练习

**练习 H.1.** 检查本附录采用的转移函数 $e_\infty=z^de_0$ 与通常的 $\mathcal O(d)$ 定义一致。

**练习 H.2.** 对 $d=1,0,-1,-2,-3$ 分别写出 $H^0$ 和 $H^1$ 的基。

**练习 H.3.** 用 H.4.1 直接证明 $\chi(\mathbb P^1,\mathcal O(d))=d+1$。

**练习 H.4.** 写出留数配对在 $d=-3$ 时的矩阵。
