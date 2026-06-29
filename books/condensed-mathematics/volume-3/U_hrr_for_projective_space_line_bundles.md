# 附录 U：射影空间线丛的 Hirzebruch-Riemann-Roch

## U.0 目标

本附录证明 Hirzebruch-Riemann-Roch 在

$$
X=\mathbb P^n,\qquad E=\mathcal O(d)
$$

情形下的公式：

$$
\chi(\mathbb P^n,\mathcal O(d))
=
\int_{\mathbb P^n}\operatorname{ch}(\mathcal O(d))\operatorname{td}(T_{\mathbb P^n}).
$$

左侧由附录 S 的 Čech 计算给出；右侧由 Chow/cohomology 环中的形式幂级数计算给出。这个例子把 HRR 的所有符号压缩到一个可复核等式。

## U.1 Cohomology 环与积分

令

$$
H=c_1(\mathcal O(1))\in H^2(\mathbb P^n,\mathbb Z).
$$

则

$$
H^\bullet(\mathbb P^n,\mathbb Q)
\cong
\mathbb Q[H]/(H^{n+1}),
$$

且积分归一化为

$$
\int_{\mathbb P^n}H^n=1.
$$

因此对任意多项式 $P(H)$，

$$
\int_{\mathbb P^n}P(H)
$$

等于 $H^n$ 的系数。

## U.2 Chern character 与 Todd class

**命题 U.1.** 有

$$
\operatorname{ch}(\mathcal O(d))=e^{dH}.
$$

**证明.** 线丛 $L$ 的 Chern character 为

$$
\operatorname{ch}(L)=e^{c_1(L)}.
$$

而 $c_1(\mathcal O(d))=dH$。证毕。

**命题 U.2.** 有

$$
\operatorname{td}(T_{\mathbb P^n})
=
\left(\frac{H}{1-e^{-H}}\right)^{n+1}.
$$

**证明.** Euler sequence 给出

$$
0\to\mathcal O
\to
\mathcal O(1)^{\oplus(n+1)}
\to
T_{\mathbb P^n}
\to0.
$$

Todd class 对短正合列乘法：

$$
\operatorname{td}(B)=\operatorname{td}(A)\operatorname{td}(C).
$$

又 $\operatorname{td}(\mathcal O)=1$，线丛 $\mathcal O(1)$ 的 Todd class 为

$$
\frac{H}{1-e^{-H}}.
$$

于是得到公式。证毕。

## U.3 HRR 右侧的系数计算

HRR 右侧等于

$$
[H^n]\,
e^{dH}\left(\frac{H}{1-e^{-H}}\right)^{n+1}.
$$

**引理 U.3.** 对任意整数 $d$，

$$
[H^n]\,
e^{dH}\left(\frac{H}{1-e^{-H}}\right)^{n+1}
=
\binom{d+n}{n},
$$

其中右侧按多项式

$$
\frac{(d+1)(d+2)\cdots(d+n)}{n!}
$$

解释。

**证明.** 系数可写为 residue：

$$
[H^n]F(H)=\operatorname{Res}_{H=0}\frac{F(H)}{H^{n+1}}\,dH.
$$

代入

$$
F(H)=e^{dH}\left(\frac{H}{1-e^{-H}}\right)^{n+1}
$$

得

$$
\operatorname{Res}_{H=0}
\frac{e^{dH}}{(1-e^{-H})^{n+1}}\,dH.
$$

令 $u=e^H$。则 $dH=du/u$，且

$$
1-e^{-H}=1-u^{-1}=\frac{u-1}{u}.
$$

于是 residue 变为

$$
\operatorname{Res}_{u=1}
\frac{u^d}{((u-1)/u)^{n+1}}\frac{du}{u}
=
\operatorname{Res}_{u=1}
\frac{u^{d+n}}{(u-1)^{n+1}}\,du.
$$

该 residue 是 $u^{d+n}$ 在 $u=1$ 的 Taylor 展开中 $(u-1)^n$ 的系数。广义二项式展开给该系数为

$$
\binom{d+n}{n}.
$$

证毕。

## U.4 与 Čech 计算比较

**定理 U.4（$\mathbb P^n$ 线丛 HRR）.** 对所有 $d\in\mathbb Z$，

$$
\chi(\mathbb P^n,\mathcal O(d))
=
\int_{\mathbb P^n}
\operatorname{ch}(\mathcal O(d))\operatorname{td}(T_{\mathbb P^n}).
$$

**证明.** 附录 S 给出

$$
\chi(\mathbb P^n,\mathcal O(d))
=
\binom{d+n}{n}
$$

按同一多项式解释。命题 U.1、U.2 和引理 U.3 给出右侧也等于该数。证毕。

## U.5 低维检验

当 $n=1$ 时，

$$
\operatorname{td}(T_{\mathbb P^1})
=
\left(\frac{H}{1-e^{-H}}\right)^2
=
1+H
$$

在 $H^2=0$ 中成立。因此

$$
\int_{\mathbb P^1}e^{dH}(1+H)
=
[H](1+dH)(1+H)
=d+1.
$$

当 $n=2$ 时，

$$
\chi(\mathbb P^2,\mathcal O(d))
=
\frac{(d+1)(d+2)}2.
$$

例如 $d=-1,-2$ 时结果为 $0$，与附录 S 中所有上同调消失一致；$d=-3$ 时结果为 $1$，对应 $H^2(\mathcal O(-3))$ 一维。

## 练习

1. 直接展开 $n=2$ 时的 Todd class，验证 $H^2$ 系数给出 $\frac{(d+1)(d+2)}2$。
2. 对 $n=3$，用 residue 方法计算 $\int e^{dH}\operatorname{td}(T_{\mathbb P^3})$。
3. 解释为什么 Euler sequence 足以计算 $T_{\mathbb P^n}$ 的 Todd class。
4. 说明本附录证明的 HRR 情形为何不能替代一般 proper smooth variety 的 HRR。
