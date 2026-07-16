# 熵、散度与互信息

一次观测的自信息可以写成 $-\log p(X)$，但 Shannon 熵是这一随机量的期望，只由分布决定。KL 散度比较两个分布，却不是距离；互信息则把联合分布与边缘乘积分布比较。本章始终在有限字母表上工作，因此所有和式有限；零概率项按 [NOTATION.md](NOTATION.md) 的约定处理。

## 8.1 熵与条件熵

**定义 8.1（Shannon 熵）。** 若随机变量 $X$ 取有限集合 $\mathcal X$ 值，分布为 $p_X$，定义

$$
H(X)=-\sum_{x\in\mathcal X}p_X(x)\log p_X(x).
$$

这里及第 9、10 章中的 $\log$ 均以 $2$ 为底，故熵的单位为 bit。因为 $-u\log u\ge0$ 且有限字母表上的每项有限，$H(X)\in[0,\log|\mathcal X|]$；上界将在推论 8.5 中证明。

**定义 8.2（联合熵与条件熵）。** 对有限随机变量 $(X,Y)$，定义

$$
H(X,Y)=-\sum_{x,y}p(x,y)\log p(x,y).
$$

对每个满足 $p_Y(y)>0$ 的 $y$，令 $p(x\mid y)=p(x,y)/p_Y(y)$，并定义

$$
H(X\mid Y)
=\sum_{y:p_Y(y)>0}p_Y(y)
\left[-\sum_xp(x\mid y)\log p(x\mid y)\right].
$$

零概率 $y$ 上的条件分布可以任取，不影响上式。多变量条件熵如 $H(X\mid Y,Z)$ 采用同一定义。

**命题 8.1（熵的链式法则）。** 有限随机变量满足

$$
H(X,Y)=H(Y)+H(X\mid Y).
$$

更一般地，

$$
H(X_1,\ldots,X_n)
=\sum_{k=1}^nH(X_k\mid X_1,\ldots,X_{k-1}),
$$

其中第一项解释为 $H(X_1)$。

**证明.** 只对 $p(x,y)>0$ 的项求和；此时 $p_Y(y)>0$ 且 $p(x,y)=p_Y(y)p(x\mid y)$。于是

$$
\begin{aligned}
H(X,Y)
&=-\sum_{x,y}p(x,y)\log p_Y(y)
  -\sum_{x,y}p(x,y)\log p(x\mid y)\\
&=H(Y)+H(X\mid Y).
\end{aligned}
$$

对 $n$ 归纳：把 $(X_1,\ldots,X_{n-1})$ 看成一个有限值随机变量，应用二变量公式，再使用归纳假设。证毕。

## 8.2 KL 散度与 Gibbs 不等式

**定义 8.3（KL 散度）。** 对同一有限集合 $\mathcal A$ 上的概率分布 $P,Q$，定义

$$
D(P\|Q)=\sum_{a\in\mathcal A}P(a)\log\frac{P(a)}{Q(a)}\in[0,+\infty],
$$

其中若存在 $a$ 使 $P(a)>0$、$Q(a)=0$，则相应项及整个散度为 $+\infty$。

证明非负性时必须保留对数底的常数。对 $u>0$，

$$
\log u=\frac{\ln u}{\ln2}\le\frac{u-1}{\ln2},
$$

且等号当且仅当 $u=1$。

**定理 8.2（Gibbs 不等式）。** 对有限集合上的概率分布 $P,Q$，

$$
D(P\|Q)\ge0,
$$

且等号成立当且仅当 $P=Q$。

**证明.** 若某点满足 $P(a)>0,Q(a)=0$，则 $D(P\|Q)=+\infty$，结论成立。否则令 $S=\{a:P(a)>0\}$。对 $a\in S$ 置 $r_a=Q(a)/P(a)>0$。由上面的自然对数不等式，

$$
\begin{aligned}
-D(P\|Q)
&=\sum_{a\in S}P(a)\log r_a\\
&\le\frac1{\ln2}\sum_{a\in S}P(a)(r_a-1)\\
&=\frac1{\ln2}\left(\sum_{a\in S}Q(a)-1\right)\le0.
\end{aligned}
$$

故散度非负。若 $D(P\|Q)=0$，上述两个不等式都必须取等：对每个 $a\in S$ 有 $r_a=1$，且 $Q(S)=1$。因此 $Q(a)=P(a)$ 在 $S$ 上成立，并且 $Q$ 在 $S^c$ 上为零，即 $P=Q$。反向显然。证毕。

## 8.3 互信息与条件互信息

**定义 8.4（互信息与条件互信息）。** 对有限随机变量 $(X,Y)$，定义

$$
I(X;Y)=D(P_{XY}\|P_XP_Y).
$$

由于 $p(x,y)>0$ 蕴含 $p_X(x)p_Y(y)>0$，该散度总是有限。对有限三元组 $(X,Y,Z)$，定义

$$
I(X;Y\mid Z)
=\sum_{z:p_Z(z)>0}p_Z(z)
D\!\left(P_{XY\mid Z=z}\middle\|P_{X\mid Z=z}P_{Y\mid Z=z}\right).
$$

由定理 8.2，$I(X;Y)\ge0$、$I(X;Y\mid Z)\ge0$。等号分别刻画无条件独立和对 $p_Z$-几乎每个 $z$ 的条件独立。

**定理 8.3（互信息恒等式与链式法则）。** 有限随机变量满足

$$
I(X;Y)=H(X)-H(X\mid Y)
=H(Y)-H(Y\mid X)
=H(X)+H(Y)-H(X,Y),
$$

以及

$$
I(X;Y\mid Z)=H(X\mid Z)-H(X\mid Y,Z).
$$

此外，

$$
I(X;Y,Z)=I(X;Z)+I(X;Y\mid Z).
$$

**证明.** 在 $p(x,y)>0$ 的点上展开：

$$
\begin{aligned}
I(X;Y)
&=\sum_{x,y}p(x,y)
  \log\frac{p(x,y)}{p_X(x)p_Y(y)}\\
&=\sum_{x,y}p(x,y)\log p(x\mid y)
  -\sum_xp_X(x)\log p_X(x)\\
&=H(X)-H(X\mid Y).
\end{aligned}
$$

交换 $X,Y$ 得到第二式，再结合命题 8.1 得到第三式。对每个 $p_Z(z)>0$ 的条件分布应用已证的二变量恒等式，再以 $p_Z(z)$ 加权求和，得到条件互信息公式。最后，

$$
\begin{aligned}
I(X;Z)+I(X;Y\mid Z)
&=H(X)-H(X\mid Z)
  +H(X\mid Z)-H(X\mid Y,Z)\\
&=I(X;Y,Z).
\end{aligned}
$$

证毕。

从非负性立即得到“增加条件不能增加条件熵”：

$$
H(X\mid Y,Z)\le H(X\mid Z),
\qquad
H(X\mid Y)\le H(X).
$$

## 8.4 数据处理不等式

**定义 8.5（有限 Markov 关系）。** 记 $X\to Y\to Z$，若存在条件分布 $W(z\mid y)$ 使

$$
p(x,y,z)=p(x,y)W(z\mid y)
$$

对所有 $x,y,z$ 成立。等价地，在每个正概率 $Y=y$ 下，$X$ 与 $Z$ 条件独立。

**定理 8.4（有限数据处理不等式）。** 若 $X\to Y\to Z$，则

$$
I(X;Z)\le I(X;Y).
$$

**证明.** Markov 分解给出 $I(X;Z\mid Y)=0$。对 $I(X;Y,Z)$ 按两种次序使用定理 8.3：

$$
I(X;Y,Z)=I(X;Y)+I(X;Z\mid Y)=I(X;Y),
$$

而

$$
I(X;Y,Z)=I(X;Z)+I(X;Y\mid Z)\ge I(X;Z).
$$

合并即得结论。证毕。

**推论 8.5（有限字母表熵上界）。** 若 $X$ 取值于非空有限集合 $\mathcal X$，则

$$
0\le H(X)\le\log|\mathcal X|,
$$

且上界取等当且仅当 $X$ 在 $\mathcal X$ 上均匀分布。

**证明.** 非负性来自定义。令 $U$ 为 $\mathcal X$ 上均匀分布。定理 8.2 给出

$$
0\le D(P_X\|U)
=\sum_xp_X(x)\log\bigl(p_X(x)|\mathcal X|\bigr)
=\log|\mathcal X|-H(X).
$$

Gibbs 不等式的等号条件给出上界的等号条件。证毕。

## 8.5 例子：二元对称噪声

**例 8.1（均匀输入通过二元对称噪声）。**

令 $X\sim\operatorname{Bernoulli}(1/2)$，$N\sim\operatorname{Bernoulli}(\varepsilon)$ 与 $X$ 独立，并令 $Y=X\oplus N$。则 $Y$ 仍均匀。给定 $X=x$ 后，$Y$ 的不确定性就是噪声 $N$ 的不确定性，故

$$
H(Y)=1,
\qquad
H(Y\mid X)=h_2(\varepsilon),
$$

从而

$$
I(X;Y)=1-h_2(\varepsilon).
$$

这里还没有证明这是所有输入分布中的最大值；容量最优性以及 BSC、BEC 的完整计算放在第 9 章。

## 练习

**练习 8.1.** 直接用 Gibbs 不等式证明推论 8.5，并逐步写出等号条件。

**练习 8.2.** 计算 Bernoulli$(p)$ 的熵，并求 $p=0,1,1/2$ 时的值。

**练习 8.3.** 证明 $I(X;Y)=0$ 当且仅当有限随机变量 $X,Y$ 独立。

**练习 8.4.** 从定理 8.3 和条件互信息非负性证明 $H(X\mid Y,Z)\le H(X\mid Y)$。

**练习 8.5.** 若 $Z=f(Y)$ 为 $Y$ 的确定函数，说明 $X\to Y\to Z$，并证明 $I(X;Z)\le I(X;Y)$。
