# Lebesgue 积分、期望与不等式

有限概率表上的平均值可以逐项相加；一般概率空间没有可供枚举的样本点，平均值必须由测度积分定义。积分不仅给出一个数，还规定了何时可以相加、取极限以及交换极限与期望。本章把这些接口逐一写清，并始终排除未定义的 $+\infty-\infty$。尤其要区分三种情形：非负函数可以允许无穷积分，可积函数允许作线性运算，而逐点收敛只有在单调或受控条件下才能穿过积分号。第 6 章的概率收敛论证和第 9 章的信息密度估计都将直接调用这些接口。

## 3.1 期望与 $L^p$ 空间

**定义 3.1（期望）。** 设 $(\Omega,\mathcal F,\mathbb P)$ 为概率空间。

1. 若 $X:\Omega\to[0,\infty]$ 可测，则
   $$
   \mathbb E[X]=\int_\Omega X\,d\mathbb P\in[0,\infty].
   $$
2. 若 $X:\Omega\to\mathbb R$ 可测，令 $X^+=\max(X,0)$、$X^-=\max(-X,0)$。当 $\mathbb E[X^+]<\infty$ 与 $\mathbb E[X^-]<\infty$ 同时成立时，称 $X$ 可积，并定义
   $$
   \mathbb E[X]=\mathbb E[X^+]-\mathbb E[X^-].
   $$

可积性等价于 $\mathbb E|X|<\infty$。若正部和负部的期望都为 $+\infty$，本书不定义 $\mathbb E[X]$。

**定义 3.2（$L^p$）。** 对 $1\le p<\infty$，令

$$
L^p(\Omega,\mathcal F,\mathbb P)
=\{X:X\text{ 可测且 }\mathbb E|X|^p<\infty\}/\!\sim,
$$

其中 $X\sim Y$ 表示 $X=Y$ 几乎处处，并记 $\|X\|_p=(\mathbb E|X|^p)^{1/p}$。正文常把等价类与其任一可测版本写成同一符号；凡等式涉及 $L^p$ 对象，默认按几乎处处理解。

期望的代数性质不是形式符号规则，而来自非负简单函数逼近。

**命题 3.1（期望的线性、单调性与绝对值界）。** 设 $X,Y\in L^1$，$a,b\in\mathbb R$。则：

1. $aX+bY\in L^1$，且
   $$
   \mathbb E[aX+bY]=a\mathbb E[X]+b\mathbb E[Y];
   $$
2. 若 $X\le Y$ 几乎处处，则 $\mathbb E[X]\le\mathbb E[Y]$；
3. $|\mathbb E[X]|\le\mathbb E|X|$。

对非负可测 $U,V$ 和 $a,b\ge0$，第一条仍成立于扩展实数意义，只要右端不出现 $0\cdot\infty$；约定 $0\cdot\infty=0$ 时也可统一书写。

**证明.** 先设 $U,V\ge0$。取非负简单函数 $U_n\uparrow U$、$V_n\uparrow V$。简单函数积分的定义直接给出

$$
\int(aU_n+bV_n)\,d\mathbb P
=a\int U_n\,d\mathbb P+b\int V_n\,d\mathbb P.
$$

因为 $aU_n+bV_n\uparrow aU+bV$，单调收敛定理 EI-2a 允许两边取极限，得到非负情形的线性。若 $0\le U\le V$，则 $V=U+(V-U)$，故 $\mathbb E[V]\ge\mathbb E[U]$。

对可积 $X,Y$，点态恒等式

$$
X^++Y^++(X+Y)^-=X^-+Y^-+(X+Y)^+
$$

中的各项均可积。对两边使用刚证得的非负线性并移项，得到 $\mathbb E[X+Y]=\mathbb E[X]+\mathbb E[Y]$；数乘同理。由 $X\le Y$ 得 $0\le Y-X$，线性给出单调性。最后由 $-|X|\le X\le|X|$ 和单调性得到

$$
-\mathbb E|X|\le\mathbb E X\le\mathbb E|X|,
$$

即第三条。证毕。

**定理 3.2（Cauchy--Schwarz 不等式）。** 若 $X,Y\in L^2$，则 $XY\in L^1$，且

$$
|\mathbb E[XY]|\le \|X\|_2\|Y\|_2.
$$

特别地，因为 $\mathbf 1_\Omega\in L^2$ 且 $\|\mathbf 1_\Omega\|_2=1$，有 $L^2\subseteq L^1$。

**证明.** 点态不等式 $2|XY|\le X^2+Y^2$ 先给出 $XY\in L^1$。若 $\|Y\|_2=0$，则 $Y=0$ 几乎处处，结论成立。否则对任意 $t\in\mathbb R$，

$$
0\le\mathbb E[(X-tY)^2]
=\mathbb E[X^2]-2t\mathbb E[XY]+t^2\mathbb E[Y^2].
$$

取 $t=\mathbb E[XY]/\mathbb E[Y^2]$，得到

$$
0\le\mathbb E[X^2]-\frac{(\mathbb E[XY])^2}{\mathbb E[Y^2]}.
$$

整理即得所求。证毕。

**定义 3.3（方差与协方差）。** 若 $X,Y\in L^2$，定义

$$
\operatorname{Var}(X)=\mathbb E[(X-\mathbb E X)^2],
\qquad
\operatorname{Cov}(X,Y)=\mathbb E[(X-\mathbb E X)(Y-\mathbb E Y)].
$$

Cauchy--Schwarz 保证这些量有限。展开平方并使用命题 3.1 可得

$$
\operatorname{Var}(X)=\mathbb E[X^2]-(\mathbb E X)^2.
$$

## 3.2 概率不等式

**定理 3.3（Markov 与 Chebyshev 不等式）。** 设 $X\ge0$ 可测，$a>0$。则

$$
\mathbb P(X\ge a)\le \frac{\mathbb E[X]}a,
$$

其中右端允许为 $+\infty$。若 $Y\in L^2$，则对 $t>0$，

$$
\mathbb P(|Y-\mathbb E Y|\ge t)
\le \frac{\operatorname{Var}(Y)}{t^2}.
$$

**证明.** 点态有 $X\ge a\mathbf 1_{\{X\ge a\}}$。由期望单调性，

$$
\mathbb E[X]\ge a\mathbb P(X\ge a),
$$

这给出 Markov 不等式。对 $Y$ 令 $X=(Y-\mathbb E Y)^2$、$a=t^2$，并注意事件 $\{X\ge t^2\}$ 等于 $\{|Y-\mathbb E Y|\ge t\}$，即得 Chebyshev 不等式。证毕。

**定理 3.4（有限 Jensen 不等式）。** 设 $I\subseteq\mathbb R$ 为区间，$\varphi:I\to\mathbb R$ 为凸函数，$x_1,\ldots,x_m\in I$，$\lambda_i\ge0$ 且 $\sum_i\lambda_i=1$。则

$$
\varphi\left(\sum_{i=1}^m\lambda_i x_i\right)
\le \sum_{i=1}^m\lambda_i\varphi(x_i).
$$

**证明.** 对 $m$ 归纳。$m=1,2$ 分别是恒等式和凸性的定义。设结论对 $m-1$ 个点成立。若 $\lambda_m=1$，结论平凡；否则令

$$
\alpha=1-\lambda_m>0,
\qquad
y=\sum_{i<m}\frac{\lambda_i}{\alpha}x_i\in I.
$$

由二点凸性和归纳假设，

$$
\varphi(\alpha y+\lambda_mx_m)
\le\alpha\varphi(y)+\lambda_m\varphi(x_m)
\le\sum_{i=1}^m\lambda_i\varphi(x_i).
$$

证毕。

## 3.3 积分与极限的精确接口

以下三项是本书不重证的 Lebesgue 积分外部输入。它们的假设不可互换。

**外部输入定理 3.5（Lebesgue 收敛定理，EI-2）。** 设 $(S,\mathcal S,\mu)$ 为测度空间。

1. **EI-2a，单调收敛。** 若 $f_n:S\to[0,\infty]$ 可测且 $f_n\uparrow f$ 几乎处处，则 $f$ 可测，并且
   $$
   \int f_n\,d\mu\uparrow\int f\,d\mu.
   $$
2. **EI-2b，Fatou 引理。** 若 $f_n:S\to[0,\infty]$ 可测，则
   $$
   \int\liminf_{n\to\infty}f_n\,d\mu
   \le\liminf_{n\to\infty}\int f_n\,d\mu.
   $$
3. **EI-2c，控制收敛。** 若 $f_n:S\to\mathbb R$ 可测，$f_n\to f$ 几乎处处，并存在 $g\in L^1(\mu)$ 使 $|f_n|\le g$ 几乎处处对所有 $n$ 成立，则 $f\in L^1(\mu)$，且
   $$
   \int|f_n-f|\,d\mu\to0,
   \qquad
   \int f_n\,d\mu\to\int f\,d\mu.
   $$

这里没有假设 $\mu(S)<\infty$；只有在使用常数作为控制函数时才需要有限测度。来源与版本定位见 [SOURCES.md](SOURCES.md)。本书使用 EI-2a 定义和计算非负积分，使用 EI-2b 控制下极限，使用 EI-2c 证明支配条件下的 $L^1$ 收敛。

**推论 3.6（期望的两个稳定性接口）。** 在概率空间上：

1. 若 $X_n\to X$ 几乎处处，且存在 $Y\in L^1$ 使 $|X_n|\le Y$ 几乎处处对所有 $n$ 成立，则 $X\in L^1$，并且 $\mathbb E|X_n-X|\to0$；
2. 若 $\mathbb E|X_n-X|\to0$，则 $\mathbb E X_n\to\mathbb E X$。

**证明.** 第一条中，极限给出 $|X|\le Y$ 几乎处处，从而 $|X_n-X|\le2Y$；对 $|X_n-X|$ 使用 EI-2c。第二条由命题 3.1 的绝对值界给出

$$
|\mathbb E X_n-\mathbb E X|
\le\mathbb E|X_n-X|\to0.
$$

证毕。

## 3.4 例子：尾概率如何累加为期望

**例 3.1（整数值随机变量的尾和公式）。**

设 $X$ 取值于 $\{0,1,2,\ldots\}\cup\{+\infty\}$。对每个 $\omega$，扩展实数意义下有

$$
X(\omega)=\sum_{k=1}^{\infty}\mathbf 1_{\{X(\omega)\ge k\}}.
$$

令部分和 $S_m=\sum_{k=1}^m\mathbf 1_{\{X\ge k\}}$，则 $S_m\uparrow X$。EI-2a 给出

$$
\mathbb E[X]
=\lim_{m\to\infty}\sum_{k=1}^m\mathbb P(X\ge k)
=\sum_{k=1}^{\infty}\mathbb P(X\ge k),
$$

两边可以同时为 $+\infty$。例如若 $\mathbb P(X=k)=1/[k(k+1)]$ 对 $k\ge1$，则 $\mathbb P(X\ge k)=1/k$，故 $\mathbb E X=+\infty$。这个例子也说明：随机变量几乎处处有限并不保证可积。

## 练习

**练习 3.1.** 用 Markov 不等式证明：若 $X\in L^1$，则 $\mathbb P(|X|\ge a)\to0$ 当 $a\to\infty$。

**练习 3.2.** 对 $X\sim\operatorname{Bernoulli}(p)$，从定义计算 $\mathbb E[X]$ 与 $\operatorname{Var}(X)$。

**练习 3.3.** 若有限值随机变量 $X$ 取 $x_i\in I$ 的概率为 $\lambda_i$，从定理 3.4 推出 $\varphi(\mathbb E X)\le\mathbb E[\varphi(X)]$。

**练习 3.4.** 设 $X_n=n\mathbf 1_{(0,1/n)}$ 定义在带 Lebesgue 概率的 $(0,1)$ 上。证明 $X_n\to0$ 几乎处处，但 $\mathbb E X_n\not\to0$，并指出 EI-2c 的哪个假设失败。

**练习 3.5.** 若 $X,Y\in L^2$，证明 $\operatorname{Var}(X+Y)=\operatorname{Var}(X)+\operatorname{Var}(Y)+2\operatorname{Cov}(X,Y)$。
