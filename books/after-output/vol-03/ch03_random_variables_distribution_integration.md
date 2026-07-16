# 第三章 随机变量、分布与积分

随机变量是从结果空间到可测值空间的函数，分布是原测度沿该函数的推前，期望则是相对于该分布或原测度的积分。这三个对象常在自然语言中被混为“一个随机数”，但它们携带不同的定义域、可测结构与等价关系。

区分函数本身和它的分布，才能说明两个实现何时同分布却不逐点相同，也才能判断一次观测究竟支持样本值陈述、总体期望陈述还是积分估计。本章据此建立后续收敛、校准与复现判据共同使用的语言。

## 第二章：随机变量、分布与积分

在同一个两次掷骰实验中，可以记录第一个点数、两次点数之和，也可以只记录二者是否
相等。这些结果拥有共同的样本空间，却是不同的函数。概率论把“随机量”放在这些函数
上，而不是把某个数称为天生随机。

第一章已经保证可测函数能把结果事件拉回样本空间。本章进一步把概率沿函数推到结果
空间，并通过 Lebesgue 积分汇总结果。非负函数积分的构造性质作为测度论输入使用；
从该输入到期望、矩和尾界的推导则在正文中完成。

### P2.1 随机变量与分布


**定义 P2.1（随机元素）.** 设 $(\Omega,\mathcal F,\mathbb P)$ 是概率空间，$(S,\mathcal S)$ 是可测空间。可测映射

$$
X:(\Omega,\mathcal F)\longrightarrow(S,\mathcal S)
$$

称为取值于 $S$ 的随机元素。$S=\mathbb R$ 时称实随机变量。

随机变量不是“会自行变化的数”，而是从样本点到结果的函数。随机性由定义域上的 $\mathbb P$ 提供。

**定义 P2.2（分布）.** $X$ 的分布是 $S$ 上的推前概率测度

$$
\mathcal L(X)=\mathbb P_X\coloneqq\mathbb P\circ X^{-1},
\qquad
\mathbb P_X(B)=\mathbb P(X\in B),\quad B\in\mathcal S.
$$

两个随机变量可以定义在不同概率空间上却有相同分布。相同分布不意味着它们是同一个函数，也不规定二者的联合分布。

**定义 P2.3（分布函数）.** 对实随机变量 $X$，定义

$$
F_X(x)=\mathbb P(X\le x),\qquad x\in\mathbb R.
$$

$F_X$ 的这些解析性质不是额外的光滑性假设，而是概率测度连续性的结果。

**命题 P2.4（分布函数的基本性质）.** 对每个实随机变量 $X$，$F_X$ 单调不减、右连续，并满足

$$
\lim_{x\to-\infty}F_X(x)=0,
\qquad
\lim_{x\to+\infty}F_X(x)=1.
$$

**证明.** 若 $x\le y$，则 $\{X\le x\}\subseteq\{X\le y\}$，故单调性来自概率测度的单调性。若 $x_n\downarrow x$，则

$$
\{X\le x_n\}\downarrow\{X\le x\};
$$

概率测度从上连续，故 $F_X(x_n)\downarrow F_X(x)$。取 $x_n=x+1/n$ 即得右连续性。最后，事件 $\{X\le n\}$ 随 $n\to\infty$ 递增到 $\Omega$，而 $\{X\le-n\}$ 递减到空集，因为 $X$ 只取有限实值。分别使用从下、从上连续性即得两个极限。证毕。

**定义 P2.5（密度）.** 设 $(S,\mathcal S,\lambda)$ 是 $\sigma$-有限测度空间。若 $\mathbb P_X\ll\lambda$，则由 Radon--Nikodym 定理存在 $\mathcal S$-可测函数 $p:S\to[0,\infty]$，使

$$
\mathbb P_X(B)=\int_Bp\,d\lambda.
$$

称 $p$ 为相对于 $\lambda$ 的密度。因为 $\int p\,d\lambda=1$，$p$ 在 $\lambda$-几乎处处有限；它也只在 $\lambda$-几乎处处意义下唯一。离散分布相对于计数测度有密度；Dirac 分布相对于 Lebesgue 测度没有密度。没有指定参考测度或没有验证绝对连续性时，符号“$p(x)$”不具有密度的含义。

### P2.2 Lebesgue 积分与期望


**外部输入 P2.6（Lebesgue 积分）.** 对测度空间 $(S,\mathcal S,\mu)$，存在非负可测函数的积分 $\int f\,d\mu\in[0,\infty]$，满足单调性、简单函数逼近、单调收敛定理和 Fatou 引理。对实函数 $f=f^+-f^-$，若至少一个 $\int f^+d\mu,\int f^-d\mu$ 有限，则可定义扩展积分；若二者都有限，则称 $f$ 可积。支配收敛定理将在第五章以所需版本重述。该构造见 [PROBABILITY_SOURCES.md](PROBABILITY_SOURCES.md) 中的标准测度论资料，本书不重建 Caratheodory 扩张与全部积分理论。

**定义 P2.7（期望）.** 对实随机变量 $X$，若 $X$ 的扩展积分有定义，记

$$
\mathbb E[X]=\int_\Omega X(\omega)\,d\mathbb P(\omega).
$$

若 $\mathbb E|X|<\infty$，称 $X$ 可积，记 $X\in L^1(\mathbb P)$。若 $\mathbb E[X^2]<\infty$，称平方可积，记 $X\in L^2(\mathbb P)$。

若 $\mathbb E[X^+]=\mathbb E[X^-]=\infty$，则 $\mathbb E[X]$ 未定义，不能以“$\infty-\infty=0$”补定义。

**定理 P2.8（推前积分公式）.** 设 $X:(\Omega,\mathcal F)\to(S,\mathcal S)$ 是随机元素，$g:S\to\overline{\mathbb R}$ 可测。若 $g\ge0$，或 $g\circ X$ 可积，则

$$
\mathbb E[g(X)]=\int_Sg(x)\,d\mathbb P_X(x).
$$

**证明.** 若 $g=\mathbf1_B$，等式是推前测度定义：

$$
\int_\Omega\mathbf1_B(X(\omega))\,d\mathbb P
=\mathbb P(X\in B)=\mathbb P_X(B).
$$

由有限线性性，等式对非负简单函数成立。对非负可测 $g$，取递增简单函数 $g_n\uparrow g$，则 $g_n\circ X\uparrow g\circ X$，两侧分别用单调收敛定理取极限。对可积 $g\circ X$，分别应用于 $g^+$ 和 $g^-$ 后相减。证毕。

这一定理允许计算时离开可能十分复杂的样本空间：只要知道 $X$ 的分布，就可以在
结果空间上积分。证明从示性函数开始，是因为示性函数恰好把积分还原成事件概率；
简单函数逼近再把这一事实扩展到一般可测函数。

### P2.3 矩与方差


**定义 P2.9（矩与方差）.** 对整数 $k\ge1$，若 $X^k\in L^1$，定义 $k$ 阶原点矩；若 $X\in L^2$，定义方差：

$$
m_k=\mathbb E[X^k],
\qquad
\operatorname{Var}(X)=\mathbb E[(X-\mathbb E X)^2].
$$

**命题 P2.10（方差恒等式）.** 若 $X\in L^2$，则

$$
\operatorname{Var}(X)=\mathbb E[X^2]-(\mathbb E X)^2.
$$

并且对 $a,b\in\mathbb R$，

$$
\operatorname{Var}(aX+b)=a^2\operatorname{Var}(X).
$$

**证明.** 令 $m=\mathbb E X$。平方展开并用期望线性性：

$$
\mathbb E[(X-m)^2]=\mathbb E[X^2]-2m\mathbb E[X]+m^2
=\mathbb E[X^2]-m^2.
$$

第二式由 $aX+b-\mathbb E[aX+b]=a(X-m)$ 得到。所有项有限：逐点不等式 $|X|\le1+X^2$ 给出 $L^2\subseteq L^1$；而 $2|XY|\le X^2+Y^2$ 保证下文协方差中的乘积可积。证毕。

**定义 P2.11（协方差）.** 若 $X,Y\in L^2$，定义

$$
\operatorname{Cov}(X,Y)=\mathbb E[(X-\mathbb EX)(Y-\mathbb EY)].
$$

协方差为零不蕴含独立；例如令 $X$ 在 $\{-1,0,1\}$ 上均匀，$Y=X^2$，则 $\mathbb E[XY]=\mathbb E[X^3]=0$ 且 $\mathbb EX=0$，所以协方差为零，但 $Y$ 是 $X$ 的函数。

**例（一个三点分布的完整计算）.** 令 $X$ 以概率 $1/4,1/2,1/4$ 分别取
$0,1,2$。相对于计数测度，它的密度就是质量函数

$$
p(0)=\frac14,
\qquad p(1)=\frac12,
\qquad p(2)=\frac14.
$$

由推前积分公式，

$$
\mathbb E X=0\cdot\frac14+1\cdot\frac12+2\cdot\frac14=1,
$$

而

$$
\mathbb E[X^2]
=0^2\cdot\frac14+1^2\cdot\frac12+2^2\cdot\frac14
=\frac32.
$$

所以 $\operatorname{Var}(X)=3/2-1=1/2$。这里每一步只使用分布；原概率空间上
哪些样本点映到 $0,1,2$，并不影响这些单变量数值。

### P2.4 基本不等式


矩把整个分布压缩成少数数值。压缩会丢失信息，但仍能控制某些事件：Markov
不等式把一个非负变量的均值转换成尾概率上界，Chebyshev 不等式再把这一思路用于
中心化平方。

**定理 P2.12（Markov 不等式）.** 若 $X:\Omega\to[0,\infty]$ 可测且 $a>0$，则

$$
\mathbb P(X\ge a)\le\frac{\mathbb E[X]}{a},
$$

其中允许 $\mathbb E[X]=\infty$，此时右侧为 $\infty$，结论正确但无信息量。

**证明.** 逐点有 $X\ge a\mathbf1_{\{X\ge a\}}$。积分单调性给出

$$
\mathbb E[X]\ge a\mathbb P(X\ge a).
$$

除以正数 $a$ 即得。证毕。

**推论 P2.13（Chebyshev 不等式）.** 若 $X\in L^2$，则对每个 $\varepsilon>0$，

$$
\mathbb P(|X-\mathbb EX|\ge\varepsilon)
\le\frac{\operatorname{Var}(X)}{\varepsilon^2}.
$$

**证明.** 对非负随机变量 $(X-\mathbb EX)^2$ 和阈值 $\varepsilon^2$ 应用 Markov 不等式。证毕。

**定理 P2.14（有限 Jensen 不等式）.** 设 $I\subseteq\mathbb R$ 是区间，$\varphi:I\to\mathbb R$ 为凸函数，$x_1,\ldots,x_n\in I$，$\lambda_i\ge0$ 且 $\sum_i\lambda_i=1$。则

$$
\varphi\left(\sum_{i=1}^n\lambda_ix_i\right)
\le\sum_{i=1}^n\lambda_i\varphi(x_i).
$$

**证明.** $n=1$ 平凡，$n=2$ 是凸性定义。假设结论对 $n-1$ 成立。若 $\lambda_n=1$，结论平凡；否则令 $s=1-\lambda_n>0$ 与 $y=s^{-1}\sum_{i<n}\lambda_ix_i$。归纳假设给出

$$
\varphi(y)\le s^{-1}\sum_{i<n}\lambda_i\varphi(x_i).
$$

再对 $sy+\lambda_nx_n$ 使用二点凸性并代入上式，得到结论。证毕。

一般积分型 Jensen 不等式需要凸函数的支撑线或逼近论。后文使用时会明确其可积性条件。

### P2.5 同分布与耦合


到目前为止，单个随机变量的期望只由它的分布决定。一旦问题同时涉及 $X$ 与 $Y$，
两个边缘分布通常不够；还需要说明二者如何放在同一个概率空间上。

**定义 P2.15（耦合）.** 给定可测空间 $(S,\mathcal S)$ 上的概率测度 $\mu,\nu$，一个 $(\mu,\nu)$-耦合是 $(S\times S,\mathcal S\otimes\mathcal S)$ 上边缘分布分别为 $\mu,\nu$ 的概率测度 $\pi$。同一对边缘分布通常有许多耦合。

如果只知道实随机变量 $X\sim\mu$ 与 $Y\sim\nu$，则通常无法计算 $\mathbb P(X=Y)$ 或 $\mathbb E[XY]$；前者还要求对角线属于乘积 $\sigma$-代数，后者还要求乘积可积。这些量依赖联合分布。把“两个采样来自同一模型”解释成“它们独立”也是额外假设。

耦合把两个边缘分布放进同一个联合模型，但并不唯一。第三章将研究一个特别重要的
联合结构：当联合分布恰好分解为边缘分布的乘积时，我们称随机变量独立。

### 练习


**练习 P2.1.** 设 $X$ 在 $\{0,1,2\}$ 上的概率分别为 $1/4,1/2,1/4$。计算 $\mathbb EX$ 与 $\operatorname{Var}(X)$。

**练习 P2.2.** 给出两个同分布但不几乎处处相等的随机变量。

**练习 P2.3.** 证明若 $X=Y$ 几乎处处，则 $\mathcal L(X)=\mathcal L(Y)$；说明逆命题为何不成立。

**练习 P2.4.** 用 Markov 不等式证明：若 $X\ge0$ 且 $\mathbb EX=0$，则 $X=0$ 几乎处处。

**练习 P2.5.** 为两个 Bernoulli$(1/2)$ 边缘分布构造独立、完全相同和完全相反三种耦合。

## 分布与统计量接口

### S8.2 期望、方差与基本不等式


若 $X$ 可积，期望为 $\mathbb E[X]$。基本不等式给出从期望到概率的有限桥梁：

$$
\mathbb P(X\ge a)\le \frac{\mathbb E[X]}{a}\quad (X\ge0),
$$

$$
\mathbb P(|X-\mathbb E X|\ge \epsilon)
\le
\frac{\operatorname{Var}(X)}{\epsilon^2}.
$$

这些结论提醒我们：平均表现不等于单次保证。模型在评测集上的平均正确率不能推出当前回答正确，只能在明确抽样模型下给出统计陈述。
