# 附录 D 概率与决策证明内核

本附录给出本书讨论不确定性、评分、校准、解码和有限决策时所需的最小概率与决策核。适用范围是可测概率空间、有限或可积随机变量、有限分布上的信息量与评分规则、有限行动集合上的 Bayes 决策，以及有限词表自回归分布。它不替代完整测度论、渐近统计、因果推断或决策理论课程。

先修包括：集合与函数、有限求和、基本积分记号、凸函数、对数函数和条件期望的初步语言。所有概率陈述都必须绑定到样本空间、随机变量和信息条件；一个数值概率不自动说明现实事实、因果关系或规范责任。

## D.1 概率空间、随机变量与推前

**定义 D.1（概率空间）.** 概率空间是三元组

$$
(\Omega,\mathcal F,\mathbb P),
$$

其中 $\Omega$ 是样本空间，$\mathcal F$ 是其上的 $\sigma$-代数，$\mathbb P:\mathcal F\to[0,1]$ 满足 $\mathbb P(\Omega)=1$ 且对两两不交的可数事件族 $(A_i)$ 有

$$
\mathbb P\left(\bigcup_i A_i\right)=\sum_i\mathbb P(A_i).
$$

**定义 D.2（随机变量）.** 从 $(\Omega,\mathcal F)$ 到可测空间 $(E,\mathcal E)$ 的随机变量是可测映射

$$
X:\Omega\to E,
$$

即对所有 $B\in\mathcal E$，有 $X^{-1}(B)\in\mathcal F$。

**定义 D.3（推前分布）.** 随机变量 $X$ 的分布是 $\mathbb P$ 在 $X$ 下的推前测度

$$
X_\#\mathbb P(B)=\mathbb P(X^{-1}(B))
=\mathbb P(X\in B).
$$

若 $E$ 是有限集合，则分布由质量函数

$$
p_X(e)=\mathbb P(X=e)
$$

给出。

**边界 D.4.** “模型有概率”“答案有概率”“事件有概率”都不是完整句子。必须说明 $\Omega$ 是采样随机流、训练样本、用户总体、服务端路由、评测抽样、观察者信息状态，还是其它对象。不同样本空间上的概率不能直接比较。

## D.2 Markov、Chebyshev 与 Jensen 不等式

**定理 D.5（Markov 不等式）.** 若 $X\ge0$ 且 $\mathbb E[X]<\infty$，则对任意 $a>0$，

$$
\mathbb P(X\ge a)\le \frac{\mathbb E[X]}{a}.
$$

**证明.** 对每个 $\omega$，有逐点不等式

$$
a\,\mathbf 1_{\{X\ge a\}}(\omega)\le X(\omega).
$$

两边取期望：

$$
a\,\mathbb P(X\ge a)
=\mathbb E[a\mathbf 1_{\{X\ge a\}}]
\le \mathbb E[X].
$$

除以 $a$ 即得结论。证毕。

**定理 D.6（Chebyshev 不等式）.** 若 $X$ 有有限均值 $\mu$ 和有限方差 $\sigma^2$，则对任意 $\epsilon>0$，

$$
\mathbb P(|X-\mu|\ge\epsilon)
\le
\frac{\sigma^2}{\epsilon^2}.
$$

**证明.** 令

$$
Y=(X-\mu)^2.
$$

则 $Y\ge0$ 且 $\mathbb E[Y]=\sigma^2$。事件 $|X-\mu|\ge\epsilon$ 等同于 $Y\ge\epsilon^2$。对 $Y$ 应用 Markov 不等式：

$$
\mathbb P(|X-\mu|\ge\epsilon)
=\mathbb P(Y\ge\epsilon^2)
\le \frac{\mathbb E[Y]}{\epsilon^2}
=\frac{\sigma^2}{\epsilon^2}.
$$

证毕。

**定理 D.7（Jensen 不等式）.** 设 $I\subset\mathbb R$ 是开区间，$\varphi:I\to\mathbb R$ 为凸函数，随机变量 $X$ 取值于 $I$，且 $X$ 与 $\varphi(X)$ 可积。若 $\mu=\mathbb E[X]\in I$，则

$$
\varphi(\mathbb E[X])\le \mathbb E[\varphi(X)].
$$

**证明.** 凸函数在开区间内的每一点都有支撑直线：存在 $g\in\mathbb R$ 使对所有 $x\in I$，

$$
\varphi(x)\ge \varphi(\mu)+g(x-\mu).
$$

把 $x$ 换成 $X(\omega)$，得到逐点不等式

$$
\varphi(X)\ge \varphi(\mu)+g(X-\mu).
$$

两边取期望：

$$
\mathbb E[\varphi(X)]
\ge
\varphi(\mu)+g(\mathbb E[X]-\mu)
=
\varphi(\mu).
$$

证毕。

**反例 D.8（平均不推出单次保证）.** 若一个系统在 $100$ 个同分布任务上的平均损失为 $0.01$，Markov 不等式只能给出诸如

$$
\mathbb P(\text{损失}\ge0.1)\le0.1
$$

这类依赖抽样模型的概率界。它不能推出下一次具体运行损失小于 $0.1$，也不能推出高风险样本没有集中在某个子群体。

## D.3 独立性与弱大数律

**定义 D.9（独立事件与随机变量）.** 事件 $A,B\in\mathcal F$ 独立，若

$$
\mathbb P(A\cap B)=\mathbb P(A)\mathbb P(B).
$$

事件族 $(A_i)_{i\in I}$ 相互独立，若任意有限子族 $i_1,\ldots,i_k$ 满足

$$
\mathbb P(A_{i_1}\cap\cdots\cap A_{i_k})
=
\prod_{r=1}^k\mathbb P(A_{i_r}).
$$

随机变量族 $(X_i)$ 独立，若它们生成的 $\sigma$-代数相互独立；等价地，在有限取值情形下，任意取值组合的联合概率分解为边缘概率乘积。

**反例 D.10（两两独立不等于相互独立）.** 令 $U,V$ 为独立公平比特，取值于 $\{0,1\}$，并令

$$
W=U\oplus V
$$

为异或。则 $U,V,W$ 中任意两个都是独立公平比特。但三者不相互独立，因为

$$
U\oplus V\oplus W=0
$$

恒成立，联合取值只落在四个三元组上，而不是八个三元组上。

**定理 D.11（弱大数律，有限方差版）.** 设 $X_1,X_2,\ldots$ 两两独立，且同分布，满足

$$
\mathbb E[X_i]=\mu,\qquad \operatorname{Var}(X_i)=\sigma^2<\infty.
$$

令样本均值

$$
\bar X_n=\frac1n\sum_{i=1}^n X_i.
$$

则对任意 $\epsilon>0$，

$$
\mathbb P(|\bar X_n-\mu|\ge\epsilon)\to0.
$$

**证明.** 由线性性，

$$
\mathbb E[\bar X_n]=\mu.
$$

由于两两独立，协方差项为零：

$$
\operatorname{Var}(\bar X_n)
=
\operatorname{Var}\left(\frac1n\sum_{i=1}^n X_i\right)
=
\frac1{n^2}\sum_{i=1}^n\operatorname{Var}(X_i)
=
\frac{\sigma^2}{n}.
$$

对 $\bar X_n$ 应用 Chebyshev 不等式：

$$
\mathbb P(|\bar X_n-\mu|\ge\epsilon)
\le
\frac{\operatorname{Var}(\bar X_n)}{\epsilon^2}
=
\frac{\sigma^2}{n\epsilon^2}.
$$

右侧随 $n\to\infty$ 收敛到 $0$。证毕。

**边界 D.12.** 独立性是联合分布的分解性质，不是因果无关。弱大数律说明样本均值在概率意义下接近期望；它不说明单个样本正确，不说明几乎处处收敛，不处理重尾无限方差情形，也不消除评测集与部署分布不一致的问题。

## D.4 KL 非负与对数评分的严格适当性

设 $\mathcal Y$ 是有限集合。分布 $p,q$ 满足 $p_y\ge0$、$q_y\ge0$ 且总和为 $1$。若存在 $y$ 使 $p_y>0$ 但 $q_y=0$，定义

$$
D_{\mathrm{KL}}(p\Vert q)=+\infty.
$$

否则

$$
D_{\mathrm{KL}}(p\Vert q)
=
\sum_{y\in\mathcal Y}p_y\log\frac{p_y}{q_y},
$$

其中 $0\log(0/q)$ 约定为 $0$。

**定理 D.13（KL 非负性）.** 对有限分布 $p,q$，

$$
D_{\mathrm{KL}}(p\Vert q)\ge0.
$$

若 $q_y>0$ 对所有 $p_y>0$ 成立，则等号成立当且仅当 $p=q$。

**证明.** 若存在 $p_y>0,q_y=0$，结论由定义成立。否则在集合 $S=\{y:p_y>0\}$ 上计算：

$$
\begin{aligned}
-D_{\mathrm{KL}}(p\Vert q)
&=
\sum_{y\in S}p_y\log\frac{q_y}{p_y}\\
&\le
\log\left(\sum_{y\in S}p_y\frac{q_y}{p_y}\right)
\qquad\text{(Jensen 不等式)}\\
&=
\log\left(\sum_{y\in S}q_y\right)
\le
\log 1
=0.
\end{aligned}
$$

所以 $D_{\mathrm{KL}}(p\Vert q)\ge0$。若等号成立，Jensen 的等号要求 $q_y/p_y$ 在 $S$ 上为常数，同时最后一步要求 $\sum_{y\in S}q_y=1$。于是 $q$ 在 $S$ 外无质量，且在 $S$ 上由归一化可得 $q_y=p_y$。反之 $p=q$ 时 KL 为 $0$。证毕。

**定义 D.14（对数评分）.** 预测者报告分布 $q$，真实结果 $Y\sim p$。对数损失为

$$
L(q,Y)=-\log q_Y.
$$

期望对数损失为

$$
\mathbb E_p[L(q,Y)]
=
-\sum_y p_y\log q_y.
$$

**定理 D.15（对数评分严格适当）.** 若允许报告所有在 $p$ 的支撑上为正的分布 $q$，则期望对数损失在 $q=p$ 处唯一最小。

**证明.** 写

$$
H(p)=-\sum_y p_y\log p_y.
$$

则

$$
\begin{aligned}
\mathbb E_p[-\log q_Y]
&=
-\sum_y p_y\log q_y\\
&=
-\sum_y p_y\log p_y
+\sum_y p_y\log\frac{p_y}{q_y}\\
&=
H(p)+D_{\mathrm{KL}}(p\Vert q).
\end{aligned}
$$

由 KL 非负性，该值至少为 $H(p)$，且等号当且仅当 $q=p$。证毕。

**边界 D.16.** 严格适当性说的是：若真实数据分布为 $p$，以期望对数损失评价时，诚实报告 $p$ 最优。它不说明预测者知道 $p$，不说明样本有限时经验最小者等于 $p$，不说明被预测事件就是用户真正关心的事件。

## D.5 Brier 分解

考虑二元结果 $Y\in\{0,1\}$ 与预测 $F\in[0,1]$。Brier 损失为

$$
(F-Y)^2.
$$

设

$$
m(F)=\mathbb E[Y\mid F],
\qquad
\bar y=\mathbb E[Y].
$$

**定理 D.17（Brier 分解）.** 若 $F$ 只取有限多个值，则

$$
\mathbb E[(F-Y)^2]
=
\mathbb E[(F-m(F))^2]
+\mathbb E[m(F)(1-m(F))].
$$

并且

$$
\mathbb E[(F-Y)^2]
=
\underbrace{\bar y(1-\bar y)}_{\text{不确定性}}
-
\underbrace{\mathbb E[(m(F)-\bar y)^2]}_{\text{分辨率}}
+
\underbrace{\mathbb E[(F-m(F))^2]}_{\text{可靠性}}.
$$

**证明.** 写

$$
F-Y=(F-m(F))+(m(F)-Y).
$$

平方并取条件于 $F$ 的期望。交叉项为

$$
\mathbb E[(F-m(F))(m(F)-Y)\mid F]
=(F-m(F))\mathbb E[m(F)-Y\mid F]
=0.
$$

因此

$$
\mathbb E[(F-Y)^2\mid F]
=(F-m(F))^2+\mathbb E[(m(F)-Y)^2\mid F].
$$

在给定 $F$ 后，$Y$ 是均值为 $m(F)$ 的 Bernoulli 变量，所以

$$
\mathbb E[(m(F)-Y)^2\mid F]=m(F)(1-m(F)).
$$

再取期望得到第一式。

第二式来自全方差公式。因为 $Y$ 是 Bernoulli 变量，

$$
\operatorname{Var}(Y)=\bar y(1-\bar y).
$$

又

$$
\operatorname{Var}(Y)
=
\mathbb E[\operatorname{Var}(Y\mid F)]
+\operatorname{Var}(\mathbb E[Y\mid F])
=
\mathbb E[m(F)(1-m(F))]
+\mathbb E[(m(F)-\bar y)^2].
$$

移项并代入第一式，得到第二式。证毕。

**边界 D.18.** Brier 分解中的可靠性、分辨率和不确定性依赖被预测的二元事件与分组变量 $F$。连续预测要使用条件期望或分箱近似。Brier 分数低不自动说明解释正确、来源可靠或行动可接受；它只评价声明事件上的概率预测质量。

## D.6 有限 Bayes 行动

设状态集合 $\Theta$ 与行动集合 $\mathcal A$ 都有限。给定后验分布 $p(\theta)$ 与损失函数

$$
L:\mathcal A\times\Theta\to\mathbb R,
$$

行动 $a$ 的后验风险为

$$
R(a\mid p)=\sum_{\theta\in\Theta}p(\theta)L(a,\theta).
$$

**定义 D.19（Bayes 行动）.** 若

$$
a^*\in\operatorname*{arg\,min}_{a\in\mathcal A}R(a\mid p),
$$

则称 $a^*$ 为给定后验和损失下的 Bayes 行动。

**定理 D.20（有限 Bayes 行动存在）.** 若 $\mathcal A$ 有限且损失有限，则 Bayes 行动存在。若允许随机化行动，则至少存在一个确定性 Bayes 行动不劣于任何随机化行动。

**证明.** 有限集合 $\mathcal A$ 上的实数集合 $\{R(a\mid p):a\in\mathcal A\}$ 有最小值，因此最小者存在。若随机化行动用分布 $\rho$ 表示，则其风险为

$$
\sum_{a\in\mathcal A}\rho(a)R(a\mid p),
$$

这是确定性行动风险的凸组合，不可能小于其中最小的确定性风险。因此随机化不能改善最小风险，只能在平局或额外约束下有用。证毕。

**反例 D.21（损失改变行动）.** 设疾病状态 $\Theta=\{0,1\}$，后验 $\mathbb P(\theta=1)=0.1$。行动为治疗 $T$ 或不治疗 $N$。若漏治损失为 $100$、误治损失为 $1$，则

$$
R(T)=0.9,\qquad R(N)=10,
$$

Bayes 行动是治疗。若误治损失改为 $20$，则

$$
R(T)=18,\qquad R(N)=10,
$$

Bayes 行动变为不治疗。同一后验在不同损失下给出不同最优行动。

**边界 D.22.** Bayes 行动是相对于后验、行动集合和损失函数的最优性。它不自动给出伦理许可、法律授权、资源公平或组织责任。高风险 AI 系统中，损失函数本身通常是需要审计的规范对象。

## D.7 有限自回归分布归一化

设词表 $\mathcal V$ 有限，另有终止符 $\mathtt{EOS}$。给定上下文 $c$，自回归模型在每个前缀 $s_{<t}$ 后给出条件分布

$$
p_t(\cdot\mid c,s_{<t})
$$

于 $\mathcal V\cup\{\mathtt{EOS}\}$ 上，满足

$$
\sum_{v\in\mathcal V\cup\{\mathtt{EOS}\}}
p_t(v\mid c,s_{<t})=1.
$$

为避免无限长度问题，设最大步数为 $N$。若前 $N-1$ 步尚未输出 $\mathtt{EOS}$，则第 $N$ 步强制终止。

**定义 D.23（有限停机序列概率）.** 对长度 $T<N$ 的序列

$$
s=(v_1,\ldots,v_{T-1},\mathtt{EOS}),\qquad v_i\in\mathcal V,
$$

定义

$$
P(s\mid c)=\prod_{t=1}^{T}
p_t(v_t\mid c,v_{<t}),
$$

其中 $v_T=\mathtt{EOS}$。对长度 $N$ 的强制终止序列

$$
s=(v_1,\ldots,v_{N-1},\mathtt{EOS}),
$$

定义

$$
P(s\mid c)=
\prod_{t=1}^{N-1}
p_t(v_t\mid c,v_{<t}),
$$

其中前 $N-1$ 步均未终止，第 $N$ 步的终止概率视为 $1$。

**定理 D.24（有限自回归归一化）.** 上述所有长度不超过 $N$ 且以 $\mathtt{EOS}$ 结束的序列概率之和为 $1$。

**证明.** 令 $S_t$ 为运行到第 $t$ 步前仍未终止的前缀集合，并令 $M_t$ 为这些前缀的总概率质量。初始 $M_1=1$。在任意未终止前缀 $s_{<t}$ 上，第 $t$ 步条件分布把该前缀质量分成两部分：输出 $\mathtt{EOS}$ 的质量成为终止序列，输出 $\mathcal V$ 中 token 的质量进入下一步未终止前缀。因为条件概率总和为 $1$，该前缀的质量被完全分配，没有丢失也没有重复。

对 $t=1,\ldots,N-1$ 重复此分配，所有已经终止的序列质量加上仍未终止质量始终为 $1$。第 $N$ 步把剩余未终止质量全部赋给强制终止序列。因此长度不超过 $N$ 的终止序列总质量为 $1$。证毕。

**边界 D.25.** 固定长度 token 序列的归一化只需嵌套求和即可；可变长度生成还要说明终止规则。温度、top-k、top-p、重复惩罚或非法 token 过滤会改变条件分布；只要每一步重新归一化并且停机协议明确，得到的仍是某个解码器分布，但未必是原始模型分布。

## D.8 温度与熵单调性

设有限集合上 logits 为 $z_1,\ldots,z_K$。对温度 $T>0$，定义

$$
q_T(i)=\frac{\exp(z_i/T)}{Z(T)},
\qquad
Z(T)=\sum_{j=1}^K\exp(z_j/T).
$$

熵为

$$
H(T)=-\sum_i q_T(i)\log q_T(i).
$$

**定理 D.26（温度熵单调性）.** 对固定 logits，$H(T)$ 关于 $T$ 单调不减。更精确地，

$$
\frac{dH}{dT}
=
\frac{\operatorname{Var}_{q_T}(z)}{T^3}
\ge0.
$$

若所有 logits 相等，则导数恒为 $0$；若 logits 不全相等，则对每个有限 $T>0$ 导数为正。

**证明.** 令

$$
\beta=\frac1T,
\qquad
q_\beta(i)=\frac{\exp(\beta z_i)}{Z(\beta)},
\qquad
Z(\beta)=\sum_j\exp(\beta z_j).
$$

记

$$
\mathbb E_\beta[z]=\sum_i q_\beta(i)z_i.
$$

由

$$
\log q_\beta(i)=\beta z_i-\log Z(\beta)
$$

可得熵

$$
\begin{aligned}
H(\beta)
&=-\sum_iq_\beta(i)\log q_\beta(i)\\
&=-\beta\mathbb E_\beta[z]+\log Z(\beta).
\end{aligned}
$$

先计算两个导数。第一，

$$
\frac{d}{d\beta}\log Z(\beta)
=
\frac{1}{Z(\beta)}\sum_i z_i\exp(\beta z_i)
=
\mathbb E_\beta[z].
$$

第二，

$$
\begin{aligned}
\frac{d}{d\beta}\mathbb E_\beta[z]
&=
\frac{d}{d\beta}\left(
\frac{\sum_i z_i\exp(\beta z_i)}{Z(\beta)}
\right)\\
&=
\frac{\sum_i z_i^2\exp(\beta z_i)}{Z(\beta)}
-
\left(
\frac{\sum_i z_i\exp(\beta z_i)}{Z(\beta)}
\right)^2\\
&=
\mathbb E_\beta[z^2]-\mathbb E_\beta[z]^2\\
&=
\operatorname{Var}_{q_\beta}(z).
\end{aligned}
$$

因此

$$
\begin{aligned}
\frac{dH}{d\beta}
&=
-\mathbb E_\beta[z]
-\beta\frac{d}{d\beta}\mathbb E_\beta[z]
+\frac{d}{d\beta}\log Z(\beta)\\
&=
-\beta\operatorname{Var}_{q_\beta}(z).
\end{aligned}
$$

又

$$
\frac{d\beta}{dT}=-\frac1{T^2},
\qquad
\beta=\frac1T.
$$

链式法则给出

$$
\frac{dH}{dT}
=
\frac{dH}{d\beta}\frac{d\beta}{dT}
=
\left(-\frac1T\operatorname{Var}_{q_T}(z)\right)
\left(-\frac1{T^2}\right)
=
\frac{\operatorname{Var}_{q_T}(z)}{T^3}.
$$

方差非负。由于 $q_T$ 对所有 $i$ 都赋正质量，方差为零当且仅当所有 $z_i$ 相等。证毕。

**边界 D.27.** 该单调性要求 logits 固定、支持集合固定，并使用完整 softmax。若 top-k 或 top-p 的候选集合随温度改变，若 logits 来自不同上下文，或若另有重复惩罚与过滤规则，熵未必按同一公式变化。熵上升也不等同于创造性、真实性或任务质量上升。

## D.9 概率测度、完备化与可测复合

概率测度的可数可加性蕴含以下常用性质。把这些性质写出，是为了明确后文哪些步骤只用概率公理，哪些步骤需要更强的积分定理。

**定理 D.28（概率测度的基本性质）.** 设 $(\Omega,\mathcal F,\mathbb P)$ 为概率空间，$A,B,A_n\in\mathcal F$。则：

1. $\mathbb P(\varnothing)=0$，且 $\mathbb P(A^c)=1-\mathbb P(A)$；
2. 若 $A\subseteq B$，则 $\mathbb P(A)\le \mathbb P(B)$，并且

   $$
   \mathbb P(B\setminus A)=\mathbb P(B)-\mathbb P(A);
   $$

3. 可数次可加性给出并集界

   $$
   \mathbb P\left(\bigcup_{n\ge1}A_n\right)
   \le \sum_{n\ge1}\mathbb P(A_n);
   $$

4. 若 $A_n\uparrow A$，即 $A_n\subseteq A_{n+1}$ 且 $A=\bigcup_nA_n$，则

   $$
   \mathbb P(A_n)\uparrow\mathbb P(A);
   $$

5. 若 $A_n\downarrow A$，即 $A_{n+1}\subseteq A_n$ 且 $A=\bigcap_nA_n$，则

   $$
   \mathbb P(A_n)\downarrow\mathbb P(A).
   $$

**证明.** 因为 $\Omega=\Omega\mathbin{\dot\cup}\varnothing$，有限可加性给出

$$
1=1+\mathbb P(\varnothing),
$$

故 $\mathbb P(\varnothing)=0$。又因为 $\Omega=A\mathbin{\dot\cup}A^c$，所以

$$
1=\mathbb P(A)+\mathbb P(A^c).
$$

若 $A\subseteq B$，则 $B=A\mathbin{\dot\cup}(B\setminus A)$，于是

$$
\mathbb P(B)=\mathbb P(A)+\mathbb P(B\setminus A)\ge\mathbb P(A),
$$

同时得到第二条中的差公式。

对任意 $(A_n)$，令

$$
B_1=A_1,
\qquad
B_n=A_n\setminus\bigcup_{j<n}A_j\quad(n\ge2).
$$

则 $B_n$ 两两不交、$B_n\subseteq A_n$，且 $\bigcup_nB_n=\bigcup_nA_n$。因此

$$
\mathbb P\left(\bigcup_nA_n\right)
=\sum_n\mathbb P(B_n)
\le\sum_n\mathbb P(A_n).
$$

若 $A_n\uparrow A$，令 $B_1=A_1$ 且 $B_n=A_n\setminus A_{n-1}$。这些 $B_n$ 两两不交，且

$$
A=\bigcup_{n\ge1}B_n,
\qquad
A_m=\bigcup_{n=1}^mB_n.
$$

故

$$
\mathbb P(A_m)=\sum_{n=1}^m\mathbb P(B_n)
\uparrow
\sum_{n\ge1}\mathbb P(B_n)=\mathbb P(A).
$$

最后，若 $A_n\downarrow A$，则 $A_n^c\uparrow A^c$。由刚证明的递增连续性和补集公式，

$$
1-\mathbb P(A_n)=\mathbb P(A_n^c)
\uparrow \mathbb P(A^c)=1-\mathbb P(A),
$$

所以 $\mathbb P(A_n)\downarrow\mathbb P(A)$。这里使用了 $\mathbb P(\Omega)=1<\infty$；这正是递减连续性在概率空间中无需额外有限性条件的原因。证毕。

**定义 D.29（完备概率空间）.** 若 $Z\in\mathcal F$、$\mathbb P(Z)=0$ 且 $N\subseteq Z$ 总能推出 $N\in\mathcal F$，则称 $(\Omega,\mathcal F,\mathbb P)$ 完备。

**定理 D.30（概率空间的完备化）.** 令

$$
\mathcal N
=
\{N\subseteq\Omega:\text{存在 }Z\in\mathcal F,
\ N\subseteq Z,\ \mathbb P(Z)=0\},
$$

并定义

$$
\overline{\mathcal F}
=
\{A\subseteq\Omega:\text{存在 }B\in\mathcal F,
\ A\mathbin{\triangle}B\in\mathcal N\}.
$$

对 $A\in\overline{\mathcal F}$，任选满足 $A\triangle B\in\mathcal N$ 的 $B\in\mathcal F$，定义

$$
\overline{\mathbb P}(A)=\mathbb P(B).
$$

则 $(\Omega,\overline{\mathcal F},\overline{\mathbb P})$ 是完备概率空间，并且它扩张 $(\Omega,\mathcal F,\mathbb P)$。此外，它是最小的完备扩张：任何包含 $\mathcal F$ 且延拓 $\mathbb P$ 的完备概率空间都包含 $\overline{\mathcal F}$，并在其上等于 $\overline{\mathbb P}$。

**证明.** 先注意，$\mathcal N$ 对取子集和可数并封闭。取子集的结论直接来自定义；若 $N_i\subseteq Z_i\in\mathcal F$ 且 $\mathbb P(Z_i)=0$，则

$$
\bigcup_iN_i\subseteq\bigcup_iZ_i,
\qquad
\mathbb P\left(\bigcup_iZ_i\right)
\le\sum_i\mathbb P(Z_i)=0,
$$

故 $\bigcup_iN_i\in\mathcal N$。

因为 $\Omega\triangle\Omega=\varnothing\in\mathcal N$，所以 $\Omega\in\overline{\mathcal F}$。若 $A\triangle B\in\mathcal N$，则

$$
A^c\triangle B^c=A\triangle B\in\mathcal N,
$$

故 $A^c\in\overline{\mathcal F}$。若 $A_i\triangle B_i\in\mathcal N$ 且 $B_i\in\mathcal F$，则

$$
\left(\bigcup_iA_i\right)
\triangle
\left(\bigcup_iB_i\right)
\subseteq
\bigcup_i(A_i\triangle B_i)\in\mathcal N.
$$

所以 $\overline{\mathcal F}$ 是 $\sigma$-代数。

若同一 $A$ 同时满足 $A\triangle B\in\mathcal N$ 与 $A\triangle C\in\mathcal N$，则

$$
B\triangle C
\subseteq(A\triangle B)\cup(A\triangle C)
$$

是零测集的子集。特别地，$\mathbb P(B\setminus C)=\mathbb P(C\setminus B)=0$，故 $\mathbb P(B)=\mathbb P(C)$。因此 $\overline{\mathbb P}$ 定义良好。

现设 $A_i\in\overline{\mathcal F}$ 两两不交，并选 $B_i\in\mathcal F$ 使 $A_i\triangle B_i\in\mathcal N$。令

$$
C_1=B_1,
\qquad
C_i=B_i\setminus\bigcup_{j<i}B_j.
$$

则 $C_i$ 两两不交。对 $i\ne j$，由于 $A_i\cap A_j=\varnothing$，有

$$
B_i\cap B_j
\subseteq(A_i\triangle B_i)\cup(A_j\triangle B_j),
$$

故 $B_i\cap B_j$ 为零测集。于是 $B_i\triangle C_i$ 是有限个零测集的并，仍为零测集；从而 $A_i\triangle C_i\in\mathcal N$，并且 $\mathbb P(C_i)=\mathbb P(B_i)$。又有

$$
\left(\bigcup_iA_i\right)
\triangle
\left(\bigcup_iC_i\right)
\in\mathcal N.
$$

因此

$$
\overline{\mathbb P}\left(\bigcup_iA_i\right)
=\mathbb P\left(\bigcup_iC_i\right)
=\sum_i\mathbb P(C_i)
=\sum_i\overline{\mathbb P}(A_i).
$$

归一化由 $\overline{\mathbb P}(\Omega)=\mathbb P(\Omega)=1$ 得到，所以 $\overline{\mathbb P}$ 是概率测度。对 $A\in\mathcal F$ 取 $B=A$，可见它延拓 $\mathbb P$。

若 $A\subseteq Z\in\overline{\mathcal F}$ 且 $\overline{\mathbb P}(Z)=0$，选 $B\in\mathcal F$ 使 $Z\triangle B\in\mathcal N$。此时 $\mathbb P(B)=0$，而

$$
A\subseteq Z\subseteq B\cup(Z\triangle B).
$$

右侧属于 $\mathcal N$，故 $A\in\mathcal N\subseteq\overline{\mathcal F}$ 且 $\overline{\mathbb P}(A)=0$。所以完备化确实完备。

最后，设 $(\Omega,\mathcal G,Q)$ 是一个完备扩张。若 $A\in\overline{\mathcal F}$ 且 $A\triangle B\subseteq Z$，其中 $B,Z\in\mathcal F$、$\mathbb P(Z)=0$，则 $Q(Z)=0$。完备性给出 $A\triangle B\in\mathcal G$，进而 $A\in\mathcal G$，并且

$$
Q(A)=Q(B)=\mathbb P(B)=\overline{\mathbb P}(A).
$$

这就证明了最小性。证毕。

完备化的接口作用是：若两个随机变量只在零测集上不同，则在完备化后，修改零测集上的取值不会破坏可测性。它没有改变原有可测事件的概率，也没有把任意不可测集合变成可测集合。

**定理 D.31（可测映射的复合）.** 设 $(E,\mathcal E)$、$(F,\mathcal H)$、$(G,\mathcal G)$ 是可测空间。若 $f:E\to F$ 对 $\mathcal E/\mathcal H$ 可测，$g:F\to G$ 对 $\mathcal H/\mathcal G$ 可测，则 $g\circ f:E\to G$ 对 $\mathcal E/\mathcal G$ 可测。

**证明.** 对任意 $C\in\mathcal G$，$g$ 的可测性给出 $g^{-1}(C)\in\mathcal H$，再由 $f$ 的可测性得到

$$
(g\circ f)^{-1}(C)=f^{-1}(g^{-1}(C))\in\mathcal E.
$$

这正是复合映射可测。证毕。

## D.10 推前测度的积分公式

设 $X:(\Omega,\mathcal F,\mathbb P)\to(E,\mathcal E)$ 可测，记 $\mu=X_\#\mathbb P$。以下公式把“对随机变量的函数取期望”和“对随机变量的分布积分”严格连接起来。

**定理 D.32（推前积分公式）.** 对任意可测函数 $h:E\to[0,\infty]$，

$$
\int_E h(x)\,\mu(dx)
=
\int_\Omega h(X(\omega))\,\mathbb P(d\omega).
$$

对指标函数和非负简单函数，此公式只由推前定义与有限求和得到；一般非负情形还需要单调收敛定理。

**证明（指标函数）.** 若 $h=\mathbf 1_B$，其中 $B\in\mathcal E$，则

$$
\int_E\mathbf 1_B\,d\mu
=\mu(B)
=\mathbb P(X^{-1}(B))
=\int_\Omega\mathbf 1_{X^{-1}(B)}\,d\mathbb P
=\int_\Omega\mathbf 1_B(X)\,d\mathbb P.
$$

**证明（非负简单函数）.** 把简单函数写成互不相交标准形

$$
h=\sum_{j=1}^m c_j\mathbf 1_{B_j},
\qquad c_j\ge0,
$$

其中 $B_j\in\mathcal E$ 两两不交。由指标函数情形和有限求和，

$$
\begin{aligned}
\int_Eh\,d\mu
&=\sum_{j=1}^m c_j\mu(B_j)\\
&=\sum_{j=1}^m c_j\mathbb P(X^{-1}(B_j))\\
&=\int_\Omega\sum_{j=1}^m c_j\mathbf 1_{X^{-1}(B_j)}\,d\mathbb P\\
&=\int_\Omega h(X)\,d\mathbb P.
\end{aligned}
$$

**外部输入（单调收敛定理）.** 若 $(f_n)$ 是同一测度空间上的非负可测函数且 $f_n\uparrow f$，则

$$
\int f_n\uparrow\int f,
$$

积分允许取 $+\infty$。这一一般积分定理在此作为外部输入，不在本附录证明。

**证明（一般非负函数，调用单调收敛）.** 对 $n\ge1$ 定义

$$
h_n(x)
=2^{-n}\left\lfloor 2^n\min\{h(x),n\}\right\rfloor.
$$

每个 $h_n$ 是取有限多个值的非负可测简单函数，并且 $h_n\uparrow h$。由已经证明的简单函数情形，

$$
\int_Eh_n\,d\mu=\int_\Omega h_n(X)\,d\mathbb P.
$$

分别在 $(E,\mathcal E,\mu)$ 与 $(\Omega,\mathcal F,\mathbb P)$ 上调用单调收敛定理，得到

$$
\int_Eh\,d\mu
=\lim_n\int_Eh_n\,d\mu
=\lim_n\int_\Omega h_n(X)\,d\mathbb P
=\int_\Omega h(X)\,d\mathbb P.
$$

证毕。

若 $h$ 为实值且 $\int|h|\,d\mu<\infty$，把 $h=h^+-h^-$ 分解并分别应用非负公式，即得

$$
\mathbb E[h(X)]=\int_Eh\,d(X_\#\mathbb P).
$$

## D.11 独立性在变换与矩下的保持

**定理 D.33（独立变量的可测变换仍独立）.** 设 $X_i:(\Omega,\mathcal F,\mathbb P)\to(E_i,\mathcal E_i)$，$i=1,\ldots,n$，相互独立。若 $g_i:(E_i,\mathcal E_i)\to(F_i,\mathcal H_i)$ 可测，则

$$
g_1(X_1),\ldots,g_n(X_n)
$$

相互独立。

**证明.** 任取 $C_i\in\mathcal H_i$。由可测性，$g_i^{-1}(C_i)\in\mathcal E_i$。于是

$$
\begin{aligned}
\mathbb P\left(\bigcap_{i=1}^n\{g_i(X_i)\in C_i\}\right)
&=\mathbb P\left(\bigcap_{i=1}^n\{X_i\in g_i^{-1}(C_i)\}\right)\\
&=\prod_{i=1}^n\mathbb P(X_i\in g_i^{-1}(C_i))\\
&=\prod_{i=1}^n\mathbb P(g_i(X_i)\in C_i).
\end{aligned}
$$

这就是变换后变量的相互独立性。证毕。

**定理 D.34（有限值域独立变量的乘积期望）.** 若实随机变量 $X_1,\ldots,X_n$ 相互独立且各自只有有限值域，则

$$
\mathbb E\left[\prod_{i=1}^nX_i\right]
=
\prod_{i=1}^n\mathbb E[X_i].
$$

**证明.** 设 $X_i$ 的值域为 $S_i$。有限值域保证以下所有和与期望有限。由独立性，

$$
\begin{aligned}
\mathbb E\left[\prod_{i=1}^nX_i\right]
&=\sum_{(x_1,\ldots,x_n)\in\prod_iS_i}
\left(\prod_{i=1}^nx_i\right)
\mathbb P(X_1=x_1,\ldots,X_n=x_n)\\
&=\sum_{(x_1,\ldots,x_n)}
\prod_{i=1}^n\left[x_i\mathbb P(X_i=x_i)\right]\\
&=\prod_{i=1}^n\sum_{x_i\in S_i}x_i\mathbb P(X_i=x_i)\\
&=\prod_{i=1}^n\mathbb E[X_i].
\end{aligned}
$$

第三行只是有限乘积的分配律。证毕。

**定理 D.35（有限值域两两独立变量的方差可加性）.** 若实随机变量 $X_1,\ldots,X_n$ 两两独立且各自只有有限值域，则

$$
\operatorname{Var}\left(\sum_{i=1}^nX_i\right)
=
\sum_{i=1}^n\operatorname{Var}(X_i).
$$

**证明.** 记 $\mu_i=\mathbb E[X_i]$。对 $i\ne j$，定理 D.33 说明 $X_i-\mu_i$ 与 $X_j-\mu_j$ 独立；再用定理 D.34 的二变量情形，

$$
\mathbb E[(X_i-\mu_i)(X_j-\mu_j)]
=\mathbb E[X_i-\mu_i]\,\mathbb E[X_j-\mu_j]=0.
$$

因此

$$
\begin{aligned}
\operatorname{Var}\left(\sum_iX_i\right)
&=\mathbb E\left[\left(\sum_i(X_i-\mu_i)\right)^2\right]\\
&=\sum_i\mathbb E[(X_i-\mu_i)^2]
+2\sum_{i<j}\mathbb E[(X_i-\mu_i)(X_j-\mu_j)]\\
&=\sum_i\operatorname{Var}(X_i).
\end{aligned}
$$

证毕。

方差可加只需两两独立；把联合分布分解成所有边缘分布的乘积则需要相互独立。两者不可混用。

## D.12 有限信息下的条件期望与 Bayes 公式

先处理有限 $\sigma$-代数。这样可以把条件期望直接写成有限个原子上的平均，而不暗中调用一般的 Radon--Nikodym 定理。

**引理 D.36（有限 $\sigma$-代数的原子分割）.** 若 $\mathcal G$ 是 $\Omega$ 上的有限 $\sigma$-代数，则存在有限个两两不交的非空集合 $G_1,\ldots,G_m\in\mathcal G$，其并为 $\Omega$，且 $\mathcal G$ 中每个集合都是某些 $G_j$ 的并。这样的 $G_j$ 称为 $\mathcal G$ 的原子。

**证明.** 对 $\omega,\omega'\in\Omega$，定义

$$
\omega\sim\omega'
\quad\Longleftrightarrow\quad
\text{对所有 }A\in\mathcal G,
\ \omega\in A\text{ 当且仅当 }\omega'\in A.
$$

这是等价关系。因为 $\mathcal G$ 只有有限多个集合，每个等价类都可写成有限个 $\mathcal G$ 中集合或其补集的交，因此属于 $\mathcal G$。等价类至多有 $2^{|\mathcal G|}$ 个，故只有有限个非空类；它们两两不交且覆盖 $\Omega$。按等价关系的定义，任意 $A\in\mathcal G$ 要么包含某个等价类的全部点，要么与该类不交，所以 $A$ 是若干等价类的并。证毕。

**定理 D.37（有限 $\sigma$-代数上的条件期望存在唯一）.** 设 $X$ 是可积实随机变量，$\mathcal G\subseteq\mathcal F$ 是有限 $\sigma$-代数，其原子为 $G_1,\ldots,G_m$。定义

$$
Y
=
\sum_{j:\,\mathbb P(G_j)>0}
\frac{\mathbb E[X\mathbf 1_{G_j}]}{\mathbb P(G_j)}
\mathbf 1_{G_j},
$$

并在零概率原子上令 $Y=0$。则 $Y$ 可积且 $\mathcal G$-可测，并对每个 $A\in\mathcal G$ 满足

$$
\int_A Y\,d\mathbb P=\int_A X\,d\mathbb P.
$$

满足这两个条件的随机变量在几乎处处意义下唯一，记为 $\mathbb E[X\mid\mathcal G]$。

**证明.** $Y$ 在每个原子上为常数，所以由引理 D.36 可知它 $\mathcal G$-可测。并且

$$
\mathbb E[|Y|]
=\sum_{j:\,\mathbb P(G_j)>0}
\left|\mathbb E[X\mathbf 1_{G_j}]\right|
\le\sum_j\mathbb E[|X|\mathbf 1_{G_j}]
=\mathbb E[|X|]<\infty.
$$

任意 $A\in\mathcal G$ 是若干原子的并。对正概率原子，定义直接给出

$$
\int_{G_j}Y\,d\mathbb P
=\mathbb E[X\mathbf 1_{G_j}]
=\int_{G_j}X\,d\mathbb P;
$$

对零概率原子，两边都为 $0$。对组成 $A$ 的原子求和，即得积分恒等式。

为证唯一，设 $Y_1,Y_2$ 都满足条件，令 $W=Y_1-Y_2$。则 $W$ 可积且 $\mathcal G$-可测，并对每个 $A\in\mathcal G$ 有 $\int_AW\,d\mathbb P=0$。集合 $A_+=\{W>0\}$ 属于 $\mathcal G$。若 $\mathbb P(A_+)>0$，则因

$$
A_+=\bigcup_{k\ge1}\{W\ge1/k\},
$$

至少存在 $k$ 使 $\mathbb P(W\ge1/k)>0$，从而

$$
\int_{A_+}W\,d\mathbb P
\ge\frac1k\mathbb P(W\ge1/k)>0,
$$

与积分恒等式矛盾。因此 $\mathbb P(W>0)=0$。对 $-W$ 同理，$\mathbb P(W<0)=0$，故 $Y_1=Y_2$ 几乎处处。证毕。

**定理 D.38（有限条件期望的线性、保序与塔式性质）.** 设 $X,Y$ 可积，$a,b\in\mathbb R$，且 $\mathcal H\subseteq\mathcal G\subseteq\mathcal F$ 都是有限 $\sigma$-代数。则：

1. 线性：

   $$
   \mathbb E[aX+bY\mid\mathcal G]
   =a\mathbb E[X\mid\mathcal G]
   +b\mathbb E[Y\mid\mathcal G]
   \quad\text{a.s.};
   $$

2. 保序：若 $X\le Y$ a.s.，则

   $$
   \mathbb E[X\mid\mathcal G]
   \le\mathbb E[Y\mid\mathcal G]
   \quad\text{a.s.};
   $$

3. 塔式性质：

   $$
   \mathbb E[\mathbb E[X\mid\mathcal G]\mid\mathcal H]
   =\mathbb E[X\mid\mathcal H]
   \quad\text{a.s.}
   $$

**证明.** 令 $X_{\mathcal G}=\mathbb E[X\mid\mathcal G]$、$Y_{\mathcal G}=\mathbb E[Y\mid\mathcal G]$。随机变量 $aX_{\mathcal G}+bY_{\mathcal G}$ 可积且 $\mathcal G$-可测。对任意 $A\in\mathcal G$，

$$
\begin{aligned}
\int_A(aX_{\mathcal G}+bY_{\mathcal G})\,d\mathbb P
&=a\int_AX\,d\mathbb P+b\int_AY\,d\mathbb P\\
&=\int_A(aX+bY)\,d\mathbb P.
\end{aligned}
$$

由定理 D.37 的唯一性得到线性。

若 $X\le Y$ a.s.，考察 $\mathcal G$ 的任一正概率原子 $G_j$。由定理 D.37 的显式公式，

$$
\mathbb E[X\mid\mathcal G]\big|_{G_j}
=\frac{\mathbb E[X\mathbf 1_{G_j}]}{\mathbb P(G_j)}
\le
\frac{\mathbb E[Y\mathbf 1_{G_j}]}{\mathbb P(G_j)}
=\mathbb E[Y\mid\mathcal G]\big|_{G_j}.
$$

所有零概率原子的并仍为零测集，因此保序性几乎处处成立。

最后，$\mathbb E[X\mid\mathcal G]$ 可积。对任意 $A\in\mathcal H$，因为 $\mathcal H\subseteq\mathcal G$，有

$$
\int_A\mathbb E[X\mid\mathcal G]\,d\mathbb P
=\int_AX\,d\mathbb P.
$$

所以 $\mathbb E[\mathbb E[X\mid\mathcal G]\mid\mathcal H]$ 满足 $\mathbb E[X\mid\mathcal H]$ 的定义；由唯一性，两者几乎处处相等。证毕。

**定理 D.39（有限 Bayes 公式）.** 设状态变量 $\Theta$ 与观测变量 $Y$ 都只有有限值域。记

$$
\pi(\theta)=\mathbb P(\Theta=\theta),
\qquad
L(y\mid\theta)=\mathbb P(Y=y\mid\Theta=\theta)
$$

其中当 $\pi(\theta)>0$ 时，$L(y\mid\theta)$ 是通常的条件概率；当 $\pi(\theta)=0$ 时约定 $L(y\mid\theta)=0$。若 $\mathbb P(Y=y)>0$，则

$$
\mathbb P(\Theta=\theta\mid Y=y)
=
\frac{L(y\mid\theta)\pi(\theta)}
{\sum_{\vartheta}L(y\mid\vartheta)\pi(\vartheta)}.
$$

**证明.** 由有限条件概率定义，

$$
\mathbb P(\Theta=\theta\mid Y=y)
=\frac{\mathbb P(\Theta=\theta,Y=y)}{\mathbb P(Y=y)}.
$$

若 $\pi(\theta)>0$，乘法公式给出

$$
\mathbb P(\Theta=\theta,Y=y)
=L(y\mid\theta)\pi(\theta);
$$

若 $\pi(\theta)=0$，联合概率和右侧都为 $0$。状态事件 $\{\Theta=\vartheta\}$ 构成有限分割，所以

$$
\mathbb P(Y=y)
=\sum_\vartheta\mathbb P(Y=y,\Theta=\vartheta)
=\sum_\vartheta L(y\mid\vartheta)\pi(\vartheta).
$$

分母因假设为正。代入即得结论。证毕。

**外部输入（一般条件期望存在性）.** 在任意概率空间上，若 $X$ 可积且 $\mathcal G\subseteq\mathcal F$ 是任意子 $\sigma$-代数，则存在可积、$\mathcal G$-可测的 $Y$，使

$$
\int_A Y\,d\mathbb P=\int_A X\,d\mathbb P
\quad\text{对所有 }A\in\mathcal G,
$$

并且 $Y$ 在 a.s. 意义下唯一。一般证明依赖 Radon--Nikodym 定理，本附录把它作为外部输入。定理 D.37--D.38 已经在有限 $\sigma$-代数范围内给出了不依赖该输入的完整证明。

## D.13 四种随机收敛

以下随机变量均为实值。除分布收敛可以在不同概率空间上定义外，比较 a.s.、$L^p$ 与概率收敛时，默认 $X_n,X$ 定义在同一概率空间上。

**定义 D.40（四种收敛）.** 设 $p>0$。

1. $X_n$ 几乎处处收敛到 $X$，记 $X_n\to X$ a.s.，若

   $$
   \mathbb P(\{\omega:X_n(\omega)\to X(\omega)\})=1.
   $$

2. $X_n$ 在 $L^p$ 中收敛到 $X$，记 $X_n\to X$ in $L^p$，若 $\mathbb E|X_n-X|^p<\infty$ 且

   $$
   \mathbb E|X_n-X|^p\to0.
   $$

3. $X_n$ 依概率收敛到 $X$，记 $X_n\to X$ in probability，若对每个 $\epsilon>0$，

   $$
   \mathbb P(|X_n-X|>\epsilon)\to0.
   $$

4. $X_n$ 依分布收敛到 $X$，记 $X_n\Rightarrow X$，若对 $X$ 的分布函数 $F(x)=\mathbb P(X\le x)$ 的每个连续点 $x$，

   $$
   F_n(x)=\mathbb P(X_n\le x)\to F(x).
   $$

当 $p\ge1$ 时，第二条由范数 $\|Z\|_p=(\mathbb E|Z|^p)^{1/p}$ 给出；$0<p<1$ 时同一收敛定义仍有意义，但该表达式不是范数。

**定理 D.41（收敛方式的基本蕴含）.** 在同一概率空间上，对任意 $p>0$，

$$
X_n\to X\text{ in }L^p
\quad\Longrightarrow\quad
X_n\to X\text{ in probability}
\quad\Longrightarrow\quad
X_n\Rightarrow X,
$$

并且

$$
X_n\to X\text{ a.s.}
\quad\Longrightarrow\quad
X_n\to X\text{ in probability}.
$$

**证明（$L^p\Rightarrow$ 概率）.** 对任意 $\epsilon>0$，把 Markov 不等式用于非负随机变量 $|X_n-X|^p$：

$$
\mathbb P(|X_n-X|>\epsilon)
=\mathbb P(|X_n-X|^p>\epsilon^p)
\le\frac{\mathbb E|X_n-X|^p}{\epsilon^p}
\longrightarrow0.
$$

**证明（a.s. $\Rightarrow$ 概率）.** 固定 $\epsilon>0$，令

$$
A_m=\bigcup_{n\ge m}\{|X_n-X|>\epsilon\}.
$$

则 $A_m\downarrow A_\infty$，其中 $A_\infty$ 是事件“$|X_n-X|>\epsilon$ 发生无穷多次”。几乎处处收敛给出 $\mathbb P(A_\infty)=0$。由定理 D.28 的递减连续性，

$$
\mathbb P(A_m)\downarrow0.
$$

对任意 $n\ge m$，

$$
\mathbb P(|X_n-X|>\epsilon)\le\mathbb P(A_m),
$$

故左侧趋于 $0$。这里关键使用了底层空间是概率空间，因而总测度有限。

**证明（概率 $\Rightarrow$ 分布）.** 设 $x$ 是 $F$ 的连续点，固定 $\epsilon>0$。集合包含关系给出

$$
\{X\le x-\epsilon\}\setminus\{|X_n-X|>\epsilon\}
\subseteq\{X_n\le x\}
$$

以及

$$
\{X_n\le x\}
\subseteq
\{X\le x+\epsilon\}\cup\{|X_n-X|>\epsilon\}.
$$

因此

$$
F(x-\epsilon)-\mathbb P(|X_n-X|>\epsilon)
\le F_n(x)
\le F(x+\epsilon)+\mathbb P(|X_n-X|>\epsilon).
$$

先令 $n\to\infty$，再令 $\epsilon\downarrow0$。由 $F$ 在 $x$ 连续，左右两端都趋于 $F(x)$，故 $F_n(x)\to F(x)$。证毕。

**反例 D.42（逆蕴含一般不成立）.** 以下例子同时界定上述箭头不能反向使用。

1. **$L^p$ 收敛不推出 a.s. 收敛。** 在 $([0,1),\mathcal B,\lambda)$ 上，对每个 $k\ge0$ 和 $j=0,\ldots,2^k-1$，令

   $$
   A_{2^k+j}=[j2^{-k},(j+1)2^{-k}),
   \qquad
   X_{2^k+j}=\mathbf 1_{A_{2^k+j}}.
   $$

   当 $n=2^k+j$ 时，

   $$
   \mathbb E|X_n|^p=2^{-k}\to0,
   $$

   所以 $X_n\to0$ in $L^p$，并因而依概率收敛。但每个 $\omega$ 在每一级 $k$ 恰落入一个 $A_{2^k+j}$；同时每一级从 $k\ge1$ 起也有其它指标使 $X_n(\omega)=0$。故每条样本路径都有无穷多个 $1$ 和无穷多个 $0$，不收敛到 $0$。

2. **a.s. 收敛不推出 $L^p$ 收敛。** 固定 $p>0$，在 $((0,1],\mathcal B,\lambda)$ 上令

   $$
   X_n=n^{1/p}\mathbf 1_{(0,1/n]}.
   $$

   对每个 $\omega>0$，充分大的 $n$ 满足 $1/n<\omega$，故 $X_n(\omega)=0$，所以 $X_n\to0$ a.s.；但

   $$
   \mathbb E|X_n|^p=n\lambda((0,1/n])=1,
   $$

   因而不在 $L^p$ 中收敛。这个例子也说明依概率收敛不推出 $L^p$ 收敛。

3. **分布收敛不推出概率收敛。** 令 $X$ 为取 $-1$ 与 $1$ 各以概率 $1/2$ 的变量，并令 $X_n=-X$。每个 $X_n$ 与 $X$ 同分布，所以 $X_n\Rightarrow X$；但

   $$
   \mathbb P(|X_n-X|>1)=1,
   $$

   所以不依概率收敛。

由此，a.s. 收敛与 $L^p$ 收敛一般互不蕴含；二者都推出概率收敛，而概率收敛再推出分布收敛。

## D.14 三个外部概率定理及其调用边界

以下三条是标准但证明明显超出本附录有限证明内核的结果。它们在后文只能按所列假设调用。

**外部输入（强大数律，i.i.d. 可积版）.** 若 $X_1,X_2,\ldots$ 是独立同分布实随机变量，且

$$
\mathbb E|X_1|<\infty,
\qquad
\mu=\mathbb E[X_1],
$$

则

$$
\frac1n\sum_{i=1}^nX_i\longrightarrow\mu
\quad\text{a.s.}
$$

此处只把结论用于 i.i.d. 且一阶绝对矩有限的固定样本序列。它不给有限样本误差率，也不覆盖依赖数据、分布漂移或由数据自适应选定停止时刻的情形。

**外部输入（中心极限定理，i.i.d. 有限方差版）.** 若 $X_1,X_2,\ldots$ 独立同分布，

$$
\mathbb E[X_1]=\mu,
\qquad
0<\sigma^2=\operatorname{Var}(X_1)<\infty,
$$

则

$$
\frac{\sum_{i=1}^nX_i-n\mu}{\sigma\sqrt n}
\Rightarrow N(0,1).
$$

这是分布收敛，不是任意有限 $n$ 下的正态恒等式；仅凭该陈述不能给出统一尾界，也不能覆盖无限方差、强依赖或数据依赖标准化。

**外部输入（Hoeffding 不等式）.** 设 $X_1,\ldots,X_n$ 相互独立，并存在确定常数 $a_i\le b_i$ 使

$$
\mathbb P(a_i\le X_i\le b_i)=1.
$$

令 $S_n=\sum_iX_i$，并设 $V=\sum_i(b_i-a_i)^2>0$。则对每个 $t>0$，

$$
\mathbb P(S_n-\mathbb E[S_n]\ge t)
\le
\exp\left(
-\frac{2t^2}{V}
\right),
$$

并且

$$
\mathbb P(|S_n-\mathbb E[S_n]|\ge t)
\le
2\exp\left(
-\frac{2t^2}{V}
\right).
$$

若 $V=0$，则每个 $X_i=a_i=b_i$ a.s.，$S_n-\mathbb E[S_n]=0$ a.s.，尾概率对 $t>0$ 为 $0$。调用时必须同时核对独立性和几乎处处有界性。该版本不直接适用于无界损失、共享随机效应、序列相关、训练后挑选出的评测子集或可选停止程序。

## D.15 有限分布上的 log-sum 与数据处理

本节所有字母表均为有限集合，所有 KL 都采用 D.4 的扩展实数约定。

**定理 D.43（log-sum 不等式）.** 对有限个非负数 $a_i,b_i$，令 $A=\sum_i a_i$、$B=\sum_i b_i$。采用 $0\log(0/b)=0$ 以及 $a\log(a/0)=+\infty$（$a>0$）的约定，则

$$
\sum_i a_i\log\frac{a_i}{b_i}
\ge
A\log\frac AB
$$

在两边有定义的扩展实数意义下成立。特别地，当 $A,B>0$ 且 $a_i>0$ 蕴含 $b_i>0$ 时，两边都有限。

**证明.** 若 $A=0$，则所有 $a_i=0$，两边都按约定为 $0$。若 $A>0$ 而 $B=0$，则至少一个 $a_i>0$ 且所有 $b_i=0$，两边都为 $+\infty$。现设 $A,B>0$。若某个 $a_i>0,b_i=0$，左侧为 $+\infty$，不等式成立。否则定义概率分布

$$
p_i=\frac{a_i}{A},
\qquad
q_i=\frac{b_i}{B}.
$$

直接整理得

$$
\begin{aligned}
\sum_i a_i\log\frac{a_i}{b_i}
-A\log\frac AB
&=A\sum_i p_i\log\frac{p_i}{q_i}\\
&=A D_{\mathrm{KL}}(p\Vert q)\\
&\ge0,
\end{aligned}
$$

最后一步由定理 D.13。证毕。

**定义 D.44（有限 Markov 核）.** 从有限集合 $\mathcal X$ 到有限集合 $\mathcal Y$ 的 Markov 核是数族

$$
K(y\mid x)\ge0,
\qquad
\sum_{y\in\mathcal Y}K(y\mid x)=1
$$

对每个 $x\in\mathcal X$ 成立。分布 $p$ 经过 $K$ 后的输出分布为

$$
(pK)(y)=\sum_xp(x)K(y\mid x).
$$

**定理 D.45（有限 KL 的数据处理不等式）.** 对有限集合上的分布 $p,q$ 和 Markov 核 $K$，

$$
D_{\mathrm{KL}}(pK\Vert qK)
\le D_{\mathrm{KL}}(p\Vert q).
$$

**证明.** 若 $D_{\mathrm{KL}}(p\Vert q)=+\infty$，结论平凡。以下设其有限，因此 $p(x)>0$ 时必有 $q(x)>0$。对每个固定 $y$，把 log-sum 不等式用于

$$
a_x=p(x)K(y\mid x),
\qquad
b_x=q(x)K(y\mid x).
$$

得到

$$
(pK)(y)\log\frac{(pK)(y)}{(qK)(y)}
\le
\sum_xp(x)K(y\mid x)
\log\frac{p(x)K(y\mid x)}{q(x)K(y\mid x)}.
$$

当 $p(x)K(y\mid x)=0$ 时相应项按约定为 $0$；当它为正时，有限输入 KL 保证 $q(x)>0$，且 $K(y\mid x)>0$，故核因子可以约去。对 $y$ 求和并交换有限求和次序，

$$
\begin{aligned}
D_{\mathrm{KL}}(pK\Vert qK)
&\le
\sum_{x,y}p(x)K(y\mid x)\log\frac{p(x)}{q(x)}\\
&=\sum_xp(x)\log\frac{p(x)}{q(x)}
\sum_yK(y\mid x)\\
&=D_{\mathrm{KL}}(p\Vert q).
\end{aligned}
$$

证毕。

**定理 D.46（有限双射下 KL 不变）.** 若 $\phi:\mathcal X\to\mathcal Y$ 是有限集合之间的双射，并定义

$$
p^\phi(y)=p(\phi^{-1}(y)),
\qquad
q^\phi(y)=q(\phi^{-1}(y)),
$$

则

$$
D_{\mathrm{KL}}(p^\phi\Vert q^\phi)
=D_{\mathrm{KL}}(p\Vert q).
$$

**证明.** 双射只重命名求和指标，并保持零概率支撑关系。因此

$$
\begin{aligned}
D_{\mathrm{KL}}(p^\phi\Vert q^\phi)
&=\sum_{y\in\mathcal Y}
p(\phi^{-1}(y))
\log\frac{p(\phi^{-1}(y))}{q(\phi^{-1}(y))}\\
&=\sum_{x\in\mathcal X}p(x)\log\frac{p(x)}{q(x)}\\
&=D_{\mathrm{KL}}(p\Vert q).
\end{aligned}
$$

若任一侧因支撑不匹配为 $+\infty$，双射使另一侧出现完全对应的不匹配，等式仍成立。证毕。

数据处理不等式允许随机映射丢失区分 $p$ 与 $q$ 的信息；双射有确定性逆映射，所以没有信息丢失，KL 恰好保持。

## D.16 有限 Markov 核的单随机源实现

**定理 D.47（单一均匀随机变量的逆变换实现）.** 设 $K$ 是从有限集合 $\mathcal X$ 到有限集合

$$
\mathcal Y=\{y_1,\ldots,y_m\}
$$

的 Markov 核。存在确定性函数

$$
F:\mathcal X\times[0,1)\to\mathcal Y
$$

使得：若 $U\sim\operatorname{Unif}[0,1)$，则对每个固定 $x$，

$$
\mathbb P(F(x,U)=y_j)=K(y_j\mid x).
$$

更一般地，若随机变量 $X$ 与同一个 $U$ 独立，则

$$
\mathbb P(F(X,U)=y_j\mid X=x)=K(y_j\mid x)
$$

对每个满足 $\mathbb P(X=x)>0$ 的 $x$ 成立。

**证明.** 对每个 $x$ 定义累计和

$$
c_0(x)=0,
\qquad
c_j(x)=\sum_{r=1}^jK(y_r\mid x),
\quad j=1,\ldots,m.
$$

则 $0=c_0(x)\le\cdots\le c_m(x)=1$。令

$$
F(x,u)=y_j
\quad\Longleftrightarrow\quad
c_{j-1}(x)\le u<c_j(x).
$$

零长度区间对应零概率输出，不影响定义；所有半开区间两两不交并覆盖 $[0,1)$。均匀分布按区间长度赋概率，所以

$$
\mathbb P(F(x,U)=y_j)
=c_j(x)-c_{j-1}(x)
=K(y_j\mid x).
$$

若 $X$ 与 $U$ 独立，则给定 $X=x$ 不改变 $U$ 的均匀分布，故同一计算给出条件分布公式。因为 $\mathcal X,\mathcal Y$ 有限，而每个逆像是有限个形如 $\{x\}\times[c_{j-1}(x),c_j(x))$ 的并，$F$ 对离散 $\sigma$-代数与 Borel $\sigma$-代数可测。证毕。

一个 $U$ 已足以实现一次有限核采样；定理并未声称重复调用时可以反复复用同一 $U$ 而仍得到条件独立样本。若需要独立重复采样，必须提供相应的独立随机源或一个已证明等价的随机流构造。

## D.17 有限随机分配下的 ATE 识别

设协变量 $X$ 取有限值域 $\mathcal X$，处理 $Z\in\{0,1\}$，潜在结果 $Y(1),Y(0)$ 是有限值域实随机变量，观察结果为 $Y$。定义平均处理效应

$$
\operatorname{ATE}=\mathbb E[Y(1)-Y(0)].
$$

以下三个条件是识别结论的一部分，而不是从观测数据自动推出的事实。

1. **条件独立性（随机分配）**：

   $$
   (Y(1),Y(0))\perp Z\mid X.
   $$

   在有限值域中，这表示对每个正概率协变量层 $x$，给定 $X=x$ 后，处理分配与两个潜在结果的联合分布分解。

2. **正性**：对每个 $\mathbb P(X=x)>0$ 的 $x$，

   $$
   0<e(x):=\mathbb P(Z=1\mid X=x)<1.
   $$

3. **一致性**：

   $$
   Y=Y(Z)=ZY(1)+(1-Z)Y(0)
   \quad\text{a.s.}
   $$

**定理 D.48（有限分层随机分配下的 ATE 识别）.** 在上述条件下，

$$
\operatorname{ATE}
=
\sum_{x\in\mathcal X}\mathbb P(X=x)
\left[
\mathbb E(Y\mid Z=1,X=x)
-
\mathbb E(Y\mid Z=0,X=x)
\right],
$$

其中只对 $\mathbb P(X=x)>0$ 的层求和。等价地，

$$
\operatorname{ATE}
=
\mathbb E\left[
\frac{ZY}{e(X)}
-
\frac{(1-Z)Y}{1-e(X)}
\right].
$$

**证明.** 固定一个正概率层 $x$。正性保证事件 $\{Z=z,X=x\}$ 对 $z=0,1$ 都有正概率，所以相应条件均值有定义。设 $S_z$ 是 $Y(z)$ 的有限值域。由条件独立性，对每个 $y\in S_z$，

$$
\mathbb P(Y(z)=y\mid Z=z,X=x)
=\mathbb P(Y(z)=y\mid X=x).
$$

乘以 $y$ 并对有限集合 $S_z$ 求和，得到

$$
\mathbb E[Y(z)\mid Z=z,X=x]
=\mathbb E[Y(z)\mid X=x].
$$

由一致性，在事件 $\{Z=z\}$ 上有 $Y=Y(z)$，因此

$$
\mathbb E[Y\mid Z=z,X=x]
=\mathbb E[Y(z)\mid Z=z,X=x]
=\mathbb E[Y(z)\mid X=x].
$$

对 $z=1$ 与 $z=0$ 相减，再按 $X$ 的有限分布求和，得到

$$
\begin{aligned}
&\sum_x\mathbb P(X=x)
\left[
\mathbb E(Y\mid Z=1,X=x)
-\mathbb E(Y\mid Z=0,X=x)
\right]\\
&\quad=
\sum_x\mathbb P(X=x)
\left[
\mathbb E(Y(1)\mid X=x)
-\mathbb E(Y(0)\mid X=x)
\right]\\
&\quad=\mathbb E[Y(1)-Y(0)].
\end{aligned}
$$

这证明了分层公式。

再证明逆概率加权公式。固定正概率层 $x$，利用一致性、条件独立性和 $e(x)>0$，

$$
\begin{aligned}
\mathbb E\left[\frac{ZY}{e(X)}\,
\middle|\,X=x\right]
&=\frac1{e(x)}
\mathbb E[ZY(1)\mid X=x]\\
&=\frac1{e(x)}
\mathbb P(Z=1\mid X=x)
\mathbb E[Y(1)\mid Z=1,X=x]\\
&=\mathbb E[Y(1)\mid X=x].
\end{aligned}
$$

同理，由 $1-e(x)>0$，

$$
\mathbb E\left[\frac{(1-Z)Y}{1-e(X)}\,
\middle|\,X=x\right]
=\mathbb E[Y(0)\mid X=x].
$$

按有限个 $x$ 求全期望并相减，即得第二个公式。证毕。

识别公式只说明：在三项假设成立时，观测分布唯一决定所写的 ATE 泛函。若存在未记录的共同原因、某层只接受一种处理、实际处理偏离记录处理，或不同处理版本不能由一个二元 $Z$ 表示，则证明中的等号会在对应步骤失效。

## D.18 分布函数、密度与有限机制核

**命题 D.49（分布函数的基本性质）.** 设 $X$ 为实随机变量，定义其分布函数

$$
F_X(t)=\mathbb P(X\le t),\qquad t\in\mathbb R.
$$

则 $F_X$ 单调不减、右连续，并且

$$
\lim_{t\to-\infty}F_X(t)=0,
\qquad
\lim_{t\to+\infty}F_X(t)=1.
$$

**证明.** 若 $s\le t$，则 $\{X\le s\}\subseteq\{X\le t\}$，故由测度单调性，$F_X(s)\le F_X(t)$。

固定 $t\in\mathbb R$，取任意严格递减且收敛到 $t$ 的序列 $t_n$。事件 $A_n=\{X\le t_n\}$ 递减，且

$$
\bigcap_{n\ge1}A_n=\{X\le t\}.
$$

由定理 D.28 的上连续性，$F_X(t_n)\to F_X(t)$，所以 $F_X$ 右连续。

事件 $\{X\le n\}$ 随 $n$ 递增到 $\Omega$，而 $\{X\le -n\}$ 随 $n$ 递减到空集。再次使用定理 D.28 的下连续性和上连续性，得到

$$
F_X(n)\longrightarrow1,
\qquad
F_X(-n)\longrightarrow0.
$$

结合单调性即得两端的一般极限。证毕。

**外部输入 D.RN（Radon--Nikodym 定理的密度特例）.** 记 $\lambda$ 为 $\mathbb R$ 上的 Lebesgue 测度。若 $X$ 的分布 $\mu_X$ 满足 $\mu_X\ll\lambda$，则存在可测函数 $f_X:\mathbb R\to[0,\infty)$，使得

$$
\mu_X(A)=\int_A f_X\,d\lambda
$$

对每个 Borel 集 $A$ 成立；$f_X$ 在 $\lambda$-几乎处处相等的意义下唯一，并且 $\int_{\mathbb R}f_X\,d\lambda=1$。本书不在此重证一般 Radon--Nikodym 定理。

条件 $\mu_X\ll\lambda$ 不能删除。例如常值随机变量 $X=c$ 的分布是点质量 $\delta_c$；单点集 $\{c\}$ 的 Lebesgue 测度为 $0$，但 $\delta_c(\{c\})=1$，故 $\delta_c$ 不存在相对于 $\lambda$ 的密度。“随机变量有分布”因此不等于“随机变量有概率密度函数”。

**定理 D.50（有限机制核的归一化与截断干预）.** 设 $X_1,\ldots,X_m$ 分别取值于有限非空集合 $\mathcal X_1,\ldots,\mathcal X_m$。给定一个有向无环图，并按拓扑序编号，使每个父节点集合 $\operatorname{pa}(i)$ 都包含于 $\{1,\ldots,i-1\}$。对每个 $i$，设

$$
k_i(x_i\mid x_{\operatorname{pa}(i)})\ge0,
\qquad
\sum_{x_i\in\mathcal X_i}
k_i(x_i\mid x_{\operatorname{pa}(i)})=1
$$

对每个父节点取值成立。则

$$
p(x_1,\ldots,x_m)
=\prod_{i=1}^m k_i(x_i\mid x_{\operatorname{pa}(i)})
$$

是 $\prod_i\mathcal X_i$ 上的概率质量函数。

固定 $j$ 与 $a\in\mathcal X_j$。把第 $j$ 个核替换为

$$
k_j^{\operatorname{do}(a)}(x_j\mid x_{\operatorname{pa}(j)})
=\mathbf 1\{x_j=a\},
$$

其余核保持不变，则截断乘积

$$
p^{\operatorname{do}(X_j=a)}(x_1,\ldots,x_m)
=\mathbf 1\{x_j=a\}\prod_{i\ne j}
k_i(x_i\mid x_{\operatorname{pa}(i)})
$$

也归一化为 $1$。

**证明.** 所有因子非负。按 $x_m,x_{m-1},\ldots,x_1$ 的次序对乘积求和。求和到 $x_i$ 时，所有指标大于 $i$ 的因子已经被消去；剩余因子中只有 $k_i$ 含有 $x_i$，因为每个剩余节点的父节点编号都更小。因此

$$
\sum_{x_i\in\mathcal X_i}
k_i(x_i\mid x_{\operatorname{pa}(i)})=1
$$

逐步消去每个因子，最终总和为 $1$。

对干预后的乘积使用同一逆拓扑求和。除第 $j$ 步外仍使用各核的归一化；第 $j$ 步使用

$$
\sum_{x_j\in\mathcal X_j}\mathbf 1\{x_j=a\}=1.
$$

故干预后的总和也为 $1$。证毕。

定理 D.50 给出的是一个已声明机制模型内部的分布语义。它本身不说明这些核能由观测联合分布唯一恢复，也不说明两个具有同一观测分布的机制模型具有同一干预分布。后者属于因果识别问题，需要像定理 D.48 那样另列结构假设与识别证明。

## 练习

**练习 D.1.** 给出一个非负随机变量，使 Markov 不等式在某个 $a$ 上取等号；再给出一个例子说明均值很小仍可能存在低概率大损失。

**练习 D.2.** 用 Chebyshev 不等式证明：若 $X_i$ 两两独立但不要求同分布，且 $\operatorname{Var}(X_i)\le C$、$\mathbb E[X_i]=\mu_i$，则 $\frac1n\sum_i(X_i-\mu_i)$ 依概率收敛到 $0$。

**练习 D.3.** 对三分类分布 $p=(1/2,1/3,1/6)$，写出期望对数损失与 KL 项的分解，并说明为什么报告 $q=p$ 最优。

**练习 D.4.** 固定 logits $(2,0,-1)$，写出 $q_T$、$H(T)$ 和 $\frac{dH}{dT}$ 的表达式，并判断 $T\to0$ 与 $T\to\infty$ 的熵极限。

**练习 D.5.** 设 $\mathcal G$ 由有限分割 $G_1,\ldots,G_m$ 生成。直接使用原子公式计算 $\mathbb E[X\mid\mathcal G]$，并证明若 $X$ 本身 $\mathcal G$-可测，则 $\mathbb E[X\mid\mathcal G]=X$ a.s.。

**练习 D.6.** 构造一个依概率收敛但不在 $L^1$ 中收敛的随机变量序列；显式计算其尾概率与一阶绝对矩，并指出为什么定理 D.41 中使用的 Markov 不等式不能反向使用。

**练习 D.7.** 对给定的二元输入分布 $p,q$ 和二元 Markov 核 $K$，显式计算 $D_{\mathrm{KL}}(p\Vert q)$ 与 $D_{\mathrm{KL}}(pK\Vert qK)$，并找出数据处理不等式严格取小于号和取等号的各一个例子。

**练习 D.8.** 在三个协变量层的随机分配模型中，给定各层概率、处理倾向与两个处理组的条件均值，分别用分层公式和逆概率加权公式表示 ATE；再说明若某一正概率层满足 $e(x)=0$，证明的哪一步失效。
