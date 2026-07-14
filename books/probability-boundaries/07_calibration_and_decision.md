# 第七章：预测、校准与决策

一个预报系统把样本分成两组，并分别报出正例概率 $1/4$ 与 $3/4$。事后统计却发现，
前一组的正例率是 $1/2$，后一组的正例率是 $1$。这个系统有区分能力，但两个报告都
偏低。把所有样本统一报告总体正例率可以获得校准，却会丢掉组间差异。校准、分辨率
和损失因此不是同一个性质。

第四章的条件期望给出总体校准的精确定义，第六章的评分规则则提供比较预测的方法。
本章先把二分类 Brier 风险分解成可解释的三项，再把概率预测接到带损失的行动选择上。

## 7.1 二分类概率预测

设 $Y:(\Omega,\mathcal F,\mathbb P)\to\{0,1\}$ 与 $P:(\Omega,\mathcal F,\mathbb P)\to[0,1]$ 是同一概率空间上的随机变量，其中 $[0,1]$ 带 Borel $\sigma$-代数。$P$ 可以是模型输出、后处理后的概率或人类报告；校准始终是相对于当前联合分布 $\mathcal L(P,Y)$ 的性质。

**定义 7.1（总体校准）.** 若

$$
\mathbb E[Y\mid\sigma(P)]=P\quad\text{几乎处处},
$$

则称 $P$ 对 $Y$ 校准。

若 $P$ 只取有限个值，该条件等价于：对每个 $p$ 满足 $\mathbb P(P=p)>0$，

$$
\mathbb P(Y=1\mid P=p)=p.
$$

零概率的预测值不受该定义约束。有限样本可靠性图只能估计总体条件频率，分箱方式会引入分辨率与偏差。

**定义 7.2（条件发生率）.** 记

$$
Q\coloneqq\mathbb E[Y\mid\sigma(P)].
$$

$Q$ 是由预测值可区分的信息下的真实条件发生率。校准等价于 $P=Q$ 几乎处处。
由 $0\le Y\le1$ 与条件期望的保序性，$0\le Q\le1$ 几乎处处；后文选定满足该界的版本。

## 7.2 Brier 分解

**定理 7.3（Brier 正交分解）.** 设 $Y\in\{0,1\}$，$P\in[0,1]$，并令 $Q=\mathbb E[Y\mid\sigma(P)]$。则

$$
\mathbb E[(Y-P)^2]
=\mathbb E[(Y-Q)^2]+\mathbb E[(Q-P)^2].
$$

此外，若 $\pi=\mathbb EY$，则

$$
\mathbb E[(Y-P)^2]
=\pi(1-\pi)-\operatorname{Var}(Q)+\mathbb E[(Q-P)^2].
$$

**证明.** 写

$$
Y-P=(Y-Q)+(Q-P).
$$

平方展开后的交叉项期望为

$$
\mathbb E[(Y-Q)(Q-P)]
=\mathbb E\!\left[\mathbb E[(Y-Q)(Q-P)\mid\sigma(P)]\right].
$$

$Q-P$ 是 $\sigma(P)$-可测且有界，由拉出性质，上式等于

$$
\mathbb E[(Q-P)\mathbb E[Y-Q\mid\sigma(P)]]=0.
$$

得到第一式。再由 $Y^2=Y$、拉出性质和塔式法则，

$$
\mathbb E[YQ]
=\mathbb E\!\left[Q\,\mathbb E[Y\mid\sigma(P)]\right]
=\mathbb E[Q^2],
$$

故直接展开可得

$$
\mathbb E[(Y-Q)^2]=\mathbb E[Y^2]-\mathbb E[Q^2]
=\pi-\mathbb E[Q^2].
$$

又由全期望 $\mathbb E Q=\mathbb EY=\pi$，所以 $\operatorname{Var}(Q)=\mathbb E[Q^2]-\pi^2$，故

$\mathbb E[(Y-Q)^2]=\pi(1-\pi)-\operatorname{Var}(Q)$。代入即得。证毕。

**例（分解一个失准但有分辨率的预测器）.** 令 $P$ 以相同概率取 $1/4$ 与
$3/4$，并令相应的条件正例率 $Q$ 分别为 $1/2$ 与 $1$。于是

$$
\pi=\mathbb EY=\mathbb EQ=\frac34,
\qquad
\operatorname{Var}(Q)=\frac1{16},
\qquad
\mathbb E[(Q-P)^2]=\frac1{16}.
$$

给定 $Q=1/2$ 时，Bernoulli 条件方差为 $1/4$；给定 $Q=1$ 时为零，因此

$$
\mathbb E[(Y-Q)^2]=\frac12\cdot\frac14=\frac18.
$$

Brier 风险为 $1/8+1/16=3/16$。第二种写法也给出

$$
\pi(1-\pi)-\operatorname{Var}(Q)+\mathbb E[(Q-P)^2]
=\frac3{16}-\frac1{16}+\frac1{16}=\frac3{16}.
$$

$\operatorname{Var}(Q)$ 记录两组真实发生率的差异，$\mathbb E[(Q-P)^2]$ 则记录
报告与组内发生率的偏离；这个算例显示二者可以同时为正。

**推论 7.4（给定预测信息的最优平方预测）.** 对任意 $\sigma(P)$-可测且平方可积的随机变量 $R$，

$$
\mathbb E[(Y-R)^2]
=\mathbb E[(Y-Q)^2]+\mathbb E[(Q-R)^2].
$$

因此 $Q$ 在所有 $\sigma(P)$-可测平方可积预测中最小化平方风险，并且最优者在几乎处处意义下唯一。

**证明.** 令 $H=Q-R$。因为 $Q$ 有界且 $R\in L^2\subseteq L^1$，所以 $H\in L^1$ 且为 $\sigma(P)$-可测。令 $H_n=(-n)\vee(H\wedge n)$。定理 4.5 与 $\mathbb E[Y-Q\mid\sigma(P)]=0$ 给出

$$
\begin{aligned}
\mathbb E[(Y-Q)H_n]
&=\mathbb E\!\left[\mathbb E[(Y-Q)H_n\mid\sigma(P)]\right]\\
&=\mathbb E\!\left[H_n\mathbb E[Y-Q\mid\sigma(P)]\right]=0.
\end{aligned}
$$

又因 $|H_n-H|\le |H|\mathbf1_{\{|H|>n\}}$，积分尾部趋于零，所以 $H_n\to H$ in $L^1$。结合 $|Y-Q|\le1$，

$$
|\mathbb E[(Y-Q)(H_n-H)]|
\le\mathbb E|H_n-H|\to0.
$$

故 $\mathbb E[(Y-Q)(Q-R)]=0$。对 $Y-R=(Y-Q)+(Q-R)$ 平方展开即得分解。第二项非负，且等于零当且仅当 $R=Q$ 几乎处处。证毕。

这里 $\mathbb E[(Y-Q)^2]$ 只是在给定信息 $\sigma(P)$ 和平方损失下的不可约项，不是脱离信息集与损失函数的“本体噪声”。$\mathbb E[(Q-P)^2]$ 是可靠性误差，$\operatorname{Var}(Q)$ 表示预测信息区分不同条件发生率的程度。一个恒报基准率的预测可以完全校准，却没有分辨率。

## 7.3 校准不是充分性能标准

若 $P\equiv\pi=\mathbb P(Y=1)$，则 $P$ 校准，但它不区分样本。反之，高分辨率预测可能存在系统偏差。评分规则把校准与锐度综合到单一期望损失，但不同任务的行动代价并不相同。

多分类校准有多种不等价版本：对完整概率向量条件化、逐类边缘校准、top-label 校准和分组校准。声称“模型已校准”必须说明采用的版本、数据分布和评估误差。

## 7.4 决策与 Bayes 行动

设有限状态集 $\Theta$、有限行动集 $\mathcal A$，损失 $\ell:\mathcal A\times\Theta\to\mathbb R$ 有界。令 $\Theta_0$ 是状态随机变量，$\mathcal G$ 是决策时可用信息。

**定义 7.5（决策规则）.** $\mathcal G$-可测随机变量 $A:\Omega\to\mathcal A$ 称为可行决策规则。其风险为

$$
\mathcal R(A)=\mathbb E[\ell(A,\Theta_0)].
$$

**定理 7.6（Bayes 行动的逐信息最优性）.** 对每个 $a\in\mathcal A$ 定义条件风险的一个版本

$$
r_a=\mathbb E[\ell(a,\Theta_0)\mid\mathcal G].
$$

固定一个行动次序，并令 $A^*$ 在每个样本点选择使所选版本 $r_a$ 最小的第一个行动。则 $A^*$ 是 $\mathcal G$-可测的，且对每个 $\mathcal G$-可测决策规则 $A$，

$$
\mathcal R(A^*)\le\mathcal R(A).
$$

**证明.** 因 $\mathcal A$ 有限，可同时选定全部 $r_a$ 的版本；改变任一版本只会在有限个零集的并上改变 $A^*$，不影响风险。每个事件

$$
\{A^*=a_j\}=\{r_{a_j}<r_{a_i}\text{ 对 }i<j\}
\cap\{r_{a_j}\le r_{a_i}\text{ 对 }i>j\}
$$

属于 $\mathcal G$，故 $A^*$ 可测。逐点有 $r_{A^*}\le r_A$。利用 $A$ 的有限值与条件期望拉出性质，

$$
\mathbb E[\ell(A,\Theta_0)\mid\mathcal G]
=\sum_{a\in\mathcal A}\mathbf1_{\{A=a\}}r_a=r_A.
$$

同一计算给出

$$
\mathbb E[\ell(A^*,\Theta_0)\mid\mathcal G]=r_{A^*}.
$$

由 $A^*$ 的逐点定义，$r_{A^*}\le r_A$ 几乎处处。取期望并使用全期望与保序性，得到 $\mathcal R(A^*)\le\mathcal R(A)$。证毕。

定理先在每个信息状态上比较条件风险，再用全期望汇总；这正是“逐信息最优”能够推出
总体风险最优的原因。有限行动集同时保证最小值达到，并让固定次序的选择器自动可测。

该定理依赖有限行动集和给定损失函数。若行动集无限，风险下确界可能不被任何行动达到；即使逐点达到，也还要证明选择器可测。概率分布不自行指定该做什么；行动选择还需要代价、约束和效用。

## 7.5 不确定性的分解边界

“偶然不确定性”（aleatoric uncertainty）常指给定模型信息后结果仍有随机性；“认知不确定性”（epistemic uncertainty）常指参数、模型类或数据不足造成的不确定。二者不是脱离模型的唯一分解。改变特征集、参数化、先验或观察层级，会改变哪些部分被归入哪一类。

因此严谨表述应说明：

- 概率相对于哪个联合模型；
- 条件信息是什么；
- 参数被视为固定未知量还是随机变量；
- 分解是后验方差、集成方差、条件熵还是其他量。

## 7.6 分布漂移

校准是相对于联合分布 $\mathcal L(P,Y)$ 的性质。部署分布改变后，原校准关系未必保留。仅协变量分布变化、标签机制变化和选择偏差有不同修正条件，不能统称为“漂移后重新校准即可”。

概率预测只有放在联合分布中才能谈校准，只有给出损失才能导出行动。第八章把视角从
统计预测转向实际执行：即使目标核已经给定，程序仍需通过随机源和确定性状态转移把它
实现出来。

## 练习

**练习 7.1.** 证明恒报总体正例率的预测在总体分布上校准。

**练习 7.2.** 构造两个都校准但分辨率不同的有限预测器，并比较 Brier 分数。

**练习 7.3.** 对二分类误判代价 $c_{10},c_{01}>0$ 推导最优概率阈值。

**练习 7.4.** 给出一个分布变化使原本校准的预测器失去校准的例子。

**练习 7.5.** 解释“模型熵高”与“决策风险高”为何既不等价也不互相蕴含。

**练习 7.6.** 给出一个单状态、无限行动集的决策问题，使风险下确界存在但没有 Bayes 行动，从而说明定理 7.6 的有限性假设承担了什么责任。
