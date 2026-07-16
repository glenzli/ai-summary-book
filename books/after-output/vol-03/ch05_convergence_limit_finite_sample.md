# 第五章 收敛、极限定理与有限样本界

几乎处处收敛、依概率收敛、依分布收敛和均方收敛分别控制不同对象，它们之间只有在明确条件下才能推出。大数定律与中心极限定理描述渐近行为，却不直接回答有限数据下误差超过阈值的概率有多大。

教材把收敛模式、极限定理和集中不等式置于同一章，是为了防止用极限符号替代样本量论证。后续任何“随着运行次数增加而稳定”的主张，都必须指出收敛关系、所需矩条件，以及有限样本时实际采用的置信界或误差界。

## 第五章：收敛、不等式与大数定律

“误差趋于零”可能指逐条样本路径最终稳定，也可能只指偏差超过固定阈值的概率趋于
零，还可能指分布函数在极限中逼近。不同说法允许的换极限、取期望和统计解释并不相同。

一个简单的尖峰序列已经能看出差别。令 $U$ 在 $[0,1]$ 上均匀，并设
$X_n=n\mathbf1_{\{U\le1/n\}}$。除零测点 $U=0$ 外，每条路径最终都等于零，
但 $\mathbb E X_n=1$ 始终不变。本章建立四种标准收敛概念，随后用它们精确陈述
样本均值的大数律。支配收敛、Vitali 收敛与中心极限定理按正文所列版本作为外部输入。

### P5.1 四种收敛


除非另有说明，本节的 $X_n,X$ 都是实随机变量。几乎处处、$L^p$ 与依概率收敛要求它们定义在同一概率空间上；依分布收敛只比较分布，可以定义在不同概率空间上。

**定义 P5.1（几乎处处收敛）.** 若

$$
\mathbb P(\{\omega:X_n(\omega)\to X(\omega)\})=1,
$$

则记 $X_n\to X$ a.s.

事件 $\{X_n\to X\}$ 是可测的，因为它可写为

$$
\bigcap_{m\ge1}\bigcup_{N\ge1}\bigcap_{n\ge N}
\{|X_n-X|<1/m\}.
$$

**定义 P5.2（$L^p$ 收敛）.** 对 $p\ge1$，若每个 $X_n,X\in L^p$ 且

$$
\mathbb E|X_n-X|^p\to0,
$$

则记 $X_n\to X$ in $L^p$。

**定义 P5.3（依概率收敛）.** 若对每个 $\varepsilon>0$，

$$
\mathbb P(|X_n-X|>\varepsilon)\to0,
$$

则记 $X_n\xrightarrow{\mathbb P}X$。

**定义 P5.4（依分布收敛）.** 若对每个有界连续函数 $f:\mathbb R\to\mathbb R$，

$$
\mathbb E[f(X_n)]\to\mathbb E[f(X)],
$$

则记 $X_n\Rightarrow X$。这一定义只依赖边缘分布，因而不要求 $X_n$ 与 $X$ 位于同一概率空间。

**定理 P5.5（$L^p$ 收敛蕴含依概率收敛）.** 若 $X_n\to X$ in $L^p$，$p\ge1$，则 $X_n\xrightarrow{\mathbb P}X$。

**证明.** 对 $\varepsilon>0$，对非负随机变量 $|X_n-X|^p$ 应用 Markov 不等式：

$$
\mathbb P(|X_n-X|>\varepsilon)
\le\frac{\mathbb E|X_n-X|^p}{\varepsilon^p}\to0.
$$

证毕。

**定理 P5.6（几乎处处收敛蕴含依概率收敛）.** 若 $X_n\to X$ a.s.，则 $X_n\xrightarrow{\mathbb P}X$。

**证明.** 固定 $\varepsilon>0$，令

$$
A_m=\bigcup_{n\ge m}\{|X_n-X|>\varepsilon\}.
$$

$A_m$ 随 $m$ 递减，且其交集是事件 $\{|X_n-X|>\varepsilon\text{ 无穷多次}\}$。几乎处处收敛说明该交集概率为零。由概率从上连续性，$\mathbb P(A_m)\downarrow0$。又对每个 $n\ge m$，
$\mathbb P(|X_n-X|>\varepsilon)\le\mathbb P(A_m)$，故所求概率趋于零。证毕。

**外部输入 P5.7（依概率收敛蕴含依分布收敛）.** 若 $X_n\xrightarrow{\mathbb P}X$，则 $X_n\Rightarrow X$。这是 Portmanteau 理论的标准推论。反向一般不成立；若 $X_n$ 与 $X\equiv c$ 定义在同一概率空间上，则 $X_n\Rightarrow c$ 等价于 $X_n\xrightarrow{\mathbb P}c$。所用版本与来源见 [PROBABILITY_SOURCES.md](PROBABILITY_SOURCES.md) 的弱收敛条目。

**定理 P5.8（依概率收敛的几乎处处收敛子列）.** 若 $X_n\xrightarrow{\mathbb P}X$，则存在严格递增的指标序列 $(n_k)$，使 $X_{n_k}\to X$ 几乎处处。

**证明.** 由依概率收敛，可递归选取 $n_k>n_{k-1}$，使

$$
\mathbb P(|X_{n_k}-X|>2^{-k})\le2^{-k}.
$$

令 $E_k=\{|X_{n_k}-X|>2^{-k}\}$。对每个 $m$，由并集上界

$$
\mathbb P\left(\bigcup_{k\ge m}E_k\right)
\le\sum_{k\ge m}2^{-k}\longrightarrow0.
$$

事件 $\limsup_kE_k=\bigcap_m\bigcup_{k\ge m}E_k$ 的概率因而为零。在其补集上，只有有限多个 $E_k$ 发生，所以最终 $|X_{n_k}-X|\le2^{-k}\to0$。证毕。

**定义 P5.9（一致可积）.** 一族可积随机变量 $\mathcal X$ 称为一致可积，若

$$
\lim_{M\to\infty}
\sup_{Z\in\mathcal X}
\mathbb E\bigl[|Z|\mathbf1_{\{|Z|>M\}}\bigr]=0.
$$

**外部输入 P5.10（支配收敛与 Vitali 收敛）.** 本书使用以下两个标准版本：

1. 若 $X_n\to X$ 几乎处处，$p\ge1$，并存在 $Y\in L^p$ 使 $|X_n|\le Y$ 对所有 $n$ 几乎处处成立，则 $X\in L^p$ 且 $X_n\to X$ in $L^p$。
2. 若 $X_n\xrightarrow{\mathbb P}X$，并且族 $\{X_n:n\ge1\}$ 一致可积，则 $X\in L^1$ 且 $X_n\to X$ in $L^1$。反之，$L^1$ 收敛蕴含依概率收敛，并且 $\{X_n:n\ge1\}$ 一致可积。

第一项对 $|X_n-X|^p$ 应用 Lebesgue 支配收敛定理；第二项是 Vitali 收敛定理。完整证明依赖积分的绝对连续性与截断论证，作为测度论外部输入，来源见 [PROBABILITY_SOURCES.md](PROBABILITY_SOURCES.md)。

因此无附加条件时，图式中只有

$$
L^p\Longrightarrow\mathbb P,
\qquad
\text{a.s.}\Longrightarrow\mathbb P\Longrightarrow\text{distribution}.
$$

几乎处处收敛与 $L^p$ 收敛彼此一般不可比较；依概率收敛也不保证 $L^1$ 收敛。定理 P5.8 只保证可选出几乎处处收敛的子列，不把原序列升级为几乎处处收敛。

章首的尖峰序列把这条边界算得很清楚。对每个 $u>0$，取 $N>1/u$，则
$n\ge N$ 时 $u>1/n$，所以 $X_n(u)=0$；故 $X_n\to0$ 几乎处处。另一方面，

$$
\mathbb E|X_n-0|=n\,\mathbb P(U\le1/n)=1,
$$

所以它不在 $L^1$ 中收敛到零。问题来自越来越高、越来越窄而总面积不变的尖峰；
这里不存在一个可积函数同时支配全部 $X_n$，族 $\{X_n\}$ 也不一致可积。

### P5.2 样本均值


设 $X_1,X_2,\ldots$ 独立同分布，且 $\mu=\mathbb EX_1$ 存在。定义

$$
\overline X_n=\frac1n\sum_{i=1}^nX_i.
$$

样本均值的方差为何会缩小，是弱大数律最直接的机制：独立性消去协方差，除以 $n$
又把方差从 $n\sigma^2$ 缩到 $\sigma^2/n$。下一定理把这个计算转换成尾概率界。

**定理 P5.11（弱大数律，i.i.d. 有限方差版本）.** 若 $X_1,X_2,\ldots$ 独立同分布且 $\operatorname{Var}(X_1)=\sigma^2<\infty$，令 $\mu=\mathbb E X_1$。则

$$
\overline X_n\xrightarrow{\mathbb P}\mu.
$$

更具体地，对每个 $\varepsilon>0$，

$$
\mathbb P(|\overline X_n-\mu|\ge\varepsilon)
\le\frac{\sigma^2}{n\varepsilon^2}.
$$

**证明.** 由期望线性性，$\mathbb E\overline X_n=\mu$。由独立和的方差公式，

$$
\operatorname{Var}(\overline X_n)
=\frac1{n^2}\sum_{i=1}^n\operatorname{Var}(X_i)
=\frac{\sigma^2}{n}.
$$

对 $\overline X_n$ 使用 Chebyshev 不等式即得概率界；右侧趋于零，所以依概率收敛。证毕。

证明只用了期望线性、独立和的方差公式与 Chebyshev 不等式。有限方差正是在最后一步
提供 $1/n$ 控制的假设；强大数律能把条件放宽到可积，但需要更深的截断与路径论证。

该定理陈述的是重复独立采样下样本均值的收敛。它不保证有限样本误差很小，不保证非独立样本，也不保证数据分布在采集过程中保持不变。

**外部输入 P5.12（Kolmogorov 强大数律，i.i.d. 可积版本）.** 若 $X_1,X_2,\ldots$ 独立同分布且 $\mathbb E|X_1|<\infty$，则

$$
\overline X_n\to\mathbb EX_1\quad\text{几乎处处}.
$$

本书使用该结论说明长期频率的路径级收敛，不重证其截断、最大不等式与 Borel--Cantelli 论证。来源见 [PROBABILITY_SOURCES.md](PROBABILITY_SOURCES.md)。

**外部输入 P5.13（Lindeberg--Levy 中心极限定理）.** 若 $X_1,X_2,\ldots$ 独立同分布，$\mathbb EX_1=\mu$，且 $0<\operatorname{Var}(X_1)=\sigma^2<\infty$，则

$$
\frac{\sqrt n(\overline X_n-\mu)}{\sigma}
\Rightarrow N(0,1).
$$

该定理给出归一化误差的极限分布，不说有限 $n$ 时误差恰好服从正态分布。若方差为零，标准化无定义且样本均值恒等于 $\mu$；若方差无限，该版本不适用。所用 i.i.d. 版本与来源见 [PROBABILITY_SOURCES.md](PROBABILITY_SOURCES.md)。

### P5.3 浓缩界与误用边界


Chebyshev 界只使用方差，通常较松。“变量有界时有指数尾界”需要独立性和明确常数，不能只靠有界性口号替代。

**外部输入 P5.14（Hoeffding 不等式）.** 设 $X_1,\ldots,X_n$ 相互独立，并且存在实数 $a_i\le b_i$ 使

$$
\mathbb P(a_i\le X_i\le b_i)=1
$$

对每个 $i$ 成立。令 $S_n=\sum_{i=1}^n(X_i-\mathbb EX_i)$。则对每个 $t>0$，

$$
\mathbb P(S_n\ge t)
\le\exp\left(-\frac{2t^2}{\sum_{i=1}^n(b_i-a_i)^2}\right),
$$

并且

$$
\mathbb P(|S_n|\ge t)
\le2\exp\left(-\frac{2t^2}{\sum_{i=1}^n(b_i-a_i)^2}\right).
$$

其中假设 $\sum_i(b_i-a_i)^2>0$。这是 Hoeffding 1963 年论文的有界独立变量版本，本书不重证指数矩方法。若某些变量退化，可令 $a_i=b_i$ 并删去对应项；若全部退化，则 $S_n=0$ 几乎处处，尾概率直接为零。

“大概率”必须带参数。例如“事件 $A_n$ 以大概率发生”通常表示 $\mathbb P(A_n)\to1$；若只给一个固定事件，应写明确数值下界。表达式 $1-o(1)$ 还必须说明趋于无穷的变量。

### P5.4 从总体到数据集


统计定理的概率空间通常包含数据采样。训练完成后固定的数据集是一个实现对象，不再自动携带“每次都重新抽样”的物理过程。用同一个固定测试集反复调参会改变选择机制；即使每个候选模型的单次评估无偏，最终被选择模型的报告分数也可能产生选择偏差。

一个有限反例已经足够说明该逻辑。令两个候选模型的评估误差 $Z_1,Z_2$ 独立且各以概率 $1/2$ 取 $-1$ 或 $1$，于是 $\mathbb EZ_i=0$。若选择报告分数较大的模型，则报告误差为 $\max(Z_1,Z_2)$；它只在二者都为 $-1$ 时等于 $-1$，其余情形等于 $1$，所以

$$
\mathbb E[\max(Z_1,Z_2)]
=-\tfrac14+\tfrac34
=\tfrac12>0.
$$

偏差来自“按同一噪声评估选择最大值”的机制，不是大数律失效。

这不是大数律的反例，而是抽样与选择过程已经不满足原定理的固定假设。

收敛概念规定了“极限”究竟控制路径、矩还是分布；大数律又把这种控制建立在明确的
采样机制上。下一章不再研究样本数量趋于无穷，而研究如何用熵、散度和评分规则比较
两个已经给定的概率分布。

### 练习


**练习 P5.1.** 构造依概率收敛但不几乎处处收敛的随机变量序列。

**练习 P5.2.** 构造 $X_n\to0$ 依概率成立但 $\mathbb E|X_n|\not\to0$ 的例子。

**练习 P5.3.** 用弱大数律给出公平硬币正面频率偏离 $1/2$ 至少 $0.1$ 的 Chebyshev 上界。

**练习 P5.4.** 说明中心极限定理为何不蕴含 $\overline X_n$ 自身收敛到正态分布。

**练习 P5.5.** 列出把一个固定基准集分数解释为总体性能估计时所需的抽样与选择假设。

**练习 P5.6.** 对 i.i.d. Bernoulli$(1/2)$ 变量，分别用 Chebyshev 与 Hoeffding 不等式界定 $\mathbb P(|\overline X_n-1/2|\ge\varepsilon)$，比较两者随 $n$ 的衰减速度。
