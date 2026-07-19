# 第六章 熵、校准与置信

模型给出一组概率后，人们自然想问：它有多不确定？这个数可信么？能否据此拒答？这些问题不能只靠最大概率回答，需要区分分布的离散程度、预测与频率的一致性，以及任务决策的代价。

## 6.1 下一 Token 熵

对词表分布 $q$，Shannon 熵为

$$
H(q)=-\sum_{i\in V}q_i\log q_i.
$$

若 $q$ 集中于一个 token，熵接近 0；若在 $K$ 个 token 上均匀，熵为 $\log K$。熵描述当前 token 分布有多分散，不说明高概率候选是否真实。

tokenizer 改变词表和序列单位，因此不同 tokenizer 的单 token 熵不可直接比较。上下文长度、语言和位置也会系统影响熵。

## 6.2 序列熵

自回归序列的条件熵满足链式法则：

$$
H(Y_{1:T}\mid C)
=\sum_{t=1}^{T}
H(Y_t\mid Y_{<t},C).
$$

右侧每一项是对随机前缀 $Y_{<t}$ 再取期望，不能用某一条 greedy 路径上的 token 熵简单相加替代完整序列熵。实际估计常用 Monte Carlo 采样，结果依赖停止规则和最大长度。

## 6.3 熵不等于“模型不知道”

问题“写一个颜色词”本来就允许许多答案，模型高熵可能合理；问题“2+2 等于多少”若高熵，则更可能反映能力不足或提示歧义。反之，训练数据中一致重复的错误也可能使模型低熵地回答错误。

所以熵必须结合任务条件和外部标签解释。它是分布形状统计量，不是知识状态的直接读数。

## 6.4 NLL 与 Brier Score

对分类结果 $Y$ 和预测概率向量 $q$，常见 proper scoring rules 包括负对数似然

$$
S_{log}(q,Y)=-\log q_Y,
$$

以及 Brier score

$$
S_B(q,Y)=\sum_k(q_k-\mathbf1\{Y=k\})^2.
$$

在真实标签分布固定的理想条件下，它们的期望由报告真实分布最小化。log score 对给真实类别极低概率的惩罚很强；Brier score 有界，更直接反映平方误差。选择指标应匹配任务，不应只挑对某模型更有利的一个。

## 6.5 校准的定义

对二分类预测分数 $S\in[0,1]$ 与标签 $Y\in\{0,1\}$，完全校准要求

$$
\mathbb P(Y=1\mid S=s)=s
$$

在有定义的分数水平上成立。直观上，所有报 0.7 的样本中，约 70% 应为正类。

校准是相对于样本分布定义的。一个模型可在测试集上校准，在时间变化、国家变化或难度变化后失准；整体校准也可能掩盖子群体内的系统偏差。

## 6.6 Reliability Diagram 与 ECE

实践中把分数划入区间 $B_m$，比较每箱平均置信与经验准确率：

$$
\operatorname{conf}(B_m)
=\frac1{|B_m|}\sum_{i\in B_m}S_i,
$$

$$
\operatorname{acc}(B_m)
=\frac1{|B_m|}\sum_{i\in B_m}Y_i.
$$

Expected Calibration Error 常写为

$$
\operatorname{ECE}
=\sum_m\frac{|B_m|}{n}
\left|
\operatorname{acc}(B_m)-
\operatorname{conf}(B_m)
\right|.
$$

ECE 依赖分箱边界、样本量和是否用最大类概率，有限样本下也有偏。它适合作为诊断摘要，不是校准的完整证明；应同时给 reliability plot、样本数和 proper score。

## 6.7 Temperature Scaling

在验证集上选一个标量 $T$，用 $z/T$ 重新 softmax，可以改善分类概率校准而不改变 argmax 顺序。其参数应只在独立校准集上拟合，再在测试集评估。

temperature scaling 只修正全局 logit 尺度。若错误依输入类型、子群体或类别而变化，一个标量无法解决；分布转移后也需重新验证。

## 6.8 从 Token 到答案置信

开放式回答没有天然单一“正确类”。常见构造包括：

- 固定选项的归一化序列分数；
- 对答案字符串所有可接受变体聚合概率；
- 多次采样后按语义等价簇计算频率；
- 训练独立 verifier 或 correctness predictor；
- 让模型输出自报置信度。

每种方法都引入额外假设。语义聚类依赖等价判定器，verifier 可能与生成器共享偏差，自报置信度也必须用外部标签校准。没有任何一种自动把 token 概率变成命题真值概率。

## 6.9 Semantic Entropy

若多条不同措辞表达同一含义，可以先把回答划分为语义簇 $C_1,\ldots,C_K$，再聚合簇概率

$$
Q(C_k)=\sum_{y\in C_k}q(y),
$$

并计算 $H(Q)$。这比逐字符串熵更接近“答案意义是否分歧”。

但开放生成空间无法完全枚举，簇边界也可能由另一个模型判断。实践中的 semantic entropy 是采样与聚类近似，必须报告采样数、聚类规则和失败案例。

## 6.10 Ensemble 分歧

对模型分布 $q_\Theta$，ensemble 平均为 $\bar q=\mathbb E q_\Theta$。一个常用分解是

$$
H(\bar q)
=\mathbb E_\Theta[H(q_\Theta)]
+I(Y;\Theta\mid x).
$$

第一项反映平均模型内熵，互信息项反映模型身份与预测之间的分歧。这个解释只有在 ensemble 样本确实代表所声称的模型分布时才成立；把同一模型几个 temperature 样本当作参数 posterior 是错误的。

## 6.11 选择性预测与拒答

设置信分数为 $s(x)$，只在 $s(x)\ge\tau$ 时回答。coverage 是被回答样本比例，selective risk 是这些样本上的错误率：

$$
\operatorname{coverage}(\tau)
=\mathbb P(s(X)\ge\tau),
$$

$$
R(\tau)
=\mathbb E[
\ell(\hat Y,Y)
\mid s(X)\ge\tau].
$$

理想置信分数应随阈值提高而降低风险。实际系统应报告完整 risk–coverage 曲线，并把拒答本身的业务代价纳入决策。

## 6.12 从概率到行动

若动作 $a$ 的损失为 $L(a,y)$，Bayes 决策规则选择

$$
a^*(x)=\arg\min_a
\sum_y q(y\mid x)L(a,y).
$$

同一概率在不同损失矩阵下可产生不同动作。医疗筛查对漏诊和误报的成本不同，内容推荐与自动付款的风险也不同。概率预测不能替代效用、权限和风险上限的设计。

## 6.13 本章结论

熵测量分布离散程度，proper score 评估概率预测，校准比较置信与经验频率，选择性预测把置信转化为回答或拒答。所有这些量都依赖明确事件、标签和数据分布；它们可以支持决策，却不能自动认证一次自然语言回答。
