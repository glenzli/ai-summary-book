# 第六章 熵、Proper Score、校准与选择性预测

模型输出概率后，人们会问：分布有多分散？概率预测是否可信？系统能否据此拒答？这三个问题分别对应熵、概率评分与校准、选择性决策。它们共享概率记号，却不互相等价。

## 6.1 先确定不确定性的对象

至少存在四种常见对象：

1. 下一 token 分布 $q_\theta(Y_t\mid y_{<t},c)$；
2. 完整序列或语义事件分布 $Q_g(C\mid c)$；
3. 给定输入时，任务标签或回答正确性 $P(Z=1\mid X)$；
4. 不同训练模型 $q_\Theta$ 之间的预测分歧。

某一步 token 熵不能直接替代整段答案正确性的概率，模型间分歧也不能由同一 checkpoint 的重复采样替代。后续每个指标都要声明它作用于哪一对象。

## 6.2 下一 token 熵

对有限词表分布 $q$，Shannon 熵为

$$
H(q)=-\sum_{i\in V}q_i\log q_i.
$$

若 $q$ 集中于单一 token，熵为 0；若在 $K$ 个 token 上均匀，熵为 $\log K$。熵测量分布的离散程度，不使用真实标签。

tokenizer 改变词表和序列单位，因而不同 tokenizer 的单 token 熵不可直接比较。语言、位置、上下文长度和解码前后阶段也会系统改变熵。

高熵可能是任务合理多解，也可能是提示含糊或模型能力不足；低熵可能是正确掌握，也可能是数据中一致错误导致的自信误判。

## 6.3 序列熵的链式法则

对固定长度随机序列，条件熵满足

$$
H(Y_{1:T}\mid C)
=\sum_{t=1}^{T}
H(Y_t\mid Y_{<t},C).
$$

更完整地，右侧每项是

$$
\mathbb E_{Y_{<t}\mid C}
\left[
H(q_t(\cdot\mid Y_{<t},C))
\right].
$$

所以不能把某一条 greedy 路径上的 token 熵简单相加，当作完整序列熵；greedy 前缀没有按模型路径分布取期望。

可变长回答还需把 EOS 或停止时间纳入随机变量。实践中用 Monte Carlo 估计序列熵时，结果依赖最大长度、截断、采样分布和尾部处理。

## 6.4 Proper scoring rule 的定义

设真实类别分布为 $p$，模型报告 $q$，结果为 $Y$。采用“损失越小越好”的约定，评分规则 $S(q,Y)$ 称为 proper，如果

$$
\mathbb E_{Y\sim p}S(p,Y)
\le
\mathbb E_{Y\sim p}S(q,Y)
$$

对所有 $p,q$ 成立；若等号只在 $q=p$ 时成立，则称 strictly proper。

proper 性质针对重复事件的期望评分。它不保证有限测试集上真实报告每次都赢，也不说明报告的 $q$ 已经是世界真值概率；后者仍取决于事件和数据总体是否定义正确。

## 6.5 Log score 与 Brier score

分类问题中常用负对数评分

$$
S_{\mathrm{log}}(q,Y)=-\log q_Y
$$

和 Brier score

$$
S_B(q,Y)=\sum_{k=1}^K
(q_k-\mathbf 1\{Y=k\})^2.
$$

对数评分的期望差等于 KL：

$$
\mathbb E_pS_{\mathrm{log}}(q,Y)
-\mathbb E_pS_{\mathrm{log}}(p,Y)
=D_{\mathrm{KL}}(p\Vert q)\ge0.
$$

Brier score 也有直接分解：

$$
\mathbb E_pS_B(q,Y)
=\lVert q-p\rVert_2^2
+1-\lVert p\rVert_2^2.
$$

第二项与 $q$ 无关，故 Brier 由 $q=p$ 唯一最小化。两者都是 strictly proper，但敏感性不同：log score 对真实类别获得极小概率的情况惩罚无界；多分类 Brier 有界于 $[0,2]$。

accuracy 只检查 argmax，不是整个概率向量的 strictly proper score。两个模型可有相同 accuracy，却在错误概率大小和下游决策风险上完全不同。

## 6.6 强校准、类别校准与置信校准

令概率分类器输出随机向量

$$
Q=Q(X)\in\Delta^{K-1}.
$$

**分布校准**要求对每个类别 $k$，

$$
\mathbb E[\mathbf 1\{Y=k\}\mid Q]=Q_k
\quad\text{几乎处处成立。}
$$

**类别校准**只要求随机变量等式

$$
\mathbb E[\mathbf 1\{Y=k\}\mid Q_k]=Q_k
\quad\text{a.s.}
$$

**最大置信校准**令

$$
\widehat Y=\arg\max_kQ_k,
\qquad S=\max_kQ_k,
$$

其中并列时采用固定规则，并要求

$$
\mathbb E[\mathbf 1\{Y=\widehat Y\}\mid S]=S
\quad\text{a.s.}
$$

若 $Q_k$ 或 $S$ 连续，事件 $Q_k=s$、$S=s$ 通常具有零概率，因此逐点写法依赖正则条件概率版本，并只能在相应分布几乎处处解释。以上条件期望等式避免把零概率点误写成额外约束。

后两者比完整向量校准弱。只检查最大类概率，可能看不见其余类别质量怎样分配；而不同错误类别在决策中可能有完全不同成本。

校准总是相对于 $(X,Y)$ 的评测分布和标签协议定义。同一模型可总体校准但在语言、时间或难度子群内失准。

## 6.7 校准不等于信息充分

设二分类总体正例率为 $\pi$。恒定预测

$$
Q(X)=(1-\pi,\pi)
$$

在总体上完全校准，因为所有报 $\pi$ 的样本中正例频率正是 $\pi$。但它完全不利用 $X$，没有个体区分能力。

因此校准必须与 sharpness、resolution 或 discrimination 一同评价。直观上，在保持校准的预测中，希望概率能随输入变化并区分不同风险；AUROC 等排序指标可测区分能力，却又不保证概率尺度校准。

另一个反例是完美分类但概率失准：若每个样本 argmax 都正确，模型 accuracy 为 1；若它始终只给正确类 0.6，则最大置信校准仍失败，因为报 0.6 的样本实际 100% 正确。

## 6.8 Reliability diagram 与 ECE

对二元正确性 $Z_i=\mathbf 1\{Y_i=\widehat Y_i\}$ 和最大置信 $S_i$，将分数划入 bins $B_m$；以下只对非空 bin 计算：

$$
\operatorname{conf}(B_m)
=\frac1{|B_m|}\sum_{i\in B_m}S_i,
$$

$$
\operatorname{acc}(B_m)
=\frac1{|B_m|}\sum_{i\in B_m}Z_i.
$$

常见 ECE 为

$$
\widehat{\operatorname{ECE}}
=\sum_m\frac{|B_m|}{n}
\left|
\operatorname{acc}(B_m)
-\operatorname{conf}(B_m)
\right|.
$$

它不是一个无条件可靠的“校准证明”，原因包括：

- 数值依赖 bin 数量、边界和等宽/等频策略；
- 有限样本的箱内准确率有抽样误差；
- 只用最大置信会忽略完整向量；
- 粗分箱可以让相反方向的失准互相抵消；
- 同一数据上选分箱或校准器再报告会产生乐观偏差。

例如等量两组样本落入同一 bin：第一组置信 0.7、准确率 0.9，第二组置信 0.9、准确率 0.7。合并后平均置信与准确率都为 0.8，该 bin 的 ECE 贡献为 0，尽管两个分数水平都失准。

因此应同时报告 reliability plot、每箱样本数和区间、NLL/Brier 等 proper score，并做类别与关键子群诊断。

还不能把 ECE 直接当成 strictly proper 的训练目标。将所有样本退化为总体基率预测，可能改善校准却摧毁分辨能力；有限样本校准指标本身还可能奖励对分数做不诚实的合并。校准诊断、概率评分和任务决策应分别报告。

## 6.9 Temperature scaling 的能力边界

在独立 calibration set 上选择标量 $T>0$，令

$$
Q_T(X)=\operatorname{softmax}(z(X)/T),
$$

通常通过最小化校准集 NLL 拟合。正 $T$ 不改变类别排序与 argmax，因此可在保持 accuracy 的同时修正全局 logit 尺度。

它不能修复：

- 输入类型或类别特异的失准；
- 错误候选排序；
- 候选集定义错误；
- 开放回答到正确性事件的映射错误；
- 分布漂移后条件关系改变。

校准器参数必须在与最终测试集独立的数据上拟合。若反复查看测试 ECE 选择 $T$，测试集已参与训练。

## 6.10 从 token 概率到答案置信

开放回答没有天然的有限正确类。常见置信构造包括：

- 固定候选答案的条件化序列概率；
- 对可接受字符串集合聚合模型质量；
- 多次采样后估计语义事件频率；
- 训练独立 correctness predictor 或 verifier；
- 让生成模型自报数字或置信措辞。

每种构造定义了新的随机变量。候选概率遗漏候选外质量；字符串集合很难枚举；verifier 可能与生成器共享错误；自报概率只是另一段生成文本。任何分数只有在明确标签和独立样本上经过 proper-score 与校准检验后，才可解释为特定任务的经验置信。

“模型给所选答案的平均 token 概率”不是完整答案正确性的概率。它既受长度和 tokenizer 影响，也没有把同义回答聚合为事件。

## 6.11 语义事件与 semantic entropy

若语义映射 $g$ 把文本划分为有限或可数个互斥簇 $(C_k)_{k\in\mathcal K}$，精确簇概率是

$$
Q(C_k)=\sum_{y:g(y)=C_k}q(y),
$$

semantic entropy 定义为

$$
H_g(Y)=-\sum_{k\in\mathcal K}Q(C_k)\log Q(C_k),
$$

其值允许为 $+\infty$。它对被 $g$ 合并的同义措辞不变，因而比逐字符串熵更接近“回答意义是否分歧”。但该定义与实际估计之间有三层误差：

1. 只能从巨大输出空间抽取有限样本；
2. 聚类或双向蕴含判定器会误合并、误拆分；
3. 样本可能来自 temperature/top-p 后分布而非基础 $q$。

若 $n_k$ 是 $n$ 个样本中簇 $k$ 的计数，插件估计

$$
\widehat H_g
=-\sum_{k:n_k>0}\frac{n_k}{n}
\log\frac{n_k}{n}
$$

通常因未见稀有簇而向下偏。报告 semantic entropy 时必须给出采样数、解码协议、聚类规则、未判定样本和人工复核误差；它不是不依赖语义模型的“内部真值”。

## 6.12 Ensemble 分歧与互信息

对训练运行分布中的模型 $q_\Theta$，固定输入 $x$，并假设所讨论事件空间有限，令

$$
\bar q=\mathbb E_\Theta q_\Theta.
$$

有恒等分解

$$
H(\bar q)
=\mathbb E_\Theta H(q_\Theta)
+I(Y;\Theta\mid x),
$$

其中

$$
I(Y;\Theta\mid x)
=\mathbb E_\Theta
D_{\mathrm{KL}}(q_\Theta\Vert\bar q)\ge0.
$$

第一项是平均模型内熵，第二项量化知道模型身份后预测熵减少多少，即模型间分歧。等式本身严格成立；把第二项解释为“epistemic uncertainty”则还要求 ensemble 样本代表所声称的模型集合。

一个手工挑选的模型集合、不同架构的产品集合与近似 posterior 样本对应不同 $\Theta$ 分布，数值不可脱离构造比较。

## 6.13 选择性预测与拒答

设置信分数为 $s(X)$，选择器为

$$
a_\tau(X)=\mathbf 1\{s(X)\ge\tau\}.
$$

coverage 与 selective risk 定义为

$$
C(\tau)=P(a_\tau(X)=1),
$$

$$
R(\tau)
=\mathbb E[\ell(\widehat Y,Y)
\mid a_\tau(X)=1],
$$

前提是 $C(\tau)>0$。完整 risk-coverage 曲线比单一阈值更能展示拒答代价。

高分阈值并不从定义上保证风险单调下降；只有当 $s$ 确实按条件错误风险正确排序时才会出现理想曲线。阈值应在验证或校准数据上选择，再在独立测试和部署子群上评估。

若阈值预先固定，测试样本 iid，接受的 $m$ 个损失独立且位于 $[0,1]$，Hoeffding 界给出

$$
R(\tau)
\le
\widehat R(\tau)
+\sqrt{\frac{\log(1/\delta)}{2m}}
$$

条件于 $m>0$，以至少 $1-\delta$ 的概率成立。若用同一测试集搜索许多阈值、请求相关或部署漂移，该简单保证不再适用。

conformal 或风险控制方法能在交换性等明确条件下给出更专门的有限样本保证；它们也不会在任意漂移下自动维持条件正确率。

## 6.14 从概率到行动

若动作 $a$ 的损失为 $L(a,y)$，Bayes 决策规则为

$$
a^*(x)
=\arg\min_a
\sum_yq(y\mid x)L(a,y).
$$

概率预测与损失函数共同决定动作。相同的 1% 失败概率，在自动付款、内容推荐和可撤销草稿中对应不同决策；拒答、转人工与延迟也有成本。

若 $q$ 未校准、事件定义错位或部署分布改变，代入精确决策公式仍会得到错误行动。数学最优只相对于输入的概率模型与损失矩阵成立。

## 6.15 本章结论

熵描述分布形状，strictly proper score 评价整个概率报告，校准比较预测与重复事件频率，选择性预测则把置信分数转化为回答或拒答。校准不蕴含信息充分，ECE 也不是校准证书；开放回答还必须显式构造语义事件。下一章将把这些量放进可执行实验，处理 Monte Carlo 区间、相关样本、配对比较和分布漂移。
