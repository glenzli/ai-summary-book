# 符号与术语

## 概率层次

| 符号或术语 | 本卷含义 |
|---|---|
| $P_W$ | 所讨论的世界或任务总体；通常不由训练数据直接观测 |
| $P_R$ | 经记录与发布机制选择后的分布 |
| $P_D$ | 经采集、过滤和数据构造后的观测分布 |
| $P_{\mathrm{train}}$ | 来源混合、采样与 token 权重实际定义的训练分布 |
| $\widehat P_n,\widehat P_w$ | 未加权与加权经验分布 |
| $q_\theta(y\mid x)$ | 固定参数基础模型对 token 或序列的条件分布 |
| $K_{\theta,\delta}(y\mid x)$ | 模型加解码配置 $\delta$ 后诱导的生成分布 |
| $P_{\mathrm{sys}}(y\mid r)$ | 给定版本化请求 $r$，继续经过路由、过滤、工具、失败处理和显示映射后的系统可观察分布 |
| $g(Y)$ | 将文本映射为任务或语义事件后的随机变量 |

## 模型与训练

| 符号或术语 | 本卷含义 |
|---|---|
| $z_i$ | token $i$ 的 softmax 前 logit；只在差值上有概率意义 |
| $\theta$ | 一个固定 checkpoint 的参数 |
| $\Theta$ | 训练算法在规定随机化设计下产生的参数随机变量 |
| $\mathcal T$ | 固定架构下允许的参数空间 |
| $R(\theta)$ | 某个总体分布上的期望风险 |
| $\widehat R_n(\theta)$ | 有限数据及其权重上的经验风险 |
| KL projection | 在受限模型族中最小化输入加权 KL；不是欧氏正交投影 |
| teacher forcing | 训练或评分时使用数据/候选的真实前缀作为下一步条件 |
| base distribution | 原始模型 logits 经规定 softmax 得到的分布 |
| decoded distribution | temperature、截断、约束等逐步处理后形成的分布 |

## 四类变异

| 术语 | 本卷用法 |
|---|---|
| 数据或条件内不确定性 | 在所采用可见条件下，数据/任务仍允许多个结果 |
| 模型或参数不确定性 | 有限数据、模型族与训练运行造成的模型间预测差异 |
| 随机算法 | 初始化、洗牌、dropout、采样等显式消费伪随机状态的步骤 |
| 系统非确定性 | 并行顺序、动态批处理、滚动部署、缓存或工具状态造成的变化 |
| model-internal variation | 固定 checkpoint 后由解码和环境产生的运行内变异 |
| model-between variation | 不同训练运行或明确模型集合之间的预测变异 |
| aleatoric / epistemic | 仅在条件集和模型集合明确时使用的文献术语；不覆盖全部执行差异 |

## 评分、校准与决策

| 术语 | 本卷含义 |
|---|---|
| NLL / log score | 已观测结果的 $-\log q_Y$ |
| perplexity | 平均 token NLL 的指数；跨 tokenizer 不宜直接比较 |
| entropy | 分布离散程度 $H(q)=-\sum q\log q$，不使用真值标签 |
| cross-entropy | $H(p,q)=-\sum p\log q$ |
| KL divergence | $D_{\mathrm{KL}}(p\Vert q)=\sum p\log(p/q)$；非对称且不是距离 |
| proper score | 期望上由报告目标分布达到最优的概率评分规则 |
| strictly proper | 期望最优报告唯一为目标分布 |
| distribution calibration | $P(Y=k\mid Q)=Q_k$ 对每类几乎处处成立 |
| classwise calibration | $P(Y=k\mid Q_k=s)=s$ |
| confidence calibration | 最大类预测正确率在给定最大置信下等于该置信 |
| ECE | 基于分箱的经验校准误差；依赖分箱和有限样本，不是校准证明 |
| coverage | 选择器愿意回答的样本比例 |
| selective risk | 条件于系统回答时的期望损失 |
| semantic event | 由显式映射把多个字符串聚合成的语义类别 |

## 统计分析

| 术语 | 本卷含义 |
|---|---|
| estimand | 希望从规定总体和随机机制中估计的目标量 |
| estimator | 由观测数据计算 estimand 的规则 |
| standard deviation | 个体随机结果的离散程度 |
| standard error | 某个估计量在重复抽样中的离散程度 |
| iid | 独立同分布；实验假设，不是重复请求自动具备的性质 |
| paired comparison | 在同一实验单位上比较两个系统并分析差值 |
| cluster/hierarchical bootstrap | 按真实依赖层级重采样，而非把全部输出视为独立 |
| importance weight | 目标分布相对于源分布的密度比 $dQ/dP$ |
| support / positivity | 目标有质量处源分布也必须有质量，方可由重加权识别 |
| identifiable | 所有产生同一观测分布的机制都给出相同目标量 |
| $do(X=x)$ | 因果模型中主动设置 $X$ 的干预，不等同于观察条件 $X=x$ |

所有概率陈述都应注明事件、条件、模型或系统层、目标总体和时间窗口。若这些对象不同，相同记号或相同数值也不表示同一个概率。
