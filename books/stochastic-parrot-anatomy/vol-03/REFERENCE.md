# 符号与术语

| 符号或术语 | 本卷含义 |
|---|---|
| $P_D$ | 经过记录、选择和处理后形成的数据分布 |
| $P_{train}$ | 数据混合与采样权重实际定义的训练分布 |
| $q_\theta(y\mid x)$ | 固定参数模型对输出的条件分布 |
| $z_i$ | token $i$ 的 softmax 前 logit |
| $\Theta$ | 训练算法在随机状态和执行环境作用下产生的参数随机变量 |
| empirical risk | 在有限训练样本及其权重上求得的平均损失 |
| cross-entropy | $H(p,q)=-\sum p\log q$ |
| KL divergence | $D_{KL}(p\Vert q)=\sum p\log(p/q)$；非对称且不是距离 |
| NLL | negative log-likelihood，已观测目标的负对数概率 |
| perplexity | 平均 token NLL 的指数；跨 tokenizer 不宜直接比较 |
| entropy | 分布离散程度 $H(q)=-\sum q\log q$ |
| calibration | 给定预测分数的样本，其经验频率与该分数相符的性质 |
| proper score | 期望上由报告目标分布最小化的概率评分规则 |
| aleatoric / 条件内多样性 | 给定所采用条件后仍存在的结果多样性 |
| epistemic / 模型不确定性 | 有限数据、模型族与优化造成的可约减模型差异 |
| model-internal variation | 固定 checkpoint 后由采样或环境产生的输出变化 |
| model-between variation | 不同训练运行或模型样本之间的预测变化 |
| base distribution | 基础模型原始 logits 经标准 softmax 得到的分布 |
| system distribution | 经温度、截断、约束、过滤与停止后的用户可见分布 |
| iid | 独立同分布；实验假设，不是重复请求自动具备的性质 |
| semantic event | 由外部规则把多个字符串聚合成同一含义事件 |
| $do(X=x)$ | 因果模型中主动设置 $X$ 的干预算子，不等同于观察条件 $X=x$ |

所有概率陈述都应注明事件、条件、模型/系统层和数据总体。若这些对象不同，相同的数值记号不表示同一个概率。
