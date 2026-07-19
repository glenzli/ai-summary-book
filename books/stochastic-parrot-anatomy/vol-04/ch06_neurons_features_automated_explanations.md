# 第六章 神经元、Feature 与自动解释

最直接的内部观察，是为某个神经元或 feature 收集最高 activation 样本，再尝试用一句话概括它。这个方法产生了许多有启发性的发现，也最容易把相关模式误命名为概念实体。

## 6.1 Activation Dataset

给定单位 $u$，对语料 token 位置 $(x,p)$ 记录 activation

$$
a_u(x,p).
$$

研究者通常检查：

- 最大正 activation 样本；
- 最小或负 activation 样本；
- activation 分位数与长尾；
- 单词、位置、语言和上下文分层；
- 与随机样本匹配的反例；
- 同一文本中的邻近 token。

只看 top-20 容易产生 selection bias：一个单位的大部分中等 activation 可能与 top 样本的故事不符。

## 6.2 Neuron 标签是一条预测假说

若把单位命名为“日期神经元”，应能预测：

- 未见日期格式也会激活；
- 相似数字但非日期的文本较少激活；
- 改变上下文使数字从编号变成日期时 activation 改变；
- activation 强度与下游日期行为存在可检验关系。

标签越宽泛，越容易事后解释所有样本，也越难证伪。高质量说明应包含正条件、排除条件和失败边界。

## 6.3 Polysemanticity

单个神经元可能对多个不相关模式激活，例如既对 DNA 序列又对法律短语响应。原因可能包括：

- 特征在同一标量通道中叠加；
- top 样本遗漏了另一激活簇；
- 上下文门控使同一方向在不同区域承担不同作用；
- 人类标签粒度不合适。

不能假定规模更大的模型必然让原始神经元更单义。模型可表示的特征数量可能远多于 residual 维度。

## 6.4 Distributed Feature

反过来，一个概念可能没有任何单一高选择性神经元，而由方向

$$
v=\sum_i\alpha_i e_i
$$

或更复杂子空间表示。只检查坐标轴会遗漏旋转后的线性 feature。probe、dictionary learning 和 sparse autoencoder 试图寻找更合适的基。

这也说明“删除一个神经元未影响行为”不能证明该概念不存在：信息可能冗余分布。

## 6.5 自动解释流程

一种可扩展流程是：

1. 为目标单位收集高、中、低 activation 文本；
2. 让解释模型生成自然语言假说 $E$；
3. 在新文本上让 $E$ 预测 activation；
4. 比较预测与真实 activation；
5. 搜索反例并修订 $E$。

评分可使用相关系数、排序能力或把 activation 分箱后的分类性能。真正关键是测试样本必须独立于生成解释所见样本。

## 6.6 模拟分数的含义

若解释模型依据说明 $E$ 能预测目标 activation，说明 $E$ 捕捉了某些可泛化输入模式。但它不说明该单位的下游作用，也不保证解释覆盖分布外文本。

解释模型本身可能利用目标模型同源的语言统计，产生共同盲点。人工评审、规则合成样本和 adversarial counterexample 仍有必要。

## 6.7 Feature Visualization 与 Logit Effect

一个单位对哪些文本激活和它促进哪些输出，是两个方向：

- input-side semantics：什么条件使 $a_u$ 增大；
- output-side effect：单位通过 decoder/output weight 提高或抑制什么方向。

对 MLP neuron，可检查 $W_Uw_{out,u}$ 的高低 token；对 SAE feature，可检查 decoder vector 到 logits 的投影。这只是直接效果，后续非线性仍可能改变实际作用。

## 6.8 Concept Activation Vectors

TCAV 类方法用概念正例与对照样本训练方向 $v_C$，再计算目标对该方向的敏感性

$$
\nabla_h S\cdot v_C.
$$

它把人类定义的概念引入分析，适合检验高层属性；结果依概念数据、随机对照和层选择。不同概念方向可能相关，不能把每个 TCAV 分数当作独立因果份额。

## 6.9 抑制与负 Feature

研究常偏好高正 activation，但抑制同样重要。一个 feature 可以在否定、冲突或安全条件下减少某 logit；ReLU SAE 的非负 latent 仍可通过 decoder vector 产生负 logit effect。

完整说明应包含促进和抑制对象，避免只讲“检测到 X”而忽略它实际在计算中阻止了什么。

## 6.10 自动化不等于闭环

自动解释可以把人工逐神经元检查扩展到数十万单位，但存在三个开放问题：

- **覆盖率**：未被简短语言解释的单位占多少？
- **可信评分**：自动 grader 是否会奖励听起来合理的说明？
- **机制连接**：单位说明怎样组成行为级计算？

单元级自动标签是 feature discovery 的入口，不是完整机制图。后两章将用干预和回路连接这些单位。
