# 第三章 从 Logits 到下一个 Token

模型前向的直接输出不是中文词，也不是事实置信度，而是词表上每个 token 的一个实数分数。

## 3.1 Vocabulary Projection

设最后隐藏状态为 $h\in\mathbb R^d$，词表矩阵为 $W_U\in\mathbb R^{d\times |V|}$。模型产生

$$
z=hW_U+b,
\qquad
z\in\mathbb R^{|V|}.
$$

$z_i$ 是 token $i$ 的 logit。logit 可以任意平移而不改变 softmax；它的绝对值没有单独概率意义，差值决定相对偏好。

## 3.2 Softmax

温度为 $T>0$ 时，条件分布为

$$
p_i=
\frac{\exp(z_i/T)}
{\sum_j\exp(z_j/T)}.
$$

$T<1$ 放大 logit 差，使分布更尖；$T>1$ 压缩差异，使分布更平。温度不增加知识，只改变当前候选之间的相对采样概率。

实际实现会先减去最大 logit 以避免数值溢出。这是等价的数值稳定变换：

$$
\operatorname{softmax}(z)
=
\operatorname{softmax}(z-c\mathbf 1).
$$

## 3.3 处理器会先改 Logit

在 softmax 或选择前，运行时可能应用：

- 禁止 token 或语法约束；
- repetition/frequency penalty；
- 最小长度和停止限制；
- JSON grammar 或正则约束；
- watermark 或安全过滤；
- vocabulary mask。

因此“模型原始分布”和“服务实际采样分布”可能不同。分析 logprob 时必须说明是在处理前还是处理后。

## 3.4 Greedy

Greedy 选择最大 logit：

$$
i^*=\arg\max_i z_i.
$$

并列时仍需要固定 tie-breaking。若全部输入和数值结果固定，greedy 不调用随机数。它可以稳定地产生同一个错误，也可能因早期一个次优 token 进入之后完全不同的续写。

## 3.5 Sampling

随机采样从分布 $p$ 中抽取 token。实现常把伪随机数 $u\in[0,1)$ 映射到累积概率区间。相同 seed 只有在随机算法、随机数消费顺序和候选分布相同的条件下才可能给出相同 token。

Sampling 的随机性位于选择步骤，不表示前向网络内部每层都在随机思考。固定 dropout 关闭的推理网络可以完全确定地产生概率向量，再由外部采样器随机选择。

## 3.6 Top-k 与 Top-p

Top-k 只保留概率最高的 $k$ 个 token，再重新归一化。Top-p 选择概率质量累计达到阈值 $p_0$ 的最小候选集合。

两者都改变原分布：低概率尾部被截断，保留项概率重新放大。Top-p 的候选数量会随分布尖锐程度变化；top-k 的候选数量固定，却可能在很确定时保留多余项，在很不确定时删除大量质量。

## 3.7 一个 Token 不是一个词

选出的 ID 可能对应完整汉字、词片段、空格、标点、UTF-8 字节片段或控制标记。界面未必立即显示它：解码器可能要等待后续 token 才能组成有效字符，streaming 层也可能按更大文本块发送。

## 3.8 概率不是真实性

高 $p_i$ 只表示在当前模型、上下文和处理配置下，该 token 在候选中占较大质量。它不等于整句为真的概率，也不等于模型知道某个事实。

卷三会解释训练数据、条件混合和目标函数怎样形成这些概率。本章只确定一点：在一次执行中，logit 经过处理和选择，最终产生一个 token ID。
