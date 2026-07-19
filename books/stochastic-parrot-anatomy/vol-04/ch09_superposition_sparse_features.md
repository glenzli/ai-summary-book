# 第九章 Superposition、SAE 与稀疏表示

模型可能需要表示的可用特征远多于 residual 维度。若这些特征很少同时激活，网络可以把它们以非正交方向叠加在同一空间中；这被称为 superposition。稀疏字典方法试图从叠加 activation 中恢复更接近计算特征的坐标。

## 9.1 一个直观模型

设潜在特征 $f\in\mathbb R^m$ 稀疏，模型表示为

$$
x=Wf,
\qquad x\in\mathbb R^d,
\quad m>d.
$$

因为 $m>d$，列向量不可能全部正交。若每个样本只有少量 $f_i$ 非零，仍可能近似恢复许多特征；同时激活过多时会发生 interference。

这个 toy model 解释了 polysemantic neurons 的一种来源，但真实 Transformer 含非线性、attention 和训练动力学，不能由线性 toy model 直接推出全部内部结构。

## 9.2 Sparse Autoencoder

对模型 activation $x$，SAE 学习过完备 latent：

$$
f=\phi(W_{enc}(x-b_{dec})+b_{enc}),
$$

$$
\hat x=W_{dec}f+b_{dec}.
$$

典型目标为

$$
\mathcal L
=\mathbb E\|x-\hat x\|_2^2
+\lambda\mathbb E\|f\|_1.
$$

重构项要求保留原 activation，稀疏项限制每个样本活跃 latent 数。实现还可能使用 TopK、JumpReLU 或其他稀疏机制。

## 9.3 尺度退化

对允许 latent 正比例重缩放的参数化，例如使用正齐次激活的情形，把 latent 缩小为 $cf$、同时把 decoder columns 放大为原来的 $1/c$，可以保持重构项不变，却把 $L_1$ 代价缩小为原来的 $c$。若没有额外约束，$c\to0$ 会产生尺度退化。实践中通常规范化 decoder columns，或使用尺度不变的稀疏代价。

因此报告 SAE 目标时必须包括 normalization、activation preprocessing、latent expansion 和稀疏度定义；只给 $\lambda$ 无法比较不同实现。

## 9.4 Feature 的输入与输出两面

latent $f_i$ 的 encoder direction 决定什么 activation 触发它，decoder vector $d_i$ 决定它如何重构 residual state。二者不必相同。

解释 feature 时应同时检查：

- 高 activation 自然样本；
- 由说明生成的合成正反例；
- decoder 对 logits 或下游 features 的作用；
- 插入、删除 feature 的行为效应；
- 跨层、位置和语料的稳定性。

## 9.5 Reconstruction 不等于 Mechanism Preservation

小均方重构误差不保证保留模型行为。误差若恰好落在高 logit 敏感方向，可能显著改变输出；较大误差也可能位于下游忽略子空间。

应额外报告：

- 把 $x$ 替换为 $\hat x$ 后的 loss increase；
- next-token argmax 一致率；
- 目标行为保留率；
- reconstruction error 的直接与间接 logit effect；
- 干预在原模型与 SAE 替代表示中的一致性。

## 9.6 Dead、Split 与 Absorbed Features

SAE 常出现：

- dead latent：几乎从不激活；
- feature splitting：一个语义模式被多个 latent 按上下文细分；
- feature absorption：本应独立的模式被另一个 feature 吸收；
- duplicate features：多个 decoder 方向高度相似；
- dense residual：重要结构留在重构误差中。

更大字典不单调带来更“真实”的 feature。粒度变化可能把一个宽概念拆成许多窄模式，也可能只复制方向。

## 9.7 Monosemanticity 的测量

“单义”不是直接可观察二元标签。可能指标包括：

- 自动/人工说明在 held-out activation 上的预测分数；
- 高 activation 样本的语义纯度；
- 合成反例的选择性；
- feature intervention 的行为一致性；
- 同一说明跨语料的稳定性。

这些指标依人类概念体系。一个计算上纯净但难用自然语言命名的 feature，不等于没有机制意义。

## 9.8 Transcoder 与 Cross-layer Feature

SAE 重构某个 residual activation；transcoder 则用稀疏 features 预测模块输出，试图让 latent 更接近计算单元。cross-layer transcoder 允许早层 feature 直接重构多个后续层 MLP 输出，使跨层 attribution graph 更稀疏。

代价是引入替代模型。feature 是分析器学习出的单位，不是原网络显式参数；机制忠实性必须通过 reconstruction 和 intervention 响应验证。

## 9.9 Steering

设置或放大 latent $f_i$，等价于沿 decoder direction 写入 residual：

$$
x'=x+\alpha d_i.
$$

feature steering 可改变主题、风格、拒答或实体行为，但常伴随：

- 过度重复和语言质量下降；
- 相关概念共同变化；
- 不同层或位置效果不一致；
- 小范围有效、大强度离开分布；
- feature label 与实际副作用不匹配。

steerability 证明方向具有控制作用，不证明正常生成只通过该 feature。

## 9.10 从解释稀疏表示到训练稀疏模型

另一条路线不是事后用 SAE 分解 dense model，而是在原模型训练中施加结构稀疏，使更少 residual channels、MLP 单元或连接参与每个计算。这样可能获得更容易隔离的 circuits，却可能牺牲能力或训练效率。

评价应画出能力、计算成本与可解释性之间的 Pareto frontier，而不能只展示最干净案例。稀疏架构仍需要验证 feature 标签和回路忠实性。

## 9.11 非唯一字典

稀疏编码目标未必有唯一最优字典。初始化、超参数、数据和等价旋转会产生不同 feature 集。若两个 SAE 都重构良好却给不同解释，研究者应比较其干预预测和跨 seed 稳定性。

寻找“唯一真实 feature 基”可能不是合理目标；更实际的标准是该基是否压缩计算、预测新激活并支持忠实干预。

## 9.12 结论

SAE 和 transcoder 为 polysemantic activation 提供过完备稀疏坐标，是当前机制可解释性的重要路线。它们的价值不由 feature 名称数量决定，而由重构、行为保真、干预一致性和跨分布稳定性共同决定。
