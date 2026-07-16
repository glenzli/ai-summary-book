# 附录 A.12 对比学习与 InfoNCE (Contrastive Learning & InfoNCE)

## A.12.1 问题设定：从“匹配”到“区分” (From Matching to Discrimination)

在多模态（如 CLIP）或自监督表征学习中，我们常见的数据形式是“配对样本” $(x_i, y_i)$：
- $x_i$：图像、语音、文本片段等
- $y_i$：与之匹配的另一模态描述/视图

目标不是预测一个固定类别，而是让模型学会一个相似度函数，使得“正确配对”更相似，“错误配对”更不相似。

## A.12.2 InfoNCE：把相似度矩阵变成交叉熵 (InfoNCE as Cross-Entropy)

设一批样本大小为 $N$，编码器给出归一化向量（或任意可比较向量）：
- $\mathbf{u}_i = f(x_i)$
- $\mathbf{v}_j = g(y_j)$

定义相似度（以点积/余弦为例）并引入温度系数 $\tau$：
$$ s_{ij} = \frac{\mathbf{u}_i^T\mathbf{v}_j}{\tau} $$

### A.12.2.1 单向 InfoNCE（以 $x$ 预测匹配的 $y$）

对固定的 $x_i$，把 $\{y_j\}_{j=1}^N$ 看作 $N$ 类分类问题：训练配对 $y_i$ 是指定正类，其余为 batch 内候选。后者在损失中充当 negatives，但现实数据中可能存在语义等价或多重正确配对（false negatives）。

于是我们可以定义一个“行 Softmax”：
$$ P(j\mid i) = \frac{\exp(s_{ij})}{\sum_{k=1}^{N}\exp(s_{ik})} $$

最大化正类概率等价于最小化交叉熵：
$$ \mathcal{L}_{x\to y} = -\frac{1}{N}\sum_{i=1}^{N} \log P(i\mid i) = -\frac{1}{N}\sum_{i=1}^{N}\log \frac{\exp(s_{ii})}{\sum_{j=1}^{N}\exp(s_{ij})} $$

### A.12.2.2 双向 InfoNCE（CLIP 常用）

同理也可以反过来，用 $y$ 去“检索” $x$：
$$ \mathcal{L}_{y\to x} = -\frac{1}{N}\sum_{i=1}^{N}\log \frac{\exp(s_{ii})}{\sum_{j=1}^{N}\exp(s_{ji})} $$

CLIP 常用双向损失的平均：
$$ \mathcal{L}_{\text{CLIP}} = \frac{1}{2}(\mathcal{L}_{x\to y} + \mathcal{L}_{y\to x}) $$

## A.12.3 温度系数 $\tau$：控制“分布尖锐度” (Role of Temperature)

把 $\tau$ 看成 Softmax logits 的缩放因子：
- $\tau$ 越小，logits 的绝对尺度增大，Softmax 通常更尖锐，梯度尺度也会改变。
- $\tau$ 越大，分布通常更平滑；具体训练效果仍取决于特征归一化、batch negatives 与优化器。

在实践中，$\tau$ 可以固定，也可以通过受约束的 logit scale 学习；并非所有 InfoNCE 实现都学习温度。

## A.12.4 一个常见直觉：对角线最大化 (Maximizing the Diagonal)

把 $S=[s_{ij}]$ 看作一个 $N\times N$ 相似度矩阵：
- 对角线 $s_{ii}$：正确配对
- 非对角线 $s_{ij}$：错误配对

InfoNCE 的训练效果可以直观理解为提高对角配对相对于同一行/列候选的 log-softmax 概率。它优化相对差值，不要求每个非对角元素分别降到某个绝对值；false negatives 也会使“所有非对角都应拉低”的说法失效。

这就是为什么在论文/工程里，经常用“相似度矩阵热力图”来肉眼检查训练是否正常：训练良好时，热力图会在对角线附近出现明显亮带。

## A.12.5 与“互信息下界”的关系（可选） (Optional: MI Lower Bound)

在特定联合/边缘采样方案与 critic 假设下，InfoNCE 可导出形如 $I(X;Y)\ge \log N-\mathcal L_{\mathrm{NCE}}$ 的互信息下界。有限 $N$ 会限制该界，负样本相关、false negatives 和具体 batch 构造也会影响解释，因此不能把任意对比损失值直接当成互信息估计。

这条解释对理解“为什么对比学习能学到语义表征”很有帮助，但严格证明需要更长的概率论推导，这里不展开。
