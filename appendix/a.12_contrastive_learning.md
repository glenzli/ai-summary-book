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

对固定的 $x_i$，把 $\{y_j\}_{j=1}^N$ 看作 $N$ 类分类问题：只有 $y_i$ 是正类，其余为负类。

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
- $\tau$ 越小，$s_{ij}$ 变大，Softmax 越尖锐，模型更“强硬”地区分正负样本。
- $\tau$ 越大，分布更平滑，训练信号更保守。

在实践中，$\tau$ 常作为可学习参数，让模型自动找到合适的对比强度。

## A.12.4 一个常见直觉：对角线最大化 (Maximizing the Diagonal)

把 $S=[s_{ij}]$ 看作一个 $N\times N$ 相似度矩阵：
- 对角线 $s_{ii}$：正确配对
- 非对角线 $s_{ij}$：错误配对

InfoNCE 的训练效果可以直观理解为：
- 拉高对角线元素
- 拉低非对角线元素

这就是为什么在论文/工程里，经常用“相似度矩阵热力图”来肉眼检查训练是否正常：训练良好时，热力图会在对角线附近出现明显亮带。

## A.12.5 与“互信息下界”的关系（可选） (Optional: MI Lower Bound)

InfoNCE 也常被解释为互信息 $I(X;Y)$ 的一个下界估计器：当负样本来自边缘分布、且批大小足够大时，优化 InfoNCE 会提高 $X$ 与 $Y$ 的统计依赖程度。

这条解释对理解“为什么对比学习能学到语义表征”很有帮助，但严格证明需要更长的概率论推导，这里不展开。