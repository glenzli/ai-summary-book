# 附录 A.10 Transformer 数学原理 (Transformer Mathematical Principles)

## A.10.1 缩放点积注意力的数学推导 (Derivation of Scaled Dot-Product Attention)

Transformer 的核心公式是：
$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

### A.10.1.1 为什么需要除以 $\sqrt{d_k}$？ (Why Scale?)

为了理解缩放因子的必要性，我们考察点积的统计性质。

为得到尺度直觉，假设 $Q$ 和 $K$ 的元素 $q_i, k_i$ 独立同分布且服从标准正态分布。真实网络中的投影向量通常相关且不严格服从该分布，因此这是初始化附近的教学模型：
$$ q_i, k_i \sim \mathcal{N}(0, 1) $$

由于它们是独立的，均值为 0：
$$ \mathbb{E}[q_i] = \mathbb{E}[k_i] = 0 $$
$$ \text{Var}(q_i) = \text{Var}(k_i) = 1 $$

考察它们的点积 $x = \sum_{i=1}^{d_k} q_i k_i$：

**均值**：
$$ \mathbb{E}[x] = \mathbb{E}\left[\sum_{i=1}^{d_k} q_i k_i\right] = \sum_{i=1}^{d_k} \mathbb{E}[q_i]\mathbb{E}[k_i] = 0 $$

**方差**：
$$ \text{Var}(x) = \text{Var}\left(\sum_{i=1}^{d_k} q_i k_i\right) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) $$

由于独立性，$\text{Var}(XY) = \mathbb{E}[X^2 Y^2] - (\mathbb{E}[XY])^2$。
当 $X, Y \sim \mathcal{N}(0, 1)$ 时，$\mathbb{E}[X^2]=1, \mathbb{E}[Y^2]=1, \mathbb{E}[XY]=0$。
所以 $\text{Var}(q_i k_i) = 1 \cdot 1 - 0 = 1$。

因此，点积 $x$ 的总方差为：
$$ \text{Var}(x) = \sum_{i=1}^{d_k} 1 = d_k $$

这表明，点积结果的标准差是 $\sqrt{d_k}$。当 $d_k$ 很大（如 512）时，点积的值域范围会非常大（例如 $\pm 20$ 甚至更大）。

### A.10.1.2 Softmax 的梯度消失问题 (Gradient Vanishing in Softmax)

Softmax 函数 $\sigma(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$。
其导数（雅可比矩阵）为：
$$ \frac{\partial \sigma_i}{\partial z_j} = \sigma_i (\delta_{ij} - \sigma_j) $$

当输入 $z$ 的数值很大时（例如某一项远大于其他项），Softmax 的分布会变得非常“尖锐”（接近 One-hot 分布）。
此时，$\sigma_i$ 要么接近 1，要么接近 0。
*   如果 $\sigma_i \approx 1$，则导数 $\approx 1(1-1) = 0$。
*   如果 $\sigma_i \approx 0$，则导数 $\approx 0(0-0) = 0$。

这意味着该 Softmax 行的雅可比会变小，沿这一路径的梯度可能衰减；它不等于整个网络的所有梯度都归零，因为损失结构、其他注意力行和残差路径仍会传递信号。

在上述假设下，除以 $\sqrt{d_k}$ 将点积方差归一化为 1，使 Softmax 不易仅因维度增大而过早饱和。它改善尺度条件，但不能保证所有头或整个深层网络的梯度稳定。

### A.10.1.3 Self-Attention 矩阵运算的完整展开 (Matrix Expansion of Self-Attention)

为了让大家脑海中“跑通”整个过程，我们将 $Q, K, V$ 的矩阵乘法完全展开。这有助于理解为什么 Attention 本质上是一个**基于内容的寻址 (Content-based Addressing)** 和**加权平均 (Weighted Averaging)** 过程。

假设我们有两个单词的序列 (Sequence Length $L=2$)，特征维度为 3 ($d_k=d_v=3$)。

#### Step 1: 计算相似度 ($QK^T$)

首先，我们将 Query 矩阵和 Key 矩阵的转置相乘。
$$
Q = \begin{bmatrix} \mathbf{q}_1^T \\ \mathbf{q}_2^T \end{bmatrix}, \quad
K^T = \begin{bmatrix} \mathbf{k}_1 & \mathbf{k}_2 \end{bmatrix}
$$

$$
\text{Scores} = QK^T =
\begin{bmatrix} \mathbf{q}_1^T \\ \mathbf{q}_2^T \end{bmatrix}
\begin{bmatrix} \mathbf{k}_1 & \mathbf{k}_2 \end{bmatrix}
=
\begin{bmatrix}
\mathbf{q}_1^T \mathbf{k}_1 & \mathbf{q}_1^T \mathbf{k}_2 \\
\mathbf{q}_2^T \mathbf{k}_1 & \mathbf{q}_2^T \mathbf{k}_2
\end{bmatrix}
$$

展开看每一个元素，例如第一行第二列的元素：
$$ \text{Score}_{12} = \mathbf{q}_1 \cdot \mathbf{k}_2 = q_{11}k_{21} + q_{12}k_{22} + q_{13}k_{23} $$
这代表了 **Query 1 与 Key 2 的相似度**（未归一化）。得到的矩阵是一个 $L \times L$ 的方阵。

#### Step 2: 归一化为概率 (Softmax)

经过 Scale 和 Softmax 后，我们将 Scores 转化为概率分布矩阵 $A$（Attention Weights）：

$$
A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) =
\begin{bmatrix}
\alpha_{11} & \alpha_{12} \\
\alpha_{21} & \alpha_{22}
\end{bmatrix}
$$
其中每一行的和为 1（例如 $\alpha_{11} + \alpha_{12} = 1$）。

#### Step 3: 信息聚合 ($A \times V$)

最后一步，用计算出的权重去“加权提取” Value 矩阵中的信息。

$$
V = \begin{bmatrix} \mathbf{v}_1^T \\ \mathbf{v}_2^T \end{bmatrix} = \begin{bmatrix} v_{11} & v_{12} & v_{13} \\ v_{21} & v_{22} & v_{23} \end{bmatrix}
$$

$$
\text{Output} = A V =
\begin{bmatrix}
\alpha_{11} & \alpha_{12} \\
\alpha_{21} & \alpha_{22}
\end{bmatrix}
\begin{bmatrix} \mathbf{v}_1^T \\ \mathbf{v}_2^T \end{bmatrix}
$$

根据矩阵乘法规则，结果的每一行是 $V$ 中行的线性组合：

$$
\text{Output} =
\begin{bmatrix}
\alpha_{11}\mathbf{v}_1^T + \alpha_{12}\mathbf{v}_2^T \\
\alpha_{21}\mathbf{v}_1^T + \alpha_{22}\mathbf{v}_2^T
\end{bmatrix}
$$

让我们聚焦于**第一个 Token 的新表示**（输出的第一行 $\mathbf{z}_1$）：
$$ \mathbf{z}_1 = \alpha_{11}\mathbf{v}_1 + \alpha_{12}\mathbf{v}_2 $$

**建模解释**：
Token 1 的新向量 $\mathbf{z}_1$，本质上就是所有 Value 向量的**加权平均**。
*   如果 $\alpha_{12}$ 很大（接近 1），说明 Token 1 极其关注 Token 2。
*   结果就是：Token 2 的信息 $\mathbf{v}_2$ 会大量“流向” Token 1，使得 Token 1 的更新后表示中包含了 Token 2 的特征。
只要掩码允许，任意位置的 Value 都能在一层中直接影响当前输出，这是 Self-Attention 建模长距离依赖的重要结构条件；是否实际学到正确依赖仍取决于 Q/K/V 投影、训练数据与优化。

## A.10.2 位置编码的性质 (Properties of Positional Encoding)

原始 Transformer 使用正弦位置编码：
$$ PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}}) $$
$$ PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}}) $$

### A.10.2.1 相对位置偏移 (Relative Position Shift)

我们希望证明：对于任意偏移 $k$，位置 $pos+k$ 的编码可以由位置 $pos$ 的编码通过线性变换得到。
即存在矩阵 $M_k$，使得 $PE(pos+k) = M_k \cdot PE(pos)$。

考察一对 $(2i, 2i+1)$ 维度的频率 $\omega_i = \frac{1}{10000^{2i/d_{model}}}$。
位置编码向量在该维度上可以看作复数 $e^{j \omega_i \cdot pos}$ 的实部和虚部。

$$ PE(pos) \sim \begin{pmatrix} \sin(\omega_i pos) \\ \cos(\omega_i pos) \end{pmatrix} $$

对于位置 $pos+k$：
$$
\begin{aligned}
\sin(\omega_i (pos+k)) &= \sin(\omega_i pos)\cos(\omega_i k) + \cos(\omega_i pos)\sin(\omega_i k) \\
\cos(\omega_i (pos+k)) &= \cos(\omega_i pos)\cos(\omega_i k) - \sin(\omega_i pos)\sin(\omega_i k)
\end{aligned}
$$

这可以写成矩阵乘法形式：

$$
\begin{pmatrix} \sin(\omega_i (pos+k)) \\ \cos(\omega_i (pos+k)) \end{pmatrix}
=
\begin{pmatrix}
\cos(\omega_i k) & \sin(\omega_i k) \\
-\sin(\omega_i k) & \cos(\omega_i k)
\end{pmatrix}
\begin{pmatrix} \sin(\omega_i pos) \\ \cos(\omega_i pos) \end{pmatrix}
$$

这是一个**旋转矩阵 (Rotation Matrix)**。
这意味着固定偏移在每个频率对子空间里对应一个确定的线性旋转，为注意力层学习相对位置关系提供结构。模型是否实际学会该变换、能否外推到训练长度之外，仍取决于数据、参数和训练范围。

## A.10.3 Self-Attention 的复杂度：为什么是 $O(L^2)$？ (Complexity)

设序列长度为 $L$，隐藏维度为 $d$，单头 Key/Query 维度为 $d_k$。

1.  **相似度矩阵**：$S = QK^T \in \mathbb{R}^{L\times L}$。
    - 计算代价约为 $O(L^2 d_k)$（本质是矩阵乘法）。
    - 朴素实现会物化 $O(L^2)$ 的分数/概率中间量；FlashAttention 等分块精确算法可避免把完整矩阵写回 HBM，因此额外工作显存不必是 $O(L^2)$。

2.  **加权求和**：$Z = \text{softmax}(S)V$。
    - 代价约为 $O(L^2 d_v)$。

因此，标准稠密 Self-Attention 的算术量仍随 $L^2$ 增长；朴素实现的中间显存也为二次方，而 IO-aware 精确实现可显著降低中间存储与访存。长上下文研究因而同时探索稀疏/滑窗注意力、FlashAttention、分块并行和缓存压缩，不能把它们都归为降低同一种复杂度。

## A.10.4 因果掩码：禁止偷看未来 (Causal Mask)

在 Decoder-only 或 Transformer Decoder 的自回归训练中，我们需要保证位置 $i$ 只能关注 $j\le i$。

令掩码矩阵 $M\in\mathbb{R}^{L\times L}$：
$$ M_{ij} = \begin{cases} 0 & i \ge j \\ -\infty & i < j \end{cases} $$

把它加到 Softmax 之前的 logits 上：
$$ \text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V $$

由于 $e^{-\infty}=0$，Softmax 会把所有 $i<j$ 的概率质量压成 0，从而在数学上严格地实现“看不见未来”。
