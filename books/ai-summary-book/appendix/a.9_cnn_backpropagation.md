# 附录 A.9 CNN 反向传播原理 (CNN Backpropagation Principles)

在正文中我们提到，卷积层对输入的梯度 $\nabla_X \mathcal{L}$ 等价于 **误差项 $\boldsymbol{\delta}$ 与翻转后的卷积核的卷积**。本附录提供严谨的数学证明。

## A.9.1 证明目标

证明：$\nabla_X \mathcal{L} = \delta * \text{rot180}(K)$

## A.9.2 定义

*   **前向互相关 (Cross-Correlation)**：这是深度学习中 Conv2D 的标准实现。
    $$ y_{i,j} = (X \star K)_{i,j} = \sum_{m} \sum_{n} x_{i+m, j+n} \cdot k_{m,n} $$
    *(注意：这里的下标是 $i+m$, $j+n$)*
    
*   **数学卷积 (Convolution)**：
    $$ (A * B)_{i,j} = \sum_{m} \sum_{n} A_{i-m, j-n} \cdot B_{m,n} $$
    *(注意：这里的下标是 $i-m$, $j-n$)*

## A.9.3 证明过程

假设我们已经从 **后端 MLP** 或后续层接收到了误差梯度矩阵 $\boldsymbol{\delta}$（其中 $\delta_{i,j} = \frac{\partial \mathcal{L}}{\partial y_{i,j}}$）。我们需要计算 Loss 对当前层输入像素 $x_{a,b}$ 的梯度。

根据链式法则：
$$ \frac{\partial \mathcal{L}}{\partial x_{a,b}} = \sum_{i,j} \frac{\partial \mathcal{L}}{\partial y_{i,j}} \frac{\partial y_{i,j}}{\partial x_{a,b}} = \sum_{i,j} \delta_{i,j} \frac{\partial y_{i,j}}{\partial x_{a,b}} $$

考察前向公式 $y_{i,j} = \sum_{m,n} x_{i+m, j+n} k_{m,n}$。
只有当 $i+m=a$ 且 $j+n=b$ 时，这一项才包含 $x_{a,b}$。
这意味着 $i = a-m$ 且 $j = b-n$。

我们将 $i, j$ 替换为 $a, b, m, n$ 的表达式，代入求和公式：
$$ \frac{\partial \mathcal{L}}{\partial x_{a,b}} = \sum_{m} \sum_{n} \delta_{a-m, b-n} \cdot k_{m,n} $$

观察这个式子：
$$ \sum_{m} \sum_{n} \delta_{a-m, b-n} \cdot k_{m,n} $$
这正是 $\delta$ 和 $K$ 的**数学卷积**公式（形式为 $A_{a-m} B_{m}$）。
即：
$$ \nabla_X \mathcal{L} = \delta *_{math} K $$

## A.9.4 与 rot180 的关系

由于深度学习框架（如 PyTorch/TensorFlow）通常只提供**互相关**算子 (conv2d)，我们需要用互相关算子来实现数学卷积。
互相关公式是 $A_{a+m} B_m$。
为了凑出 $A_{a-m} B_m$，我们需要将 $B$ 的下标符号取反。
令 $K'_{m,n} = K_{-m, -n}$（即旋转 180 度），代入互相关公式：
$$ (\delta \star K')_{a,b} = \sum_{m,n} \delta_{a+m, b+n} K'_{m,n} = \sum_{m,n} \delta_{a+m, b+n} K_{-m,-n} $$
令 $p = -m, q = -n$，则：
$$ = \sum_{p,q} \delta_{a-p, b-q} K_{p,q} $$
这恰好回到了数学卷积的形式。

## A.9.5 结论

因此，我们在正文中得到结论：
$$ \nabla_X \mathcal{L} = \delta *_{math} K \equiv \text{Conv2D}(\text{input}=\delta, \text{kernel}=\text{rot180}(K)) $$
即：**输入梯度等于误差图与翻转后卷积核的互相关。**
