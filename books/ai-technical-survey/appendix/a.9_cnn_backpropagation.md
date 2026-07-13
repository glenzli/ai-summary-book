# 附录 A.9 CNN 反向传播原理 (CNN Backpropagation Principles)

在正文中我们提到，在特定索引约定下，卷积层对输入的梯度 $\nabla_X \mathcal{L}$ 可写成 **误差项 $\boldsymbol{\delta}$ 与翻转卷积核的互相关**。本附录先在简化条件下证明该式，再说明一般 Conv2D 的扩展。

## A.9.1 证明目标

证明单位步幅情形下：

$$
\nabla_X\mathcal L
=\delta *_{math} K
=\delta\star\operatorname{rot180}(K),
$$

其中 $*_{math}$ 表示数学卷积，$\star$ 表示深度学习框架常用的互相关。

## A.9.2 定义

以下 A.9.2-A.9.5 主体推导采用这些假设：单个样本、单输入通道、单输出通道、`stride=1`、`dilation=1`、无 groups，并省略 bias 与后续非线性。为简化变元替换，核用以 0 为中心的有限整数 offset 记号；0-based 数组中的 `rot180` 对应 $k_{K_h-1-m,K_w-1-n}$。输入按前向 padding 规则在边界外补 0，$\delta_{i,j}$ 也在有效输出下标外视为 0。对于 `valid` 前向卷积，最终输入梯度对应 full 形状；对于显式 padding，还需按原输入区域裁剪。没有这些边界约定，下面的求和范围与输出形状并不完整。

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

因此，在上述单通道、单位 stride/dilation 与补零约定下：
$$ \nabla_X \mathcal{L} = \delta *_{math} K \equiv \text{Conv2D}(\text{input}=\delta, \text{kernel}=\text{rot180}(K)) $$
即：**输入梯度等于误差图与翻转后卷积核的互相关**，并按前向 padding 约定取相应区域。

## A.9.6 多通道与一般 Conv2D

多输入/输出通道时，前向互相关为

$$
y_{o,i,j}=\sum_c\sum_{m,n}x_{c,i+m,j+n}\,k_{o,c,m,n},
$$

所以还要对所有输出通道求和：

$$
\frac{\partial\mathcal L}{\partial x_{c,a,b}}
=\sum_o\sum_{m,n}\delta_{o,a-m,b-n}\,k_{o,c,m,n}.
$$

对一般步幅 $s_h,s_w$、膨胀 $d_h,d_w$ 和 padding $p_h,p_w$，更稳妥的前向索引是

$$
y_{o,i,j}
=\sum_c\sum_{m,n}
x_{c,\,i s_h-p_h+m d_h,\,j s_w-p_w+n d_w}
k_{o,c,m,n}.
$$

对应输入梯度可写成

$$
\frac{\partial\mathcal L}{\partial x_{c,a,b}}
=\sum_{o,i,j,m,n}\delta_{o,i,j}k_{o,c,m,n}
\,\mathbf 1\!\left[a=i s_h-p_h+m d_h,\;
b=j s_w-p_w+n d_w\right].
$$

框架通常用 **transposed convolution** 式索引实现这一步：按 stride 在误差图位置间插入对应间隔，按 dilation 访问核元素，再处理 padding、output padding 与裁剪。此时不能把单位步幅下的同形状公式无条件套用到所有 Conv2D 配置。
