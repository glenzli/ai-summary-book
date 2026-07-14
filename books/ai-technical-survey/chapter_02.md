# 第二章 深度学习基础：训练、CNN 与序列模型

多层网络能表示异或，并不等于它会从数据中自动找到那组权重。随机初始化之后，误差必须穿过每一层传回参数；训练集上的改进还必须转化为未见样本上的表现。深度学习真正困难的部分，常常不在“这个函数是否存在”，而在“有限数据和有限计算下，优化过程会找到什么函数”。

图像和序列把这个问题具体化。卷积利用局部连接和参数共享，把同一种视觉模式放到不同位置复用；循环网络让同一状态更新跨时间复用，LSTM 再用门控制信息保留。三类结构都不是任意堆层，而是在数据中加入不同的归纳偏置。反向传播、正则化与优化器则负责让这些偏置真正变成可训练系统。
<a id="section-2-1"></a>

## 2.1 深度学习的理论基石：从泛化到反向传播
### 2.1 Theoretical Foundations: From Generalization to Backpropagation

第一章比较了人工特征、经典预测器与端到端表示学习。本章进一步讨论深度网络如何训练和泛化，并以 CNN、RNN 与 LSTM 为主要架构例子；经典统计学习方法仍在结构化数据和资源受限场景中广泛使用。

在进入这些架构前，需要先建立三个相互关联的基础：经验风险与泛化、反向模式自动微分，以及基于梯度的优化。

深度学习训练综合使用 **反向传播** (Backpropagation)、**统计学习与泛化分析** (Generalization) 以及 **数值优化** (Optimization)。本节给出教材所需的最小数学口径，并明确哪些结论只在特定损失或模型假设下成立。

这些工具跨架构复用，但具体泛化界、优化性质与数值行为仍取决于模型、数据和训练算法。

#### 2.1.1 机器学习的铁律：泛化与过拟合 (Generalization & Overfitting)

我们训练模型的终极目标从来不是在 **训练集** 上拿满分，而是在未见过的 **测试集** 上表现良好。这种举一反三的能力被称为 **泛化 (Generalization)**。

##### 1. 经验风险与期望风险
从数学上讲，我们试图最小化所有可能数据的**期望风险 (Expected Risk)** $R(f)$，但上帝视角的真实分布 $P(x,y)$ 是不可知的。因此，我们只能退而求其次，最小化在训练集上的**经验风险 (Empirical Risk)** $\hat{R}(f)$：

$$ \hat{R}(f) = \frac{1}{N} \sum_{i=1}^N L(y_i, f(x_i)) $$

**过拟合 (Overfitting)** 指训练风险较低而未见分布上的风险明显更高。拟合有限样本的偶然模式或噪声是一种常见原因，但分布偏移、数据泄漏和评测选择也要另行区分。

##### 2. 偏差-方差分解 (Bias-Variance Decomposition)
在平方损失回归的经典设定下，预测误差可以分解为三部分（详细推导见 [附录 A.3](appendix/a.3_statistical_learning_theory.md)）：

$$ \text{Error} = \text{Bias}^2 + \text{Variance} + \text{Noise} $$

*   **偏差 (Bias)**：不同训练集上所得预测的平均值相对真实回归函数的系统误差；模型限制过强时常出现高偏差。
*   **方差 (Variance)**：训练集变化引起预测变化的程度；高方差可能造成过拟合，但不能仅由参数量判断。
*   **不可约误差 (Noise)**：数据本身的固有噪声，它给可达到的期望误差设定下限。

下图展示经典的概念性权衡：在一些模型族中，复杂度增加会降低偏差并提高方差。现代过参数化网络可能出现双下降等不同曲线，因此这不是关于所有模型的单调定律。

```mermaid
graph LR
    subgraph Underfitting ["欠拟合 (High Bias)"]
        direction TB
        C1["模型过于简单"] --> R1["捕捉不到数据特征"]
    end

    subgraph SweetSpot ["最佳平衡点 (Sweet Spot)"]
        direction TB
        C2["复杂度适中"] --> R2["泛化能力最强"]
    end

    subgraph Overfitting ["过拟合 (High Variance)"]
        direction TB
        C3["模型过于复杂"] --> R3["死记硬背噪音"]
    end

    Underfitting --> SweetSpot --> Overfitting

    linkStyle default stroke:#333,stroke-width:2px;
    style Underfitting fill:#F5F5F5,stroke:#666666,stroke-width:2px,color:#000000
    style SweetSpot fill:#D5E8D4,stroke:#82B366,stroke-width:2px,color:#000000
    style Overfitting fill:#F8CECC,stroke:#B85450,stroke-width:2px,color:#000000
```

<img src="chapter_02/images/bias_variance_tradeoff.png" width="80%" />

##### 3. 复杂度的度量：VC 维 (VC Dimension)
在上图中，横轴代表“Model Complexity”。但如何从数学上定量描述一个模型的复杂度？统计学习理论引入了 **VC 维 (Vapnik-Chervonenkis Dimension)** 的概念。

*   **直观定义**：VC 维衡量了模型能够“打散” (Shatter) 多少个样本。
    *   例如，二维平面上的仿射直线分类器可以打散某组一般位置的 3 个点，但不存在可被它打散的 4 点集合，所以其 VC 维为 3；正方形四角上的 XOR 标记直观展示了其中一种不可线性分情形。
*   **泛化界 (Generalization Bound)**：
    对二分类 0-1 损失，在独立同分布、$N\ge d_{VC}$ 等经典条件下，一类常见的高概率界可概括为

$$
R(f) \le \hat R(f) +
O\!\left(\sqrt{\frac{d_{VC}\log(2eN/d_{VC})+\log(1/\delta)}{N}}\right).
$$

    具体常数和对数形式因定理版本而异，样本不足时该界也可能是空泛的。较高 VC 维会增大容量项，但样本量、置信度和经验误差共同决定界；它不是“高 VC 维必然过拟合”的判据。

这为“奥卡姆剃刀”原则提供了统计学习意义上的解释：**在训练误差相近且候选模型都能解释数据时，更低复杂度的模型通常具有更紧的泛化界**。但现代深度学习还受到优化隐式偏置、数据规模和模型结构的影响，不能把 VC 维较低简单等同于一定更好。（关于 VC 维的详细定义与泛化误差界公式，详见 **[附录 A.3](appendix/a.3_statistical_learning_theory.md)**）

#### 2.1.2 应对过拟合：正则化 (Regularization)

深度神经网络通常参数很多，具备拟合训练数据乃至噪声的能力，但参数量并不能单独预测其泛化误差。显式正则化、数据增强、优化算法的隐式偏置、模型结构和数据规模会共同决定最终解。这里先介绍**正则化**：在目标或训练过程中加入约束与先验偏好。（关于其贝叶斯解释与详细推导，请见 **[附录 A.4](appendix/a.4_regularization.md)**）

##### 1. L1 与 L2 正则化 (Mathematical Definition)

最经典的正则化方法是通过修改**损失函数**，在其中加入对参数规模的惩罚项（Penalty Term）。
$$ J(\mathbf{w}) = \text{Loss}(\mathbf{w}) + \lambda \cdot \Omega(\mathbf{w}) $$
其中 $\lambda$ 是控制约束强度的超参数。

*   **L2 正则化 (Ridge)**：
    $$ \Omega(\mathbf{w}) = \frac{1}{2} \|\mathbf{w}\|^2_2 = \frac{1}{2} \sum w_i^2 $$
    它倾向于收缩参数的整体 $\ell_2$ 范数，但不保证参数均匀，也不单独保证泛化改善。
*   **L1 正则化 (Lasso)**：
    $$ \Omega(\mathbf{w}) = \|\mathbf{w}\|_1 = \sum |w_i| $$
    它倾向于让参数**变得稀疏**（许多 $w_i$ 直接变成 0）。

**几何直观 (Geometric Intuition)**：
如下图所示，我们将二次 Loss 的等高线与约束区域画在同一个平面上。当约束处于活跃状态且满足常规光滑条件时，最先接触可行域的等高线给出边界最优点。
*   **L2 (圆形)**：切点通常在圆周上，$w$ 变小但不为 0。
*   **L1 (菱形)**：非光滑边界与坐标轴尖角使最优解更容易出现精确的 $w=0$，从而产生稀疏性；是否稀疏仍取决于数据与正则强度。

<img src="chapter_02/images/regularization_geometry.png" width="90%" />

##### 2. 工程视角：权重衰减 (Weight Decay)

在实际的深度学习框架（如 PyTorch）中，我们通常不直接修改 Loss 函数公式，而是设置优化器的 **`weight_decay`** 参数。为什么？

*   **L2 正则化与权重衰减**
    如果我们对 L2 正则化的 Loss 求导并代入 SGD 更新公式，会发现：
    $$ w_{t+1} = w_t - \eta (\nabla Loss + \lambda w_t) = \underbrace{(1 - \eta\lambda)}_{\text{Decay}}w_t - \eta \nabla Loss $$
    这意味着：在普通 SGD 中，**L2 正则化等价于在每次更新时将权重“衰减”一小部分**。对 Adam 这类自适应优化器，这个等价关系一般不再严格成立，因此现代大模型训练常使用 AdamW 形式的解耦权重衰减。这也解释了术语 **Weight Decay** 的由来。

*   **谁更常用？**
    *   **L2 / Weight Decay** 是现代深度学习中最常见的基础正则化之一，尤其在 ResNet、Transformer、BERT 等架构中常与优化器、数据增强、归一化和早停等策略配合使用。
    *   **L1** 常用于希望得到稀疏参数或特征选择的场景，也可与 L2 组合成 Elastic Net 等正则。

##### 3. Dropout：随机失活
**Dropout** 是神经网络中常用的随机正则化技巧：训练时对激活施加伯努利掩码，推理时使用相应的期望缩放约定。

**数学定义 (Mathematical Formulation)**：
设 $\mathbf{h}$ 为隐藏层输出，$p$ 为**丢弃概率 (Dropout Rate)**。我们引入一个服从伯努利分布的随机掩码向量 $\mathbf{r}$：
$$ \mathbf{r} \sim \text{Bernoulli}(1-p) $$

*   **训练时**：应用掩码 $\tilde{\mathbf{h}} = \mathbf{r} \odot \mathbf{h}$。
*   **测试时**：为了平衡期望值 ($E_{train}[\tilde{\mathbf{h}}] = (1-p)\mathbf{h}$)，我们需要对权重或输出进行缩放，这就是 **Weight Scaling**。
    $$ \mathbf{h}_{test} = (1-p) \mathbf{h} $$
    *(注：现代框架常采用 **Inverted Dropout**，即在训练时直接除以 $1-p$，从而免去测试时的缩放)*

（关于掩码向量的生成、数值计算示例及反向传播细节，请见 **[附录 A.4.4](appendix/a.4_regularization.md#a44-dropout-的数学机制详解-mathematics-of-dropout)**）

**集成学习视角 (Ensemble View)**：
*   **训练时 (Ensemble)**：每次 Iteration，我们都在训练一个新的、更稀疏的子网络。由于网络参数是共享的，这相当于我们在同时训练指数级个 ($2^N$) 不同的神经网络。
*   **测试时 (Averaging)**：所有神经元开启。上述的 Weight Scaling 或 Inverted Dropout 近似于对大量共享参数的稀疏子网络做模型平均。对特定损失和近似假设下可以得到几何平均式的解释，但在一般深度网络中，更稳妥的说法是：Dropout 提供了一种高效的随机集成近似。

从这个角度看，Dropout 并非简单的“扔掉”信息，而是一种极其高效的**集成学习 (Ensemble Learning)**。

<div align="center">
  <img src="chapter_02/images/dropout_ensemble.png" width="80%" />
</div>

#### 2.1.3 通用近似定理 (Universal Approximation Theorem)

在 1.3.4 节中，我们从直观层面介绍了**通用近似定理**。XOR 的可表示性可以直接构造，通用近似定理讨论的则是更一般的函数类：在明确的激活函数、定义域和逼近范数条件下，网络族为何能逼近紧致域上的连续函数。本章转入这一性质的数学表述与构造思路。

前文讨论了如何约束训练过程以改善泛化；这里转向另一个独立问题：给定足够的网络宽度，模型族能够表示哪些函数。表达能力强不等于一定能优化到、从有限数据学到或在未见样本上泛化。

**定理陈述**：
> 在相应定理条件下（例如连续且非多项式的激活函数），具有足够多隐藏单元的**单隐层前馈网络**在紧致集合上可以一致逼近连续函数。Cybenko (1989) 的原始结果使用满足其判别条件的 sigmoid 激活；ReLU 的结论来自后续推广。

**数学表达（统一记号）**：
令 $K\subset\mathbb R^n$ 为紧致集，$\sigma$ 满足所采用定理的条件：Cybenko (1989) 使用连续 sigmoidal/discriminatory 激活；ReLU 则由后续非多项式激活结论覆盖。对任意 $f\in C(K)$ 和 $\epsilon>0$，存在整数 $N$ 及参数 $v_i,\mathbf w_i,b_i$，使得
$$ F(\mathbf x) = \sum_{i=1}^{N} v_i \sigma(\mathbf w_i^T\mathbf x + b_i), $$
$$ \sup_{\mathbf x\in K}|F(\mathbf x) - f(\mathbf x)| < \epsilon. $$

这表明，单隐层神经网络在连续函数空间中是**稠密 (Dense)** 的。

**直观证明 (Bump Function 构造法)**：

在一维连续函数的情形，可以通过构造局部“积木”来理解逼近过程。
在一维中，若干平移、缩放的 ReLU 线性组合可以构造分段线性折线或三角形“帽函数”；把足够多局部基函数组合起来，可逼近连续曲线。这个直观只说明构造思路，正式定理还需要激活条件、紧致定义域与一致逼近论证。

<img src="chapter_01/images/universal_approximation.png" width="80%" />

这意味着：在连续函数、紧致定义域和足够宽的网络等条件下，神经网络具有很强的函数逼近能力。它说明的是表达能力的存在性，不保证有限数据下的可学习性、可优化性或泛化能力。（详细数学证明请见 **[附录 A.5](appendix/a.5_universal_approximation.md)**）

#### 2.1.4 训练引擎：反向传播 (Backpropagation)

有了模型和目标，通常用**梯度法**迭代寻找低损失参数；非凸神经网络训练不保证得到全局最优解。高效计算这些梯度的核心方法，是 Rumelhart、Hinton 与 Williams 在 1986 年系统推广的**反向传播 (Backpropagation)**。

##### 1. 核心思想：计算图与链式法则
反向传播的本质是**链式法则 (Chain Rule)** 在**计算图 (Computational Graph)** 上的高效应用。
想象整个神经网络是一个巨大的管道系统（计算图）。
*   **前向传播 (Forward Pass)**：数据像水流一样，从输入层流向输出层，经过层层加权和激活，最终计算出 Loss。
*   **反向传播 (Backward Pass)**：**误差信号**（梯度）像水流回溯一样，从 Loss 出发，沿着原本的路径逆流而上，计算每个阀门（权重）对总误差的“贡献度”。

我们可以用下图直观对比这两股“水流”：

```mermaid
graph LR
    %% 样式定义 (遵循全书统一配色)
    classDef input fill:#F5F5F5,stroke:#666666,stroke-width:2px,color:#000000;
    classDef hidden fill:#DAE8FC,stroke:#6C8EBF,stroke-width:2px,color:#000000;
    classDef output fill:#E1D5E7,stroke:#9673A6,stroke-width:2px,color:#000000;
    classDef loss fill:#F8CECC,stroke:#B85450,stroke-width:2px,color:#000000;

    subgraph Process ["深度学习的每一次迭代 (One Iteration)"]
        direction LR
        x((Input)):::input
        h[Hidden Layers]:::hidden
        y((Output)):::output
        L{Loss}:::loss

        %% 前向传播：实线
        x ==>|"1. Forward (Data)"| h
        h ==>|"1. Forward (Activation)"| y
        y ==>|"1. Forward (Prediction)"| L

        %% 反向传播：虚线
        L -.->|"2. Backward (Gradient)"| y
        y -.->|"2. Backward (Chain Rule)"| h
        h -.->|"2. Backward (Update Weights)"| x
    end

    %% 连线样式
    linkStyle 0,1,2 stroke:#4CAF50,stroke-width:3px;
    linkStyle 3,4,5 stroke:#FF5722,stroke-width:3px,stroke-dasharray: 5 5;

    %% 子图样式
    style Process fill:#FFFFFF,stroke:#666666,stroke-dasharray: 5 5,color:#000000
```

##### 2. 矩阵版反向传播推导 (The Four Fundamental Equations)
为了高效计算，现代深度学习框架（如 PyTorch）都采用**矩阵运算**。我们定义第 $l$ 层的加权输入为 $\mathbf{z}^{(l)}$，激活输出为 $\mathbf{a}^{(l)}$。
$$ \mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}, \quad \mathbf{a}^{(l)} = \sigma(\mathbf{z}^{(l)}) $$

我们需要求 Loss 对权重 $\mathbf{W}^{(l)}$ 和偏置 $\mathbf{b}^{(l)}$ 的梯度。这就引入了一个中间量——**误差项 (Error Term) $\boldsymbol{\delta}^{(l)}$**，它表示“第 $l$ 层神经元的加权输入 $\mathbf{z}^{(l)}$ 对最终 Loss 的敏感程度”：
$$ \boldsymbol{\delta}^{(l)} \equiv \frac{\partial L}{\partial \mathbf{z}^{(l)}} $$

有了这个定义，反向传播可以概括为四个简洁优美的公式（BP 四大公式）：

**(1) 输出层误差 (Output Layer Error)**
计算最后一层 $L$ 的误差。这取决于损失函数的导数 $\nabla_{\mathbf{a}}L$ 和激活函数的导数 $\sigma'$。
$$ \boldsymbol{\delta}^{(L)} = \nabla_{\mathbf{a}}L \odot \sigma'(\mathbf{z}^{(L)}) $$
*   $\odot$ 代表 Hadamard 积（逐元素相乘）。这告诉我们：如果激活函数进入饱和区（$\sigma' \approx 0$），梯度就会在这里消失。

**(2) 误差反向传播 (Error Backpropagation)**
如何从第 $l+1$ 层的误差推导出第 $l$ 层的误差？误差通过权重矩阵的**转置** $(\mathbf{W}^{(l+1)})^T$ 往回传。
$$ \boldsymbol{\delta}^{(l)} = \left[ (\mathbf{W}^{(l+1)})^T \boldsymbol{\delta}^{(l+1)} \right] \odot \sigma'(\mathbf{z}^{(l)}) $$
*   **直观理解**：第 $l+1$ 层的误差 $\boldsymbol{\delta}^{(l+1)}$ 被权重矩阵 $W$ “加权分配”回了第 $l$ 层。

**(3) 权重的梯度 (Gradient w.r.t Weights)**
有了本层的误差 $\boldsymbol{\delta}^{(l)}$，权重的梯度就是“误差”乘以“上一层的输入”。
$$ \frac{\partial L}{\partial \mathbf{W}^{(l)}} = \boldsymbol{\delta}^{(l)} (\mathbf{a}^{(l-1)})^T $$
*   这解释了为什么输入 $\mathbf{a}^{(l-1)}$ 也不能过大或过小，因为它直接充当了梯度的系数。

**(4) 偏置的梯度 (Gradient w.r.t Bias)**
$$ \frac{\partial L}{\partial \mathbf{b}^{(l)}} = \boldsymbol{\delta}^{(l)} $$

（关于这四个公式的详细数学推导、矩阵维度分析及计算图的局部梯度视角，请见 **[附录 A.6 反向传播的数学推导](appendix/a.6_backpropagation.md)**）

##### 3. 完整的动态流程图
下图展示了一个包含两层网络的完整前向与反向过程。前向传播计算 Loss，反向传播计算梯度并更新参数。

```mermaid
graph TD
    %% 全局样式
    classDef input fill:#F5F5F5,stroke:#666666,stroke-width:2px,color:#333;
    classDef weights fill:#FFF2CC,stroke:#D6B656,stroke-width:2px,color:#333;
    classDef neuron fill:#DAE8FC,stroke:#6C8EBF,stroke-width:2px,color:#333;
    classDef loss fill:#F8CECC,stroke:#B85450,stroke-width:2px,color:#333;
    classDef grad fill:#FFFFFF,stroke:#C62828,stroke-width:2px,stroke-dasharray: 5 5,color:#C62828;

    %% ---------------------------------------------------------
    %% Forward Pass (Top Down)
    %% ---------------------------------------------------------
    subgraph ForwardPass ["1. 前向传播 (Forward Pass)"]
        direction TB
        X(("Input x")):::input

        %% Layer 1
        W1["W<sub>1</sub>, b<sub>1</sub>"]:::weights
        Z1("Linear z<sub>1</sub>"):::neuron
        A1("ReLU a<sub>1</sub>"):::neuron

        %% Layer 2
        W2["W<sub>2</sub>, b<sub>2</sub>"]:::weights
        Z2("Linear z<sub>2</sub>"):::neuron
        A2("Softmax a<sub>2</sub>"):::neuron

        %% Loss
        L{"Loss"}:::loss
        Target(("Target y")):::input

        X --> W1 --> Z1 --> A1
        A1 --> W2 --> Z2 --> A2
        A2 --> L
        Target --> L
    end

    %% ---------------------------------------------------------
    %% Backward Pass (Bottom Up / Side)
    %% ---------------------------------------------------------
    subgraph BackwardPass ["2. 反向传播 (Backward Pass)"]
        direction TB

        %% Gradients - Complex Math uses LaTeX
        dL_dA2{{"$$\frac{\partial L}{\partial a_2}$$"}}:::grad
        dL_dW2{{"$$\frac{\partial L}{\partial W_2}$$"}}:::grad
        dL_dA1{{"$$\frac{\partial L}{\partial a_1}$$"}}:::grad
        dL_dW1{{"$$\frac{\partial L}{\partial W_1}$$"}}:::grad

        %% Updates - Simple Subscripts use HTML
        UpdW2["Update W<sub>2</sub>"]:::weights
        UpdW1["Update W<sub>1</sub>"]:::weights
    end

    %% Linking Forward and Backward
    L -.->|"Compute Grad"| dL_dA2
    dL_dA2 -.->|"Chain Rule"| dL_dW2
    dL_dW2 -.->|"Update"| UpdW2

    dL_dA2 -.->|"Backprop Error"| dL_dA1
    dL_dA1 -.->|"Chain Rule"| dL_dW1
    dL_dW1 -.->|"Update"| UpdW1

    %% Visual connection to original weights
    UpdW2 -.- W2
    UpdW1 -.- W1

    %% ---------------------------------------------------------
    %% Layout Tweaks
    %% ---------------------------------------------------------
    style ForwardPass fill:#FFFFFF,stroke:#333,stroke-width:1px
    style BackwardPass fill:#FFFFFF,stroke:#333,stroke-width:1px

    %% Color backward links to match gradient nodes (Dark Red)
    linkStyle 8,9,10,11,12,13,14,15 stroke:#C62828,stroke-width:2px;
```

> **核心概念说明：Softmax 函数**
>
> 在上图中，我们看到输出层使用了 `Softmax`。这是多分类任务（如识别 1000 种物体）的标准配置。
> 它将神经网络输出的任意实数（Logits）转换为**概率分布**：
> 1.  所有概率非负 ($p_i \ge 0$)。
> 2.  所有概率之和为 1 ($\sum p_i = 1$)。
>
> **公式**：
> $$ \text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}} $$
>
> *（关于 Softmax 及其与交叉熵损失函数的常见配合与求导推导，请详见 **[附录 A.7 Softmax 与 Cross-Entropy](appendix/a.7_softmax_crossentropy.md)**）*

#### 2.1.5 优化器：从 SGD 到 Adam

计算出随机小批量梯度估计 $g_t$ 后，如何更新参数 $w$？朴素的 **SGD (Stochastic Gradient Descent)** 沿该估计的负梯度方向更新；它是全数据最陡下降方向的随机近似：
$$ w_{t+1} = w_t - \eta g_t $$
但在面对复杂的非凸优化曲面（如狭长的峡谷或鞍点）时，SGD 往往表现不佳：它会在峡谷壁之间剧烈震荡（Zig-Zag），收敛极慢。现代优化器引入了**动量 (Momentum)** 和**自适应 (Adaptivity)** 两个核心概念来解决这一问题。（关于优化器的详细数学推导与进阶分析，请见 **[附录 A.8](appendix/a.8_advanced_optimization.md)**）

##### 1. Momentum (动量法)
**直观理解**：模拟物理中的小球滚下山坡。小球不会时刻顺着当前的坡度切线方向（梯度）走，而是拥有**惯性**。如果梯度方向改变（震荡），惯性会平滑路径；如果梯度方向一致（下坡），小球会不断加速。

**数学定义**：
我们引入速度变量 $v$，它是梯度的**指数移动平均 (Exponential Moving Average, EMA)**：
$$ v_t = \beta v_{t-1} + (1-\beta) g_t $$
$$ w_{t+1} = w_t - \alpha v_t $$
*   $\beta$ (通常约 0.9) 类似于“摩擦系数”，决定了惯性的大小。
*   **效果**：在峡谷中，垂直于谷底的震荡分量互相抵消，而平行于谷底的加速分量互相叠加，从而加速收敛。

<img src="chapter_02/images/sgd_vs_momentum.png" width="60%" />

##### 2. Adaptive Learning Rate (自适应学习率)
**直观理解**：不同的参数扮演不同的角色。对于经常更新的参数（如常见词的 Embedding），我们希望步长小一点以微调；对于很少更新的稀疏参数（如生僻词），一旦出现，我们希望步长大一点以抓住机会。
**RMSProp** 通过衡量梯度的“大小”来动态调整每个参数的学习率。

**数学定义**：
我们计算梯度平方的移动平均 $s_t$（即二阶矩估计）：
$$ s_t = \beta s_{t-1} + (1-\beta) g_t^2 $$
$$ w_{t+1} = w_t - \frac{\eta}{\sqrt{s_t + \epsilon}} g_t $$
*   当梯度很大（陡峭）时，$s_t$ 变大，分母变大，有效学习率减小（防止震荡）。
*   当梯度很小（平坦）时，$s_t$ 变小，分母变小，有效学习率增大（加速通过平原）。

<img src="appendix/images/rmsprop_vs_sgd.png" width="60%" />

##### 3. Adam (Adaptive Moment Estimation)
Adam 及其 AdamW 变体是深度学习中最常用的优化器家族之一。

**为什么需要 Adam？**
*   **动量法的局限**：虽然解决了震荡，但在极度拉伸的峡谷中，单一的全局学习率可能导致在平缓方向上移动极其缓慢。
*   **自适应法的局限**：上面展示的基础 RMSProp 公式只有二阶缩放、没有 Adam 那样显式的一阶矩；RMSProp 也存在带动量的实现，不能笼统说该算法必然“缺乏惯性”。

**Adam 的集大成之道**：
Adam 优雅地结合了二者。它同时维护**一阶矩 $m_t$**（动量，提供惯性）和**二阶矩 $v_t$**（自适应缩放，调整步长）。

**数学定义**：
1.  **一阶矩与二阶矩**：
    $$ m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \quad \text{(Momentum)} $$
    $$ v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \quad \text{(RMSProp)} $$
2.  **参数更新**：
    $$ w_{t+1} = w_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} $$
    *(注：$\hat{m}, \hat{v}$ 为偏差修正后的值。典型参数：$\beta_1=0.9, \beta_2=0.999$)*

**优化器轨迹总览 (The Grand Comparison)**：
我们将所有优化器的轨迹绘制在同一张图上，可以清晰地看到进化的脉络：

*   **SGD (红色)**：受困于峡谷的陡峭壁面，剧烈震荡，寸步难行。
*   **Momentum (蓝色)**：引入惯性，冲破了震荡，但步长单一。
*   **RMSProp (紫色)**：自适应调整步长，能够快速穿越平缓区，但路径有抖动。
*   **Adam (绿色)**：结合了一阶动量与二阶自适应缩放，在许多非凸问题上更快、更稳；但它并不保证总是优于 SGD，最终效果仍取决于学习率、正则化、数据规模和目标函数形状。

<img src="chapter_02/images/optimizer_comparison_all.png" width="60%" />

#### 2.1.6 权重初始化：打破对称与保持方差

在构建好深度神经网络的架构后，我们需要给每个参数赋一个初始值。不恰当的初始化会显著增加梯度消失、梯度爆炸或激活饱和的风险，因而可能使训练变慢甚至失败。

##### 1. 直观理解：传声筒游戏 (Telephone Game)
想象深度网络是一个有 100 层接力者的传声筒游戏。
*   **输入信号**（数据）就像初始的悄悄话。
*   **层与层之间**的传递，就是将信号乘以权重 $W$。
*   **如果 $W$ 太小 (比如 0.01)**：声音会越来越小，传到第 10 层时已经听不见了（**梯度消失**）。
*   **如果 $W$ 太大 (比如 1.5)**：声音会越来越大，传到第 10 层时变成了震耳欲聋的噪音（**梯度爆炸**）。
*   **全零或同值初始化**：在常见全连接层中，同一层的隐藏单元若以完全相同的参数开始，就会得到相同的激活和梯度，并在确定性更新下继续同步，无法学习彼此不同的特征；多层网络把所有权重置零时，还可能在初始步骤阻断前层梯度。具体输出是否为 0 取决于偏置、激活函数和架构，但核心问题都是**对称性没有被打破**。

##### 2. 数学原理：方差守恒 (Variance Preservation)
为了让信号在深层网络中较稳定地传播，初始化通常试图使相邻层激活或梯度的尺度保持在同一数量级；这是一种依赖分布假设的近似设计目标，不是对训练全过程的严格保证。
设 $y = w_1 x_1 + \dots + w_n x_n$，假设各项所需的独立性（或至少不相关性）成立，且 $w_i$ 与 $x_i$ 独立并具有零均值，则：
$$ \text{Var}(y) = \text{Var}\left(\sum w_i x_i\right) = \sum \text{Var}(w_i x_i) = \sum \text{Var}(w_i)\text{Var}(x_i) $$
$$ \text{Var}(y) = n \cdot \text{Var}(w) \cdot \text{Var}(x) $$
若把 $\text{Var}(y) \approx \text{Var}(x)$ 作为这一理想化模型下的目标，就得到：
$$ n \cdot \text{Var}(w) = 1 \implies \text{Var}(w) = \frac{1}{n} $$

##### 3. 初始化的演进
下图展示了一个 10 层网络在不同初始化策略下的激活值分布演变：

<img src="chapter_02/images/weight_initialization.png" width="100%" />

*   **Small Random (0.01)**：激活值迅速向 0 坍缩，信号丢失。
*   **Large Random (1.0)**：激活值迅速向 -1 和 1 两端饱和（对于 Tanh），导致梯度为 0。
*   **Xavier Initialization (Glorot Init)**:
    *   **适用**：Sigmoid / Tanh 激活函数。
    *   **公式**：$W \sim U(-\sqrt{6/(n_{in}+n_{out})}, \sqrt{6/(n_{in}+n_{out})})$
    *   **效果**：在相应假设和示例设置下，可减缓激活尺度随深度迅速收缩或膨胀；它不保证激活保持正态，也不保证任意架构中的训练稳定。
*   **He Initialization (Kaiming Init)**:
    *   **适用**：ReLU 及其变种。
    *   **原理**：对零均值、对称的理想化输入，ReLU 会把约一半样本置零，并使输出二阶矩约为输入的一半。为补偿这一尺度损失，初始权重方差取 $2/n_{in}$。
    *   **公式**：若正态分布第二个参数表示方差，则 $W \sim \mathcal{N}(0, 2/n_{in})$；等价地，其标准差为 $\sqrt{2/n_{in}}$。它是 ReLU 网络常用初始化原则之一。
    *   **效果**：在图示设置下，经过多层 ReLU 后的信号尺度较稳定；残差连接、归一化、实际数据分布和有限宽度都会改变这一近似分析。

------------------------------------------------------------------------------------------------

本节建立了后续架构共用的训练口径：泛化与过拟合、正则化、通用近似的存在性、反向传播、优化器和初始化。接下来分别讨论两类常见结构化输入：
1.  **空间数据（图像）**：我们将看到 CNN 如何通过“卷积”捕捉局部特征，解决计算机视觉问题。
2.  **时间数据（序列）**：我们将看到 RNN 及其进化体 LSTM 如何通过“记忆”捕捉时序依赖，解决自然语言处理问题。

<a id="section-2-2"></a>

## 2.2 卷积神经网络：视觉皮层的数学抽象
### 2.2 Convolutional Neural Networks (CNN)

图像不是一组没有位置关系的数。相邻像素常共同形成边缘与纹理，同一种局部形状也可能出现在画面任何位置。早期视觉神经科学对局部感受野的研究提供了重要启发，但卷积神经网络 (CNN) 更准确的技术出发点，是把这种局部性与位置复用写进参数共享方式。

一个卷积核先在小窗口内计算响应，再把同一组权重滑过整张图；浅层响应可组合成更大尺度的形状。猫耳出现在左上角或右下角时，同一个检测器都能产生响应。这个机制同时减少参数并保留空间结构，随后再由下采样、非线性与更深层组合扩大感受野。

#### 2.2.1 归纳偏置：为什么是卷积？ (The Inductive Bias)

若把 $1000\times1000$ 图像直接连到 1000 个隐层单元，单层就有约 $10^9$ 个权重，而且拉平操作没有显式保留二维邻域。CNN 让每个输出只连接一个局部感受野，并让同一卷积核在所有空间位置共享。前者假设近邻结构优先，后者使输入平移时特征图相应平移，也就是标准卷积的平移等变性。

对标准卷积层，这两点使其参数量不随输入图像的空间尺寸增长，而由卷积核大小和输入/输出通道数决定，从而显著降低参数开销。完整 CNN 若含 Flatten 后的全连接层，整体参数量仍可能依赖输入尺寸。

#### 2.2.2 卷积的算术原理 (The Arithmetic of Convolution)

虽然名为“卷积”，但在深度学习中，我们实际使用的是**互相关 (Cross-Correlation)** 运算（不涉及卷积核的翻转）。

##### 1. 离散卷积公式 (Matrix View)
对于单通道图像 $I$ 和卷积核 $K$，在局部窗口内的运算本质上就是两个矩阵的**点积**（Frobenius Inner Product）。
假设局部输入窗口为 $\mathbf{X}_{local}$，卷积核为 $\mathbf{K}$：
$$
\mathbf{X}_{local} = \begin{bmatrix} x_{11} & x_{12} \\ x_{21} & x_{22} \end{bmatrix}, \quad
\mathbf{K} = \begin{bmatrix} k_{11} & k_{12} \\ k_{21} & k_{22} \end{bmatrix}
$$
则输出值 $y$ 为对应元素相乘求和：
$$
y = \langle \mathbf{X}_{local}, \mathbf{K} \rangle = \sum \sum x_{ij} k_{ij} = x_{11}k_{11} + x_{12}k_{12} + x_{21}k_{21} + x_{22}k_{22}
$$
*   **直观意义**：这是输入局部区域与卷积核的未归一化相关响应。若二者的符号与结构匹配，响应的绝对值可能较大；由于它还受二者范数和偏置影响，不能直接当作归一化相似度。

##### 2. 核心超参数与几何维度
一个卷积层由输入/输出通道数 $C_{in},C_{out}$、卷积核大小 $K$、填充 $P$ 与步长 $S$ 共同确定。$K$ 决定单层局部窗口，$P$ 决定边缘如何参与，$S$ 决定窗口每次移动多远；这些量同时决定输出几何尺寸和计算量。

下图直观展示了卷积、填充和步长的几何关系：

<img src="chapter_02/images/cnn_spatial.png" width="100%" />

##### 3. 输出尺寸公式 (The Magic Formula)
给定输入尺寸 $H_{in} \times W_{in}$，输出特征图的尺寸可以通过以下公式计算：

$$ H_{out} = \left\lfloor \frac{H_{in} + 2P - K}{S} \right\rfloor + 1 $$
$$ W_{out} = \left\lfloor \frac{W_{in} + 2P - K}{S} \right\rfloor + 1 $$

*   **Same Padding**：若希望输出尺寸不变 ($H_{out} = H_{in}$)，且 $S=1$，需设置 $P = \frac{K-1}{2}$（要求 $K$ 为奇数）。
*   **Valid Padding**：$P=0$，不填充，尺寸会逐层减小。
*   **下采样 (Downsampling)**：当 $S=2$ 时，特征图尺寸减半 ($H_{out} \approx H_{in}/2$)，常用于替代池化层。

##### 4. 通道视角的卷积 (Channel View: The 3D Nature)
初学者常误以为卷积是 2D 运算，实际上它是 **3D 运算**。
假设输入是 RGB 图像 ($C_{in}=3$)，卷积核的形状实际上是 $K \times K \times C_{in}$（如 $3 \times 3 \times 3$）。

$$ \text{Output}(x,y) = \sum_{c=1}^{C_{in}} \sum_{i,j} \text{Input}(x+i, y+j, c) \cdot \text{Kernel}(i, j, c) + \text{Bias} $$

一个卷积核跨越全部 $C_{in}$ 个输入通道并在通道上求和，产生一个二维特征图；使用 $C_{out}$ 组独立卷积核便得到 $C_{out}$ 个输出通道。因此参数量为 $\text{Params}=(K\times K\times C_{in}+1)\times C_{out}$，与输入图像的 $H\times W$ 空间尺寸无关。RGB 边缘检测器由此可以同时使用红、绿、蓝三个通道，而不是分别生成三个互不相干的结果。

#### 2.2.3 池化层：信息的压缩与筛选 (Pooling Layers)

卷积层负责提取特征，而池化层负责**压缩特征**。它通常跟在卷积层之后。

##### 1. 为什么需要池化？
*   **降低计算量**：当池化采用大于 1 的步长进行下采样时，会减小特征图尺寸并降低后续层的计算开销。
*   **局部平移稳定性（非严格不变性）**：最大池化或平均池化可能降低网络对局部小位移的敏感度，但有限窗口和步长会引入边界效应；输入平移一个像素时，池化输出一般不保证不变。更强的平移不变性通常来自卷积、下采样、数据增强与全局聚合的共同作用。
*   **扩大感受野**：通过下采样，后续卷积核能“看到”更大的原图区域。

##### 2. 常见操作
*   **最大池化 (Max Pooling)**：选出局部窗口内的最大值。
    *   *直觉*：只保留最显著的特征（如最亮的边缘），丢弃背景噪声。
*   **平均池化 (Average Pooling)**：计算局部窗口的平均值。
    *   *直觉*：平滑特征，保留背景信息。

$$
\text{Input: } \begin{bmatrix} 1 & 3 \\ 2 & 9 \end{bmatrix} \quad \xrightarrow{\text{Max}} 9, \quad \xrightarrow{\text{Avg}} 3.75
$$

*   **全局平均池化 (Global Average Pooling, GAP)**：将整个 $H \times W$ 的特征图取平均，得到一个数值。这在现代网络（如 ResNet）中常用于替代全连接层。

#### 2.2.4 训练与学习动力学 (Training and Learning Dynamics)

很多初学者容易误解一点：以为卷积核（Kernel）是像 Photoshop 滤镜一样，由工程师预先定义好的（如 Sobel 算子）。

**事实并非如此。** 在 CNN 中，卷积核 $\mathbf{K}$ 就是神经网络的**权重 (Weights)**。它们初始化时是随机噪声，通过**端到端 (End-to-End)** 的训练，网络自动“学会”了应该长成什么样来提取有用的特征。

##### 1. 架构全景图：从像素到决策 (The Architecture Panorama)

从宏观视角看，标准的 CNN 分类器在结构上正是由 **前端的“卷积特征提取器”** 和 **后端的“传统 MLP 分类器”** 串联而成的。

这一架构把 **前端感知 (Perception)** 与 **后端决策 (Decision)** 组织成一个端到端系统，并通过 **特征提取 (Feature Extraction)**、**空间压缩 (Spatial Compression)** 与 **逻辑决策 (Logical Decision)** 这三个功能阶段层层递进：

我们以经典的 ResNet-50 为例，追踪张量 (Tensor) 在这三个阶段中的形态演变：

```mermaid
graph LR
    %% 样式定义
    classDef tensor fill:#F5F5F5,stroke:#666666,color:#000000,shape:rect;
    classDef op fill:#DAE8FC,stroke:#6C8EBF,color:#000000,shape:rounded;
    classDef vector fill:#FFF2CC,stroke:#D6B656,color:#000000,shape:rect;

    subgraph FrontEnd ["Part 1: 卷积前端 (The Convolutional Eye)"]
        style FrontEnd fill:#FFFFFF,stroke:#6C8EBF,stroke-width:2px,stroke-dasharray: 5 5
        direction TB

        subgraph Stage1 ["Stage 1: 特征提取 (Body)"]
            direction TB
            Input["Raw Image<br/>(H, W, 3)"]:::tensor --> Conv1["Conv Layers"]:::op
            Conv1 --> Feats["Feature Maps<br/>(H/32, W/32, 2048)"]:::tensor
        end

        subgraph Stage2 ["Stage 2: 空间压缩 (Neck)"]
            direction TB
            Feats --> GAP["Global Avg Pool"]:::op
            GAP --> Vector["Feature Vector<br/>(1, 2048)"]:::vector
        end
    end

    subgraph BackEnd ["Part 2: MLP 后端 (The MLP Brain)"]
        style BackEnd fill:#FFFFFF,stroke:#D6B656,stroke-width:2px,stroke-dasharray: 5 5
        direction TB

        subgraph Stage3 ["Stage 3: 逻辑决策 (Head)"]
            direction TB
            Vector --> FC["Linear Classifier"]:::op
            FC --> Probs["Class Probabilities<br/>(1, 1000)"]:::vector
        end
    end

    style Stage1 fill:#F9F9F9,stroke:#D6D6D6,stroke-dasharray: 0
    style Stage2 fill:#F9F9F9,stroke:#D6D6D6,stroke-dasharray: 0
    style Stage3 fill:#F9F9F9,stroke:#D6D6D6,stroke-dasharray: 0
```

**张量演变视图 (Tensor Evolution View)**：

$$
\underbrace{\begin{bmatrix} H \\ W \\ 3 \end{bmatrix}}_{\text{Image}}
\xrightarrow{\text{Conv}}
\underbrace{\begin{bmatrix} \downarrow \\ \downarrow \\ \uparrow \end{bmatrix}}_{\text{Spatial } \downarrow, \text{ Channel } \uparrow}
\xrightarrow{\text{GAP}}
\underbrace{\begin{bmatrix} 1 \\ 1 \\ C \end{bmatrix}}_{\text{Vector}}
\xrightarrow{\text{FC}}
\underbrace{\begin{bmatrix} N_{class} \end{bmatrix}}_{\text{Logits}}
$$

我们将整个流水线的功能模块（三段式）与其宏观结构（两段式）对应起来：

**Part 1: 卷积前端 (The Convolutional Front-End)**
负责“看懂”图像，承担了前两个功能阶段：
*   **Stage 1: 特征提取 (Feature Extraction)**
    *   **动作**：堆叠卷积层 (Conv) 和下采样 (Pooling/Stride)。
    *   **目的**：**"空间换语义"**。
    *   **形态变化**：空间维度 ($H, W$) 不断缩小，通道维度 ($C$) 不断增加。
*   **Stage 2: 空间压缩 (Spatial Compression / Neck)**
    *   **动作**：全局平均池化 (Global Average Pooling, GAP) 或 Flatten。
    *   **目的**：**"三维转一维"**。
    *   **直观理解**：GAP 将每个通道的特征图（如猫耳特征）坍缩为一个标量，最终形成一个语义向量。这标志着从“图像处理”到“向量处理”的质变。

**Part 2: MLP 后端 (The MLP Back-End)**
负责基于特征进行“决策”，承担了最后一个功能阶段：
*   **Stage 3: 逻辑决策 (Logical Decision / Head)**
    *   **动作**：全连接层 (Linear Layer) + Softmax。
    *   **目的**：**"特征映射到类别"**。
    *   **直观理解**：这本质上是一个简单的逻辑回归或 MLP。它基于前端提供的“体检报告”（特征向量），计算分类概率。

##### 2. 训练循环 (The Training Loop)
这是一个**监督学习 (Supervised Learning)** 过程。

1.  **初始化**：卷积核 $\mathbf{K}$ 初始化为高斯噪声。此时网络输出也是噪声。
2.  **前向传播**：图片输入，经过层层随机卷积，输出预测结果。
3.  **计算损失**：对比预测与真实标签，计算 Loss。
4.  **反向传播**：计算 Loss 对卷积核的梯度。
5.  **参数更新**：使用梯度下降更新卷积核。

##### 3. 数学推导：卷积层的反向传播 (Mathematical Derivation)

为了避开繁琐的求和符号下标 ($\sum_{i,j}$)，最直观的方法是利用 **im2col (image-to-column)** 技术，将卷积运算转换为我们熟悉的**矩阵乘法**。这也是 Caffe、PyTorch 等框架底层的加速实现方式。

**(1) 前向传播的矩阵化 (im2col)**

假设我们把输入 $X$ 中的每一个 $k \times k$ 局部感受野窗口拉平成一个 **列向量**，并将它们按顺序排列成一个大矩阵 $X_{col}$。

**直观示例 (Matrix Expansion Example)**：
假设输入 $X$ 为 $3 \times 3$，卷积核 $K$ 为 $2 \times 2$，步长 $S=1$（无填充）。

$$ X = \begin{bmatrix} x_{11} & x_{12} & x_{13} \\ x_{21} & x_{22} & x_{23} \\ x_{31} & x_{32} & x_{33} \end{bmatrix} $$

我们按行滑动提取 4 个局部窗口，并将其拉平为列向量：
1.  **左上窗口** $\to [x_{11}, x_{12}, x_{21}, x_{22}]^T$
2.  **右上窗口** $\to [x_{12}, x_{13}, x_{22}, x_{23}]^T$
3.  **左下窗口** $\to [x_{21}, x_{22}, x_{31}, x_{32}]^T$
4.  **右下窗口** $\to [x_{22}, x_{23}, x_{32}, x_{33}]^T$

拼接成 $X_{col}$ 矩阵（形状 $[k^2, N] = [4, 4]$）：

$$
X_{col} =
\begin{bmatrix}
x_{11} & x_{12} & x_{21} & x_{22} \\
x_{12} & x_{13} & x_{22} & x_{23} \\
x_{21} & x_{22} & x_{31} & x_{32} \\
x_{22} & x_{23} & x_{32} & x_{33}
\end{bmatrix}
$$
*(注：矩阵的每一列代表一个滑动窗口，每一行对应卷积核的一个权重位置)*

此时，卷积计算 $Y = X * K$ 就变成了简单的矩阵乘法：
$$ Y_{vec} = W_{row} \cdot X_{col} $$

*   $W_{row}$: 卷积核拉平后的行向量，形状 $[1, k^2]$。
*   $X_{col}$: 展开后的输入矩阵，形状 $[k^2, N]$，其中 $N$ 是输出像素的总数。
*   $Y_{vec}$: 输出特征图拉平后的向量，形状 $[1, N]$。

**(2) 反向传播 (Backward)**
现在，问题变成了对标准矩阵乘法 $Y = WX$ 求导。假设我们已知从 **后端 MLP** 或后续层回传来的误差梯度 $\delta_{vec}$，我们可以直接复用全连接层的结论（完整推导见 [附录 A.6](appendix/a.6_backpropagation.md)）：

*   **对权重的梯度 $\nabla_W \mathcal{L}$**：
    已知全连接层规则 $\frac{\partial L}{\partial W} = \delta \cdot X^T$，直接代入得：
    $$ \nabla_{W_{row}} \mathcal{L} = \delta_{vec} \cdot X_{col}^T $$
    <span style="background-color: #DAE8FC; color: black; padding: 2px 4px; border-radius: 4px;">直观含义</span> $\delta_{vec}$ 是误差， $X_{col}^T$ 包含了所有局部输入块。这个矩阵乘法本质上就是将**误差**与**对应的局部输入**相乘并求和。这正是**互相关**的定义。

*   **对输入的梯度 $\nabla_X \mathcal{L}$**：
    已知全连接层规则 $\frac{\partial L}{\partial X} = W^T \cdot \delta$，直接代入得：
    $$ \nabla_{X_{col}} \mathcal{L} = W_{row}^T \cdot \delta_{vec} $$
    这里求出的是“展开后矩阵 $X_{col}$”的梯度。为了得到原始图像 $X$ 的梯度，我们需要执行 **col2im** 操作（im2col 的逆过程）。
    <span style="background-color: #DAE8FC; color: black; padding: 2px 4px; border-radius: 4px;">关键点</span> 由于输入图像中的一个像素 $x_{i,j}$ 会出现在多个滑动窗口中（即在 $X_{col}$ 中出现了多次），col2im 操作会将这些位置的梯度**累加**起来。
    *   **权重转置** ($W^T$)：意味着梯度的反向传播使用了权重的转置。
    *   **梯度累加** (col2im)：意味着误差被分摊回所有感受野。

    在这里采用“前向算子实现为互相关”的深度学习约定时，输入梯度可写成 **$\delta$ 与旋转 $180^\circ$ 的卷积核做适当 padding 的互相关**；同一运算也可按数学卷积的记号重写。两种名称的差异来自核是否预先翻转（详见 **[附录 A.9](appendix/a.9_cnn_backpropagation.md)**）。

**结论总结**：
1.  $\nabla_K \mathcal{L} = X * \delta$ （输入与误差的互相关）
2.  $\nabla_X \mathcal{L} = \delta * \text{rot180}(K)$ （误差与翻转核的互相关）

通过矩阵视角，我们不需要纠结下标索引，就能清晰地看到：卷积层的反向传播，本质上就是全连接层反向传播在**权值共享**约束下的特殊形式。

##### 4. 学习动力学：从边缘到语义 (Learning Dynamics)

理解了数学机制，我们来看看通过这种机制，CNN 到底学到了什么。
我们可以把 CNN 想象成一个**搭积木**的过程。每一层都在利用上一层提供的简单积木，搭建出更复杂的积木。

*   **浅层 (Layer 1-2)：视觉原语**
    *   网络的最底层（靠近输入）通常学习到了类似于 **Gabor 滤波器** 的特征。
    *   **功能**：检测各个方向的**边缘**（横线、竖线）、**颜色斑点**、**纹理**。
    *   **直观类比**：这就像我们在画画前，先准备好各种线条和颜料。

*   **中层 (Layer 3-4)：部件组合**
    *   网络的中层开始将边缘和纹理组合成有意义的**局部部件**。
    *   **功能**：检测眼睛、耳朵、车轮、窗户等。
    *   **直观类比**：用线条画出了“圆形”、“三角形”等几何形状。

*   **深层 (Layer 5+)：语义实体**
    *   网络的深层（靠近输出）将部件组合成完整的**物体概念**。
    *   **功能**：检测人脸、猫、汽车等完整对象，甚至包括场景（卧室、海滩）。
    *   **直观类比**：将几何形状组合成了一幅完整的画作。

**特征可视化 (Feature Visualization)**：
下图展示了在 ImageNet 上训练好的 CNN，不同层级神经元被激活得最强烈的输入模式：

<img src="chapter_02/images/cnn_feature_hierarchy.png" width="100%" />
*(图注：左侧浅层关注纹理，右侧深层关注物体部件。)*

#### 2.2.5 感受野：管中窥豹 (Receptive Field)

**感受野 (Receptive Field, RF)** 是指输出特征图上的一个像素点，在原始输入图像上“看到”的区域大小。
*   **意义**：RF 决定了神经元能利用多大范围的上下文信息。如果 RF 小于目标物体（如只能看到大象的一条腿），网络就无法识别出整体。

##### 1. 感受野的逐层累积
感受野并非一成不变，而是随着网络加深而扩大。
*   **Layer 1**：$3 \times 3$ 卷积的 RF 是 $3 \times 3$。
*   **Layer 2**：在 Layer 1 的基础上再叠一个 $3 \times 3$ 卷积。Layer 2 的一个点看 Layer 1 的 $3 \times 3$，而 Layer 1 的每个点又看输入的 $3 \times 3$。最终 Layer 2 在输入上的 RF 是 $5 \times 5$。

**通项公式 (Recursive Formula)**：
令 $R_l$ 为第 $l$ 层特征图对应的输入感受野大小，$S_l, K_l$ 分别为第 $l$ 层的步长和核大小。
$$ R_l = R_{l-1} + (K_l - 1) \times \prod_{i=1}^{l-1} S_i $$
*(注：$R_0 = 1$)*

*   **直观解读**：
    *   **线性增长**：如果所有 stride $S=1$，感受野随层数线性增加 ($R_L \approx L \times K$)。
    *   **指数增长**：如果存在 stride $S=2$（下采样），累积步长 $\prod S_i$ 会迅速变大，感受野将呈指数级爆发。这就是为什么深层 CNN (如 ResNet) 能够捕捉全局语义。

##### 2. 有效感受野 (Effective Receptive Field)
理论 RF 往往非常大（甚至超过图片尺寸），但研究（Luo et al., 2016）发现，像素对输出的实际贡献并非均匀分布，而是服从**高斯分布**。
*   **中心强化**：RF 中心的像素影响最大。
*   **边缘衰减**：RF 边缘的像素影响微乎其微。
这意味着，虽然理论上神经元“看”到了全图，但它真正聚焦的还是中心区域。这解释了为什么我们需要 Attention 机制来打破这种局部聚焦的限制。

#### 2.2.6 经典架构演进：从 AlexNet 到 ResNet

几个代表性架构展示了卷积网络如何在表示能力、优化难度与计算成本之间取舍。这里比较 AlexNet、VGG、Inception 与 ResNet 所引入的具体结构变化，而不把网络深度本身当作进步尺度。

##### 1. AlexNet (2012)：大规模卷积训练
*   **深度**：8层 (5个卷积层 + 3个全连接层)。
*   **贡献**：
    *   **实证突破**：在大规模图像数据集 (ImageNet) 上清楚显示，多层卷积网络在端到端特征学习方面可以显著超过依赖手工特征 (SIFT/HOG) 的传统流程。
    *   **工程突破**：系统性使用 **ReLU** (缓解梯度饱和)、**Dropout** (降低过拟合风险) 和 **GPU 并行训练**。
*   **局限**：卷积核很大 ($11 \times 11, 5 \times 5$)，参数量巨大（大部分集中在最后的全连接层）。

##### 2. VGGNet (2014)：小卷积核的规则堆叠
*   **深度**：19层。
*   **核心思想：模块化 (Modularity)**。
    *   它抛弃了 AlexNet 中杂乱的大卷积核，全网统一使用 **$3 \times 3$** 的微小卷积核。
    *   **为什么是 $3 \times 3$？**
        *   堆叠两个 $3 \times 3$ 卷积层，感受野等于一个 $5 \times 5$。
        *   但参数量更少 ($2 \times 3^2 = 18 < 5^2 = 25$)。
        *   且多了层非线性激活，表达能力更强。
*   **局限**：全连接层依然导致参数量惊人 (138M)，且层数达到 20 层左右时，训练开始变得极其困难（梯度消失）。

##### 3. GoogLeNet (Inception) (2014)：多尺度融合
*   **深度**：22层。
*   **核心思想：宽度代替深度**。
    *   **Inception Block**：在同一层并行使用 $1 \times 1, 3 \times 3, 5 \times 5$ 卷积核，让网络自己决定该用大视野还是小视野。
        *   **$1 \times 1$ 卷积**：它先混合并压缩通道数，再施加代价更高的空间卷积，因此可显著减少计算量；这一位置通常称为瓶颈层 (Bottleneck)。

##### 4. ResNet (2015)：残差参数化
*   **深度**：152层（甚至可达 1000 层）。
*   **背景：退化问题 (Degradation Problem)**
    研究者在继续加深普通网络时发现一个反直觉现象：**更深的网络在训练集上的误差反而更高了**。这通常不是过拟合（过拟合是训练好、测试差），而是网络优化变难，即所谓退化问题。
*   **核心创新：残差学习 (Residual Learning)**
    ResNet 引入了 **Skip Connection (跨层连接)**，将输入 $x$ 直接加到输出上，即 $H(x) = F(x) + x$。
    *   **为什么叫“残差”？**
        假设这一层的目标是拟合最优映射 $H(x)$。
        *   **传统网络**：让非线性层 $F(x)$ 直接去拟合 $H(x)$。
        *   **ResNet**：让 $F(x)$ 去拟合 $H(x) - x$。这里的 $H(x) - x$（目标值减去输入值）在数学上就叫做**残差 (Residual)**。
    *   **直观理解（微调策略）**：
        想象你要把一张模糊图片($x$)修复为清晰图片($H(x)$)。
        *   **不带残差**：你需要从白纸开始，完整地画出清晰图片。
        *   **带残差**：你直接保留模糊图片，只需要画出“丢失的细节”（即残差 $F(x)$），然后把它叠加上去。在许多层中，学习“差值”比学习“全量”更容易优化。
    *   **恒等映射 (Identity Mapping)**：
        如果某一层其实什么都不需要做（特征已经足够好），网络只需将 $F(x)$ 的权重推向 0，就能较容易地实现 $H(x) = x$。而在传统网络中，要稳定拟合 $F(x) = x$ 这种恒等变换会更困难。
    *   **恒等梯度路径**：在反向传播时，Skip Connection 提供额外的恒等项，缓解深层网络的退化问题和部分梯度衰减，但并不消除所有优化困难。

```mermaid
graph LR
    %% 样式定义
    classDef data fill:#DAE8FC,stroke:#6C8EBF,color:#000000;
    classDef op fill:#F5F5F5,stroke:#666666,color:#000000;
    classDef act fill:#D5E8D4,stroke:#82B366,color:#000000;
    classDef sum fill:#FFF2CC,stroke:#D6B656,color:#000000;

    subgraph ResidualBlock ["Residual Block"]
        direction LR
        x[Input x]:::data --> w1[Weight Layer]:::op
        w1 --> relu1[ReLU]:::act
        relu1 --> w2[Weight Layer]:::op
        w2 --> add((+)):::sum
        x -->|Identity Mapping| add
        add --> relu2[ReLU]:::act
        relu2 --> y[Output Hx]:::data
    end

    style add stroke-width:2px,stroke:#D6B656
```

#### 2.2.7 现代设计原语 (Modern Primitives)

在 ResNet 之后，CNN 的设计转向了更高效的算子，旨在移动端部署和极大规模扩展。我们从数学角度来看这些优化是如何实现的。

##### 1. 深度可分离卷积 (Depthwise Separable Convolution) - MobileNet
标准卷积同时处理**空间信息**（$H \times W$）和**通道信息**（$C_{in} \to C_{out}$）。MobileNet 将其拆解为两步：
1.  **Depthwise Conv (DW)**：每个输入通道只由一个卷积核处理（不跨通道）。
2.  **Pointwise Conv (PW)**：用 $1 \times 1$ 卷积在通道间混合信息。

**计算量对比**：
假设输入 $H \times W \times C_{in}$，输出 $C_{out}$，核大小 $K \times K$。
*   **标准卷积**：$H \cdot W \cdot C_{in} \cdot C_{out} \cdot K^2$
*   **DW + PW**：$\underbrace{H \cdot W \cdot C_{in} \cdot K^2}_{\text{Depthwise}} + \underbrace{H \cdot W \cdot C_{in} \cdot C_{out} \cdot 1^2}_{\text{Pointwise}}$
*   **压缩比**：
    $$ \frac{\text{DW+PW}}{\text{Standard}} = \frac{K^2 C_{in} + C_{in} C_{out}}{K^2 C_{in} C_{out}} = \frac{1}{C_{out}} + \frac{1}{K^2} $$
    若 $K=3$，计算量通常能减少到原来的 **1/8** 到 **1/9**，是移动端模型的基石。

##### 2. 分组卷积 (Group Convolution) - ResNeXt
将输入通道 $C_{in}$ 分成 $g$ 组，每组独立进行卷积，最后再拼接。
*   **数学意义**：标准卷积的权重矩阵是全满的，而分组卷积强制权重矩阵为**块对角矩阵 (Block-diagonal Matrix)**。
*   **稀疏连接**：这是一种结构化的稀疏性。它减少了参数量（约减少 $g$ 倍），也可能通过限制通道间任意混合而带来一定的正则化效果。它与 Transformer 中 **Multi-head Attention** 都体现了“把表示空间分成若干子空间分别处理，再进行融合”的思想，但二者不是严格的一一对应物。

##### 3. 挤压与激励 (Squeeze-and-Excitation, SE) - SENet
这是一类早期的**通道注意力 (Channel Attention)** 机制，用于学习输入相关的通道重标定系数。
过程分为三步：
1.  **Squeeze (全局池化)**：将空间信息 $H \times W$ 压缩为一个实数，得到通道描述符 $z \in \mathbb{R}^C$。
    $$ z_c = \frac{1}{H \times W} \sum_{i=1}^H \sum_{j=1}^W u_c(i,j) $$
2.  **Excitation (自适应权重)**：通过两个全连接层（降维再升维）学习通道间的非线性关系，生成权重向量 $s$。
    $$ s = \sigma(W_2 \cdot \text{ReLU}(W_1 z)) $$
3.  **Scale (加权)**：将权重 $s_c$ 乘回原特征图的对应通道。
    $$ \tilde{u}_c = s_c \cdot u_c $$
**效果**：几乎不增加计算量，却能显著提升模型对关键特征的敏感度。

---

CNN 长期是计算机视觉主干。Vision Transformer 展示了较弱视觉先验在大规模训练下的竞争力，而 CNN 的局部性与平移等变性在数据效率、延迟和部分低层视觉任务中仍然有用。ConvNeXt 等工作也说明，卷积与 Transformer 的训练配方和设计原语可以相互借鉴；不存在对所有任务都“不可替代”的单一架构。
<a id="section-2-3"></a>

## 2.3 循环神经网络：序列动力学
### 2.3 Recurrent Neural Networks (RNN)

卷积在空间位置上复用局部检测器，**循环神经网络 (RNN)** 则在时间位置上复用状态更新。读到句子中的 `bank` 时，当前解释不仅取决于这个 token，也取决于此前是否出现 `river` 或 `account`；语音与其他时间序列同样需要把过去信息带到当前步骤。

#### 2.3.1 序列建模与参数共享 (Sequence Modeling & Parameter Sharing)

训练样本之间是否独立，与一个样本内部的序列依赖是两件事。对序列输入 $x^{(1)},x^{(2)},\dots,x^{(T)}$，模型必须显式处理同一样本内的位置关系；普通 MLP 不会自动把前一位置的计算状态传给后一位置。

RNN 在所有时间步 (Time Steps) 上复用同一组状态转移参数。参数量因此不随序列长度增长，同一个递推规则也可作用于不同长度的序列；可处理任意长度不等于能无损保存任意久远的信息。

##### 1. 计算图展开 (Unrolling the Graph)
把 RNN 沿时间展开，就会看到同一个单元反复使用：第 $t$ 步输入 $\mathbf x_t$ 与上一状态 $\mathbf h_{t-1}$ 共同产生新状态 $\mathbf h_t$，再由它得到输出 $\mathbf y_t$。连接相邻状态的 $\mathbf W_{hh}$ 在所有时间步共享，因此参数量不随序列长度增长；序列变长时，计算步数和状态所承担的信息压力仍会增长。

```mermaid
graph LR
    %% 样式定义
    classDef input fill:#F5F5F5,stroke:#666666,color:#000000;
    classDef hidden fill:#DAE8FC,stroke:#6C8EBF,color:#000000;
    classDef output fill:#FFF2CC,stroke:#D6B656,color:#000000;
    classDef weight fill:#E1D5E7,stroke:#9673A6,color:#000000;

    subgraph Unrolled ["时间展开视图 (Unrolled View)"]
        direction LR
        x1["x<sub>1</sub>"]:::input --> h1(("h<sub>1</sub>")):::hidden
        x2["x<sub>2</sub>"]:::input --> h2(("h<sub>2</sub>")):::hidden
        x3["x<sub>3</sub>"]:::input --> h3(("h<sub>3</sub>")):::hidden

        h0(("h<sub>0</sub>")):::hidden -->|"W<sub>hh</sub>"| h1
        h1 -->|"W<sub>hh</sub>"| h2
        h2 -->|"W<sub>hh</sub>"| h3
        h3 --> ...

        h1 --> y1["y<sub>1</sub>"]:::output
        h2 --> y2["y<sub>2</sub>"]:::output
        h3 --> y3["y<sub>3</sub>"]:::output
    end

    %% 增加边框以避免标题遮挡，使用虚线风格
    style Unrolled fill:#FFFFFF,stroke:#D6D6D6,stroke-dasharray: 5 5
```

#### 2.3.2 动力学状态方程 (State Equations)

RNN 可以被看作是一个离散时间的**动力系统 (Dynamical System)**。它的核心是**隐状态 (Hidden State)** $\mathbf{h}_t$，充当系统的“记忆”。

##### 1. 状态更新方程 (State Update)
在时刻 $t$，隐状态 $\mathbf{h}_t$ 由当前的输入 $\mathbf{x}_t$ 和上一时刻的状态 $\mathbf{h}_{t-1}$ 共同决定：

$$ \mathbf{h}_t = \tanh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h) $$

*   $\mathbf{W}_{hh} \in \mathbb{R}^{d_h \times d_h}$：**状态-状态权重** (State-to-State weights)，控制记忆如何随时间演化。
*   $\mathbf{W}_{xh} \in \mathbb{R}^{d_h \times d_x}$：**输入-状态权重** (Input-to-State weights)，将新输入写入记忆。
*   $\tanh$：**非线性激活**。选用 $\tanh$ 是因为它将值压缩在 $(-1, 1)$ 之间，能一定程度上防止状态值在多次迭代后爆炸（相比于 ReLU）。

##### 2. 输出方程 (Output Equation)
输出通常基于当前的隐状态：

$$ \mathbf{y}_t = \text{softmax}(\mathbf{W}_{hy} \mathbf{h}_t + \mathbf{b}_y) $$

##### 3. 动力学解释
如果我们将输入 $\mathbf{x}_t$ 视为外部驱动力，RNN 就是一个**非线性受迫振动系统**。
*   如果 $\mathbf{x}_t = 0$（无输入），局部动力学由 $\operatorname{diag}(\tanh'(\mathbf z_t))\mathbf W_{hh}$ 的 Jacobian 决定，而不是只由 $\mathbf W_{hh}$ 的特征值决定。线性化模型中，小于 1 的算子增益倾向于收缩，大于 1 的增益可能放大扰动；非线性饱和、非正规矩阵和随时间变化的状态都会改变这一简单图景。

---

#### 2.3.3 随时间反向传播 (BPTT - Backpropagation Through Time)

训练 RNN 的算法叫 BPTT。本质上，它就是将 RNN 展开成一个深层网络，然后应用标准的反向传播。但不同的是，所有层**共享**同一组权重 $\mathbf{W}_{hh}$。

##### 1. 误差项的递归推导 (Derivation of Error Term)
我们定义 $t$ 时刻隐状态的梯度为 $\delta_t = \frac{\partial \mathcal{L}}{\partial \mathbf{h}_t}$。
隐状态 $\mathbf{h}_t$ 在计算图中流向了两个分支：
1.  **当前输出**：参与计算 $\mathbf{y}_t$，产生当前时刻的损失 $\mathcal{L}_t$。
2.  **未来状态**：传递给 $\mathbf{h}_{t+1}$，影响未来所有时刻的损失。

根据多元微积分的链式法则，总梯度 $\delta_t$ 是这两部分梯度之和：

$$ \delta_t = \underbrace{\frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t}}_{\text{直接梯度}} + \underbrace{\left(\frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t}\right)^T \delta_{t+1}}_{\text{传递梯度}} $$

这是一个**时间逆序的递归公式**。这意味着我们可以从最后时刻 $T$ 开始（此时没有未来项），反向一步步算出所有时刻的 $\delta_t$。

##### 2. 参数梯度的具体计算 (Gradient Calculation)
一旦计算出 $\delta_t$，我们就可以算出各个参数在 $t$ 时刻的梯度贡献。
回顾状态方程 $\mathbf{h}_t = \tanh(\mathbf{z}_t)$，其中 $\mathbf{z}_t = \mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h$。
根据链式法则，先计算激活函数前的梯度 $\mathbf{d}_t$：
$$ \mathbf{d}_t = \frac{\partial \mathcal{L}}{\partial \mathbf{z}_t} = \delta_t \circ \tanh'(\mathbf{z}_t) $$
*注：$\circ$ 表示逐元素乘积 (Hadamard product)。*

由于所有时间步共享参数，总梯度是所有时刻贡献的累加：

$$ \begin{aligned} \frac{\partial \mathcal{L}}{\partial \mathbf{W}_{hh}} &= \sum_{t=1}^T \mathbf{d}_t \mathbf{h}_{t-1}^T \\ \frac{\partial \mathcal{L}}{\partial \mathbf{W}_{xh}} &= \sum_{t=1}^T \mathbf{d}_t \mathbf{x}_t^T \\ \frac{\partial \mathcal{L}}{\partial \mathbf{b}_h} &= \sum_{t=1}^T \mathbf{d}_t \end{aligned} $$

这些公式展示了 BPTT 的可计算性：通过一次反向遍历，我们收集了更新所有参数所需的信息。

##### 3. 梯度流图解 (Gradient Flow)

```mermaid
graph RL
    %% 样式定义
    classDef loss fill:#F8CECC,stroke:#B85450,color:#000000;
    classDef grad fill:#D5E8D4,stroke:#82B366,color:#000000;
    classDef weight_grad fill:#FFF2CC,stroke:#D6B656,color:#000000;

    subgraph TimeStep_t_plus_1 ["Time Step t+1"]
        direction TB
        L_next["Loss<sub>t+1</sub>"]:::loss --> d_next(("δ<sub>t+1</sub>")):::grad
    end

    subgraph TimeStep_t ["Time Step t"]
        direction TB
        L_curr["Loss<sub>t</sub>"]:::loss --> d_curr(("δ<sub>t</sub>")):::grad
    end

    subgraph TimeStep_t_minus_1 ["Time Step t-1"]
        direction TB
        L_prev["Loss<sub>t-1</sub>"]:::loss --> d_prev(("δ<sub>t-1</sub>")):::grad
    end

    %% 时间轴反向传播
    d_next -->|"∂h<sub>t+1</sub>/∂h<sub>t</sub>"| d_curr
    d_curr -->|"∂h<sub>t</sub>/∂h<sub>t-1</sub>"| d_prev

    %% 权重的梯度累加
    d_next -.->|Accumulate| dW["∇W<sub>hh</sub>"]:::weight_grad
    d_curr -.->|Accumulate| dW
    d_prev -.->|Accumulate| dW

    style dW stroke-width:3px,stroke-dasharray: 5 5
```

---

#### 2.3.4 数学困境：梯度消失与爆炸 (The Vanishing/Exploding Gradient Problem)

BPTT 的链式法则没有缺口，但长序列会产生很长的 Jacobian 乘积，数值上可能衰减或放大。问题因此不在反向传播规则本身，而在状态转移的长期条件数。

##### 1. 逐步推导 Jacobian 矩阵
先考察单步 Jacobian $\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}}$。
考虑状态更新公式 $\mathbf{h}_t = \tanh(\mathbf{z}_t)$，其中 $\mathbf{z}_t = \mathbf{W}_{hh} \mathbf{h}_{t-1} + \dots$。

根据链式法则，单步反向传播的 Jacobian 矩阵为：
$$ \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}} = \underbrace{\text{diag}(\tanh'(\mathbf{z}_t))}_{\text{激活函数的导数}} \cdot \underbrace{\mathbf{W}_{hh}}_{\text{权重矩阵}} $$

##### 2. 时间轴上的连乘效应
当梯度需要从 $t$ 时刻传回遥远的 $k$ 时刻（$t \gg k$）时，我们需要将中间所有的单步 Jacobian 矩阵相乘：

$$ \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_k} = \prod_{j=k+1}^t \frac{\partial \mathbf{h}_j}{\partial \mathbf{h}_{j-1}} = \prod_{j=k+1}^t \left( \text{diag}(\tanh'(\mathbf{z}_j)) \cdot \mathbf{W}_{hh} \right) $$

这里反复出现 $\mathbf{W}_{hh}$ 与状态相关激活导数的乘积。就像 $0.9^{100} \approx 0$ 和 $1.1^{100} \approx 13780$ 一样，连续 Jacobian 的收缩或放大会使梯度范数近似指数变化。

##### 3. 线性化的谱分析
若暂时忽略非线性并反复使用同一个正规矩阵，谱半径可以给出长期趋势的直觉；对一般非正规矩阵和有限步梯度，奇异值与完整 Jacobian 乘积更直接控制范数。因此下面只是标量化启发，不是充分必要条件。令 $\rho(\mathbf W_{hh})$ 表示谱半径：

*   **收缩直觉 ($\rho < 1$)**：
    *   在上述简化条件下，反复乘法趋于衰减。
    *   **后果**：模型“遗忘”了长距离的历史信息。例如在处理长句时，句首的主语无法影响句尾的动词形式。
*   **放大直觉 ($\rho > 1$)**：
    *   某些方向可能增长；是否造成梯度爆炸仍取决于激活导数和完整矩阵乘积。
    *   **后果**：权重更新过大，导致 Loss 震荡甚至溢出 (NaN)。

##### 4. 解决方案
*   **梯度裁剪 (Gradient Clipping)**：
        *   若 $\|\mathbf{g}\| > \text{threshold}$，则令 $\mathbf{g} \leftarrow \mathbf{g} \cdot \frac{\text{threshold}}{\|\mathbf{g}\|}$。这会限制单次更新的范数，但不消除造成放大的长期 Jacobian。
*   **合理的初始化策略 (Initialization Strategy)**：
    *   **$\mathbf{W}_{hh}$ (Recurrent Weights)**：常用**正交初始化 (Orthogonal Initialization)**。正交矩阵的奇异值为 1，可在训练初期减少线性变换本身对梯度范数的缩放；非线性和后续参数更新仍会改变这一性质。
    *   **$\mathbf{W}_{xh}$ (Input Weights)**：推荐使用 **Xavier/Glorot** 初始化（配合 $\tanh$）或 **Kaiming/He** 初始化（配合 ReLU），确保输入信号的方差在传播过程中保持稳定。
    *   **$\mathbf{b}_h$ (Bias)**：通常初始化为 **0**。但在使用 LSTM 的遗忘门时，有时会初始化为正数（如 1.0）以鼓励模型在训练初期“记住”信息。
*   **ReLU 激活函数**：
    *   $\text{ReLU}'(x)$ 在正区间为 1，可减少饱和导数造成的衰减；循环权重仍可能导致爆炸，负区间的零导数也会丢失梯度。

    <img src="chapter_02/images/activation_gradients.png" width="80%" />

*   **门控机制 (LSTM/GRU)**：
    *   通过带门控的加法状态路径改善长期信用分配，是重要缓解方案，但遗忘门连乘仍可能衰减，也不能消除所有长序列瓶颈。详见 **2.4** 节。

---

#### 2.3.5 经典变体：双向 RNN 与 深层 RNN

##### 1. 双向 RNN (Bi-directional RNN)
在很多任务中，上下文不仅来自“过去”，也来自“未来”。
例如填空题："The ___ sat on the mat."
为了填出 "cat"，我们需要看前面的 "The"，也要看后面的 "sat"。

```mermaid
graph LR
    %% 样式定义
    classDef forward fill:#DAE8FC,stroke:#6C8EBF,color:#000000;
    classDef backward fill:#FFF2CC,stroke:#D6B656,color:#000000;
    classDef input fill:#F5F5F5,stroke:#666666,color:#000000;

    subgraph Inputs
        direction TB
        x1["x<sub>1</sub>"]:::input
        x2["x<sub>2</sub>"]:::input
        x3["x<sub>3</sub>"]:::input
    end

    subgraph Forward ["前向层 (Forward)"]
        direction LR
        h_f1(("→h<sub>1</sub>")):::forward --> h_f2(("→h<sub>2</sub>")):::forward
        h_f2 --> h_f3(("→h<sub>3</sub>")):::forward
    end

    subgraph Backward ["后向层 (Backward)"]
        direction RL
        h_b3(("←h<sub>3</sub>")):::backward --> h_b2(("←h<sub>2</sub>")):::backward
        h_b2 --> h_b1(("←h<sub>1</sub>")):::backward
    end

    x1 --> h_f1 & h_b1
    x2 --> h_f2 & h_b2
    x3 --> h_f3 & h_b3

    h_f1 & h_b1 --> y1["y<sub>1</sub>"]
    h_f2 & h_b2 --> y2["y<sub>2</sub>"]
    h_f3 & h_b3 --> y3["y<sub>3</sub>"]
```

Bi-RNN 包含两个独立的层：
*   **前向层 (Forward Layer)**：$\vec{\mathbf{h}}_t$ 从 $t=1$ 读到 $T$。
*   **后向层 (Backward Layer)**：$\overleftarrow{\mathbf{h}}_t$ 从 $t=T$ 读到 $1$。
*   **输出**：通常将两者拼接 $\mathbf{y}_t = [\vec{\mathbf{h}}_t; \overleftarrow{\mathbf{h}}_t]$。

##### 2. 深层 RNN (Stacked/Deep RNN)
类似于 CNN，我们可以堆叠多个 RNN 层来提取更高层级的特征。
第 $l$ 层的输入是第 $l-1$ 层的隐状态：
$$ \mathbf{h}_t^{(l)} = \sigma(\mathbf{W}_{hh}^{(l)} \mathbf{h}_{t-1}^{(l)} + \mathbf{W}_{xh}^{(l)} \mathbf{h}_t^{(l-1)} + \mathbf{b}^{(l)}) $$

#### 2.3.6 RNN 在 Transformer 时代的位置

RNN 的递推形式适合按序接收数据，但它同时带来两项可量化限制：训练时的时间串行深度，以及固定维状态对历史信息的压缩。

##### 1. 串行深度与状态瓶颈
*   **计算效率低（串行依赖）**：
    *   由于 $h_t$ 必须等待 $h_{t-1}$，RNN 在单条序列的时间轴上难以并行；批维、特征维和矩阵运算仍可使用 GPU 并行。相较可并行处理各位置的 Transformer，这一串行深度会限制长序列训练吞吐。
*   **信息压缩瓶颈（Memory Bottleneck）**：
    *   RNN 把历史压缩进固定大小状态 $h_t$。有限状态会造成信息容量与可检索性的取舍；LSTM 门控可以延长保留时间，但不保证任意早期细节可被精确恢复。

##### 2. Transformer 的不同取舍
2017 年后，Transformer 在许多 NLP 基准和大规模预训练系统中逐步取代 RNN 成为主流主干：
*   **并行训练**：Self-Attention 机制允许同时计算序列中所有位置的关系，打破了训练阶段按时间步递推的串行限制。
*   **全局视野**：Transformer 可以在上下文窗口内直接建模任意两个位置之间的依赖，而不必把全部历史压缩进单个隐状态，因此大幅缓解了长距离依赖问题。

##### 3. 递推结构仍适用的场景
虽然在通用大模型（LLM）领域 Transformer 仍是主流主干，但循环状态、状态空间模型和线性时间序列模型在下列约束下仍有明确价值：

*   **极低资源推理 (Edge AI)**：
    *   Transformer 在推理时需要缓存历史信息 (KV Cache)，显存占用随序列长度线性增长 $O(L)$。
    *   RNN 在推理时只需要维护一个状态 $h_t$，内存占用是恒定的 $O(1)$。这使得 RNN 极适合**嵌入式设备、实时语音处理**等对延迟和内存极其敏感的场景。
*   **处理流式长序列**：
    *   RNN 天生适合处理流式数据（Streaming Data），可以持续读入新 token 并更新有限状态；不过由于状态维度有限，它也会遗忘或压缩早期信息。Transformer 则通常受限于上下文窗口长度和 KV Cache 成本。
*   **现代递推与状态空间架构 (RWKV / Mamba)**：
    *   **RWKV** 与 **Mamba (State Space Models)** 采用不同的参数化和并行算法，但都研究“训练时利用并行 scan/等价形式、流式推理时维护固定维度状态”的权衡。实际内存仍随层数、batch 和状态维度增长，只是不随已处理序列长度线性累积 KV Cache。
    *   这说明**循环状态记忆**仍是有价值的设计原语；Mamba 属于选择性状态空间模型，与传统 RNN 有递推相似性，但参数化和并行算法不同。
<a id="section-2-4"></a>

## 2.4 LSTM 与门控状态更新
### 2.4 Long Short-Term Memory (LSTM)

普通 RNN 的长期 Jacobian 由状态相关矩阵反复相乘。LSTM 增加一条带门控的加法状态路径，使模型可以学习接近恒等的跨步传递；下面由更新方程直接检查这条路径何时保留梯度，并说明它如何进入 Seq2Seq 架构。

#### 2.4.1 近恒等状态路径 (Constant Error Carousel, CEC)

标准 RNN 的时间 Jacobian 连乘可能使梯度衰减或爆炸；饱和的 $\tanh$ 导数会进一步促成衰减，但并非每一步都必然变小。
Hochreiter & Schmidhuber (1997) 的洞察是：为了通过时间反向传播误差，我们需要一个导数为 1 的单元。
$$ \frac{\partial \mathbf{C}_t}{\partial \mathbf{C}_{t-1}} \approx \mathbf{I} $$
这就是细胞状态 $\mathbf{C}_t$ 的设计初衷。如果遗忘门长期接近 1，误差信号可以沿着 $\mathbf{C}_t$ 的线性路径保留很久；如果门值明显小于 1，衰减仍会按乘法规律累积。

#### 2.4.2 LSTM 的两类状态与三个门

**关键区别：双轨道记忆 (Dual State)**
LSTM 把状态拆成两条轨道。细胞状态 $\mathbf C_t$ 通过“保留旧值加写入新值”的加法路径跨时间传递，隐状态 $\mathbf h_t$ 则由细胞状态经过 $\tanh$ 和输出门形成，供当前预测与下一步门计算使用。遗忘门控制旧信息保留比例，输入门控制候选内容写入比例，输出门控制当前暴露多少状态。比如处理主谓一致时，网络可以在读到主语后让相关细胞分量保持，在遇到新从句时有选择地更新，而不必把全部历史每步重新编码。

**架构图解**：

```mermaid
graph LR
    %% 样式定义
    classDef cell fill:#D5E8D4,stroke:#82B366,stroke-width:2px,color:#000000;
    classDef op fill:#DAE8FC,stroke:#6C8EBF,stroke-width:2px,color:#000000;
    classDef add fill:#F8CECC,stroke:#B85450,stroke-width:2px,color:#000000;

    subgraph LSTM_Cell ["LSTM Cell"]
        direction LR
        Xt[Input Xt]
        Ht_1[Prev Hidden Ht-1]
        Ct_1[Prev Cell Ct-1]

        %% Gates
        Ht_1 & Xt --> ForgetGate[Forget Gate ft]
        Ht_1 & Xt --> InputGate[Input Gate it]
        Ht_1 & Xt --> OutputGate[Output Gate ot]
        Ht_1 & Xt --> CellUpdate[Candidate C't]

        %% Operations
        ForgetGate -->|x| Mult1((x)):::op
        Ct_1 --> Mult1

        InputGate -->|x| Mult2((x)):::op
        CellUpdate -->|tanh| Mult2

        Mult1 --> Add((+)):::add
        Mult2 --> Add
        Add --> Ct[New Cell Ct]:::cell

        Ct -->|tanh| TanhCt
        OutputGate -->|x| Mult3((x)):::op
        TanhCt --> Mult3
        Mult3 --> Ht[New Hidden Ht]
    end
```

#### 2.4.3 前向传播方程组

设 $\mathbf{x}_t$ 为输入，$\mathbf{h}_{t-1}$ 为上一隐状态。所有 $\mathbf{W} \in \mathbb{R}^{d_h \times (d_x + d_h)}$。

1.  **遗忘门 (Forget Gate)**：决定 $\mathbf{C}_{t-1}$ 中多少信息被保留。
    $$ \mathbf{f}_t = \sigma(\mathbf{W}_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f) $$
    *(矩阵展开形式)*：
    $$ \mathbf{f}_t = \sigma \left( \begin{bmatrix} \mathbf{W}_{fh} & \mathbf{W}_{fx} \end{bmatrix} \begin{bmatrix} \mathbf{h}_{t-1} \\ \mathbf{x}_t \end{bmatrix} + \mathbf{b}_f \right) $$
2.  **输入门 (Input Gate)**：决定多少新信息 $\tilde{\mathbf{C}}_t$ 被写入。
    $$ \mathbf{i}_t = \sigma(\mathbf{W}_i \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i) $$
    $$ \tilde{\mathbf{C}}_t = \tanh(\mathbf{W}_C \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_C) $$
3.  **细胞状态更新 (Cell Update)**：
    $$ \mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{C}}_t $$
    *   **加法更新**：这是 LSTM 避免梯度消失的关键。相比于 RNN 的矩阵乘法更新，加法运算的导数性质更优。
4.  **输出门 (Output Gate)**：
    $$ \mathbf{o}_t = \sigma(\mathbf{W}_o \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o) $$
    $$ \mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{C}_t) $$

#### 2.4.4 梯度流分析 (Gradient Flow Analysis)

考虑 $\mathbf{C}_t$ 对 $\mathbf{C}_{t-1}$ 的偏导数：
$$ \frac{\partial \mathbf{C}_t}{\partial \mathbf{C}_{t-1}} = \frac{\partial}{\partial \mathbf{C}_{t-1}} (\mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{C}}_t) $$
$$ = \text{diag}(\mathbf{f}_t) + \underbrace{\mathbf{C}_{t-1} \odot \sigma'(...) \mathbf{W}_f \dots}_{term_1} + \underbrace{\mathbf{i}_t \odot \tanh'(...) \mathbf{W}_C \dots}_{term_2} + \dots $$

通常，$term_1, term_2$ 等间接项较小，主导项是 **$\mathbf{f}_t$**。
$$ \frac{\partial \mathbf{C}_T}{\partial \mathbf{C}_k} \approx \prod_{t=k+1}^T \text{diag}(\mathbf{f}_t) $$

<img src="chapter_02/images/lstm_forget_gate_retention.svg" width="85%" />

*   当遗忘门 $\mathbf{f}_t \approx \mathbf{1}$（开启）且持续稳定时，梯度可以在较长时间跨度内保持较大幅度；但只要每一步都略小于 1，长期保留量仍会指数式下降。
*   网络可以学习在特定时刻将 $\mathbf{f}_t$ 设为 0（重置记忆），或设为 1（保持记忆）。

---

#### 2.4.5 GRU (Gated Recurrent Unit)

GRU (Cho et al., 2014) 是 LSTM 的简化版，合并了 $\mathbf{C}_t$ 和 $\mathbf{h}_t$，并将遗忘门和输入门合并为更新门 $\mathbf{z}_t$。

$$ \mathbf{z}_t = \sigma(\mathbf{W}_z \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t]) $$
$$ \mathbf{r}_t = \sigma(\mathbf{W}_r \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t]) $$
$$ \tilde{\mathbf{h}}_t = \tanh(\mathbf{W} \cdot [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t]) $$
$$ \mathbf{h}_t = (\mathbf{1} - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t $$

*   **参数与计算**：GRU 的门和状态较少，单步参数量通常低于同宽度 LSTM；实际速度与效果仍取决于实现、宽度和任务。
*   **数学直觉**：$\mathbf{z}_t$ 在 0 和 1 之间插值，显式地建模“保持旧状态”与“写入新状态”的权衡。

---

#### 2.4.6 Sequence-to-Sequence (Seq2Seq) 与注意力前奏

Seq2Seq 用一个递推编码器读取输入序列，再由递推解码器生成长度可变的输出。其统计目标是条件分布 $P(\mathbf{Y}|\mathbf{X})$，其中 $\mathbf{X}=(\mathbf{x}_1, \dots, \mathbf{x}_N)$，$\mathbf{Y}=(\mathbf{y}_1, \dots, \mathbf{y}_M)$。

```mermaid
graph LR
    subgraph Encoder
    x1[x1] --> enc_h1((h1))
    x2[x2] --> enc_h2((h2))
    enc_h1 --> enc_h2
    end

    enc_h2 -- Context Vector c --> dec_s0

    subgraph Decoder
    dec_s0((s0)) --> dec_s1((s1)) --> y1[y1]
    y1 --> dec_s2((s2)) --> y2[y2]
    end

    style Encoder fill:#DAE8FC,stroke:#6C8EBF,stroke-width:2px,color:#000000
    style Decoder fill:#FFF2CC,stroke:#D6B656,stroke-width:2px,color:#000000
    style dec_s0 fill:#E1D5E7,stroke:#9673A6,stroke-width:2px,color:#000000
```

1.  **Encoder**：
    Encoder 读取输入序列 $\mathbf{X}$，并生成隐状态序列：
    $$ \mathbf{h}_t = \text{LSTM}_{enc}(\mathbf{x}_t, \mathbf{h}_{t-1}) $$
    最终的上下文向量由最后一个隐状态得到：
    $$ \mathbf{c} = \mathbf{h}_N \quad (\text{Context Vector}) $$
2.  **Decoder**：
    Decoder 根据上下文向量 $\mathbf{c}$ 和上一步输出 $\mathbf{y}_{t-1}$ 更新自身状态 $\mathbf{s}_t$：
    $$ \mathbf{s}_t = \text{LSTM}_{dec}(\mathbf{y}_{t-1}, \mathbf{s}_{t-1}, \mathbf{c}) $$
    计算当前时刻输出词的概率分布：
    $$ P(\mathbf{y}_t | \mathbf{y}_{<t}, \mathbf{X}) = \text{softmax}(\mathbf{W}_{out} \mathbf{s}_t) $$
