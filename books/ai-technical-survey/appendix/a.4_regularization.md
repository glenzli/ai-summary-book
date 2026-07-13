# 附录 A.4 正则化与 Dropout (Regularization & Dropout)

## A.4.1 拉格朗日乘数法 (Lagrange Multiplier) 复习

拉格朗日乘数法是解决 **带约束优化问题** 的通用数学工具。它是理解正则化“约束优化”视角的数学基础。

### 1. 形式化定义 (Formal Definition)

**原始问题 (Primal Problem)**：
假设我们需要最小化一个多元函数 $f(\mathbf{x})$，同时受到一个等式约束 $g(\mathbf{x}) = 0$ 的限制：
$$
\begin{aligned}
\min_{\mathbf{x}} \quad & f(\mathbf{x}) \\
\text{s.t.} \quad & g(\mathbf{x}) = 0
\end{aligned}
$$

**拉格朗日函数 (The Lagrangian)**：
为分析上述约束问题，我们引入标量 $\lambda$（拉格朗日乘子）并构造拉格朗日函数。它服务于驻点、KKT 或对偶鞍点条件，不等于把 $\mathbf x$ 与 $\lambda$ 放在一起做一次普通无约束最小化：
$$ \mathcal{L}(\mathbf{x}, \lambda) = f(\mathbf{x}) + \lambda g(\mathbf{x}) $$

**极值的必要条件**：
在 $f,g$ 可微且满足约束资格条件（单个等式时可取 $\nabla g(\mathbf x^*)\ne0$）下，若 $\mathbf{x}^*$ 是局部约束最优解，则存在乘子 $\lambda^*$ 使拉格朗日函数满足一阶必要条件：

1.  **对 $\mathbf{x}$ 求偏导**：
    $$ \nabla_{\mathbf{x}} \mathcal{L} = \nabla f(\mathbf{x}) + \lambda \nabla g(\mathbf{x}) = 0 \implies \nabla f(\mathbf{x}) = -\lambda \nabla g(\mathbf{x}) $$
    *(这说明在最优解处，目标函数的梯度与约束函数的梯度平行)*

2.  **对 $\lambda$ 求偏导**：
    $$ \nabla_{\lambda} \mathcal{L} = g(\mathbf{x}) = 0 $$
    *(这确保了约束条件被满足)*

**必要条件的推导 (Derivation)**
为什么最优解必须满足上述条件？我们可以通过 **正交分解** 来证明。

1.  **切向量与梯度垂直**：
    假设我们在约束曲面 $g(\mathbf{x})=0$ 上移动，路径为 $\mathbf{x}(t)$，且 $\mathbf{x}(0)=\mathbf{x}^*$。
    由于始终在约束面上，$g(\mathbf{x}(t)) \equiv 0$。对 $t$ 求导：
    $$ \frac{d}{dt} g(\mathbf{x}(t)) = \nabla g(\mathbf{x}) \cdot \mathbf{x}'(t) = 0 $$
    这说明：**约束函数的梯度 $\nabla g$ 垂直于约束曲面上任意切向量 $\mathbf{v} = \mathbf{x}'(0)$**。

2.  **目标函数的极值性**：
    如果 $\mathbf{x}^*$ 是 $f(\mathbf{x})$ 的局部极小值点，那么沿着约束曲面的任何方向移动，目标函数的值都不应该减小（一阶变化率为0）。
    $$ \frac{d}{dt} f(\mathbf{x}(t)) \bigg|_{t=0} = \nabla f(\mathbf{x}^*) \cdot \mathbf{x}'(0) = \nabla f(\mathbf{x}^*) \cdot \mathbf{v} = 0 $$
    这说明：**目标函数的梯度 $\nabla f$ 也垂直于约束曲面上任意切向量 $\mathbf{v}$**。

3.  **梯度共线**：
    既然 $\nabla f$ 和 $\nabla g$ 都垂直于同一个切平面（即垂直于所有的 $\mathbf{v}$），在几何上它们必须 **共线**（假设 $\nabla g \neq 0$）。
    因此，一定存在一个标量 $\lambda$，使得：
    $$ \nabla f(\mathbf{x}^*) = -\lambda \nabla g(\mathbf{x}^*) $$
    移项即得 $\nabla f + \lambda \nabla g = 0$，这正是 $\nabla_{\mathbf{x}} \mathcal{L} = 0$。

### 2. 几何直觉 (Geometric Intuition)

理解了公式后，我们再看图：

<img src="images/lagrange_geometric.png" width="60%" />

*   **登山的比喻**：
    想象我们在爬山（寻找 $f(\mathbf{x})$ 的最低点），但被限制只能在一条特定的小路上行走（约束 $g(\mathbf{x})=0$，即上图中的红线）。
*   **相交 (Intersecting) vs 相切 (Tangent)**：
    *   如果我们走在红线上的某点，发现 $f(\mathbf{x})$ 的等高线（蓝圈）与红线 **相交** (Cross)，说明红线穿过了等高线，我们一定可以顺着红线往“圈内”走，找到更低的点。
    *   只有当 $f(\mathbf{x})$ 的等高线与红线 **相切** (Touch) 时，我们才没法再通过沿红线移动来降低 $f$ 的值了。这时我们就找到了极值点。
*   **梯度的关系**：
    “相切”在数学上就意味着两条曲线在接触点有相同的法线方向。函数的法线方向就是梯度方向。因此，**最优解处，两个梯度方向平行**。

### 3. 对应到机器学习正则化
范数惩罚与范数约束在**适当条件下**可以通过拉格朗日对偶联系起来，但不是无条件一一对应。考虑不等式约束

$$
\min_{\mathbf w}\;\operatorname{Loss}(\mathbf w)
\quad\text{s.t.}\quad \Omega(\mathbf w)\le C.
$$

其拉格朗日函数为

$$
\mathcal L(\mathbf w,\lambda)
=\operatorname{Loss}(\mathbf w)+\lambda\bigl(\Omega(\mathbf w)-C\bigr),
\qquad \lambda\ge 0.
$$

在凸性、约束资格条件（例如 Slater 条件）和强对偶成立时，KKT 条件还要求原始可行性、对偶可行性与互补松弛

$$
\lambda\bigl(\Omega(\mathbf w)-C\bigr)=0.
$$

对某个活跃预算 $C$，可能存在乘子 $\lambda$，使约束问题与 $\operatorname{Loss}(\mathbf w)+\lambda\Omega(\mathbf w)$ 共享最优解；反过来，一个惩罚问题的解可对应预算 $C=\Omega(\mathbf w_\lambda)$。但映射未必唯一，约束不活跃时可有 $\lambda=0$，非凸问题或多重最优解下也不能宣称完全等价。实践中，$C$ 与 $\lambda$ 都是控制容量偏好的不同超参数化。

---

## A.4.2 正则化的贝叶斯视角 (Bayesian View of Regularization)

除了上述的“约束优化”视角，正则化还可以从概率统计的视角——即贝叶斯推断中 **先验知识 (Prior Knowledge)** 的自然体现——来理解。

### 1. 频率学派与贝叶斯视角 (Frequentist and Bayesian Views)
*   **频率学派**通常把未知参数 $\mathbf{w}$ 视为固定量，并用重复抽样分布分析估计量与检验程序。极大似然估计（MLE）是常用方法之一，但不是频率统计的全部。
*   **贝叶斯视角**用先验分布表达对未知参数的不确定性，再通过观测数据得到后验分布。这里说参数“服从分布”是推断模型的一部分，不必理解为参数在一次已固定的现实系统里不断随机变化。

### 2. MAP (极大后验估计) 推导
贝叶斯公式的核心是：
$$ \text{Posterior} \propto \text{Likelihood} \times \text{Prior} $$
$$ P(\mathbf{w}|\mathcal{D}) \propto P(\mathcal{D}|\mathbf{w}) P(\mathbf{w}) $$

我们的目标是寻找最可能的 $\mathbf{w}$，即最大化后验概率（MAP）：
$$
\begin{aligned}
\mathbf{w}_{MAP} &= \arg\max_{\mathbf{w}} \log P(\mathbf{w}|\mathcal{D}) \\
&= \arg\max_{\mathbf{w}} \left( \underbrace{\log P(\mathcal{D}|\mathbf{w})}_{\text{Log-Likelihood}} + \underbrace{\log P(\mathbf{w})}_{\text{Log-Prior}} \right) \\
&= \arg\min_{\mathbf{w}} \left( \underbrace{-\log P(\mathcal{D}|\mathbf{w})}_{\text{Loss Function}} + \underbrace{- \log P(\mathbf{w})}_{\text{Regularization}} \right)
\end{aligned}
$$

*   **直观解释**：
    *   **Likelihood** 代表“数据说了算”：数据希望参数变成什么样（通常为了拟合数据，参数会变大、变复杂）。
    *   **Prior** 代表“先验信仰”：在没看数据前，我们认为参数应该是什么样（通常我们认为世界是简单的，参数应该接近 0）。
    *   **Posterior** 是两者的博弈与平衡。

<img src="images/bayesian_update.png" width="80%" />

上图展示了一个一维参数 $w$ 的更新过程：
*   **Likelihood (蓝色虚线)**：数据告诉我们要把 $w$ 设在 4 附近。
*   **Prior (黄色虚线)**：我们预设 $w$ 应该在 0 附近。
*   **Posterior (红色实线)**：最终的估计值在 0 和 4 之间。正则化就像一根橡皮筋，把参数从数据拟合的“过拟合点”往回拉，拉向 0 点。

### 3. 为什么 L2 对应高斯，L1 对应拉普拉斯？

不同的正则化项对应了不同的先验假设。

**Case 1: L2 正则化 $\iff$ 各向同性高斯先验 (Isotropic Gaussian Prior)**
假设 $d$ 维权重 $\mathbf{w}$ 服从 $\mathcal{N}(\mathbf 0, \tau^2 I_d)$：
$$ P(\mathbf{w}) \propto \exp\left( -\frac{\|\mathbf{w}\|^2}{2\tau^2} \right) $$
取负对数后：
$$ -\log P(\mathbf{w}) \propto \|\mathbf{w}\|^2 $$
这正是 **L2 正则化**。高斯分布在 0 附近是平滑的凸起，它希望权重集中在 0 附近，但对于稍微偏离 0 的小权重也能容忍。

**Case 2: L1 正则化 $\iff$ 独立拉普拉斯先验 (Independent Laplace Prior)**
假设各坐标独立且 $w_j\sim\operatorname{Laplace}(0,b)$，则联合密度满足：
$$ P(\mathbf{w}) \propto \exp\left( -\frac{\|\mathbf{w}\|_1}{b} \right) $$
取负对数后：
$$ -\log P(\mathbf{w}) \propto \|\mathbf{w}\|_1 $$
这正是 **L1 正则化**。

<img src="images/bayesian_priors.png" width="80%" />

**由图可见区别**：
*   **高斯分布 (蓝色)**：在 $x=0$ 处是圆滑的。它虽然通过概率密度“压制”大权重，但不会强制权重为 0。
*   **拉普拉斯分布 (黄色)**：其负对数先验是 $|w|/b$，在 $w=0$ 处不可微。L1-MAP 容易产生精确零的关键是这个目标函数的**非光滑折点**及相应最优性条件，而不是概率密度对单点产生物理“吸力”；连续先验在任意单点上的概率质量仍为 0。

---

## A.4.3 优化视角：权重衰减与梯度更新 (Weight Decay in Optimization)

在正文 2.1.2 节中，我们提到了工程上常用的 **Weight Decay**。本节推导普通、未预条件 SGD 下 L2 penalty 与比例权重衰减的代数对应；对 Adam 等自适应方法，这一等价关系一般不成立。

### 1. L2 正则化 $\Leftrightarrow$ 权重比例衰减 (Proportional Decay)
假设目标函数包含 L2 正则项：
$$ J(w) = J_{orig}(w) + \frac{1}{2}\lambda \|w\|^2 $$

**梯度计算**：
$$ \nabla J(w_t) = \nabla J_{orig}(w_t) + \lambda w_t $$

**SGD 更新规则**：
我们将梯度代入 SGD 更新公式：
$$
\begin{aligned}
w_{t+1} &= w_t - \eta \nabla J(w_t) \\
        &= w_t - \eta (\nabla J_{orig}(w_t) + \lambda w_t) \\
        &= w_t - \eta \nabla J_{orig}(w_t) - \eta \lambda w_t \\
        &= \underbrace{(1 - \eta \lambda)}_{\text{Decay Factor}} w_t - \eta \nabla J_{orig}(w_t)
\end{aligned}
$$
**结论**：在上述普通 SGD 更新、相同学习率与系数约定下，把 L2 penalty 的梯度加入更新，与同时把旧权重乘以 $1-\eta\lambda$ 给出同一个迭代式。这就是 **Weight Decay** 名称的由来；加入动量、预条件、参数分组或自适应缩放后必须重新核对更新式，不能直接沿用这一结论。

### 2. L1 次梯度与 Proximal 更新
假设目标函数包含 L1 正则项：
$$ J(w) = J_{orig}(w) + \lambda \|w\|_1 $$

**梯度计算**：
由于 $|w|$ 的导数是符号函数 $\text{sign}(w)$（在 0 处不可导，通常取次梯度）：
$$ \nabla J(w) = \nabla J_{orig}(w) + \lambda \text{sign}(w) $$

**SGD 更新规则**：
$$
\begin{aligned}
w_{t+1} &= w_t - \eta (\nabla J_{orig}(w) + \lambda \text{sign}(w)) \\
        &= w_t - \eta \nabla J_{orig}(w) - \eta \lambda \text{sign}(w)
\end{aligned}
$$
这个 vanilla 次梯度更新会朝 0 推动参数，但离散步长可能越过 0 或在其附近振荡；算法没有“到 0 后自动截断并保持为 0”的保证。精确稀疏解通常用 proximal gradient 的 soft-thresholding 更清楚地解释。先对光滑损失走一步

$$ u_t=w_t-\eta\nabla J_{orig}(w_t), $$

再应用 L1 的 proximal operator：

$$
w_{t+1}
=\operatorname{prox}_{\eta\lambda\|\cdot\|_1}(u_t)
=\operatorname{sign}(u_t)\max\bigl(|u_t|-\eta\lambda,0\bigr).
$$

当 $|u_t|\le\eta\lambda$ 时，soft-thresholding 会把坐标精确置零；后续是否保持为零仍取决于光滑损失梯度。

---

## A.4.4 Dropout 的数学机制详解 (Mathematics of Dropout)

本节详细拆解 Dropout 在前向传播、反向传播及测试阶段的数学细节，帮助理解“掩码”与“缩放”的本质。

### 1. 伯努利掩码 (Bernoulli Mask)
在 Dropout 中，核心操作是生成一个 **掩码向量 (Mask Vector)** $\mathbf{r} \in \{0, 1\}^d$。这个向量的每一个元素 $r_j$ 都是独立从伯努利分布中采样的：
$$ r_j \sim \text{Bernoulli}(1-p) $$
这意味着：
*   $P(r_j=1) = 1-p$ （保留）
*   $P(r_j=0) = p$ （丢弃）

### 2. 训练阶段 (Training Phase)
假设某层神经元的原始输出向量为 $\mathbf{h} = [h_1, h_2, \dots, h_d]^T$。
应用 Dropout 后的输出 $\tilde{\mathbf{h}}$ 是 $\mathbf{h}$ 与 $\mathbf{r}$ 的 **Hadamard 积 (Element-wise Product)**，记作 $\odot$。

**操作公式**：
$$ \tilde{\mathbf{h}} = \mathbf{r} \odot \mathbf{h} $$

**数值示例**：
假设 $\mathbf{h} = [10, 20, 30, 40]$，丢弃率 $p=0.5$。
我们随机采样得到掩码 $\mathbf{r} = [1, 0, 1, 0]$。
则应用计算如下：
$$
\tilde{\mathbf{h}} =
\begin{bmatrix} 1 \\ 0 \\ 1 \\ 0 \end{bmatrix} \odot
\begin{bmatrix} 10 \\ 20 \\ 30 \\ 40 \end{bmatrix} =
\begin{bmatrix} 1 \times 10 \\ 0 \times 20 \\ 1 \times 30 \\ 0 \times 40 \end{bmatrix} =
\begin{bmatrix} 10 \\ 0 \\ 30 \\ 0 \end{bmatrix}
$$
可以看到，第 2 和第 4 个神经元的输出被强制置为 0，相当于它们在本次迭代中“消失”了。

### 3. 缩放的必要性 (Scaling)
为什么测试时需要缩放？让我们看期望值。
在训练时，$\tilde{h}_i$ 的期望是：
$$ \mathbb{E}[\tilde{h}_i] = \mathbb{E}[r_i \cdot h_i] = P(r_i=1) h_i + P(r_i=0) \cdot 0 = (1-p)h_i $$

这意味着，下一层神经元接收到的信号总强度（期望值）只有原来的 $(1-p)$ 倍。
如果测试时我们全开（所有神经元都工作），信号强度会恢复为 $1$ 倍（$h_i$）。
这种 **训练/测试时的信号强度不匹配** 会导致网络预测失效。

**解决方案：Inverted Dropout**
为了避免在测试时修改代码，现代框架（如 PyTorch）通常采用 **Inverted Dropout**：**在训练时就提前把信号放大**。

*   **训练时**：$\tilde{\mathbf{h}} = \frac{1}{1-p} (\mathbf{r} \odot \mathbf{h})$
*   **测试时**：直接使用 $\mathbf{h}$，**不做任何改动**。

**验证期望**：
此时训练时的期望变为：
$$ \mathbb{E}[\tilde{h}_i] = \mathbb{E}\left[ \frac{r_i}{1-p} h_i \right] = \frac{1-p}{1-p} h_i = h_i $$
在把 $\mathbf h$ 视为给定时，这使训练期随机掩码输出的条件期望与测试期输出一致；单次样本的激活和更高阶矩仍不同。

### 4. 反向传播 (Backpropagation)
Dropout 不仅影响前向传播，也影响反向传播。（关于反向传播的详细数学原理，请参考 **[附录 A.6 反向传播 (Backpropagation)](a.6_backpropagation.md)**）。
设 Loss 对输出 $\tilde{\mathbf{h}}$ 的梯度为 $\frac{\partial L}{\partial \tilde{\mathbf{h}}}$。
根据链式法则，Loss 对原始 $\mathbf{h}$ 的梯度为：

$$
\frac{\partial L}{\partial \mathbf{h}}
= \frac{\partial L}{\partial \tilde{\mathbf{h}}}
\odot \frac{\mathbf r}{1-p}.
$$

这是前文 **inverted dropout** 前向式的反向传播；若采用未缩放的 $\tilde{\mathbf h}=\mathbf r\odot\mathbf h$，这里才不含 $1/(1-p)$。

**直观含义**：
如果 $r_i=0$（神经元被 Drop），那么梯度乘以 0 也会变成 0。
对该掩码覆盖的样本/位置，这条激活路径的梯度为 0；共享权重仍可能从同一 batch 的其他样本、空间位置或未被丢弃路径获得梯度，因此不能笼统称整个神经元或参数在一轮迭代中“冻结”。
