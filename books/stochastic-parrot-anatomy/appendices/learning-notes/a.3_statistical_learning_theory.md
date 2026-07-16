# 附录 A.3 统计学习理论 (Statistical Learning Theory)
## Appendix A.3 Statistical Learning Theory

本附录将为 1.4 节中的机器学习核心概念提供严谨的数学推导，重点涵盖偏差-方差分解的完整证明以及 VC 维理论简介。

### A.3.1 偏差-方差分解 (Bias-Variance Decomposition) 的完整推导

我们在正文中提到了泛化误差可以分解为偏差、方差和噪音。这里给出严格的推导。

#### 1. 问题设定
假设真实数据生成模型为 $y=f(\mathbf{x})+\epsilon$，其中噪声满足 $\mathbb E[\epsilon\mid\mathbf x]=0$、$\operatorname{Var}(\epsilon\mid\mathbf x)=\sigma^2$（为简化取同方差）。训练集 $\mathcal D$ 自身包含独立采样的训练噪声，并据此得到模型 $\hat f(\mathbf x;\mathcal D)$。在固定测试输入 $\mathbf x$ 处，令独立测试标签为 $y^*=f(\mathbf x)+\epsilon^*$，其中 $\epsilon^*$ 与 $\mathcal D$ 独立。

$$
\operatorname{Error}(\mathbf{x})
=\mathbb{E}_{\mathcal{D},\epsilon^*}
\left[(y^*-\hat f(\mathbf{x};\mathcal D))^2\right].
$$

#### 2. 推导步骤
为了简化符号，简写 $\hat{f}(\mathbf{x}; \mathcal{D})$ 为 $\hat{f}$。
利用 $y^*=f+\epsilon^*$，展开平方项：

$$
\begin{aligned}
\operatorname{Error}(\mathbf{x})
&= \mathbb{E}_{\mathcal{D},\epsilon^*}
\left[(f + \epsilon^* - \hat{f})^2\right] \\
&= \mathbb{E}_{\mathcal{D},\epsilon^*}
\left[(f - \hat{f})^2 + (\epsilon^*)^2 + 2\epsilon^*(f - \hat{f})\right]
\end{aligned}
$$

由于 $\epsilon^*$ 与 $\mathcal D$ 独立且条件均值为 0，交叉项期望为 0：
$$
\mathbb E_{\mathcal D,\epsilon^*}[2\epsilon^*(f-\hat f)]
=2\mathbb E_{\epsilon^*}[\epsilon^*]\,
\mathbb E_{\mathcal D}[f-\hat f]=0,
$$
且 $\mathbb E[(\epsilon^*)^2]=\sigma^2$。

现在关注主要项 $\mathbb{E}_{\mathcal{D}} \left[ (f - \hat{f})^2 \right]$。
这是一个关于随机变量 $\hat{f}$ 的二阶矩。我们利用恒等式 $\mathbb{E}[X^2] = (\mathbb{E}[X])^2 + \text{Var}(X)$ 的变体。
在此，我们在公式中人为引入 $\mathbb{E}_{\mathcal{D}}[\hat{f}]$（记为 $\bar{f}$，即模型的平均预测）：

$$
\begin{aligned}
\mathbb{E}_{\mathcal{D}} \left[ (f - \hat{f})^2 \right] &= \mathbb{E}_{\mathcal{D}} \left[ (f - \bar{f} + \bar{f} - \hat{f})^2 \right] \\
&= \mathbb{E}_{\mathcal{D}} \left[ (f - \bar{f})^2 + (\bar{f} - \hat{f})^2 + 2(f - \bar{f})(\bar{f} - \hat{f}) \right]
\end{aligned}
$$

*   第一项 $(f - \bar{f})^2$ 是常数（相对于 $\mathcal{D}$），即 **偏差平方 ($\text{Bias}^2$)**。
*   第二项 $\mathbb{E}_{\mathcal{D}}[(\bar{f} - \hat{f})^2]$ 正是 $\hat{f}$ 的方差，即 **方差 (Variance)**。
*   第三项交叉项：$2(f - \bar{f}) \mathbb{E}_{\mathcal{D}}[\bar{f} - \hat{f}] = 2(f - \bar{f})(\bar{f} - \bar{f}) = 0$。

#### 3. 最终结果
将所有项合并：
$$
\operatorname{Error}(\mathbf{x})
=\underbrace{\left(f(\mathbf{x})-\mathbb E_{\mathcal D}[\hat f(\mathbf{x};\mathcal D)]\right)^2}_{\text{Bias}^2}
+\underbrace{\mathbb E_{\mathcal D}\!\left[\left(\hat f(\mathbf{x};\mathcal D)-\mathbb E_{\mathcal D}[\hat f(\mathbf{x};\mathcal D)]\right)^2\right]}_{\text{Variance}}
+\underbrace{\sigma^2}_{\text{Independent test noise}}.
$$

这一分解定义了平方损失回归中的偏差、方差与不可约测试噪声。在部分经典模型族中，增加复杂度常降低偏差并提高方差；现代过参数化模型可能出现双下降等不同现象，因此下图只是概念性曲线。

<img src="../../vol-01/chapter_02/images/bias_variance_tradeoff.png" width="80%" />

---

### A.3.2 VC 维与泛化界 (Statistical Learning Theory: VC Dimension & Generalization Bounds)

为什么在训练集上表现好 ($E_{in} \approx 0$)，就意味着在测试集上也表现好 ($E_{out} \approx 0$)？这并非理所当然。统计学习理论（Statistical Learning Theory, SLT）通过引入 **VC 维** 回答了这个问题。

#### 1. 学习的可行性：从霍夫丁不等式开始

对于一个 **固定** 的假设模型 $h$，**霍夫丁不等式 (Hoeffding's Inequality)** 告诉我们，训练误差 $E_{in}(h)$ 和泛化误差 $E_{out}(h)$ 之间的差距大于 $\epsilon$ 的概率是非常小的：

$$ P(|E_{in}(h) - E_{out}(h)| > \epsilon) \le 2 \exp(-2N\epsilon^2) $$

这意味着对于单个模型，只要样本量 $N$ 足够大，$E_{in}$ 就是 $E_{out}$ 的良好估计。

但是，机器学习是 **从假设空间 $\mathcal{H}$ 中挑选** 一个最好的 $h$。如果 $\mathcal{H}$ 包含无数个模型，我们如何保证我们选出来的那个 $g$ 不是恰好在训练集上“撞大运”表现好（Bad Sample），而在测试集上表现差的呢？

#### 2. 增长函数与打散 (Growth Function & Shattering)

为了解决无限个假设的问题，Vapnik 和 Chervonenkis 提出了一个天才的想法：**虽然参数是连续无穷的，但模型对 $N$ 个数据点的分类结果（Labeling）是有限的。**

对于二分类问题，N 个数据点最多有 $2^N$ 种标签组合。
定义 **增长函数 (Growth Function)** $m_{\mathcal{H}}(N)$：假设空间 $\mathcal{H}$ 在 $N$ 个数据点上能产生的 **最大不同二分（分类组合）数量**。

*   **打散 (Shattering)**：如果 $\mathcal{H}$ 能对 $N$ 个点的 **所有** $2^N$ 种可能性都进行分类，我们称 $\mathcal{H}$ 能 **打散** 这 $N$ 个点。此时 $m_{\mathcal{H}}(N) = 2^N$。

#### 3. VC 维 (VC Dimension) 的定义

**VC 维 ($d_{VC}$)** 是衡量假设空间 $\mathcal{H}$ 容量（复杂度）的核心指标。

> **定义**：$d_{VC}$ 是满足“增长函数 $m_{\mathcal{H}}(N) = 2^N$”的 **最大** $N$。
> 换句话说，它是模型能够完全打散的 **最大样本数量**。

*   如果 $N \le d_{VC}$，模型有可能打散这 $N$ 个点。
*   如果 $N > d_{VC}$，模型 **一定** 无法打散这 $N$ 个点（即总存在某种标签组合，模型学不会）。

**经典案例：仿射线性分类器的 VC 维**

这里必须先排除一个常见但错误的上界证明。画出正方形四角并赋予 XOR 标签，只能说明**这一组四点在这一种标记下**不可线性分；VC 维的上界要求证明任意 $d+2$ 点都存在某种无法实现的标记。下面用 Radon 分割完成全部量词。

令 $\mathcal H_d$ 是 $\mathbb R^d$ 上带偏置的仿射半空间类：

$$
h_{\mathbf w,b}(\mathbf x)
=
\begin{cases}
+1, & \mathbf w^\mathsf T\mathbf x+b\ge 0,\\
-1, & \mathbf w^\mathsf T\mathbf x+b<0.
\end{cases}
$$

**命题**：$\operatorname{VCdim}(\mathcal H_d)=d+1$。

**证明（下界）**：取 $d+1$ 个仿射独立点 $\mathbf x_1,\ldots,\mathbf x_{d+1}$。仿射独立等价于增广向量

$$
\widetilde{\mathbf x}_i=(\mathbf x_i,1)\in\mathbb R^{d+1}
$$

线性独立，因此它们构成 $\mathbb R^{d+1}$ 的一组基。对任意标签 $y_i\in\{-1,+1\}$，线性方程组

$$
\mathbf w^\mathsf T\mathbf x_i+b=y_i,
\qquad i=1,\ldots,d+1,
$$

都有唯一解 $(\mathbf w,b)$。因为右侧严格等于 $\pm1$，所得分类器在每个点上实现指定标签。于是这 $d+1$ 个点被打散，故 $\operatorname{VCdim}(\mathcal H_d)\ge d+1$。

**Radon 分割（书内证明）**：任取 $d+2$ 个点 $\mathbf x_1,\ldots,\mathbf x_{d+2}\in\mathbb R^d$。它们的 $d+2$ 个增广向量位于 $d+1$ 维空间，必线性相关，所以存在不全为零的系数 $\lambda_i$ 使

$$
\sum_{i=1}^{d+2}\lambda_i\mathbf x_i=0,
\qquad
\sum_{i=1}^{d+2}\lambda_i=0.
$$

令 $I=\{i:\lambda_i>0\}$、$J=\{j:\lambda_j<0\}$。两者都非空，并且

$$
A:=\sum_{i\in I}\lambda_i
=-\sum_{j\in J}\lambda_j>0.
$$

归一化后得到

$$
\sum_{i\in I}\frac{\lambda_i}{A}\mathbf x_i
=
\sum_{j\in J}\frac{-\lambda_j}{A}\mathbf x_j.
$$

等式两侧都是凸组合，故 $\operatorname{conv}\{\mathbf x_i:i\in I\}$ 与 $\operatorname{conv}\{\mathbf x_j:j\in J\}$ 相交。系数为零的点可任意归入一侧，不影响这个交点。

**证明（上界）**：对任意 $d+2$ 点取上述 Radon 分割，把 $I$ 一侧标为 $+1$，把 $J$ 一侧标为 $-1$，其余零系数点任意标记。假设某个仿射分类器实现了这些标签，并令两凸包的公共点为 $\mathbf z$。由 $I$ 侧的凸组合和仿射性，

$$
\mathbf w^\mathsf T\mathbf z+b
=
\sum_{i\in I}\frac{\lambda_i}{A}
(\mathbf w^\mathsf T\mathbf x_i+b)
\ge 0.
$$

由 $J$ 侧的凸组合，每一项都严格小于零，所以

$$
\mathbf w^\mathsf T\mathbf z+b
=
\sum_{j\in J}\frac{-\lambda_j}{A}
(\mathbf w^\mathsf T\mathbf x_j+b)
<0,
$$

矛盾。因此任意 $d+2$ 点都不能被打散，$\operatorname{VCdim}(\mathcal H_d)\le d+1$。结合下界即得结论。$\square$

在 $d=2$ 时，这个证明覆盖所有四点构型。若四点处于凸位置，Radon 分割由相交的两条对角线给出；同一条对角线的两个端点同色、两条对角线异色，正是常见的 XOR 图。若一点落在另外三点的三角形内，则把内部点标成一类、三个顶点标成另一类，同样无法由直线分开。共线、重合等退化情形也已包含在增广向量的线性相关论证中。因此，下面的 XOR 图是上界证明的一个可视化实例，而不是证明的全部。

<img src="images/vc_dimension.png" width="90%" />

#### 4. 关键引理：Sauer's Lemma

我们已经知道，当 $N \le d_{VC}$ 时，$m_{\mathcal{H}}(N) = 2^N$。那么当 $N > d_{VC}$ 时，增长函数会发生什么变化呢？

**Sauer's Lemma** 给出了增长函数的上界：
如果 $d_{VC}$ 有限，则对于任意 $N$：
$$ m_{\mathcal{H}}(N) \le \sum_{i=0}^{d_{VC}} \binom{N}{i} $$

*   **多项式上界**：当 $N > d_{VC}$ 时，我们可以利用不等式 $\binom{N}{i} \le N^i$，得到一个更宽松但直观的界：
    $$ m_{\mathcal{H}}(N) \le (N+1)^{d_{VC}} $$

这意味着：
1.  **Break Point**：一旦 $N$ 超过了 $d_{VC}$，增长函数 $m_{\mathcal{H}}(N)$ 就从 **指数级增长** ($2^N$) 突然“折断”为 **多项式级增长** ($N^{d_{VC}}$)。
2.  **意义**：这一性质至关重要。因为在霍夫丁不等式中，右边是 $M \cdot \exp(-N)$。如果 $M$ 是指数级增长的 ($2^N$)，它会抵消掉 $\exp(-N)$ 的衰减，导致误差界无法收敛。但如果是多项式级增长，$\exp(-N)$ 最终会战胜 $N^{d_{VC}}$，保证概率收敛到 0。

#### 5. 泛化误差界 (Generalization Bound) 的完整推导流程

有了 Sauer's Lemma，我们可以完成最后一步证明。

**Step 1: 幽灵样本技巧 (Ghost Sample Trick)**
为了处理无限的 $E_{out}$，我们引入第二组大小为 $N$ 的“幽灵数据集” $\mathcal{D}'$。我们证明：如果 $E_{in}$ 和 $E_{out}$ 差别很大，那么 $E_{in}$ 和 $E'_{in}$ (在幽灵数据上的误差) 差别很大的概率也是有界的。
$$ P(\sup |E_{in} - E_{out}| > \epsilon) \le 2 P(\sup |E_{in} - E'_{in}| > \epsilon/2) $$

**Step 2: 有限的二分法 (Effective Hypotheses)**
现在我们只看 $2N$ 个数据点（$\mathcal{D} + \mathcal{D}'$）。在这 $2N$ 个点上，模型最多只能产生 $m_{\mathcal{H}}(2N)$ 种不同的分类结果。
我们将“无限假设空间”的问题转化为了“有限二分空间”的问题。

**Step 3: 联合界 (Union Bound) 与 Hoeffding**
应用联合界和无放回抽样的 Hoeffding 不等式：
$$ P(\dots) \le 2 \cdot m_{\mathcal{H}}(2N) \cdot 2 \exp\left( -2 \left(\frac{\epsilon}{2}\right)^2 N \right) $$
整理得：
$$ P(\sup |E_{in} - E_{out}| > \epsilon) \le 4 (2N)^{d_{VC}} \exp\left( -\frac{1}{8} N \epsilon^2 \right) $$

**Step 4: 求解 $\epsilon$**
令上述概率的上界为 $\delta$：
$$ \delta = 4 (2N)^{d_{VC}} \exp\left( -\frac{1}{8} N \epsilon^2 \right) $$
反解出 $\epsilon$，即得到我们熟悉的 **VC 泛化界**：

$$ E_{out}(h) \le E_{in}(h) + \underbrace{\sqrt{\frac{8}{N} \left( d_{VC} \ln (2N) + \ln \frac{4}{\delta} \right)}}_{\text{Complexity Penalty } \Omega} $$
*(注：为简化展示，此处使用了 $m_{\mathcal{H}}(N) \approx N^{d_{VC}}$ 的近似形式，严谨形式略有差异但不影响结论)*

**适用域内的核心结论**：
对二分类假设类，在分布无关 PAC 学习框架及适当可测性条件下，有限 VC 维与该类的 PAC 可学习性相对应；相应的 agnostic/一致收敛版本也以有限 VC 维刻画容量。这不是关于回归、结构化预测、分布依赖学习或“所有机器学习”的充要条件。

有限 $d_{VC}$ 使统一泛化罚项随 $N$ 增大而趋于 0，从而以高概率控制所有 $h\in\mathcal H$ 的经验风险与总体风险差距。要得到低总体风险，还需要假设类中存在足够好的预测器并由学习算法找到低经验风险解；泛化界本身不保证近似误差或优化误差很小。
