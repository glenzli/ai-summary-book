# 附录 A.5 通用近似定理 (Universal Approximation Theorem)
## Appendix A.5 Proof Outline and a One-Dimensional ReLU Construction

本附录给出通用近似定理（Universal Approximation Theorem, UAT）的证明概要，并用一维 ReLU 网络说明分段线性逼近。定理讨论特定函数空间中的稠密性，不保证可训练性、样本效率或泛化。

### A.5.1 定理陈述 (Statement of the Theorem)

需要区分几类常被合并引用的结果：

*   **Cybenko (1989)**：若 $\sigma$ 是连续 sigmoidal 函数，即 $\sigma(t)\to 0$（$t\to-\infty$）且 $\sigma(t)\to 1$（$t\to+\infty$），则有限个 ridge functions $\sigma(\mathbf w^T\mathbf x+b)$ 的线性组合在 $C([0,1]^n)$ 中按上确界范数稠密。更一般地，Cybenko 的证明使用 **discriminatory** 条件。
*   **Hornik (1991) 及后续推广**：普适性不依赖 sigmoid 的单调形状。对具有偏置的单隐层网络，在相应正则条件下，激活函数“不是（几乎处处）多项式”给出稠密性的关键刻画；Leshno 等人 (1993) 给出了这一非多项式条件。ReLU 是连续、非多项式激活，因此属于后续结论，而不是 Cybenko 原始 sigmoidal 定理的例子。

以连续函数的一致逼近为例：对紧致集合 $K\subset\mathbb R^n$、$f\in C(K)$ 和任意 $\epsilon>0$，在上述相应激活条件下，存在有限宽单隐层网络

$$
g(\mathbf{x}) = \sum_{i=1}^{N} v_i\,\sigma(\mathbf{w}_i^T \mathbf{x} + b_i)
$$

使得

$$ \sup_{\mathbf{x} \in K} |f(\mathbf{x}) - g(\mathbf{x})| < \epsilon $$

这里的“任意精度”只针对指定紧致域、函数类和范数，也以允许网络宽度随 $f$ 与 $\epsilon$ 变化为前提。

---

### A.5.2 基于 Hahn-Banach 定理的证明思路 (Proof Outline)

这里取 $K=[0,1]^n$ 且 $\sigma$ 为连续 sigmoidal 函数，概述 Cybenko 的泛函分析证明思路；ReLU 的普适性不由这一组 sigmoidal 前提直接推出。证明使用 **Hahn-Banach 定理** 和 **Riesz 表示定理**。

#### 1. 定义函数空间
记 $C(K)$ 为定义在紧致集合 $K$ 上的所有连续函数的集合，赋予上确界范数（Sup-Norm）：
$$ \|f\| = \sup_{\mathbf{x} \in K} |f(\mathbf{x})| $$

我们定义神经网络能够表示的函数子空间为 $S$：
$$ S = \text{span} \{ \sigma(\mathbf{w}^T \mathbf{x} + b) \mid \mathbf{w} \in \mathbb{R}^n, b \in \mathbb{R} \} $$
即所有可能的单层神经网络函数的线性组合。

我们需要证明：$S$ 在 $C(K)$ 中是**稠密 (Dense)** 的。也就是说，对于 $C(K)$ 中的任意 $f$，都在 $S$ 的闭包 $\bar{S}$ 中。

#### 2. 反证法 (Proof by Contradiction)
假设 $S$ 不是稠密的，即 $\bar{S} \neq C(K)$。
根据泛函分析中的推论，如果 $\bar{S} \neq C(K)$，那么一定存在一个非零的**有界线性泛函 (Bounded Linear Functional)** $L$，使得 $L$ 在子空间 $S$ 上的作用全为 0，但 $L$ 本身不为 0（即存在某个 $f$ 使 $L(f) \neq 0$）。
这利用了 Hahn-Banach 定理的一个推论。

根据 **Riesz 表示定理 (Riesz Representation Theorem)**，在紧致集 $K$ 上的连续函数空间 $C(K)$ 上的任意有界线性泛函 $L$，都可以表示为一个正则波莱尔测度 (Regular Borel Measure) $\mu$ 的积分：
$$ L(h) = \int_K h(\mathbf{x}) d\mu(\mathbf{x}) $$

因此，我们的假设等价于：存在一个非零测度 $\mu$，使得对于所有 $h \in S$，都有：
$$ \int_K \sigma(\mathbf{w}^T \mathbf{x} + b) d\mu(\mathbf{x}) = 0, \quad \forall \mathbf{w}, b $$

#### 3. 利用 Sigmoid 的判别性质 (Discriminatory Property)
Cybenko 证明了对于 Sigmoid 类的 Sigmoidal 函数（$x \to -\infty$ 时 $\sigma \to 0$，$x \to +\infty$ 时 $\sigma \to 1$），它是**判别性 (Discriminatory)** 的。

这意味着，如果对于所有 $\mathbf{w}, b$ 都有：
$$ \int_K \sigma(\mathbf{w}^T \mathbf{x} + b) d\mu(\mathbf{x}) = 0 $$
那么必然蕴含 $\mu = 0$。

证明利用缩放后的 sigmoidal 函数逼近由超平面定义的阶跃，并据此表明该有限符号测度在相应半空间族上为零，最终推出 $\mu=0$。这里不是声称单个 sigmoid 能逼近任意示性函数，而是说明该函数族足以分离非零测度。

#### 4. 矛盾
如果你能证明 $\mu = 0$，这就与“存在非零泛函 $L$”矛盾了。
因此，假设不成立，$S$ 在 $C(K)$ 中是稠密的。证毕。

---

### A.5.3 一维 ReLU 的分段线性构造 (One-Dimensional Construction)

这里给出一个诚实但限于一维的构造。记 $(u)_+=\operatorname{ReLU}(u)$。

1.  **四个 ReLU 构造梯形帽函数**

    对 $0<a<b$ 和平移量 $t$，定义

    $$
    h_{t,a,b}(x)=\frac{1}{a}\left[
    (x-t)_+-(x-t-a)_+-(x-t-b)_+ +(x-t-a-b)_+
    \right].
    $$

    它由**四个** ReLU 组成：在 $[t,t+a]$ 线性上升，在 $[t+a,t+b]$ 保持为 1，在 $[t+b,t+a+b]$ 线性下降，其余位置为 0。若写成 $(x)_+-(x-a)_+-(x-b)_++(x-c)_+$，必须满足 $0<a<b$ 且 $c=a+b$ 才有上述紧支撑梯形。连续 ReLU 网络不能精确表示有跳跃的不连续矩形指示函数。

2.  **用分段线性插值逼近连续函数**

    对区间 $[A,B]$ 上的连续函数 $f$，一致连续性保证：当分割网格足够细时，连接采样点 $(x_j,f(x_j))$ 的分段线性插值 $p$ 满足 $\|f-p\|_\infty<\epsilon$。任意连续分段线性函数都可写成

    $$
    p(x)=\alpha+\beta x+\sum_{j=1}^{m}\gamma_j(x-x_j)_+,
    $$

    其中 $\gamma_j$ 是相邻线段斜率的变化量。因此，单隐层 ReLU 网络可以在一维紧区间上一致逼近连续函数。若网络定义不单列仿射直连项，可用 $x=(x)_+-(-x)_+$ 表示线性项，并用输出偏置表示常数项。

3.  **多维边界**

    直接写 $\sum_i g_i(x_i)$ 只能得到坐标可加函数，不能构造一般的局部矩形或表示变量交互。多维普适性依赖 $\sigma(\mathbf w^T\mathbf x+b)$ 这类 ridge functions 的正式密度定理，不能由上述一维插值论证直接推出。

<img src="../chapter_02/images/universal_approximation_bump.png" width="80%" />

---

### A.5.4 局限性 (Limitations)

通用近似定理虽然保证了**存在性**，但没有告诉我们：
1.  **如何找到**这个网络（优化问题）。
2.  需要**多少**神经元（效率问题）。对没有额外结构的一般高维函数类，逼近率可能受到维数灾难；对具有组合、局部或对称结构的函数类，深层架构有时能更高效。UAT 本身不证明“深度一定更省参数”。
