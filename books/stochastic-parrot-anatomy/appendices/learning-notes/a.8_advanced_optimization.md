# 附录 A.8 自适应优化器与 AdamW

本附录固定 Momentum、AdaGrad、RMSProp、Adam 和 AdamW 的一组常用更新约定，并解释哪些等价关系成立。算法名称不能替代具体公式：不同库对动量缩放、$\epsilon$ 位置、weight decay 与参数分组可能采用不同约定。

以下乘法、除法、平方与平方根均按坐标进行。令
$g_t=\nabla F_t(\theta_{t-1})$ 表示第 $t$ 步 stochastic gradient。
除非另行说明，一阶、二阶累积量均从零初始化，$\epsilon>0$，各动量系数在训练期间固定。

## A.8.1 Momentum 的两种尺度约定

EMA 形式的 momentum 写作

$$
m_t=\beta m_{t-1}+(1-\beta)g_t,
$$

$$
\theta_t=\theta_{t-1}-\eta_tm_t,
\qquad 0\le\beta<1.
\tag{A.8.1}
$$

另一种常见实现省略 $(1-\beta)$：

$$
v_t=\beta v_{t-1}+g_t,
\qquad
\theta_t=\theta_{t-1}-\widetilde\eta_tv_t.
\tag{A.8.2}
$$

当 $m_0=v_0=0$ 且超参数固定时，$m_t=(1-\beta)v_t$；要得到相同参数更新需令
$\widetilde\eta_t=(1-\beta)\eta_t$。因此不同实现中的“学习率相同”不表示动力学相同。

展开 (A.8.1)：

$$
m_t=(1-\beta)\sum_{j=1}^t\beta^{t-j}g_j
\quad(m_0=0).
\tag{A.8.3}
$$

权重和为 $1-\beta^t$，时间常数约为 $-1/\log\beta\approx1/(1-\beta)$。momentum 可以平滑部分高频随机变化，但不保证逃离任意鞍点，也不存在对所有目标优于 SGD 的次序。

## A.8.2 AdaGrad 与 RMSProp

AdaGrad 的一组对角形式为

$$
v_t=v_{t-1}+g_t^2,
$$

$$
\theta_t
=\theta_{t-1}
-\eta_t\frac{g_t}{\sqrt{v_t}+\epsilon}.
\tag{A.8.4}
$$

$v_t$ 单调不减。若某坐标的平方梯度和发散，该坐标的预条件系数趋于零；若平方梯度和有限，不能作此结论。稀疏坐标往往累积得较慢，但这不表示稀有特征必然更有信息。

RMSProp 用指数移动平均替代永久累积，其中 $0\le\rho<1$：

$$
v_t=\rho v_{t-1}+(1-\rho)g_t^2,
$$

$$
\theta_t
=\theta_{t-1}
-\eta_t\frac{g_t}{\sqrt{v_t}+\epsilon}.
\tag{A.8.5}
$$

有些实现使用 $\sqrt{v_t+\epsilon}$，或另加一阶 momentum；这些公式在有限精度和 $v_t$ 很小时并不等价。$v_t$ 是平方梯度的近期尺度，不是 Hessian，也不区分信号与噪声。

## A.8.3 Adam 与偏差修正

Adam 同时维护一阶与二阶原始矩的 EMA，其中 $0\le\beta_1,\beta_2<1$：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,
\tag{A.8.6}
$$

$$
\widehat m_t=\frac{m_t}{1-\beta_1^t},
\qquad
\widehat v_t=\frac{v_t}{1-\beta_2^t},
\tag{A.8.7}
$$

$$
\theta_t
=\theta_{t-1}
-\eta_t\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon}.
\tag{A.8.8}
$$

若 $m_0=0$，则

$$
m_t=(1-\beta_1)
\sum_{j=1}^t\beta_1^{t-j}g_j,
$$

而这些系数之和是 $1-\beta_1^t$，所以 (A.8.7) 把它们归一化为 $1$。对恒定均值的平稳梯度过程，这消除零初始化造成的均值缩小；对非平稳序列，它只是归一化权重和，不保证 $\widehat m_t$ 对“当前真实梯度”无偏。$\widehat v_t$ 同理。

### 收敛反例不能省略

原始 Adam 公式不是无条件收敛定理。Reddi、Kale 与 Kumar 构造了一维凸在线损失序列，使 Adam 的自适应有效步长行为导致非消失 regret；因此即使各步损失凸，也不能只凭 (A.8.6)--(A.8.8) 宣称收敛。

AMSGrad 的修正之一是逐坐标维护

$$
\overline v_t=\max(\overline v_{t-1},v_t)
$$

并在分母使用 $\sqrt{\overline v_t}$，从而避免该预条件尺度按坐标减小。其收敛保证仍依赖有界域、梯度、步长等论文假设；它不是对任意非凸训练的全局最优证明。

## A.8.4 AdamW 与 L2 penalty 的区别

若先把 L2 penalty 梯度并入 Adam：

$$
g_t=\nabla F_t(\theta_{t-1})+\lambda\theta_{t-1},
\tag{A.8.9}
$$

则 $\lambda\theta_{t-1}$ 同时进入 $m_t$ 和 $v_t$，并接受历史依赖的逐坐标预条件。这仍是在对 penalized objective 取随机梯度，但不等于 plain SGD 的固定比例 weight decay。

AdamW 令矩估计只使用原损失梯度，取 $\lambda\ge0$，
$g_t=\nabla F_t(\theta_{t-1})$，再执行

$$
\boxed{
\theta_t
=(1-\eta_t\lambda)\theta_{t-1}
-\eta_t
\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon}.}
\tag{A.8.10}
$$

因此 decay 不进入矩状态，也不被 $1/\sqrt{\widehat v_t}$ 按坐标缩放。当 $0\le\eta_t\lambda\le1$ 时，第一项才是通常意义下的无符号翻转收缩。若学习率调度变化，(A.8.10) 中每步实际收缩因子也随 $\eta_t$ 变化。某些 API 允许把 decay 写成不同顺序或使用独立调度，必须查看实际更新式。

对 bias、LayerNorm/RMSNorm scale、embedding 或其他参数是否关闭 decay 是参数分组策略，不是 AdamW 定义自动推出的结论。AdamW 也不等于“完整 Bayesian 正则化”；MAP 边界见[附录 A.4](a.4_regularization.md)。

## A.8.5 使用这些公式时应报告什么

至少应同时给出：

- 优化器的精确变体与库版本；
- $\eta_t$ 调度、warmup 和总更新步数；
- $\beta_1,\beta_2,\epsilon$ 及 $\epsilon$ 的位置；
- gradient clipping、loss scaling 与梯度累积；
- weight decay 数值、是否与学习率相乘、参数分组；
- batch/token 归一化方式。

这些量共同定义离散优化过程。只写“使用 AdamW”不足以复现实验，也不能推出收敛或泛化排序。

## A.8.6 来源

- Duchi, Hazan & Singer, [*Adaptive Subgradient Methods for Online Learning and Stochastic Optimization*](https://jmlr.org/papers/v12/duchi11a.html), 2011。
- Hinton, Srivastava & Swersky, [*Neural Networks for Machine Learning, Lecture 6a*](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf), 2012。RMSProp 的课程讲义来源。
- Kingma & Ba, [*Adam: A Method for Stochastic Optimization*](https://arxiv.org/abs/1412.6980), 2015。
- Reddi, Kale & Kumar, [*On the Convergence of Adam and Beyond*](https://openreview.net/forum?id=ryQu7f-RZ), 2018。
- Loshchilov & Hutter, [*Decoupled Weight Decay Regularization*](https://arxiv.org/abs/1711.05101), 2019。
