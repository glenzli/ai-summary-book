# 附录 A.4 正则化、MAP 与 Dropout 的边界

本附录只保留卷一第二章需要的三条接口：范数约束与惩罚目标何时相关，L1/L2 与 MAP 在什么建模前提下对应，以及 dropout 的缩放与梯度怎样计算。AdamW 的算法细节放在[附录 A.8](a.8_advanced_optimization.md)。

## A.4.1 约束问题与惩罚问题

考虑

$$
\min_\theta F(\theta)
\quad\text{s.t.}\quad
\Omega(\theta)\le C
\tag{A.4.1}
$$

与

$$
\min_\theta F(\theta)+\lambda\Omega(\theta),
\qquad \lambda\ge0.
\tag{A.4.2}
$$

二者不是对任意 $C,\lambda$ 自动一一对应。

**命题 A.4.1（惩罚解给出一个约束解）** 若 $\lambda>0$，且 $\theta_\lambda$ 是 (A.4.2) 的全局极小点，则它也是取
$C=\Omega(\theta_\lambda)$ 时 (A.4.1) 的全局极小点。

**证明** 对任意满足 $\Omega(\theta)\le C$ 的 $\theta$，

$$
F(\theta_\lambda)+\lambda C
\le F(\theta)+\lambda\Omega(\theta)
\le F(\theta)+\lambda C.
$$

消去 $\lambda C$ 得 $F(\theta_\lambda)\le F(\theta)$。$\square$

反方向需要更多条件。若 $F,\Omega$ 是正常（proper）、下半连续凸函数，存在属于其有效域相对内部的严格可行点 $\bar\theta$，满足 $\Omega(\bar\theta)<C$（Slater 条件），且约束最优解存在，则强对偶和 KKT 条件给出某个乘子 $\lambda^*\ge0$，满足

$$
0\in\partial F(\theta^*)+\lambda^*\partial\Omega(\theta^*),
\qquad
\lambda^*(\Omega(\theta^*)-C)=0.
$$

此时 $\theta^*$ 也是相应惩罚问题的解。映射仍可能不唯一；约束不活跃时可有 $\lambda^*=0$。非凸网络、局部极小点和近似优化下，不能只凭拉格朗日直觉宣称两个训练过程等价。

## A.4.2 MAP 与范数惩罚

设数据为 $\mathcal D$，参数相对于固定基准测度有先验密度 $p(\theta)$，似然为 $p(\mathcal D\mid\theta)$。若后验可正规化，则

$$
p(\theta\mid\mathcal D)
\propto p(\mathcal D\mid\theta)p(\theta).
$$

MAP 点估计满足

$$
\theta_{\mathrm{MAP}}
\in\arg\min_\theta
\left[-\log p(\mathcal D\mid\theta)-\log p(\theta)\right].
\tag{A.4.3}
$$

因此“正则项等于负对数先验”只在训练损失确为相应负对数似然、系数尺度一致且参数化固定时成立。

### Gaussian 与 Laplace 例子

若 $\theta\sim\mathcal N(0,\tau^2I)$，则

$$
-\log p(\theta)
=\frac1{2\tau^2}\|\theta\|_2^2+\text{constant}.
$$

若各坐标独立且 $\theta_j\sim\operatorname{Laplace}(0,b)$，则

$$
-\log p(\theta)
=\frac1b\|\theta\|_1+\text{constant}.
$$

系数取决于损失采用总和还是均值。若
$-\log p(\mathcal D\mid\theta)=\sum_{i=1}^N\ell_i(\theta)$，高斯先验给出系数 $1/(2\tau^2)$；若优化平均损失
$N^{-1}\sum_i\ell_i$，同一个 MAP 问题中的 L2 系数变为 $1/(2N\tau^2)$。所以“L2 就是标准高斯先验”没有自动确定训练代码里的 weight decay 数值。

### MAP 不是完整 Bayesian 推断

Bayesian 预测分布需要积分

$$
p(y\mid x,\mathcal D)
=\int p(y\mid x,\theta)p(\theta\mid\mathcal D)\,d\theta,
$$

而 MAP 只保留一个后验众数。它不会给出后验不确定性的完整表示。此外，连续密度的众数在非线性重参数化下会受 Jacobian 影响；MAP 一般不具有参数化不变性。故不能把普通 L2 训练后的单个 checkpoint 直接称为“Bayesian 神经网络”。

L1-MAP 中可能出现的精确零，来自 $\|\theta\|_1$ 在零点的非光滑最优性条件。对光滑损失 $F$，proximal gradient 的一步为

$$
u_t=\theta_t-\eta\nabla F(\theta_t),
$$

$$
\theta_{t+1,j}
=\operatorname{sign}(u_{t,j})
\max(|u_{t,j}|-\eta\lambda,0).
\tag{A.4.4}
$$

普通次梯度更新不具备 (A.4.4) 的显式阈值操作，不能保证坐标一到零就永久保持为零。

## A.4.3 L2 penalty、weight decay 与 AdamW

对 plain SGD，若目标为

$$
J(\theta)=F(\theta)+\frac\lambda2\|\theta\|^2,
$$

则

$$
\theta_{t+1}
=(1-\eta_t\lambda)\theta_t
-\eta_t\nabla F(\theta_t).
\tag{A.4.5}
$$

因此在这一更新约定下，L2 penalty 与同步比例衰减代数相同。对 Adam，把 $\lambda\theta_t$ 并入梯度会使它进入一阶、二阶矩估计并接受逐坐标预条件；这不再等于 (A.4.5)。AdamW 则把参数衰减与损失梯度的矩估计解耦。它是一种优化算法定义，不自动对应某个固定先验下的 MAP。

## A.4.4 Dropout 的前向与反向

令丢弃率 $p\in[0,1)$、保留率 $q=1-p$，并令掩码坐标独立满足

$$
r_j\sim\operatorname{Bernoulli}(q).
$$

现代实现通常使用 inverted dropout：

$$
\widetilde h=\frac rq\odot h
\quad\text{（训练）},
\qquad
\widetilde h=h
\quad\text{（评估）}.
\tag{A.4.6}
$$

在给定 $h$ 且掩码独立时，

$$
\mathbb E_r[\widetilde h\mid h]=h,
\qquad
\operatorname{Var}_r(\widetilde h_j\mid h_j)
=\frac pqh_j^2.
\tag{A.4.7}
$$

若上游梯度为 $g=\nabla_{\widetilde h}L$，则同一次前向使用的掩码给出

$$
\nabla_hL=\frac rq\odot g.
\tag{A.4.8}
$$

掩码为零只切断该样本和该位置的这条激活路径；共享参数仍可从 batch 中其他路径得到梯度。

### 期望匹配不等于精确模型平均

(A.4.7) 只匹配被 dropout 激活的一阶条件期望。一般有

$$
\mathbb E_r[\phi(\widetilde h)]
\ne\phi(\mathbb E_r[\widetilde h])
$$

对非线性 $\phi$ 成立。例如 $q=1/2$、标量 $h\ne0$、$\phi(z)=z^2$ 时，$\widetilde h$ 以相同概率取 $0$ 或 $2h$，所以

$$
\mathbb E[\widetilde h^2]=2h^2
\ne h^2=(\mathbb E[\widetilde h])^2.
$$

因此评估时关闭 dropout 不是对所有随机子网络输出的精确平均，只是成本低且广泛使用的确定性近似。dropout 是否改善泛化是依赖数据、架构和训练配方的经验问题。

## A.4.5 来源

- Boyd & Vandenberghe, [*Convex Optimization*](https://web.stanford.edu/~boyd/cvxbook/), 2004。约束、对偶与 KKT 条件。
- Parikh & Boyd, [*Proximal Algorithms*](https://doi.org/10.1561/2400000003), 2014。软阈值与 proximal gradient。
- Srivastava et al., [*Dropout: A Simple Way to Prevent Neural Networks from Overfitting*](https://jmlr.org/papers/v15/srivastava14a.html), 2014。
- Loshchilov & Hutter, [*Decoupled Weight Decay Regularization*](https://arxiv.org/abs/1711.05101), 2019。
