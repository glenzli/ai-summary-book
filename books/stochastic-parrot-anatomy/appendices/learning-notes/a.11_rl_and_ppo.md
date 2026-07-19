# 附录 A.11 策略梯度、PPO 与 DPO

卷一第五章说明后训练流水线。本附录只推导三个数学接口：策略梯度估计量、PPO clipped surrogate，以及 KL 正则化最优策略到 DPO 损失的变换。三者依赖的策略与数据分布不同，不能仅因公式都含 log probability 就混为一个算法。

## A.11.1 自回归生成作为有限时域决策过程

对 prompt $x$，令状态

$$
s_t=(x,y_{<t}),
$$

动作 $a_t=y_t$ 是下一个 token，策略为

$$
\pi_\theta(a_t\mid s_t).
$$

状态转移在给定 token 后通常是确定的前缀追加，直到 EOS 或长度上限。奖励可以只在终止时给出，也可以包含逐 token KL、格式或过程反馈。以下策略梯度推导假定用于更新时 reward 函数固定，不显式依赖 $\theta$；若 reward 与策略参数共享并参与求导，还会出现额外导数项。

轨迹概率可分解为

$$
p_\theta(\tau)
=p(x)\prod_{t=0}^{T-1}
\pi_\theta(a_t\mid s_t)
p(s_{t+1}\mid s_t,a_t).
\tag{A.11.1}
$$

只有策略因子依赖 $\theta$。

## A.11.2 Likelihood-ratio 策略梯度

令总回报为 $R(\tau)$，目标

$$
J(\theta)=\mathbb E_{\tau\sim p_\theta}[R(\tau)].
$$

在可交换微分与积分且轨迹支持不随参数突变等条件下，

$$
\begin{aligned}
\nabla_\theta J(\theta)
&=\mathbb E_{\tau\sim p_\theta}
[R(\tau)\nabla_\theta\log p_\theta(\tau)]\\
&=\mathbb E
\left[
R(\tau)\sum_t
\nabla_\theta\log\pi_\theta(a_t\mid s_t)
\right].
\end{aligned}
\tag{A.11.2}
$$

若逐步 reward 为 $r_t$、折扣因子为 $\gamma\in[0,1]$，并且

$$
R(\tau)=\sum_{k=0}^{T-1}\gamma^kr_k,
$$

定义从 $t$ 开始、重新以零次幂计数的 return

$$
G_t=\sum_{k=t}^{T-1}\gamma^{k-t}r_k.
$$

由因果性，动作 $a_t$ 之前的 reward 已由历史决定；它们与该步 score 项相乘后的条件期望为零，故精确的有限时域 likelihood-ratio 梯度为

$$
\nabla_\theta J(\theta)
=\mathbb E\left[
\sum_{t=0}^{T-1}\gamma^tG_t
\nabla_\theta\log\pi_\theta(a_t\mid s_t)
\right].
$$

当 $\gamma=1$ 时外部因子消失。某些实现改用折扣占用分布采样状态，或直接优化按时间步平均的 surrogate，因而不显式写 $\gamma^t$；那是目标与采样测度的约定，不是从上述轨迹目标中代数消去该因子。

对任意只依赖状态、在 actor 求导时视为 stop-gradient 的 baseline $b(s_t)$，

$$
\mathbb E_{a_t\sim\pi_\theta}
[b(s_t)\nabla_\theta\log\pi_\theta(a_t\mid s_t)]
=b(s_t)\nabla_\theta\sum_a\pi_\theta(a\mid s_t)=0.
$$

故把 $G_t$ 替换为 $G_t-b(s_t)$ 不改变上述估计量期望，并可能降低方差。常用选择是

$$
V^\pi(s_t)=\mathbb E[G_t\mid s_t],
\qquad
A^\pi(s_t,a_t)=Q^\pi(s_t,a_t)-V^\pi(s_t).
$$

但 value function 不是对所有参数化都严格最小化梯度估计方差的标量 baseline。若记 $z_t=\nabla_\theta\log\pi_\theta(a_t\mid s_t)$，条件二阶矩有限且分母非零，则最小化 $\mathbb E[\|z_t(G_t-b)\|^2\mid s_t]$ 的值是

$$
b^*(s_t)
=\frac{\mathbb E[G_t\|z_t\|^2\mid s_t]}
{\mathbb E[\|z_t\|^2\mid s_t]}.
$$

$V^\pi$ 是稳定且易估计的标准选择；只有在相应加权不改变条件均值时才与 $b^*$ 相同。

## A.11.3 Value critic 与 GAE

令 critic 为 $V_\phi(s)$，TD residual 为

$$
\delta_t
=r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t),
\tag{A.11.3}
$$

终止状态取 $V_\phi=0$。Generalized Advantage Estimation 的截断形式为

$$
\widehat A_t^{\mathrm{GAE}(\gamma,\lambda)}
=\sum_{l=0}^{T-t-1}(\gamma\lambda)^l\delta_{t+l},
\qquad \lambda\in[0,1].
\tag{A.11.4}
$$

$\lambda$ 较小通常更依赖 critic bootstrap、方差较低而偏差可能较大；$\lambda$ 接近 $1$ 更接近 Monte Carlo return。这个说法依赖 critic 误差和轨迹长度，不是单调性能定理。对只有终止 reward 的语言模型，早期 token 的 advantage 主要通过回报传播和 critic 基线形成。

## A.11.4 PPO clipped surrogate

用冻结的旧策略 $\pi_{\mathrm{old}}$ 采样轨迹，取 $0<\epsilon<1$，并在这些样本上定义

$$
r_t(\theta)
=\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\mathrm{old}}(a_t\mid s_t)}.
\tag{A.11.5}
$$

分母必须在已采样动作上为正。PPO 的 clipped surrogate 是

$$
L^{\mathrm{clip}}(\theta)
=\mathbb E_t\left[
\min\left(
r_t(\theta)\widehat A_t,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)
\widehat A_t
\right)
\right].
\tag{A.11.6}
$$

符号分情况解释：

- 若 $\widehat A_t>0$，增大动作概率有利；当 $r_t>1+\epsilon$ 时，第二项把进一步增益截平。
- 若 $\widehat A_t<0$，减小动作概率有利；当 $r_t<1-\epsilon$ 时，第二项把进一步增益截平。

在另外两个方向上，`min` 不会截平纠错梯度。因此 (A.11.6) **不是**硬约束
$r_t\in[1-\epsilon,1+\epsilon]$，也不保证每次更新后的 KL 小于某阈值。

一种常见联合最大化目标为

$$
L^{\mathrm{clip}}
-c_v\mathbb E_t[(V_\phi(s_t)-\widehat V_t)^2]
+c_H\mathbb E_t[H(\pi_\theta(\cdot\mid s_t))],
\tag{A.11.7}
$$

并可另加到参考策略的 KL 控制。实际算法还由 rollout batch、每批 epoch 数、mini-batch 顺序、advantage 标准化、value clipping 和 early stopping 共同定义。PPO 原论文给出实用 surrogate，而不是 TRPO 式严格单调改进保证。

## A.11.5 参考策略与旧策略不是同一个对象

$\pi_{\mathrm{old}}$ 是产生当前 rollout 的行为策略，用在 (A.11.5) 的重要性比率中；每轮通常更新。

$\pi_{\mathrm{ref}}$ 是 RLHF 目标中的正则化锚点，常在一段训练中冻结。分布级目标写作

$$
J(\pi)
=\mathbb E_{x\sim\mathcal D}
\left[
\mathbb E_{y\sim\pi(\cdot\mid x)}r(x,y)
-\beta D_{\mathrm{KL}}
(\pi(\cdot\mid x)\|\pi_{\mathrm{ref}}(\cdot\mid x))
\right].
\tag{A.11.8}
$$

二者有时初始化为同一 checkpoint，但承担的数学角色不同。

## A.11.6 KL 正则化最优策略

固定 $x$，记 $q_y=\pi_{\mathrm{ref}}(y\mid x)$。假设输出空间有限或求和绝对收敛，reward 有限，并只在 $q_y>0$ 的支持上优化。考虑

$$
\max_{\pi\in\Delta}
\sum_y\pi_yr_y
-\beta\sum_y\pi_y\log\frac{\pi_y}{q_y},
\qquad \beta>0.
\tag{A.11.9}
$$

目标对支持内部的 $\pi$ 严格凹。加入约束 $\sum_y\pi_y=1$ 的乘子 $\lambda$，驻点满足

$$
r_y-\beta\left(\log\frac{\pi_y}{q_y}+1\right)+\lambda=0.
$$

整理并归一化得到唯一最优解

$$
\boxed{
\pi^*(y\mid x)
=\frac{1}{Z(x)}
\pi_{\mathrm{ref}}(y\mid x)
\exp\left(\frac{r(x,y)}{\beta}\right)}
\tag{A.11.10}
$$

$$
Z(x)=\sum_y\pi_{\mathrm{ref}}(y\mid x)e^{r(x,y)/\beta}.
$$

若参考策略对某输出概率为零，任何给它正概率的策略都有无限 forward KL；(A.11.10) 不会创造参考支持之外的概率质量。

## A.11.7 从 Bradley--Terry 到 DPO

假设同一 prompt 下的偏好概率满足 Bradley--Terry 模型

$$
P(y^+\succ y^-\mid x)
=\sigma(r(x,y^+)-r(x,y^-)).
\tag{A.11.11}
$$

由 (A.11.10) 反解

$$
r(x,y)
=\beta\log\frac{\pi^*(y\mid x)}
{\pi_{\mathrm{ref}}(y\mid x)}
+\beta\log Z(x).
$$

同一 $x$ 的 $\log Z(x)$ 在 reward 差中抵消。用待训练策略 $\pi_\theta$ 参数化 $\pi^*$，定义

$$
\Delta_\theta
=\beta\left[
\log\frac{\pi_\theta(y^+\mid x)}
{\pi_{\mathrm{ref}}(y^+\mid x)}
-
\log\frac{\pi_\theta(y^-\mid x)}
{\pi_{\mathrm{ref}}(y^-\mid x)}
\right].
\tag{A.11.12}
$$

假设观测偏好对上的 $\pi_\theta$ 与 $\pi_{\mathrm{ref}}$ 序列概率均为正。对偏好对的负对数似然就是

$$
\boxed{
L_{\mathrm{DPO}}(\theta)
=-\mathbb E_{(x,y^+,y^-)}
[\log\sigma(\Delta_\theta)].}
\tag{A.11.13}
$$

自回归序列概率必须包含全部被建模 token：

$$
\log\pi_\theta(y\mid x)
=\sum_{t=1}^{|y|}
\log\pi_\theta(y_t\mid x,y_{<t}).
\tag{A.11.14}
$$

因此 EOS、截断和长度进入标准 DPO 目标。擅自改为 token 平均会定义不同的隐式 reward。DPO 的推导依赖 (A.11.8) 的 forward-KL 目标、Bradley--Terry 偏好模型和参考支持；它不等价于任意 reward、任意偏好噪声或任意 PPO-RLHF 实现。离线偏好数据若不能覆盖训练后策略常见输出，分类损失也不会自动修复这一分布缺口。

## A.11.8 来源

- Williams, [*Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning*](https://doi.org/10.1007/BF00992696), 1992。
- Schulman et al., [*High-Dimensional Continuous Control Using Generalized Advantage Estimation*](https://arxiv.org/abs/1506.02438), 2016。
- Schulman et al., [*Proximal Policy Optimization Algorithms*](https://arxiv.org/abs/1707.06347), 2017。
- Ouyang et al., [*Training Language Models to Follow Instructions with Human Feedback*](https://arxiv.org/abs/2203.02155), 2022。
- Rafailov et al., [*Direct Preference Optimization: Your Language Model Is Secretly a Reward Model*](https://arxiv.org/abs/2305.18290), 2023。
