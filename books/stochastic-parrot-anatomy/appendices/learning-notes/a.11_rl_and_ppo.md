# 附录 A.11 强化学习与 PPO 原理 (RL & PPO Principles)

## A.11.1 强化学习基础 (RL Basics)

在 RLHF 中，我们将大模型微调建模为一个 RL 问题：
*   **Agent**: LLM (Policy $\pi_\theta$).
*   **State/Environment**: Prompt、已生成前缀以及生成终止规则；奖励模型、验证器和系统约束在轨迹结束或中间步骤提供反馈。
*   **Action**: 生成下一个 Token。
*   **Reward**: 可以来自奖励模型、人类偏好代理、规则或可执行验证器，不限于单一 RM 分数。

我们的目标是最大化期望累积奖励：
$$ J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)] $$

## A.11.2 策略梯度 (Policy Gradient)

最直接的方法是使用策略梯度定理：
$$ \nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t} \nabla_\theta \log \pi_\theta(a_t | s_t) A_t \right] $$
其中 $A_t$ 是优势函数 (Advantage Function)，衡量当前动作比平均水平好多少。

朴素 Monte Carlo policy-gradient 估计通常方差较高，更新也可能迅速偏离采样策略：
1.  **步长难以确定**: 更新太小收敛慢，更新太大导致策略崩溃。
2.  **数据效率受限**: 朴素 on-policy 更新只能在策略尚未偏离采样策略太远时复用轨迹。

## A.11.3 PPO: 近端策略优化 (Proximal Policy Optimization)

PPO 的核心思想是：**限制策略更新的幅度，防止新策略偏离旧策略太远。**

### A.11.3.1 重要性采样 (Importance Sampling)

为了复用旧策略 $\pi_{old}$ 采样的数据来更新新策略 $\pi_\theta$，我们引入重要性采样比率 $r_t(\theta)$：
$$ r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{old}(a_t | s_t)} $$

当 $\theta = \theta_{old}$ 时，$r_t = 1$。

### A.11.3.2 截断目标函数 (Clipped Objective)

PPO 的 clipped surrogate 由两部分取最小值构成，对会把概率比率推得过远的有利更新截平：

$$ L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t) \right] $$

*   $\epsilon$: 超参数，通常为 0.1 或 0.2。
*   **第一项** $r_t(\theta) A_t$: 标准的 TRPO 代理目标。
*   **第二项** $\text{clip}(\dots) A_t$: 截断该样本对代理目标的贡献；它不会把实际策略比率硬性约束在 $[1-\epsilon, 1+\epsilon]$ 内。

**直观理解**：
1.  如果 $A_t > 0$（动作很好）：我们希望增加该动作的概率 ($r_t > 1$)。但为了稳定，如果 $r_t > 1+\epsilon$，就不再给予额外的奖励梯度。
2.  如果 $A_t < 0$（动作很差）：我们希望减少该动作的概率 ($r_t < 1$)。但为了稳定，如果 $r_t < 1-\epsilon$，就不再给予额外的惩罚梯度。

这种机制通常提高更新稳定性，但 PPO clipping 不提供 TRPO 式的严格单调改进保证，实际结果仍取决于优势估计、学习率、epoch 数和 KL 漂移。

## A.11.4 KL 散度与 RLHF (KL Divergence in RLHF)

在一种常见的 KL 正则化表述中，优化目标是

$$
J(\pi)
=\mathbb E_{x\sim\mathcal D}\left[
\mathbb E_{y\sim\pi(\cdot\mid x)}r(x,y)
-\beta D_{KL}\!\left(
\pi(\cdot\mid x)\,\|\,\pi_{ref}(\cdot\mid x)
\right)
\right].
$$

这里第一项是样本级奖励的期望，第二项是给定 prompt 后两个完整输出分布之间的 KL，二者不能在记号上混为同一个 $R(x,y)$。实现时也常对采样输出使用整形分数

$$
\widetilde r(x,y)
=r(x,y)-\beta\log\frac{\pi(y\mid x)}{\pi_{ref}(y\mid x)},
$$

因为对 $y\sim\pi(\cdot\mid x)$ 取期望正好恢复上述 KL 项。

### A.11.4.1 为什么要减去 KL？

从正则化角度看，这相当于在最大化奖励的同时，用参考策略 $\pi_{ref}$ 限制策略漂移。在控制即推断等特定表述中，参考策略也可承担先验的角色；一般情况下直接称其为参考分布更准确。

对离散输出空间，KL 项展开为

$$
D_{KL}\!\left(\pi\|\pi_{ref}\right)
=\sum_y\pi(y\mid x)
\log\frac{\pi(y\mid x)}{\pi_{ref}(y\mid x)}.
$$

如果完全忽略奖励并只最小化该 KL，最优点是 $\pi=\pi_{ref}$。
如果完全忽略 KL，策略偏移缺少这一锚定，更可能利用奖励模型漏洞；是否发生以及表现为何种模式取决于奖励与优化过程。

该项是在**优化外部奖励**与**保持接近参考策略**之间做正则化权衡；这不同于强化学习中通常所说的 exploration-exploitation 权衡。

## A.11.5 DPO：绕过显式 RL 的偏好优化 (Direct Preference Optimization)

上一节把偏好优化写成期望奖励减去分布级 KL 正则项。

DPO 的关键观察是：在很多工程实践里，我们并不一定要显式地训练 $R_{RM}$、也不一定要跑 PPO。只要我们有偏好数据 $(x, y_w, y_l)$，就可以把“胜者应该比败者更像人类喜欢”的约束，直接写成一个可优化的对数似然目标。

### A.11.5.1 从 Bradley-Terry 偏好模型到优化目标

一个常见假设是：人类偏好服从 Bradley-Terry / Logit 模型，即“胜者胜出概率”由一个打分差决定：
$$ P(y_w \succ y_l \mid x) = \sigma(\Delta(x, y_w, y_l)) $$

对 KL 正则化奖励最优化问题，其最优策略满足

$$
r(x,y)=\beta\log\frac{\pi^*(y\mid x)}{\pi_{ref}(y\mid x)}+\beta\log Z(x),
$$

其中 $Z(x)$ 与候选输出 $y$ 无关。把这个关系代入 Bradley-Terry 模型时，两个候选共享的 $\beta\log Z(x)$ 抵消，得到打分差
$$ \Delta(x, y_w, y_l) = \beta\Big[(\log \pi_\theta(y_w|x) - \log \pi_\theta(y_l|x)) - (\log \pi_{ref}(y_w|x) - \log \pi_{ref}(y_l|x))\Big] $$

那么对偏好数据最大化对数似然，得到的就是 DPO 损失：
$$ \mathcal{L}_{\text{DPO}} = -\log \sigma\big(\Delta(x, y_w, y_l)\big) $$

### A.11.5.2 直观理解：为什么它像“带锚的策略梯度”？

- **胜者/败者差分**：只关心 $\log \pi_\theta(y_w|x) - \log \pi_\theta(y_l|x)$，等价于“相对偏好”的学习信号。
- **参考策略锚定**：减去 $\log \pi_{ref}(\cdot)$，会惩罚那些虽然能赢，但会把策略分布推得过远的更新方向（这与 RLHF 里 KL Penalty 的角色一致）。
- **$\beta$ 的作用**：它缩放策略相对参考策略的对数比，并在 DPO 的理论来源中对应 KL 正则系数/温度。较大的 KL 系数意味着最优策略更受参考模型约束；在有限数据和具体优化器下，不能只看损失中的乘法位置就断言“$\beta$ 越大越激进”。

DPO 不需要在线采样循环、显式奖励模型训练和 critic，因此实现链路通常比 PPO-RLHF 短；稳定性与最终效果仍取决于偏好数据、参考模型、$\beta$ 和分布偏移。
