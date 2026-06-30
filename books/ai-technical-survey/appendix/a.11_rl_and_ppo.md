# 附录 A.11 强化学习与 PPO 原理 (RL & PPO Principles)

## A.11.1 强化学习基础 (RL Basics)

在 RLHF 中，我们将大模型微调建模为一个 RL 问题：
*   **Agent**: LLM (Policy $\pi_\theta$).
*   **Environment**: 用户 Prompt 及对话上下文。
*   **Action**: 生成下一个 Token。
*   **Reward**: 奖励模型 (Reward Model) 给出的分数。

我们的目标是最大化期望累积奖励：
$$ J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)] $$

## A.11.2 策略梯度 (Policy Gradient)

最直接的方法是使用策略梯度定理：
$$ \nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t} \nabla_\theta \log \pi_\theta(a_t | s_t) A_t \right] $$
其中 $A_t$ 是优势函数 (Advantage Function)，衡量当前动作比平均水平好多少。

然而，直接使用 Policy Gradient 极不稳定：
1.  **步长难以确定**: 更新太小收敛慢，更新太大导致策略崩溃。
2.  **数据效率低**: 每一批数据用完即弃。

## A.11.3 PPO: 近端策略优化 (Proximal Policy Optimization)

PPO 的核心思想是：**限制策略更新的幅度，防止新策略偏离旧策略太远。**

### A.11.3.1 重要性采样 (Importance Sampling)

为了复用旧策略 $\pi_{old}$ 采样的数据来更新新策略 $\pi_\theta$，我们引入重要性采样比率 $r_t(\theta)$：
$$ r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{old}(a_t | s_t)} $$

当 $\theta = \theta_{old}$ 时，$r_t = 1$。

### A.11.3.2 截断目标函数 (Clipped Objective)

PPO 的目标函数 $L^{CLIP}$ 由两部分取最小值构成，形成一个“悲观”的下界：

$$ L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t) \right] $$

*   $\epsilon$: 超参数，通常为 0.1 或 0.2。
*   **第一项** $r_t(\theta) A_t$: 标准的 TRPO 代理目标。
*   **第二项** $\text{clip}(\dots) A_t$: 将比率强制限制在 $[1-\epsilon, 1+\epsilon]$ 之间。

**直观理解**：
1.  如果 $A_t > 0$（动作很好）：我们希望增加该动作的概率 ($r_t > 1$)。但为了稳定，如果 $r_t > 1+\epsilon$，就不再给予额外的奖励梯度。
2.  如果 $A_t < 0$（动作很差）：我们希望减少该动作的概率 ($r_t < 1$)。但为了稳定，如果 $r_t < 1-\epsilon$，就不再给予额外的惩罚梯度。

这种机制保证了策略迭代的单调提升，避免了剧烈震荡。

## A.11.4 KL 散度与 RLHF (KL Divergence in RLHF)

在 RLHF 中，完整的 Reward 不仅包含 RM 的打分，还包含 KL 惩罚。

$$ R(x, y) = R_{RM}(x, y) - \beta D_{KL}(\pi_\theta(\cdot|x) || \pi_{ref}(\cdot|x)) $$

### A.11.4.1 为什么要减去 KL？

从贝叶斯角度看，这相当于在最大化奖励的同时，添加了一个**先验约束 (Prior)**。这个先验就是初始的 SFT 模型 $\pi_{ref}$。

展开 KL 项：
$$ D_{KL} = \sum \pi_\theta(y|x) \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)} $$

如果我们完全忽略 $R_{RM}$，只最小化 KL，那么 $\pi_\theta$ 就会坍缩回 $\pi_{ref}$。
如果我们完全忽略 KL，$\pi_\theta$ 就会为了高分利用 RM 的漏洞（输出乱码或重复模式）。

该项本质上是在 **Exploration (探索高分)** 和 **Exploitation (保持语言能力)** 之间寻求平衡。

## A.11.5 DPO：绕过显式 RL 的偏好优化 (Direct Preference Optimization)

上一节我们把 RLHF 写成“奖励 - KL”的形式：
$$ R(x, y) = R_{RM}(x, y) - \beta D_{KL}(\pi_\theta(\cdot|x) || \pi_{ref}(\cdot|x)) $$

DPO 的关键观察是：在很多工程实践里，我们并不一定要显式地训练 $R_{RM}$、也不一定要跑 PPO。只要我们有偏好数据 $(x, y_w, y_l)$，就可以把“胜者应该比败者更像人类喜欢”的约束，直接写成一个可优化的对数似然目标。

### A.11.5.1 从 Bradley-Terry 偏好模型到优化目标

一个常见假设是：人类偏好服从 Bradley-Terry / Logit 模型，即“胜者胜出概率”由一个打分差决定：
$$ P(y_w \succ y_l \mid x) = \sigma(\Delta(x, y_w, y_l)) $$

如果我们希望这个打分差来自策略相对参考策略的对数比（把 KL 锚定显式写进去），可以令：
$$ \Delta(x, y_w, y_l) = \beta\Big[(\log \pi_\theta(y_w|x) - \log \pi_\theta(y_l|x)) - (\log \pi_{ref}(y_w|x) - \log \pi_{ref}(y_l|x))\Big] $$

那么对偏好数据最大化对数似然，得到的就是 DPO 损失：
$$ \mathcal{L}_{\text{DPO}} = -\log \sigma\big(\Delta(x, y_w, y_l)\big) $$

### A.11.5.2 直观理解：为什么它像“带锚的策略梯度”？

- **胜者/败者差分**：只关心 $\log \pi_\theta(y_w|x) - \log \pi_\theta(y_l|x)$，等价于“相对偏好”的学习信号。
- **参考策略锚定**：减去 $\log \pi_{ref}(\cdot)$，会惩罚那些虽然能赢，但会把策略分布推得过远的更新方向（这与 RLHF 里 KL Penalty 的角色一致）。
- **$\beta$ 的作用**：$\beta$ 越大，模型会更激进地追随偏好；$\beta$ 越小，模型更保守、更贴近 $\pi_{ref}$。

这也是为什么在很多开源实践中，DPO 往往更“省事”且更稳定：它把 RLHF 的核心权衡压缩成了一个直接监督目标。
