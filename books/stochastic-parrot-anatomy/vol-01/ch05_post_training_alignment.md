# 第五章 后训练、对齐与模型适配

预训练让模型学习训练分布中的条件规律，却不直接规定它怎样响应指令、何时拒答、怎样调用工具，或怎样在多个可接受回答中取舍。后训练使用规模较小但目标更明确的数据和反馈，改变模型在给定接口下的行为。

“对齐”不是单一损失，也不是已经完成的性质。必须说明模型要对齐谁的偏好、在什么任务和权限边界内、以什么证据评价。本章先给出训练目标，再讨论数据与系统边界。

![后训练方法的关系](chapter_05/images/post_training_taxonomy.svg)

![从数据到部署评测的后训练流程](chapter_05/images/post_training_pipeline.svg)

## 5.1 方法地图：数据从哪里来，梯度怎样产生

后训练方法可以按监督信号与采样方式区分：

| 方法 | 训练记录 | 直接优化的量 | 是否需要当前策略在线采样 |
| --- | --- | --- | --- |
| SFT | $(x,y^*)$ | 参考回答的 token likelihood | 否 |
| reward model | $(x,y^+,y^-)$ | 偏好比较 likelihood | 否 |
| PPO-RLHF | prompt、策略轨迹、reward | 估计的期望正则化奖励 | 是 |
| DPO | $(x,y^+,y^-)$ 与参考策略 | 胜负回答的隐式 reward 差 | 否 |
| 可验证 RL / GRPO 类方法 | prompt、多条策略轨迹、验证 reward | 组内相对优势下的策略目标 | 通常是 |
| 蒸馏 | 教师分布、回答或轨迹 | 教师行为的拟合损失 | 视数据生成而定 |

“在线”是指训练过程中用正在更新或其近邻版本的策略生成新轨迹，不等于互联网在线服务。离线偏好优化可以更简单、可复现，却受固定候选分布限制；在线方法能探索新输出，也更容易进入 reward 或验证器未覆盖的区域。

## 5.2 Supervised Fine-Tuning

### 5.2.1 对话序列化与 loss mask

设一条多轮记录经 chat template 序列化为

$$
z=(z_1,\ldots,z_T),
$$

其中包括 system、user、assistant、工具结果、角色边界和停止 token。定义 $m_t=1$ 表示位置 $t$ 的目标 token 参与训练，并假设 $\sum_t m_t>0$，则

$$
\mathcal L_{\mathrm{SFT}}(z)
=-
\frac{1}{\sum_{t=1}^{T}m_t}
\sum_{t=1}^{T}m_t
\log p_\theta(z_t\mid z_{<t}).
$$

常见做法只令 assistant 回答区域的 $m_t=1$，但也可以训练工具调用、system 模板或用户 token。选择会改变模型学到的条件分布。attention mask 决定 $z_t$ 能读取哪些前缀，loss mask 决定该位置是否被监督，二者仍不可混用。

若 batch 内先对每条样本求平均再对样本平均，短回答与长回答权重相同；若先汇总全部有效 token 再归一化，每个 token 权重相同。两种目标都合理，但并不等价。数据配方必须声明归一化单位、样本权重和截断方式。

### 5.2.2 SFT 能教什么

SFT 同时教给模型任务内容、角色边界、格式、语气、停止方式和工具调用语法。高质量答案不只是“最终结论正确”：序列化中的隐藏模板、空白、结束标志和错误处理都会被拟合。

SFT 的支持集仍有限。训练集中从未出现的组合任务，不会因“指令微调”这个名称自动解决；同一答案被大量复制也会提高特定文风和模板的经验权重。

## 5.3 偏好数据与 Reward Model

### 5.3.1 偏好不是事实标签

对同一条件 $x$，偏好记录给出候选 $y^+$ 与 $y^-$，表示前者按声明的 rubric 更可取。偏好可来自人工比较、规则、模型裁判、可执行测试或它们的组合。

“更受偏好”不等于“更真实”。若候选生成器很弱，比较数据只覆盖一小块输出空间；若标注者奖励自信、长度或固定格式，模型会学习这些代理。数据记录至少应包含候选来源、生成参数、rubric 版本、标注者类型、分歧与弃权。

### 5.3.2 Bradley-Terry reward model

一种常见假设是偏好概率只由两个标量 reward 的差决定：

$$
P_\phi(y^+\succ y^-\mid x)
=\sigma\left(r_\phi(x,y^+)-r_\phi(x,y^-)\right).
$$

对已观察胜负对的负对数似然为

$$
\mathcal L_{\mathrm{RM}}
=-\mathbb E_{(x,y^+,y^-)}
\log\sigma\left(r_\phi(x,y^+)-r_\phi(x,y^-)\right).
$$

因为损失只含差值，对同一 prompt 加任意常数 $c(x)$，即令

$$
r'_\phi(x,y)=r_\phi(x,y)+c(x),
$$

不会改变偏好概率。故 pairwise 数据不识别跨 prompt 的绝对 reward 零点；把两个不同 prompt 的 raw score 直接比较，需要额外校准假设。

reward model 只在候选与标注分布附近经过验证。策略主动寻找高分输出时，可能进入模型外推区域并利用代理漏洞，这就是 reward hacking 的一个来源。

## 5.4 KL 正则化的 RLHF 目标

给定 prompt 分布 $x\sim\mathcal D$、参考策略 $\pi_{\mathrm{ref}}$ 和 reward $r(x,y)$，一种标准的分布级目标是

$$
J(\pi)
=\mathbb E_{x\sim\mathcal D}
\left[
\mathbb E_{y\sim\pi(\cdot\mid x)}r(x,y)
-\beta D_{\mathrm{KL}}
\left(\pi(\cdot\mid x)\|\pi_{\mathrm{ref}}(\cdot\mid x)\right)
\right].
$$

$\beta>0$ 控制偏离参考策略的代价。对固定 $x$ 和离散输出空间，加入归一化约束后对 $\pi(y\mid x)$ 求驻点，可得

$$
\pi^*(y\mid x)
=\frac{1}{Z(x)}
\pi_{\mathrm{ref}}(y\mid x)
\exp\left(\frac{r(x,y)}{\beta}\right),
$$

$$
Z(x)=\sum_y\pi_{\mathrm{ref}}(y\mid x)
\exp\left(\frac{r(x,y)}{\beta}\right).
$$

该解定义在 $\pi_{\mathrm{ref}}$ 的支持集上；若 $\pi_{\mathrm{ref}}(y\mid x)=0$，任何给该 $y$ 正概率的策略都会产生无限 KL。这个闭式解说明 reward 倾向与参考概率共同决定最优分布。$\beta$ 越大，在 reward 尺度固定时策略越受参考分布约束。

采样实现常把完整输出的整形 reward 写为

$$
\widetilde r(x,y)
=r(x,y)-\beta
\log\frac{\pi(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)},
$$

因为对 $y\sim\pi$ 取期望会恢复 KL 项。对自回归策略，序列 log-ratio 又等于各 token log-ratio 之和。参考策略 $\pi_{\mathrm{ref}}$ 是正则化锚点，不等于稍后用于重要性采样的旧策略 $\pi_{\mathrm{old}}$。

![奖励提升与参考策略约束之间的权衡](chapter_05/images/kl_anchor_tradeoff.png)

## 5.5 PPO 在语言模型后训练中的位置

把状态写成 $s_t=(x,y_{<t})$，动作写成下一个 token $a_t=y_t$。用旧策略 $\pi_{\mathrm{old}}$ 采样轨迹后，定义概率比

$$
\rho_t(\theta)
=\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\mathrm{old}}(a_t\mid s_t)}.
$$

PPO 的 clipped surrogate 最大化目标为

$$
J_{\mathrm{clip}}(\theta)
=\mathbb E_t\left[
\min\left(
\rho_t(\theta)\widehat A_t,
\operatorname{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)
\widehat A_t
\right)
\right].
$$

$\widehat A_t$ 由 reward、value critic 与回报估计产生。clip 使会继续放大代理目标的过大概率比停止贡献相同方向的增益，但它不把实际策略硬约束在区间内，也不保证真实性或安全性单调提高。完整训练还包含 value loss、entropy 或 KL 控制、rollout 批次和多轮 epoch。

RLHF 不是 PPO 的同义词。reward 建模、轨迹采样、优势估计、KL 锚定和策略更新是不同模块；PPO 只承担其中的策略优化。逐步推导见[强化学习与 PPO 附录](../appendices/learning-notes/a.11_rl_and_ppo.md)，来源见 [Schulman et al., 2017](SOURCE_NOTES.md#ref-schulman-ppo-2017) 与 [Ouyang et al., 2022](SOURCE_NOTES.md#ref-ouyang-2022)。

## 5.6 DPO：从正则化最优策略到分类损失

由上一节的闭式最优策略反解 reward：

$$
r(x,y)
=\beta\log\frac{\pi^*(y\mid x)}
{\pi_{\mathrm{ref}}(y\mid x)}
+\beta\log Z(x).
$$

把它代入 Bradley-Terry 模型时，同一 prompt 的 $\beta\log Z(x)$ 在胜负差中抵消。用待训练策略 $\pi_\theta$ 表示 $\pi^*$，得到

$$
\Delta_\theta(x,y^+,y^-)
=\beta\left[
\log\frac{\pi_\theta(y^+\mid x)}
{\pi_{\mathrm{ref}}(y^+\mid x)}
-
\log\frac{\pi_\theta(y^-\mid x)}
{\pi_{\mathrm{ref}}(y^-\mid x)}
\right],
$$

$$
\mathcal L_{\mathrm{DPO}}
=-\mathbb E\log\sigma(\Delta_\theta).
$$

DPO 因而可以直接用偏好对训练策略，不必先拟合显式 reward model，也不需要在这一步在线 rollout 或训练 critic。这个推导依赖 KL 正则化最优策略与 Bradley-Terry 偏好假设；它不是“任意偏好数据上都等价于任意 RLHF 实现”的结论。

序列 log probability 是 token log probability 之和：

$$
\log\pi_\theta(y\mid x)
=\sum_{t=1}^{|y|}\log\pi_\theta(y_t\mid x,y_{<t}).
$$

长度因此进入目标。把总和改成长度平均可以改变长度偏置，却也改变标准 DPO 所对应的隐式 reward。偏好候选的长度分布、截断与结束 token 必须共同检查。DPO 来源见 [Rafailov et al., 2023](SOURCE_NOTES.md#ref-rafailov-dpo-2023)。

## 5.7 采样、筛选与在线数据闭环

### 5.7.1 Best-of-$N$ 与 rejection sampling

给定 prompt，从策略独立采样 $N$ 个候选并选择 reward 最大者：

$$
y^*=\arg\max_{1\le i\le N}r(x,y_i),
\qquad y_i\sim\pi(\cdot\mid x).
$$

增加 $N$ 不会降低所用 reward 的期望最大值，却使最终输出不再服从原策略 $\pi$，并近似按 $N$ 倍增加生成开销。若把获胜回答回灌 SFT，产生的是经过选择器重加权的离线数据；选择器偏差会一起被蒸馏。

### 5.7.2 在线与离线的分布差异

离线偏好集中的候选来自某个行为策略 $\mu$。训练后的 $\pi_\theta$ 若远离 $\mu$，数据不再覆盖它常生成的失败。在线循环可以用新策略补充候选，但 reward model、规则和人工审核也必须跟随扩展；仅“重新采样”不能自动修复监督信号的盲区。

## 5.8 可验证奖励、GRPO 与推理强化

数学、代码和形式任务可以用最终答案、单元测试或证明检查器提供较客观反馈。对同一 prompt $x$，从旧策略采样 $G$ 个回答 $y_1,\ldots,y_G$，得到 reward $r_i$。组内标准化优势可写为

$$
\widehat A_i
=\frac{r_i-\bar r}
{\sqrt{G^{-1}\sum_{j=1}^{G}(r_j-\bar r)^2}+\varepsilon},
\qquad
\bar r=\frac1G\sum_{j=1}^{G}r_j,
\qquad \varepsilon>0.
$$

GRPO 类目标用组内相对优势替代单独训练的 value critic。一个代表性 clipped 形式是

$$
J_{\mathrm{GRPO}}(\theta)
=\frac1G\sum_{i=1}^{G}\frac1{|y_i|}
\sum_{t=1}^{|y_i|}
\min\left(
\rho_{i,t}\widehat A_i,
\operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)
\widehat A_i
\right)
-\beta\mathcal K,
$$

其中 $\rho_{i,t}$ 是新旧策略 token 概率比，$\mathcal K$ 表示对参考策略的 KL 估计。具体实现对 advantage、KL 估计和 token 归一化有不同变体，引用方法名时应给出公式版本。

减去组均值严格消除 reward 的加性平移。若 $\varepsilon=0$ 且标准差非零，正比例缩放 reward 也不改变 $\widehat A_i$；实际使用 $\varepsilon>0$ 时，这一尺度不变性只在组标准差远大于 $\varepsilon$ 时近似成立。若一组 reward 全同，分子全为零，策略优势信号恰为零。组内标准化可缓和 prompt 间尺度差异，但不保证完全消除它，也不消除 rollout 成本、reward hacking 或 on-policy 分布漂移。

可验证 reward 通常只证明测试通过，不证明推理文本忠实反映内部因果过程。模型可能利用测试缺口、浮点容差、环境状态或答案泄漏。隐藏测试、沙箱、过程检查与对抗样本分别覆盖不同风险。来源见 [Shao et al., 2024](SOURCE_NOTES.md#ref-shao-deepseekmath-2024) 与 [DeepSeek-AI, 2025](SOURCE_NOTES.md#ref-deepseek-r1-2025)。

## 5.9 拒答与安全后训练

安全训练常包含危险请求、边界案例、良性近邻和替代帮助。只增加拒答样本容易造成过度拒答；只奖励任务完成又可能让模型忽略风险。需要分开评价：

- 明确危险任务的拒绝；
- 良性近邻任务的能力保持；
- 含糊请求的澄清；
- 多轮诱导和上下文污染；
- 工具环境中的实际权限遵守。

模型侧训练只是系统安全的一层。真实工具权限、身份、资源配额和副作用确认仍由第十一章的运行时控制。一个模型说“我不能执行”不等于系统已经撤销其权限。

## 5.10 后训练数据工程

后训练数据远少于预训练数据，单条样本的模板和质量影响更大。管线应覆盖任务与难度分层、候选生成、去重、人工审核、rubric 版本、污染检查和失败样本回流。

合成数据可以扩大规模，但教师错误、固定文风和同质化也会被复制。合格记录应区分：人工原始样本、模型候选、验证器接受、模型裁判选择和人工修订。仅把最终记录统称“高质量数据”，无法追踪偏差来自何处。

继续预训练与 SFT 也不能互换：

- **continued pretraining** 延续语言建模目标，主要改变领域语料分布、术语和文体；
- **SFT** 在序列化任务上提高指定回答的条件 likelihood，直接塑造接口行为；
- **偏好优化或 RL** 在多个候选之间施加相对或 reward 信号。

它们可以串联或交错。数据混合、学习率、步数与保留集决定学会新领域和遗忘旧能力之间的取舍。

## 5.11 蒸馏

设教师与学生在温度 $T$ 下的 token 分布为 $p_{\mathrm{teach},T}$ 与 $p_{\mathrm{stud},T}$。logit 蒸馏的一项常写为

$$
\mathcal L_{\mathrm{KD}}
=T^2\sum_t
D_{\mathrm{KL}}
\left(p_{\mathrm{teach},T}(\cdot\mid c_t)
\|p_{\mathrm{stud},T}(\cdot\mid c_t)\right).
$$

$T^2$ 用于补偿 softmax 温度对梯度尺度的影响，实践中还可与真实标签交叉熵混合。若教师只通过 API 暴露文本，蒸馏就退化为对教师采样回答或轨迹做 SFT、筛选或偏好训练，无法获得完整 soft target。

学生获得的是蒸馏数据中可学习的教师行为，不是教师权重或完整内部机制。能力受学生架构、容量、tokenizer、采样分布和选择器限制。

## 5.12 LoRA、QLoRA 与参数高效适配

对冻结权重 $W_0\in\mathbb R^{d_{out}\times d_{in}}$，LoRA 把更新限制为

$$
W'=W_0+\Delta W,
\qquad
\Delta W=\frac{\alpha}{r}BA,
$$

其中

$$
A\in\mathbb R^{r\times d_{in}},
\qquad
B\in\mathbb R^{d_{out}\times r},
\qquad
r\ll\min(d_{in},d_{out}).
$$

该层训练参数量是

$$
r(d_{in}+d_{out}),
$$

而完整矩阵有 $d_{in}d_{out}$ 个参数。常把一个因子初始化为零，使训练开始时 $\Delta W=0$；$\alpha/r$ 控制 adapter 更新尺度。选哪些 query、key、value、output 或 MLP 矩阵作为目标，和秩 $r$ 同样重要。

QLoRA 冻结量化后的底模权重，在矩阵运算时按 kernel 需要反量化或使用低精度计算，只对 LoRA 参数反向与维护优化器状态。它降低训练显存，不表示底模量化误差消失，也不保证任意 adapter 可无损合并到任意量化工件。

来源见 [Hu et al., 2021](SOURCE_NOTES.md#ref-hu-lora-2021) 与 [Dettmers et al., 2023](SOURCE_NOTES.md#ref-dettmers-qlora-2023)。

![LoRA 的低秩更新结构](chapter_05/images/lora_diagram.png)

![LoRA 秩与适配效果的示意关系](chapter_05/images/lora_rank_tradeoff.png)

## 5.13 模型合并

若多个微调模型来自同一底模，最简单的 task-vector 组合为

$$
W_{\mathrm{merge}}
=W_0+\sum_k\alpha_k(W_k-W_0).
$$

权重形状相同只是可执行条件，不是功能兼容保证。不同微调轨迹可能在同一参数方向上产生冲突；adapter 的秩、缩放和目标模块也可能不同。合并后必须重新评价，不能把各子模型分数直接相加。

![开放模型、适配器与派生工件的关系](chapter_05/images/open_model_ecosystem.svg)

## 5.14 后训练怎样评价

评测首先固定模型工件、chat template、system prompt、解码参数、工具版本和最大长度。否则比较的可能是两套服务配置，而非两种后训练方法。

| 方面 | 例子 |
| --- | --- |
| 指令跟随 | 格式、约束、多轮一致性 |
| 任务能力 | 知识、代码、数学、领域任务 |
| 偏好 | 盲化人评与学习型裁判校准 |
| 安全 | 危险、良性近邻、越狱和工具场景 |
| 校准 | 不知道时的表达、选择性回答 |
| 回归 | 预训练能力、语言与关键子群 |

对成对比较，若 $w_i\in\{0,\tfrac12,1\}$ 分别表示失败、平局、获胜，则估计胜率为

$$
\widehat p=\frac1n\sum_{i=1}^{n}w_i.
$$

置信区间应以独立抽样单位重采样；若同一 prompt 有多名标注者，不能把相关的每次点击都当作独立题目。位置随机化、匿名输出、rubric 一致性和裁判对长度/格式的偏差都需检查。

平均胜率不能替代失败分析。应保存 prompt-level 配对结果，分别查看能力提升来自哪些子群、哪些旧能力退化，以及 reward 或裁判是否在奖励可见风格而非任务正确性。

后训练把基座模型变成某种助手或策略，但部署成本与在线延迟由另一套机制决定。下一章因此从行为塑造转向推理效率与模型服务。

主要来源包括 InstructGPT、PPO、DPO、DeepSeekMath、GRPO、Constitutional AI、蒸馏、LoRA 与 QLoRA，统一登记在[卷内来源表](SOURCE_NOTES.md)。
