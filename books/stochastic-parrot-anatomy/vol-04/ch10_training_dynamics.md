# 第十章 训练动力学与机制形成

最终 checkpoint 只展示训练终点。保存中间 checkpoints 并重复同一解释实验，可以观察 feature、回路和行为何时出现，区分逐渐加强、突然重组与后训练覆盖。

## 10.1 Checkpoint 轨迹

设训练步为 $t$，模型参数为 $\theta_t$。对固定行为集和内部指标，记录

$$
B_t(x),\qquad a_{u,t}(x),
\qquad C_t(x).
$$

$C_t$ 可以是 probe 性能、head ablation effect 或 circuit fidelity。只有最终两个 checkpoint 无法分辨连续演化和短暂阶段。

## 10.2 坐标对齐问题

不同 checkpoints 的神经元坐标可能发生置换或旋转。直接比较 neuron 42 的 activation，未必追踪同一功能。

可用 weight matching、CCA/CKA、feature decoder 相似度或共同 anchor 数据对齐表示。对齐算法本身可能强行制造连续性，应与行为和干预签名共同验证。

## 10.3 Grokking

在某些算法任务上，模型先记忆训练集，训练准确率很高而验证准确率低；继续优化后验证性能突然提高，被称为 grokking。机制研究发现表示与电路可能经历稀疏化或算法重组。

这是一类受控现象，不应把任何晚期性能提升都称为 grokking。需同时展示训练/验证曲线、数据规模、正则化和内部机制变化。

## 10.4 “涌现”与测量尺度

能力曲线在离散指标上可能看似突然：连续 logit 改善跨过 exact-match 阈值后，准确率从 0 跳到 1。解释训练动力学时应同时观察连续 loss、margin 和内部读出，避免把指标阈值误当机制瞬间产生。

真正的机制转变仍可能发生，但需要 checkpoint 密度、多个 seeds 和内部结构证据。

## 10.5 Feature Formation

对一个候选 feature，可追踪：

- activation selectivity 何时出现；
- decoder/logit effect 何时稳定；
- upstream 输入路径何时形成；
- ablation effect 是否与行为同步；
- feature 是否先宽泛后分化；
- 不同训练 seed 是否重复出现等价结构。

“先能 probe 读出，后才被下游使用”与“使用和读出同步出现”对应不同学习故事。

## 10.6 数据阶段与后训练

继续预训练、SFT 和偏好优化可能：

- 新增行为而复用原有 features；
- 改变已有 feature 的触发阈值；
- 增加新的抑制或拒答路径；
- 重写 late-layer output mapping；
- 造成能力遗忘或概念漂移。

比较 base 与 instruct model 时，不能只看同名 neuron；应对齐模型、控制模板，并区分基础知识表示与助手策略。

## 10.7 Model Diffing

若两个模型只相差一个训练阶段，可以研究参数差、activation 差和行为差。直接参数减法

$$
\Delta\theta=\theta_B-\theta_A
$$

在相同初始化和连续训练时有意义；独立训练模型存在置换与路径差异，参数差很难解释。

更稳妥的是使用同一输入集，比较 activation distributions、probe/circuit signatures，并用 patching 把 B 的中间状态移入 A 或反向移入。

## 10.8 数据归因的困难

要问“哪条训练数据造成这个 feature”，需要连接数十亿步梯度更新。influence functions、gradient similarity、data attribution 和训练重放都提供近似，但受非凸路径、optimizer state 和数据交互限制。

检索到与输出相似的训练文本，只说明表面近邻；不能单独证明因果记忆来源。强结论需要受控增删数据、重复训练或可追踪的小模型实验。

## 10.9 多 Seed 的机制稳定性

若同一任务在多个训练 seed 上都学会，内部实现可能：

- 在 neuron 坐标上不同，但存在可对齐子空间；
- 使用不同冗余 heads；
- 共享高层算法而低层 feature 不同；
- 完全采用不同 shortcut。

机制解释的“普遍性”应区分同 checkpoint 跨输入、同配方跨 seed、同架构跨规模和跨架构四个层级。

## 10.10 发展性可解释性的价值

训练轨迹可以检验静态解释难以区分的假说：如果 feature 被声称是行为的前置表示，它应在行为形成前或同时出现；如果它只是结果相关副产物，可能在行为之后才稳定。

时间顺序不是充分因果证据，却能排除部分叙事，并帮助选择更有信息量的干预 checkpoint。

## 10.11 结论

训练动力学把“模型里有什么”扩展为“它怎样形成”。checkpoint 对齐、连续指标和多 seed 是关键；没有这些控制，终点差异很容易被讲成过度确定的学习故事。
