# 第十一章 推理过程、Chain of Thought 与监测

推理模型会输出步骤化文字，内部也可能在可见答案前使用隐藏 token 或 latent computation。可见 chain of thought 提供了新的观察面，但它仍是模型生成的文本，不必忠实记录全部内部因果过程。

## 11.1 四种“过程”

应区分：

1. Transformer 层间的内部 activation 过程；
2. 自回归生成的可见 reasoning tokens；
3. 服务端未显示的 hidden reasoning tokens；
4. Agent 的工具调用与外部状态轨迹。

它们可以相互影响，却不是同一对象。工具日志可真实记录 API 调用，但自然语言“我先查询了数据库”可能只是描述。

## 11.2 CoT 的行为作用

要求模型生成中间步骤会把这些 token 加入后续上下文，从而改变条件分布：

$$
q(a\mid x,r)
$$

与直接回答 $q(a\mid x)$ 不同。CoT 可以充当 scratchpad，让模型分解任务；它也可能产生冗长但无效的合理化。

性能提升证明 reasoning text 对生成过程有功能作用，不证明每句话都对应内部真实原因。

## 11.3 Faithfulness 测试

可见推理的忠实性可通过：

- 修改中间结论，看最终答案是否随之改变；
- 插入错误提示，观察模型是否使用却不承认；
- 保持答案、改变 CoT，比较内部和输出；
- 隐藏或截断某些步骤；
- 要求简短与详细推理，检查核心依据是否稳定；
- 将 CoT 交给独立 verifier 检查每一步。

这些测试只能针对所定义的依赖关系。模型可能有多条正确推理路径，措辞变化不一定是不忠实。

## 11.4 Outcome 与 Process Supervision

outcome supervision 只根据最终答案给奖励；process supervision 对中间步骤提供标签。后者可以直接塑造可见推理分布，但标签正确性和粒度决定学到什么。

一个步骤 verifier 能识别局部错误，不保证全局策略最优；奖励易检查步骤也可能鼓励模板化解释。需要同时评估最终能力、步骤有效性与分布外泛化。

## 11.5 从 CoT 到内部 Circuit

可以把 reasoning token 当作普通 token，研究：

- 哪些 earlier tokens 通过 attention 影响当前步骤；
- 某中间结论在哪层可读出；
- patch 某步骤对应 activation 是否改变后续答案；
- 同一算法在无 CoT 与有 CoT 条件下是否复用回路。

复杂推理可能使用分布式、动态和 prompt-specific circuits。单一局部 attribution graph 很少等于完整算法证明。

## 11.6 Internal Monitor

训练 probe 或 classifier，从 activation 预测欺骗、危险知识、任务进度或即将发生的错误，是“监测”路线。它可以在输出出现前提供信号。

监测器需要独立评估：

- 正负标签怎样获得；
- 是否只检测表面 topic；
- 对新策略和新语言是否泛化；
- 模型是否能在优化压力下规避；
- false positive/negative 的部署代价；
- 监测器访问哪一模型版本与层。

高 probe AUROC 不等于可可靠控制行为。

## 11.7 可解释性与对抗适应

若模型训练目标知道监测器，可能学会在不触发已知 feature 的情况下完成行为。静态 probe 在自然分布上有效，不保证在对抗优化后有效。

防护方向包括多层随机化监测、隐藏监测器细节、机制多样化、训练时 adversarial evaluation 和权限隔离。但可解释性不应成为唯一安全边界。

## 11.8 多模态推理

视觉问答中的可见解释可能提到图像区域。验证时可将区域遮挡、替换 patch features 或 patch cross-attention，检查答案与说明是否共同变化。

若 explanation 指向红车但模型实际依赖背景水印，attention heatmap 可能仍看起来合理。需要输入反事实与内部干预联合验证 grounding。

## 11.9 扩散与迭代生成过程

扩散模型的中间状态是逐步去噪 latent。研究可追踪概念何时可由 probe 读出、cross-attention 在不同时间怎样作用，以及修改某时段 feature 如何影响布局或纹理。

“早期决定构图、后期决定细节”是常见经验概括，不是所有模型和所有概念的定理。应通过 time-resolved intervention 验证具体目标。

## 11.10 监测与解释的边界

一个 monitor 可以很准确但不可解释；一个漂亮 feature 说明也可能监测性能很差。部署目标若是早期预警，应优先报告 detection metrics、distribution shift 和规避测试；研究目标若是机制理解，则需进一步追踪 feature 的上下游回路。

## 11.11 结论

CoT 是有功能的生成状态，也是可能不忠实的自然语言报告。过程研究应把可见 token、隐藏 activation、工具轨迹和监测器分开验证。它扩展了可解释性的观察面，却没有消除对反事实和干预证据的需求。
