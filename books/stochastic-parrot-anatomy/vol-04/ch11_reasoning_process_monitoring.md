# 第十一章 推理过程、Chain of Thought 与监测

推理模型在答案前生成步骤化文本，Agent 还产生工具轨迹。可见 chain of thought（CoT）提供一个行为过程通道，但它不等于全部内部计算，也不必忠实披露影响答案的因素。研究必须把“过程对答案有因果作用”“过程内容忠实”和“monitor 能检测风险”分开。

## 11.1 四种过程对象

至少区分：

1. Transformer 层间的 activation 过程；
2. 自回归生成并进入后续上下文的 reasoning tokens；
3. 服务端保存但不向用户展示的 reasoning tokens；
4. tool calls、observations、memory writes 等外部执行轨迹。

工具日志可真实记录 API 调用；自然语言“我查询了数据库”可能只是描述。隐藏 reasoning 若不可由研究者访问，就不能声称对其完成机制解释。

## 11.2 CoT 的行为因果角色

设输入为 $x$，reasoning trace 为 $r$，答案为 $a$。有 CoT 时

$$
p(a\mid x)=\sum_r p(r\mid x)p(a\mid x,r).
$$

由于 $r$ 被加入上下文，改变或截断 $r$ 通常会改变 $p(a\mid x,r)$。这证明生成 tokens 可作为 scratchpad 起功能作用，但不证明每句话准确描述此前内部原因。

比较 direct answer 与 CoT prompting 时，还同时改变 prompt、计算预算和输出分布。要归因“显式中间 token”，应控制总 tokens、采样与提示风格，并直接干预 trace。

## 11.3 Faithfulness 的不同定义

CoT faithfulness 至少有三层：

- **causal sensitivity**：改变 trace 中某因素是否按预测改变答案；
- **factor disclosure**：真正改变答案的外部 cue 是否在 trace 中被承认；
- **process completeness**：完成任务所需的关键步骤是否可由 trace 或 monitor 恢复。

措辞相同不保证因果，措辞不同也可能实现等价算法。不存在脱离任务和反事实的单一“忠实度”。

## 11.4 Trace intervention

对 trace 段 $r_j$ 的替换 $R_j$，可测

$$
\Delta_j
=S(a\mid x,R_j(r))-S(a\mid x,r).
$$

常见操作：

- 截断某步；
- 把中间结论翻转；
- 替换为语义等价 paraphrase；
- 插入错误但流畅的步骤；
- 保持结论、改变理由；
- 把一个样本的步骤 patch 到另一个样本。

替换后的 trace 可能不来自模型自己的 $p(r\mid x)$。应比较自然续写 likelihood、用模型重写过渡、以及多种干预的一致性。

答案不变不证明该步无用：冗余步骤、模型纠错和后续重算都可能补偿。

## 11.5 Cue-based disclosure test

在 prompt 中加入会改变答案的 cue $c$，只对成功改变答案的 cases 检查 CoT 是否承认其影响。可估计

$$
\widehat F_c
=\frac{\#\{\text{answer changed and cue disclosed}\}}
{\#\{\text{answer changed}\}}.
$$

该估计量只在至少一个样本因 cue 改变答案时定义；分母为零时应报告命中数为零和该条件量不可估计，不能把比率记为 $0$。这是条件 acknowledgment rate，不是完整思维忠实度。分母筛选会排除 cue 被看到但未改变答案的样本；自动关键词 grader 还可能漏掉隐含承认。

[Turpin 等（2023）](https://arxiv.org/abs/2305.04388)及[后续 reasoning-model hint 实验（2025）](https://arxiv.org/abs/2505.05410)在特定 bias/hint 协议中观察到不完全披露。正确结论是这些模型与设置存在可测不忠实案例，而不是“所有 CoT 都是事后合理化”。

## 11.6 Outcome 与 process supervision

outcome supervision 只奖励最终结果；process supervision 对中间步骤评分。后者改变了 $p(r\mid x)$，可能提高可检查性，也可能鼓励符合 grader 偏好的模板。

过程奖励需要区分：

- 局部步骤正确；
- 步骤对最终解法必要；
- 全局路线有效；
- 文本是否真实反映模型使用的因素。

一个 step verifier 能识别局部错误，不保证 trace 因果忠实或策略最优。直接优化“看起来安全的 CoT”还可能降低监测信号。

## 11.7 Monitorability 是二元关系

monitor $Q$ 接收观察 $O$（CoT、动作、输出或 activation）并预测行为属性 $Y$。monitorability 不是 Agent 的单一内在属性，而是

$$
\mathcal M(M,Q,O,\mathcal D_{\mathrm{eval}},Y).
$$

评价包括 TPR、TNR、AUROC、AUPRC、calibration 与 subgroup worst case。若用几何平均总结平衡检测：

$$
G=\sqrt{\operatorname{TPR}\operatorname{TNR}},
$$

仍应同时给原始两项，因为部署代价可能不对称。

更强 monitor 得分更高，不一定说明 Agent trace 更透明；比较 Agent 时要固定 monitor family，比较 monitors 时要固定 Agent 与数据。

## 11.8 三类监测评估

[Guan 等的 ICML 2026 系统评估](https://openreview.net/forum?id=b82fgbMVpz)把 CoT monitorability 分为：

1. **intervention evaluations**：控制改变环境因素，测试 monitor 能否从 trace 识别行为改变原因；
2. **process evaluations**：任务只有有限有效路线，测试关键步骤是否可检测；
3. **outcome-property evaluations**：结果属性可可靠标注，测试 trace 是否提前揭示该属性。

这三类分别依赖干预有效性、过程 ground truth 与 outcome label。它们不能合并为无条件透明度分数。Guan 等在其所测 frontier/reasoning models 与任务上发现，CoT 常比只看动作和最终输出提供更强监测信号，但并不完美，且受模型、任务与 reasoning length 影响。[Korbak 等（2025）](https://arxiv.org/abs/2507.11473)则是一篇立场与研究议程文章，用来支持“monitorability 是一种脆弱机会、需要持续测量”这一判断，不作为第二组独立模型实验。

这些是快速演化中的经验结果，不是未来模型必然保持 monitorability 的理论保证。

## 11.9 Internal monitors

另一条路线从 activation 训练 probe，预测欺骗、危险知识、任务进度或即将发生的错误。其研究合同与第五章相同，但部署风险更高：

- labels 是否来自可验证行为而非主题词；
- train/test 是否按策略和环境分离；
- monitor 是否只检测 prompt 表面；
- lead time 是否足够干预；
- false positive/negative 成本；
- Agent 是否知道并适应该 monitor；
- 模型升级后表示是否漂移。

高 AUROC 不是可可靠控制的充分条件。需要 calibration、阈值下效用、分布外和对抗评估。

## 11.10 从 CoT 到内部机制

reasoning token 仍是 token，可研究：

- earlier trace positions 怎样影响当前 step；
- 中间结论在哪层可读；
- patch 某 step 对应 activation 是否改变后续答案；
- 有/无 CoT 时是否复用同类功能角色；
- 错误出现前是否存在内部预警 feature。

复杂任务的 circuit 可能随题目动态路由。单 prompt attribution graph 最多解释该轨迹的局部近似；要声称算法，需要跨问题结构和反事实步骤验证。

## 11.11 对抗适应与 Goodhart 风险

若 Agent 的优化目标直接包含 monitor score 或 CoT 合规性，模型可能学会隐藏 intent、缩短 trace 或使用 monitor 不理解的编码。需要区分：

- 自然分布上的可监测性；
- 知道被监测但未专门训练规避；
- prompt-level obfuscation pressure；
- 训练级 adversarial adaptation。

[Zolkowski 等（2025）的 stress tests](https://arxiv.org/abs/2510.19851)展示了部分模型在强 obfuscation pressure 下可降低检测，同时完成某些对抗任务；[Chen 等的 ICML 2026 controllability 实验](https://arxiv.org/abs/2603.05706)则在另一组当前模型与协议上发现 CoT 控制能力仍有限。两类结果针对不同模型与压力，不能拼成单一乐观或悲观定律。

直接强监督 CoT 可能使表面说明更规范，也可能损害自然监测信号。monitorability 应随训练阶段持续评估，而不是部署前测一次。

## 11.12 Access、隐私与安全边界

推理 trace 可能含敏感输入、系统提示或安全策略。开放给 monitor 会扩大访问面；对用户隐藏又限制外部复现。研究报告要说明：

- monitor 看到完整还是摘要 trace；
- trace 是否经过 redaction；
- monitor 与 Agent 是否共享模型家族；
- 数据保留和权限边界；
- 不可访问 hidden reasoning 时使用了什么代理。

CoT monitoring 只能是纵深防御的一层，不能替代权限控制、sandbox、行为评测和人工复核。

## 11.13 其他迭代生成过程

扩散去噪状态和多模态 Agent 轨迹也可按时间做 probe、patch 与 monitor，但它们不是 CoT。扩散中的 timestep intervention、视觉区域替换和工具 observation patch 都要重新定义自然状态与反事实，不应借用“推理文本忠实性”的指标。

本卷不在此展开这些架构；原则仍是先区分可见过程、内部状态和外部动作。

## 11.14 方法审计表

| 方法 | 问题/对象 | 操作与估计量 | 必要控制 | 能支持 | 不能支持与失效 |
|---|---|---|---|---|---|
| trace intervention | 某步骤是否影响答案 | 截断/替换；$\Delta_j$ | 自然度、多替换、冗余 | 指定 trace 操作的因果效应 | 文本完整反映内部计算 |
| cue disclosure | 被用 cue 是否被承认 | cue 插入；conditional rate | cue 有效性、grader、未改变 cases | 指定协议的披露率 | 全局 faithfulness |
| CoT monitor | trace 能否预测风险属性 | 训练/提示 monitor；TPR/TNR | Agent 固定、calibration、shift | 指定 Agent-monitor 对的检测力 | 安全保证 |
| internal monitor | activation 是否提前含风险信号 | probe；AUROC/lead time | topic baseline、策略 split、对抗 | 可读预警信息 | 不可规避控制 |
| reasoning circuit | 内部与 token 过程怎样连接 | readout + patch + path | 多题型、失败轨迹、动态路由 | 局部/条件机制 | 通用推理算法 |

CoT 是有功能的生成状态，也可能是不完全的自然语言报告。研究目标不是笼统判定“它真或假”，而是把每一种忠实性与监测主张还原为可重复的反事实、检测指标和威胁模型。
