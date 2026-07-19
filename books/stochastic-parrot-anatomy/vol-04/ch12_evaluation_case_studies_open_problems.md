# 第十二章 评估、案例与开放问题

可解释性方法最大的共同难题是缺少普遍 ground truth。真实大模型不附带人类可读源代码；研究者既发现机制又评价发现，容易让同一假设同时决定题目、指标和成功标准。本章给出跨路线评估协议，并用案例说明怎样从观察走向有限而可辩护的机制结论。

## 12.1 七个评估维度

1. **predictiveness**：解释能否预测新输入上的 activation、edge 或行为；
2. **behavioral faithfulness**：简化模型是否复现自然运行输出；
3. **intervention faithfulness**：解释与原模型是否响应相同干预；
4. **completeness**：未解释部分还承担多少目标 effect；
5. **stability**：对数据、seed、阈值、checkpoint 与规模是否稳定；
6. **scalability**：计算和人工成本怎样随模型与上下文增长；
7. **usefulness**：解释是否帮助人预测、发现或修复具体问题。

单一分数不能覆盖七项。研究应先声明主目标，再把其他项作为约束和失败边界。

## 12.2 解释评估的基本张量

把结果组织为四个轴：

$$
(\text{inputs},\text{models},\text{interventions},\text{metrics}).
$$

至少在 inputs 上分 discovery/validation/test，在 models 上区分 checkpoint/seed/scale，在 interventions 上包含自然、删除、替换和组合，在 metrics 上同时含目标行为与副作用。

只在一个轴上扩展样本量不能代替其他轴。例如十万个 prompts 仍不能证明跨 seed 稳定；十个模型也不能弥补所有干预都 off-manifold。

## 12.3 Ground Truth 的梯度

可使用不同强度的已知机制环境：

- 手写 toy network；
- 编译程序得到的 Transformer；
- 植入已知 backdoor/circuit 的模型；
- 由 causal abstraction 目标训练的半合成模型；
- 有可证明算法的受控任务；
- 自然模型中经多方法支持的 reference cases。

ground truth 越强，生态有效性通常越弱。InterpBench（2024）与 MIB（2025）分别提供半合成 circuit 和多模型任务级比较；它们能比较方法，但 benchmark 上排名不能自动外推到开放域大模型。

## 12.4 Behavioral 与 intervention fidelity

对解释模型 $E$，自然运行误差为

$$
L_{beh}=\mathbb E_{x\sim\mathcal D_{\mathrm{eval}}}
[d(F_E(x),F_M(x))].
$$

对干预 battery $\mathcal I$，

$$
L_{int}
=\mathbb E_{\substack{x\sim\mathcal D_{\mathrm{eval}}\\
I\sim\Pi_I(\cdot\mid x)}}
[d(F_E(I_E,x),F_M(I_M,x))].
$$

$\Pi_I(\cdot\mid x)$ 是声明的条件干预核，规定目标节点、来源样本、替换值与剂量；$I_E$、$I_M$ 是同一抽象干预在解释模型和原模型中的对齐实现。$d$ 可为 logit MSE、KL、任务 loss 或 response-curve distance。不同机制可在自然运行给同一输出，因此 $L_{int}$ 通常更能区分替代解释；但它也受干预对齐和 off-manifold 影响。

必须含原模型成功与失败样本，否则解释可能只拟合成功子集。

## 12.5 Completeness 与残差账本

每种分析器都有未解释部分：

- attribution 未分配的交互；
- circuit 补图；
- SAE reconstruction error；
- monitor 的 false negatives；
- 自动说明未覆盖 units。

建立残差账本：

$$
\text{target effect}
=\text{explained effect}
+\text{residual}
+\text{interaction error},
$$

这里只是报告结构，非线性下未必存在唯一加法分解。若不能精确相加，就分别报告 keep/remove/combined interventions，不能把残差静默设为零。

## 12.6 统计控制与搜索偏差

可解释性常扫描数万 units、layers、prompts 与 thresholds。最低统计纪律包括：

- discovery/validation/final test 分离；
- 以模板、文档或任务为 cluster 的 bootstrap；
- 预先指定主要 score 与方向；
- 对大规模 unit 搜索做 false-discovery 控制或独立复验；
- 报告全部 seeds 和失败 runs；
- 阈值—稀疏度—保真度曲线；
- 与随机、结构匹配和简化 baselines 比较；
- effect size 与区间，而不只给 $p$ 值。

自动 explainer、judge 和 circuit search 都属于模型选择过程。最终 test 不能再用于修订说明或阈值。

## 12.7 稳定性矩阵

不要用一个“robust”概括所有推广。分别测：

| 变化轴 | 需要固定/对齐 | 合格证据 |
|---|---|---|
| paraphrase/entity | 任务语义 | effect 分布与失败簇 |
| domain/language | 标签含义与 tokenization | 独立数据性能 |
| analysis seed | 数据与模型 | feature/circuit 匹配分布 |
| model seed | 配方与能力 | 功能角色和干预签名 |
| checkpoint | 时间轴与 anchor inputs | 机制轨迹 |
| scale | 训练 tokens/loss 与角色定义 | 跨规模角色复现 |
| method hyperparameters | 目标 estimand | sensitivity/Pareto 曲线 |

跨模型“相同机制”应说明允许的等价：相同 neuron、对齐方向、同类角色，还是相同高层算法。

## 12.8 Human usefulness

若目标是帮助人类，评估应让受试者完成实际任务：

- 预测模型在哪些输入失败；
- 找到 backdoor 触发条件；
- 选择最小副作用的修复；
- 判断模型依据是否符合政策；
- 决定何时升级人工复核。

与无解释、随机解释、行为 examples、原始 logits 和等时长 baseline 比较。测正确率、时间、calibration 与过度信任。主观“解释有道理”不能替代决策改善。

## 12.9 案例一：IOI/指代回路

一个完整研究链可为：

1. 定义 IOI logit difference 与模板分层；
2. 建立名字、位置和句法的 minimal pairs；
3. 用 attention/readout 形成 name-mover、inhibition 等角色假说；
4. 在 discovery set 做 head/edge patch；
5. 构造候选 circuit，分别测 keep、remove 和 minimality；
6. 在新模板与模型失败样本检验；
7. 做组合干预与 backup-path 分析；
8. 在独立 seed 上按功能角色重新定位，而非复制 head 编号。

[Wang 等（2023）的 GPT-2 Small IOI circuit](https://arxiv.org/abs/2211.00593) 是这种路线的重要实例。它支持特定模型和任务的精细机制研究，不是“语言指代已经被普遍解释”。

## 12.10 案例二：事实回忆

目标为主体—关系问题中正确 object 与竞争 object 的 logit difference。研究链：

1. 多关系、多模板、多语言和虚构实体行为集；
2. logit lens/probe 定位 subject、relation、object 的可读性；
3. 多种 corruption 下做双向 causal tracing；
4. 区分信息生成、传递与直接 logit 写入；
5. 用 feature/MLP 分析提出候选转换；
6. path patch 连接 subject、relation-conditioned route 与 object output；
7. 用替换主体/关系的 counterfactual 预测答案变化；
8. 报告 circuit 外 residual 与参数编辑副作用。

causal tracing 峰值只能说明指定恢复操作有效，不能单独证明事实唯一存储层。

## 12.11 案例三：SAE 安全 Feature

若声称 SAE feature 表示“危险生物请求”并可用于控制：

1. 说明 SAE 训练分布与 reconstruction/稀疏指标；
2. 在 benign/dual-use/malicious matched prompts 上测 feature；
3. 与 neuron、linear probe、随机与 supervised directions 比较；
4. 在 held-out domain 测自动说明和 firing；
5. 删除/插入 feature 并画剂量曲线；
6. 测拒答、帮助性、语言质量和无关安全类别；
7. 在多个 SAE seeds 做 feature/subspace matching；
8. 把 reconstruction error 对安全行为的影响计入。

只有 firing 高不能证明 feature 是危险性机制；只有 steering 成功也不能证明它在自然生成中被使用。

## 12.12 案例四：CoT 风险监测

设目标是从 Agent CoT 预测 reward hacking。完整评估包括：

1. 由环境 verifier 给行为标签，而非由同一 monitor 自标；
2. 分开 CoT、动作和最终输出观察面；
3. 报告 TPR/TNR、AUPRC、calibration 与 lead time；
4. 环境级 held-out，而非同任务随机 split；
5. intervention test 验证 monitor 能识别行为变化原因；
6. 自然、prompt obfuscation 与训练适应三种压力；
7. 不同 monitor 能力和模型家族；
8. 失败案例进入权限和人工升级策略。

这得到的是特定 Agent-monitor-system 的控制证据，不是 CoT 总体忠实性或系统安全证明。

## 12.13 Triangulation 不是投票

多方法一致只有在它们误差来源相对独立时才增强证据。gradient、gradient-based circuit search 与用同一梯度评分的验证不是三份证据；它们共享线性化假设。

更强组合是：

- 行为反事实定义目标；
- 不同数据上的 probe/feature 观察定位；
- 真实 patch 建立内部 effect；
- circuit 预测新组合干预；
- replacement residual 量化遗漏；
- 跨 seed 功能角色复验。

方法之间若冲突，不应以多数投票消除；冲突本身提示基线、交互或作用域不同。

## 12.14 统一主张模板

推荐最终结论写成：

> 对模型 $M$ 的 checkpoint $\theta$，在输入分布 $\mathcal D_{\mathrm{eval}}$、目标 $S$ 与干预族 $\mathcal I$ 下，方法 $A$ 定位到对象 $U$。在独立 test 上，估计量为 $\hat\tau$（含区间）；相对 baselines，$U$ 预测并在指定操作下改变 $S$。候选解释覆盖给定范围，仍不能排除替代机制 $H_1,H_2$，且在轴 $Z$ 上未验证。

它比“模型使用了一个负责 X 的 feature”更长，却精确给出可复现内容和证据上限。

## 12.15 截至 2026 年中的开放问题

- 什么 feature 定义能在保持功能的同时跨 seed 稳定？
- 可识别性需要哪些真实 activation 可检验的假设？
- 怎样评价 SAE 与 transcoder，而不依赖可能失真的代理 benchmark？
- 怎样量化 circuit completeness，又不把原模型全部放回图中？
- 局部 attribution graph 怎样推广为条件化全局机制？
- QK 动态路由与高阶 feature interactions 怎样进入可扩展图？
- 机制角色在训练、后训练、规模与架构间何时稳定？
- CoT monitorability 在更强优化压力下是否保持？
- internal monitor 怎样抵抗表示漂移与策略规避？
- 解释怎样转化为可验证修复，而不制造过度信任？

2026 年的新 benchmark、可识别 SAE 和 monitorability 研究正在改进这些问题的测量；它们尚未解决上述开放问题。各项工作的版本与证据等级见[本卷一手资料表](SOURCES.md)。

## 12.16 全卷收束

现代模型可解释性研究的核心不是寻找一种万能可视化，而是建立不同证据之间的语法：

- 行为实验规定待解释事实；
- 梯度、attention、probe 与自动说明产生候选；
- 干预区分相关、可控与被使用；
- circuits 把局部效应组合成计算假说；
- SAE/transcoder 改变分析基，同时引入非唯一性与替代误差；
- 训练动力学检验机制如何形成和是否稳定；
- CoT 与 monitor 提供新的过程观测面，但不构成安全证明；
- 综合评估把残差、失败和替代解释保留下来。

作为研究生/博士研究路线教材，本卷的学习目标是使读者能够从问题定义走到可复现实验和有限结论。学科仍有开放问题；严谨性不在于宣称模型已被完全解剖，而在于清楚标出当前方法仍不能知道什么。
