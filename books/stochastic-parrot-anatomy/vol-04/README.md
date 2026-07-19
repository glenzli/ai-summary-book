# 卷四 模型可解释性的研究路线

前两卷说明模型怎样建成、一次输出怎样执行，卷三解释概率怎样进入训练与生成。本卷只研究一个更窄的问题：**在给定模型、输入分布与行为指标后，我们怎样提出并检验关于内部计算的假说？**

这不是“读心术”，也不是一套已经成熟的统一理论。现代模型可解释性包含若干证据标准不同、分析单位不同的研究路线。attention 图、probe、activation patching、SAE feature 与 attribution graph 不能互相替代；它们分别观察、读出、干预或近似不同对象。

## 阅读前提与边界

读者应熟悉 Transformer 的前向计算、基本线性代数、微分与统计估计。卷一负责模型结构，卷三负责概率基础；本卷只在解释估计量需要时重述公式，不重复训练与采样原理。

本卷集中于已训练模型的行为与内部计算。数据卡、系统日志、部署审计和形式验证很重要，但不与机制解释混为一谈。多模态、扩散和 Agent 只在方法边界处出现，不另起平行综述。

## 全卷研究合同

每一种方法都用同一组问题审计：

| 项目 | 必须说明的内容 |
|---|---|
| 研究问题 | 描述行为、定位可读信息、估计干预效应，还是提出机制？ |
| 对象 | 输入、logit、residual state、head、neuron、direction、SAE latent、edge 或过程文本？ |
| 操作 | 观察、求导、训练读出器、删除、替换、重采样、投影或构造替代模型？ |
| 估计量 | 哪个标量或分布量化结果，基线和归一化是什么？ |
| 必要控制 | 哪些混杂、随机基线、负对照、held-out 数据和副作用必须检查？ |
| 能支持的结论 | 在什么模型、输入分布和干预族上成立？ |
| 不能支持的结论 | 哪一种更强主张仍缺证据？ |
| 失效模式 | 饱和、shortcut、off-manifold、冗余、替代误差、搜索偏差或跨 seed 不稳定？ |

贯穿全卷的记号是研究设定

$$
\mathfrak E=(M,\theta,\tau,\mathcal D_{\mathrm{eval}},S,\mathcal U,\mathcal I),
$$

其中 $M_\theta$ 是冻结的模型版本，$\tau$ 是 tokenizer 与序列化协议，$\mathcal D_{\mathrm{eval}}$ 是主张针对的输入分布，$S$ 是行为标量，$\mathcal U$ 是允许观察的内部单位，$\mathcal I$ 是允许实施的干预族。删除其中任何一项，“某组件解释了行为”通常都不是可复现主张。

## 章节路线

0. [可解释性究竟在研究什么](ch00_what_interpretability_studies.md)：研究对象、证据等级、可识别性与解释等价类。
1. [我们在模型内部看什么](ch01_internal_objects.md)：计算图中的张量、hook site、坐标依赖与读写路径。
2. [行为与反事实：先确认模型做了什么](ch02_behavior_counterfactuals.md)：目标指标、minimal pairs、对照分布与行为签名。
3. [梯度与输入归因](ch03_gradients_saliency.md)：局部敏感性、路径归因、遮挡、交互与 sanity checks。
4. [Attention、Residual 与 Logit Lens](ch04_attention_residual_readout.md)：路由、写入、直接读出与中间预测。
5. [Probes 与表示几何](ch05_probes_representation_geometry.md)：可解码性、control tasks、几何不变性与 causal probing。
6. [神经元、Feature 与自动解释](ch06_neurons_features_automated_explanations.md)：单位语义、反例搜索、自动说明与功能验证。
7. [Ablation、Patching 与因果追踪](ch07_interventions_patching.md)：节点干预、路径效应、off-manifold 与因果抽象。
8. [Circuits、路径与 Attribution Graphs](ch08_circuits_paths.md)：子图假说、faithfulness、completeness、替代模型与自动发现。
9. [Superposition、SAE 与稀疏表示](ch09_superposition_sparse_features.md)：字典学习、非唯一性、尺度退化、transcoder 与任务级评估。
10. [训练动力学与机制形成](ch10_training_dynamics.md)：checkpoint 对齐、机制出现、跨 seed 与跨规模稳定性。
11. [推理过程、Chain of Thought 与监测](ch11_reasoning_process_monitoring.md)：可见过程、因果忠实性、monitorability 与对抗适应。
12. [评估、案例与开放问题](ch12_evaluation_case_studies_open_problems.md)：统一评估协议、案例链、统计控制与研究边界。

辅助材料：[符号与术语](GLOSSARY.md) · [一手资料与证据状态](SOURCES.md)

## 结论纪律

本卷反复使用以下证据上限：

- activation pattern 支持共现描述，不自动支持概念或因果主张；
- held-out probe 支持指定读出器下的可解码性，不自动支持下游使用；
- steering 支持指定方向的控制能力，不自动支持正常前向依赖；
- patching 支持指定替换操作下的内部效应，不自动给出唯一自然中介；
- 稀疏回路支持目标分布上的压缩机制模型，不自动成为全模型完整解剖；
- 自动解释分数支持说明的预测能力，不自动支持 feature 的下游功能；
- CoT 可用于监测时仍只是一个有条件的观测通道，不是安全证明。

卷首给出统一研究检查表，各方法章以审计表压缩对象、操作、控制与证据上限；它们构成阅读论文和设计实验的报告标准。

下一卷不再提高证据等级，而转问这些有限技术事实允许怎样的第一人称、理解与责任语言。
