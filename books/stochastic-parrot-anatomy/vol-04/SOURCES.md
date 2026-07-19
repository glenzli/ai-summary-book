# 一手资料与证据状态

本表是卷四正文中作者—年份引用的定位表，检索截止到 **2026-07-19**。优先列方法原论文、同行评审版本和作者公开技术报告；综述只用于导航，不作为经验结论的唯一依据。

状态标记：

- **[同行评审]**：正式会议或期刊论文；结论仍只在论文实验范围内成立。
- **[机构/作者研究报告]**：公开方法、实验与代码，但未必经过匿名同行评审。
- **[预印本]**：尚不能把新假说或单组经验结果写成领域定论。
- **[工作坊]**：用于记录新方向或受限实验，证据权重低于正式主会论文。

## 研究规范、行为与反事实

- Finale Doshi-Velez and Been Kim, [*Towards A Rigorous Science of Interpretable Machine Learning*](https://arxiv.org/abs/1702.08608), 2017，**[预印本]**。提出解释对象与评估需要操作化；本卷不采用其中任何单一定义作为定论。
- Marco Tulio Ribeiro et al., [*Beyond Accuracy: Behavioral Testing of NLP Models with CheckList*](https://aclanthology.org/2020.acl-main.442/), ACL 2020，**[同行评审]**。能力、invariance 与 directional expectation 测试。
- Matt Gardner et al., [*Evaluating Models' Local Decision Boundaries via Contrast Sets*](https://aclanthology.org/2020.findings-emnlp.117/), Findings of EMNLP 2020，**[同行评审]**。局部反事实与对照集；不提供内部机制识别。
- Atticus Geiger et al., [*Causal Abstractions of Neural Networks*](https://proceedings.neurips.cc/paper/2021/hash/4f5c422f4d49a5a807eda27434231040-Abstract.html), NeurIPS 2021，**[同行评审]**；Atticus Geiger et al., [*Inducing Causal Structure for Interpretable Neural Networks*](https://proceedings.mlr.press/v162/geiger22a.html), ICML 2022，**[同行评审]**。interchange interventions 与受限高层因果抽象。
- Denis Sutter et al., [*The Non-Linear Representation Dilemma: Is Causal Abstraction Enough for Mechanistic Interpretability?*](https://papers.nips.cc/paper_files/paper/2025/hash/dbb98528c9870377f3f0d133aae6050b-Abstract-Conference.html), NeurIPS 2025，**[同行评审]**。说明不受限高容量对齐可使因果抽象变得空泛；正文只采用其方法警告。

## 梯度、遮挡与归因

- Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman, [*Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps*](https://arxiv.org/abs/1312.6034), 2013，**[预印本/早期方法论文]**。输入梯度 saliency。
- Mukund Sundararajan, Ankur Taly, and Qiqi Yan, [*Axiomatic Attribution for Deep Networks*](https://proceedings.mlr.press/v70/sundararajan17a.html), ICML 2017，**[同行评审]**。Integrated Gradients、implementation invariance 与端点 completeness。
- Daniel Smilkov et al., [*SmoothGrad: Removing Noise by Adding Noise*](https://arxiv.org/abs/1706.03825), 2017，**[预印本]**。邻域噪声平均 saliency。
- Scott Lundberg and Su-In Lee, [*A Unified Approach to Interpreting Model Predictions*](https://arxiv.org/abs/1705.07874), NeurIPS 2017，**[同行评审]**。SHAP 统一框架；文本中的缺失分布仍须另行定义。
- Pieter-Jan Kindermans et al., [*The (Un)reliability of Saliency Methods*](https://arxiv.org/abs/1711.00867), 2017，**[预印本]**。输入变换与归因不变性反例。
- Julius Adebayo et al., [*Sanity Checks for Saliency Maps*](https://proceedings.neurips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html), NeurIPS 2018，**[同行评审]**。参数与标签随机化检查。

## Attention、Residual 与中间读出

- Sarthak Jain and Byron C. Wallace, [*Attention is not Explanation*](https://aclanthology.org/N19-1357/), NAACL 2019，**[同行评审]**；Sarah Wiegreffe and Yuval Pinter, [*Attention is not not Explanation*](https://aclanthology.org/D19-1002/), EMNLP 2019，**[同行评审]**。两文共同用于界定 attention weights 的证据范围，不作二元裁决。
- Samira Abnar and Willem Zuidema, [*Quantifying Attention Flow in Transformers*](https://arxiv.org/abs/2005.00928), ACL 2020，**[同行评审]**。attention rollout/flow 及与其他重要性指标的比较。
- Nelson Elhage et al., [*A Mathematical Framework for Transformer Circuits*](https://transformer-circuits.pub/2021/framework/index.html), 2021，**[作者研究报告]**。QK/OV、residual 与 composition 框架。
- Nora Belrose et al., [*Eliciting Latent Predictions from Transformers with the Tuned Lens*](https://arxiv.org/abs/2303.08112), 2023，**[预印本]**。中间 residual 的学习型词表读出及校准。

## Probes、表示几何与概念方向

- John Hewitt and Percy Liang, [*Designing and Interpreting Probes with Control Tasks*](https://aclanthology.org/D19-1275/), EMNLP 2019，**[同行评审]**。control tasks 与 selectivity。
- Tiago Pimentel et al., [*Information-Theoretic Probing for Linguistic Structure*](https://aclanthology.org/2020.acl-main.420/), ACL 2020，**[同行评审]**。以信息估计解释 probe 容量问题。
- Elena Voita and Ivan Titov, [*Information-Theoretic Probing with Minimum Description Length*](https://aclanthology.org/2020.emnlp-main.14/), EMNLP 2020，**[同行评审]**。online coding/MDL probing。
- Shauli Ravfogel et al., [*Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection*](https://aclanthology.org/2020.acl-main.647/), ACL 2020，**[同行评审]**。迭代线性投影删除；不保证所有非线性信息消失。
- Yanai Elazar et al., [*Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals*](https://aclanthology.org/2021.tacl-1.10/), TACL 2021，**[同行评审]**。投影删除与行为测试。
- Simon Kornblith et al., [*Similarity of Neural Network Representations Revisited*](https://proceedings.mlr.press/v97/kornblith19a.html), ICML 2019，**[同行评审]**。linear CKA 及表示相似性的不变性。
- Been Kim et al., [*Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors*](https://proceedings.mlr.press/v80/kim18d.html), ICML 2018，**[同行评审]**。TCAV 与随机概念对照。
- Marc E. Canby et al., [*How Reliable are Causal Probing Interventions?*](https://aclanthology.org/2025.ijcnlp-long.47/), IJCNLP-AACL 2025，**[同行评审]**。在所测方法与模型上量化 causal probing 的 completeness–selectivity 权衡。
- Laure Ciernik et al., [*Objective Drives the Consistency of Representational Similarity across Datasets*](https://proceedings.mlr.press/v267/ciernik25a.html), ICML 2025，**[同行评审]**。表示相似性对 stimulus dataset 与训练目标的依赖。

## 神经元与自动解释

- David Bau et al., [*Network Dissection: Quantifying Interpretability of Deep Visual Representations*](https://openaccess.thecvf.com/content_cvpr_2017/html/Bau_Network_Dissection_Quantifying_CVPR_2017_paper.html), CVPR 2017，**[同行评审]**。以带标签概念数据测 unit selectivity；结果范围主要是视觉网络。
- Steven Bills et al., [*Language Models Can Explain Neurons in Language Models*](https://openai.com/index/language-models-can-explain-neurons-in-language-models/), 2023，**[机构研究报告]**。自动生成与模拟 neuron 说明；作者结果同时显示大量说明分数有限。
- Arnau Marin-Llobet and Javier Ferrando, [*Automated Interpretability and Feature Discovery in Language Models with Agents*](https://arxiv.org/abs/2605.01555), 2026，**[预印本]**。迭代竞争假说和反例搜索的自动化路线；正文不把单组结果写成自动解释已闭环。

## 内部干预与模型编辑

- Kevin Meng et al., [*Locating and Editing Factual Associations in GPT*](https://arxiv.org/abs/2202.05262), NeurIPS 2022，**[同行评审]**。事实回忆的 causal tracing 与 ROME 编辑；不支持事实唯一存储层。
- Fred Zhang and Neel Nanda, [*Towards Best Practices of Activation Patching in Language Models*](https://arxiv.org/abs/2309.16042), ICLR 2024，**[同行评审]**。corruption、metric 与 patching 设计的敏感性。
- Marc Canby et al. 2025 causal probing 论文同样用于第七章关于 projection intervention 的 completeness 与 selectivity。
- Atticus Geiger et al. 2021/2022 的 causal abstraction 工作用于 interchange intervention；正文明确限制 alignment map 容量。

## Circuits、路径与自动发现

- Catherine Olsson et al., [*In-context Learning and Induction Heads*](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html), 2022，**[作者研究报告]**。induction head 行为、composition 与训练 phase change；范围为所测小型 Transformer。
- Kevin Wang et al., [*Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small*](https://arxiv.org/abs/2211.00593), ICLR 2023，**[同行评审]**。IOI circuit、faithfulness、completeness 与 minimality 案例。
- Lawrence Chan et al., [*Causal Scrubbing: a Method for Rigorously Testing Interpretability Hypotheses*](https://www.alignmentforum.org/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing), 2022，**[作者研究报告]**。高层假说、条件 resampling 与严格反例测试。
- Arthur Conmy et al., [*Towards Automated Circuit Discovery for Mechanistic Interpretability*](https://arxiv.org/abs/2304.14997), NeurIPS 2023，**[同行评审]**。ACDC 的自动 edge 搜索与任务级评估。
- János Kramár et al., [*AtP*: An Efficient and Scalable Method for Localizing LLM Behaviour to Components*](https://arxiv.org/abs/2403.00745), 2024，**[预印本]**。attribution patching 的高效近似与饱和修正。
- Emmanuel Ameisen et al., [*Circuit Tracing: Revealing Computational Graphs in Language Models*](https://transformer-circuits.pub/2025/attribution-graphs/methods.html), 2025，**[机构/作者研究报告]**。cross-layer transcoder、local replacement model、error nodes 与 attribution graph。
- Harish Kamath et al., [*Tracing Attention Computation Through Feature Interactions*](https://transformer-circuits.pub/2025/attention-qk/index.html), 2025，**[机构/作者研究报告]**。把 feature interaction 分析扩展到 attention QK；仍是局部替代图研究。

## Superposition、SAE 与 Transcoder

- Nelson Elhage et al., [*Toy Models of Superposition*](https://transformer-circuits.pub/2022/toy_model/index.html), 2022，**[作者研究报告]**。稀疏 feature 的线性 toy model；不作为真实 LLM superposition 的普遍证明。
- Trenton Bricken et al., [*Towards Monosemanticity*](https://transformer-circuits.pub/2023/monosemantic-features/index.html), 2023，**[机构研究报告]**；Adly Templeton et al., [*Scaling Monosemanticity*](https://transformer-circuits.pub/2024/scaling-monosemanticity/), 2024，**[机构研究报告]**。SAE feature discovery 与规模化实例。
- Robert Huben et al., [*Sparse Autoencoders Find Highly Interpretable Features in Language Models*](https://openreview.net/forum?id=F76bwRSLeK), ICLR 2024，**[同行评审]**。小型语言模型中的稀疏字典与 feature 评估。
- Joshua Engels et al., [*Not All Language Model Features Are Linear*](https://arxiv.org/abs/2405.14860), 2024，**[预印本]**。非线性/多维 feature 结构的经验案例，限制“所有概念都是单方向”的表述。
- Samuel Marks et al., [*Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models*](https://proceedings.iclr.cc/paper_files/paper/2025/hash/3ba4d47a83e498c2b1a0868cba20f6de-Abstract-Conference.html), ICLR 2025，**[同行评审]**。SAE feature circuits、显式 error terms 与编辑；结论限于所测任务。
- Jacob Dunefsky, Philippe Chlenski, and Neel Nanda, [*Transcoders Find Interpretable LLM Feature Circuits*](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2b8f4db0464cc5b6e9d5e6bea4b9f308-Abstract-Conference.html), NeurIPS 2024，**[同行评审]**。用稀疏 transcoder 近似 MLP 输入输出关系。
- David Chanin et al., [*A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders*](https://openreview.net/forum?id=R73ybUciQF), NeurIPS 2025，**[同行评审]**。受控任务中的 splitting/absorption 失败模式。
- Aleksandar Makelov, Georg Lange, and Neel Nanda, [*Towards Principled Evaluations of Sparse Autoencoders for Interpretability and Control*](https://openreview.net/forum?id=1Njl73JKjB), ICLR 2025，**[同行评审]**。以受监督任务字典比较 SAE disentanglement 与 control。
- Adam Karvonen et al., [*SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability*](https://proceedings.mlr.press/v267/karvonen25a.html), ICML 2025，**[同行评审]**。八项 SAE 评估与多架构比较；指标有效性本身仍需审计。
- Aaron Mueller et al., [*MIB: A Mechanistic Interpretability Benchmark*](https://proceedings.mlr.press/v267/mueller25a.html), ICML 2025，**[同行评审]**。circuit 与 causal-variable localization；正文只引用其所测模型/任务上的比较结果。
- Walter Nelson, Theofanis Karaletsos, and Francesco Locatello, [*Toward Identifiable Sparse Autoencoders*](https://openreview.net/forum?id=miLK9YcxtA), ICML 2026，**[同行评审]**。分析标准 SAE 跨 run 不稳定并提出 iSAE；不等于现实 activation 的唯一语义字典已被证明。
- David Chanin, [*Are Sparse Autoencoder Benchmarks Reliable?*](https://arxiv.org/abs/2605.18229), 2026，**[预印本]**。对 SAEBench 指标的 reseed、合成真值与可区分性审计；作为最新负面证据，不写成最终裁决。

## 训练动力学与机制稳定性

- Neel Nanda et al., [*Progress Measures for Grokking via Mechanistic Interpretability*](https://arxiv.org/abs/2301.05217), ICLR 2023，**[同行评审]**。模运算任务中算法回路形成与 progress measures。
- Olsson et al. 2022 induction-head 报告也提供训练 phase-change 实例；正文明确限制到所测模型。
- Curt Tigges et al., [*LLM Circuit Analyses Are Consistent Across Training and Scale*](https://proceedings.neurips.cc/paper_files/paper/2024/hash/47c7edadfee365b394b2a3bd416048da-Abstract-Conference.html), NeurIPS 2024，**[同行评审]**。在特定 Pythia 规模与任务上比较功能 circuits；只作为有限迁移证据。
- Xu Wang et al., [*Towards Understanding Fine-Tuning Mechanisms of LLMs via Circuit Analysis*](https://openreview.net/forum?id=45EIiFd6Oa), ICML 2025，**[同行评审]**。受控数学任务上的 fine-tuning circuit 变化。
- Jason Gross et al., [*Compact Proofs of Model Performance via Mechanistic Interpretability*](https://openreview.net/forum?id=2zWbzx50mH), NeurIPS 2024，**[同行评审]**。在小型受控 Transformer 上跨 151 seeds 构造计算机辅助性能证明；不外推到通用 LLM。

## Chain of Thought、Reasoning 与 Monitorability

- Miles Turpin et al., [*Language Models Don't Always Say What They Think*](https://arxiv.org/abs/2305.04388), NeurIPS 2023，**[同行评审]**。biasing cue 与 CoT 披露的反事实实验。
- Tamera Lanham et al., [*Measuring Faithfulness in Chain-of-Thought Reasoning*](https://arxiv.org/abs/2307.13702), 2023，**[预印本]**。截断、错误注入等多种 faithfulness probes。
- Yanda Chen et al., [*Reasoning Models Don't Always Say What They Think*](https://arxiv.org/abs/2505.05410), 2025，**[预印本/机构研究]**。特定 hint 设置中的 reasoning-model 披露实验；不外推为所有 CoT 性质。
- Tomek Korbak et al., [*Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety*](https://arxiv.org/abs/2507.11473), 2025，**[预印本/跨机构立场与研究综述]**。提出 monitorability 机会与脆弱性，不是安全保证。
- Artur Zolkowski et al., [*Can Reasoning Models Obfuscate Reasoning? Stress-Testing Chain-of-Thought Monitorability*](https://arxiv.org/abs/2510.19851), 2025，**[预印本]**。prompt-level obfuscation stress tests；结果依模型与压力设置。
- Melody Y. Guan et al., [*Monitoring Monitorability*](https://openreview.net/forum?id=b82fgbMVpz), ICML 2026 spotlight，**[同行评审]**。intervention、process、outcome-property 三类评估与 agent-monitor scaling。
- Yueh-Han Chen et al., [*Reasoning Models Struggle to Control Their Chains of Thought*](https://arxiv.org/abs/2603.05706), ICML 2026，**[同行评审]**。当前所测模型的 CoT controllability 与训练/推理条件；与 obfuscation 研究的协议不同。
- Austin Meek et al., [*Measuring Chain-of-Thought Monitorability Through Faithfulness and Verbosity*](https://arxiv.org/abs/2510.27378), 2025，**[预印本]**。区分 cue faithfulness 与关键因素覆盖；不提供完整内部过程真值。

## 评估基准

- Rohan Gupta et al., [*InterpBench: Semi-Synthetic Transformers for Evaluating Mechanistic Interpretability Techniques*](https://proceedings.neurips.cc/paper_files/paper/2024/hash/a8f7d43ae092d9a5295775eb17f3f4f7-Abstract-Datasets_and_Benchmarks_Track.html), NeurIPS 2024 Datasets and Benchmarks，**[同行评审]**。用 SIIT 构造半合成 known circuits。
- Mueller et al. 2025 MIB：用于跨 circuit/causal-variable localization 方法比较，已发表于 ICML 2025，属于 **[同行评审]** 结果。
- Karvonen et al. 2025 SAEBench 与 Chanin 2026 audit：分别代表 SAE 多指标基准与对其可靠性的最新审计；二者应并读。
- Gross et al. 2024：展示受控模型中从机制理解走向形式性能界的可能，也明确暴露结构外误差累积问题。

## 引用与外推纪律

1. “某论文发现”默认只指该论文的模型、任务、数据和干预协议。
2. preprint、机构报告和 workshop 结果可以更新研究地图，不用作无条件定理。
3. SAE feature、attribution graph、CoT monitor 和 causal probe 的新指标本身也需要 sanity checks 与 ground truth 审计。
4. 自动说明分数、probe 准确率、steering 成功、自然行为匹配与 intervention faithfulness 是不同量。
5. 截至 2026 年 7 月，feature 可识别性、全局 circuit、replacement-model 忠实性、跨规模机制稳定性和对抗条件下 CoT monitorability 仍是开放问题。
