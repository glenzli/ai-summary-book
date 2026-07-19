# 资料源

本表优先列出方法原论文和公开研究报告。经验结论只在相应模型、任务和协议范围内使用；预印本或实验室报告不写成全模型定理。

## 归因、Attention 与 Probe

- Mukund Sundararajan, Ankur Taly, and Qiqi Yan, [*Axiomatic Attribution for Deep Networks*](https://proceedings.mlr.press/v70/sundararajan17a.html), ICML 2017。Integrated Gradients、completeness 及其公理来源。
- Julius Adebayo et al., [*Sanity Checks for Saliency Maps*](https://proceedings.neurips.cc/paper_files/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html), NeurIPS 2018。参数与标签随机化检查。
- Sarthak Jain and Byron C. Wallace, [*Attention is not Explanation*](https://aclanthology.org/N19-1357/), NAACL 2019；Sarah Wiegreffe and Yuval Pinter, [*Attention is not not Explanation*](https://aclanthology.org/D19-1002/), EMNLP 2019。用于界定 attention weights 的证据范围，而非给出普遍的二元裁决。
- John Hewitt and Percy Liang, [*Designing and Interpreting Probes with Control Tasks*](https://aclanthology.org/D19-1275/), EMNLP 2019。probe control task 与 selectivity。
- Yanai Elazar et al., [*Amnesic Probing*](https://aclanthology.org/2021.tacl-1.10/), TACL 2021。投影删除与行为测试。
- Nora Belrose et al., [*Eliciting Latent Predictions from Transformers with the Tuned Lens*](https://arxiv.org/abs/2303.08112), 2023。中间 residual 表示的学习型词表读出。

## 干预与回路

- Nelson Elhage et al., [*A Mathematical Framework for Transformer Circuits*](https://transformer-circuits.pub/2021/framework/index.html), 2021。QK/OV 分解与 residual 路径框架。
- Kevin Wang et al., [*Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small*](https://arxiv.org/abs/2211.00593), ICLR 2023。IOI circuit 及 faithfulness、completeness、minimality 实例。
- Kevin Meng et al., [*Locating and Editing Factual Associations in GPT*](https://arxiv.org/abs/2202.05262), NeurIPS 2022。事实回忆上的 causal tracing 与模型编辑；不支持“事实唯一存储在单层”。
- Fred Zhang and Neel Nanda, [*Towards Best Practices of Activation Patching in Language Models*](https://arxiv.org/abs/2309.16042), ICLR 2024。patch metric 与 corruption 选择敏感性。
- Arthur Conmy et al., [*Towards Automated Circuit Discovery for Mechanistic Interpretability*](https://arxiv.org/abs/2304.14997), NeurIPS 2023。自动 edge 搜索及其任务范围。
- Emmanuel Ameisen et al., [*Circuit Tracing: Revealing Computational Graphs in Language Models*](https://transformer-circuits.pub/2025/attribution-graphs/methods.html), 2025。cross-layer transcoder、local replacement model、attribution graph、剪枝与干预验证；该工作明确保留 reconstruction error 和 attention pattern 限制。
- Harish Kamath et al., [*Tracing Attention Computation Through Feature Interactions*](https://transformer-circuits.pub/2025/attention-qk/index.html), 2025。把 feature interactions 扩展到 attention QK 路径的研究路线。

## 神经元、Superposition 与稀疏表示

- Steven Bills et al., [*Language Models Can Explain Neurons in Language Models*](https://openai.com/index/language-models-can-explain-neurons-in-language-models/), OpenAI 2023。自动生成并模拟评分 neuron 说明；作者报告多数说明分数仍低且不解释 downstream mechanism。
- Nelson Elhage et al., [*Toy Models of Superposition*](https://transformer-circuits.pub/2022/toy_model/index.html), 2022。稀疏特征超额表示的 toy model。
- Trenton Bricken et al., [*Towards Monosemanticity*](https://transformer-circuits.pub/2023/monosemantic-features/index.html), 2023；Adly Templeton et al., [*Scaling Monosemanticity*](https://transformer-circuits.pub/2024/scaling-monosemanticity/), 2024。SAE feature discovery 与规模化实例。
- Robert Huben et al., [*Sparse Autoencoders Find Highly Interpretable Features in Language Models*](https://openreview.net/forum?id=F76bwRSLeK), ICLR 2024。小语言模型上的 SAE 评估。
- David Chanin et al., [*A is for Absorption*](https://openreview.net/forum?id=R73ybUciQF), NeurIPS 2025；Aleksandar Makelov, Georg Lange, and Neel Nanda, [*Towards Principled Evaluations of Sparse Autoencoders for Interpretability and Control*](https://openreview.net/forum?id=1Njl73JKjB), ICLR 2025。feature splitting/absorption 与任务级 SAE 控制评估。
- Leo Gao et al., [*Weight-sparse Transformers Have Interpretable Circuits*](https://arxiv.org/abs/2511.13653), 2025。训练时 weight sparsity、任务回路剪枝以及能力—可解释性 frontier；结果限于较小代码模型与受控任务。

## 训练动力学、推理与评估

- Neel Nanda et al., [*Progress Measures for Grokking via Mechanistic Interpretability*](https://arxiv.org/abs/2301.05217), ICLR 2023。受控算法任务中的训练阶段与回路形成。
- Miles Turpin et al., [*Language Models Don't Always Say What They Think*](https://arxiv.org/abs/2305.04388), NeurIPS 2023；Tamera Lanham et al., [*Measuring Faithfulness in Chain-of-Thought Reasoning*](https://arxiv.org/abs/2307.13702), 2023。CoT 反事实忠实性实验。
- Yanda Chen et al., [*Reasoning Models Don't Always Say What They Think*](https://arxiv.org/abs/2505.05410), 2025。特定提示线索设置中的推理模型披露实验，不外推为所有 CoT 的性质。
- Rohan Gupta et al., [*InterpBench: Semi-Synthetic Transformers for Evaluating Mechanistic Interpretability Techniques*](https://proceedings.neurips.cc/paper_files/paper/2024/hash/a8f7d43ae092d9a5295775eb17f3f4f7-Abstract-Datasets_and_Benchmarks_Track.html), NeurIPS 2024。具有半合成 ground truth 的 circuit benchmark。

## 引用边界

- 本卷未声称 SAE features、attribution graphs 或 sparse circuits 已给出大型模型的完整解剖。
- 自动说明分数、probe 准确率、steering 成功与机制忠实性是不同指标。
- 截至 2026 年中，feature 定义、全局 circuit、替代模型忠实性、CoT 监测稳健性和规模化评估仍是开放问题。
