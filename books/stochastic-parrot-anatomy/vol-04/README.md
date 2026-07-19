# 卷四 模型可解释性的研究路线

前两卷说明模型怎样建成、一次输出怎样执行；卷三解释概率怎样形成。本卷转向内部机制，但不假装可解释性已经是一门闭合理论。不同方法看到不同对象，也支持不同强度的结论。

本卷按当前研究路线展开：先定义行为和内部张量，再从相关读出走向显式干预、回路与稀疏特征，最后讨论训练动力学、推理过程和方法评估。

0. [可解释性究竟在研究什么](ch00_what_interpretability_studies.md)
1. [我们在模型内部看什么](ch01_internal_objects.md)
2. [行为与反事实：先确认模型做了什么](ch02_behavior_counterfactuals.md)
3. [梯度与输入归因](ch03_gradients_saliency.md)
4. [Attention、Residual 与 Logit Lens](ch04_attention_residual_readout.md)
5. [Probes 与表示几何](ch05_probes_representation_geometry.md)
6. [神经元、Feature 与自动解释](ch06_neurons_features_automated_explanations.md)
7. [Ablation、Patching 与因果追踪](ch07_interventions_patching.md)
8. [Circuits、路径与 Attribution Graphs](ch08_circuits_paths.md)
9. [Superposition、SAE 与稀疏表示](ch09_superposition_sparse_features.md)
10. [训练动力学与机制形成](ch10_training_dynamics.md)
11. [推理过程、Chain of Thought 与监测](ch11_reasoning_process_monitoring.md)
12. [评估、案例与开放问题](ch12_evaluation_case_studies_open_problems.md)

辅助材料：[符号与术语](GLOSSARY.md) · [资料源](SOURCES.md)

本卷的基本纪律是：activation pattern 不是概念证明，probe 不是使用证明，steering 不是正常机制证明，局部 circuit 也不是全局完整解剖。可信解释需要行为、预测和干预证据相互校验。
