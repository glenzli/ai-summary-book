# 卷三 模型中的概率从何而来

卷二展示概率在一次执行中的位置；本卷追问这些数值怎样形成。主线从世界与语言的非唯一性出发，经过数据采集、交叉熵训练、logits 与序列概率，再进入解码、训练运行差异、校准和实验分析。

本卷不是另一部概率论教材。正文只展开直接解释模型的概率概念，一般基础集中在短附录中。

0. [为什么确定的机器仍使用概率语言](ch00_why_probability.md)
1. [世界、数据与隐上下文](ch01_world_data_hidden_context.md)
2. [学习条件分布](ch02_learning_distributions.md)
3. [Logits、Token 概率与序列概率](ch03_logits_sequence_probability.md)
4. [解码怎样改变输出分布](ch04_decoding_transforms_distribution.md)
5. [训练随机性与模型间差异](ch05_training_randomness_model_variation.md)
6. [熵、校准与置信](ch06_entropy_calibration_confidence.md)
7. [怎样分析模型中的概率](ch07_model_probability_analysis.md)
8. [概率语言的边界](ch08_limits_probability_language.md)

辅助材料：[概率基础的最小工具箱](APPENDIX_PROBABILITY.md) · [符号与术语](REFERENCE.md) · [资料源](SOURCES.md)

本卷的完成标准不是让读者重新证明一般测度论，而是能回答：一个概率属于哪一层，它受哪些训练与运行变量影响，怎样通过实验估计，以及为什么它不能直接等同于真值或因果。
