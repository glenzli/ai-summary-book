# 卷三 模型中的概率从何而来

卷二沿一次模型执行追踪 logits 和 token；本卷改问这些概率怎样形成、怎样被解码改变、怎样从有限实验估计，以及哪些结论无法由概率单独识别。主线是

$$
\text{世界与隐情境}
\to\text{观测和训练分布}
\to\text{学得条件分布}
\to\text{logits 与序列事件}
\to\text{解码输出与完整系统}
\to\text{统计估计与决策}.
$$

本卷不是另一部概率论教材。标准条件概率、期望、方差、Monte Carlo、区间与 KL 集中在[概率附录](APPENDIX_PROBABILITY.md)；正文只引入能直接约束模型概率解释的公式。

## 本卷完成后应能做到

- 区分数据条件内不确定性、模型/参数不确定性、随机算法和系统非确定性；
- 说明交叉熵何时学习数据条件分布，何时只得到受限模型族中的 KL 最优近似；
- 从 logits 正确计算赔率、序列概率、候选条件概率与语义事件质量；
- 把 temperature、top-k/top-p、约束、best-of-N 和过滤写成分布变换；
- 区分熵、proper score、校准、ECE、模型间分歧和选择性风险；
- 为黑盒采样、模型比较和多 checkpoint 实验选择正确重复单位与区间；
- 判断分布漂移、语义测量误差和支持缺口造成的不可识别结论。

## 章节

0. [为什么确定的机器仍使用概率语言](ch00_why_probability.md)：六层概率对象与四类变异。
1. [世界、数据与隐上下文](ch01_world_data_hidden_context.md)：选择过程、经验分布、隐变量混合与观测等价。
2. [学习条件分布](ch02_learning_distributions.md)：总体/经验风险、严格适当对数损失与 KL 投影。
3. [Logits、Token 概率与序列事件](ch03_logits_sequence_probability.md)：softmax 等价类、EOS、候选与语义推前分布。
4. [解码怎样诱导输出分布](ch04_decoding_transforms_distribution.md)：逐步随机核、截断、约束、搜索、过滤与扩散路径。
5. [训练随机性、参数变异与系统非确定性](ch05_training_randomness_model_variation.md)：训练运行分布、方差分解与复现层级。
6. [熵、Proper Score、校准与选择性预测](ch06_entropy_calibration_confidence.md)：校准层级、ECE 反例、semantic entropy 与 risk-coverage。
7. [模型概率的实验分析](ch07_model_probability_analysis.md)：estimand、Monte Carlo、相关样本、配对/层级 bootstrap 与漂移估计。
8. [概率分析的识别边界](ch08_limits_probability_language.md)：真值、语义、因果、分布外和反馈回路的边界。

辅助材料：[概率与估计的最小工具箱](APPENDIX_PROBABILITY.md) · [符号与术语](REFERENCE.md) · [资料源](SOURCES.md)

## 建议阅读路线

- 连续精读：0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8。
- 模型输出分析：0 → 3 → 4 → 6 → 7 → 8。
- 训练研究设计：0 → 1 → 2 → 5 → 7 → 8。
- 黑盒服务审计：0 → 4 → 6 → 7 → 8。

每章由定义、条件、证明或反例和可执行分析边界组成；读者应能把任一“模型概率”还原为明确事件、条件、系统层和估计程序，而不是只记住术语。
