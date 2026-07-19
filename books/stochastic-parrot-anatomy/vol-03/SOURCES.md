# 资料源

## 概率、信息与统计基础

- Thomas M. Cover and Joy A. Thomas, *Elements of Information Theory*, 2nd ed., Wiley, 2006。用于熵、交叉熵、KL、链式法则与信息论记号。
- Larry Wasserman, *All of Statistics*, Springer, 2004。用于有限样本估计、区间、bootstrap 与基础统计推断。
- Judea Pearl, *Causality*, 2nd ed., Cambridge University Press, 2009。用于观察条件与干预分布的区分。

## 语言建模与训练目标

- Yoshua Bengio et al., [*A Neural Probabilistic Language Model*](https://www.jmlr.org/papers/v3/bengio03a.html), JMLR 2003。用于神经条件语言模型的基本形式。
- Ashish Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762), 2017。用于 Transformer 条件序列建模背景。
- Rafael Rafailov et al., [*Direct Preference Optimization: Your Language Model is Secretly a Reward Model*](https://arxiv.org/abs/2305.18290), NeurIPS 2023。用于偏好对上的序列概率比目标。
- Chuan Guo et al., [*On Calibration of Modern Neural Networks*](https://arxiv.org/abs/1706.04599), ICML 2017。用于 temperature scaling、reliability diagram 与现代神经网络校准。

## 不确定性与语言模型分析

- Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell, [*Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*](https://arxiv.org/abs/1612.01474), NeurIPS 2017。用于模型间预测分歧和 ensemble 背景。
- Stephanie Lin, Jacob Hilton, and Owain Evans, [*Teaching Models to Express Their Uncertainty in Words*](https://arxiv.org/abs/2205.14334), TMLR 2022。用于自然语言置信表达需要外部校准的背景。
- Sebastian Farquhar et al., [*Detecting Hallucinations in Large Language Models Using Semantic Entropy*](https://www.nature.com/articles/s41586-024-07421-0), Nature 2024。用于把采样回答按语义簇聚合的不确定性方法；正文同时说明其采样与聚类假设。
- Ari Holtzman et al., [*The Curious Case of Neural Text Degeneration*](https://arxiv.org/abs/1904.09751), ICLR 2020。用于 nucleus sampling 与开放生成解码。

## 解释边界

- 模型 softmax、解码后频率、世界事件概率和自报置信在正文中被视为不同对象。上述来源不支持把任一对象无条件替换成另一个。
- 统计区间依赖采样与独立性假设；因果结论依赖干预或可辩护的识别假设；事实结论依赖模型概率之外的证据。
