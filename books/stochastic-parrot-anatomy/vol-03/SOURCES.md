# 资料源

本表服务于卷三的定义、方法与边界，不试图替代概率论、统计学或机器学习完整书目。一手论文链接已于 2026-07 核对；正文中的数学结论以给出的条件为准，论文中的经验结果不被外推为普遍定理。

## 概率、信息与评分规则

- Thomas M. Cover and Joy A. Thomas, *Elements of Information Theory*, 2nd ed., Wiley, 2006。用于熵、条件熵链式法则、交叉熵、KL 与互信息。
- Larry Wasserman, *All of Statistics*, Springer, 2004。用于期望、方差、Monte Carlo、区间与基础统计推断。
- Tilmann Gneiting and Adrian E. Raftery, [*Strictly Proper Scoring Rules, Prediction, and Estimation*](https://doi.org/10.1198/016214506000001437), JASA 2007。用于 proper/strictly proper score、log score 与 Brier score 的一般框架。
- Tilmann Gneiting, Fadoua Balabdaoui, and Adrian E. Raftery, [*Probabilistic Forecasts, Calibration and Sharpness*](https://doi.org/10.1111/j.1467-9868.2007.00587.x), JRSS B 2007。用于区分校准与信息分辨能力。
- Edwin B. Wilson, [*Probable Inference, the Law of Succession, and Statistical Inference*](https://doi.org/10.1080/01621459.1927.10502953), JASA 1927。用于二项比例 Wilson score 区间。
- Bradley Efron, [*Bootstrap Methods: Another Look at the Jackknife*](https://doi.org/10.1214/aos/1176344552), *Annals of Statistics* 1979。用于非参数 bootstrap 的原始方法来源；正文另行强调配对和层级必须按实验单位保留。
- Charles J. Geyer, [*Practical Markov Chain Monte Carlo*](https://doi.org/10.1214/ss/1177011137), *Statistical Science* 1992。用于相关 Monte Carlo 均值的积分自相关时间与有效样本量；正文只在平稳性和相关可和条件下使用该近似。

## 数据、分词与条件语言模型

- Yoshua Bengio et al., [*A Neural Probabilistic Language Model*](https://www.jmlr.org/papers/v3/bengio03a.html), JMLR 2003。用于神经条件语言模型和最大似然背景。
- Ashish Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762), 2017。用于自回归/条件序列建模的 Transformer 背景；架构细节主要见卷一。
- Taku Kudo and John Richardson, [*SentencePiece: A Simple and Language Independent Subword Tokenizer and Detokenizer for Neural Text Processing*](https://aclanthology.org/D18-2012/), EMNLP 2018。用于 tokenizer、detokenizer 与概率单位的工程背景。
- Rafael Rafailov et al., [*Direct Preference Optimization: Your Language Model is Secretly a Reward Model*](https://arxiv.org/abs/2305.18290), NeurIPS 2023。用于偏好对上的序列概率比目标；卷三不把该目标解释为唯一真实回答分布。

## 解码与诱导分布

- Ari Holtzman et al., [*The Curious Case of Neural Text Degeneration*](https://arxiv.org/abs/1904.09751), ICLR 2020。用于 nucleus sampling 与开放文本截断解码的经验背景。
- Matthew Finlayson et al., [*Closing the Curious Case of Neural Text Degeneration*](https://openreview.net/forum?id=dONpC9GL1o), ICLR 2024。用于截断采样机制与 softmax 分布误差的后续分析；正文只采用支持被删除与逐步重归一化等可直接验证结论。
- Jonathan Ho, Ajay Jain, and Pieter Abbeel, [*Denoising Diffusion Probabilistic Models*](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html), NeurIPS 2020。用于扩散模型反向随机转移的基本形式。

## 训练运行与模型间变异

- Alex Kendall and Yarin Gal, [*What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?*](https://proceedings.neurips.cc/paper_files/paper/2017/hash/2650d6089a6d640c5e85b2b88265dc2b-Abstract.html), NeurIPS 2017。用于 aleatoric/epistemic 术语背景；正文说明该二分不能替代随机算法与系统非确定性。
- Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell, [*Simple and Scalable Predictive Uncertainty Estimation Using Deep Ensembles*](https://proceedings.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html), NeurIPS 2017。用于 ensemble 预测分歧背景。
- Thibault Sellam et al., [*The MultiBERTs: BERT Reproductions for Robustness Analysis*](https://openreview.net/forum?id=K0E_F0gFDgA), ICLR 2022。用于多预训练 seed、工件结论与训练程序结论的区分，以及多层 bootstrap 设计。
- Yupei Du, Dong Nguyen, and Naoaki Okazaki, [*Measuring the Instability of Fine-Tuning*](https://aclanthology.org/2023.acl-long.342/), ACL 2023。用于微调运行间差异不能由单一标准差充分描述的经验背景。

## 校准与不确定性

- Chuan Guo et al., [*On Calibration of Modern Neural Networks*](https://proceedings.mlr.press/v70/guo17a.html), ICML 2017。用于 temperature scaling、reliability diagram 与现代神经分类器校准。
- Juozas Vaicenavicius et al., [*Evaluating Model Calibration in Classification*](https://proceedings.mlr.press/v89/vaicenavicius19a.html), AISTATS 2019。用于完整概率向量校准、最大置信校准的区别及校准评估细节。
- Rebecca Roelofs et al., [*Mitigating Bias in Calibration Error Estimation*](https://proceedings.mlr.press/v151/roelofs22a.html), AISTATS 2022。用于分箱 ECE 的有限样本偏差、等宽/等频分箱差异与估计器选择问题。
- Jason Hartline, Lunjia Hu, and Yifan Wu, [*A Perfectly Truthful Calibration Measure*](https://proceedings.mlr.press/v336/hartline26a.html), COLT 2026。用于区分“测量校准”与“以诚实概率报告为唯一期望最优”的指标激励；正文不展开其新指标，只保留 ECE 不是 strictly proper 训练目标这一边界。
- Yaniv Ovadia et al., [*Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty under Dataset Shift*](https://proceedings.neurips.cc/paper_files/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html), NeurIPS 2019。用于分布漂移下准确率与校准退化的经验基准。
- Zhengbao Jiang et al., [*How Can We Know When Language Models Know? On the Calibration of Language Models for Question Answering*](https://aclanthology.org/2021.tacl-1.57/), TACL 2021。用于生成式问答中候选概率和正确性校准的语言模型实例。
- Stephanie Lin, Jacob Hilton, and Owain Evans, [*Teaching Models to Express Their Uncertainty in Words*](https://arxiv.org/abs/2205.14334), TMLR 2022。用于自然语言自报置信需要外部任务检验的背景。
- Saurav Kadavath et al., [*Language Models (Mostly) Know What They Know*](https://arxiv.org/abs/2207.05221), 2022。用于模型预测自身答案正确性的实验路线；正文不把其经验发现外推为所有模型的保证。
- Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar, [*Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation*](https://openreview.net/forum?id=tWS-S_aRDRe), ICLR 2023。用于把不同字符串按语义等价类聚合的定义与采样近似。
- Sebastian Farquhar et al., [*Detecting Hallucinations in Large Language Models Using Semantic Entropy*](https://www.nature.com/articles/s41586-024-07421-0), *Nature* 2024。用于 semantic entropy 的扩展实验；正文明确保留采样、聚类与测量误差条件。

## 选择性预测与有限样本风险

- Yonatan Geifman and Ran El-Yaniv, [*SelectiveNet: A Deep Neural Network with an Integrated Reject Option*](https://proceedings.mlr.press/v97/geifman19a.html), ICML 2019。用于 coverage、selective risk 与 risk-coverage 评价。
- Stephen Bates et al., [*Distribution-Free, Risk-Controlling Prediction Sets*](https://doi.org/10.1145/3478535), JACM 2021。用于独立校准数据上有限样本风险控制的背景；正文不声称其保证在任意漂移或依赖请求下保持。

## 比较实验与分布漂移

- Philipp Koehn, [*Statistical Significance Tests for Machine Translation Evaluation*](https://aclanthology.org/W04-3250/), EMNLP 2004。用于按句子单位进行配对 bootstrap 的 NLP 早期实例。
- Rotem Dror et al., [*The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing*](https://aclanthology.org/P18-1128/), ACL 2018。用于根据实验设计和指标选择统计比较程序。
- Masashi Sugiyama, Matthias Krauledat, and Klaus-Robert Müller, [*Covariate Shift Adaptation by Importance Weighted Cross Validation*](https://www.jmlr.org/papers/v8/sugiyama07a.html), JMLR 2007。用于 covariate shift 条件、密度比加权和支持要求。

## 因果与解释边界

- Judea Pearl, *Causality*, 2nd ed., Cambridge University Press, 2009。用于观察条件 $P(Y\mid X)$ 与干预分布 $P(Y\mid do(X))$ 的区分。
- 模型 softmax、解码后频率、语义事件概率、世界事件概率和自报置信在本卷中始终是不同对象。上列来源都不支持在缺少事件映射、标签、校准与分布条件时将它们无条件等同。
