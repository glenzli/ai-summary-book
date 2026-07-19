# 学习与模型数学讲义

这些讲义是卷一的公式核验层。每篇都说明假设、导出结论并标记不能外推的边界；正文负责技术叙事，附录不重复整章介绍。默认使用自然对数、欧氏内积；标量损失对向量的梯度写成列向量，batch/token 在矩阵公式中通常放在行轴。

## 条目与先修关系

| 条目 | 最小先修 | 何时使用 | 不负责什么 |
| --- | --- | --- | --- |
| [A.1 优化基础](a.1_optimization_basics.md) | 多元微积分、期望 | 核对下降引理、SGD 无偏性与强凸收敛条件 | 非凸深度网络的全局收敛 |
| [A.2 感知机收敛](a.2_perceptron_convergence.md) | 向量内积、Cauchy--Schwarz | 核对 Novikoff 错误次数界 | 不可分数据与最大间隔学习 |
| [A.3 统计学习](a.3_statistical_learning_theory.md) | 初等概率、组合计数 | 核对偏差--方差、VC 维、Sauer 引理与 ERM 界 | 任意生成任务的泛化理论 |
| [A.4 正则化与 MAP](a.4_regularization.md) | A.1、Bayes 公式 | 区分约束/惩罚、MAP/Bayesian、L2/decay 与 dropout | 把单个 checkpoint 解释成后验分布 |
| [A.5 通用逼近](a.5_universal_approximation.md) | 连续函数、紧致性；一般证明需泛函分析 | 核对 Cybenko/Leshno 假设及一维 ReLU 构造 | 优化、宽度、样本量与泛化保证 |
| [A.6 反向传播](a.6_backpropagation.md) | 矩阵乘法、链式法则 | 核对单样本和 batch affine 梯度形状 | 卷积与 attention 的专用索引 |
| [A.7 Softmax 与交叉熵](a.7_softmax_crossentropy.md) | 微分、概率单纯形 | 核对梯度、Hessian、token mask 与 log-sum-exp | 概率校准和世界不确定性 |
| [A.8 AdamW](a.8_advanced_optimization.md) | A.1、A.4 | 对照 Momentum/AdaGrad/RMSProp/Adam/AdamW 更新式 | 仅凭优化器名称推出收敛或优劣 |
| [A.9 CNN 反向传播](a.9_cnn_backpropagation.md) | A.6、离散索引 | 核对 bias/kernel/input 梯度与 transposed convolution | 把伴随算子当作卷积逆运算 |
| [A.10 Attention 反向传播](a.10_transformer_math.md) | A.6、A.7 | 核对 masked attention 的 Q/K/V 梯度和多头共享 | 重复 Transformer 整层架构与 RoPE |
| [A.11 PPO 与 DPO](a.11_rl_and_ppo.md) | 条件期望、KL、A.7 | 核对策略梯度、GAE、PPO clipping 与 DPO 推导 | 任意偏好模型之间的无条件等价 |
| [A.12 InfoNCE](a.12_contrastive_learning.md) | A.7、KL 与互信息 | 核对双向损失、表示梯度和 MI 下界采样条件 | 把相似度直接解释为事实概率 |

## 阅读约定

- 定义、命题和定理中的假设属于结论的一部分；正文简写不能覆盖附录的适用边界。
- 算法式描述一个离散更新，不自动构成收敛定理。出现保证时，本文会单列假设或给出反例。
- `loss mask` 决定哪些位置计入目标；`attention mask` 决定前向可见集合，两者不可互换。
- A.4 中的参数先验、A.11 中的参考策略 $\pi_{\mathrm{ref}}$ 与 PPO 旧策略 $\pi_{\mathrm{old}}$ 是不同对象，应以各篇定义为准。
- 主要概率记号可查阅卷三的[概率工具箱](../../vol-03/APPENDIX_PROBABILITY.md)；概率校准、模型不确定性和可解释性方法仍分别以卷三、卷四正文为准。

本目录的用途是让读者从正文返回可核验的公式、证明和来源，同时保持卷一的技术叙事连续。
