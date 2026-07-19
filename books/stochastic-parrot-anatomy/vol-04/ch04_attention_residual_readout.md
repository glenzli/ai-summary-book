# 第四章 Attention、Residual 与 Logit Lens

attention 热图是 Transformer 最直观的内部图像，但“看向哪里”不是“为什么这样回答”的完整解释。本章把 attention pattern、value 内容、residual contribution 和中间层读出放在同一计算链中。

## 4.1 Attention Pattern 只给路由权重

head 输出为

$$
o_p=W_O\sum_j a_{p,j}W_Vx_j.
$$

即使 $a_{p,j}$ 很大，来源位置 $j$ 的 value 可能接近零，或写入的方向与目标 logit 无关。反之，较小权重乘以很大的 value 也可能重要。

所以解释至少应检查：

1. QK：为何目标位置选择这些来源；
2. OV：从来源读取了什么并写到哪个 residual 方向；
3. downstream：写入怎样影响后续层与 logits。

## 4.2 Attention 头的功能标签

研究者常称某 head 为 previous-token、induction、name-mover 或 delimiter head。这些名称是行为摘要，通常在一个输入分布上成立，而不是架构强制类型。

验证一个标签应测试：

- attention pattern 是否在新样本上出现；
- value 写入是否携带所声称的信息；
- ablation 是否损害相应行为；
- 其他 heads 是否能补偿；
- 不相关任务是否也依赖该 head。

一个 head 可以多功能；同一功能也可以分布在多个 heads。

## 4.3 Attention Rollout

跨层 rollout 把 attention matrices 与 residual identity 混合后相乘，试图估计输入 token 到高层位置的总体连接。它能显示潜在路由路径，但通常忽略 value 变换、MLP、非线性和 head 间抵消。

因此 rollout 是结构可达性或混合权重近似，不是输出贡献守恒分解。

## 4.4 Residual Decomposition

由于 residual addition，最终 state 可展开为初始 embedding 与各模块写入之和：

$$
x_L=x_0+
\sum_{\ell=0}^{L-1}
(a_\ell+m_\ell),
$$

其中 $a_\ell,m_\ell$ 分别为 attention 与 MLP 输出。若暂忽略最终 LayerNorm，每项可以投影到 unembedding 方向，得到 direct logit attribution。

这个分解精确描述向量相加，却不把 credit 唯一分给机制：早期模块可能通过改变后续模块输入产生主要间接效果，而其直接投影很小。

## 4.5 Logit Lens

logit lens 把中间 residual state 直接通过最终 normalization/unembedding，观察“若现在就输出，哪些 token 较高”：

$$
z_\ell^{lens}=W_U\operatorname{LN}_f(x_\ell).
$$

它常显示候选答案何时在 residual stream 中变得可线性读出。但中间 state 并未按最终层分布训练，直接应用最终 LayerNorm 和 unembedding 可能产生系统偏差。

## 4.6 Tuned Lens

tuned lens 为每层学习一个 affine translator $T_\ell$，再投影：

$$
z_\ell^{tuned}=W_U T_\ell(x_\ell).
$$

它可以改善中间预测质量，却引入额外训练模型。读出的信息可能由 translator 提取，而不是原模型在该层已经以最终输出坐标显式表示。应同时报告 translator 容量、训练数据和 baseline。

## 4.7 Logit Attribution 的正负抵消

某组件可直接提高答案 token logit，也可更强提高竞争 token。研究二选一行为时，最好投影到差向量

$$
u=W_{U,a}-W_{U,b},
$$

并计算 $u^\top c$。这比单看 $W_{U,a}^{\top}c$ 更对应决策。

组件间可相互抵消。仅展示正贡献会产生选择偏差，完整图应保留显著负贡献和抑制路径。

## 4.8 多模态 Attention 图

视觉语言模型的 cross-attention 可以映射文本位置到视觉 tokens，但视觉 token 可能覆盖 patch、region 或压缩后的混合表示。把权重插值到像素上形成热图，会引入空间平滑和分辨率假象。

若要声称模型依据某物体回答，应进一步遮挡区域、patch activation 或替换视觉 feature，并检查目标输出变化。热图对齐物体只是相关证据。

## 4.9 Attention 争论的正确落点

“attention 是/不是解释”过于笼统。更准确的是：

- attention weights 是模型真实计算的一部分；
- 单独权重通常不足以解释输出；
- 在明确 head、value 和下游目标时，它可成为机制解释的一部分；
- 替代 attention pattern 能否保持输出，取决于模型和允许的替代集合；
- 解释质量应由预测与干预测试决定，而非方法名称。

## 4.10 结论

attention 说明条件路由，residual decomposition 说明模块写入，logit lens 说明中间表示对输出词表的可读性。三者结合可以形成机制假说，但只有后续 probe 与 intervention 才能区分“可读出”“被使用”和“必要路径”。
