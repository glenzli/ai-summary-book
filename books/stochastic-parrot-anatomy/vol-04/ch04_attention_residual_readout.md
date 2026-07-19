# 第四章 Attention、Residual 与 Logit Lens

attention 热图是 Transformer 最直观的内部图像，但“向哪里分配权重”不是“哪个输入导致输出”的完整答案。本章只研究模型已经计算出的路由、写入和线性读出，不把它们提前升级为因果回路。

## 4.1 三个不同研究问题

对目标位置 $p$ 的 head $h$，区分：

1. **routing**：哪些 source positions $j$ 获得权重；
2. **content**：这些位置的 value 携带什么；
3. **effect**：head result 怎样直接或间接改变目标 $S$。

head 输出为

$$
o_p^h=\sum_j c_{p\leftarrow j}^{h},
\qquad
c_{p\leftarrow j}^{h}
=A_{p,j}^{h}W_O^hW_V^h r_j.
$$

$A_{p,j}^h$ 只属于第一问，$c_{p\leftarrow j}^h$ 同时包含路由与线性 value 写入，完整 effect 还要经过后续非线性。

## 4.2 Attention pattern 的正确估计量

研究 pattern 时，常用：

- 对目标 source set $J$ 的 mass $\sum_{j\in J}A_{p,j}^h$；
- pattern entropy $-\sum_jA_{p,j}^h\log A_{p,j}^h$；
- argmax source 的命中率；
- 相对位置或语法关系条件下的平均权重；
- paired input 前后的 pattern 差。

平均权重要按有效序列长度和 mask 对齐。把 padding、BOS 或 delimiter 的高权重混入实体平均，会产生位置 shortcut。跨样本汇总还应报告分布而不是只画平均 heatmap。

高权重 source 的 $W_Vr_j$ 可能很小，或经 $W_O$ 写入与目标无关方向。相反，小权重乘大 value 仍可重要。

## 4.3 QK 与 OV 分解

若 query/key 投影后没有位置相关变换，并忽略 bias，score 可写为

$$
s_{p,j}^h
=r_p^\top (W_Q^h)^\top W_K^h r_j/\sqrt{d_h}.
$$

采用 RoPE 时，若 $q_p=R_pW_Q^hr_p$、$k_j=R_jW_K^hr_j$，则应写成

$$
s_{p,j}^h
=r_p^\top(W_Q^h)^\top R_p^\top R_jW_K^hr_j/\sqrt{d_h}.
$$

其他相对位置机制还可能加入单独的 $b_{p,j}$ 或改变 kernel。因而固定矩阵 $(W_Q^h)^\top W_K^h$ 只在无位置变换的简化模型中定义完整 source-selection 双线性形式；$W_O^hW_V^h$ 则定义读取后写入 residual 的线性变换。这一 QK/OV 分解有助于提出问题：

- query 依赖哪个上游 feature；
- key 怎样标记候选位置；
- OV 是否复制 token、抑制候选或写入关系信息；
- 哪个下游组件读取该写入。

权重矩阵分析是全局潜能，实际 activation 分析是输入条件下的实现。仅凭 QK/OV 的 singular vectors 不能说明自然语料上确实启用。

## 4.4 Source contribution 与 norm 陷阱

可以计算 source contribution 的 norm

$$
\|c_{p\leftarrow j}^h\|_2
$$

或对目标 unembedding difference $u$ 的未归一化直接投影

$$
\widetilde d_{p\leftarrow j}^h
=u^\top c_{p\leftarrow j}^h.
$$

若最终读出含 $\operatorname{Norm}_f$，在当前最终 residual $x_{L,p}$ 处的一阶 DLA 应为

$$
d_{p\leftarrow j}^{h,(1)}
=u^\top J_{\operatorname{Norm}_f}(x_{L,p})
c_{p\leftarrow j}^h.
$$

只有忽略 normalization，或已把冻结 normalization 的线性部分吸收到 $u$ 中时，才可把 $\widetilde d$ 直接称为 DLA。norm 指标不关心方向，一阶 DLA 只关心当前点的直接 logit 方向。大 norm 可能被后续抑制；小直接投影可能通过改变下游 attention 产生大间接效应。二者不能互称“真实贡献”。

LayerNorm/RMSNorm 还会耦合 residual 坐标。若把 source contributions 分别归一化再求和，通常不等于对总和归一化后的 logit，应明确采用冻结 norm、局部线性化还是实际重跑。

## 4.5 功能标签是一条分布主张

previous-token、induction、name-mover 或 delimiter head 是行为摘要，不是架构类型。验证标签至少需要：

- 在新模板、新实体和失败样本上测 pattern；
- 检查 value 写入与所称功能一致；
- 预测该 head 的输出或目标 logit effect；
- 用 matched heads、位置和权重熵作基线；
- 通过 ablation/patching 检查行为效应；
- 测试其他任务是否也使用该 head。

同一 head 可多功能，同一功能可由多个 heads 冗余实现。标签应带输入条件，如“在分布 $\mathcal D_{\mathrm{eval}}$ 的答案位置执行 name-copying”，而不是“这是名字神经元”。

## 4.6 Attention rollout 与 flow

rollout 常把每层平均 attention 与 identity 混合，例如

$$
\widetilde A^{\ell}=\alpha A^{\ell}+(1-\alpha)I,
\qquad
R^{L\leftarrow0}
=\widetilde A^{L-1}\cdots\widetilde A^0.
$$

它估计在只保留 attention mixing 的近似图中，输入位置到高层位置的可达权重。它通常忽略：

- value 与 output projections；
- head 间符号和方向差异；
- MLP 路径；
- normalization 与门控；
- attention pattern 对 activation 的输入依赖。

所以 rollout 是结构流近似。它可与删除或 gradient 指标相关，但这种相关不把它变成输出贡献守恒分解。

## 4.7 Residual decomposition

最终 residual 的加法分解为

$$
x_L=x_0+
\sum_{\ell=0}^{L-1}
\sum_h o_{\ell}^{h}
+\sum_{\ell=0}^{L-1}m_{\ell}.
$$

这允许把各 module result 投影到某一 output direction。对 pre-norm 模型，module result 是自然加法单位；对并行 block、post-norm 或混合架构，需要按实际计算图重写。

分解回答“最终向量由哪些加数构成”，不唯一回答“哪些组件应获得因果 credit”。若早期 head 只为后期 MLP 准备 feature，其 direct projection 可以接近零。

## 4.8 Logit lens

logit lens 把中间 residual state 直接经最终 norm 和 unembedding：

$$
z_{\ell}^{lens}=W_U\operatorname{Norm}_f(x_\ell)+b_U.
$$

它估计“若用最终读出器读取当前 state，什么 token 可线性读出”。可报告：

- 目标 token rank、logit 或 logit difference；
- 与最终 next-token distribution 的 KL divergence；
- 层间候选排序的变化；
- 在原模型成功/失败样本上的差异。

中间层没有被要求处于最终 residual 分布。读出较差可能是坐标错位，不表示信息不存在；读出较好也不证明此 token 已被下游采用。

## 4.9 Tuned lens 与读出器容量

tuned lens 为每层学习 translator $T_\ell$：

$$
z_\ell^{tuned}=W_UT_\ell(x_\ell)+b_U.
$$

它可以更准确预测最终 distribution，却引入辅助模型。必须报告：

- $T_\ell$ 是 affine、低秩还是非线性；
- 训练语料、目标与正则化；
- 相对 identity、均值和随机 translator 的增益；
- held-out domain 与 token positions；
- translator 是否只校正层间尺度/旋转，还是执行复杂计算。

读出器越强，结果越接近“信息可被外部模型提取”，越远离“原模型已显式形成该预测”。

## 4.10 Attention 的非唯一读出

对固定 values，可能存在多个 pattern $A'$ 使

$$
\sum_jA'_jv_j\approx\sum_jA_jv_j.
$$

这说明仅从输出不总能反推唯一 attention pattern；但原模型实际计算出的 $A$ 仍是计算图中的真实变量。正确问题不是“attention 是不是解释”，而是：当前主张需要 pattern 的真实性、输出敏感性，还是完整机制？

替代 pattern 实验还必须满足 simplex、mask 和自然 score 约束。任意优化出的 $A'$ 可能不是任何合法 query/key activation 会产生的 pattern。

## 4.11 从观察到因果验证

候选 head/route 的最低验证链是：

1. pattern 在 held-out 输入上满足预注册关系；
2. source contribution 与目标内容相符；
3. head result 预测目标 $S$ 的变化；
4. 替换 source token 或 key feature 后，pattern 按假说改变；
5. patch/ablate head 或 edge 后，行为 effect 符合预测；
6. matched random head、同层其他 head 与位置基线较弱；
7. 副作用和替代路径被报告。

前四步产生机制候选，第五步才加入显式内部干预。

## 4.12 多模态 readout 的边界

cross-attention 权重可映射文本到视觉 tokens，但视觉 token 可能覆盖 patch、region 或压缩混合表示。插值到像素的热图增加了视觉平滑，不能当作像素级因果 mask。

若声称模型依据某物体回答，应联合使用区域反事实、视觉 feature patch 与答案效应。attention 对齐物体只支持路由共现。

## 4.13 方法审计表

| 方法 | 问题/对象 | 操作与估计量 | 必要控制 | 能支持 | 不能支持与失效 |
|---|---|---|---|---|---|
| pattern analysis | head 路由到哪里 | 观察 $A$；mass/熵/关系命中 | padding、位置、matched heads | 条件路由规律 | value 内容、输出原因 |
| source contribution | 路由后写入什么 | $A_{pj}W_OW_Vr_j$；norm；$u^\top J_{\mathrm{Norm}_f}(x_L)c$ | final norm 约定、正负 effect | 局部一阶线性写入 | 后续完整效应 |
| rollout/flow | 跨层位置连接如何累积 | attention 矩阵乘积 | residual 权重、ablation 比较 | 简化图中的可达性 | 全模型信息流 |
| residual DLA | 哪些加数直接对齐输出 | 加法分解与 $u^\top J_{\mathrm{Norm}_f}(x_L)c$ | 线性化点、竞争 token | 局部一阶直接读出 | 间接因果 credit |
| logit lens | 中间层能否被最终头读取 | final norm + unembedding | 成败样本、层分布偏移 | 最终坐标下可读性 | 下游实际使用 |
| tuned lens | 简单 translator 能否预测终态 | 训练 $T_\ell$；KL/accuracy | 容量、数据、identity baseline | 指定读出器可解码性 | 原模型已有显式 logits |

attention、residual decomposition 和 lens 都是原模型内部的真实读出面，但支持的是不同强度的可读性与路由结论。它们最适合生成候选路径；关于“被使用”与“必要”的结论留给干预章节。
