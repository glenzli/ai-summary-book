# 第二章 Prefill：一次 Transformer 前向

Prefill 接收完整输入 token 序列，一次计算所有输入位置的隐藏状态，并为后续解码建立 KV cache。它是用户按下发送后、首 token 出现前最主要的模型计算阶段。

## 2.1 输入形状

设 batch size 为 $B$，输入长度为 $n$，模型宽度为 $d$。embedding 后张量形状为

$$
X_0\in\mathbb R^{B\times n\times d}.
$$

为简化说明，下面省略 batch 维。每一行对应一个 token 位置，但它会在层间不断吸收其他可见位置的信息。

## 2.2 第一层发生什么

对 pre-norm decoder 层，可以把主要计算写成

$$
U_\ell=X_\ell+
\operatorname{Attn}_\ell(\operatorname{Norm}(X_\ell)),
$$

$$
X_{\ell+1}=U_\ell+
\operatorname{MLP}_\ell(\operatorname{Norm}(U_\ell)).
$$

attention 在位置之间交换信息，MLP 在每个位置的特征维度上变换。残差连接让层更新累加到同一主表示。

## 2.3 Causal Attention 的矩阵

每层从规范化输入得到 query、key 和 value。对长度 $n$ 的序列，attention score 具有 $n\times n$ 的位置矩阵。因果掩码把右上三角区域屏蔽，位置 $i$ 只能读取 $1$ 到 $i$。

虽然最终回答尚未生成，prefill 中每个输入位置的计算可以并行完成，因为所有输入 token 已知。所谓“自回归一定逐 token 训练”是不准确的：训练和 prefill 可以并行，只有未知输出必须按生成顺序确定。

## 2.4 信息没有变成一句摘要

模型不会先把整段 prompt 压成一个自然语言摘要再回答。每个位置都保持一个 $d$ 维向量，各层以分布式方式更新。最后输入位置通常承担预测首个输出 token 的直接读出，但它的表示已经通过 attention 汇集可见前缀。

这也解释了为什么仅观察某个神经元或注意力头很难概括整个运行：输出来自多层、多个位置和残差通道的组合。卷四将讨论怎样观察这些中间量。

## 2.5 KV Cache 的建立

每层会保存输入位置的 key 和 value：

$$
K_\ell^{1:n},\qquad V_\ell^{1:n}.
$$

后续生成新 token 时，不需要重新计算旧位置的 K/V。缓存内容依赖模型、token 序列、位置和精度；它不是可以在任意模型之间通用的文本摘要。

## 2.6 位置与长度

RoPE 等位置机制在构造 query/key 时加入位置相关旋转。位置编号通常从输入开头连续增长；前缀缓存复用和截断都必须保持位置约定一致。

输入越长，稠密 attention 的 prefill 计算和中间矩阵越大。具体实现可以用 FlashAttention、稀疏注意力或分块减少内存访问，但这些优化不改变本章的逻辑数据流。

## 2.7 固定前向是否唯一

在抽象数学模型中，固定参数和输入给出固定张量。实际机器还涉及浮点精度、并行归约、kernel 和硬件。通常这些差异很小，但在接近决策边界时可能改变最终 token 排序。

本卷不发展一般浮点误差理论。理解一次运行只需记住：权重与 token 相同是必要信息，却不总是逐位复现的全部条件。

## 2.8 最后一层

经过 $L$ 层后，得到 $X_L\in\mathbb R^{n\times d}$。取最后一个可预测位置的表示 $h$，经过最终 normalization 和词表投影，得到下一章的 logits。

Prefill 到此结束。服务系统已经建立缓存，但尚未选出任何新 token。
