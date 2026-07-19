# 附录 A.10 Attention 的矩阵与反向传播

卷一第三章已经给出 Transformer block、multi-head/GQA、归一化、RoPE、MoE 与复杂度。本附录不重复整层架构，只逐步核对 scaled dot-product attention 的定义域、形状与完整反向传播。

## A.10.1 前向算子与可见集合

取单个样本、单个 attention head。令

$$
Q\in\mathbb R^{n_q\times d_h},
\qquad
K\in\mathbb R^{n_k\times d_h},
\qquad
V\in\mathbb R^{n_k\times d_v}.
$$

对每个 query 行 $i$，给定非空可见集合
$\mathcal V_i\subseteq\{1,\ldots,n_k\}$。定义

$$
S_{ij}=\frac{q_i^\mathsf Tk_j}{\sqrt{d_h}}
\quad (j\in\mathcal V_i),
\tag{A.10.1}
$$

$$
A_{ij}=
\begin{cases}
\dfrac{\exp(S_{ij}-m_i)}
{\sum_{r\in\mathcal V_i}\exp(S_{ir}-m_i)},
&j\in\mathcal V_i,\\[8pt]
0,&j\notin\mathcal V_i,
\end{cases}
$$

$$
m_i=\max_{j\in\mathcal V_i}S_{ij},
\tag{A.10.2}
$$

以及

$$
O=AV\in\mathbb R^{n_q\times d_v}.
\tag{A.10.3}
$$

$A\in\mathbb R^{n_q\times n_k}$ 的每行非负且和为 $1$，所以每个输出行是可见 value 行的凸组合。mask 是离散的模型定义，本附录不对其求导。若某个 $\mathcal V_i$ 为空，则 (A.10.2) 无定义；全遮蔽行不能靠普通 softmax 自动获得合理值。

Self-attention 有 $n_q=n_k$，Q/K/V 来自同一序列的不同投影。Cross-attention 可有 $n_q\ne n_k$；query 来自 decoder，key/value 来自 encoder。

## A.10.2 缩放因子的假设与结论

设某个 query/key 对的坐标满足：

$$
\mathbb E[q_r]=\mathbb E[k_r]=0,
\quad
\operatorname{Var}(q_r)=\sigma_q^2,
\quad
\operatorname{Var}(k_r)=\sigma_k^2,
$$

$q_r$ 与 $k_r$ 独立，并且乘积 $q_rk_r$ 在不同 $r$ 间互不相关。则

$$
\operatorname{Var}(q^\mathsf Tk)
=\sum_{r=1}^{d_h}
\operatorname{Var}(q_rk_r)
=d_h\sigma_q^2\sigma_k^2.
\tag{A.10.4}
$$

因此除以 $\sqrt{d_h}$ 后，初始化尺度模型下的方差不随 head dimension 线性增长。这里不需要正态分布，但需要上述零均值、独立/不相关条件。训练后的 Q/K 坐标一般不满足这些假设，所以 $1/\sqrt{d_h}$ 是尺度设计，不是“所有 attention logit 方差恒为一”的定理。

## A.10.3 Softmax 行的反向公式

设上游梯度

$$
G_O=\frac{\partial L}{\partial O}
\in\mathbb R^{n_q\times d_v}.
$$

由 $O=AV$ 的矩阵微分，

$$
\boxed{
G_V=A^\mathsf TG_O
\in\mathbb R^{n_k\times d_v}}
\tag{A.10.5}
$$

$$
\boxed{
G_A=G_OV^\mathsf T
\in\mathbb R^{n_q\times n_k}.}
\tag{A.10.6}
$$

对第 $i$ 行，把不可见坐标排除。由 softmax Jacobian
$\operatorname{diag}(a_i)-a_ia_i^\mathsf T$，令

$$
c_i=\sum_{r\in\mathcal V_i}A_{ir}(G_A)_{ir},
$$

则

$$
\boxed{
(G_S)_{ij}=
\begin{cases}
A_{ij}((G_A)_{ij}-c_i),&j\in\mathcal V_i,\\
0,&j\notin\mathcal V_i.
\end{cases}}
\tag{A.10.7}
$$

每行满足

$$
\sum_j(G_S)_{ij}=0,
$$

这对应 softmax 对整行常数平移不敏感。若某行 attention 已极度集中，Jacobian 的部分方向可能很小；残差路径和其他 head 仍可传递梯度，不能据此宣称整个 Transformer 梯度为零。

## A.10.4 Q、K、V 的完整反向

把不可见位置的 $G_S$ 定义为零后，(A.10.1) 可按完整矩阵写成

$$
S=\frac{QK^\mathsf T}{\sqrt{d_h}}
$$

并忽略常量 mask。矩阵微分给出

$$
\boxed{
G_Q=\frac{G_SK}{\sqrt{d_h}}
\in\mathbb R^{n_q\times d_h}}
\tag{A.10.8}
$$

$$
\boxed{
G_K=\frac{G_S^\mathsf TQ}{\sqrt{d_h}}
\in\mathbb R^{n_k\times d_h}.}
\tag{A.10.9}
$$

连同 (A.10.5)，这就是 attention core 对 Q/K/V 的全部梯度。忽略投影偏置，卷一采用行主权重约定：对 self-attention，若

$$
X\in\mathbb R^{n\times d},
\quad
W_Q,W_K\in\mathbb R^{d\times d_h},
\quad
W_V\in\mathbb R^{d\times d_v},
$$

$$
Q=XW_Q,
\qquad K=XW_K,
\qquad V=XW_V,
\tag{A.10.10}
$$

则

$$
G_{W_Q}=X^\mathsf TG_Q,
\quad
G_{W_K}=X^\mathsf TG_K,
\quad
G_{W_V}=X^\mathsf TG_V,
$$

$$
G_X=G_QW_Q^\mathsf T+G_KW_K^\mathsf T+G_VW_V^\mathsf T.
\tag{A.10.11}
$$

[A.6 的 batch affine 公式](a.6_backpropagation.md#a63-batch-行主公式)把权重写成 $d_{\mathrm{out}}\times d_{\mathrm{in}}$ 并在前向使用 $W^\mathsf T$；令其中的 $W$ 等于这里的 $W_Q^\mathsf T$ 即得到同一公式。两种记法只是权重存储朝向不同，不能在同一等式中混用。cross-attention 的 query 与 key/value 来自不同输入，此时只对实际共享的输入支路求和。

## A.10.5 Batch、多头与 GQA 的形状

卷一使用

$$
Q\in\mathbb R^{B\times H_q\times n_q\times d_h},
$$

$$
K\in\mathbb R^{B\times H_{\mathrm{kv}}\times n_k\times d_h},
\qquad
V\in\mathbb R^{B\times H_{\mathrm{kv}}\times n_k\times d_v}.
$$

标准 MHA 有 $H_{\mathrm{kv}}=H_q$；MQA 有 $H_{\mathrm{kv}}=1$；GQA 介于二者之间。每个 query head $r$ 使用映射 $g(r)$ 指定的 KV head，并独立应用 (A.10.1)--(A.10.9)。反向时，共享同一 KV head 的多个 query head 对 $G_K,G_V$ 的贡献必须求和。

各 head 输出 concat 后形状为
$B\times n_q\times(H_qd_v)$，再经过输出投影。reshape 和 transpose 本身不改变数值，只改变索引；反向必须执行其逆置换，不能把 head 轴与序列轴混合。

## A.10.6 因果性与复杂度边界

自回归 self-attention 取

$$
\mathcal V_i=\{j:j\le i\}
$$

并再去除 padding 位置。训练时可以并行计算所有行，因为未来位置在每一行的定义域外；并行执行不破坏因果依赖。

单个 head 的主要算术量为

$$
O(n_qn_kd_h+n_qn_kd_v),
$$

朴素实现还会物化 $O(n_qn_k)$ 的 $S,A$。FlashAttention 一类分块算法用在线 softmax 减少高带宽内存往返，并在反向中重算局部统计；在实数算术语义下它计算同一个精确 attention 函数，不把稠密 attention 的算术量改为线性。有限精度下，不同分块与归约顺序仍可产生舍入差异。

位置编码、RoPE、完整 pre-norm block 与参数账本见卷一第三章；重复这些内容不会增加本附录的推导价值。

## A.10.7 来源

- Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762), 2017。
- Dao et al., [*FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*](https://arxiv.org/abs/2205.14135), 2022。
