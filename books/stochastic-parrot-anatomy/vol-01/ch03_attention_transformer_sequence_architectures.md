# 第三章 注意力与 Transformer

循环网络按时间步更新状态，长距离信息必须穿过一条连续路径。注意力改用内容相关的读取：当前位置不只继承上一状态，而是直接计算它与其他位置的关系。Transformer 再把这种读取变成主要序列算子，使训练可以在位置维度上并行。

本章回答结构问题：一层 Transformer 实际计算什么，张量形状怎样变化，位置信息和掩码怎样进入。服务优化留给第六章，单次自回归运行留给卷二。

## 3.1 记号与内容寻址

设 batch 大小为 $B$，序列长度为 $n$，模型宽度为 $d$。一批输入表示为

$$
X\in\mathbb R^{B\times n\times d}.
$$

不涉及 batch 交互时，下文省略第一维，把单条序列写成

$$
X=(x_1,\ldots,x_n)^\top\in\mathbb R^{n\times d}.
$$

注意力为每个位置构造 query，并把它与所有可见位置的 key 比较，再对对应 value 加权。它是可微的内容寻址，不是数据库中的精确查找：输出通常是多个 value 的线性组合，寻址规则也由数据学习。

最初的 encoder-decoder 注意力用于机器翻译：解码器在每一步读取编码器的不同位置。self-attention 则让 query、key 和 value 来自同一序列。

## 3.2 单头缩放点积注意力

取每头 query/key 维数 $d_h$，value 维数 $d_v$。由输入线性投影得到

$$
Q=XW_Q\in\mathbb R^{n\times d_h},\qquad
K=XW_K\in\mathbb R^{n\times d_h},\qquad
V=XW_V\in\mathbb R^{n\times d_v}.
$$

第 $i$ 个 query 对第 $j$ 个 key 的分数、权重和输出分别为

$$
s_{ij}=\frac{q_i^\top k_j}{\sqrt{d_h}}+M_{ij},
\qquad
a_{ij}=\frac{e^{s_{ij}}}{\sum_{r=1}^{n}e^{s_{ir}}},
\qquad
z_i=\sum_{j=1}^{n}a_{ij}v_j.
$$

矩阵形式为

$$
\operatorname{Attn}(Q,K,V)
=
\operatorname{softmax}_{\mathrm{row}}
\left(\frac{QK^\top}{\sqrt{d_h}}+M\right)V.
$$

$QK^\top\in\mathbb R^{n\times n}$ 是位置关系矩阵，输出 $Z\in\mathbb R^{n\times d_v}$。若某一行至少有一个可见位置，则 $a_{ij}\ge 0$ 且 $\sum_j a_{ij}=1$，所以该行输出位于 value 行向量的凸包中。后续输出投影和残差相加会打破这一凸组合限制。

### 3.2.1 为什么除以 $\sqrt{d_h}$

作为尺度分析，假设每个 $q_{ir}$ 与对应的 $k_{jr}$ 独立、均值为 $0$、方差为 $1$，并假设不同坐标的乘积 $q_{ir}k_{jr}$ 两两不相关，则

$$
\operatorname{Var}(q_i^\top k_j)
=\sum_{r=1}^{d_h}\operatorname{Var}(q_{ir}k_{jr})
=d_h.
$$

除以 $\sqrt{d_h}$ 后，分数方差保持在常数量级，softmax 不会仅因头维数增加而更容易饱和。训练后的分量并不满足上述独立同分布假设；这个推导解释的是初始化尺度，而不是对任意已训练模型分数分布的定理。

### 3.2.2 数值稳定的 masked softmax

实现中不会真的存储 IEEE 的 $-\infty$ 再盲目计算。对第 $i$ 行可见集合 $\mathcal V_i$，稳定写法是

$$
m_i=\max_{j\in\mathcal V_i}s_{ij},
\qquad
a_{ij}=
\begin{cases}
\dfrac{e^{s_{ij}-m_i}}
{\sum_{r\in\mathcal V_i}e^{s_{ir}-m_i}},&j\in\mathcal V_i,\\[8pt]
0,&j\notin\mathcal V_i.
\end{cases}
$$

每一行必须至少保留一个可见位置；否则分母为零，常见张量实现会产生 `NaN`。padding mask、因果 mask 与局部窗口 mask 应在定义可见集合时合并，而不是在 softmax 后再把权重清零，因为后者不再保证行和为 $1$。

注意力权重只是当前前向计算中的混合系数。它可以帮助观察信息汇聚，却不自动等于语义重要性或因果解释；这一区分在卷四展开。

![注意力权重示例](chapter_03/images/attention_alignment_heatmap.png)

## 3.3 因果掩码与 padding 掩码

自回归语言模型在位置 $i$ 只允许读取 $j\le i$ 的位置：

$$
M^{\mathrm{causal}}_{ij}=
\begin{cases}
0,&j\le i,\\
-\infty,&j>i.
\end{cases}
$$

若第 $j$ 个位置是 padding，还需令所有有效 query 都不能读取它。常见组合是

$$
M=M^{\mathrm{causal}}+M^{\mathrm{pad}},
$$

其中任何一项禁止的位置最终都不可见。训练可以同时计算所有位置的损失，因为掩码在矩阵内部阻止未来信息泄漏；生成仍需等上一个 token 确定后再构造下一步输入。训练并行和生成串行并不矛盾。

![因果掩码如何阻止读取未来位置](chapter_03/images/causal_mask_demo.png)

BERT 一类双向编码器通常没有因果掩码，但仍需处理 padding；encoder-decoder 模型在 encoder 中双向读取，在 decoder 中使用因果 self-attention，并通过 cross-attention 读取 encoder 输出。

## 3.4 多头、Multi-Query 与 Grouped-Query Attention

设 query 头数为 $H_q$，KV 头数为 $H_{kv}$，通常令 $H_qd_h=d$，并假设 key 与 value 的每头维数都是 $d_h$。投影后 reshape 为

$$
Q\in\mathbb R^{B\times H_q\times n\times d_h},
\qquad
K,V\in\mathbb R^{B\times H_{kv}\times n\times d_h}.
$$

令映射 $g:\{1,\ldots,H_q\}\to\{1,\ldots,H_{kv}\}$ 指定每个 query 头使用哪一组 KV，则

$$
Z^{(r)}=
\operatorname{Attn}
\left(Q^{(r)},K^{(g(r))},V^{(g(r))}\right),
$$

$$
\operatorname{MHA}(X)
=\operatorname{Concat}
\left(Z^{(1)},\ldots,Z^{(H_q)}\right)W_O.
$$

三种常见情形只是 $g$ 的不同选择：

| 结构 | $H_{kv}$ | KV 共享方式 |
| --- | ---: | --- |
| Multi-Head Attention | $H_q$ | 每个 query 头有独立 KV 头 |
| Grouped-Query Attention | $1<H_{kv}<H_q$ | 一组 query 头共享一个 KV 头 |
| Multi-Query Attention | $1$ | 全部 query 头共享同一 KV 头 |

忽略 bias，注意力投影的参数量为

$$
P_{\mathrm{attn}}
=2dH_qd_h+2dH_{kv}d_h.
$$

当 $H_qd_h=d$ 时，它等于 $2d^2+2dH_{kv}d_h$；标准 MHA 再令 $H_{kv}=H_q$，得到 $4d^2$。减少 KV 头既减少 $K,V$ 投影参数，也按比例减少第六章所述的 KV cache；query 数和输出宽度并未因此缩小。

朴素实现会物化形状为 $B\times H_q\times n\times n$ 的分数或概率张量，注意力算术量为 $O(Bn^2d)$。FlashAttention 一类算法通过分块和在线 softmax 减少高带宽内存读写，不改变精确 attention 的数学函数，也不把其算术复杂度改成线性。

不同头可以学习不同位置和内容关系，但“一个头对应一个人类概念”不是结构保证。头的功能取决于层、输入分布以及与其他组件的组合。

MQA、GQA 与 IO-aware 精确 attention 的来源分别见 [Shazeer, 2019](SOURCE_NOTES.md#ref-shazeer-mqa-2019)、[Ainslie et al., 2023](SOURCE_NOTES.md#ref-ainslie-gqa-2023) 和 [Dao et al., 2022](SOURCE_NOTES.md#ref-dao-flashattention-2022)。

## 3.5 MLP、残差流与归一化

注意力在位置之间混合信息，逐位置 MLP 在特征维度上变换。对 pre-norm decoder block，一种常见写法是

$$
U_\ell=X_\ell+
\operatorname{Attn}_\ell(\operatorname{Norm}_{\ell,1}(X_\ell)),
$$

$$
X_{\ell+1}=U_\ell+
\operatorname{MLP}_\ell(\operatorname{Norm}_{\ell,2}(U_\ell)).
$$

dropout、残差缩放或额外 gate 可插在子层输出处；它们是否存在属于模型定义。所谓 **residual stream** 不是额外模块，而是 $X_0,U_0,X_1,\ldots$ 这条不断被各子层读取和写回的主表示通道。

对单个 token 向量 $x\in\mathbb R^d$，LayerNorm 与 RMSNorm 可分别写为

$$
\operatorname{LN}(x)
=\gamma\odot
\frac{x-\mu(x)\mathbf 1}
{\sqrt{d^{-1}\sum_{j=1}^{d}(x_j-\mu(x))^2+\varepsilon}}
+\beta,
$$

$$
\operatorname{RMSNorm}(x)
=\gamma\odot
\frac{x}{\sqrt{d^{-1}\sum_{j=1}^{d}x_j^2+\varepsilon}}.
$$

LayerNorm 同时去中心与缩放；RMSNorm 只按均方根缩放。$\gamma,\beta$ 是否存在、$\varepsilon$ 的数值、归一化位于子层前还是子层后都会影响训练，不能把“使用 Transformer”当作完整架构描述。

现代 decoder 常使用 gated MLP。SwiGLU 的一个常见形式为

$$
\operatorname{SwiGLU}(x)
=\left(\operatorname{SiLU}(xW_g)\odot xW_u\right)W_d,
\qquad
\operatorname{SiLU}(z)=z\sigma(z),
$$

其中 $W_g,W_u\in\mathbb R^{d\times d_{ff}}$，$W_d\in\mathbb R^{d_{ff}\times d}$。忽略 bias，其参数量是 $3dd_{ff}$。标准两层 FFN 的参数量是 $2dd_{ff}$；若标准 FFN 取 $d_{ff}=4d$，则为保持近似相同参数预算，SwiGLU 常取 $d_{ff}$ 接近 $8d/3$，再按硬件粒度取整。这是预算换算，不是必须遵守的架构定律。

RMSNorm 与 gated FFN 的研究入口见 [Zhang & Sennrich, 2019](SOURCE_NOTES.md#ref-zhang-rmsnorm-2019) 和 [Shazeer, 2020](SOURCE_NOTES.md#ref-shazeer-glu-2020)。

## 3.6 位置信息与 RoPE 的相对位移性质

若不注入位置，self-attention 对输入位置的共同置换具有等变性，无法仅从内容区分先后顺序。绝对位置 embedding 直接把位置向量加到 token 表示。原始 Transformer 的正弦编码使用不同频率：

$$
\operatorname{PE}(p,2i)=\sin(p\theta_i),
\qquad
\operatorname{PE}(p,2i+1)=\cos(p\theta_i).
$$

RoPE 不把位置向量加进 residual stream，而是在每个二维 query/key 子空间中旋转。以下假设被旋转的维数为偶数；若模型只旋转部分坐标，则同一推导只作用于该偶数维子空间。令

$$
R_i(p)=
\begin{bmatrix}
\cos(p\theta_i)&-\sin(p\theta_i)\\
\sin(p\theta_i)&\cos(p\theta_i)
\end{bmatrix},
$$

并把各 $R_i(p)$ 组成分块对角矩阵 $R(p)$。若

$$
q_p=R(p)\widetilde q_p,
\qquad
k_s=R(s)\widetilde k_s,
$$

则利用旋转矩阵的群性质得到

$$
q_p^\top k_s
=\widetilde q_p^\top R(p)^\top R(s)\widetilde k_s
=\widetilde q_p^\top R(s-p)\widetilde k_s.
$$

因此位置对点积的作用显式依赖相对位移 $s-p$，同时仍保留内容向量。相对位置 bias、ALiBi 等方法则直接在注意力分数中加入距离相关项。

位置插值或频率缩放可以把既有位置范围映射到更长索引，但“张量能够运行”不等于“模型已经学会长距离检索与组合”。外推质量还取决于训练长度、长文本数据、attention 实现和评测任务。

RoPE 的原始构造见 [Su et al., 2021](SOURCE_NOTES.md#ref-su-rope-2021)。

![位置编码随位置变化的示例](chapter_03/images/positional_encoding.png)

## 3.7 从 embedding 到 logits：完整形状账本

设词表大小为 $|V|$，token embedding 矩阵为 $E\in\mathbb R^{|V|\times d}$。decoder-only 模型可以概括为：

```text
token ids [B, n]
-> token/position representation [B, n, d]
-> [norm -> causal attention -> residual
    -> norm -> gated MLP -> residual] x L
-> final norm [B, n, d]
-> vocabulary projection [B, n, |V|]
-> logits
```

以 $B=2,n=4,d=8,H_q=H_{kv}=2,d_h=4$ 为例，单层主要张量为：

| 张量 | 形状 |
| --- | --- |
| $X$ | $[2,4,8]$ |
| $Q,K,V$ reshape 后 | $[2,2,4,4]$ |
| 每头分数 $QK^\top$ | $[2,2,4,4]$ |
| 每头加权结果 | $[2,2,4,4]$ |
| concat 后 | $[2,4,8]$ |
| attention 输出投影后 | $[2,4,8]$ |
| logits | $[2,4,|V|]$ |

把最后隐藏状态 $h_i\in\mathbb R^d$ 视为列向量，unembedding 得到

$$
\ell_i=W_Uh_i+b,
\qquad
W_U\in\mathbb R^{|V|\times d}.
$$

若输入 embedding 表为 $E\in\mathbb R^{|V|\times d}$，采用 weight tying 时可令 $W_U=E$；否则输入 embedding 与输出投影独立。实现中的 batch 张量把 hidden state 放在最后一轴时，等价地右乘 $W_U^\top$。logit 仍不是概率；softmax、温度和采样由卷三系统解释，卷二则跟踪它们在一次生成中的实际调用。

## 3.8 Encoder、Decoder 与 Cross-Attention

三种主干对应不同信息流：

| 结构 | 可见上下文 | 典型目标 |
| --- | --- | --- |
| encoder-only | 输入内双向 | 掩码恢复、分类、表示学习 |
| decoder-only | 左侧或声明前缀 | 下一 token 预测、开放生成 |
| encoder-decoder | encoder 双向，decoder 因果 | 条件生成、翻译、摘要 |

在 cross-attention 中，query 来自 decoder 当前状态 $Y\in\mathbb R^{m\times d}$，key/value 来自 encoder 输出 $H\in\mathbb R^{n\times d}$：

$$
Q=YW_Q,
\qquad
K=HW_K,
\qquad
V=HW_V.
$$

于是分数矩阵形状为 $m\times n$。decoder 的因果约束由其 self-attention 负责；cross-attention 通常可读取全部有效源位置，只屏蔽源序列 padding。架构不等于用途：encoder 可用于检索，decoder 隐状态也可用于表示，具体能力还取决于数据和目标。

## 3.9 MoE 与条件计算

Mixture-of-Experts 通常把部分逐 token MLP 替换为 $E$ 个专家。对 token 表示 $x$，路由器给出

$$
p_e(x)=\frac{\exp(w_e^\top x)}{\sum_{j=1}^{E}\exp(w_j^\top x)},
\qquad
S(x)=\operatorname{TopK}\{p_e(x)\}_{e=1}^{E}.
$$

在选中集合内重新归一化后，输出为

$$
y=\sum_{e\in S(x)}
\frac{p_e(x)}{\sum_{j\in S(x)}p_j(x)}E_e(x).
$$

MoE 扩大总参数容量，同时只让每个 token 激活 $k$ 个专家。它还引入三个普通 dense MLP 没有的约束：

1. 每个专家的容量有限。若 batch 中分给某专家的 token 超过容量，系统必须丢弃、重路由或扩大缓冲。
2. Top-$k$ 选择会造成负载不均。以 top-1 路由为例，一类辅助项写成

   $$
   \mathcal L_{\mathrm{bal}}
   =E\sum_{e=1}^{E}f_e\bar p_e,
   $$

   其中 $f_e$ 是实际分给专家 $e$ 的 token 比例，$\bar p_e$ 是 batch 内平均路由概率。具体模型还可能加入 router z-loss 或使用无辅助损失的均衡策略。
3. 专家跨设备放置时，token 表示需要 all-to-all 通信；激活 FLOPs 相近不保证墙钟时间相近。

因此必须分开报告总参数、每 token 激活参数、路由负载和通信成本。

top-1 稀疏专家、容量和均衡项的代表性来源见 [Fedus, Zoph & Shazeer, 2021](SOURCE_NOTES.md#ref-switch-transformer-2021)。

## 3.10 注意力之外的状态空间序列模型

连续时间线性状态空间模型可写为

$$
\dot h(t)=Ah(t)+Bx(t),
\qquad
y(t)=Ch(t)+Dx(t).
$$

若在步长 $\Delta$ 内把输入视为常量，精确离散化给出

$$
\bar A=e^{\Delta A},
\qquad
\bar B=\int_0^\Delta e^{\tau A}B\,d\tau,
$$

$$
h_t=\bar A h_{t-1}+\bar Bx_t,
\qquad
y_t=Ch_t+Dx_t.
$$

递推表面上是串行的，但仿射状态更新可组合：

$$
(A_2,b_2)\circ(A_1,b_1)
=(A_2A_1,A_2b_1+b_2).
$$

该运算满足结合律，因此训练时可用 parallel scan 计算整段状态。选择性状态空间模型进一步让步长、输入映射或读取映射依赖 $x_t$，使保留和遗忘受当前内容控制；推理时则递推更新有限维状态。

注意力显式保留可按内容读取的历史 KV，SSM 把历史压进状态。二者有不同的记忆、并行与硬件特征，混合架构也可在不同层使用两者。没有一种序列算子在全部长度、硬件和任务上占优。

选择性 SSM 与结构化状态空间对偶视角分别见 [Gu & Dao, 2023](SOURCE_NOTES.md#ref-gu-mamba-2023) 和 [Dao & Gu, 2024](SOURCE_NOTES.md#ref-dao-gu-mamba2-2024)。

![选择性状态空间模型的信息流](chapter_03/images/mamba_ssm_flow.svg)

## 3.11 本章边界与核对清单

看到一份 Transformer 配置时，至少应能回答：

1. $d,L,H_q,H_{kv},d_h,d_{ff}$ 分别是多少；
2. attention mask 与 loss mask 是否被区分；
3. 使用哪种位置机制，训练与部署长度各是多少；
4. norm、MLP、bias、weight tying 和残差顺序怎样定义；
5. 若使用 MoE，每 token 选几个专家，容量与均衡如何处理；
6. 输出 logits 的词表与 tokenizer 是否匹配。

Transformer 规定一次前向怎样计算，却没有说明参数为何形成某种能力。预训练目标、数据和优化把结构变成模型；后训练再改变它如何响应用户。第四章进入预训练模型谱系，第五章讨论后训练。

主要来源包括 Transformer、RMSNorm、GLU 变体、MQA/GQA、RoPE、Switch Transformer 与状态空间模型，统一登记在[卷内来源表](SOURCE_NOTES.md)。需要逐元素核对基础 attention 推导时，参见[Transformer 数学附录](../appendices/learning-notes/a.10_transformer_math.md)。
