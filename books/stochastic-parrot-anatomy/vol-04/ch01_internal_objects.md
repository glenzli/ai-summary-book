# 第一章 我们在模型内部看什么

解释方法操作的不是抽象“思想”，而是前向传播中具体张量。理解这些对象的形状和作用，是避免把热图、向量和神经元名称过度心理化的第一步。

## 1.1 Residual Stream

对 decoder-only Transformer，位置 $p$、层 $\ell$ 的 residual state 记为

$$
x_{\ell,p}\in\mathbb R^{d_{\mathrm{model}}}.
$$

简化的 pre-norm 层可写为

$$
x'_{\ell,p}
=x_{\ell,p}+
\operatorname{Attn}_\ell(\operatorname{LN}(x_\ell))_p,
$$

$$
x_{\ell+1,p}
=x'_{\ell,p}+
\operatorname{MLP}_\ell(\operatorname{LN}(x'_{\ell,p})).
$$

residual stream 是各模块共同读写的通信通道。一个方向可能同时被多个组件写入，也被多个下游组件读取；它不是一组彼此独立的语义槽位。

## 1.2 Attention Head

第 $h$ 个 head 对位置 $p$ 的 query 和来源位置 $j$ 的 key 计算

$$
s_{p,j}^{(h)}
=\frac{q_p^{(h)}\cdot k_j^{(h)}}{\sqrt{d_h}}+M_{p,j},
$$

$$
a_{p,j}^{(h)}=\operatorname{softmax}_j(s_{p,j}^{(h)}).
$$

输出为

$$
o_p^{(h)}
=W_O^{(h)}\sum_j a_{p,j}^{(h)}v_j^{(h)}.
$$

attention pattern $a_{p,j}$ 告诉我们从哪些位置混合 value；它没有单独说明 value 携带什么，也没有说明混合结果怎样影响目标 logit。解释一个 head 通常要同时研究 QK 选择机制与 OV 写入机制。

## 1.3 MLP Neuron

一层 MLP 可写为

$$
m=W_{out}\phi(W_{in}x+b_{in})+b_{out}.
$$

中间单元 $i$ 的 activation 为

$$
a_i=\phi(w_{in,i}^{\top}x+b_i),
$$

它沿 $w_{out,i}$ 方向写回 residual stream。所谓“激活神经元”通常是在看 $a_i$ 对不同 token/上下文的数值，而不是看到一个神经元输出自然语言概念。

一个神经元可在不相关输入上都激活，称为 polysemantic；一个概念也可分布在许多神经元和方向上。

## 1.4 Feature 与 Direction

“feature”没有唯一架构定义。研究中它可能指：

- 一个原始神经元；
- activation 空间中的线性方向 $v$；
- probe 的决策法向量；
- sparse autoencoder 的 latent 单元；
- transcoder 或 cross-layer feature；
- 某个行为上由实验定义的低维子空间。

写作时必须说明 feature 如何构造，不能把不同方法得到的同名 feature 当作同一对象。

## 1.5 Logits 与 Direct Logit Attribution

最终 state 经 unembedding 得到

$$
z=W_U\operatorname{LN}(x_L)+b.
$$

若暂时忽略最终 normalization 的非线性，一个 residual contribution $c$ 对 token $t$ 的直接 logit contribution 近似为

$$
\operatorname{DLA}_t(c)=W_{U,t}^{\top}c.
$$

这只测量 $c$ 通过直接 residual 路径对输出的线性作用。若 $c$ 改变后续 attention、MLP 门控或 normalization，完整间接效应可能不同。

## 1.6 Activation、Weight 与 Gradient

- weight 是训练后保存的参数，跨输入固定；
- activation 是某次前向在特定输入上产生的状态；
- gradient 是目标对 weight、activation 或输入的局部导数。

大 weight 不保证对应 activation 常出现，大 activation 不保证对目标重要，大 gradient 也只描述局部敏感性。三者回答不同问题。

## 1.7 Token Position

同一层同一通道在不同位置的 activation 是不同节点。对语言模型，位置关系尤其重要：一个 head 可在目标位置读取主语位置的信息，另一个 feature 只在答案首 token 处写入 logit。

因此“第 12 层 feature 42 激活”仍不完整，还应记录序列、token position、前后文和 normalization 约定。

## 1.8 Hook 与缓存

研究代码通常通过 hook 保存或修改中间张量。需要明确 hook 位于：

- layer norm 前或后；
- attention/MLP 输入或输出；
- 单个 head 合并前或后；
- residual addition 前或后；
- prefill 还是某个 decode step。

同名 `layer_10_output` 在不同框架中可能指不同张量。可重复实验应记录模型架构图和精确 hook site。

## 1.9 多模态内部对象

视觉语言模型还包括 patch/token 表示、视觉 encoder、connector、cross-attention 与共享 decoder。图像区域与文本 token 的对应可能经过 pooling 或重采样，不能假定一个视觉 token 等于一个可见物体。

扩散模型的对象则包括时间条件、latent feature map、cross-attention 和去噪预测；其“某一步 activation”属于全局迭代状态，不等同于自回归模型某一 token 的 hidden state。

## 1.10 本章结论

可解释性看到的是向量、activation、attention weights、参数和局部导数。人类概念是对这些对象提出的解释假说。后续每种方法都必须说明它读出或干预哪个对象，以及从该对象到行为结论之间还缺哪些验证。
