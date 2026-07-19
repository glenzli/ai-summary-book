# 第一章 我们在模型内部看什么

解释方法操作的不是抽象“思想”，而是前向计算图中的参数、activation 与局部导数。一个内部单位是否有意义，取决于它在计算图中的位置、坐标约定和允许操作；同名张量在不同实现中未必是同一对象。

## 1.1 前向计算图与索引约定

对 batch 中第 $b$ 个样本、位置 $p$、层 $\ell$，记 residual state 为

$$
x_{b,\ell,p}\in\mathbb R^{d_{\mathrm{model}}}.
$$

省略 batch 下标后，一个 pre-norm decoder block 写为

$$
r_{\ell,p}=\operatorname{Norm}^{A}_{\ell}(x_{\ell,p}),
$$

$$
x'_{\ell,p}=x_{\ell,p}+a_{\ell,p},
\qquad
a_{\ell,p}=\operatorname{Attn}_{\ell}(r_{\ell,\le p})_p,
$$

$$
s_{\ell,p}=\operatorname{Norm}^{M}_{\ell}(x'_{\ell,p}),
$$

$$
x_{\ell+1,p}=x'_{\ell,p}+m_{\ell,p},
\qquad
m_{\ell,p}=\operatorname{MLP}_{\ell}(s_{\ell,p}).
$$

这里至少已有六类可 hook 对象：`resid_pre`、normalized attention input、attention result、`resid_mid`、normalized MLP input 与 MLP result。把它们统称为“第 $\ell$ 层 activation”会丢失干预语义。

## 1.2 Residual stream 是通信空间

residual stream 是各子层共同读写的 $d_{\mathrm{model}}$ 维通道。忽略 dropout，可展开为

$$
x_{L,p}=x_{0,p}+
\sum_{\ell=0}^{L-1}
\bigl(a_{\ell,p}+m_{\ell,p}\bigr).
$$

这是向量加法的精确恒等式，不是 credit assignment 定理。早层写入 $c$ 即使最终直接投影很小，也可能通过改变后续 query、gate 或 normalization 产生主要间接效应。

一个方向可由多个模块写入，也可被多个下游模块读取。因而 residual 坐标不是彼此独立的语义寄存器；分析方向 $v$ 时要同时问：谁写入 $v$，谁对 $v$ 敏感，二者在什么输入上连接。

## 1.3 Attention head 的三类对象

第 $h$ 个 head 的 query、key、value 为

$$
q_p^{h}=W_Q^h r_p,\qquad
k_j^{h}=W_K^h r_j,\qquad
v_j^{h}=W_V^h r_j.
$$

其 score、pattern 与写回结果为

$$
s_{p,j}^{h}
=\frac{(q_p^h)^\top k_j^h}{\sqrt{d_h}}+M_{p,j},
\qquad
A_{p,j}^{h}=\operatorname{softmax}_j(s_{p,j}^{h}),
$$

$$
o_p^h=W_O^h\sum_j A_{p,j}^h v_j^h.
$$

所以“研究一个 head”至少可能指：

1. QK score 或 attention pattern：选择了哪个 source position；
2. value：source state 中什么内容被读取；
3. result $o_p^h$：经 $W_O^h$ 后向 residual 写入什么；
4. head 对下游节点或目标 logit 的效应。

只有 pattern 没有 OV 与下游读出，最多得到路由假说。

## 1.4 MLP neuron 与 gated feature

普通 MLP 可写为

$$
m=W_{out}\phi(W_{in}s+b_{in})+b_{out}.
$$

中间单元 $i$ 的 activation 和写入为

$$
u_i=\phi(w_{in,i}^{\top}s+b_i),
\qquad
c_i=u_iw_{out,i}.
$$

对 SwiGLU 等 gated MLP，单元贡献更接近

$$
u_i=\operatorname{SiLU}(w_{g,i}^{\top}s)
(w_{v,i}^{\top}s),
$$

不能只缓存其中一个分支便称为完整“神经元 activation”。模型实现还可能融合 kernel；研究代码要从数学计算点而不是变量名确定 hook。

## 1.5 Feature、Direction 与 Subspace

“feature”没有唯一架构定义。它可指：

- 原始 neuron 坐标；
- residual 或 MLP 空间中的方向 $v$；
- probe 的决策法向量；
-由 paired differences 得到的均值方向；
- SAE latent 及其 encoder/decoder vectors；
- transcoder feature；
- 相对于行为定义的低维子空间。

若 feature 是方向，只给向量不够。还需给 activation convention，例如

$$
f_v(x)=v^\top(x-\mu),
$$

其中是否中心化、$v$ 是否单位归一、在哪个 token position 计算都会改变数值。对非正交字典，encoder score 与 decoder 投影也不是同一个量。

## 1.6 坐标对称性与对象可识别性

设隐藏表示作可逆换基 $x'=Ax$。若相邻权重相应变换，模型函数可以保持不变：

$$
W'_{in}=W_{in}A^{-1},\qquad
W'_{out}=AW_{out}.
$$

一般 Transformer 的 normalization、逐元素非线性和 tied weights 会限制可允许的 $A$，但置换、缩放或局部等价仍足以说明：坐标值本身不总是函数可识别量。

因此应区分：

- **架构可定位**：这个 head/neuron 在实现中有固定索引；
- **功能可识别**：在允许的等价变换后仍能由行为或干预签名辨认；
- **语义可识别**：给定概念假说在 held-out 条件下优于替代说明。

跨 seed 比较 neuron 编号通常没有意义；要比较的是经明确对齐后的方向、子空间或功能签名。

## 1.7 Normalization 改变干预语义

RMSNorm 可写为

$$
\operatorname{RMSNorm}(x)
=g\odot
\frac{x}{\sqrt{d^{-1}\|x\|_2^2+\varepsilon}}.
$$

在 norm 前加方向 $\alpha v$ 会同时改变分母，因而影响所有归一化坐标；在 norm 后加同一方向则是不同操作。LayerNorm 还减去均值，使全一方向具有特殊不变性。

研究必须记录：

- norm 前还是后；
- 替换整个向量还是一个投影；
- 是否保持原向量范数；
- 后续是否重新计算 normalization；
- 混合精度和缓存值的 dtype。

这些不是工程细节，而是干预定义的一部分。

## 1.8 Logits、读出与 Direct Logit Attribution

最终 logits 为

$$
z=W_U\operatorname{Norm}_f(x_L)+b_U.
$$

对候选 $a,b$，定义 unembedding difference

$$
u_{a,b}=W_{U,a}-W_{U,b},
\qquad
S=z_a-z_b.
$$

若在当前最终 residual $x_L$ 处对 normalization 作一阶线性化，一个 residual contribution $c$ 的局部 direct logit attribution 为

$$
\operatorname{DLA}^{(1)}_{a,b}(c;x_L)
=u_{a,b}^{\top}J_{\operatorname{Norm}_f}(x_L)c
=\bigl(J_{\operatorname{Norm}_f}(x_L)^{\top}u_{a,b}\bigr)^{\top}c.
$$

若把 normalization 忽略为恒等映射，上式才约化为 $u_{a,b}^{\top}c$；若在某个参考点冻结 normalization statistics，则应把相应仿射线性部分写入有效读出方向。严格说，最终 normalization 使 $S$ 不再是各 $c$ 的无条件线性和。常见处理有三种：使用实际 normalized decomposition、在当前点线性化，或明确声明忽略 normalization。三者不能静默混用。

## 1.9 Weight、Activation、Gradient 与 Jacobian

- weight $\theta$ 跨输入固定；
- activation $u(x)$ 属于一次具体前向；
- gradient $\nabla_u S$ 是当前点附近的局部敏感性；
- Jacobian $J_{v\leftarrow u}=\partial v/\partial u$ 描述小扰动怎样传播。

若上游扰动为 $\delta u$，一阶下游变化是

$$
\delta v\approx J_{v\leftarrow u}\delta u.
$$

大 weight 不保证 activation 常出现，大 activation 不保证目标重要，大 gradient 也可能由饱和、局部曲率和参数化改变。任何“重要性”都必须绑定目标与操作。

## 1.10 Token position 与生成时间

同层同通道在不同位置是不同计算节点。自回归生成还要区分 prefill 中的 prompt positions 与 decode step $t$ 新增的位置。KV cache 中保存的是历史 key/value；patch 某一步 cache 与重新运行完整前缀并不总是同一实验。

最小坐标应写成

$$
u=(\text{module},\ell,p,h,i,t,\text{hook convention}),
$$

其中不适用的索引可省略。报告“第 12 层 feature 42”而不含位置与序列条件，通常不足以复现。

## 1.11 Hook、缓存与数值一致性

在开展解释前先做三项校验：

1. **identity hook**：读出再原样写回，输出应在容差内不变；
2. **recompute check**：缓存的局部计算应能重建模块输出；
3. **batch invariance**：单样本与批处理结果应在数值容差内一致。

还要记录 fused attention、dropout、quantization、tensor parallel 和 stochastic kernels。若 hook 改变了执行路径，后续差异可能来自研究工具而不是目标干预。

## 1.12 其他架构中的对象

视觉语言模型还包含 patch/token 表示、vision encoder、connector、cross-attention 与共享 decoder；一个视觉 token 可能是重采样后的混合区域。扩散模型包含时间条件、latent feature map、cross-attention 与每个去噪步的状态。MoE 还要记录 router logits、被选 experts 与合并权重。

本卷方法可以迁移，但不能把 decoder-only Transformer 的“层—位置—head”本体直接套用。首先应为目标架构重画可干预计算图。

## 1.13 内部对象审计表

| 对象 | 典型操作 | 直接估计量 | 能支持 | 不能单独支持 | 常见失效 |
|---|---|---|---|---|---|
| attention pattern | 观察/替换 pattern | source mass、熵、输出变化 | 路由规律或指定替换效应 | value 内容与完整原因 | 高权重低 value、替代 pattern |
| residual direction | 投影/加向量 | projection、$\Delta S$ | 可读性或控制效应 | 唯一概念坐标 | 换基、norm、混杂方向 |
| neuron | 缓存/删除 | activation、ablation effect | 坐标级相关或效应 | 单义概念 | polysemanticity、冗余 |
| SAE latent | 编码/替换/steer | firing、reconstruction、行为差 | 学习基中的预测或控制 | 原模型天然 feature | 非唯一字典、误差路径 |
| edge/path | path patch | 路径恢复量 | 指定路径效应 | 全局完整机制 | 不一致状态、交互 |
| gradient | 求导 | 局部导数 | 当前点敏感性 | 有限自然反事实 | 饱和、坐标依赖 |

本章的结论是方法论性的：内部对象先是计算图节点或分析坐标，后才可能成为语义与机制单位。后续每章都必须说明它读取、训练或干预了哪一个精确对象。
