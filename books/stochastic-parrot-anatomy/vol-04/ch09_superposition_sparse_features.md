# 第九章 Superposition、SAE 与稀疏表示

原始 neuron 坐标常同时响应多种模式。superposition 假说认为，模型可把许多稀疏出现的特征编码在少量非正交方向中；SAE 与 transcoder 试图学习更稀疏的分析坐标。本章重点不是展示可命名 feature，而是说明字典何时可比较、何时不唯一，以及怎样检验替代表示保留机制。

## 9.1 线性 toy model 的解释范围

设潜在 feature $f\in\mathbb R^m$ 稀疏，表示

$$
x=Wf,
\qquad x\in\mathbb R^d,
\quad m>d.
$$

当每个样本只有少量 $f_i$ 非零时，非正交列 $W_{:,i}$ 可以共享 $d$ 维空间；共同激活时产生 interference。这个模型说明“feature 数大于 neuron 数”在稀疏条件下可能实现。

它不是关于真实 Transformer 的已证定理。真实 activation 可能没有固定线性生成字典，feature 可上下文依赖，attention 与 MLP 还构成动态非线性计算。toy model 的作用是生成可检验假说。

## 9.2 SAE 的基本参数化

对中心化 activation $x-b_{dec}$，ReLU SAE 可写为

$$
f=\operatorname{ReLU}
(W_{enc}(x-b_{dec})+b_{enc}),
$$

$$
\hat x=W_{dec}f+b_{dec}.
$$

典型目标为

$$
\mathcal L
=\mathbb E\|x-\hat x\|_2^2
+\lambda\mathbb E\|f\|_1.
$$

TopK SAE 则只保留 preactivation 最大的 $k$ 个坐标；JumpReLU 学习非零阈值。不同稀疏门控定义不同 estimand，不能只用相同的“平均 $L_0$”直接比较。

必须报告：hook site、activation 预处理、字典宽度 $m$、稀疏机制、decoder 约束、训练 token 分布、dead-feature 处理与重构 dtype。

## 9.3 尺度退化

对正齐次激活，若某 latent 与 decoder column 可独立重缩放，令

$$
f_i'=cf_i,\qquad d_i'=d_i/c,\qquad c>0.
$$

则

$$
f_i'd_i'=f_id_i,
$$

所以 reconstruction 不变，而

$$
\|f_i'\|_1=c\|f_i\|_1.
$$

若 decoder norm 无约束，取 $c\to0$ 可任意减小稀疏惩罚。这证明仅写“重构加 $L_1$”不足以定义优化问题。常见修复是约束

$$
\|d_i\|_2=1
$$

或采用尺度不变惩罚。比较 latent activation 大小时必须先确认尺度规范一致。

## 9.4 最低限度的非唯一性

即便固定 decoder norm，SAE 仍至少有 permutation symmetry：对任意置换矩阵 $P$，

$$
f'=Pf,\qquad
W'_{dec}=W_{dec}P^\top
$$

给出相同重构。这个对称性可通过 feature matching 消除，不构成实质语义差异。

更严重的是：

- 相关 feature 可被不同方向组合；
- 一个宽 feature 可 split 为多个上下文子 feature；
- 罕见 feature 可被常见 feature absorption；
- 多个近重复 decoder columns 可共享样本；
- 重要信息可留在 dense reconstruction error。

稀疏字典学习只有在关于生成过程、稀疏度、字典 incoherence 与样本覆盖的附加条件下才可能可识别。真实 LLM activation 不自动满足这些条件。

## 9.5 Encoder、Decoder 与 Latent 不是同一对象

第 $i$ 个 latent 的 encoder row 决定哪些 $x$ 触发它，decoder column $d_i$ 决定它怎样重构。非正交字典下二者不必平行。

因此 feature 报告至少包含：

- preactivation 与 post-gate firing；
- encoder-selected 自然样本；
- decoder 对 residual/logit 的直接写入；
- 删除/插入 latent 的行为 effect；
- 与相近 decoder features 的共同激活；
- reconstruction error 的相关方向。

把 decoder 高投影 token 当作输入语义，或把 top activation 样本当作输出功能，都会混淆两面。

## 9.6 Reconstruction 与 sparsity 的基础指标

常用 activation fidelity 包括 explained variance

$$
\operatorname{EV}
=1-\frac{\mathbb E\|x-\hat x\|_2^2}
{\mathbb E\|x-\mathbb E x\|_2^2}
$$

和 cosine similarity。EV 只在 $\mathbb E\|x-\mathbb Ex\|_2^2>0$ 时定义；activation 在评价分布上几乎处处为常量时，应报告该退化情形而不是令分母为零。稀疏度可报告平均 $L_0$、firing frequency 分布与 activation entropy。

这些指标回答“字典怎样压缩 activation”，不回答 feature 是否单义或有因果用途。[SAEBench（ICML 2025）](https://proceedings.mlr.press/v267/karvonen25a.html)组织了八项评估，而 [Chanin（2026）的复现实验与合成真值审计](https://arxiv.org/abs/2605.18229)发现部分指标对 reseed、可区分性和已知结构并不稳健。二者合在一起仍不足以建立“代理指标越高，下游解释用途必然越好”的可靠单调关系；后者尤其是新近预印本结果，应视为当前证据而非最终标准。

## 9.7 Mechanism preservation

令外部输入 $a\sim\mathcal D_{\mathrm{eval}}$，被分析位置的原 activation 为 $x(a)$，SAE 重构为 $\hat x(a)$。在同一输入上把该内部 activation 替换为重构值，测

$$
\Delta L_{recon}
=\mathbb E_{a\sim\mathcal D_{\mathrm{eval}}}
\left[
L(M_{x(a)\leftarrow\hat x(a)};a)-L(M;a)
\right].
$$

还应报告 next-token KL、argmax agreement、目标任务保留率与特定干预响应。小均方误差不保证小行为误差：误差可能落在高 Jacobian 方向。

一阶可检查

$$
\Delta S_{\mathrm{err}}(a)
:=S(M;a)-S(M_{x(a)\leftarrow\hat x(a)};a)
\approx
\nabla_xS(M;a)^\top(x(a)-\hat x(a)),
$$

其中梯度在原 activation $x(a)$ 处计算。若取 $S=L$，则 $\Delta S_{\mathrm{err}}(a)$ 恰是对应逐样本 replacement-minus-original loss 增量的负值；对任意其他 score $S$，两者没有这种符号关系。若采用相反差值 convention，线性项也应反号。最终仍需真实重跑。若 error node 对目标贡献显著，任何只在 SAE latent 上构造的 circuit 都不完整。

## 9.8 Dead、Split、Absorbed 与 Composite Features

- **dead**：几乎不在训练/验证数据激活；
- **split**：一个粗模式分散到多个条件 feature；
- **absorbed**：某模式在通常场景被另一 feature 表示，在关键例外中漏失；
- **duplicate**：多个 decoder direction 与 firing set 高度重合；
- **composite**：一个 latent 在不同上下文混合多个机制变量。

更宽字典不单调产生更真实 feature。评估 splitting 需要已知 labels 或跨字典层级匹配；只看 feature 名称会把细粒度拆分误认为发现更多概念。

## 9.9 Monosemanticity 与自动说明

单义性不是可直接观察的二元真值。可分解为：

- 输入说明在 held-out 样本上的预测性；
- 反例选择性；
- 不同语料 domain 的稳定性；
- decoder effect 的一致性；
- 干预后行为是否符合说明；
- 与相邻 features 的可区分性。

自然语言 grader 偏好容易命名的语义模式，可能低估纯计算 feature。自动解释分数高也可能由 token identity shortcut 获得；必须与随机方向、原 neurons 和 supervised task directions 比较。

## 9.10 任务级因果评估

对已知因果变量 $Y$ 和任务行为 $S$，可检验 SAE features 是否：

1. 稀疏读出 $Y$；
2. 在 matched counterfactual 中改变 $Y$ 而少改其他属性；
3. 删除后降低相关行为；
4. 插入后产生方向一致的变化；
5. 比 neuron、PCA、probe direction 与 supervised dictionary 更有效；
6. 在 held-out templates 上保持。

[MIB（ICML 2025）](https://proceedings.mlr.press/v267/mueller25a.html)在其任务和模型上报告，受监督 DAS 在 causal-variable localization 中优于所测无监督 SAE，SAE features 未优于 neurons。这限制的是“SAE 已普遍胜出”的说法，不证明 SAE 在所有任务无用。

## 9.11 跨 Seed 的 Feature Matching

给两个使用非负 latent coefficient 的 SAE，其单位 decoder matrices 为 $D^{(1)},D^{(2)}$。feature 方向匹配应保留符号，例如使用

$$
C_{ij}=\langle d_i^{(1)},d_j^{(2)}\rangle
$$

做最大权重二分匹配，再同时比较 firing correlation 与干预 effect。对 ReLU/JumpReLU 编码，$d\mapsto-d$ 通常不能由 latent 符号翻转抵消，反向 decoder direction 对应相反干预，不能因绝对 cosine 相同就视为同一 feature。只有研究无向一维子空间时才可使用 $|\langle d_i,d_j\rangle|$，并必须把结果标作子空间匹配而非 SAE feature 匹配。只按 decoder cosine 匹配仍可能把 input-side 功能不同的方向视为同一 feature。

稳定性应至少报告：

- 可一对一匹配的 feature 比例；
- matched cosine 与 activation correlation；
- top-example/说明一致率；
- 干预 effect 一致率；
- unmatched features 是否落在共同子空间。

[ICML 2026 的 identifiable SAE 工作](https://openreview.net/forum?id=miLK9YcxtA)给出了提高跨 run 稳定性的参数化与经验结果；它表明可识别性可被工程约束改善，但没有证明任意真实 activation 存在唯一语义字典。

## 9.12 Transcoder 与 Cross-layer Transcoder

SAE 重构状态 $x$；transcoder 以模块输入 $x$ 预测模块输出 $y=F(x)$：

$$
f=\phi(W_{enc}x+b),
\qquad
\hat y=W_{dec}f+b_y.
$$

目标为输出重构与稀疏度，因此 latent 更接近输入条件下的模块贡献。cross-layer transcoder 允许早层 features 预测多个后续 MLP outputs，用于构造跨层稀疏 attribution graph。

代价是替代模型假设更强。它可能绕过原模型中的中间步骤；低 output error 不证明 latent 对应原网络天然节点。需要比较中间 intervention responses 与 error paths。

## 9.13 Steering 与控制用途

沿 decoder direction 干预

$$
x'=x+\alpha d_i
$$

或直接修改 latent $f_i$，可改变主题、拒答或风格。评估要包含剂量曲线、反方向、matched random/neuronal directions、语言质量与非目标行为。

feature label 与 steering utility 是不同指标。可命名 feature 不一定最好控制行为，强 steering direction 也可能通过异常大 norm 和广泛副作用起效。

## 9.14 方法审计表

| 方法 | 问题/对象 | 操作与估计量 | 必要控制 | 能支持 | 不能支持与失效 |
|---|---|---|---|---|---|
| SAE reconstruction | 稀疏字典能否压缩 activation | 编码/解码；EV、$L_0$ | decoder norm、dead rate、数据分布 | 指定字典的稀疏重构 | 真实 feature 恢复 |
| auto-interpretation | latent 是否有可预测说明 | top/随机样本与模拟评分 | surface baselines、独立 grader | input-side 可说明性 | 下游机制 |
| task causal eval | feature 是否承载目标变量 | delete/insert/counterfactual | neurons、probe、supervised dictionary | 指定任务功能 | 全局单义性 |
| cross-seed matching | feature 是否可重复 | assignment + firing/effect 对齐 | random null、子空间比较 | 采用匹配规则的稳定性 | 唯一字典 |
| transcoder graph | 稀疏替代能否复现模块计算 | 预测 module outputs | error nodes、原模型干预 | 替代机制与候选回路 | 原网络天然分解 |

SAE 与 transcoder 提供有用的稀疏分析基，但 feature 的存在不是由名称决定，也不因重构损失低而获得机制地位。目前能稳健得到的结论是：它们是需要跨 seed、任务干预和替代误差共同验证的学习型坐标系。
