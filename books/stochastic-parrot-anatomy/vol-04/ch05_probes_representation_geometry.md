# 第五章 Probes 与表示几何

probe 训练一个辅助模型，从内部 activation 预测属性。它回答“信息是否可从这里被某类读出器提取”，是表示研究的核心工具；它不自动回答原模型是否使用了该信息。

## 5.1 线性 Probe

给定 activation $h_i\in\mathbb R^d$ 和标签 $y_i$，二分类线性 probe 学习

$$
\hat y_i=\sigma(w^\top h_i+b).
$$

held-out 准确率高说明标签在这些 activation 中近似线性可分。$w$ 定义一个读出方向，但其坐标大小受 activation 尺度和正则化影响。

## 5.2 Decodability 不等于 Use

深层网络的表示维度很高，许多输入属性即使不参与当前行为也可能保留。probe 能读出 $Y$，只说明 $I(H;Y)$ 或某种可预测关系非零，不说明模型下游沿 probe 方向读取了它。

要研究使用，需要：

- 检查下游 weight 与该方向的连接；
- 沿方向干预并观察目标行为；
- 与保持其他信息的对照方向比较；
- 验证干预效应在新输入上符合假说。

## 5.3 Probe 容量

非线性 probe 越强，越可能从复杂表示中提取标签，也越可能记忆数据或自己完成任务。probe 性能必须与容量、训练样本量和 baseline 一起报告。

常见控制包括：

- 线性 vs MLP probe；
- 随机特征或未训练模型；
- 随机标签 control task；
- 不同层与 embedding baseline；
- 以 minimum description length 衡量读出复杂度。

## 5.4 Selectivity

若 probe 对真实标签准确率为 $A_{true}$，对保留输入统计但随机映射的 control label 准确率为 $A_{ctrl}$，selectivity 可粗略写为

$$
S=A_{true}-A_{ctrl}.
$$

它帮助识别 probe 记忆能力，但 control task 怎样构造会改变结果。不能把一个 selectivity 数字视为统一的“表示质量”。

## 5.5 表示相似性

比较模型、层或训练阶段时，可以使用：

- centered kernel alignment (CKA)；
- canonical correlation analysis (CCA)；
- representational similarity matrix；
- Procrustes alignment；
- 子空间 principal angles。

这些方法对旋转、缩放或可逆变换具有不同不变性。高相似表示可能支持不同下游计算，低坐标相似也可能只是基变化。指标必须匹配要忽略的对称性。

## 5.6 Direction 与 Subspace

一个二元属性可能沿单一方向近似线性编码；多类别或连续结构常占据子空间。若用均值差

$$
v=\mathbb E[H\mid Y=1]-
\mathbb E[H\mid Y=0],
$$

它混合类内协方差与潜在混杂。Fisher LDA、PCA 或多任务 probe 会产生不同方向。

“模型有一个诚实方向”只有在方向跨模板、主题、语言和模型状态稳定，并能预测或干预诚实行为时才有意义。

## 5.7 表示几何的可视化

t-SNE、UMAP 和二维 PCA 可以展示样本聚类，但投影会丢失高维关系。t-SNE/UMAP 的全局距离尤其不能直接解释；参数和随机 seed 会改变图形。

可视化应配合定量 held-out 分类、邻域保持率或距离统计。二维上分开的颜色不是高维因果结构的证明。

## 5.8 Token、Position 与 Pooling

句级属性可从最后 token、平均 pooling、特定 delimiter 或所有位置 attention pooling 读出。不同选择对应不同问题。

如果标签可由句长、末尾标点或模板位置预测，probe 可能利用 shortcut。应建立只含这些表面变量的 baseline，并用长度、模板平衡数据。

## 5.9 Amnesic 与 Causal Probing

一种思路是投影掉 probe 子空间，再观察行为：

$$
h'=h-P_Vh.
$$

若目标行为下降，说明该子空间携带有用信息。但投影可能同时删除与标签相关的其他内容并把 activation 推到分布外；行为不变也可能因冗余编码或下游恢复。

更强设计需要 matched random subspaces、activation norm 控制、重构误差报告和恢复实验。

## 5.10 Probe 的正确结论形式

推荐写：

> 在模型 M 的第 $\ell$ 层、位置规则 P 上，给定数据分布 D 和正则化线性读出器，属性 Y 在 held-out 模板上达到某性能；该结果说明 Y 可线性读出，尚未证明下游生成使用该方向。

这种表述看似克制，实际更精确，也为下一步干预留下明确任务。

## 5.11 结论

probes 测量信息可读性，表示几何研究信息怎样组织。它们适合比较层、模型和训练阶段，也最容易被误写成“模型已经知道并使用”。从 readout 到 mechanism 的桥梁，是第七章的干预与第八章的回路组合。
