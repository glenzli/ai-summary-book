# 第五章 Probes 与表示几何

probe 训练辅助读出器，从内部 activation 预测研究者定义的属性。它回答“在给定数据与读出器类别下，信息能否被提取”。表示几何则比较样本关系、方向和子空间。两者都不自动说明原模型下游使用了所发现的信息。

## 5.1 Probe 的统计对象

给定冻结模型产生的 activation-label 样本

$$
\mathcal D_P=\{(h_i,y_i,g_i)\}_{i=1}^{n},
\qquad h_i\in\mathbb R^d,
$$

其中 $g_i$ 是模板、文档或实体 group。二分类线性 probe 学习

$$
p_\phi(y=1\mid h)=\sigma(w^\top h+b)
$$

并最小化带正则的交叉熵。研究对象不是训练准确率，而是指定 split 与分布上的泛化风险

$$
R_P(\phi)=\mathbb E_{(H,Y)\sim P_{\mathrm{test}}}
[\ell(p_\phi(Y\mid H),Y)].
$$

其中 $P_{\mathrm{test}}$ 是按预先声明的 group split 诱导的 held-out activation–label 分布。必须按 $g_i$ 分组切分，避免同一模板或句子不同 token 泄漏到训练与测试。

## 5.2 Decodability 不等于 Use

held-out 性能高说明 $Y$ 可由 probe 类 $\mathcal F$ 从 $H$ 读出：

$$
\inf_{f\in\mathcal F}R_P(f)
$$

相对 baseline 较低。它不说明模型的真实下游映射属于 $\mathcal F$，也不说明当前行为依赖 $Y$。

高维表示可能保留大量输入属性；一个强 probe 还可能自己重建任务。研究“使用”需要检查下游读取、方向干预和与目标行为的条件关系。

## 5.3 Probe 容量与学习曲线

probe 越强，越能发现可提取信息，也越容易记忆数据。不要用“线性总是更科学”替代问题定义：

- 若问题是线性可读性，应限制为线性 probe；
- 若问题是信息是否存在，可比较逐步增加容量的读出器；
- 若问题是读出难度，应报告样本复杂度或 description length。

至少绘制训练样本数 $n$ 与 held-out 性能的学习曲线，并比较 linear、low-rank、MLP、随机特征和只看表面变量的 baselines。

## 5.4 Control tasks 与 selectivity

Hewitt–Liang control task 保留输入 type 等统计结构，却随机分配标签，使 probe 只能依靠记忆。若真实性能为 $A_{true}$、控制性能为 $A_{ctrl}$，可报告

$$
\operatorname{Sel}=A_{true}-A_{ctrl}.
$$

这不是统一“表示质量”。control label 的粒度、样本复用和 probe 容量都会改变 $A_{ctrl}$。应同时给两项原始结果及多个随机 control assignments。

其他必要控制包括：

- 未训练或随机权重模型的同层表示；
- embedding、词袋、长度、位置和 token identity baselines；
- label permutation；
- 同容量 probe 在邻层和随机子空间上的结果；
- balanced 与自然 prevalence 两种评估。

## 5.5 Calibration、类别不平衡与不确定性

准确率会隐藏类别不平衡和置信度。按问题报告 balanced accuracy、AUROC、AUPRC、cross-entropy 与 calibration。测试集要按独立 group bootstrap 得到区间。

在扫描许多层、位置和方向后挑最高分会产生 winner's curse。discovery 集用于选层，最终 test 集只评一次；否则置信区间没有覆盖所进行的搜索。

## 5.6 Minimum Description Length

MDL probing 把“标签能否被简短地从表示编码”写成 code length。取

$$
0=t_0<t_1<\cdots<t_K=n,
$$

并记 $H_{a:b}=(h_a,\ldots,h_b)$、$Y_{a:b}=(y_a,\ldots,y_b)$。在线编码近似为

$$
L_{online}(Y\mid H)
=t_1\log_2|\mathcal Y|
-\sum_{k=1}^{K-1}
\log_2 p_{\phi_k}
(Y_{t_k+1:t_{k+1}}
\mid H_{t_k+1:t_{k+1}}),
$$

其中 $\phi_k$ 只在前 $t_k$ 个样本上训练；若 probe 对 block 内标签条件独立编码，则

$$
p_{\phi_k}(Y_{a:b}\mid H_{a:b})
=\prod_{i=a}^{b}p_{\phi_k}(y_i\mid h_i).
$$

若采用序列式编码器，则应以它实际给出的联合条件概率替换该乘积。较短 code 表示以较少数据即可学习读出。

结果依 block schedule、优化器和 probe 类。MDL 是操作化的读出复杂度，不是模型内部概念的 Kolmogorov complexity。

## 5.7 Direction 与 Subspace

均值差方向

$$
v_{mean}=\mathbb E[H\mid Y=1]
-\mathbb E[H\mid Y=0]
$$

混合类内协方差、模板和 norm 差异。线性分类器方向、Fisher LDA、PCA 与 partial least squares 解决不同目标，所得“概念方向”不应互换。

多类别或连续变量通常占据子空间 $V\in\mathbb R^{d\times k}$。比较子空间时可用 principal angles，而不是逐向量匹配；子空间内任意正交旋转可能代表同一读出族。

方向主张应至少跨模板、主题、语言和 position 稳定，并明确符号、归一化与中心化。

## 5.8 表示相似性与 CKA

给同一 $n$ 个刺激在两个系统中的中心化表示 $X\in\mathbb R^{n\times p}$、$Y\in\mathbb R^{n\times q}$，linear CKA 为

$$
\operatorname{CKA}(X,Y)
=\frac{\|X^\top Y\|_F^2}
{\|X^\top X\|_F\,\|Y^\top Y\|_F}.
$$

该式要求 $\|X^\top X\|_F>0$ 且 $\|Y^\top Y\|_F>0$；若某组中心化表示全为零，linear CKA 未定义，应报告退化表示而不是置零。CKA 对各向同性缩放和正交变换不变，但不对一般可逆线性变换不变。CCA、Procrustes、RSA 和 CKA 忽略的对称性不同；选择指标等于声明哪些差异不重要。

相似性还依刺激集。两个模型在一个数据集上 CKA 高，不保证在新 domain 仍高，也不保证下游函数相同。应在多个独立 stimulus sets 上重复并联系行为差异。

## 5.9 可视化不是几何检验

t-SNE、UMAP 和二维 PCA 会丢失高维关系。t-SNE/UMAP 的全局距离、簇面积和簇间空白尤其不能直接解释；seed 与超参数可改变图形。

可视化应配套：

- held-out 分类或回归；
- 邻域保持率；
- 类内/类间距离及 bootstrap；
- 在原高维空间计算的子空间角或 CKA；
- 对长度、位置等混杂残差化后的结果。

二维颜色分开只适合作为探索图。

## 5.10 Identifiability 与 probe equivalence

若多个方向 $w$ 在数据支持集上给近似相同预测，probe 不能识别唯一方向。高维共线、过参数化和标签 shortcut 都会扩大等价类。

可通过以下方式量化：

- 多个 probe seeds 的方向夹角与预测一致率；
- 正则强度改变后的方向稳定性；
- 在反事实数据上的 disagreement；
- 对 null directions 的敏感性；
- 子空间而非单向量的可重复性。

“模型有一个 $Y$ 方向”只有在允许等价变换后仍能稳定定位，且该方向具有独立功能预测时才是强主张。

## 5.11 Amnesic 与 causal probing

投影删除 probe 子空间：

$$
h'=(I-P_V)h
$$

并观察行为变化，试图从可读性推进到使用证据。但该操作同时面临：

- **completeness**：是否真的删除了目标属性的全部可读表示；
- **selectivity**：是否尽量保留非目标属性；
- **off-manifold**：$h'$ 是否像自然模型 activation；
- **redundancy**：目标信息是否可由其他方向恢复；
- **downstream compensation**：原模型是否用不同非线性读取。

可分别训练 post-intervention adversary 测剩余目标可读性，并训练多组非目标 probes 测 collateral damage。[Canby 等（2025）的系统评估](https://aclanthology.org/2025.ijcnlp-long.47/)显示，completeness 与 selectivity 在所测方法和模型中存在经验权衡；这是一项受协议限制的结果，不是所有表示的普遍定理。

必要 baselines 包括 matched random subspace、相同维数与 norm 的投影、只删除表面混杂方向、以及可逆恢复或 counterfactual insertion。

## 5.12 从 probe 到机制的桥梁

一个较完整的链条是：

1. probe 在 group-held-out 数据上读出 $Y$；
2. control tasks 排除记忆与表面 shortcut；
3. 方向在 seeds 和合理正则区间内稳定；
4. 下游 weight/Jacobian 对该子空间敏感；
5. 小幅、selective 干预按剂量改变目标行为；
6. matched controls 不产生同样效应；
7. 干预效应在新模板上符合预注册预测。

即使完成这些步骤，通常也只得到“一个被模型使用的表示子空间”，还未给出其上下游完整 circuit。

## 5.13 方法审计表

| 方法 | 问题/对象 | 操作与估计量 | 必要控制 | 能支持 | 不能支持与失效 |
|---|---|---|---|---|---|
| linear probe | 属性是否线性可读 | 冻结表示训练线性模型；held-out risk | group split、表面 baseline、control task | 指定类下可解码性 | 下游使用、唯一方向 |
| nonlinear probe | 属性能否被较强读出器恢复 | MLP/核方法；风险与学习曲线 | 容量、样本量、随机表示 | 更宽函数类的可提取性 | 简单显式表示 |
| MDL probe | 读出学习需要多少数据 | online code length | schedule、优化、baseline | 操作化读出复杂度 | 内部算法复杂度 |
| CKA/CCA | 两组表示在何种不变性下相似 | 样本几何统计 | 多数据集、行为对照 | 指定度量的表示相似 | 功能或机制等价 |
| amnesic probing | 删除读出子空间是否改行为 | 投影；行为差与剩余 decodability | random subspace、selectivity、manifold | 指定投影的干预效应 | 唯一自然中介 |

probe 与几何路线的严格结论是“某信息以何种复杂度可从某表示读出，以及表示在指定对称性下怎样相似”。把它写成“模型知道并使用”会越过证据边界。
