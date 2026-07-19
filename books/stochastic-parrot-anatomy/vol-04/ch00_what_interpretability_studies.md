# 第零章 可解释性究竟在研究什么

“打开模型看一看”不是充分的研究问题。现代模型内部有数十亿参数和按 token 产生的海量 activation；把它们全部打印出来只会得到更多数字。可解释性研究的任务，是在明确行为、输入分布、内部对象和允许干预后，提出能够预测新观察并可能被反例推翻的机制或表示假说。

## 0.1 研究设定而不是孤立图像

把一次研究写成

$$
\mathfrak E=(M,\theta,\tau,\mathcal D_{\mathrm{eval}},S,\mathcal U,\mathcal I).
$$

- $M_\theta$：冻结的模型与 checkpoint；
- $\tau$：tokenizer、chat template 和截断规则；
- $\mathcal D_{\mathrm{eval}}$：主张所针对的输入分布；
- $S$：连续行为指标，例如 logit difference；
- $\mathcal U$：允许分析的内部单位及 hook site；
- $\mathcal I$：允许的删除、替换、投影或重采样操作。

模型名称并不足够。同一权重在不同 system prompt、tokenization、解码协议和工具环境中形成不同系统。内部结论也不脱离 $\mathcal D_{\mathrm{eval}}$：在 IOI 模板上稳定的 circuit 不是“全部指代消解机制”。

## 0.2 四类问题与证据上限

可解释性研究常回答四类不同问题：

1. **行为描述**：什么输入使输出指标怎样变化？
2. **信息定位**：某属性能从哪里被指定读出器恢复？
3. **因果归因**：对内部变量施加指定干预会怎样改变输出？
4. **机制解释**：哪些变量以什么组合规则实现一段可重复计算？

“第十二层可以线性预测法语”属于第二类；“投影掉一个法语方向后法语生成率下降”属于第三类；只有继续说明该方向怎样形成、由哪些下游权重读取、在何种输入上失效，才进入第四类。

| 证据层级 | 典型方法 | 能支持的最高结论 |
|---|---|---|
| 描述性 | top activations、热图、相关性 | 单位与模式在样本分布上共同出现 |
| 预测性 | held-out probe、自动说明模拟 | 指定表示包含可泛化读出信息 |
| 干预性 | ablation、patching、steering | 指定操作会改变指定行为 |
| 机制性 | 可组合回路、因果抽象、干预保真 | 某部分计算在目标分布与干预族上由所述结构实现 |

后一级通常需要前一级证据，但没有任何单项实验会自动升级主张。

## 0.3 解释目标与作用域

目标 $S$ 可以是：

- 两个候选 token 的 logit difference；
- 一段答案的条件 log likelihood；
- 拒答、工具调用或格式满足率；
- benchmark 上的连续 margin；
- 某 feature 的 activation；
- 一个 monitor 对未来行为的预测性能。

解释还要声明作用域：

- **instance-local**：一个 prompt 或一条生成轨迹；
- **template-local**：一族结构固定、实体变化的输入；
- **distributional**：明确分布 $\mathcal D_{\mathrm{eval}}$ 上的平均或条件机制；
- **cross-model**：跨 checkpoint、seed、规模或架构。

从局部图推广到全局机制需要新的数据与估计，不是增加几个案例即可完成。

## 0.4 解释作为部分模型

把解释记作 $E$。它可以是自然语言规则、线性方向、因果图、稀疏子回路或 replacement model。解释的价值在于压缩原模型，同时保留与目标有关的预测。

若 $F_M(x)$ 与 $F_E(x)$ 分别是原模型和解释模型的目标输出，可定义行为误差

$$
L_{\mathrm{beh}}(E;\mathcal D_{\mathrm{eval}})
=\mathbb E_{x\sim\mathcal D_{\mathrm{eval}}}
\bigl[d(F_E(x),F_M(x))\bigr].
$$

仅匹配自然运行还不够。令 $\Pi_{\mathcal I}(\cdot\mid x)$ 为给定输入后的干预核，定义干预误差

$$
L_{\mathrm{int}}(E;\mathcal D_{\mathrm{eval}},\Pi_{\mathcal I})
=\mathbb E_{\substack{x\sim\mathcal D_{\mathrm{eval}}\\
I\sim\Pi_{\mathcal I}(\cdot\mid x)}}
\bigl[d(F_E(I_E,x),F_M(I_M,x))\bigr].
$$

这种写法允许干预选择依赖 $x$；若实验先独立抽取干预，则 $\Pi_{\mathcal I}(\cdot\mid x)$ 退化为不依赖 $x$ 的分布。未声明联合抽样规则时，$L_{\mathrm{int}}$ 不是完整的 estimand。

一个替代模型可在 $L_{\mathrm{beh}}$ 很低时仍有很高的 $L_{\mathrm{int}}$：它拟合了输入输出，却没有复制被干预时的机制响应。

## 0.5 Faithfulness、Completeness 与可理解性

本卷区分三个经常混用的目标：

- **faithfulness**：解释在其声明的输入与干预范围内是否跟随原模型计算；
- **completeness**：解释是否覆盖目标行为所需的相关效应，而不是只挑选最易命名部分；
- **human comprehensibility**：解释是否足够压缩，能被人检查和使用。

faithfulness 没有脱离测试协议的单一数值。保留子图、删除补图、重放 intervention 和预测 held-out edge effect 测的是不同方面。completeness 也不等于把所有节点放回图中；那会得到原模型副本而失去解释压缩。

“完整而可理解”之间存在真实张力。论文应画出稀疏度与保真度曲线，而不是只报告一个看起来最漂亮的阈值。

## 0.6 可识别性：我们能否区分多个解释

若两个解释 $E_1,E_2$ 对所有 $x\in\operatorname{supp}(\mathcal D_{\mathrm{eval}})$ 和 $I\in\mathcal I$ 都给出相同预测，

$$
F_{E_1}(I,x)=F_{E_2}(I,x),
$$

则在当前研究设定下它们不可区分。不能据此宣布其中一个是唯一真实机制。

不可识别性的来源包括：

1. **有限数据**：测试没有覆盖能区分假说的输入；
2. **有限干预**：只做节点删除，未测试路径或组合效应；
3. **参数对称性**：隐藏空间可在保持函数不变时置换、缩放或换基；
4. **冗余机制**：多个路径在自然分布上互相备份；
5. **分析器非唯一**：不同 probe、SAE 或剪枝阈值得到不同坐标；
6. **过强映射类**：若允许任意复杂的对齐函数，几乎任何表示都可能被映射为目标概念。

因此“可识别”必须写成：在给定假说类 $\mathcal H$、分布 $\mathcal D_{\mathrm{eval}}$、干预族 $\mathcal I$ 与容差 $\varepsilon$ 下，是否只有一个等价类通过检验。约束对齐映射的容量不是审美选择，而是防止解释变得空泛的必要条件。

## 0.7 内部对象不是天然概念

neuron、attention head、residual direction、SAE latent 和 attribution-graph node 都是架构或分析器定义的单位。它们不天然对应“否定”“法国”“欺骗”或“推理”。

给单位命名应转化为可证伪命题：

$$
E(u)\Longrightarrow
\begin{cases}
\text{新正例上 activation 或 effect 增大},\\
\text{匹配负例上不增大},\\
\text{干预响应符合所述功能},\\
\text{明确列出的边界条件下允许失败}.
\end{cases}
$$

无法预测反例的名称只是事后故事。

## 0.8 Causal 不等于 Natural

神经网络前向图是确定性计算图，因此可以精确设置内部节点并测量输出变化。这给出了**干预因果效应**。但干预值可能不来自自然前向分布，节点集合也可能不是唯一的高层变量。

所以应区分：

- “在 $a\leftarrow a'$ 操作下，$S$ 改变了多少”；
- “现实输入变化通过自然中介 $a$ 产生多少效应”；
- “$a$ 是人类概念 $C$ 的唯一内部实现”。

第一句可由实验直接估计；后两句分别需要关于干预有效性、无混杂、表示映射和唯一性的额外假设。

## 0.9 一份统一的方法记录

对任一方法 $A$，至少保存以下记录：

| 字段 | 记录内容 |
|---|---|
| question | 要区分的具体假说 |
| unit | 层、位置、张量形状、hook 前后关系 |
| operation | 观察、训练、微分或赋值规则 |
| estimand | 目标总体量，例如 $\mathbb E_{x\sim\mathcal D_{\mathrm{eval}}}[\Delta S(x)]$ |
| estimator | 样本统计量、方差与置信区间 |
| controls | 负对照、随机基线、替代解释与副作用 |
| claim ceiling | 该设计能支持的最强语句 |
| failure | 结果在何种情况下不再有效 |

“估计量”与“估计器”不同：前者是想知道的总体量，后者是有限样本上的计算规则。只报告热图而不定义二者，难以比较方法和复现结论。

## 0.10 主要路线及其互补关系

| 路线 | 直接对象 | 首要产物 | 主要缺口 |
|---|---|---|---|
| 行为反事实 | 输入与输出 | 行为签名 | 内部算法不可识别 |
| 梯度归因 | 局部导数或路径积分 | 敏感性/归因图 | 基线、饱和与交互 |
| attention/readout | 路由权重与 residual | 候选信息流 | 可读出不等于使用 |
| probe/几何 | activation 样本 | 可解码属性与空间关系 | probe 容量与混杂 |
| neuron/feature | 单位 activation | 语义假说 | polysemanticity 与选择偏差 |
| intervention | 内部节点 | 指定操作下的效应 | off-manifold 与干预语义 |
| circuits/path | 组件子图 | 组合机制 | 剪枝、冗余与局部性 |
| SAE/transcoder | 学习出的稀疏基 | feature 与替代图 | 非唯一性与重构误差 |
| training dynamics | checkpoint 序列 | 机制形成轨迹 | 对齐与跨 seed 稳定性 |
| CoT/monitoring | 生成过程与监测信号 | 可观测过程指标 | 不忠实与对抗适应 |

没有一条路线单独构成“模型显微镜”。可靠工作通常从行为签名出发，用相关方法定位，再用干预与 held-out 预测检验，最后量化未解释残差。

## 0.11 研究检查表

- 模型、checkpoint、tokenizer、精度与代码版本是否固定？
- 目标行为和连续指标是否在看内部张量前定义？
- discovery、模型选择与最终验证数据是否分离？
- 内部对象和 hook site 是否精确到层、位置及规范化前后？
- 干预值是否来自自然分布，若不是是否量化偏离？
- 是否主动检验至少一个替代解释？
- 是否同时报告成功样本、失败样本和效应分布？
- 主张是否限定到 $\mathcal D_{\mathrm{eval}}$、$\mathcal I$ 和模型版本？
- 是否区分读出、控制、正常使用与完整机制？
- 跨 seed 或规模结论是否先定义了可接受的对齐等价类？

本章给出的不是可解释性的终极定义，而是一套证据语法。后续章节各自采用这套语法，避免把不同方法产生的对象强行混合为同一种“解释”。
