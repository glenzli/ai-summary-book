# 第六章 神经元、Feature 与自动解释

单位级解释通常从“哪些输入使这个内部单位激活”开始，再尝试用自然语言概括。它适合大规模发现候选 feature，也最容易把 top examples 的相关模式误命名为模型中的概念实体。

## 6.1 先定义单位与采样总体

给定单位 $u$、输入 $x$ 和位置 $p$，记录

$$
a_u(x,p).
$$

$u$ 可以是 MLP neuron、residual direction、head result projection 或 SAE latent。对每一种都要声明 normalization、符号、位置选择和是否聚合多个 token。

activation dataset 应从预先说明的语料分布 $D_A$ 抽样，并保留：

- 最大正值、最小值与接近零样本；
- 全体分位数与 firing frequency；
- 语言、文档、token type、位置和上下文分层；
- 同文档相关性与重复片段；
- 原模型成功和失败样本。

只看 top-$k$ 相当于条件在极端尾部，不能描述典型 activation。

## 6.2 Top examples 的选择偏差

假设单位同时响应常见模式 $A$ 与罕见但幅度更高的模式 $B$。top-$k$ 可能几乎全是 $B$，随机正 activation 样本却以 $A$ 为主。说明必须区分：

- **trigger coverage**：哪些输入条件能激活；
- **activation magnitude**：各条件下幅度；
- **population prevalence**：各条件在语料中占比；
- **behavioral effect**：激活后对目标有什么作用。

这四项不能由一张排序截图同时回答。

## 6.3 标签是可证伪预测

若把 $u$ 命名为“日期单位”，说明 $E_u$ 应预测：

1. 未见日期格式仍产生高 activation；
2. 匹配数字长度和位置但非日期的文本较低；
3. 同一字符串在日期与编号上下文中响应不同；
4. 说明可预测 held-out activation 的排序或分箱；
5. 若声称功能，还应预测插入/删除 $u$ 对日期相关行为的方向。

高质量说明包含正条件、排除条件、上下文依赖和已知失败边界。越宽泛的标签越容易事后解释，也越难证伪。

## 6.4 Polysemanticity 与分布式表示

一个 neuron 可在多个难统一的模式上激活，原因可能是 superposition、上下文门控、样本簇遗漏或人类标签粒度不当。可以对高 activation 样本做聚类，但簇数与 embedding 模型会影响结果。

反过来，一个属性可能沿方向

$$
v=\sum_i\alpha_i e_i
$$

或子空间分布，任何单 neuron 都不选择性。单 neuron ablation 无效不证明信息不存在；单 neuron 可命名也不证明该坐标是最自然 feature 基。

## 6.5 Input-side 与 output-side 语义

对 MLP neuron $i$：

- input-side：何时 $u_i=\phi(w_{in,i}^\top h+b_i)$ 增大；
- output-side：$u_iw_{out,i}$ 向 residual 写入什么。

对目标方向 $q$，直接 effect 为

$$
e_i^{direct}=u_iq^\top w_{out,i}.
$$

一个单位可检测 $A$ 却写入抑制 $A$ 的方向；“检测器”与“促进器”不能由 activation 样本混同。对 SAE feature 同样要分别研究 encoder trigger 与 decoder effect。

## 6.6 自动解释的发现—验证分离

一个可审计流程是：

1. 从 discovery corpus 收集高、中、低 activation 样本；
2. explainer 根据其中一部分提出若干竞争说明 $E_1,\ldots,E_k$；
3. 依据说明生成正例、负例和边界例；
4. 在未给 explainer 的 validation corpus 测 activation 预测；
5. 主动搜索高置信反例并修订；
6. 锁定说明后，在 final test corpus 只评一次；
7. 若提出功能主张，另做内部干预。

生成说明与评分若使用同一模型、同一 prompt 模板和同一 examples，会产生共同偏差。应至少使用独立 held-out 数据，并比较不同 grader 与人类复核。

## 6.7 自动说明的估计量

令说明 $E$ 对样本给预测 $\hat a_E(x,p)$。可报告：

- Spearman rank correlation；
- activation quantile 的 AUROC/AUPRC；
- calibration error；
- top-$k$ precision 与随机抽样 recall；
- 正反合成对的配对效应；
- 反例上的最坏组性能。

单一“simulation score”会把阈值、类别平衡和 grader 误差隐藏。评分应与简单 baselines 比较，如 token identity、词频、位置、字符模式和 nearest-neighbor retrieval。

自动说明预测 activation，支持 input-side 语义；除非评分目标本身包含行为干预，它不支持 downstream mechanism。

## 6.8 反例搜索与竞争说明

解释科学的关键不是生成更流畅的标签，而是区分相近假说。例如单位 top examples 都是 Python 代码，可竞争：

- 对 Python 语言本身响应；
- 对缩进响应；
- 对冒号后的换行响应；
- 对 GitHub 文档风格响应。

为每对假说构造使其预测相反的样本，并测

$$
\Delta_{E_i,E_j}
=\mathbb E[a_u(X)\mid E_i\text{ predicts high},E_j\text{ predicts low}]
-\mathbb E[a_u(X)\mid E_i\text{ predicts low},E_j\text{ predicts high}].
$$

无法产生区分性预测的两个标签在当前数据下属于同一解释等价类。

## 6.9 Concept Activation Vectors

TCAV 类方法从人类定义的概念正例与对照样本训练方向 $v_C$，再测目标沿该方向的导数

$$
\operatorname{TCAV}_{C}
=\Pr_{x\sim\mathcal D_{\mathrm{eval}}}
[\nabla_hS(x)^\top v_C>0].
$$

它把概念监督引入内部分析。必要控制包括多个随机对照集、概念数据独立性、层选择校正及相关概念的联合分析。

TCAV 是“概念方向上的敏感性”而非该概念的独立因果份额；概念 directions 相关时，各分数不能相加。

## 6.10 从语义说明到功能验证

若说明声称 feature $u$ 参与行为 $B$，最低验证链为：

1. $E_u$ 在 held-out 输入预测 activation；
2. activation 在控制混杂后预测 $S$；
3. 删除 $u$ 在目标样本上降低 $S$，matched units 不产生同等 effect；
4. 激活 $u$ 以剂量方式提高或抑制 $S$；
5. 非目标行为与语言质量副作用可接受；
6. 上游 trigger 和下游 reader 与说明一致；
7. 在新模板与模型 checkpoint 上复验或明确不稳定。

steering 成功只证明控制能力。若自然 activation 与行为无条件相关性弱，或删除无效，不能反过来写成正常机制已被确认。

## 6.11 抑制、负值与无激活

研究常偏好高正 activation，但抑制同样重要。线性 direction 可有正负投影；ReLU/TopK SAE latent 虽非负，其 decoder vector 可降低目标 logit。

“未激活”也有多种含义：低于阈值、被 TopK 截断、preactivation 为负、或相对基线下降。报告 firing threshold 和 censoring 规则，避免把算法零值解释成概念绝对不存在。

## 6.12 覆盖率与规模化

自动解释的规模化至少有四个分母：

- 被分析单位总数；
- 有足够 firing samples 的单位数；
- 说明通过预测阈值的单位数；
- 进一步通过功能干预的单位数。

只展示最佳 feature 会高估进展。应发布完整分数分布、失败类型和资源成本。从 [Bills 等（2023）的 neuron 说明与模拟](https://openai.com/index/language-models-can-explain-neurons-in-language-models/)到 [Marin-Llobet 与 Ferrando（2026）的 Agent 化竞争假说搜索](https://arxiv.org/abs/2605.01555)，自动解释工作展示了可扩展发现与迭代反例搜索的可能，但尚未建立“大部分大型模型计算可被自然语言说明覆盖”的结论。

## 6.13 方法审计表

| 方法 | 问题/对象 | 操作与估计量 | 必要控制 | 能支持 | 不能支持与失效 |
|---|---|---|---|---|---|
| top activation inspection | 什么输入与单位共现 | 排序样本、分位数 | 随机/中值样本、分层、去重 | 极端响应模式 | 典型语义、下游作用 |
| neuron label | 一条规则能否概括响应 | 合成正反例；rank/AUROC | 竞争说明、surface baselines | 可泛化 input-side 说明 | 单义性、机制功能 |
| auto-interpretation | 能否规模化生成与评分说明 | explainer + simulator | 数据隔离、多 grader、人审 | 说明预测能力 | 模型忠实性与覆盖闭环 |
| TCAV | 人类概念方向是否影响局部目标 | 训练 $v_C$；方向导数比例 | 多对照、相关概念、层校正 | 概念方向敏感性 | 独立因果份额 |
| feature intervention | 单位是否有控制作用 | delete/insert/scale；$\Delta S$ | random units、剂量、副作用 | 指定操作的功能效应 | 正常运行唯一依赖 |

神经元与 feature 说明的研究产物应是“可预测、可反驳的单位假说”，而不是一批动听名称。下一章加入显式内部赋值，讨论这些单位在指定干预下是否真正改变行为。
