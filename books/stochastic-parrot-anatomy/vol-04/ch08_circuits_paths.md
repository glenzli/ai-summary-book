# 第八章 Circuits、路径与 Attribution Graphs

机制解释的核心单位往往不是单个 neuron，而是一组按计算顺序连接的组件。circuit 研究试图找到一个比原模型更小、能预测目标行为及干预响应的计算子图。它是一项有损建模任务，不是从模型中读取唯一线路图。

## 8.1 相对于任务定义 Circuit

令原计算图为 $G=(V,E)$，目标行为为 $(\mathcal D_{\mathrm{eval}},S)$。候选 circuit

$$
C=(V_C,E_C),\qquad C\subseteq G
$$

还必须附带补图如何处理的规则 $R$：置零、均值、resample、冻结或由 replacement model 近似。没有 $R$，仅给节点集合不能执行。

一个 circuit 可追求：

- **faithfulness**：保留 $C$ 后近似原模型目标行为与干预响应；
- **completeness**：$C$ 覆盖目标相关计算，删除它显著损害行为；
- **minimality**：移除内部元素会显著降低前两项；
- **interpretability**：节点和边有可检验的功能规则；
- **stability**：对样本、阈值、seed 与合理基线稳定。

这些目标冲突，必须报告 Pareto 曲线而非单一“最佳图”。

## 8.2 Keep 与 Remove 两种评估

设 $S_M,S_C,S_{base}$ 分别为原模型、只保留 circuit 和基线模型 score。可定义归一化 keep fidelity

$$
F_{keep}
=\frac{\mathbb E_{x\sim\mathcal D_{\mathrm{eval}}}
[S_C(x)-S_{base}(x)]}
{\mathbb E_{x\sim\mathcal D_{\mathrm{eval}}}
[S_M(x)-S_{base}(x)]},
$$

该比值只在分母远离零且 score 方向一致时有稳定含义，应同时报告原始差与逐样本分布。删除 circuit 的损伤为

$$
F_{remove}
=\mathbb E_{x\sim\mathcal D_{\mathrm{eval}}}
[S_M(x)-S_{M\setminus C}(x)].
$$

keep 高说明 $C$ 在所用补图规则下较充分；remove 大支持必要性。二者可能同时不高，因为冗余路径、非线性交互或补图替换不自然。

“completeness”在文献中命名并不统一，论文应直接给操作定义而不是只报词。

## 8.3 QK/OV Composition

attention head 的 QK 决定选择，OV 决定读写。跨层 composition 可分为：

- Q-composition：前一组件写入后层 query 使用的方向；
- K-composition：写入后层 key 标记；
- V-composition：写入被后层 value 读取；
- residual/logit composition：多个模块沿共同输出方向相加或抵消。

矩阵乘积可识别潜在通路，activation 与 path intervention 才说明具体输入上启用。权重层的高 composition score 不应单独命名为行为 circuit。

## 8.4 Induction head 作为方法范例

在 `... A B ... A` 中，一类 induction 机制可分解为：

1. 上游组件把 previous-token 信息写到当前位置；
2. induction head 在当前 `A` 查询过去“前一个 token 是 A”的位置；
3. OV 路径复制过去配对中的 `B` 信息；
4. residual 写入提高 `B` 的 logit。

这个案例的价值在于四类证据汇合：参数组合、行为签名、activation pattern 与 ablation。它不推出自然语言模型的全部 in-context learning 都由同一回路实现。

训练与规模研究中即使同类算法复现，具体 head 编号也可能改变；应比较功能角色而非坐标身份。

## 8.5 Edge 与 path patching

节点 patch 改变 sender 的全部下游影响。edge patch 只把 sender $u$ 对 receiver $v$ 的输入从 clean 替换进 corrupt run。对 path $P=(u_0,\ldots,u_k)$，连续隔离其传递。

路径 effect 不是天然可加 credit。若两条路径在 receiver 非线性交互，分别 patch 的和不等于联合 patch。至少报告：

- 单边、联合边与完整节点 effects；
- patch 顺序；
- residual identity 是否算 edge；
- attention pattern 是否冻结；
- 其他 sender 输入来自哪一个 run。

## 8.6 Causal scrubbing 与高层假说

给定高层计算图 $H$ 以及模型节点到高层变量的映射，causal scrubbing 对 $H$ 认为无关的输入差异重采样。如果 scrub 后目标行为保持，说明模型对这些变化的不变性与 $H$ 相容。

该方法检验的是假说充分性，而不是发现唯一图。结果依：

- 高层变量粒度；
- 节点—变量映射；
- resampling distribution；
- 保留行为的 metric；
- 对齐映射容量。

应加入“错误但复杂度相同”的高层图作负对照，并在 held-out intervention pairs 上比较。

## 8.7 自动 Circuit Discovery

自动方法大致分为：

1. **逐节点/边 patch**：精确但前向次数高；
2. **gradient attribution**：用一阶近似筛选 edge；
3. **mask optimization**：学习稀疏门控使目标行为保留；
4. **贪心搜索**：从输出向上游加入高 effect 边；
5. **替代模型图**：在 SAE/transcoder features 上构图。

所有方法都包含搜索超参数。阈值、稀疏正则和候选单位改变图；必须在独立验证集运行真实 ablation/patch，而不是用同一 attribution score 既发现又证明。

梯度筛选受饱和和局部线性化影响，mask optimization 可能找到对训练 prompts 特化的子图，贪心法会遗漏只有联合加入才有作用的边。

## 8.8 Faithfulness、Completeness 与 Minimality

对目标距离 $d$，行为保真误差为

$$
L_{beh}(C)=\mathbb E_{x\sim\mathcal D_{\mathrm{eval}}}
[d(F_C(x),F_M(x))].
$$

令 $\Pi(\cdot\mid x)$ 为给定输入后的干预核，更强的响应误差为

$$
L_{int}(C)
=\mathbb E_{\substack{x\sim\mathcal D_{\mathrm{eval}}\\
I\sim\Pi(\cdot\mid x)}}
[d(F_C(I_C,x),F_M(I_M,x))].
$$

若干预与输入独立，$\Pi(\cdot\mid x)$ 退化为常数核。该联合分布是 circuit faithfulness 主张的一部分，不能只列干预名称而不列抽样规则。

minimality 可用删除每条边后的边际损失分布评估：

$$
m_e=L_{beh}(C\setminus\{e\})-L_{beh}(C).
$$

若大量 $m_e\approx0$，图可能不是最小；但冗余边会使单边 $m_e$ 小而联合删除重要。还需组合或 group ablation。

## 8.9 Replacement model 与误差节点

原 MLP 或 residual 可由稀疏替代模型近似。设原模块输出 $y$，替代输出

$$
\hat y=\sum_k f_kd_k,
\qquad
\varepsilon=y-\hat y.
$$

attribution graph 若忽略 $\varepsilon$，就把未解释计算静默删除。更诚实的图把 reconstruction error 作为显式节点，并测它对目标与干预的 effect。

替代模型质量至少包含：

- activation reconstruction；
- 原模型 loss 增量与 argmax agreement；
- 目标行为保留；
- 原/替代模型的 intervention response；
- error nodes 的正负贡献。

自然输出匹配不保证替代机制匹配。

## 8.10 Attribution graphs

对固定 prompt，可把 input features、内部 sparse features、attention interactions 与 logits 作为节点，用局部 attribution 建边并按目标 effect 剪枝。这类图擅长展示一个运行中的候选信息流。

需要声明近似：

- attention pattern 是否冻结；
- feature interactions 是否只取一阶项；
- cross-layer transcoder 重构哪些模块；
- 剪枝按绝对 effect、正 effect 还是路径累积；
- error term 怎样进入；
- 图只对一个 token 还是整段生成有效。

[2025 年公开的 Circuit Tracing 方法报告](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)展示了大型模型局部 attribution graph 的可扩展构造；其后续 [attention QK 分析](https://transformer-circuits.pub/2025/attention-qk/index.html)也仍以替代模型和局部图为对象。replacement error、动态 attention 与局部化限制没有因此消失，不能将这些结果写成全模型已经完成解剖。

## 8.11 Local 到 Global

局部图固定大量非线性状态，较容易稀疏。全局 circuit 要解释 feature 何时触发、QK 路由怎样随输入变化以及备份路径何时接管。

合并局部图不能简单取并集。可定义 edge 的条件出现率

$$
p_e(c)=\Pr_{x\sim\mathcal D_{\mathrm{eval}}}
[e\in C(x)\mid c(x)=c]
$$

和 effect 分布，进一步学习可检验的 gating rule。若边在不同 context 下改变符号，平均 edge weight 没有明确机制含义。

全局化需要新样本上的图预测、条件规则和失败簇，不是更多可视化。

## 8.12 阈值、搜索与不确定性

一个 circuit 应随稀疏阈值形成路径：

$$
(|E_C(\lambda)|,L_{beh}(C_\lambda),L_{int}(C_\lambda)).
$$

报告整条曲线可以显示结论是否依一个任意阈值。还应对数据 bootstrap、patch baseline、attribution method 和随机 seed 重复搜索。

边稳定性可定义为被选择频率，但高频不等于真实；它只说明在所采样分析扰动下可重复。选择频率低的边可能属于多组等价冗余路径。

## 8.13 回路非唯一与可识别性

不同节点基、补图规则和误差容忍可得到多个行为等价 circuit。若 $C_1,C_2$ 在测试分布与干预族上满足

$$
L_{beh}(C_1)\approx L_{beh}(C_2),
\qquad
L_{int}(C_1)\approx L_{int}(C_2),
$$

就没有证据称其中一个是唯一真实图。更合理的产物是：

- 共享的稳定 core；
- 可互换的冗余 modules；
- 输入条件决定的 branches；
- 尚由 error nodes 承担的未解释部分。

机制解释可由等价类构成，而不必强求坐标级唯一性。

## 8.14 方法审计表

| 方法 | 问题/对象 | 操作与估计量 | 必要控制 | 能支持 | 不能支持与失效 |
|---|---|---|---|---|---|
| hand-built circuit | 已知角色能否组合实现行为 | keep/remove；$F_{keep},F_{remove}$ | 补图基线、失败样本、组合删除 | 目标分布上的组合机制 | 唯一全局图 |
| path patching | sender 是否经 receiver 起效 | edge 隔离；路径 effect | 状态一致性、交互、双向 patch | 指定协议的路径效应 | 自然中介唯一性 |
| causal scrubbing | 高层图的不变性是否匹配模型 | 条件 resample；行为保留 | 错误图、映射容量、held-out | 假说相容性 | 排除所有替代图 |
| automated discovery | 能否高效定位稀疏子图 | attribution/mask/search | 独立真实 patch、阈值曲线 | 候选 circuit 与效率 | 搜索 score 即忠实性 |
| attribution graph | 局部 feature 图怎样影响 logit | replacement + edge attribution | error nodes、attention 近似 | 局部替代机制 | 原模型全局完整解剖 |

circuit 的合格结论不是“发现模型真正线路图”，而是“在声明的行为分布、节点基和干预协议下，这个较小计算模型以量化误差复现了原模型的部分行为与响应”。
