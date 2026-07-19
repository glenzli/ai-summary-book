# 第十二章 评估、案例与开放问题

可解释性方法最大的共同难题是缺少普遍 ground truth。真实大模型没有附带人类可读源代码；研究者既要发现机制，又要评估自己的发现。本章给出跨路线评估框架，并以事实回忆为例展示如何组合证据。

## 12.1 六个评估维度

1. **predictiveness**：解释能否预测新输入上的 activation 或行为；
2. **faithfulness**：解释是否对应原模型实际计算；
3. **completeness**：解释覆盖目标行为多少效应；
4. **stability**：对模板、语料、seed 和方法超参数是否稳定；
5. **scalability**：能否应用到更大模型、长上下文和复杂行为；
6. **usefulness**：是否帮助人发现错误、修复模型或做出更好判断。

单一分数很难同时覆盖六项。论文应明确优化哪一项、牺牲哪一项。

## 12.2 有 Ground Truth 的环境

可在以下对象上获得较强机制真值：

- 人工写出的 toy network；
- 编译已知程序得到的 Transformer；
- 合成算法任务；
- 植入已知 backdoor 或目标的模型；
- 训练时结构受约束的 sparse model。

这些环境适合验证方法能否恢复已知 circuit，却可能比自然语言大模型简单。方法在 toy task 成功是必要证据之一，不足以证明可扩展。

## 12.3 Behavioral Fidelity

若解释模型或子回路 $E$ 近似原模型 $M$，至少比较

$$
\mathbb E_{x\sim D}
[d(F_E(x),F_M(x))],
$$

其中 $d$ 可以是 logit KL、logit difference error、argmax disagreement 或任务损失。只在挑选案例上匹配没有代表性；应包含原模型成功和失败样本。

## 12.4 Intervention Fidelity

更强评估让解释和原模型接受对应干预 $I$：

$$
F_E(I_E(x))
\approx
F_M(I_M(x)).
$$

如果 attribution graph 能匹配正常输出，却在 feature patch 后响应相反，它可能只是局部拟合。干预集合应包括增强、抑制、组合和分布内替换。

## 12.5 Human Usefulness

让受试者使用解释完成：

- 预测模型在哪些输入失败；
- 找到触发 backdoor 的条件；
- 选择有效修复或 steering；
- 判断模型依据是否与任务相符。

应与无解释、随机解释、简单行为示例和更耗时信息基线比较。人觉得解释“有道理”不是 usefulness 的充分指标。

## 12.6 稳定性与多重发现

可解释性搜索会检查海量 neurons、heads、features 和 prompts，偶然相关不可避免。应使用：

- discovery/validation 分离；
- 多 seed 的 SAE/probe；
- 预先注册的主要行为指标；
- 随机组件与随机方向 baseline；
- 跨模板、语言和实体复验；
- 公开全部扫描结果或适当多重比较控制。

## 12.7 案例：事实回忆

目标问题：“法国的首都是？”目标指标为 Paris 与竞争城市的 logit difference。

一个完整研究链可以是：

1. **行为集**：多个国家、关系和 paraphrase，另设关系反转与虚构实体对照；
2. **logit lens**：观察答案方向在哪些层变得可读；
3. **probe**：检测 subject、relation、object 信息的层/位置；
4. **causal tracing**：扰动 subject 后 patch clean activation，定位恢复路径；
5. **feature analysis**：检查相关 MLP/SAE features 的正反样本和 output effect；
6. **circuit graph**：连接 subject feature、relation-conditioned attention、object/output feature；
7. **intervention**：替换 subject 或 object feature，预测答案怎样改变；
8. **held-out**：在未参与发现的关系、语言和模板上验证。

如果只完成第二步，结论是“答案可在此层读出”；完成四至七步后，才可能提出局部机制解释。

## 12.8 替代解释

同一观察常有多个故事。事实 patch 恢复可能因为：

- patch 恢复了主体身份；
- patch 恢复了词频或语法；
- patch 让 activation 回到自然分布；
- patch 同时携带关系和答案。

对照实验应专门区分这些解释。好的研究不是给最顺畅的故事增加更多热图，而是设计最容易让故事失败的反例。

## 12.9 当前开放问题

- 怎样定义 feature，使其跨输入、层和模型稳定？
- 怎样量化 circuit 的 completeness，而不把所有计算重新放回图中？
- replacement model 的机制忠实性如何标准化？
- SAE 的 feature splitting、absorption 和误差如何影响安全结论？
- 如何解释动态路由、MoE、多模态和扩散模型？
- 如何从局部 prompt graph 推广到全局算法？
- 模型能否对监测与解释方法进行策略性规避？
- 怎样让解释真正改善人类判断，而非制造过度信任？

这些不是边角缺陷，而是该学科尚未严格闭合的核心。

## 12.10 本卷结论

可解释性不是一条从 activation 热图直接通向“模型在想什么”的捷径。行为、梯度、attention、probe、feature、intervention 与 circuit 分别提供不同证据；稀疏表示和 attribution graph 提高了规模化可能，也引入替代模型误差。

当前最可靠的实践是 triangulation：用多种方法提出同一可证伪机制，在 held-out 行为、内部读出和干预响应上共同验证，并把未解释部分明确留下。下一卷不再重复这些技术，而改用第一人称追问：当一个由训练数据、概率和外部系统共同构成的模型说“我”时，这个“我”究竟指向什么。
