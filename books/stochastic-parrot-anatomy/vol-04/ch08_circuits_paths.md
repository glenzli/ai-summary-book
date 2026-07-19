# 第八章 Circuits、路径与 Attribution Graphs

机制解释的核心单位通常不是单个神经元，而是一组组件及其相互作用。回路研究试图把输入、features、attention 路由、MLP 变换和输出 logits 连接成一个可检验的计算图。

## 8.1 什么是 Circuit

相对于行为 $B$ 和输入分布 $D$，circuit 是原模型计算图的一个子图 $C$，它应满足某种：

- **faithfulness**：保留 $C$ 后能近似原模型目标行为；
- **completeness**：删除 $C$ 后行为显著受损；
- **minimality/parsimony**：去掉其中重要部分会降低解释力；
- **interpretability**：节点和边可由可检验规则描述。

这些目标可能冲突。更小的 circuit 更易读，却可能遗漏冗余路径；高保真子图可能仍有数千节点。

## 8.2 QK 与 OV Circuits

attention head 可以分成：

- QK circuit：哪些 source position 获得较高 attention；
- OV circuit：被选 source 的表示怎样转换并写回 residual。

组合 heads 时，前一 head 写入的方向可以被后一 head 的 query/key 或 value 读取，形成 composition。只看每个 head 独立 pattern 会漏掉这种跨层依赖。

## 8.3 Induction Head 范例

在模式 `... A B ... A` 中，induction head 倾向在第二个 `A` 处预测 `B`。一种机制分解是：

1. previous-token head 把前一个 token 信息写到当前位置；
2. induction head 的 query 在当前 `A` 寻找过去表示“前一个 token 是 A”的位置；
3. 其 value 路径复制该位置后续 token `B` 的信息；
4. unembedding 提高 `B` logit。

这个案例重要，不是因为所有语言行为都归结为 induction，而是它展示了行为签名、权重分析、activation 和 ablation 如何汇合为可组合解释。

## 8.4 Path Patching

普通 activation patching 会同时影响所有下游路径。path patching 只让 source 的 clean effect 沿指定 edge 或路径进入 target，其余输入保持 corrupt 值。

实现通常需要缓存多次 run，并精确控制 sender、receiver 和中间节点。它能区分“head A 的输出有用”和“head A 通过 head B 被使用”，代价是干预语义更复杂，容易产生不一致 activation 组合。

## 8.5 Causal Scrubbing

给定抽象计算图假说 $H$，causal scrubbing 把模型节点映射到假说变量，并对假说认为不相关的输入差异进行 resample。若 scrub 后模型仍保持目标行为，说明模型对这些变化的不变性与假说一致。

它是严格测试假说充分性的方向，但结果依：

- 抽象图如何选择；
- 节点到变量的映射；
- resampling distribution；
- 行为保留指标。

通过测试说明假说与干预相容，不证明不存在另一个同样相容的机制。

## 8.6 从 Attribution 到 Graph

对固定输入，可把 embeddings、内部 features 和输出 logits 作为节点，用梯度线性化、direct attribution 或替代模型构造边权。再按目标 logit 的贡献剪枝，得到可浏览子图。

剪枝阈值决定图的稀疏性。应报告：

- 被保留输出 effect 比例；
- reconstruction/error nodes 的贡献；
- 不同阈值下结论是否稳定；
- 正负路径与 suppression；
- 图是否只对单 prompt 局部成立。

## 8.7 Replacement Model

原始 MLP 神经元常 polysemantic。近期 circuit tracing 方法用 transcoder 或 cross-layer transcoder 近似原 MLP 输出，再以稀疏 feature 作为节点。

抽象地，若原 MLP 输出为 $y^\ell$，替代模型给出

$$
\hat y^\ell
=\sum_{k\le\ell}
W_{dec}^{k\to\ell}a^k,
$$

并最小化 reconstruction error 加 sparsity penalty。得到的 feature graph 更易读，但解释对象先是替代模型。必须比较：

- 原模型与替代模型输出一致率；
- patch/steering 在两者中的响应是否一致；
- reconstruction error 是否携带目标行为；
- 冻结 attention pattern 等近似遗漏什么。

## 8.8 Local 与 Global Circuit

局部 attribution graph 固定 prompt 和许多非线性状态，较容易稀疏化；全局 circuit 要跨 prompts 解释 QK 路由与 feature 激活规则，难度更高。

把多个局部图合并时会出现：

- 同一语义由不同 features 实现；
- 相似 feature 标签实为不同计算角色；
- 边在不同输入上改变符号或强度；
- 图规模随输入长度和行为多样性爆炸。

全局化需要聚类、条件规则和 held-out 预测，而不只是把图取并集。

## 8.9 Faithfulness 评估

对 circuit $C$，可比较原模型目标分数 $S_M(x)$ 与 circuit/replacement 分数 $S_C(x)$：

$$
\mathbb E_{x\sim D}
\left[(S_M(x)-S_C(x))^2\right],
$$

以及 argmax 一致率、行为成功率和干预响应一致性。只匹配最终 token 不够：不同机制可以给同一输出。

机制忠实性更强的测试是让两者接受同一组内部 perturbations，比较输出和中间 feature 的响应曲线。

## 8.10 回路不是唯一分解

神经网络存在基变换、冗余和非线性交互。不同节点基、剪枝目标和干预定义可能得到不同但同样有预测力的 circuits。研究目标不是声称找到唯一“真正线路图”，而是找到能压缩、预测并经受干预的机制模型。

## 8.11 结论

circuit 研究把单位级发现组织成计算过程。它的强度来自路径干预和行为保真度，脆弱处则是替代基、剪枝和局部化。下一章专门讨论为什么原始神经元难以作为节点，以及 SAE、transcoder 与稀疏模型怎样改变解释基。
