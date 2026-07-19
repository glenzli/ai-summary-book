# 符号与术语

| 符号或术语 | 本卷含义 |
|---|---|
| token | 固定 tokenizer 词表中的离散 ID；不必对应完整字符或词 |
| context | chat template 渲染、截断并分词后实际送入模型的序列 |
| hidden state | 某位置在 Transformer 层中的向量表示 |
| residual stream | 残差连接持续读写的表示通道 |
| logit $z_i$ | token $i$ 在 softmax 前的实数分数 |
| $p_\theta(y_t\mid y_{<t},c)$ | 参数为 $\theta$ 的模型在条件 $c$ 与前缀下的下一 token 分布 |
| prefill | 对输入上下文并行前向并建立 KV cache 的阶段 |
| decode step | 基于已有缓存计算并选择下一 token 的一步 |
| KV cache | attention 层保存的历史 key/value 张量 |
| temperature | softmax 前缩放 logit 差异的参数 |
| top-k / top-p | 截断候选集合并重新归一化的采样规则 |
| stop sequence | 运行时用于终止返回的 token 或文本序列；可能跨 token 边界 |
| latent state | 扩散或流模型迭代更新、最后再解码为媒体的连续表示 |
| denoising step | 扩散采样中对整个带噪状态的一次更新 |
| guidance | 把条件预测与基线预测组合以加强条件影响的机制 |
| tool proposal | 模型生成的结构化调用候选；本身没有执行外部操作 |
| commit boundary | 外部副作用越过后不能靠取消当前请求自动撤回的边界 |
| idempotency key | 让执行器识别同一逻辑写操作重试的键 |
| trace | 按时间排列的请求构造、模型计算、运行时处理与外部事件记录 |

同一术语在不同生成范式中可能指不同的“步”。本卷始终显式区分 decode step、denoising step、integration step 与输出文本中书写的 reasoning step。
