# 符号与术语

本表只规定卷二中的执行含义。同一词在训练理论、概率分析或可解释性研究中可能有更细定义。

## 输入与模型

| 符号或术语 | 本卷含义 |
|---|---|
| request object | 消息、模型工件、模板/tokenizer、生成配置、限制与可选工具目录的结构化输入 |
| rendered input | chat template 对结构化消息与工具描述生成的确定字节或 token 段 |
| tokenizer artifact | 规范化、pre-tokenization、词表、分词模型、特殊 token 与后处理规则的版本化组合 |
| token | 固定 tokenizer 词表中的离散 ID；不必对应完整字符或词 |
| context | 渲染、分词并执行预算策略后真正进入模型的有效 token 序列 |
| input IDs | 词表 ID 的整数张量，通常形如 [B,n] |
| attention mask | 指明有效位置和允许 attention 边的输入；padding token 不会自动被忽略 |
| position descriptor | position IDs 或等价的位置/旋转参数 |
| residual stream | Transformer 层内由 attention、MLP 与残差连接持续更新的主表示 |
| hidden state | 某批次、位置和层上的表示向量 |
| raw logits | 模型词表投影直接产生、尚未经过运行时处理器的分数 |

## Prefill 与 Decode

| 符号或术语 | 本卷含义 |
|---|---|
| prefill | 对全部提示位置执行因果前向、建立各层 KV 并产生首步 logits |
| KV cache | 每层按逻辑 token 位置保存、供后续 attention 读取的 key/value |
| logical cache length | cache 所代表的有效 token 数；不同于物理页数或分配容量 |
| decode step | 消费一个已选 token、追加其 K/V 并产生下一步 raw logits 的模型调用 |
| selection step | 对当前 logits 应用处理器并确定一个 token；发生在相应 decode step 之前 |
| $z_t$ | 以提示和 $y_{1:t-1}$ 为条件、用于选择 $y_t$ 的 raw logits |
| processed logits | mask、惩罚、温度等按版本化顺序作用后的 logits |
| candidate support | 最终仍具有非零选择概率的 token 集合 |
| decoder state | token 历史、计数、grammar/stop 自动机、RNG 游标和步号 |
| processor order | logit 处理器的有序列表；顺序变化可改变候选和选择 |
| grammar state | 结构化输出约束在当前字节前缀上的自动机状态 |
| RNG counter | 定位一次随机数消费的 stream/subsequence/counter 状态 |

## Streaming 与终止

| 符号或术语 | 本卷含义 |
|---|---|
| incremental detokenizer | 跨 token 保存字节/清理状态并产生可显示增量的解码器 |
| stop matcher | 在 token 或增量字节流上跨边界识别停止序列的状态机 |
| safe prefix | 已不可能成为停止串组成部分、可以不可撤回发送的字节前缀 |
| stream delta | 带请求内单调序号的增量字节事件 |
| backpressure | 客户端消费速度低于服务产生速度时对调度或缓冲形成的压力 |
| finish reason | EOS、stop sequence、长度、取消、deadline、错误等版本化终止枚举 |
| DRAINING | 不再选择 token，但仍有安全字节或终态事件待发送的生命周期状态 |
| terminal state | FINISHED、CANCELLED 或 FAILED；终态之后不得发送普通 delta |

## 其他迭代生成

| 符号或术语 | 本卷含义 |
|---|---|
| latent state | 连续扩散或 flow 反复更新、最后再解码为媒体的整体张量 |
| denoising step | 扩散 sampler 按一个 timestep 更新整个连续或离散状态的一步 |
| scheduler | 把模型预测、当前状态与时间参数组合成下一扩散状态的算法 |
| prediction type | 模型输出被解释为 noise、sample、score、velocity 等哪一种参数化 |
| guidance | 组合有条件与基线预测以修改 scheduler 输入的机制 |
| NFE | number of function evaluations；数值求解中实际网络评估次数 |
| integration step | ODE solver 使用一个或多个向量场评估更新连续状态的一步 |
| discrete diffusion | 在有限状态空间中按扰动与逆转移更新整个离散序列/网格 |
| mask set $M_k$ | 掩码式生成第 $k$ 轮仍未确认或待修订的位置集合 |

## 工具执行

| 符号或术语 | 本卷含义 |
|---|---|
| ToolSpec | 名称、版本、schema、副作用类别、能力要求和重试策略的接口 |
| Proposal | 模型生成的工具名与原始参数字节；本身没有副作用 |
| Invocation | 运行时规范化、验证、授权并绑定主体后准备执行的调用 |
| capability | 运行时授予主体、资源与操作范围的能力，不由模型自我声明 |
| confirmation digest | 用户确认所绑定的规范化 Invocation 摘要 |
| idempotency key | 在规定 scope 和工具版本内识别同一逻辑写意图的键 |
| commit boundary | 外部副作用首次已成为系统状态或无法由当前请求单方面阻止的边界 |
| OUTCOME_UNKNOWN | 调用可能已经提交，但当前运行时没有充分证据判定结果 |
| tool result | 执行器的结构化结果或错误；编码后成为后续模型调用的新输入 |
| trace | 由 trace ID 关联、带父子关系与单调事件序号的一次执行记录 |

## 核心记号

- $x_{1:m}$：提示 token；
- $y_{1:t-1}$：第 $t$ 次选择前已经选出的输出 token；
- $s_k$：某种迭代生成器在第 $k$ 步的完整状态；
- $u_k=f_\theta(s_k,k,c)$：模型对当前状态与条件的输出；
- $s_{k+1}=F(s_k,u_k,k,\xi_k)$：sampler 或运行时更新；
- $c$：固定或版本化的条件编码；
- $\xi_k$：该步可选的显式随机输入。

“decode step”“denoising step”“integration step”和输出文本中写出的 reasoning step 不是同一单位；比较步数前必须先说明状态与更新核。
