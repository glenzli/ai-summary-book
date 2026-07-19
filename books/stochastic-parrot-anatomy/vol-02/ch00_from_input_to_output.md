# 导论 从输入到输出：一张执行地图

卷一回答模型怎样被设计、训练和部署。本卷把观察尺度缩到一次请求：用户按下发送键之后，哪些字节、token、张量和事件依次出现，哪个组件在每一步拥有决定权，失败又最早可能发生在哪里。

贯穿本卷的第一个请求是：

> 请用一句话解释为什么天空通常呈蓝色。

屏幕上最终只出现一句中文，但这句话不是模型一次性写入界面的。应用先确定请求与上下文，tokenizer 产生整数序列，模型完成 prefill，解码器反复选择 token，增量反分词器产生字节片段，传输层把事件送到客户端。若出现工具调用，运行时还会暂停文本生成，进入具有独立权限和失败语义的外部系统。

本卷把“生成”限定为这条可执行时间线，不借它讨论模型训练所得概率的哲学含义，也不把一次中间激活直接解释成概念。概率来源属于卷三；内部表示的观测与干预属于卷四。

## 0.1 请求不是一段字符串

一次请求至少包含以下版本化对象：

```text
Request = {
    request_id,
    messages,
    model_snapshot,
    template_version,
    tokenizer_version,
    generation_config,
    runtime_limits,
    tool_catalog?,
    client_metadata?
}
```

`messages` 是结构化消息，而不是已经渲染好的 prompt。`generation_config` 决定温度、截断采样、停止条件和输出上限；`runtime_limits` 决定超时、排队、资源与取消策略；`tool_catalog` 只声明当前调用可见的工具接口。模型名称若只是可变别名，就不足以唯一确定 `model_snapshot`。

**定义 0.1（执行实例）**　执行实例是请求对象、所有被解析到的版本化工件、初始随机源状态以及运行时环境标识的组合。两个界面显示相同用户文字，不推出它们属于同一执行实例。

## 0.2 五类状态

为了避免把不同组件压成一个“模型状态”，本卷区分五类状态：

| 状态 | 典型内容 | 主要所有者 |
|---|---|---|
| 应用状态 $A$ | 消息、模板、检索结果、工具描述、截断策略 | 应用运行时 |
| 模型状态 $M$ | token IDs、位置、隐藏状态、KV cache、下一步原始 logits | 推理引擎 |
| 解码状态 $D$ | 已选 token、计数器、约束自动机、随机数游标、停止匹配器 | 解码器 |
| 传输状态 $S$ | 未发送字节、事件序号、背压、连接与取消状态 | 流式服务 |
| 外部状态 $W$ | 文件、数据库、远程 API、设备及其事务状态 | 工具执行器与外部系统 |

一次纯文本回答通常改变 $A,M,D,S$，不直接改变 $W$。一次写工具调用可能改变全部五类状态。这里的划分不是为了建立一般程序逻辑，而是为了回答一个具体问题：某次偏差首先出现在哪个状态中。

## 0.3 阶段与数据产品

一次 decoder-only 文本生成可分成以下阶段：

| 阶段 | 输入 | 主要输出 | 尚未发生的事 |
|---|---|---|---|
| 请求规范化 | 网络字节、API 参数 | 已验证的请求对象 | 尚无模型输入 |
| 上下文渲染 | 消息、模板、工具 schema | 确定的文本或控制符号序列 | 尚无 token |
| tokenization | 渲染结果、tokenizer | `input_ids`、mask、position IDs | 尚无神经网络计算 |
| prefill | 输入张量、模型参数 | 首步 logits、提示 KV cache | 尚未选出输出 token |
| token 选择 | logits、处理器、解码状态 | 一个 token ID | 该 token 尚未必进入缓存或界面 |
| decode | 上一步所选 token、旧缓存 | 新缓存、下一步 logits | 尚未选出再下一个 token |
| 流式返回 | token、反分词与停止状态 | 有序字节事件 | 客户端尚未必绘制 |
| 终止与清理 | 停止原因、资源句柄 | 终态事件、用量、资源释放 | 不保证撤销外部副作用 |

这个表特别区分“选择 token”和“用该 token 计算下一步”。本卷采用如下统一约定。

**约定 0.2（解码步序）**　在选择第 $t$ 个输出 token 之前：

1. KV cache 表示提示 $x_{1:m}$ 与已经处理过的输出 $y_{1:t-1}$；
2. 当前 logits $z_t$ 是以同一序列为条件的下一 token 分数；
3. 解码器从 $z_t$ 选择 $y_t$；
4. 若 $y_t$ 触发终止，可以不再把它送入模型；否则 `decode_one(y_t)` 把它写入缓存并产生 $z_{t+1}$。

因此，prefill 完成时缓存长度是 $m$，不是 $m+1$；选择首 token 后，只有执行下一次模型调用，缓存长度才成为 $m+1$。具体引擎可以在内部流水化这些动作，但逻辑依赖顺序不能颠倒。

## 0.4 一条最小事件时间线

执行记录应使用单调事件序号，而不能只依赖可能漂移的墙上时钟：

```text
e00 request.accepted
e01 request.normalized
e02 context.rendered
e03 input.tokenized
e04 prefill.started
e05 prefill.completed          # cache covers prompt; logits_1 ready
e06 token.selected(step=1)
e07 stream.delta(seq=0)?
e08 decode.completed(step=1)   # cache now includes y_1; logits_2 ready
...
eNN generation.finished(reason=...)
eNN+1 resources.released
```

时间戳可用于计算延迟，但事件序号用于确定因果顺序。异步系统中，`stream.delta` 与下一次设备计算可能重叠；记录不应伪造一个实际上不存在的全局纳秒顺序，只需保存每个组件内的顺序以及跨组件的父子关系。跨服务传播事件父子关系的标准背景见[资料源](SOURCES.md#source-tracing)。

一个最小事件至少应包含：

```text
Event = {
    trace_id,
    event_id,
    parent_event_id?,
    component,
    event_type,
    monotonic_time,
    payload_schema_version,
    payload
}
```

敏感文本和完整张量不必写入长期日志；可以保存版本、形状、长度、摘要与经过授权的抽样字段。记录是否充分，取决于它能否定位第一次差异，而不是取决于日志体积。

## 0.5 三个执行边界

一次输出至少跨越三个边界：

1. **应用边界**：用户界面、消息权限、模板、检索、工具目录和服务配置；
2. **模型边界**：确定的输入张量进入参数化前向计算，得到 logits、噪声预测或向量场；
3. **现实边界**：执行器真正读取、写入、发送或移动外部对象。

模型生成 `{"name":"send_email",...}` 仍位于第二个边界内。解析、授权也尚未发送邮件。只有外部服务接受写入，现实状态才发生变化。第六章会给出这个提交边界的状态机。

## 0.6 四个层次的可复现

“参数都一样”并不足以说明复现到了哪一层。本卷区分：

1. **输入复现**：规范化请求、渲染字节和 token IDs 相同；
2. **模型复现**：各步原始 logits 或更新张量在规定容差内相同；
3. **选择复现**：处理器顺序、随机算法与随机数消费相同，所选 token 或状态更新相同；
4. **系统复现**：排队、流式事件、取消结果和工具返回也相同。

前一层通常是后一层的必要条件，却未必充分。例如，固定 token 与权重后，不同精度和归约顺序仍可能给出略有差异的 logits；固定 logits 后，改变 top-p 与温度的先后顺序仍会改变候选集合；固定模型输出后，外部天气 API 仍可能返回新数据。

**命题 0.3（固定环境下的抽象确定性）**　令 $\omega_{\mathrm{env}}$ 记录调度许可、时钟与 deadline、客户端取消、传输成功/失败等外部事件。若请求规范化、模板、tokenizer、模型参数、全部数值算子、logit 处理器及其顺序、停止规则和选择函数均为确定函数，并固定请求输入与 $\omega_{\mathrm{env}}$，则完整纯文本执行轨迹唯一确定。

**说明。** 这是函数复合与确定状态转移的直接结果，不是对具体 GPU 服务逐位确定性的承诺。随机采样把随机源状态加入输入后，也可以视为确定的伪随机计算。若不固定 $\omega_{\mathrm{env}}$，相同请求仍可因排队、取消或传输事件得到不同系统轨迹；在环境事件首次介入前，模型与选择子轨迹才由其余固定输入决定。

## 0.7 本卷使用的不变量

后续章节会反复检查以下不变量：

1. `input_ids` 必须能够追溯到确定的模板与 tokenizer 工件；
2. 每层 KV cache 的逻辑位置集合相同，key 与 value 长度相等；
3. 在约定 0.2 的步首，缓存序列与当前 logits 的条件前缀相同；
4. 处理器只按声明顺序读取和更新解码状态；
5. 已发送的流式字节是已选 token 反分词结果的一个安全前缀；
6. `finished`、`cancelled` 与 `failed` 是终态，终态之后不能再发送普通增量；
7. 工具候选在通过解析、验证和授权前不能执行；
8. 写操作越过提交边界后，模型请求取消不等价于外部回滚。

这些不变量比“模型正在思考”“系统知道答案”一类描述更可检验。违反其中任何一项，都能对应到具体实现错误或记录缺口。

## 0.8 失败定位规则

当两次最终输出不同，按时间比较最早的数据产品：

```text
request object
-> rendered bytes/control symbols
-> token IDs and positions
-> prefill logits
-> processed candidate distribution
-> selected token
-> cache after decode
-> emitted bytes
-> client display
-> external effect
```

如果 token IDs 已不同，就没有必要先解释隐藏层；如果原始 logits 相同而候选集不同，应检查处理器；如果全部 token 相同而界面文字不同，应检查增量反分词、停止串和传输。这个“第一次分歧”原则贯穿第七章的三条完整轨迹。

## 0.9 阅读路线与学习目标

- [第一章](ch01_text_tokens_context.md)把网络输入规范化为模型实际接收的 token、mask 与位置；
- [第二章](ch02_prefill_forward_pass.md)逐层跟随 prefill，并说明 KV cache 在何处写入；
- [第三章](ch03_logits_and_next_token.md)规定 logit 处理顺序、约束状态与选择算法；
- [第四章](ch04_decode_loop_streaming.md)把单步扩展成含停止、streaming 与取消的状态机；
- [第五章](ch05_iterative_generation.md)用同一执行接口比较自回归、连续扩散、离散扩散与 flow；
- [第六章](ch06_tools_runtime_boundary.md)给出工具 schema、权限、幂等与提交边界；
- [第七章](ch07_end_to_end_traces.md)用三条含具体张量形状和事件的轨迹检验全卷。

读完本卷，读者应能仅凭一份充分的执行记录回答：实际输入是什么、当前状态是什么、下一步由谁计算、哪些规则改变了候选、何时已经终止、外部副作用是否发生，以及两次运行第一次在哪里分叉。这就是本卷的学习目标。
