# 第四章 解码循环、Streaming 与终止状态机

上一章定义了如何从一列 logits 选出一个 token。本章把这个局部转移嵌入完整请求：模型状态、解码状态、增量反分词、停止串缓冲、调度与取消必须一起前进，并在唯一终态收束。

## 4.1 请求生命周期

一个文本生成请求至少经历以下状态：

```text
CREATED -> QUEUED -> PREFILLING -> DECODING -> DRAINING -> FINISHED
               |          |            |           |
               +----------+------------+-----------+-> FAILED
                          cancel request
                                |
                                v
                       CANCEL_REQUESTED -> CANCELLED
```

这只是允许转移图，不表示取消可以抢占任意设备指令。状态含义如下：

| 状态 | 已成立的事实 | 允许的下一动作 |
|---|---|---|
| `CREATED` | 请求对象已创建 | 验证、拒绝、入队 |
| `QUEUED` | 尚未占用模型执行槽 | 调度 prefill、取消、超时 |
| `PREFILLING` | 正计算提示 cache | 完成、失败、登记取消 |
| `DECODING` | 至少有下一步 logits 或在计算下一步 | 选择、decode、登记取消 |
| `DRAINING` | 不再选择 token，仍有安全字节/终态事件待发送 | 发送完成、连接失败 |
| `CANCEL_REQUESTED` | 取消已被接受但资源可能仍在执行 | 到安全点清理 |
| `FINISHED` | 正常终止原因已固定 | 只允许读取结果 |
| `CANCELLED` | 取消终态已固定 | 只允许读取部分结果/元数据 |
| `FAILED` | 错误终态已固定 | 只允许读取错误/部分结果 |

`FINISHED`、`CANCELLED`、`FAILED` 互斥且不可逆。进入任一终态后，不得再发送普通文本增量。

## 4.2 步首状态与 off-by-one 约定

设提示为 $x_{1:m}$，已经选出的输出为 $y_{1:t-1}$。第 $t$ 次选择之前定义：

```text
StepState_t = {
    cache,                  # covers x[1:m] + y[1:t-1]
    raw_logits_t,           # next-token logits for the same prefix
    decoder_state_t,
    detokenizer_state_t,
    stop_buffer_t,
    stream_sequence_number,
    lifecycle = DECODING
}
```

条件分布写为

$$
p_\theta(y_t\mid x_{1:m},y_{1:t-1}).
$$

选择 $y_t$ 后有两条路径：

1. 若 EOS、停止串、长度、取消或错误使请求终止，不需要再计算 $z_{t+1}$；
2. 若继续，`decode_one(cache, y_t, position=m+t-1)` 把 $y_t$ 的 K/V 追加到所有层，并返回 $z_{t+1}$。

因此“已选 token 数”可以比“cache 中输出 token 数”多 1，但只存在于选择后、下一次 decode 前的短暂状态。事件记录应区分 `token.selected(t)` 与 `decode.completed(t)`。

## 4.3 完整的单路径循环

下面给出参考伪代码。它省略设备并行，却不省略影响返回字节的顺序：

```text
function generate(request):
    transition(CREATED, QUEUED)
    normalized = normalize_request(request)
    inputs = assemble_input(normalized)

    check_cancel_or_deadline()
    transition(QUEUED, PREFILLING)
    logits, cache = prefill(inputs)
    decoder = init_decoder_state(normalized.generation_config)
    text = init_incremental_detokenizer()
    stopper = init_stop_matcher(normalized.stop_sequences)
    stream = init_ordered_stream()
    transition(PREFILLING, DECODING)

    while true:
        if cancellation_requested():
            transition(DECODING, CANCEL_REQUESTED)
            break
        if deadline_exceeded():
            finish_reason = "deadline_exceeded"
            break

        token, decision_record = choose(logits, decoder)
        record token.selected(token, decision_record)
        decoder.consume(token)

        if token_is_eos(token):
            finish_reason = "eos"
            break

        new_bytes = text.push(token)
        safe_bytes, stop_hit = stopper.push(new_bytes)
        stream.emit_delta_if_nonempty(safe_bytes)

        if stop_hit:
            finish_reason = "stop_sequence"
            break
        if decoder.generated_count >= max_output_tokens:
            finish_reason = "max_output_tokens"
            break
        if cache.logical_length + 1 >= model_context_limit:
            finish_reason = "context_limit"
            break
        if grammar_requires_termination(decoder.grammar_state):
            finish_reason = "grammar_complete"
            break

        logits, cache = decode_one(
            token,
            cache,
            position=cache.next_logical_position
        )
        record decode.completed(cache.logical_length)

    if lifecycle == CANCEL_REQUESTED:
        wait_until_model_safe_point()
        discard_non_safe_stop_buffer()
        release_model_resources()
        transition_and_emit_terminal(
            CANCEL_REQUESTED, CANCELLED, reason="cancelled"
        )
        return partial_result

    transition(DECODING, DRAINING)
    if finish_reason == "stop_sequence":
        text.discard_pending_after_stop()
        stopper.discard_from_first_match()
    else:
        final_decoded_bytes = text.finish()
        safe_bytes, final_stop_hit = stopper.push(final_decoded_bytes)
        stream.emit_delta_if_nonempty(safe_bytes)
        if final_stop_hit:
            finish_reason = "stop_sequence"
            stopper.discard_from_first_match()
        else:
            stream.emit_delta_if_nonempty(stopper.flush_safe_suffix())
    release_model_resources()
    transition_and_emit_terminal(
        DRAINING, FINISHED, reason=finish_reason
    )
    return final_result
```

stop matcher 的第一次命中是吸收事件：命中位置及其后字节永不进入返回流，匹配器也不再接受可见输出。若停止串已命中，反分词器中尚未形成字符的尾部属于停止点之后的数据，应直接丢弃；此时调用 `text.finish()` 既无必要，也可能把被丢弃的不完整 UTF-8 尾部误报为请求失败。其他正常终止原因必须调用 `text.finish()`，排出合法尾部，并在残留字节不能构成合法文本时产生 decoding error；该异常进入 `FAILED`。finish 产生的字节仍经过 stop matcher，因而不能绕过停止串规则。`transition_and_emit_terminal` 在同一受控临界区固定终态并排入唯一终态事件，避免取消与正常完成各写一次结果。具体协议可以在终态后继续释放不影响结果的后台统计资源，但对客户端可见的 `completed` 必须在所有普通 delta 之后。异常路径还要保证 cache 只释放一次。

## 4.4 单 token decode 的模型转移

在第 $t$ 个 token 被选出且生成继续时，模型入口只有一个新位置：

```text
new_input_id : int[B_active, 1]
position     : int[B_active, 1]
old_cache    : per-sequence lengths m_b+t_b-1
```

这里只把本轮仍在生成的请求组成 active batch。第 $\ell$ 层为每个样本 $b$ 计算新位置的 query、key、value；query 读取该样本本层旧 K/V 与当前新 K/V，当前 key/value 写到逻辑位置 $m_b+t_b-1$。不同样本可有不同 cache 长度，分页表或 sequence ID 必须保持归属。所有层完成后，新隐藏状态投影为下一步 logits。

若层数为 $L$、KV 头数为 $H_{kv}$、头维度为 $d_h$，每选出并处理一个普通 token，逻辑 cache 增加

$$
2LH_{kv}d_h
$$

个标量。总缓存量随已处理上下文长度线性增长。物理分页、连续批处理和设备并行属于卷一的服务实现；本章只要求这些优化保持逻辑 token 顺序和 cache 不变量。系统背景见[资料源](SOURCES.md#source-serving)。

## 4.5 增量反分词器

反分词器维护自己的字节状态：

```text
DetokenizerState = {
    token_history_suffix,
    undecoded_bytes,
    normalization_or_cleanup_state
}
```

选择一个 token 不保证立即产生显示字符。字节级 token 可能只给出多字节 UTF-8 的前半部分；某些 tokenizer 的清理规则还依赖相邻 token。增量实现必须与批量 `decode(all_selected_tokens)` 在声明的接口下相容：

$$
\operatorname{concat}(\text{all emitted bytes},\text{final buffer})
=\operatorname{Decode}(y_{1:T})
$$

或明确记录为了停止串、控制 token 与清理规则而删除的区段。不能对每个 ID 独立调用无状态 `decode` 后简单拼接，除非 tokenizer 保证这种同态性质。

## 4.6 停止串必须跨 token 匹配

停止条件分为 token 级与字节/文本级：

- EOS、特殊结束 token：选择后立即终止，通常不显示；
- stop token sequence：在 token ID 流上匹配；
- stop byte/string sequence：在增量反分词后的字节流上匹配；
- grammar acceptance：由约束自动机决定；
- 长度、deadline、取消与错误：由运行时决定。

对一组字节停止串，streaming 层不能立刻发送每个新字节。它必须保留当前输出中仍可能成为某个停止串前缀的**最长后缀**。只有更早的字节是安全的。

可手算夹具如下：

```text
token 101 -> bytes "AB"
token 102 -> bytes "CD"
stop sequence = "BCD"
```

选择 token 101 后，`A` 可以发送，`B` 必须留在 stop buffer；选择 token 102 后，缓冲成为 `BCD` 并完整命中，三个停止字节均不发送。最终返回 `A`。若服务在第一次选择后发送 `AB`，它就无法在不要求客户端回滚的情况下隐藏停止串。

多个停止串同时匹配时需要确定规则，例如“选择最早结束；同一结束位置选择最长；再按配置顺序打破并列”。规则影响返回文本与 finish metadata，必须版本化。

## 4.7 有序流式协议

模型选择、服务发送和客户端绘制是三条不同时间线。一个最小协议可以发送：

```text
response.start(request_id, model_snapshot)
response.delta(sequence_no=0, bytes=...)
response.delta(sequence_no=1, bytes=...)
...
response.completed(sequence_no=N, finish_reason, usage)
```

不变量是：

1. `sequence_no` 严格递增且在请求内唯一；
2. delta 的拼接顺序由序号决定，不由网络到达时间猜测；
3. 终态事件最多一个，并位于所有普通 delta 之后；
4. 重连若允许重放，客户端按 `(request_id, sequence_no)` 去重；
5. 部分连接断开不应被记录成模型 EOS。

客户端绘制还可能按帧合并多个 delta，所以肉眼看到的停顿不能推出 token 选择的时间间隔。性能字段应分别记录 queue、prefill、decode、stop-buffer 与网络发送时间。

## 4.8 背压与缓冲上限

当客户端读取速度低于生成速度，服务有三种基本选择：

1. 暂停调度该请求的后续 decode；
2. 继续生成并在服务器缓冲；
3. 超过缓冲上限后取消或失败。

第一种降低资源利用率但限制内存；第二种维持模型吞吐，却需要有界缓冲；第三种必须给出独立于模型停止的 finish/error reason。实现不能无限增长网络缓冲，也不能在丢弃 delta 后仍声称返回了完整文本。

若请求被调度器暂挂或 KV 被换出，逻辑 `StepState` 不变。恢复后必须在同一 token、位置、解码状态和 RNG 游标继续。物理 preemption 不是一次新的生成。

## 4.9 取消的精确语义

取消至少有三个时刻：

1. **已请求**：客户端或系统控制器发出取消；
2. **已观察**：网关/调度器把请求标记为 `CANCEL_REQUESTED`；
3. **已生效**：不再安排模型步骤，资源到安全点后释放，终态成为 `CANCELLED`。

GPU kernel 通常不能在任意指令处由业务请求单独回滚。取消发生在 prefill 或 decode 执行中时，本次 kernel 可能完成；其结果应被丢弃，不能在取消终态后继续发送。客户端断开连接也不必自动等价于取消：服务可能为了缓存、计费或后台任务继续执行，协议必须明确。

deadline 是时间策略触发的独立终止原因，不是用户取消。实现内部可以复用同一安全点与资源清理机制，但外部终态必须保留 `deadline_exceeded`，不能改写成 `cancelled`。纯文本取消只影响计算与返回字节。若此前已经触发外部工具，取消不能撤销工具副作用；这属于[第六章的提交边界](ch06_tools_runtime_boundary.md#67-只读写入与结果未知)。

## 4.10 终止原因与资源清理

建议把终止原因作为枚举而非自由文本：

```text
eos
stop_sequence
grammar_complete
max_output_tokens
context_limit
cancelled
deadline_exceeded
content_interrupted
model_error
transport_error
```

`transport_error` 表示返回通道失败，不推出模型计算失败；`content_interrupted` 表示策略层中断，也不应伪装成 EOS。用量字段还要说明计数的是已选 token、已处理 token 还是已返回 token；停止串和 EOS 会使三者不同。

资源清理必须满足：cache、批处理 slot、RNG state、stream buffer 和追踪句柄各释放一次；清理失败可以记录后台错误，但不能再次打开已终止的输出流。

## 4.11 两个最小复现实验

**实验 A：跨 token 停止串。** 使用第 4.6 节的 token 101/102 和停止串 `BCD`。断言事件顺序为：

```text
selected(101) -> delta("A") -> decode(101)
-> selected(102) -> stop_hit("BCD") -> completed(stop_sequence)
```

断言返回字节恰为 `A`，cache 是否写入 token 102 由终止优化决定，但不能再产生下一步 token。

**实验 B：decode 中取消。** 使用一个在 `decode.started(step=3)` 后、`decode.completed(step=3)` 前注入的取消信号。允许设备完成本步，但断言：

1. 不发生 `token.selected(step=4)`；
2. 最终状态是 `CANCELLED`；
3. 终态后没有 delta；
4. cache 与批处理 slot 最终释放一次；
5. 已发送前缀保持不变，不发送伪造的回滚事件。

## 4.12 失败条件与可观测字段

| 失败 | 违反的不变量 | 可观察结果 |
|---|---|---|
| EOS 被送给普通 detokenizer 并显示 | 控制 token 终止顺序 | 界面出现控制文本 |
| stop matcher 只看单 token | 跨 token 匹配 | 停止串泄漏或漏停 |
| token 已选但 cache 被标成已写 | 条件前缀同步 | 下一步位置 off-by-one |
| 两个终态竞争写入 | 终态互斥 | 同时出现 completed 与 cancelled |
| 背压缓冲无上限 | 有界资源 | 慢客户端耗尽内存 |
| 断连被记为 EOS | 原因语义 | 把传输失败误作模型决定 |
| 重连不按序号去重 | 流有序性 | 客户端显示重复片段 |

一次完整循环的记录至少包括：生命周期转移、每步 token 选择、cache 逻辑长度、停止匹配器状态、已发送字节序号、取消观察时刻、终止原因和资源释放结果。至此，自回归文本的执行语义已经完整描述；下一章用同样的“状态、更新、停止”框架比较其他生成方式。
