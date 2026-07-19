# 第六章 工具调用：Schema、权限与提交边界

模型生成一段结构化调用，不等于工具已经执行。一次工具交互至少包含候选生成、解析、schema 验证、语义规范化、授权、执行、结果编码与后续模型调用。本章只讨论这条执行链，不把它扩张成一般 Agent 方法论或组织治理讨论。

## 6.1 四个不同对象

工具系统首先要区分：

```text
ToolSpec    # 当前运行时允许模型看到的版本化接口
Proposal    # 模型生成的名称与参数候选
Invocation  # 运行时规范化、授权后真正准备执行的调用
Result      # 执行器返回的结构化结果或错误
```

参考数据结构如下：

```text
ToolSpec = {
    name,
    version,
    input_schema,
    schema_dialect,
    side_effect_class,
    required_capabilities,
    timeout_policy,
    retry_policy
}

Proposal = {
    model_call_id,
    proposal_id,
    name,
    arguments_bytes
}

Invocation = {
    invocation_id,
    tool_name,
    tool_version,
    normalized_arguments,
    subject,
    granted_capabilities,
    idempotency_key?,
    deadline
}

Result = {
    invocation_id,
    status,
    value_or_error,
    commit_status,
    external_resource_id?,
    observed_at
}
```

模型只直接产生 Proposal。Invocation 中的主体、权限、deadline 和幂等键来自运行时，不能由模型通过在参数里自我声明而获得。

## 6.2 工具调用状态机

一次调用的允许状态可写成：

```text
PROPOSED
  -> PARSED
  -> VALIDATED
  -> AUTHORIZED
  -> PREPARED
  -> EXECUTING
  -> COMMITTED
  -> OBSERVED
```

旁路终态包括：

```text
PARSE_REJECTED
VALIDATION_REJECTED
AUTHORIZATION_REJECTED
CANCELLED_BEFORE_COMMIT
FAILED_BEFORE_COMMIT
OUTCOME_UNKNOWN
FAILED_AFTER_COMMIT
```

不是每个只读工具都需要显式 PREPARED/COMMITTED；这两个状态主要用来说明写操作。关键是运行时不能从 PROPOSED 直接跳到 EXECUTING。

**不变量 6.1**

1. 状态只能沿允许边转移，拒绝终态不能重新执行；
2. 实际工具名和版本必须存在于本请求的 ToolSpec 集合；
3. Invocation 参数必须是 Proposal 经确定性解析、规范化与验证的结果；
4. 执行使用的权限不能超过当前主体被授予的 capability；
5. COMMITTED 之后，取消当前模型请求不能把调用改写为“未发生”。

## 6.3 从 token 到 Proposal

工具候选仍由普通 token 生成。运行时可以在生成结束后解析 JSON，也可以在[第三章的 grammar 处理器](ch03_logits_and_next_token.md#34-硬约束与语法状态)中限制可选 token。两条路径的差别是：

- **事后解析**：模型可以先生成任意字节；解析失败后拒绝或请求修复；
- **约束解码**：每一步只允许仍可完成合法结构的 token，降低语法失败；
- **两者并用**：约束保证形式前缀，完成后仍由独立 parser 和 validator 检查。

即使 grammar 保证输出是 JSON，Proposal 仍应保存原始参数字节与解析后的值。否则无法判断问题来自模型字节、JSON parser，还是后续规范化。

## 6.4 Schema 验证的精确边界

考虑一个只读天气工具：

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {
    "city": {"type": "string", "minLength": 1},
    "date": {"type": "string", "format": "date"}
  },
  "required": ["city", "date"],
  "additionalProperties": false
}
```

验证必须绑定 schema dialect。JSON Schema 2020-12 中，一些 vocabulary 与 `format` 的行为取决于实现声明和配置；不能只写“使用 JSON Schema”而不记录 dialect、validator 版本与是否启用 format assertion。规范来源见[资料源](SOURCES.md#source-schema)。

验证分为三层：

1. **语法解析**：字节是否构成一个 JSON 值，是否拒绝重复键、非有限数字等实现扩展；
2. **结构验证**：字段、类型、枚举、范围、额外属性是否满足 schema；
3. **语义规范化与前置条件**：城市别名、时区、相对日期、资源是否存在、版本号是否仍有效。

Schema 不能证明参数符合用户真实意图。例如合法日期可能是错误年份；合法路径也可能指向越权文件。语义检查与授权不可由结构验证替代。

## 6.5 规范化必须先于幂等比较

相对值应在运行时转成明确值。例如请求中的“明天”不能直接交给远程 API：

```text
normalize_date(
    expression = "tomorrow",
    reference_clock = "2026-07-19T14:00:00+08:00",
    user_timezone = "Asia/Shanghai"
) -> "2026-07-20"
```

规范化接口必须固定 locale、时区、参考时钟、舍入和默认字段。对写操作，幂等请求摘要应基于**规范化后的语义参数**，否则同一意图可能因字段顺序或等价日期写法绕过去重。

一个参考规范请求摘要是

$$
d=H(
\text{tenant}\parallel
\text{tool name/version}\parallel
\text{canonical normalized arguments}
).
$$

哈希只压缩比较对象；语义由前面的规范化规则定义。

## 6.6 权限是运行时能力

自然语言工具说明可以帮助模型选择工具，却不是权限边界。授权函数读取：

$$
\operatorname{authorize}
(\text{subject},\text{capability},\text{resource},\text{context})
\longrightarrow
\{\text{allow},\text{deny},\text{confirm}\}.
$$

运行时至少应实施：

- 工具与版本 allowlist；
- 主体、租户和会话绑定；
- 资源范围，如路径前缀、数据库行、域名或日历 ID；
- 只读、创建、修改、删除能力分离；
- 金额、速率、数量和时间范围约束；
- 高影响调用的用户确认；
- 短期凭证与最小权限，凭证不进入模型上下文。

用户确认也必须绑定规范化 Invocation。若确认界面显示“创建一个 9:00 提醒”，确认后却允许模型把参数换成另一时间，确认就没有约束意义。合理做法是让确认记录包含 invocation digest；参数变化后重新验证与确认。

## 6.7 只读、写入与结果未知

只读调用通常不改变业务状态，但仍可能计费、写访问日志或消费配额。写调用具有更严格的失败语义。一次网络超时可能对应：

1. 请求尚未到达外部服务；
2. 到达但在提交前失败；
3. 已提交，响应在返回途中丢失；
4. 只完成了非原子工作流的一部分；
5. 已提交，但后续读取受一致性延迟影响。

因此 timeout 不是“未执行”的证据。对无法确定提交结果的写调用，状态应是 `OUTCOME_UNKNOWN`，而不是自动回到 PREPARED。

**定义 6.2（提交边界）**　提交边界是外部系统首次保证目标副作用已经成为其状态的一部分、或已不可由本请求单方面阻止的事件。边界的具体位置由工具协议定义，不由语言模型叙述决定。

边界前取消可阻止执行；边界后取消只能停止后续步骤。撤销需要外部系统支持的逆操作、事务回滚或补偿工作流。

## 6.8 幂等键的正确契约

幂等键不是随便附加的 UUID。执行器应以至少三元组

$$
(\text{scope},\text{tool version},\text{idempotency key})
$$

查找记录，并保存规范请求摘要 $d$。收到重复键时：

1. 若摘要相同且已有成功结果，返回原结果，不重复副作用；
2. 若摘要相同且仍在执行，返回 in-progress 或等待同一执行；
3. 若摘要不同，拒绝 key reuse；
4. 若旧记录已过保留窗口，按照明确策略拒绝或视为新调用，而不能悄悄猜测。

若业务写入和幂等记录位于同一数据库，可以在一个事务中原子提交。若副作用发生在远程系统，本地先写“准备执行”仍不能解决“远程已提交但本地未记录”的窗口；最好由远程 API 原生接受同一幂等键，或使用可查询的操作 ID、outbox/工作流与补偿协议。

HTTP 对方法幂等性的标准定义只说明重复相同请求的预期效果；具体 POST 工具是否支持应用级幂等键，必须由该工具协议声明。标准来源见[资料源](SOURCES.md#source-transactions)。

## 6.9 重试矩阵

重试决定应按失败类别和副作用语义制定：

| 结果 | 只读调用 | 带可靠幂等键的写调用 | 无幂等保证的写调用 |
|---|---|---|---|
| parse/schema 失败 | 不重试同一候选 | 不重试 | 不重试 |
| authorization deny | 不重试 | 不重试 | 不重试 |
| 明确未到执行器 | 可按预算重试 | 可按预算重试 | 可谨慎重试 |
| 429/临时 5xx | 退避并受 deadline 限制 | 同键退避重试 | 先判断是否可能提交 |
| 响应超时、提交未知 | 可重试或重新读取 | 同键查询/重试 | 标为 UNKNOWN，先核对状态 |
| 明确业务拒绝 | 只有参数改变后新调用 | 参数改变后新键 | 参数改变后新调用 |

重试计数、退避、jitter、总 deadline 和每次 invocation ID 都应进入事件轨迹。模型可以根据结构化错误提出新参数，但这是一项新的 Proposal，不是运行时对旧调用的透明重试。

## 6.10 工具结果再次成为输入

执行结果通常被编码为 tool message，再触发一次模型调用：

```json
{
  "role": "tool",
  "tool_call_id": "call_17",
  "content": {
    "status": "ok",
    "city": "上海",
    "date": "2026-07-20",
    "rain_probability": 0.72
  }
}
```

模型看到的是这个序列化表示，不是外部数据库本身。运行时必须明确：

- 大结果的截断、分页或摘要规则；
- 二进制结果如何引用，引用何时过期；
- 错误与成功是否使用可区分 schema；
- 外部文本被标记为不可信数据，而不是提升为 system 指令；
- tool call ID 是否与 Proposal/Invocation 一一对应；
- 结果中的秘密、个人数据和凭证在回填前如何过滤。

结果被截断时应有 `truncated=true` 和剩余数据的获取方式，不能把截断后的 JSON 伪装成完整结果。

## 6.11 多轮工具循环

一个有界运行时循环可以写成：

```text
while budget.model_calls_remaining > 0:
    response = call_model(context, visible_tool_specs)

    if response.kind == FINAL_TEXT:
        return response

    proposal = parse_tool_proposal(response)
    invocation = validate_normalize_authorize(proposal)

    if invocation.requires_confirmation:
        confirmation = request_user_confirmation(invocation.digest)
        if not confirmation.matches(invocation.digest):
            return cancelled_before_commit

    result = execute_with_retry_policy(invocation)
    context.append(encode_tool_result(result))
    budget.consume(model_call=1, tool_call=1, time=result.elapsed)

return budget_exhausted
```

预算至少限制模型调用数、工具调用数、总墙钟时间、累计结果字节与外部成本。循环终止是运行时决定，不应仅依赖模型愿意输出“完成”。

## 6.12 写调用的具体状态轨迹

考虑 `calendar.create_event@v1`，规范化参数是：

```json
{
  "calendar_id": "primary",
  "title": "带护照",
  "start": "2026-07-20T09:00:00+08:00",
  "duration_minutes": 10
}
```

运行时状态如下：

| 事件 | 状态 | 关键记录 |
|---|---|---|
| 模型生成参数 | `PROPOSED` | 原始 arguments bytes |
| JSON 与 schema 通过 | `VALIDATED` | validator/dialect |
| 日期与日历规范化 | `VALIDATED` | reference clock、normalized args |
| 检查 `calendar.write` | `AUTHORIZED` | subject、calendar scope |
| 用户确认摘要 | `PREPARED` | confirmation + invocation digest |
| 以 key `req-42:create-event:1` 请求外部 API | `EXECUTING` | attempt 1 |
| 外部 API 返回 event ID `evt_6K2` | `COMMITTED` | commit time、resource ID |
| 结果写回模型上下文 | `OBSERVED` | tool result bytes |

若 `COMMITTED` 后客户端取消，只能阻止后续自然语言总结；事件 `evt_6K2` 仍存在。若响应在提交后丢失，应以相同幂等键查询或重试，不能创建第二个提醒。

## 6.13 事件记录与跨服务关联

工具调用跨越模型服务、运行时与外部 API。每层应传播同一个 trace ID，并为 Proposal、Invocation 和外部 attempt 分配不同 span/event ID。W3C Trace Context 给出了跨 HTTP 服务传播 trace context 的标准接口，见[资料源](SOURCES.md#source-tracing)。

最小记录要能回答：

1. 模型看到了哪个 ToolSpec；
2. 原始 Proposal 字节是什么；
3. parser、schema validator 与规范化器分别产出什么；
4. 谁以何种 capability 授权，是否经过确认；
5. 实际执行的 Invocation 与每次重试是什么；
6. 提交状态是未提交、已提交还是未知；
7. 返回模型的 tool message 到底有哪些字节；
8. 下一次模型调用使用了哪个上下文。

这是一份执行事实记录，不是对输出真实性或制度责任的替代讨论。

## 6.14 失败条件

| 失败 | 最早边界 | 正确响应 |
|---|---|---|
| 模型调用不存在的工具版本 | Proposal/lookup | 拒绝，不做近似名称匹配 |
| JSON 可解析但多出字段 | schema | 按 `additionalProperties` 规则拒绝 |
| `format: date` 未启用 assertion | validator 配置 | 另做明确语义验证并记录 |
| 模型在 args 中自称管理员 | authorization | 忽略该声明，读取运行时主体 |
| 用户确认后参数被替换 | confirmation binding | digest 不同，重新确认 |
| 写超时被当作未执行 | execution | 标记 outcome unknown 并查询 |
| 同幂等键配不同参数 | idempotency store | 拒绝 key reuse |
| 工具网页内容被提升为指令 | result encoding | 保持不可信 tool-data 边界 |
| 取消后再次发送普通 delta | lifecycle | 终态后禁止输出 |

工具执行在 `OBSERVED` 状态结束：外部结果已经以确定格式进入新上下文，或者失败/未知状态已经被明确记录。模型随后怎样措辞是一次新的生成；外部世界是否已经改变，则由提交状态而不是那段措辞决定。
