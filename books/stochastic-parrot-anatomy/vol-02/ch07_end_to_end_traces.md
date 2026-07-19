# 第七章 三条完整执行轨迹

本章用三个固定夹具检验前六章的接口：纯文本自回归、文本到图像的潜空间扩散，以及带确认与写副作用的工具调用。夹具中的小模型、tokenizer 和数值是教学用合成对象，不冒充任何线上产品；它们的价值在于每一步都有确定输入、形状、状态与预期事件。

## 7.1 轨迹记录格式

三条轨迹使用同一事件外壳：

~~~json
{
  "trace_id": "trace-text-001",
  "event_id": "e05",
  "parent_event_id": "e04",
  "component": "decoder",
  "event_type": "token.selected",
  "logical_step": 1,
  "payload": {}
}
~~~

事件序号只在一条 trace 内排序；墙钟时间另行记录。张量事件至少保存 shape、dtype、逻辑位置和可选摘要。工具事件还保存 commit status。对敏感生产数据可以只留摘要，但本章夹具直接列出小规模 token IDs。

## 7.2 轨迹一的固定工件

用户请求：

> 天空为什么是蓝色？

使用[第一章的合成模板与 tokenizer](ch01_text_tokens_context.md#16-一个可手算的输入夹具)。词表在原 14 项上增加：

| ID | token |
|---:|---|
| 14 | 短波蓝光 |
| 15 | 的散射 |
| 16 | 通常更强 |
| 17 | 。 |
| 18 | EOS |
| 19 | reserved |

模型与解码配置固定为：

~~~text
model_id = toy-decoder-v1
B = 1, L = 2, d = 8
H_q = 2, H_kv = 1, d_h = 4
vocabulary_size = 20
dtype = float32
strategy = greedy
processor_order = [
    control_token_mask,
    minimum_length_eos_mask,
    grammar_mask,
    repetition_transforms,
    temperature_disabled,
    truncation_disabled
]
minimum_output_tokens = 1
max_output_tokens = 8
stop_sequences = []
~~~

所有权重是 fixture 的一部分，推理模式无 dropout。下文只列每步最高的几个 logits；夹具还规定未列项均低于表中第三名，因此足以唯一确定 greedy 结果。

## 7.3 轨迹一：请求到首 Token

### 阶段 A：确定输入

模板渲染为：

~~~text
<bos><system>简洁回答</system><user>天空为什么是蓝色？</user><assistant>
~~~

tokenizer 输出：

~~~text
input_ids      = [[1, 2, 7, 3, 4, 8, 9, 10, 11, 12, 5, 6]]
attention_mask = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
position_ids   = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
~~~

此时有效长度 $m=12$。输入事件是：

| 事件 | 组件 | 关键 payload |
|---|---|---|
| e00 request.accepted | gateway | request ID、模型不可变 ID |
| e01 context.rendered | application | template v1、rendered byte length |
| e02 input.tokenized | tokenizer | 上述三张量、tokenizer v1 |
| e03 prefill.started | scheduler | batch slot、logical length 12 |

### 阶段 B：Prefill 张量

embedding 输出形状是 [1,12,8]。每一层的形状账本为：

| 层内对象 | 第 0 层 | 第 1 层 |
|---|---|---|
| residual in | [1,12,8] | [1,12,8] |
| Q | [1,2,12,4] | [1,2,12,4] |
| K/V | 各 [1,1,12,4] | 各 [1,1,12,4] |
| score（逻辑形状） | [1,2,12,12] | [1,2,12,12] |
| residual out | [1,12,8] | [1,12,8] |

prefill 完成后的 cache 是：

~~~text
layer 0: K[1,1,12,4], V[1,1,12,4], positions 0..11
layer 1: K[1,1,12,4], V[1,1,12,4], positions 0..11
next_logits: [1,20]
~~~

首步原始 logits 的前三名为：

| token ID | token | raw logit |
|---:|---|---:|
| 14 | 短波蓝光 | 4.20 |
| 15 | 的散射 | 3.80 |
| 17 | 。 | 1.10 |

控制 mask 禁止 0、1、2、3、4、5、6、19；minimum-length mask 暂时禁止 EOS 18。其余处理器不改变前三名，greedy 选择 14。

| 事件 | 组件 | 关键 payload |
|---|---|---|
| e04 prefill.completed | model | cache length 12、raw logits [1,20] |
| e05 token.selected | decoder | step 1、ID 14、greedy argmax |
| e06 stream.delta | transport | seq 0、UTF-8 bytes for 短波蓝光 |

注意 e05 后、第一次 decode 前，已选输出数是 1，cache 仍长 12。这正是[第四章的步首约定](ch04_decode_loop_streaming.md#42-步首状态与-off-by-one-约定)。

## 7.4 轨迹一：逐 Token Decode

token 14 作为位置 12 的新输入进入两层模型。每层新 Q 为 [1,2,1,4]，新 K/V 各为 [1,1,1,4]；attention 读取旧 12 项与当前项，逻辑 score 为 [1,2,1,13]。完成后 cache 长 13，得到第二步 logits。

完整选择账本如下：

| 选择步 $t$ | 步首 cache 长度 | raw top-1 | logit | 选择 | 是否 decode | decode 后 cache |
|---:|---:|---:|---:|---|---|---:|
| 1 | 12 | 14 | 4.20 | 14 短波蓝光 | 是 | 13 |
| 2 | 13 | 15 | 5.00 | 15 的散射 | 是 | 14 |
| 3 | 14 | 16 | 4.70 | 16 通常更强 | 是 | 15 |
| 4 | 15 | 17 | 5.30 | 17 。 | 是 | 16 |
| 5 | 16 | 18 | 6.00 | 18 EOS | 否 | 16 |

每次普通 token 都按以下事件对出现：

~~~text
token.selected(step=t)
stream.delta(sequence_no=t-1)
decode.started(consuming=y_t, position=cache_length)
decode.completed(new_cache_length=cache_length+1)
~~~

EOS 是例外：它被选择后直接终止，不显示，也不为再下一步建立 cache。最终返回：

> 短波蓝光的散射通常更强。

终态事件为：

~~~text
e21 token.selected(step=5, id=18)
e22 generation.draining(finish_reason=eos)
e23 response.completed(sequence_no=4, output_tokens_selected=5,
                       output_tokens_returned=4)
e24 resources.released(cache_final_length=16)
~~~

该句省略了完整瑞利散射解释，是合成夹具的输出，不用于评估科学质量。轨迹只验证 token 与状态怎样前进。

## 7.5 轨迹一的复现断言

1. 输入 IDs 恰为 12 项固定序列；
2. 两层 prefill cache 逻辑长度均为 12；
3. 首步原始 top-1 为 ID 14；
4. 选择序列恰为 [14,15,16,17,18]；
5. 每个非 EOS token 使所有层 cache 增长 1；
6. EOS 不进入返回文本；
7. delta 序号为 0、1、2、3，终态序号为 4；
8. 最终状态只出现一次 FINISHED，资源释放一次。

若输入 IDs 相同但 e04 logits 已不同，检查模型/数值路径；若 raw logits 相同而选择不同，检查处理器与 tie-break；若 token 相同而字节不同，检查 tokenizer decoder 与 transport。

## 7.6 轨迹二的固定工件

文本到图像请求：

> 雨夜车站，一把留在长椅上的红伞，纪实摄影。

固定配置：

~~~text
text_encoder_id = toy-text-encoder-v2
denoiser_id = toy-latent-denoiser-v3
vae_id = toy-vae-v1
prediction_type = epsilon
latent_shape = [1,4,64,64]
latent_dtype = float32
image_shape = [1,3,512,512]
timesteps = [999,749,499,249]
sampler = ddim-v1
eta = 0
guidance_scale = 5.0
seed = 314159
prng = Philox4x32-10
normal_transform = box-muller-v1
negative_prompt = ""
~~~

这些标识只把接口固定下来；它们不指向真实发布模型。初始潜变量由 seed、PRNG、counter 0 和 normal transform 唯一生成。

## 7.7 轨迹二：条件、去噪与像素

文本 tokenizer 得到长度 12 的 ID 序列，文本编码器产生：

~~~text
conditional_embedding   c_pos: [1,12,768]
unconditional_embedding c_neg: [1,12,768]
~~~

初始状态为 $z_{999}$，形状 [1,4,64,64]。为一次 CFG 网络评估，运行时构造：

~~~text
latent_batch:    [2,4,64,64]  # same z_t duplicated
condition_batch: [2,12,768]   # c_neg followed by c_pos
timestep_batch:  [2]
epsilon_batch:   [2,4,64,64]
~~~

把输出拆为 $\epsilon_{neg},\epsilon_{pos}$，组合

$$
\widehat\epsilon
=\epsilon_{neg}+5.0(\epsilon_{pos}-\epsilon_{neg}),
$$

再由 DDIM scheduler 更新整个 [1,4,64,64] 潜变量。

| 事件 | 状态输入 | 网络输出 | scheduler 输出 |
|---|---|---|---|
| d00 condition.encoded | text IDs [1,12] | embeddings [2,12,768] | 不适用 |
| d01 latent.initialized | seed/counter | 不适用 | $z_{999}$ |
| d02 denoise.updated | $z_{999},999$ | $\epsilon$ [2,4,64,64] | $z_{749}$ |
| d03 denoise.updated | $z_{749},749$ | 同形 | $z_{499}$ |
| d04 denoise.updated | $z_{499},499$ | 同形 | $z_{249}$ |
| d05 denoise.updated | $z_{249},249$ | 同形 | $z_0$ |
| d06 vae.decoded | $z_0$ [1,4,64,64] | pixels [1,3,512,512] | 不适用 |
| d07 image.encoded | postprocessed pixels | PNG bytes | result object |
| d08 response.completed | image reference | 不适用 | FINISHED |

每个 denoise step 都替换整个潜状态；d03 的预览不是在 d02 结果后追加新区域。若提供预览，流协议应使用 replace-frame 或带 step ID 的对象，而不是文本 delta 的拼接语义。

## 7.8 轨迹二的二维数值内核

完整潜变量太大，不适合手算。另取二维 DDIM 子夹具检查 scheduler：

~~~text
alpha_bar[2] = 0.25
alpha_bar[1] = 0.64
alpha_bar[0] = 1.00
z_2 = (1.0, -1.0)
epsilon_2 = (0.2, -0.1)
epsilon_1 = (0.1, -0.2)
sigma = 0
~~~

第 2 步恢复

$$
\widehat z_0^{(2)}
=\frac{z_2-\sqrt{0.75}\,\epsilon_2}{0.5}
\approx(1.653590,-1.826795).
$$

确定性 DDIM 更新到时刻 1：

$$
z_1
=0.8\widehat z_0^{(2)}+0.6\epsilon_2
\approx(1.442872,-1.521436).
$$

再用 $\epsilon_1$：

$$
\widehat z_0^{(1)}
=\frac{z_1-0.6\epsilon_1}{0.8}
\approx(1.728590,-1.751795).
$$

因为 $\bar\alpha_0=1$，最终 $z_0=\widehat z_0^{(1)}$。实现可在声明容差内比较这三个向量。若结果不同，问题位于 scheduler 或参数化，无需先分析图像语义。

## 7.9 轨迹二的复现断言

1. 文本、negative prompt 与编码器版本产生相同条件张量；
2. seed 的 PRNG、normal transform、counter 和 latent shape 完全相同；
3. CFG batch 顺序固定为 negative 后 positive；
4. timestep 列表严格为 [999,749,499,249]；
5. 每步 model output 与 latent shape 匹配；
6. $\eta=0$ 时 denoise loop 不额外消费高斯噪声；
7. VAE scaling、decoder 与像素后处理固定；
8. 比较层级明确为 latent、pixel 或 PNG bytes。

相同 seed 而 scheduler、尺寸或文本编码不同，不构成同一轨迹。

## 7.10 轨迹三的请求与 ToolSpec

固定运行时：

~~~text
clock = 2026-07-19T14:00:00+08:00
timezone = Asia/Shanghai
subject = user-42
capabilities = [calendar.read, calendar.write:primary]
~~~

第一次和第二次模型调用还固定：

~~~text
model_id = toy-tool-decoder-v1
template_id = toy-tool-template-v1
tokenizer_id = toy-tool-tokenizer-v1
L = 2, d = 16
H_q = 4, H_kv = 2, d_h = 4
vocabulary_size = 256
dtype = float32
strategy = greedy_with_json_grammar
~~~

用户请求：

> 明天上午九点提醒我带护照，提交前让我确认。

运行时只向模型提供 calendar.create_event@v1：

~~~json
{
  "name": "calendar.create_event",
  "version": "v1",
  "side_effect_class": "write",
  "input_schema": {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
      "calendar_id": {"type": "string", "enum": ["primary"]},
      "title": {"type": "string", "minLength": 1, "maxLength": 80},
      "local_date": {"type": "string", "format": "date"},
      "local_time": {
        "type": "string",
        "pattern": "^(?:[01][0-9]|2[0-3]):[0-5][0-9]$"
      },
      "duration_minutes": {"type": "integer", "minimum": 1, "maximum": 1440}
    },
    "required": [
      "calendar_id", "title", "local_date",
      "local_time", "duration_minutes"
    ],
    "additionalProperties": false
  }
}
~~~

## 7.11 轨迹三：第一次模型调用

模板把 system 指令、ToolSpec 与用户消息序列化。该夹具记录：

~~~text
model_call_id = mc_01
input_token_count = 86
input_ids_shape = [1,86]
residual_shape = [1,86,16]
per_layer_kv_shape = K/V [1,2,86,4]
raw_logits_shape = [1,256]
prefill_cache_length = 86
generated_proposal_tokens = 27
finish_reason = tool_proposal_complete
~~~

模型产生的原始 arguments bytes 解析为：

~~~json
{
  "calendar_id": "primary",
  "title": "带护照",
  "local_date": "2026-07-20",
  "local_time": "09:00",
  "duration_minutes": 10
}
~~~

到这里没有日历事件。状态仍是 PROPOSED。

| 事件 | 组件 | 状态/结果 |
|---|---|---|
| t00 model.started | model runtime | model call mc_01 |
| t01 proposal.generated | decoder | proposal p_01、27 tokens |
| t02 proposal.parsed | JSON parser | PARSED |
| t03 schema.validated | validator | Draft 2020-12、VALIDATED |

## 7.12 轨迹三：规范化、授权与确认

规范化器结合固定时区，得到 Invocation：

~~~json
{
  "invocation_id": "inv_01",
  "tool": "calendar.create_event@v1",
  "normalized_arguments": {
    "calendar_id": "primary",
    "title": "带护照",
    "start": "2026-07-20T09:00:00+08:00",
    "duration_minutes": 10
  },
  "subject": "user-42",
  "required_capability": "calendar.write:primary"
}
~~~

授权通过，但用户要求提交前确认，所以状态从 AUTHORIZED 到 PREPARED，不进入执行。确认界面显示规范化后的时间、标题、目标日历和时区。确认记录绑定 Invocation digest：

~~~text
confirmation_id = confirm_01
invocation_digest = H(canonical invocation inv_01)
decision = approve
~~~

若用户改时间，运行时必须创建新 Invocation 与 digest，再次确认。模型不能在确认后修改参数。

| 事件 | 状态 | 关键 payload |
|---|---|---|
| t04 invocation.normalized | VALIDATED | reference clock、timezone |
| t05 authorization.allowed | AUTHORIZED | subject、capability、scope |
| t06 confirmation.requested | PREPARED | invocation digest |
| t07 confirmation.approved | PREPARED | matching digest |

## 7.13 轨迹三：执行与提交

运行时生成幂等键：

~~~text
scope = user-42/primary
tool_version = calendar.create_event@v1
idempotency_key = req-42:create-event:1
request_digest = H(normalized_arguments)
~~~

第一次且唯一一次外部 attempt 返回：

~~~json
{
  "status": "ok",
  "event_id": "evt_6K2",
  "committed_at": "2026-07-19T14:00:04+08:00",
  "start": "2026-07-20T09:00:00+08:00"
}
~~~

event ID 被外部系统确认的时刻就是本夹具的 commit boundary。

| 事件 | 状态 | 是否已有副作用 |
|---|---|---|
| t08 invocation.started | EXECUTING | 尚未确认 |
| t09 external.committed | COMMITTED | 是，evt_6K2 已存在 |
| t10 result.encoded | OBSERVED | 是，结果已成为 tool message |

若 t09 后收到取消，日历事件不消失；只能跳过第二次模型总结，最终请求状态标为 cancelled-after-commit，并返回可查询的 event ID 或恢复指引。

## 7.14 轨迹三：第二次模型调用

运行时把原消息、模型 Proposal 和 tool result 重新渲染：

~~~text
model_call_id = mc_02
input_token_count = 132
input_ids_shape = [1,132]
residual_shape = [1,132,16]
per_layer_kv_shape = K/V [1,2,132,4]
raw_logits_shape = [1,256]
tool_call_id = p_01
tool_result_status = ok
tool_result_event_id = evt_6K2
~~~

引擎可以复用与第一次调用完全相同的前缀 cache，但新增的 assistant proposal 与 tool message 必须经过 prefill；不能让模型在未看到结果时直接生成成功措辞。

第二次调用返回：

> 已在 2026 年 7 月 20 日上午 9:00 创建“带护照”提醒。

| 事件 | 组件 | 结果 |
|---|---|---|
| t11 model.started | model runtime | mc_02、132 input tokens |
| t12 text.delta | transport | 有序自然语言片段 |
| t13 response.completed | runtime | finish reason EOS |
| t14 trace.closed | tracing | external commit = true |

这句总结的依据是 tool result；真正的日历副作用已经在 t09 发生，而不是在 t12 的自然语言中发生。

## 7.15 轨迹三的失败注入

同一个 fixture 应运行四个失败分支：

1. **额外字段。** Proposal 增加 notify_all；因 additionalProperties=false，在 t03 前进入 VALIDATION_REJECTED，不得授权或执行。
2. **确认不匹配。** 确认 digest 属于 9:00，Invocation 被改为 10:00；在 t07 拒绝，必须重新确认。
3. **提交前超时。** 外部系统明确返回未提交；可按同一幂等键和预算重试。
4. **提交后响应丢失。** 本地先看到 timeout，状态为 OUTCOME_UNKNOWN；以同一键查询得到 evt_6K2 后转为 COMMITTED，不得创建第二个事件。

这些分支分别检验 schema、确认绑定、重试与提交边界，而不是检验模型是否能写出更有说服力的解释。

## 7.16 三条轨迹的状态对照

| 问题 | 文本轨迹 | 扩散轨迹 | 工具轨迹 |
|---|---|---|---|
| 初始确定对象 | 12 个提示 IDs | 条件张量 + 初始噪声 | 消息 + ToolSpec + 主体 |
| 模型状态 | residual、KV、logits | 整体 latent、timestep | 两次模型调用各自 cache/logits |
| 运行时更新 | processor、选择、stop | CFG、DDIM scheduler | parse、validate、authorize、execute |
| 不可逆点 | 已发送字节通常不回滚 | 最终文件发布可另定义 | external commit t09 |
| 随机入口 | 本夹具 greedy，无 RNG | 初始 latent RNG | 模型解码固定；外部 ID 由服务产生 |
| 终止 | EOS | schedule 完成 | tool observed 后第二次文本 EOS |
| 重放核心 | token 与事件序列 | tensor shape、schedule、RNG | Invocation、幂等键、commit status |

同一个词“输出”在三列中分别指增量文本、整体媒体与外部结果后的自然语言。只有先确定状态与不可逆点，取消、复现和失败才有明确含义。

## 7.17 第一次分歧算法

对任意两条同类轨迹 $A,B$，按拓扑顺序比较：

~~~text
for each corresponding event pair (a_i, b_i):
    compare event type and parent relation
    compare immutable artifact IDs
    compare discrete state exactly
    compare tensors under declared tolerance
    compare RNG counter before random choice
    compare external commit/result identity
    stop at the first failed comparison
~~~

之后的差异通常是首个分歧的后果，不应倒过来用最终结果猜原因。跨范式轨迹不比较“第 10 步是否相同”，而比较各自的 state/update/terminal 契约是否被满足。

## 7.18 本卷的结论边界

至此，本卷可以对“一次生成如何发生”给出完整回答：

1. 请求通过版本化模板与 tokenizer 变成确定输入张量；
2. prefill 逐层建立提示 KV，并只产生首步原始 logits；
3. 有序处理器把 logits 变成候选分布，选择器更新解码状态；
4. decode、反分词、停止串与 streaming 按状态机推进；
5. 扩散、离散修订与 flow 使用不同的整体状态和更新核；
6. 工具 Proposal 只有经验证、授权和执行后才可能越过现实提交边界；
7. 三类轨迹都能由事件、不变量、失败条件与重放记录检查。

本卷不从执行事件推导训练概率为何如此，也不从某个张量直接命名内部概念，更不从成功 trace 推出输出必然真实。它只把机器实际完成的步骤讲清楚。概率的形成由卷三继续，可解释性方法由卷四继续。
