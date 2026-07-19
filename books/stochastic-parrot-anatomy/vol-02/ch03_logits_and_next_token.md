# 第三章 从 Logits 到下一个 Token

Prefill 或一次 decode 前向的直接结果是词表上的原始 logits。真正被选中的 token 还取决于运行时处理器、约束状态、截断规则与选择器。本章把这条路径写成确定的算法，并明确哪些 logprob 可以互相比较。

## 3.1 四个不能混写的对象

设词表大小为 $|V|$，最后隐藏状态为 $h\in\mathbb R^d$。词表投影给出

$$
z^{\mathrm{raw}}=W_Uh+b_U\in\mathbb R^{|V|},
\qquad W_U\in\mathbb R^{|V|\times d}.
$$

执行中至少有四个不同对象：

1. **原始 logits** $z^{\mathrm{raw}}$：模型前向的直接读出；
2. **处理后 logits** $z^{\mathrm{proc}}$：惩罚、mask 与温度按声明顺序作用后的分数；
3. **候选支撑集** $C\subseteq V$：top-k、top-p 或 grammar 后仍可被选择的 token；
4. **实际选择分布** $p^{\mathrm{sel}}$：在最终候选集上重新归一化的分布。

同一个 token 在这四层可以有不同排名或根本不可选。API 返回的 `logprob` 若未说明位于哪一层，就不是完整定义。

## 3.2 解码状态

处理器不是只读取一列 logits 的纯函数。repetition penalty、grammar 与停止匹配都依赖历史。第 $t$ 步选择前的解码状态记作：

```text
DecoderState_t = {
    prompt_token_ids,
    selected_token_ids: y[1:t-1],
    token_counts,
    sequence_constraint_state,
    grammar_state,
    stop_matcher_state,
    rng_algorithm,
    rng_state_or_counter,
    step_index: t,
    finish_status: RUNNING
}
```

该状态不包含 Transformer KV 的数值内容；KV 属于模型状态。两者通过共同的条件前缀保持同步：在第 $t$ 步首，cache 表示 `prompt + y[1:t-1]`，当前原始 logits 也以该前缀为条件。

**不变量 3.1**　每个处理器只能读取其声明的状态，并产生新的 logits、候选 mask 或处理器状态；不能静默改写已选 token 历史。

## 3.3 处理器顺序是配置的一部分

不同推理库没有必须共享的全局顺序。本书为后续轨迹固定以下参考管线：

```text
raw logits
1. model/control-token mask
2. minimum-length EOS mask
3. grammar allowed-token mask
4. banned-sequence and no-repeat masks
5. repetition/presence/frequency transforms
6. temperature scaling
7. top-k support mask
8. stable softmax on the remaining support
9. top-p support mask and final renormalization
10. greedy or categorical selection
```

这个顺序不是宣称它优于所有实现，而是提供一个可复现约定。具体服务若采用别的顺序，必须记录有序处理器列表、每个版本和参数。尤其是温度、截断与非线性惩罚一般不可交换：

$$
P_i(P_j(z))\neq P_j(P_i(z)).
$$

硬 mask 通常用 $-\infty$ 表示不可选 token。实现应在 softmax 前断言至少有一个有限候选；空支撑集必须返回约束失败，而不能从 NaN 或全零权重中继续采样。

## 3.4 硬约束与语法状态

对当前状态 $q_t$，grammar 或 schema 约束给出允许集合

$$
A(q_t)=\{i\in V:\operatorname{bytes}(i)
\text{ 仍可扩展为合法输出}\}.
$$

处理器令

$$
z_i'=
\begin{cases}
z_i,&i\in A(q_t),\\
-\infty,&i\notin A(q_t).
\end{cases}
$$

约束不能只看 token 的显示字符串。一个 token 可能包含半个转义序列、多个 JSON 字符或 UTF-8 片段；语法状态应消费 token 对应字节，并判断当前字节前缀是否仍可到达接受状态。选择 token $y_t$ 后更新

$$
q_{t+1}=\delta(q_t,\operatorname{bytes}(y_t)).
$$

grammar 保证的是形式语言成员关系。例如 JSON grammar 可以保证括号与字符串转义闭合，却不能保证城市存在、日期符合用户意图或操作有权限。工具调用的语义验证在[第六章](ch06_tools_runtime_boundary.md)完成。

## 3.5 历史相关惩罚

令 $c_i$ 是 token $i$ 在指定历史范围内的出现次数。一种常见的 presence/frequency 变换是

$$
z_i'=z_i-\lambda_{\mathrm{presence}}\mathbf 1[c_i>0]
-\lambda_{\mathrm{frequency}}c_i.
$$

另一类 sign-aware repetition penalty 取 $r\geq1$：

$$
z_i'=\begin{cases}
z_i/r,&c_i>0\text{ 且 }z_i>0,\\
r z_i,&c_i>0\text{ 且 }z_i\leq0,\\
z_i,&c_i=0.
\end{cases}
$$

两式不是同一种算法，也不应同时被一个模糊的 `repetition_penalty` 名称替代。执行配置还必须说明计数范围是否包含 prompt、是否按 token ID 还是文本片段计数，以及特殊 token 是否排除。

no-repeat n-gram 和 banned sequence 是状态自动机：它们只禁止会使当前后缀完成某个序列的下一 token。简单地永久屏蔽序列中所有 token 会过度约束输出。

## 3.6 温度与稳定 softmax

在参考管线中，温度 $T>0$ 作用于惩罚后的有限 logits：

$$
\widetilde z_i=z_i'/T.
$$

令候选集合暂为 $C$，稳定 softmax 为

$$
p_i=
\begin{cases}
\dfrac{\exp(\widetilde z_i-m)}
{\sum_{j\in C}\exp(\widetilde z_j-m)},&i\in C,\\
0,&i\notin C,
\end{cases}
$$

其中 $m=\max_{j\in C}\widetilde z_j$。$T<1$ 放大有限 logit 差，$T>1$ 压缩差异；它不添加信息，也不等同于整句事实置信度。

`temperature = 0` 不应代入上式除零。API 常把它约定为 greedy 的别名；严格接口应直接选择 `strategy = greedy`，或者明确该特殊值的分支语义。

## 3.7 Top-k 与 Top-p 的精确定义

Top-k 在当前有限候选中保留分数最高的 $k$ 项。并列边界必须有稳定规则，例如以 token ID 升序打破并列。

Top-p 先按当前概率从高到低排序为 $i_1,\ldots,i_m$，再取满足

$$
\sum_{r=1}^{K}p_{i_r}\geq p_0
$$

的最小 $K\geq1$，保留 $i_1,\ldots,i_K$。截断后重新归一化。参考管线先 top-k 后 top-p，所以 top-p 读取的是 top-k 后的归一化概率；若实现先 top-p，候选集可能不同。

边界条件应显式规定：

- $k\leq0$ 是禁用 top-k 还是非法配置；
- $p_0\notin(0,1]$ 如何处理；
- NaN、正无穷和全部负无穷 logits 是否立即失败；
- 至少保留几个 token；
- 浮点累计到阈值附近时使用何种精度与比较规则。

Nucleus sampling 的原始工作见[资料源](SOURCES.md#source-decoding)。

## 3.8 选择器与随机数消费

Greedy 选择

$$
y_t=\arg\max_{i\in C}z_i^{\mathrm{proc}},
$$

并使用已声明的 tie-break。categorical sampling 则从最终 $p^{\mathrm{sel}}$ 取样。一种参考实现是：按 token ID 升序排列候选，生成 $u\in[0,1)$，选择首个满足累计概率大于 $u$ 的 token。

```text
function categorical(candidate_ids, probabilities, rng):
    order candidates by token_id
    u = rng.next_uniform_01()
    cumulative = 0
    for id, p in candidates:
        cumulative += p
        if u < cumulative:
            return id, rng.updated_state
    return last_candidate, rng.updated_state  # rounding guard
```

相同 seed 不自动给出相同 token。还必须固定 PRNG 算法、初始 stream/subsequence、每一步消费几个随机数、候选排序和浮点累计。动态 batch 若共享一个全局 RNG，会使其他请求改变本请求的随机数消费；更容易重放的做法是为每个请求维护独立 counter-based stream。

## 3.9 一次完整的手算选择

设 grammar 和控制 mask 后只剩 token `A`、`B`、`C`，原始相关 logits 为

$$
(2.0,1.5,0.5).
$$

历史计数为 $c_A=2,c_B=1,c_C=0$，取 presence penalty $0.1$、frequency penalty $0.2$。处理后为

$$
(1.5,1.2,0.5).
$$

再取 $T=0.8$，缩放 logits 为

$$
(1.875,1.5,0.625).
$$

稳定 softmax 约为

$$
(0.507,0.348,0.145).
$$

取 top-p $p_0=0.8$，最小前缀为 `A`,`B`，重新归一化后约为

$$
p^{\mathrm{sel}}=(0.593,0.407,0).
$$

按 `A`,`B` 的累计区间，若参考 PRNG 本步给出 $u=0.63$，选择 `B`。该例同时说明：`B` 的原始 softmax、惩罚后 softmax 和最终选择概率并不相同；记录“`B` 的概率”时必须说明是哪一个。

这个例子的复现断言是：处理器顺序不变时，最终支撑集恰为 `{A,B}`，$u=0.63$ 选择 `B`。若先做 top-p 再做惩罚，断言没有理由成立。

## 3.10 选择后的状态转移

选出 $y_t$ 后，解码器按固定顺序更新：

```text
selected_token_ids.append(y_t)
token_counts[y_t] += 1
sequence_constraint_state.consume(y_t)
grammar_state.consume(token_bytes(y_t))
stop_matcher_state.observe(y_t, token_bytes(y_t))
rng_state = rng_state_after_selection
step_index += 1
```

若 token 是 EOS、达到输出上限、grammar 已接受并要求终止，或者 stop matcher 命中，状态可以直接进入终态。否则该 token 交给下一次 `decode_one`，写入 KV 并生成新的原始 logits。

语法接受与停止不是同一个概念。一个 JSON 值已经闭合时，grammar 可处于接受状态，但协议可能仍允许尾随空白；运行时必须明确是立即停止、只允许空白/EOS，还是继续普通生成。

## 3.11 多候选解码是状态集合

Beam search 不只选择一个 token，而是维护至多 $B_w$ 个假设：

$$
\mathcal H_t
=\{(y_{1:t}^{(b)},\ell_t^{(b)},K_t^{(b)},d_t^{(b)})\}_{b=1}^{B_w}.
$$

其中 $\ell_t^{(b)}$ 是累计分数，$K_t^{(b)}$ 是 KV 逻辑分支，$d_t^{(b)}$ 是约束、停止与处理器状态。扩展后选出全局得分最高的若干分支；长度惩罚、EOS 完成规则和并列处理都属于算法。把 beam width 改为 1 不一定等价于随机采样，通常接近 greedy。

本卷后续以单路径 greedy/categorical 为主，因为它最清楚地展示流式生成；多路径算法只需遵守同样的状态所有权与停止不变量。

## 3.12 失败条件与诊断字段

| 失败 | 首次异常 | 正确处理 |
|---|---|---|
| 处理器后无有限 token | 候选支撑集为空 | 返回约束错误，不调用采样器 |
| processor 顺序未版本化 | 同 logits 得到不同候选 | 记录有序列表与版本 |
| grammar 按字符而 tokenizer 按字节 | token 边界处误放行/误拒绝 | 统一消费 token bytes |
| seed 相同但 RNG 流共享 | 并发变化时 token 分叉 | 请求级 RNG 状态或计数器 |
| stop token 同时被 min-length mask | 配置矛盾 | 在请求验证时拒绝或声明优先级 |
| 返回 raw logprob 却标作 sampled logprob | 观测语义错误 | 字段名区分阶段与归一化集合 |
| 浮点 NaN 参与排序 | 排名不稳定 | 检测并失败，不依赖库排序行为 |

一次 token 选择的最小记录包括：step、原始 logits 的可比较摘要或指定候选值、处理器有序列表、各阶段候选集大小、最终候选及 logprob、PRNG 标识与计数器、所选 token、约束状态前后值。下一章把这一转移嵌入完整的 decode、stop 与 streaming 状态机。
