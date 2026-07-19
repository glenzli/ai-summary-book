# 第二章 Prefill：整层状态演化与 KV 建立

Prefill 接收已经确定的输入张量，对全部提示位置执行一次因果前向，建立各层 KV cache，并产生首个输出 token 的原始 logits。卷一已经说明 Transformer 的设计，本章只回答执行问题：每一层读取什么、写入什么，缓存与首步 logits 在哪个时刻成立。

## 2.1 Prefill 的接口契约

对 batch size $B$、矩形长度 $n$，一个最小接口是：

```text
prefill(
    input_ids: int[B, n],
    attention_mask,
    position_descriptor,
    model_snapshot,
    numerical_config,
    cache_config
) -> PrefillResult {
    next_logits: float[B, |V|],
    kv_cache,
    valid_lengths: int[B],
    cache_metadata
}
```

`model_snapshot` 必须解析到不可变权重及适配器；`numerical_config` 至少确定权重/激活/cache 精度和关键 kernel；`cache_config` 确定 KV 布局与位置约定。逻辑接口不要求公开物理显存地址。

**前置条件 2.1**

1. 每个非 padding ID 位于词表范围内；
2. token-validity mask $G\in\{0,1\}^{B\times n}$ 与有效位置一致；
3. position descriptor 对所有有效位置有定义；
4. 模型、embedding、输出词表和 tokenizer 的词表大小兼容；
5. 推理模式下训练专用 dropout 已关闭，或其随机状态被显式记录。

违反前置条件时应在进入模型或第一层前失败，而不是产生一个看似正常但语义未定义的回答。

## 2.2 层首状态

设模型宽度为 $d$，层数为 $L$。embedding 后的残差流为

$$
R_0\in\mathbb R^{B\times n\times d}.
$$

对第 $\ell$ 层，层首的逻辑状态可记作

$$
\mathcal M_\ell=(R_\ell,G,M,P,C_{<\ell}),
$$

其中 $G[b,i]=1$ 表示样本 $b$ 的位置 $i$ 有效，$M\in(\mathbb R\cup\{-\infty\})^{B\times n\times n}$ 是由 $G$、因果规则与位置描述构造的 attention bias，$P$ 是位置描述，$C_{<\ell}$ 是已经为较低层写好的缓存。对有效 query，普通因果 self-attention 可取

$$
M[b,i,j]=
\begin{cases}
0,&G[b,i]=G[b,j]=1,\ j\le i,\\
-\infty,&G[b,i]=1\text{ 且其余情形}.
\end{cases}
$$

无效 query 行不进入 softmax，其输出随后被丢弃；kernel 若仍物化这些行，必须另用哨兵规则避免全 $-\infty$ softmax。计算第 $\ell$ 层不会修改更低层的逻辑 K/V；缓存管理器可以搬移物理页，但不能改变它们代表的 token 与位置。

下面以常见 pre-norm、grouped-query decoder 层为具体执行模板。其他架构可以替换规范化、位置编码或 MLP，但必须给出同样清楚的读写契约。

## 2.3 Attention 子层的逐张量演化

令 query 头数为 $H_q$，KV 头数为 $H_{kv}$，且 $H_{kv}\mid H_q$；头维度为 $d_h$。规范化后：

$$
\widehat R_\ell=\operatorname{Norm}_\ell(R_\ell)
\in\mathbb R^{B\times n\times d}.
$$

线性投影与 reshape 得到

$$
Q_\ell\in\mathbb R^{B\times H_q\times n\times d_h},
$$

$$
K_\ell,V_\ell\in
\mathbb R^{B\times H_{kv}\times n\times d_h}.
$$

位置机制作用后，记实际用于点积的 query/key 为 $\widetilde Q_\ell,\widetilde K_\ell$。对于 query 头 $h$，令

$$
g(h)=\left\lfloor\frac{hH_{kv}}{H_q}\right\rfloor
$$

给出它共享的 KV 头。score 张量为

$$
S_\ell[b,h,i,j]
=\frac{\langle\widetilde Q_\ell[b,h,i,:],
\widetilde K_\ell[b,g(h),j,:]\rangle}{\sqrt{d_h}}
+M[b,i,j].
$$

因果 mask 对 $j>i$ 给出 $-\infty$，padding mask 对无效 key 位置同样给出 $-\infty$。一个有效 query 行必须至少有一个有限元素；否则 softmax 的结果未定义，应该在 mask 构造阶段拒绝。

稳定 softmax 对每行有限分数先减最大值：

$$
A_\ell[b,h,i,j]
=\frac{\exp(S_\ell[b,h,i,j]-m_{bhi})}
{\sum_{k:\,S_\ell[b,h,i,k]>-\infty}
\exp(S_\ell[b,h,i,k]-m_{bhi})},
$$

其中 $m_{bhi}$ 是该行有限 score 的最大值。然后

$$
O_\ell[b,h,i,:]
=\sum_j A_\ell[b,h,i,j]V_\ell[b,g(h),j,:].
$$

各 query 头拼接并经过输出投影后写回残差流：

$$
U_\ell=R_\ell+\operatorname{OutProj}_\ell
(\operatorname{ConcatHeads}(O_\ell)).
$$

FlashAttention 一类 kernel 不必显式物化完整 $S_\ell$ 和 $A_\ell$，但其逻辑结果仍由上述张量关系定义。实现优化不能改变 mask、位置或 head-sharing 语义。基础算法来源见[资料源](SOURCES.md#source-transformer)。

## 2.4 MLP 子层与层终状态

对 gated MLP，一种常见形式是

$$
G_\ell=\phi(\operatorname{Norm}(U_\ell)W_{g,\ell})
\odot(\operatorname{Norm}(U_\ell)W_{u,\ell}),
$$

$$
R_{\ell+1}=U_\ell+G_\ell W_{d,\ell}.
$$

MLP 对每个位置独立应用同一组权重；位置间的信息交换已经发生在 attention 子层。对执行追踪而言，关键不是把每个非线性赋予自然语言含义，而是记录：层首残差、attention 更新、MLP 更新和层终残差具有相同的 $B\times n\times d$ 外形，并按固定次序相加。

第 $\ell$ 层完成时，逻辑状态转移是

$$
(R_\ell,M,P,C_{<\ell})
\longmapsto
(R_{\ell+1},M,P,C_{\leq\ell}).
$$

其中新增加的 $C_\ell$ 正是本层对有效提示位置的 key/value。

## 2.5 KV cache 在何处写入

本卷采用以下缓存契约：

```text
LayerCache[l] = {
    keys:   K_used_by_attention[B, H_kv, logical_length, d_h],
    values: V_used_by_attention[B, H_kv, logical_length, d_h],
    positions,
    valid_lengths,
    layout_metadata
}
```

若位置机制把旋转应用于 key 后再参与 attention，则这里的 `keys` 指旋转后的 key；若某模型选择缓存旋转前表示并在读取时变换，必须把这一差异写入 `layout_metadata`。两种实现都可以正确，但缓存不能跨这两种契约直接复用。

对 batch 中样本 $b$ 的有效长度 $n_b$，prefill 完成后每层应满足

$$
\operatorname{len}(K_\ell^{(b)})
=\operatorname{len}(V_\ell^{(b)})=n_b.
$$

物理实现可以分页存储、量化或为不同序列分配不连续块；逻辑读取顺序仍必须对应位置 $0,\ldots,n_b-1$ 或明确记录的 position IDs。

**缓存不变量 2.2**

1. 所有层对同一序列具有相同逻辑 token 长度；
2. 每层 K/V 的 batch、KV 头和头维度匹配模型配置；
3. cache slot 与 token/position 的映射在一次 attention 读取期间不变；
4. padding 不得伪装成可见历史位置；
5. cache 的模型、adapter、位置机制、精度和布局版本必须与当前 decode 调用兼容。

## 2.6 Prefill 伪代码

下面的伪代码刻意把逻辑张量与物理缓存分开：

```text
function prefill(input_ids, valid_mask, positions, model, cache_manager):
    assert_input_contract(input_ids, valid_mask, positions, model)
    attention_bias = build_attention_bias(
        valid_mask, positions, model.attention_rule
    )
    residual = embedding_lookup(model.embedding, input_ids)
    cache = cache_manager.allocate(
        batch_lengths(valid_mask), model.cache_layout
    )

    for l in 0 .. model.num_layers - 1:
        x = norm_l(residual)
        q, k, v = project_qkv_l(x)
        q_used, k_used = apply_position_l(q, k, positions)

        for b in 0 .. batch_size - 1:
            idx = valid_token_indices(valid_mask[b])
            cache.write_sequence(
                batch_index=b,
                layer=l,
                logical_positions=positions[b, idx],
                keys=k_used[b, :, idx, :],
                values=v[b, :, idx, :]
            )

        attn = causal_attention(
            q_used, k_used, v, attention_bias, valid_mask
        )
        residual = residual + output_projection_l(attn)
        residual = residual + mlp_l(norm_after_attn_l(residual))

    final = final_norm(residual)
    h_last = gather_last_valid_position(final, valid_mask)
    next_logits = unembedding(h_last)

    assert_cache_invariants(cache, valid_mask, model)
    return next_logits, cache, valid_lengths(valid_mask)
```

某些 fused kernel 会同时投影、旋转、写缓存和计算 attention；伪代码没有要求按独立 kernel 执行，只要求结果等价。`gather_last_valid_position` 不能简单取矩形张量的最后一列，除非 batch 使用右对齐且该列对每个样本都有效。

## 2.7 首步 logits 的条件前缀

经过 $L$ 层和最终 normalization，样本 $b$ 的最后有效提示位置表示为 $h_b\in\mathbb R^d$。词表投影得到

$$
z_1^{(b)}=W_Uh_b+b_U\in\mathbb R^{|V|},
\qquad W_U\in\mathbb R^{|V|\times d}.
$$

它表示下一位置的原始分数，条件前缀恰为该样本全部有效输入 token。此时：

- 首个输出 token 尚未被选择；
- cache 只含提示位置；
- `next_logits` 与 cache 代表同一个条件前缀；
- 若服务返回 prompt logprobs，那是额外读取各提示位置 logits 的功能，不改变首步接口。

这是[导论约定 0.2](ch00_from_input_to_output.md#03-阶段与数据产品)的 prefill 实例。

## 2.8 一个具体形状账本

取合成模型：

```text
B = 1, n = 12, d = 8, L = 2
H_q = 2, H_kv = 1, d_h = 4, |V| = 20
```

使用[第一章夹具](ch01_text_tokens_context.md#16-一个可手算的输入夹具)的 12 个 token。每层逻辑形状为：

| 对象 | 形状 |
|---|---|
| 层首残差 $R_\ell$ | `[1, 12, 8]` |
| query | `[1, 2, 12, 4]` |
| key/value | 各 `[1, 1, 12, 4]` |
| score（若物化） | `[1, 2, 12, 12]` |
| attention 输出（拼接前） | `[1, 2, 12, 4]` |
| 层终残差 $R_{\ell+1}$ | `[1, 12, 8]` |

两层缓存合计保存

$$
2\times L\times n\times H_{kv}\times d_h
=2\times2\times12\times1\times4=192
$$

个标量；开头的 2 表示 key 与 value。最终只取位置 11 的 `[8]` 向量，投影出 `[20]` 首步 logits。该账本不依赖具体参数值，适合检查 reshape、KV 头映射和缓存长度。

## 2.9 前缀缓存复用

若请求 $R'$ 的 token 前缀与已缓存请求 $R$ 的前 $k$ 个 token 完全相同，系统可以复用前 $k$ 个位置的 KV，但还必须同时满足：

1. 模型权重、adapter 与层配置相同；
2. 位置编号和位置变换参数相同；
3. attention mask 对该前缀的语义相同；
4. cache 精度与布局兼容；
5. 复用边界不切入不可独立处理的模态区段；
6. 缓存所属租户和访问策略允许复用。

一个 prefix hash 只能用于查找候选缓存，不能单独证明以上条件。命中后应核对不可变元数据；跨用户缓存还必须避免让命中时间或缓存内容泄露私有前缀。

复用不改变逻辑结果：系统把缓存视为已经对前 $k$ 个 token 完成 prefill，再对剩余后缀继续前向。若逻辑 logits 改变，就不是透明优化。

## 2.10 数值复现与逐层校验

在实数抽象中，固定参数和输入唯一确定张量。实际设备还包含低精度乘法、并行归约、近似函数和 kernel 选择。要判断差异首次出现在哪层，可以运行逐层校验：

```text
for each layer l:
    compare shape and finite-value mask
    compare cache logical lengths and positions exactly
    compare residual checksum or sampled tensor values
    compare max_abs_error and relative_error against declared tolerance
compare final raw logits and top-ranked IDs
```

摘要只能快速发现差异；浮点张量的位级摘要不同不说明误差一定有行为影响。反过来，只比较最终 top-1 相同也不足以说明分布或后续采样相同。

一个可复现实验至少固定：模型快照、adapter、输入三张量、精度、设备与 kernel 软件版本、是否允许非确定归约、cache 布局，以及比较容差。先在单 batch、无 prefix reuse 的路径建立基准，再逐项启用批处理与缓存优化。

## 2.11 Prefill 的失败条件

| 失败 | 违反的不变量 | 典型症状 |
|---|---|---|
| 左填充样本仍取矩形最后位置 | 首步条件前缀 | 短样本首 token 异常 |
| 某层漏写一个 KV 位置 | 各层长度一致 | 首次 decode shape error 或静默错位 |
| position IDs 在复用后从零重启 | token/position 映射 | cache 命中时结果改变 |
| GQA query 头映射到错误 KV 头 | 模型配置兼容 | logits 从第一层开始偏离 |
| padding key 未屏蔽 | 有效位置语义 | batch 组合改变单样本结果 |
| 复用不同 adapter 的缓存 | 快照一致 | 前缀结果污染 |
| 全掩码 query 行进入 softmax | 有限 score 存在 | NaN 传播到 logits |

Prefill 结束时只交付两个输出：提示 cache 与首步**原始** logits。温度、top-p、JSON grammar 和 repetition penalty 都尚未应用；它们属于[第三章的解码状态](ch03_logits_and_next_token.md)。
