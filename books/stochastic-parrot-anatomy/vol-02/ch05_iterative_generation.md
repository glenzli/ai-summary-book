# 第五章 自回归、扩散与 Flow 的可执行差异

“生成”不是单一循环。自回归语言模型追加离散 token；连续扩散更新整块带噪张量；离散扩散或掩码模型并行修订多个位置；flow 模型用数值积分运输连续状态。比较它们时，本章只问可执行问题：状态是什么、一次网络调用预测什么、更新核如何计算、随机数在哪里消费、何时终止。

## 5.1 共同执行接口

任何本章讨论的生成器都可以由七个对象描述：

```text
Generator = {
    state_space,
    initial_state,
    condition,
    model_evaluation,
    update_rule,
    terminal_rule,
    output_decoder
}
```

给定条件 $c$，第 $k$ 步写成

$$
u_k=f_\theta(s_k,k,c),
\qquad
s_{k+1}=F(s_k,u_k,k,\xi_k),
$$

其中 $s_k$ 是当前状态，$u_k$ 是网络输出，$F$ 是模型之外或与模型共同定义的更新规则，$\xi_k$ 是可选随机输入。这个记号用于区分“神经网络的一次评估”和“采样器的一次更新”。

复现一种生成器，至少要固定全部七个接口字段；只固定 checkpoint 与 seed 不够。

## 5.2 左到右自回归

对离散输出 $y_{1:T}$，自回归分解是

$$
p_\theta(y_{1:T}\mid c)
=\prod_{t=1}^{T}p_\theta(y_t\mid y_{<t},c).
$$

其执行状态是

```text
s_t = (fixed_condition, selected_prefix, KV_cache, decoder_state)
```

一次网络评估读取新 token 与旧 cache，产生下一 token logits；一次更新从 logits 选择 token 并追加前缀。已确定前缀通常不再修改，输出长度由 EOS、停止串或上限动态决定。

```text
logits, cache = prefill(condition_tokens)
while not terminal:
    y = choose(process(logits, decoder_state))
    update decoder_state with y
    emit safe decoded bytes from y
    if terminal_after(y): break
    logits, cache = decode_one(y, cache)
```

严格的步间依赖位于“选择后才能确定下一次模型输入”。batch 内不同请求可以并行，一次 prefill 的提示位置也可以并行，但同一路径的未知 token 不能在没有额外算法的情况下全部同时确定。

Encoder–decoder 模型仍可以自回归：encoder memory 固定为条件，decoder cache 随输出增长。网络接口不同，不改变输出侧的左到右依赖。

## 5.3 连续扩散的状态与更新

离散时间高斯扩散先定义前向扰动。对 $t=1,\ldots,T$，取 $0<\beta_t<1$，令 $\alpha_t=1-\beta_t$、$\bar\alpha_0=1$ 与 $\bar\alpha_t=\prod_{r=1}^t\alpha_r$，于是 $0<\bar\alpha_t<1$，并有

$$
x_t=\sqrt{\bar\alpha_t}x_0
+\sqrt{1-\bar\alpha_t}\,\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).
$$

采样时不执行这条前向公式，而是从 $x_T\sim\mathcal N(0,I)$ 开始，用网络预测噪声、干净样本或等价参数。以噪声预测 $\epsilon_\theta(x_t,t,c)$ 为例，一种 DDPM 形式的逆更新是

$$
x_{t-1}=
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}
\epsilon_\theta(x_t,t,c)
\right)
+\sigma_t\xi_t,
$$

其中 $\xi_t\sim\mathcal N(0,I)$，最后一步常令噪声项为零；$\sigma_t$ 由具体 sampler 决定。公式只在网络参数化、时间索引与方差约定一致时可用，不能从一种 scheduler 复制到另一种而不核对定义。

执行状态是整块张量：

```text
s_t = (x_t, timestep_index, scheduler_state, rng_state)
```

一次网络调用通常输出与 $x_t$ 同形的预测，一次更新作用于所有空间位置。中间状态可以整体预览，却不是不可变前缀。DDPM 与 DDIM 的原始来源见[资料源](SOURCES.md#source-diffusion)。

## 5.4 DDIM：同一网络，不同执行路径

先由噪声预测恢复干净样本估计：

$$
\widehat x_0=
\frac{x_t-\sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t,t,c)}
{\sqrt{\bar\alpha_t}}.
$$

实际 sampler 可沿任意严格递减日程 $T=t_0>t_1>\cdots>t_N=0$ 跳步。若当前时刻为 $t$、日程中的下一时刻为 $s<t$，DDIM 类更新写为

$$
x_s=
\sqrt{\bar\alpha_s}\widehat x_0
+\sqrt{1-\bar\alpha_s-\sigma_{t\to s}^2}\,
\epsilon_\theta(x_t,t,c)
+\sigma_{t\to s}\xi_t,
\qquad
0\le \sigma_{t\to s}^2\le 1-\bar\alpha_s.
$$

约束保证第二个平方根有定义；$\xi_t\sim\mathcal N(0,I)$，终点可令噪声项为零。相邻 DDIM 是 $s=t-1$ 的特例，采用 $999\to749$ 之类日程时必须使用 $\bar\alpha_{749}$，不能把下标写成字面上的 $t-1$。当所有 $\sigma_{t\to s}=0$ 时，给定初始噪声和数值实现，更新路径是确定的；这不表示最终样本没有随机性，因为初始 $x_T$ 仍由 seed 决定。改变 timestep 子序列、scheduler 或参数化，即使 checkpoint 与 seed 相同，也会改变轨迹。

这个例子说明 sampler 不是无关紧要的外壳：网络给出 $u_t$，scheduler 决定怎样把 $u_t$ 与 $x_t$ 组合成下一状态。

## 5.5 潜空间扩散与条件引导

潜空间扩散把媒体编码器/解码器加入接口。文本到图像的一次执行通常是：

```text
text bytes
-> text tokenizer and encoder -> condition c
seed -> PRNG -> initial latent z_T
for t in schedule:
    model prediction on z_t, t, c
    guidance combination
    scheduler update -> z_(t-1)
VAE decoder(z_0) -> pixels
postprocess -> encoded image bytes
```

若使用 classifier-free guidance，有条件与无条件预测组合为

$$
\widehat\epsilon
=\epsilon_\theta(z_t,t,\varnothing)
+w\bigl(
\epsilon_\theta(z_t,t,c)
-\epsilon_\theta(z_t,t,\varnothing)
\bigr).
$$

实现常把两份输入沿 batch 维拼接，只调用一次网络；逻辑上仍有两份预测。引导强度 $w$ 直接改变 scheduler 输入，不等同于语言模型的 temperature。

潜变量尺寸、VAE scaling factor、文本编码长度、negative prompt、scheduler timestep 与预测参数化都属于执行记录。最终 PNG 的编码器版本也可能在像素相同的情况下改变文件字节，应区分“潜变量相同”“像素相同”和“文件相同”。

## 5.6 Flow 与 ODE 积分

Flow 模型学习随时间变化的向量场：

$$
\frac{dx}{dt}=v_\theta(x,t,c).
$$

模型评估返回局部速度，数值求解器决定离散轨迹。显式 Euler 步为

$$
x_{k+1}=x_k+h_kv_\theta(x_k,t_k,c).
$$

Heun 步需要两次网络评估：

$$
k_1=v_\theta(x_k,t_k,c),
$$

$$
\widetilde x=x_k+h_kk_1,
\qquad
k_2=v_\theta(\widetilde x,t_k+h_k,c),
$$

$$
x_{k+1}=x_k+\frac{h_k}{2}(k_1+k_2).
$$

所以“20 steps”不足以直接比较成本：Euler 通常每步一次模型评估，Heun 每步两次；classifier-free guidance 又可能把一次逻辑预测变成两份条件评估。应记录 NFE（number of function evaluations）而不只记录循环次数。

一个可手算的求解器夹具是 $dx/dt=-x$、$x(0)=1$，用四个 $h=0.25$ 的 Euler 步得到

$$
x_4=(1-0.25)^4=0.31640625.
$$

精确解 $e^{-1}\approx0.367879$。换成 Heun 会得到另一离散值，说明模型/向量场相同并不保证 sampler 输出相同。Flow Matching 的执行背景见[资料源](SOURCES.md#source-flow)。

## 5.7 离散扩散的 Markov 接口

对有限词表或离散代码，D3PM 一类模型用转移矩阵 $Q_t$ 定义扰动：

$$
q(x_t\mid x_{t-1})
=\operatorname{Cat}(x_t; x_{t-1}Q_t).
$$

乘积 $\bar Q_t=Q_1\cdots Q_t$ 给出从干净状态到时刻 $t$ 的边缘扰动。逆模型根据当前整段状态预测 $p_\theta(x_{t-1}\mid x_t,c)$，或预测 $x_0$ 后再组合已知后验。

执行状态通常是固定长度的离散张量：

```text
s_t = (token_grid_or_sequence, noise_level, reverse_schedule, rng_state)
```

一次更新可以同时改写多个位置，且早期产生的 token 后续仍可能变化。传统左到右 KV cache 不能直接复用，因为任一已缓存位置的表示可能在下一轮被替换；需要专门的增量算法才能避免整段重算。

D3PM 的结构化离散转移来源见[资料源](SOURCES.md#source-discrete-diffusion)。

## 5.8 掩码式并行修订

掩码生成是离散迭代的一个可执行实例，但不应与所有 D3PM 混称。设输出长度 $n$ 已确定，初态全部为 `[MASK]`。第 $k$ 轮：

```text
logits = model(current_tokens, condition, round=k)
proposals = sample token for every masked position
confidence = score each proposal
commit_count = schedule(k, total_rounds, n)
commit highest-confidence positions with deterministic tie-break
optionally remask low-confidence previously filled positions
advance round
```

若只提交且不重掩码，mask 集合满足

$$
M_{k+1}\subseteq M_k.
$$

若允许修订，则这个单调不变量不成立，必须记录每轮哪些位置被改写。置信排序、并列规则、每位置采样随机数和提交日程都会改变最终结果。

MaskGIT 是这种并行修订的代表性实例；语言上的掩码扩散仍是快速发展的研究方向，不能仅凭执行并行性断言它已全面替代自回归模型。相关一手来源见[资料源](SOURCES.md#source-discrete-diffusion)。

## 5.9 四类算法的可计算对照

| 属性 | 自回归 | 连续扩散 | 离散扩散/掩码修订 | Flow/ODE |
|---|---|---|---|---|
| 状态 | 已定 token 前缀 + cache | 整体连续噪声张量 | 整段离散状态 + mask/noise level | 整体连续状态 |
| 一次网络输出 | 下一 token logits | 噪声/样本/score 预测 | 各位置类别预测或逆转移 | 向量场 |
| 更新单位 | 通常追加一个 token | 更新全部连续位置 | 更新/修订多个离散位置 | 数值积分一步 |
| 旧内容可改写 | 通常不可 | 可以 | 可以或逐轮锁定 | 可以 |
| 长度 | 常由 EOS 动态决定 | 张量形状预先确定 | 通常预定或另有长度机制 | 张量形状预先确定 |
| 主要串行轴 | 输出 token | 噪声 timestep | 修订 round | 积分 time |
| 自然 streaming | 强 | 通常是整体预览 | 不稳定草稿或最终结果 | 通常是整体预览 |
| 典型随机入口 | categorical selection | 初始噪声与可选逐步噪声 | 初态扰动与逐位置采样 | 初始基分布；ODE 步可确定 |
| 停止规则 | EOS/stop/长度 | scheduler 结束 | 轮数/无 mask/接受准则 | 积分终点/容差 |

“一步”只有结合该表中的状态与更新单位才有意义。语言输出中写出的 reasoning step 只是 token 内容，不是这里任何数值求解步的直接同义词。

## 5.10 统一伪代码与不变量

```text
state = initialize(condition, rng, algorithm_config)
record initial_state_metadata

for k in schedule:
    assert state is valid for k
    model_output = evaluate_model(state, k, condition)
    state, rng = update(state, model_output, k, rng, sampler_config)
    record state shape, update metadata, rng counter
    if terminal(state, k): break

output = decode_final_state(state, decoder_artifact)
```

共同不变量是：

1. model output 的参数化与 update rule 匹配；
2. schedule 的方向、索引和值与训练/采样约定一致；
3. 状态形状在每一步满足模型接口；
4. 条件编码在声明不变时保持不变；
5. 每个随机消费点可由独立 stream 或计数器定位；
6. terminal 成立后不再执行普通更新；
7. 最终 decoder 的版本属于生成工件，而不是无关显示层。

## 5.11 可复现记录

**自回归。** 模板、tokenizer、输入 IDs、模型快照、处理器顺序、逐步 token、RNG、stop 配置与 cache 位置。

**连续/潜扩散。** 文本编码器与 token、初始噪声生成算法、latent shape/dtype、网络与 VAE、预测参数化、完整 timestep 列表、sampler 公式与参数、guidance、每步额外噪声 RNG。

**离散扩散。** 词表、初始扰动或 mask、转移矩阵/噪声日程、输出长度、每轮 proposal 与确认/重掩码集合、并列规则、逐位置 RNG。

**Flow。** 基分布样本、向量场 checkpoint、积分方向、solver、步长或容差、NFE、adaptive-step 接受/拒绝轨迹。

若目标是位级重放，还必须固定数值库和硬件路径；若目标是算法级重放，则应声明张量误差与最终输出的比较准则。

## 5.12 失败条件

| 失败 | 所在接口 | 结果 |
|---|---|---|
| 把 $v$-prediction 当作 noise prediction | model output/update | 每步公式系统性错误 |
| scheduler timesteps 顺序反转 | schedule | 从数据端向噪声端移动 |
| CFG 两个 batch 条件顺序交换 | condition combination | 引导方向反转 |
| VAE scaling factor 不匹配 | output decoder | 图像对比度/幅值异常 |
| masked model 提交数降到 0 | terminal/progress | 循环不前进 |
| adaptive ODE 只记录“20 步” | solver trace | 无法恢复接受/拒绝路径 |
| 把中间扩散预览当稳定前缀 | transport semantics | 客户端错误累加而非替换画面 |
| 比较 steps 而忽略 NFE | cost accounting | 算力比较失真 |

本章没有评价哪种生成范式在所有任务上更好。它给出的结论更窄也更可检验：看到一项生成服务时，可以写出它的状态、模型输出、更新核、随机入口和停止规则，并据此判断两次轨迹能否比较。
