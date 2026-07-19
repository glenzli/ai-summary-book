# 第三章 Logits、Token 概率与序列事件

卷二沿一次前向执行展示了 logits 怎样产生。本章只研究这些数值的概率含义：哪些变换不改变分布，怎样从逐 token 条件分布得到可变长序列分布，以及何时可以把字符串、候选答案或语义命题视为一个事件。

## 3.1 Softmax 只识别相对 logit

对词表 $V$，最后隐藏状态 $h$ 经输出投影得到

$$
z=W_Uh+b,
\qquad W_U\in\mathbb R^{|V|\times d},
\qquad z\in\mathbb R^{|V|}.
$$

温度为 1 时，

$$
q_i=\frac{e^{z_i}}{\sum_je^{z_j}}.
$$

两个 token 的概率比满足

$$
\frac{q_i}{q_j}=e^{z_i-z_j},
\qquad
\log\frac{q_i}{q_j}=z_i-z_j.
$$

因此 logit 差决定赔率，单个 logit 的绝对值没有独立概率意义。

**命题 3.1（softmax 的等价类）。** 对有限向量 $z,z'$，

$$
\operatorname{softmax}(z)
=\operatorname{softmax}(z')
$$

当且仅当存在常数 $c$ 使 $z'=z+c\mathbf 1$。

**证明。** 若 $z'=z+c\mathbf 1$，分子分母都乘 $e^c$，分布不变。反之，若两个 softmax 相等，则对任意 $i,j$，

$$
e^{z_i-z_j}=\frac{q_i}{q_j}=e^{z'_i-z'_j},
$$

故 $z'_i-z_i=z'_j-z_j$。该差与坐标无关，记为 $c$。证毕。

所以模型概率只识别 logits 模去常数平移后的等价类。比较不同实现的 raw logits 时，可以先减去同一基准或使用 logit difference。

## 3.2 Log-sum-exp 与数值稳定性

定义

$$
\operatorname{LSE}(z)=\log\sum_je^{z_j}.
$$

则

$$
\log q_i=z_i-\operatorname{LSE}(z).
$$

为避免指数溢出，取 $m=\max_jz_j$ 后计算

$$
\operatorname{LSE}(z)
=m+\log\sum_je^{z_j-m},
$$

以及

$$
q_i=\frac{e^{z_i-m}}{\sum_je^{z_j-m}}.
$$

该变换在实数算术中完全等价。低精度舍入、近似指数与并行归约仍可产生微小差异；当候选接近并列或位于 top-k 边界时，微差可改变被选 token，并使后续轨迹分叉。

这类现象应写成“实际数值实现改变了计算出的 $q$”，而不是“同一个精确分布自己随机改变”。

## 3.3 条件必须包括完整机器可见前缀

第 $t$ 步分布应写为

$$
q_\theta(x_t\mid x_{<t},c),
$$

其中 $c$ 包含系统消息、用户消息、工具结果、图像编码、检索片段、chat template 控制 token 和截断后的实际输入。界面上看似相同的问题，经不同模板序列化后可以是不同条件。

因此不同提示下的概率不是同一随机变量的重复观测。若研究提示敏感性，应把提示变换写成自变量，并检查它是否同时改变语义、格式和候选位置。

模型有有限上下文窗口时，真实条件是保留下来的 token 窗口，而不是调用者原先提交但已被截掉的全部文本。

## 3.4 固定长度序列的链式概率

对固定 token 序列 $y_{1:m}$，

$$
q_\theta(y_{1:m}\mid c)
=\prod_{t=1}^{m}
q_\theta(y_t\mid c,y_{<t}),
$$

$$
\log q_\theta(y_{1:m}\mid c)
=\sum_{t=1}^{m}
\log q_\theta(y_t\mid c,y_{<t}).
$$

每一步都必须用候选序列自己的真实前缀重新前向计算。不能把首步分布中多个 token 的概率相乘得到整句概率，因为第二步以后条件已经改变。

这里 $q(y_{1:m}\mid c)$ 是“生成序列的前 $m$ 个 token 等于该前缀”的圆柱事件概率，不一定是“完整回答恰好等于该序列”的概率。

## 3.5 可变长回答必须包含终止机制

若 EOS 是终止 token，则完整有限回答 $y_{1:m}$ 的质量为

$$
q_\theta^{\mathrm{fin}}(y_{1:m}\mid c)
=\left[
\prod_{t=1}^{m}
q_\theta(y_t\mid c,y_{<t})
\right]
q_\theta(\mathrm{EOS}\mid c,y_{1:m}).
$$

只有在生成过程以概率 1 最终停止时，所有有限回答的质量才求和为 1。若模型存在永不产生 EOS 的正概率路径，有限字符串只形成次概率分布；工程上的最大长度或超时会另外把这些路径映射为“截断返回”。

因此比较不同长度完整答案时忽略 EOS，会遗漏“此处结束”本身的概率。stop string、最大 token 数和客户端截断也必须纳入系统事件，而不能事后当作显示细节。

## 3.6 长度与三种不同分数

总 log probability

$$
\ell_{\mathrm{sum}}(y)=\sum_{t=1}^m\log q_t
$$

通常随长度下降。平均 token log probability

$$
\ell_{\mathrm{avg}}(y)=\frac1m\sum_{t=1}^m\log q_t
$$

消除了线性长度尺度，却不再是某个序列事件的对数概率。带参数的长度惩罚又定义另一种排序分数。

三者回答不同问题：

- 总 logprob：模型给该 token 路径多少联合质量；
- 平均 logprob：每个 token 的平均可预测程度；
- 长度惩罚分数：搜索算法人为规定的候选偏好。

没有一种归一化在所有任务上自动等于事实性、语义质量或效用。研究报告必须说明使用的是概率还是启发式分数。

## 3.7 Perplexity 的比较条件

在 $N$ 个目标 token 上，平均 NLL 为

$$
L=-\frac1N\sum_{i=1}^N
\log q_\theta(x_i\mid x_{<i}),
$$

perplexity 定义为

$$
\operatorname{PPL}=e^L.
$$

若使用以 2 为底的对数，则为 $2^{L_2}$。PPL 是平均 token 对数损失的单调变换，不增加新信息。

跨 tokenizer 直接比较 PPL 通常不成立，因为预测单位和 $N$ 都改变。若评测相同字节串，可报告 bits per byte

$$
\operatorname{BPB}
=-\frac{1}{B\log 2}
\sum_i\log q_\theta(x_i\mid x_{<i}),
$$

其中 $B$ 为统一规范化文本的字节数。即便如此，还必须统一文档边界、上下文窗口和是否计入特殊 token。

## 3.8 从 token 序列到显示字符串

设反分词和显示规范化映射为

$$
d:\mathcal Y\to\mathcal S,
$$

其中 $\mathcal Y$ 是有限 token 序列集合，$\mathcal S$ 是显示字符串集合。用户看到字符串 $s$ 的模型诱导概率原则上为推前分布

$$
q_d(s\mid c)
=\sum_{y:d(y)=s}q^{\mathrm{fin}}_\theta(y\mid c).
$$

若 tokenizer 对生成路径给每个显示字符串唯一表示，该和式只有一项；若不同 token 序列会反分词或规范化为同一字符串，则必须聚合。客户端删除空白、截断标记或 Unicode 规范化也会改变 $d$。

因此 token 概率既不是词义概率，也不是界面字符串概率；后两者需要明确映射。

## 3.9 语义事件是额外建模层

设 $g:\mathcal S\to\mathcal C$ 把显示文本映射到语义类别，例如“回答巴黎”“回答其他城市”“拒答”“含糊”。由模型与映射共同诱导

$$
Q_g(C=k\mid c)
=\sum_{s:g(s)=k}q_d(s\mid c).
$$

这是合法的事件概率，但它依赖 $g$。同一句话可能部分表达多个命题，语境中的指代也可能使 $g$ 不唯一。若使用人工、规则或另一个模型近似 $g$，评估中还要计入判定误差。

一个 token 的 softmax 概率不能跳过 $d$ 和 $g$ 直接成为“命题为真的概率”。第六、七章将讨论如何用采样近似这类语义事件及怎样审计映射误差。

## 3.10 Forced scoring 与候选条件化

即使模型没有自由生成候选，也可用 teacher forcing 计算每个候选 $a_k$ 的完整序列 log likelihood。若候选事件互斥，其开放空间质量为 $q(a_k\mid c)$。

常见做法是在候选集合 $A=\{a_1,\ldots,a_K\}$ 内重新归一化：

$$
\widetilde q(a_k\mid c,A)
=\frac{q(a_k\mid c)}{\sum_{j=1}^Kq(a_j\mid c)}.
$$

它等于模型分布条件于“输出恰属于列出的候选表示”的结果，前提是这些事件互斥且分母非零。候选外质量被条件化掉，并非真的消失。

候选实验至少要控制：

- 标签或答案的 tokenization；
- 候选措辞、顺序、长度与 EOS；
- 是评分标签 token 还是完整答案；
- 候选是否覆盖所有有效语义类别；
- 前导空格、大小写和标点。

候选内概率可以用于排序和校准，但其总体是这个人为给定的候选任务，而不是开放问答空间。

## 3.11 原始模型分布与服务 logprob

服务端可能执行

$$
z
\xrightarrow{\text{penalty}}z'
\xrightarrow{\text{mask}}z''
\xrightarrow{\text{temperature}}q'
\xrightarrow{\text{truncate}}\widetilde q.
$$

API 返回的 logprob 可能对应原始 softmax、温度后分布、截断后分布，或只返回若干候选而没有完整归一化信息。文档未定义时，不能自行假设。

整段生成后还可能经过安全过滤、重写、工具执行和客户端清理。此时用户观察到的是完整系统的推前分布，而不只是基础模型序列概率。

## 3.12 一个可核算例子

设某一步三个 logits 为

$$
z=(2,1,0).
$$

未归一化权重为 $(e^2,e,1)$，故第一项与第二项赔率为 $e:1$，第一项与第三项为 $e^2:1$。所有 logits 加 100 后概率不变。

若温度改为 $T=2$，logits 变为 $(1,1/2,0)$，相应赔率变为 $e^{1/2}:1$ 与 $e:1$。若再只保留前两项，则第三项概率变为零，前两项按其原相对赔率重新归一化。

这个例子展示了三个互不相同的对象：原始 logit 差、温度变换后的概率和截断后的服务分布。temperature 没有添加知识，top-k 也没有证明被删除 token 不可能。

## 3.13 本章结论

softmax 只识别 logit 差；固定长度前缀概率由链式分解给出；完整回答还必须包含 EOS 或服务停止机制。字符串和语义事件是 token 序列分布经显式映射得到的推前分布，候选内归一化则额外条件化了输出空间。下一章把解码器整体视为条件随机变换，分析每一步怎样改变支持集、赔率和最终系统分布。
