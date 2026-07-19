# 第一章 文本、Token 与实际上下文

用户看到字形，网络传输字节，应用处理消息对象，tokenizer 输出整数 ID。一次生成能否被重放，首先取决于这四层是否被明确区分。本章不给出通用分词算法综述，而是规定从请求到模型输入的确定性接口。

## 1.1 输入管线

对纯文本聊天请求，一条可复现的输入管线可以写成：

```text
network bytes
-> decode and parse request
-> validate request schema
-> canonical message objects
-> render versioned chat template
-> tokenizer normalization/pre-tokenization/segmentation
-> special-token post-processing
-> context-budget policy
-> input_ids, attention_mask, position_ids
```

实际实现可以把截断放在消息级并多次重新渲染，也可以直接在 token 级截断；无论采用哪种方式，**顺序本身都是模型输入接口的一部分**。只保存用户可见字符串，无法恢复模板控制符号、工具 schema、历史删减和最终 token 序列。

把这条管线记为

$$
I=\operatorname{Assemble}_{\pi}
\bigl(\operatorname{Tok}_{\tau}(\operatorname{Render}_{\kappa}(R))\bigr),
$$

其中 $R$ 是规范化请求，$\kappa$ 是模板版本，$\tau$ 是 tokenizer 版本，$\pi$ 是上下文预算策略；$I$ 包含 `input_ids`、mask 和位置。公式的作用只是列出三个不能省略的版本参数。

## 1.2 字节、码点与规范化

UTF-8 把 Unicode 码点编码为一到四个字节；一个屏幕字形可能由多个码点组成。字符串“é”既可能是单个码点 `U+00E9`，也可能是 `U+0065 U+0301`。两者可以渲染相同，却具有不同字节序列。

Unicode 定义 NFC、NFD、NFKC 与 NFKD 等规范化形式，但应用不能擅自假定 tokenizer 使用其中某一种。NFKC 还会折叠兼容字符，可能改变对全角字符、上标或排版差异的保留方式。正确做法是：

1. 明确网络字符编码与非法字节处理策略；
2. 把规范化规则绑定到 tokenizer 工件；
3. 若应用在 tokenizer 前另做规范化，记录算法和 Unicode 数据版本；
4. 比较输入时优先比较最终字节与 token IDs，而不是渲染外观。

Unicode 规范化的标准定义见[资料源](SOURCES.md#source-input)。

**输入不变量 1.1**　同一执行配置中，规范化不能取决于机器 locale、界面字体或未记录的数据库排序规则。

## 1.3 结构化消息与聊天模板

聊天接口通常接收消息数组：

```json
[
  {"role": "system", "content": "你是一个简洁的科学助手。"},
  {"role": "user", "content": "请用一句话解释为什么天空通常呈蓝色。"}
]
```

模型却只接收模板渲染后的序列。一个示意模板是：

```text
<|system|>你是一个简洁的科学助手。<|end|>
<|user|>请用一句话解释为什么天空通常呈蓝色。<|end|>
<|assistant|>
```

真实模板中的角色边界常是保留 token，不一定先表现为普通文本。工具调用 ID、tool result、图片占位和 assistant 起始标记也可能由模板插入。因而模板函数必须明确：

- 接受哪些 role 与 content 类型；
- 消息间是否插入换行或分隔符；
- 特殊 token 由模板还是 tokenizer 后处理器添加；
- 是否在末尾添加 generation prompt；
- 工具 schema 以何种字段顺序和转义规则序列化；
- 空消息、未知 role 与重复 system 消息如何处理。

**失败条件。** 若应用先把不可信文本拼成带保留标记的普通字符串，再允许 tokenizer 把该字符串解释成控制 token，用户内容可能逃逸消息边界。保留 token 应由结构化模板插入，并对普通内容采用不产生控制语义的编码路径。

## 1.4 确定性序列化契约

JSON 对象在语义上不是有序字段列表，而模板渲染是有序字节序列。若工具 schema 或消息元数据来自 map，序列化器必须固定以下选择：

1. 字段顺序或规范化序列化规则；
2. 字符串转义、斜杠与非 ASCII 编码规则；
3. 浮点数、整数和日期的文本形式；
4. 缺省值是显式写入还是省略；
5. 行尾是 LF 还是 CRLF；
6. 模板与 schema 的版本标识。

不能用“解析后 JSON 相同”代替“渲染字节相同”：tokenizer 读取后者。若实现确实只在 token 层插入控制符号，记录应直接保存最终 token IDs，而不是假造一份等价文本。

下面的接口足以表达确定性要求：

```text
render(messages, tools, template_artifact) -> RenderedInput {
    kind: "bytes" | "token_segments",
    payload,
    template_id,
    template_digest
}
```

相同参数必须产生逐字节相同的 `payload`；若模板包含动态时钟、随机 ID 或服务注入内容，这些值必须作为显式参数进入执行记录。

## 1.5 Tokenizer 的四个阶段

许多 tokenizer 可以按四个逻辑阶段理解：

1. **Normalizer**：可选地改写 Unicode、空白或大小写；
2. **Pre-tokenizer**：按空白、字节边界或模式切分候选区段；
3. **Segmentation model**：使用 BPE、Unigram 或其他词表模型选择 token；
4. **Post-processor**：加入 BOS、EOS、角色或任务所需特殊 token。

令词表为 $V$，tokenizer 实现映射

$$
T_{\tau}:\text{byte/string input}\longrightarrow(i_1,\ldots,i_n),
\qquad i_j\in\{0,\ldots,|V|-1\}.
$$

token ID 的意义只在给定 $\tau$ 时成立。同一个 ID 在另一词表中可能对应完全不同的字节；同一句话在两个 tokenizer 下也可能具有不同长度和边界。SentencePiece 等具体实现的设计来源见[资料源](SOURCES.md#source-input)。

反分词器实现另一个接口：

```text
decoder.push(token_id) -> zero or more output bytes
decoder.finish()       -> remaining bytes or decoding error
```

它不必对单个 token 立即返回合法 UTF-8。字节级词表可能把一个多字节字符拆开；前导空格也可能编码在后续 token 中。因此 `decode([a]) + decode([b])` 未必等于增量状态机依次 `push(a); push(b)` 的界面行为。

## 1.6 一个可手算的输入夹具

为了把“模板会改变 token”变成可检验事实，定义一个只用于本书的合成 tokenizer。它使用最长匹配，词表片段如下：

| ID | token |
|---:|---|
| 0 | `<pad>` |
| 1 | `<bos>` |
| 2 | `<system>` |
| 3 | `</system>` |
| 4 | `<user>` |
| 5 | `</user>` |
| 6 | `<assistant>` |
| 7 | `简洁回答` |
| 8 | `天空` |
| 9 | `为什么` |
| 10 | `是` |
| 11 | `蓝色` |
| 12 | `？` |
| 13 | `<unk>` |

模板固定渲染为：

```text
<bos><system>简洁回答</system><user>天空为什么是蓝色？</user><assistant>
```

在“不另做规范化、控制 token 只由模板插入、最长匹配、无隐式 padding/EOS”的配置下，唯一期望输出是：

```text
[1, 2, 7, 3, 4, 8, 9, 10, 11, 12, 5, 6]
```

这个夹具可以作为输入层单元测试。至少应加入以下反例：

- 删除 `<assistant>` 后长度变为 11，测试 generation prompt；
- 把 `？` 换成 ASCII `?`，在词表无该项时得到 `<unk>`；
- 在 `天空` 中插入不可见空格，确保 token 序列改变；
- 把普通用户文本写成 `<system>`，确保它不获得控制 token 权限。

夹具不声称模拟某个商用 tokenizer；它固定了足够小的接口，使模板、特殊 token 与词表版本的关系可以手工验证。

## 1.7 上下文组装与预算

实际上下文可能包含：

```text
system/developer instructions
-> tool schemas
-> conversation history
-> retrieved documents
-> current user message
-> assistant generation prefix
```

设模型窗口上限为 $N$，保留输出预算为 $g$，则送入 prefill 的长度必须满足

$$
n_{\text{input}}+g\leq N,
$$

除非服务允许生成时再动态拒绝或扩展上下文。预算策略不能只对各段字符数求和，因为模板边界和 tokenization 会改变长度。

一个确定的消息级预算算法可以是：

```text
function fit_context(required, history, optional_docs, max_input_tokens):
    keep all required segments atomically
    render and tokenize required
    if length > max_input_tokens: reject

    for segment in priority_order(history, optional_docs):
        tentatively add segment
        render and tokenize the whole candidate
        if length <= max_input_tokens:
            keep segment

    assert current user message is present
    assert every tool result keeps its matching tool call
    assert exactly one assistant generation prefix exists
    return final rendered input and tokenization
```

生产实现可用缓存避免每次完整分词，但其结果必须等价。直接从 token 序列左侧切掉若干 ID 更便宜，却可能切入消息、JSON 或多模态占位内部；若采用这种策略，就必须明确允许的结构破坏及模型所见结果。

**上下文不变量 1.2**

1. 必需段超出预算时应拒绝，而不是静默删去权限或系统边界；
2. 工具调用与对应结果应成对保留或成对移除；
3. 每个多模态输入必须映射到接口声明的占位或 cross-attention memory，数量与形状不得丢失或串线；
4. 最终 token 数、attention mask 和 position IDs 必须一致；
5. 截断决定应进入执行记录，不能只保存截断后的界面文本。

## 1.8 多模态输入也是确定性接口

界面中的图片不会以文件名进入语言模型。一个图像输入接口至少包含：

```text
image bytes + media type
-> decode with specified library/version
-> orientation and color conversion
-> resize/crop/tile policy
-> normalize pixels
-> vision encoder
-> projected visual tokens or cross-attention memory
```

要重放输入，应保存原始媒体摘要、解码错误策略、尺寸、裁剪/分块坐标、像素归一化参数、视觉编码器与投影器版本。图片 URL 不是稳定工件：内容可能变化，访问权限也可能过期。多模态架构本身由卷一说明；本章只要求输入媒体、视觉张量与文本占位或 cross-attention memory 之间存在可验证的映射，即使一张图片对应多个内部位置。

## 1.9 模型真正接收的张量

对 batch size $B$、序列长度 $n$，常见 decoder-only 接口至少有：

```text
input_ids      : int[B, n]
attention_mask : bool/int[B, n] or broadcastable attention mask
position_ids   : int[B, n] or an equivalent position descriptor
```

embedding 矩阵 $E\in\mathbb R^{|V|\times d}$ 把 ID 转为

$$
X_0[b,j,:]=E[\texttt{input\_ids}[b,j],:]
\in\mathbb R^d.
$$

padding token 本身不会自动“消失”；attention mask 必须使填充位置不参与不应参与的计算。左填充与右填充还会影响位置编号和最后有效位置的索引。执行记录不能只保存矩形 `input_ids`，还应保存每个样本的有效长度或 mask。

## 1.10 输入阶段的失败条件

| 失败 | 最早可观察位置 | 后果 |
|---|---|---|
| 非法 UTF-8 的替换策略不同 | 请求解码 | 后续字符串和 token 不同 |
| 模板别名指向新版本 | 渲染 | 控制 token 或换行改变 |
| schema 字段遍历顺序不稳定 | 渲染 | 工具描述 token 改变 |
| tokenizer 词表与模型不匹配 | tokenization/embedding | ID 语义错误，严重时越界 |
| 截断切断工具调用与结果 | 上下文预算 | 模型收到结构不完整历史 |
| padding mask 或 position IDs 错位 | 模型入口 | 相同 ID 产生不同前向结果 |
| 图片重编码或旋转未记录 | 模态预处理 | 视觉张量无法复现 |

这些错误都发生在模型推理解释之前。将它们误诊为“模型忘记了上下文”，会绕过最直接的证据。

## 1.11 最小复现包

输入阶段的最小复现包应包含：

1. 请求 schema 版本与规范化后的消息对象；
2. 动态注入字段，例如时区、当前时间、检索结果和工具目录；
3. 模板工件的不可变 ID 或内容摘要；
4. tokenizer 配置、词表、合并/Unigram 工件与特殊 token 表；
5. 渲染后的字节或 token 段；
6. 最终 `input_ids`、attention mask、position IDs 和有效长度；
7. 截断、摘要与多模态预处理决定；
8. 任何输入错误与替换策略。

复现实验先断言第 1.6 节夹具的 ID 序列，再对真实请求逐项比较上述对象。一旦最终输入张量一致，才进入[第二章的 prefill](ch02_prefill_forward_pass.md)；若这里已经不同，后续 logits 的差异无需诉诸更深的解释。
