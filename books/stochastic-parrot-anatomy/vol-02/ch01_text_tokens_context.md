# 第一章 文本、Token 与实际上下文

用户看到的是字形，程序接收的是字节，tokenizer 输出的是整数 ID。三者相关，却不能互换。

## 1.1 字节、码点与字形

UTF-8 把 Unicode 码点编码为一到四个字节。屏幕上的一个字形还可能由多个码点组合，例如字母和组合附加符；相反，不同码点序列也可能渲染得非常相似。

因此“文本相同”可能指：

- UTF-8 字节完全相同；
- Unicode 码点序列相同；
- 经过指定规范化后相同；
- 屏幕渲染看起来相同。

tokenizer 读取的是某种确定的字符串表示，不读取人类视觉上的“同一个字”。不可见空格、换行和规范化差异都可能改变 token。

## 1.2 Tokenizer 是模型工件的一部分

tokenizer 通常包含规范化规则、pre-tokenization、词表和合并或分词算法。它实现映射

$$
T:\text{string}\longrightarrow (i_1,\ldots,i_n),
$$

其中 $i_k$ 是词表 ID。逆映射把 ID 序列还原为字节或文本，但不保证每个任意 ID 序列都对应自然、规范化的字符串。

同一句中文在不同 tokenizer 下可能得到不同长度和边界。英文常见词也可能是一个 token，带前导空格的另一个 token，或多个子词。任何具体 token ID 都必须绑定 tokenizer 版本，不能凭肉眼猜测。

## 1.3 聊天模板先于 Tokenization

用户消息通常不会单独送入模型。应用先构造模板，例如：

```text
<system>你是一个简洁的科学助手。</system>
<user>请用一句话解释为什么天空通常呈蓝色。</user>
<assistant>
```

真实模板可能使用专用控制 token，而不是这些可见标签。角色、换行、工具说明、图片占位和 assistant 起始标记都会进入 token 序列。

因此“同一个 prompt”至少要说明可见用户文本和完整模型输入中的哪一个。两款应用即使显示相同输入，也可能因系统指令和模板不同而产生不同 token。

## 1.4 上下文组装

一次上下文可能按以下顺序组装：

```text
system/developer instructions
-> tool schemas
-> conversation history
-> retrieved documents
-> current user message
-> assistant generation prefix
```

顺序不是装饰。因果模型只能读取当前位置之前的 token；材料放在窗口中的不同位置，会改变注意力路径和截断风险。

## 1.5 截断

设窗口上限为 $N$，组装后输入长度为 $n>N$。系统必须选择删除、摘要或拒绝。常见策略包括从最早历史开始删、保留系统消息、压缩工具返回或限制单份文档长度。

截断发生在模型前。若相关证据已被应用删除，模型后续没有机会“想起来”。记录长上下文问题时应先检查实际 token 序列，而不是只检查聊天界面仍显示的内容。

## 1.6 多模态占位

图片和音频可以被专用编码器转换为表示，再以特殊 token 区段、cross-attention memory 或统一 token 进入模型。界面中的一张图片可能对应几百到几千个内部位置。

模型看到的不是 PNG 文件名，而是解码、缩放、切 patch 和投影后的张量。相同图片若使用不同分辨率或裁剪，也会产生不同模型输入。

## 1.7 Embedding 查表

token ID 序列 $i_{1:n}$ 通过 embedding 矩阵 $E\in\mathbb R^{|V|\times d}$ 转成

$$
X_0=
\begin{bmatrix}
E_{i_1}\\
\vdots\\
E_{i_n}
\end{bmatrix}
\in\mathbb R^{n\times d}.
$$

位置表示、模态信息或其他输入编码随后加入或作用于这些向量。到这里，可见文字已经变成模型宽度为 $d$ 的数值序列。

## 1.8 需要保存的最小对象

为了重现输入阶段，需要：原始字节或规范化文本、聊天模板、特殊 token 规则、tokenizer 工件、完整 token ID 序列、截断结果以及多模态预处理配置。

下一章从 $X_0$ 开始，跟随它穿过 Transformer 层并形成 prefill 缓存。
