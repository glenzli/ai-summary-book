# 卷二 一次生成如何发生

卷一从历史、训练和系统结构解释模型；本卷把镜头收紧到一次具体请求。它沿真实依赖顺序追踪：请求怎样被序列化，文本怎样成为 token，prefill 怎样逐层建立 KV cache，logit 处理器怎样形成实际候选，decode 与 streaming 怎样终止，以及工具调用何时真正越过外部提交边界。

本卷的中心对象不是“模型能力”，而是**执行状态**。每章都按同一方法组织：

1. 定义输入、状态所有者和输出；
2. 给出可以实现的伪代码或状态转移；
3. 声明必须保持的不变量；
4. 列出最早失败边界；
5. 给出能够手算或重放的固定夹具。

数学只用于区分张量、条件前缀和更新核。它不承担装饰功能，也不把工程事件提升成不必要的形式逻辑。

## 本卷边界

本卷回答“一次生成在机器中怎样发生”，不重复三类问题：

- Transformer 为什么采用某种设计、模型怎样训练和服务优化：见[卷一](../vol-01/README.md)；
- token 概率从数据和目标函数中怎样形成、可以解释什么：见[卷三](../vol-03/README.md)；
- 隐藏状态、神经元、feature 与因果干预怎样研究：见[卷四](../vol-04/README.md)。

工具章节只讲 schema、权限、幂等、重试和提交状态如何参与一次执行，不扩展为一般 Agent 工程、治理或责任框架。

## 章节

0. [从输入到输出：一张执行地图](ch00_from_input_to_output.md)
1. [文本、Token 与实际上下文](ch01_text_tokens_context.md)
2. [Prefill：整层状态演化与 KV 建立](ch02_prefill_forward_pass.md)
3. [从 Logits 到下一个 Token](ch03_logits_and_next_token.md)
4. [解码循环、Streaming 与终止状态机](ch04_decode_loop_streaming.md)
5. [自回归、扩散与 Flow 的可执行差异](ch05_iterative_generation.md)
6. [工具调用：Schema、权限与提交边界](ch06_tools_runtime_boundary.md)
7. [三条完整执行轨迹](ch07_end_to_end_traces.md)

辅助材料：[符号与术语](GLOSSARY.md) · [资料源](SOURCES.md)

## 统一步序

全卷采用一个明确约定：prefill 后 cache 只含提示并已有首步 logits；第 $t$ 次选择之前，cache 含提示和前 $t-1$ 个输出；若新 token 不触发终止，下一次 decode 才把它写入 cache 并产生再下一步 logits。这个约定避免“token 已选择”“token 已进入模型”和“token 已显示”之间的 off-by-one 混乱。

## 阅读结果

读完本卷，读者应能够：

- 从请求对象恢复渲染字节、token IDs、mask 与 position IDs；
- 为 prefill 和单 token decode 写出逐层张量形状与 KV 长度；
- 重放一条有序 logit processor 管线并定位首次候选分歧；
- 区分 token selected、cache updated、bytes emitted 与 client rendered；
- 为 EOS、停止串、长度、取消、背压和失败画出合法终态；
- 写出自回归、连续扩散、离散扩散和 flow 各自的 state/update/terminal；
- 判断工具调用处于 proposed、authorized、committed 还是 outcome unknown；
- 用三条端到端轨迹的事件和不变量检查实现。

如果这些问题仍只能用“模型生成了结果”概括，读者就尚未达到本卷目标；如果每一步的输入、决定者、状态与失败边界都能被指出，读者便已掌握本卷要求的执行语义。
