# 卷二 一次生成如何发生

卷一从历史与系统结构解释模型；本卷把镜头收紧到一次请求。我们沿着真实执行顺序观察文本怎样变成 token，Transformer 怎样完成 prefill，自回归解码怎样逐步产生文本，扩散与流模型怎样迭代整个状态，以及工具调用在哪里越过模型与外部世界的边界。

本卷只保留能澄清执行过程的数学。公式用于标明张量、条件分布和状态更新，不把一次普通生成包装成一套额外的形式逻辑。

0. [从输入到输出：一张执行地图](ch00_from_input_to_output.md)
1. [文本、Token 与实际上下文](ch01_text_tokens_context.md)
2. [Prefill：一次 Transformer 前向](ch02_prefill_forward_pass.md)
3. [从 Logits 到下一个 Token](ch03_logits_and_next_token.md)
4. [解码循环与流式输出](ch04_decode_loop_streaming.md)
5. [不止自回归：迭代生成的几种形态](ch05_iterative_generation.md)
6. [工具调用与运行时边界](ch06_tools_runtime_boundary.md)
7. [三条完整执行轨迹](ch07_end_to_end_traces.md)

辅助材料：[符号与术语](GLOSSARY.md) · [资料源](SOURCES.md)

读完本卷，读者应当能够区分模型计算、解码策略、服务调度、界面显示和外部工具执行，并能沿时间线定位两次输出第一次发生分歧的位置。下一卷将从执行中的概率数值出发，追问它们如何由数据、目标函数与条件信息形成。
