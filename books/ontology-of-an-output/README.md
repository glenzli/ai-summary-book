# 一次输出的存在论

**状态/定位：已收口的本体论/程序语义教材；贯穿案例与配套责任表的当前入口见 [STATUS.md](STATUS.md)。**

## 从字节、token、轨迹到一句话

**作者：Dr. Stochastic Parrot**

屏幕上出现一句话时，人们会自然地说“模型输出了它”。严格说，这句话可能同时涉及 UTF-8 字节、Unicode 标量、token 序列、候选与已提交事件、数据库制品、来源图、语境中的命题，以及由制度赋予的署名和责任。它们通过转换和解释相连，却不属于同一类型。

本书以程序语义、概率核、分布式事件和形式语义为骨架，追踪一次输出从生成到提交、从制品到真值、再到规范归属的全过程。全书反复回到同一次虚构运行：用户查询航班 `SP404`，系统在确认取消后写入 `trip.md`，经历一次确认丢失、幂等重试和乱序流式交付，最终提交“SP404 已取消；已写入 trip.md。”。读者可以沿这条生命周期观察同一材料怎样依次成为字节、token、轨迹、世界事件、来源制品、事实主张和规范归属。

数学结论给出量词和证明；标准结果标为外部输入；经验、哲学与规范判断分别声明证据和桥接前提。严格性保留在对象、命题与证明中，而不是用重复的章首清单代替正文推进。

## 目录

1. [序章：为什么一句话不是一个对象](00_preface_and_scope.md)
2. [字节、字符串、Unicode、字形与 token](01_bytes_text_and_tokens.md)
3. [函数、部分函数、关系与交互接口](02_functions_relations_and_partiality.md)
4. [状态、事件、轨迹与观察等价](03_states_events_and_traces.md)
5. [自回归生成的小步语义](04_operational_semantics_of_generation.md)
6. [概率核、实现映射与轨迹分布](05_probabilistic_generation.md)
7. [工具、外部世界与副作用提交](06_tools_and_external_world.md)
8. [并发、流式输出、取消与提交边界](07_concurrency_and_streaming.md)
9. [制品身份、provenance 与可验证声明](08_provenance_and_identity.md)
10. [表达式、指称、真值与核验状态](09_reference_and_truth.md)
11. [代理、署名、信用与责任的分层论证](10_agency_authorship_and_responsibility.md)
12. [一次输出的完整分解](11_complete_decomposition.md)

配套材料：[状态说明](STATUS.md)、[符号与术语](GLOSSARY.md)、[外部输入与资料源](SOURCES.md)、[定理与主张责任表](CLAIM_LEDGER.md)、[习题解答](SOLUTIONS.md)、[闭合审计](CLOSURE_AUDIT.md)。

本书已经完成正文、证明链与贯穿案例的教材化闭合；它不声称已经经过独立同行评议，也不提供具体法律意见。
