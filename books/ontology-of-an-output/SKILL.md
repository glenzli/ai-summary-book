---
name: ontology-of-an-output
description: Use when writing or reviewing books/ontology-of-an-output. Enforces typed distinctions among bytes, Unicode scalars, token sequences, states, traces, kernels, artifacts, semantic claims, and normative attributions; measurable probabilistic interfaces; complete operational proofs; and explicit external-world, concurrency, provenance, truth, and responsibility boundaries.
---

# 《一次输出的存在论》写作规范

本书同时服从上级 [OET 本体严格性标准](../OET_RIGOR_STANDARD.md) 与
[人类可读教材叙事标准](../TEXTBOOK_NARRATIVE_STANDARD.md)。下列规则是本书的附加约束。

## 生命周期叙事

- 全书沿同一次虚构运行推进：查询航班 `SP404`，确认取消后写入 `trip.md`，经幂等重试和乱序流式交付，最终提交“SP404 已取消；已写入 trip.md。”。
- 每章章首从前一层无法回答的问题进入下一层，不使用固定的“本章目标”“依赖”“主线”或“本章小结”标题。
- 每章至少给出一个可逐步跟随的生命周期片段、状态轨迹、计算或反例；案例中的符号必须与正文定义同型。
- 章末用自然段说明本层已经建立的接口和下一层仍缺的对象，不用审计清单复述整章。
- 严格边界的完整枚举放在 [CLAIM_LEDGER.md](CLAIM_LEDGER.md) 与
  [CLOSURE_AUDIT.md](CLOSURE_AUDIT.md)；正文只保留当前推导真正需要的桥接假设和反例。
- 新增章节或大幅改写时，必须让 `SP404` 生命周期仍可从序章连续追踪到第十一章；不得为每章换一套互不相干的示例。

## 对象纪律

- “string”必须声明为字节、编码单元、Unicode 标量或其他有限序列。
- UTF-8 解码只在合法域上为函数；宽容解码不得冒充逆映射。
- tokenizer 必须固定完整制品、`AdmIn`、`AdmTok`、特殊 token 规则及并列选择；`Enc` 与 `Dec` 的复合必须先做类型检查。
- 函数、部分函数、关系、随机核和实现映射使用不同记号。
- 状态机结论必须列出状态、标签、终止、卡死、错误、发散与最大轨迹。
- observational equivalence 必须写出轨迹范围、观察函数、上下文类和 may/must 或概率量词。
- 概率观察必须声明可测空间、核、观察映射可测性及推前事件。
- “输出”必须说明是候选事件、已提交 token、Unicode 文本、序列化字节、呈现历史还是制品。
- 工具请求、授权、执行、commit、响应与 ingest 分开；外部世界和调度不得藏在 prompt 中。
- 内容身份、哈希身份、运行身份和来源图身份分别定义。
- 真值、证据和核验状态不得合并。
- 工程 Agent、规范主体、署名、信用与责任分别定义。

## 证明责任

- 书内形式命题给出完整假设、量词和证明；练习不承担主线缺口。
- Unicode、UTF-8、JSON、Ionescu--Tulcea、随机化引理、Lamport 顺序和 W3C PROV 以 [SOURCES.md](SOURCES.md) 的精确接口为外部输入。
- 外部输入只承担正文明确列出的结论，不从标准名称推导未声明性质。
- 哲学和规范论证必须拆成描述性前提、桥接原则、规范前提与条件结论。
- 工程建议声明威胁模型、观察边界或协议假设；法律边界不写成定理。

## 一致性检查

- 新符号同步进入 [GLOSSARY.md](GLOSSARY.md)。
- 新定理、命题、外部输入和条件论证同步进入 [CLAIM_LEDGER.md](CLAIM_LEDGER.md)。
- 每道练习在 [SOLUTIONS.md](SOLUTIONS.md) 中有同编号闭合解答。
- 章节状态同步进入 [CLOSURE_AUDIT.md](CLOSURE_AUDIT.md)。
- 验证器必须拒绝旧模板标题，并检查每章有自然导言、贯穿案例、练习、声明、署名、链接和 LaTeX 基本闭合。
- 修改后运行 `python3 validate.py`、本书严格 OET 审计和 `git diff --check`。
