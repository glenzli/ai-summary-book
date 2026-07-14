---
name: probability-boundaries
description: Use when writing, revising, or checking the rigorous Chinese textbook books/probability-boundaries. Enforces measure-theoretic probability, explicit deterministic/random interfaces, complete internal proofs, precise external inputs, source traceability, and separation of mathematical randomness from epistemic uncertainty and psychological metaphor.
---

# 《概率的边界》写作规范

本文件约束 `books/probability-boundaries/` 的写作、扩写和校订。全书同时服从上级
[OET 本体严格性标准](../OET_RIGOR_STANDARD.md) 与
[人类可读教材叙事标准](../TEXTBOOK_NARRATIVE_STANDARD.md)。前者规定对象、定理与
证明责任，后者规定这些内容如何进入连续可读的教材正文。

## 1. 对象与类型

- 概率论对象必须先声明概率空间 $(\Omega,\mathcal F,\mathbb P)$；随机变量必须给出可测空间、定义域和值域。
- “分布”指推前测度；“密度”只在已经指定参考测度并证明绝对连续后使用。
- “独立”分别区分事件、随机变量、$\sigma$-代数与随机过程。
- 条件概率与条件期望首先定义为关于子 $\sigma$-代数的对象；只在正则条件分布存在时使用逐点核。
- 算法必须区分确定性状态转移、随机种子、伪随机数生成器、采样规则和外部环境。
- 语言模型案例必须区分 logits、条件分布、解码器、随机源、浮点实现与观察者信息。

## 2. 主张终态

每个非平凡主张只能以以下状态之一结束：

1. **书内定理**：给出覆盖全部结论的证明；
2. **外部输入定理**：准确陈述使用版本、假设、用途和来源；
3. **推导或计算**：列出起点、变形与适用条件；
4. **模型假设、经验结论或解释边界**：明确不把它写成数学定理。

不得用“概率上”“大概率”“平均而言”代替具体的概率空间、收敛方式或期望对象。不得把训练目标中的概率模型直接等同于一次执行中的随机性。

## 3. 证明与外部输入

- 测度扩张、Radon--Nikodym、Fubini--Tonelli、Ionescu--Tulcea、中心极限定理等大型结果可作外部输入，但必须在 [SOURCES.md](SOURCES.md) 中定位。
- 书内主线包括推前分布、独立性的基本等价、条件期望的基本性质、Markov/Chebyshev/Jensen 不等式、弱大数律、Gibbs 不等式、对数损失严格适当性、校准分解的有限版本，以及固定随机带下算法的确定性。
- “证明思路”“证明草图”和把主线结论留作练习均不算闭合。
- 边界情形必须处理：零概率条件事件、$0\log 0$、无限期望、不存在密度、不唯一的正则条件分布版本、温度为零以及并列最大值。

## 4. 证据语言

- 数学定理用“定理/命题/引理/推论”。
- 实验观察用“经验结果”，并说明数据、测量和外推边界。
- 建模选择用“模型假设”。
- 哲学判断用“论证”，列出前提与结论，不冒充演绎定理。
- “意图”“期待”“犹豫”“自由选择”等心理词只可作为被分析的归因，不可从概率分布直接推出。

## 5. 教材叙事

- 章节文件使用两位编号。
- 不使用逐章固定的“本章目标”“依赖”“主线”“本章小结”栏目。预备知识与阅读路线
  集中写入 README 和序章；章末用自然段收束并引向下一章。
- 章首从一个可计算例子、反例、现象或前章留下的问题进入，不以写作过程说明开场。
- 定义出现前先说明旧语言缺少什么；定理前说明它解决的问题，证明后解释关键机制与
  承担责任的假设。
- 每章至少保留一个可逐步跟随的完整例子、计算或反例。例子必须实际使用本章定义，
  不能只列名称或一句场景。
- 边界说明只在具体误读风险处出现；完整证明状态与范围责任留在账本和闭合审计中。
- 定义、命题和练习使用稳定编号，例如 `定义 3.2`、`练习 3.4`。
- 每道练习在 [SOLUTIONS.md](SOLUTIONS.md) 中有答案或完整解题要点。
- 全书性符号进入 [NOTATION.md](NOTATION.md)，外部结果进入 [THEOREM_LEDGER.md](THEOREM_LEDGER.md)。

## 6. 完成检查

修改后运行：

```bash
python3 books/audit_oet_rigor.py probability-boundaries --strict
python3 books/audit_textbook_narrative.py probability-boundaries --strict
python3 books/probability-boundaries/validate.py
git diff --check
```

结构与叙事审计都不能代替数学审稿。所有通过审计的定理仍需逐项检查假设、量词和边界；
所有通过字符计数的章首也仍需人工判断是否真正把读者带入问题。
