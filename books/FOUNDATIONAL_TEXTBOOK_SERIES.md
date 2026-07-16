# 基础教材系列登记

本文档登记本轮新增的非 AI 主题基础教材与方法论伴随卷。它只记录范围和收口标准，不替代各书自己的 `CONTENT_CLOSURE_AUDIT.md`，也不把文件齐全等同于内容闭合。

## 1. 本轮选书

本轮选择三本相互支撑、又能服务现有 Langlands、string theory、category theory 等教材的基础书：

1. [数学物理基础：从几何、表示论到量子场论](mathematical-physics-foundations/README.md)
2. [概率、随机过程与信息论：从测度到熵](probability-stochastic-information/README.md)
3. [计算理论、类型论与程序语义](computation-types-semantics/README.md)
4. [如何写一本可收口的严格教材](textbook-writing-methodology/README.md)（方法论伴随卷）

## 2. 共同收口标准

每本书的审定分为两层。机械闭合只是进入内容审定的必要条件：

1. 有 `README.md`、`SKILL.md`、`NOTATION.md`、`SOURCES.md`、`THEOREM_INDEX.md`、`DEPENDENCY_GRAPH.md`、`CONTENT_CLOSURE_AUDIT.md` 和 `SOLUTIONS.md`。
2. 本地 Markdown 链接、定理索引、习题解答、编号和数学围栏没有机械缺口。
3. 每个编号练习均能在解答集中定位；每个索引条目均能回到正文。

机械闭合通过后，只有同时满足以下条件才可判为内容本体收口：

1. 正文章节形成从前置概念到主线应用的有限依赖链，且没有把主题清单冒充章节展开。
2. 非平凡陈述必须处于下列终态之一：
   - `P`：正文给出覆盖全部假设与结论的完整证明；
   - `S`：物理推导、计算或可执行方案，明确适用条件，不作为已证数学定理；
   - `E`：精确陈述的外部输入定理，登记版本、用途、来源和未重证边界；
   - `C`：猜想、原则、模型假设或审美取舍，不进入已证明结论链。
3. “证明草图”不是终态。有限主线论证必须补全；深结果只能作为 `E`，证明路线仅用于解释输入。
4. 每章至少有一个实际操作定义的完整例子、计算、程序轨迹或失败案例；名称枚举不计作例子。
5. 习题覆盖定义使用、假设边界和跨节综合；解答给出关键推导，而不只给最终答案。
6. 外部大理论只作为接口引入，不在本书中无限展开，但书名承诺的核心结果不能全部外置。
7. 来源至少定位到作者、题名、版本或年份、相关章节或定理；模糊的“经典教材”或“相关论文”不构成来源闭合。
8. `CONTENT_CLOSURE_AUDIT.md` 分别报告机械检查与人工内容审定，并列出实际检查过的代表性证明、例子和边界。
9. 严格运行 `audit_oet_rigor.py`、`audit_textbook_narrative.py`、系列验证和 `git diff --check`；脚本通过不能替代人工数学审读。

## 3. 范围边界

本轮不做站点、导航、HTML anchor 或出版排版。所有成果保持为 Markdown 教材内容本体。
