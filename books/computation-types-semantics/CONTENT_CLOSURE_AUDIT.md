# 内容闭合审计

审计日期：2026-07-16

审计范围：books/computation-types-semantics/

目标等级：有限范围内可独立审定的严格 MD 教材内容本体，不声称已经过 proof assistant 机器核验或出版审稿。

## 机械闭合

机械闭合只回答文件、链接、标签和映射是否一致，不替代数学审定。

1. 目录包含 README、局部 SKILL、符号表、来源账本、定理索引、依赖图、00--10 章、A--C 附录、逐题解答和本审计，共 22 个 Markdown 文件。
2. validate_foundational_series.py computation-types-semantics 检查本地链接、定义/定理标签、每章例子、每章练习以及练习到解答的一一映射。
3. audit_oet_rigor.py computation-types-semantics --strict 检查证明状态、来源边界和形式标记。
4. audit_textbook_narrative.py computation-types-semantics --strict 检查章节导言、重复模板和提纲化段落。
5. git diff --check 检查冲突标记、尾随空白和补丁格式。
6. 终态正文与账本不得含未完成标记、占位段落或把提纲性论证当作证明的措辞。

上述命令的最终计数与退出状态在系列终检时重新生成；本文件不以手写计数代替验证器输出。

## 内容闭合

### 主线闭合

本书冻结的证明主线为：
$$
\text{有限程序语法}
\longrightarrow
\text{可接受编号与不可判定性}
\longrightarrow
\text{lambda 归约与类型安全}
\longrightarrow
\text{操作、指称和公理语义}
\longrightarrow
\text{表达力与自动化边界}.
$$
每个箭头都有 [DEPENDENCY_GRAPH.md](DEPENDENCY_GRAPH.md) 登记的内部定理或精确外部输入。正文没有把“有类型”“终止”“指称相等”“Hoare 可证”互相偷换。

### 证明责任闭合

- [THEOREM_INDEX.md](THEOREM_INDEX.md) 登记全部 T 标签与 EI 标签。P 条目在正文给出完整证明并覆盖相应规则或构造子的全部情形。
- EI-1 至 EI-9 的对象系统、假设、结论、版本和章节或定理定位均在 [SOURCES.md](SOURCES.md) 固定。
- 外部输入之后的证明路线只解释来源方法，不进入书内证明链。
- STLC preservation/progress、MLTT 弱化与替换、System F preservation、CEK 对应、while 指称一致性和 Hoare soundness 均为书内证明；强正规化、参数性、PCF 完全抽象和 Cook 相对完备性没有被伪装成内部结果。

### 定义、例子与练习闭合

- 机器配置、程序编码、捕获避免替换、类型上下文、判断等价、求值上下文、抽象机状态、omega-cpo、偏状态函数和 Hoare 语义均在首次进入证明前定义。
- 每个编号正文章至少有一个执行轨迹、推导树、反例或有限计算，而非只有术语说明。
- 所有编号练习均在 [SOLUTIONS.md](SOLUTIONS.md) 有同号解答；解答实际回应当前题目，不沿用旧章节版本的错位答案。
- 练习不承担后续正文必需的未证结论。

## 分章实质审定

| 章节 | 内部闭合结果 | 外部边界 |
| --- | --- | --- |
| 00 | 对象语言、元语言、计算和指称层级分离 | 无 |
| 01 | 计数器机一步函数、编码和复合封闭 | EI-1 通用解释器与模型等价 |
| 02 | 停机不可判定、归约传递、Rice 定理 | EI-1、EI-2 |
| 03 | alpha 商上的替换、beta 正规形唯一性 | EI-3 Church--Rosser |
| 04 | STLC weakening、替换、preservation、progress、类型安全 | EI-4 强正规化 |
| 05 | 固定 MLTT 核心、弱化、依赖替换、regularity、规则翻译 | EI-5 弱头正规化、Nat canonicity |
| 06 | System F preservation、同构递归接口、Kleisli 合成 | EI-6 关系参数性 |
| 07 | while 大小步等价、唯一分解、CEK 双向对应 | EI-7 通用 SOS 格式定理 |
| 08 | 最小不动点、偏函数域、while 操作/指称双向一致 | EI-8 的 PCF 结果只作接口 |
| 09 | Hoare soundness、循环不变式、WLP | EI-9 相对完备性 |
| 10 | 三类可靠性分离、完全抽象方向、Rice 自动化边界 | EI-8 |
| A--C | 集合、归纳、共归纳、推导规则的局部工具 | 不新增外部定理 |

## 明确不纳入的方向

复杂度理论、并发和概率语义、线性与仿射类型、同伦类型论、编译器验证、模型检查算法、分离逻辑完整体系及 proof assistant 形式化均不在冻结主线内。它们是后续专门教材的范围，不构成本书当前主线的内容缺口。

## 残余风险

1. EI-5 对固定 MLTT 规则集的专门化是本卷最敏感的来源接口；正文和来源账本已逐项列出宇宙、eta、大消去及删除规则的边界，并主动排除了来源不能直接支持的无 eta 判等结论，但弱头正规化与 canonicity 的最终可靠性仍依赖所引外部元理论。
2. 所有 P 证明经过文本审定而非机器检查；变量新鲜性、同时替换和环境卸载是未来形式化时最值得优先核验的部分。
3. 外部输入的正确性不由本书重证。若更换版本或对象系统，必须重新审计相应 EI，而不能只改书目字符串。

这些风险已经显式隔离，不造成当前有限主线中的悬空定义、未标证明责任或未映射练习。

## 最终判定

在上述冻结范围与 P/E 证明责任下，本目录达到“可审定的教材内容本体收口”状态。该判定不等同于出版级同行评审，也不授权把范围外的大型结果当作本书已经证明。
