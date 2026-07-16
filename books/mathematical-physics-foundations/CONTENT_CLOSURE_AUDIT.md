# 内容闭合审计

本文审计教材内容本体，不把排版、插图、出版流程或题量扩充列为闭合条件。这里的“闭合”是相对于 [00_preface_and_scope.md](00_preface_and_scope.md) 的先修合同、[SKILL.md](SKILL.md) 的 P/S/E 责任以及本书明示的外部输入而言；它不表示全书从集合论开始自足，也不表示量子场论已经获得非微扰构造。

## 审定口径

教材只有同时满足以下两层标准，才判为“可审定的内容收口”：

1. 机械层没有断链、重复标签、无解答练习、残留证明草图标题、章节结构错误或空白差异。
2. 内容层中，每个主线结论的证明责任都能沿 `P/S/E`、章节依赖和来源定位追溯，且正文没有越过已声明的数学或物理边界。

机械检查通过不能替代数学审读；数学叙述看似完整也不能替代标签、引用和练习映射检查。

## 机械闭合

本书以以下四组检查作为机械发布门槛：

| 检查 | 命令 | 通过条件 |
|---|---|---|
| 系列结构、链接、标签、工作例与练习解答映射 | `python3 books/validate_foundational_series.py mathematical-physics-foundations` | 退出码为 0，且无错误条目 |
| OET 证明责任 | `python3 books/audit_oet_rigor.py mathematical-physics-foundations --strict` | strict 模式退出码为 0 |
| 教材叙事结构 | `python3 books/audit_textbook_narrative.py mathematical-physics-foundations --strict` | strict 模式退出码为 0 |
| Markdown 差异空白 | `git diff --check -- books/mathematical-physics-foundations` | 退出码为 0，且无空白错误 |

系列检查还要求 00--10 每章至少有一个带编号工作例和一个带编号练习，全部章节练习均能在 [SOLUTIONS.md](SOLUTIONS.md) 回链；OET 检查负责识别未履行的证明边界；叙事检查负责识别练习前的重复收束节。命令必须在最终文件状态上重新执行，不能以较早版本的结果代替。

### 本轮执行记录

2026-07-16 在本文件、来源账本与定理索引的最终状态上执行上述命令：系列检查报告 22 个 Markdown 文件、160 个标签、25 个练习且无错误；OET strict 与叙事 strict 均报告 `errors=0, warnings=0`；`git diff --check` 退出码为 0 且无输出。因此机械闭合成立。

## 内容闭合

### 证明责任

| 状态 | 本书采用的闭合条件 | 核查位置 |
|---|---|---|
| `P` | 相对于序章先修合同，正文给出覆盖命题全部量词、假设和结论的书内证明 | [THEOREM_INDEX.md](THEOREM_INDEX.md) 与命题所在章 |
| `E` | 正文精确陈述调用版本、说明书内用途与停止点，并在来源账本给出作者、书名、版本或年份及章节、节、定理或页码定位 | [SOURCES.md](SOURCES.md) |
| `S` | 正文只作标准物理形式推导，同时保留 regulator、微扰阶数、算符域、能标或局部规范切片等适用边界 | [THEOREM_INDEX.md](THEOREM_INDEX.md) 与相应推导 |

`P` 不含依赖深理论的证明路线；这类内容登记为 `E`。连续场的形式路径积分不被解释为无穷维测度，有限维 Gaussian/Wick 定理与受 regulator 控制的形式展开在第 8 章分开陈述。

### 主线覆盖

| 范围 | 书内闭合的主干 | 明示外部或形式边界 | 判定 |
|---|---|---|---|
| 00--01 | 证明合同、流形张量、微分形式、有限维变分 | de Rham 同构单列为 `E-A.2` | 闭合 |
| 02 | Hamilton 向量场、Poisson 括号、Hamilton 流与 moment map 守恒 | Darboux 定理为 `E-2.5` | 闭合 |
| 03 | Lie 代数、表示微分与 Schur 引理 | Peter--Weyl 与 $SU(2)$ 群级分类为 `E-3.4`、`E-3.5` | 闭合 |
| 04 | 联络、曲率、规范变换、Bianchi 与 Chern--Weil 形式闭合性 | 特征类识别为 `E-4.5` | 闭合 |
| 05 | 无界算符的域、图、闭包、伴随边界以及 Schwartz--Fourier 基础 | 闭图、自伴扩张、PVM 谱定理、Stone、Fourier 反演与 Sobolev 嵌入为 `E` | 闭合 |
| 06 | 有界动力学、有限维角动量与自旋计算 | Stone--von Neumann、Wigner 为 `E`；WKB 为 `S` | 闭合 |
| 07 | 局部场变分、Noether、能动张量与 Klein--Gordon 配对 | 全局双曲 PDE 的 Green 算子为 `E-7.5` | 闭合 |
| 08 | 有限维 Gaussian、Wick、重整化条件及 EFT 截断计算 | BPHZ 为 `E-8.4`；路径积分、Callan--Symanzik、EFT 与 Faddeev--Popov 的物理推导为 `S` | 闭合 |
| 09 | 有限粒子域上的 Fock 算符、自由场 CCR 与自由真空 Wick 定理 | Wightman 重构为 `E-9.3`；一般相互作用场不由自由构造推出 | 闭合 |
| 10 | Yang--Mills 规范不变性与场方程 | BRST、Fujikawa 为 `S`；扭 Dirac 指标为 `E-10.5` | 闭合 |
| A--C | 线性代数、测度接口、同调/指标接口与公式约定 | de Rham、测度表示、Hodge、椭圆 Fredholm 理论均逐项登记为 `E` | 闭合 |

章节直接依赖见 [DEPENDENCY_GRAPH.md](DEPENDENCY_GRAPH.md)，符号约定见 [NOTATION.md](NOTATION.md)。每个主干章均保留带编号的完整工作例；全部带编号练习在 [SOLUTIONS.md](SOLUTIONS.md) 有对应解答。

### 可接受的外部边界

以下理论不在本书内部重建，但其被调用的精确版本已登记，因此不构成本书既定范围内的断口：

- 无界自伴算符的完整扩张理论、PVM 谱定理与强连续酉群生成元理论。
- Peter--Weyl、Darboux、de Rham、Hodge、Chern--Weil 特征类识别和 Atiyah--Singer 型指标理论。
- 全局双曲 Lorentz 流形上的完整能量估计、Cauchy 问题和 Green-hyperbolic 理论。
- 四维重整化的 BPH 森林收敛定理、Wightman 重构所需的核定理与分布论技术。
- 构造性量子场论、四维相互作用场的非微扰测度、Yang--Mills 质量间隙、全局 BRST/BV 量子化和 Gribov 问题。

这些边界限制了本书可推出的结论：例如，第 8 章只能推出逐阶微扰与指定 EFT 截断下的陈述，第 9 章的严格算符构造只覆盖自由 Fock 模型，第 10 章不据形式 Jacobian 推导非微扰量子规范理论的存在性。

## 真实残余风险

1. 规范场、BRST 与异常公式的符号和归一化依赖第 4、9、10 章固定的度规、Lie 代数与 Fourier 约定；跨书引用时必须先换算约定。
2. `E-8.4` 固定 massive Euclidean $\phi^4_4$、多尺度 cutoff 与逐阶 BPH 结论；正文的非例外动量减法条件通过有限局部方案变换匹配，不包含微扰级数收敛性。
3. 第 9 章的场算符等式成立在共同的有限粒子稠密域或算符值分布意义下；一般闭包、本质自伴性和相互作用域问题不在其结论中。
4. 书目定位绑定 [SOURCES.md](SOURCES.md) 所列版本；使用其他版本时，页码可能变化，应按所列章节或定理名称复核。

## 收口判定

内容层已经相对于冻结范围、先修合同及上述外部边界闭合；“机械闭合”节所列命令也已在当前文件状态全部通过。因此，本书达到可审定的教材内容收口状态。该判定不扩大正文结论，也不取消“真实残余风险”及外部输入边界。
