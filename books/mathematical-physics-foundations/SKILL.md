---
name: mathematical-physics-foundations
description: Use when writing, extending, or reviewing the rigorous Chinese Markdown textbook in books/mathematical-physics-foundations. Requires theorem-definition-proof style, explicit status labels P/S/E, source traceability, notation consistency, and a closed mathematical-physics narrative from geometry and representation theory to quantum field theory.
---

# 数学物理基础教材写作技能

本文件约束 `books/mathematical-physics-foundations/` 中《数学物理基础：从几何、表示论到量子场论》的写作、扩写、审校和解答维护。

## 基本原则

- 正文使用中文。标准英文术语第一次出现时括注，例如“辛形式（symplectic form）”。
- 采用“问题-定义-例子-命题-证明或边界说明”的写法。重要对象不得只凭物理直觉引入。
- 非平凡结论必须使用状态标记：
  - `P`：正文给出覆盖陈述全部结论的完整证明。证明可使用序章声明的先修知识和已经建立的 `P` 结果，但不得把未证明的深结果藏在“标准论证”中。
  - `S`：标准物理形式推导。凡依赖形式路径积分、正规化、微扰渐近展开、未构造的无穷维行列式或规范轨道换元者均属此类，并须注明 cutoff、能区和截断口径。
  - `E`：外部输入定理。正文须给出本书实际使用的精确版本、全部必要假设、用途以及可追溯资料定位。
- “证明草图”不是终态。有限且属于主线的论证必须补全；依赖本书范围外理论的解释只能写成“证明路线（外部输入）”，并把陈述登记为 `E`。
- `P` 的证明结束使用 `\(\square\)`；`S` 的推导结束必须保留其物理适用边界；`E` 的证明路线不得写“证毕”。
- 外部输入不得伪装成书内定理。Sobolev 嵌入、Stone 定理、谱定理、Peter-Weyl、Atiyah-Singer、Wightman 重构、BRST/FP 量子化完整性等均须标为 `E` 或 `S`。

## 主题边界

- 主体覆盖：微分流形、张量、辛几何、Hamilton 系统、Lie 群与表示、纤维丛与联络、泛函分析、量子力学、经典场论、路径积分接口、Wightman/Fock 框架、规范场和异常。
- 不并入主体：完整代数拓扑、完整椭圆算子理论、构造性量子场论、非微扰 Yang-Mills 存在性、完整 BV-BRST 形式主义、广义相对论全书化处理、弦论全书化处理。
- 物理推导可以服务主线，但必须说明依赖的正规化、截止、微扰阶数和可交换极限问题。

## 符号和归一化

- 实流形默认有限维、Hausdorff、第二可数、光滑。
- Riemann 度量记为 $g$，辛形式记为 $\omega$，Poisson 括号记为 $\{f,h\}$。
- Lie 代数用 fraktur 字体，如 $\mathfrak g$；其表示记为 $(\rho,V)$。
- Hilbert 空间内积默认对第二个变量线性：$\langle \psi,\phi\rangle$。
- Minkowski 度量默认 mostly plus：$\eta=\operatorname{diag}(-,+,\ldots,+)$。
- 自然单位默认 $\hbar=c=1$；若恢复量纲，必须在局部说明。

## 文件格式

- 章节文件使用两位编号，例如 `03_lie_groups_lie_algebras_and_representations.md`。
- 每章 H1 后写自然导言；不得只列模板化目标。
- 定义、约定、命题、定理、推论、例子、练习使用 `**定义 3.1.**` 格式。
- 完整证明结束使用 `$\square$`；外部输入的证明路线不使用证毕符号。
- 每章至少包含若干可解练习；代表性练习在 [SOLUTIONS.md](SOLUTIONS.md) 中给出解答。
- 新增全书符号必须更新 [NOTATION.md](NOTATION.md)。
- 新增主要命题必须更新 [THEOREM_INDEX.md](THEOREM_INDEX.md)。
- 新增外部输入必须更新 [SOURCES.md](SOURCES.md) 和 [CONTENT_CLOSURE_AUDIT.md](CONTENT_CLOSURE_AUDIT.md)。

## 审查清单

编辑后检查：

- 是否在目标目录内完成修改。
- 是否声明数学对象的类别、拓扑、光滑性、定义域和边界条件。
- 是否区分有限维证明、无穷维形式推导和外部分析定理。
- 是否把群表示、守恒量、谱分解和量子数之间的关系写成可检验命题。
- 是否把路径积分、Faddeev-Popov、异常和重整化标为 `S` 或 `E`，并说明其限制。
- 是否保持章节依赖与 [DEPENDENCY_GRAPH.md](DEPENDENCY_GRAPH.md) 一致。
