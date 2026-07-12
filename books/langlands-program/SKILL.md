---
name: langlands-program
description: Use when writing, extending, or reviewing the rigorous Chinese textbook on the Langlands Program in books/langlands-program. Requires formal theorem-proof exposition, explicit local/global field conventions, adelic and representation-theoretic notation, source traceability, and a strict separation between proved material and external input theorems such as class field theory, Tate's thesis, modularity, Ribet level lowering, and geometric Langlands.
---

# Langlands 纲领教材写作技能

本技能约束 `books/langlands-program/` 中《Langlands 纲领》教材的写作、扩写、校订和审稿。

## 基本原则

- 使用中文叙述；标准英文术语第一次出现时括注，例如“自守表示（automorphic representation）”。
- 每个概念先给形式定义，再给例子、反例或边界条件；不得用类比替代定义。
- 非平凡命题必须给出完整书内证明，或明确标注为“外部输入定理”并在 `SOURCES.md` 记录来源；只给出路线的旧条目统一归入外部输入，不计作已证。
- 不使用“显然”“容易看出”等词跳过关键步骤；若步骤短，应写出使用的定义、局部-整体分解、交换图或泛性质。
- 所有域、赋值、完备化、群、表示、特征、L 函数、范畴和函子必须标明所在环境。
- 不把 Langlands 纲领写成历史随笔或科普口号；正文默认读者愿意接受研究生层级的代数数论、调和分析、表示论和代数几何。

## 范围口径

- 本书以数论 Langlands 为主线：局部域、整体域、adeles、ideles、Tate thesis、类域论、`GL(1)`、`GL(2)`、`GL(n)`、自守表示、Galois 表示、L 群、函子性和相容 L 函数。
- 几何 Langlands 作为后半部分专题处理：曲线上的 `G`-bundles、Hecke 修改、D-模或 $\ell$-adic sheaves、Hecke eigensheaves、谱侧和几何 Satake。
- 范畴语言用于表达表示范畴、层范畴、函子性、Hecke 作用和几何 Langlands 的谱-自守对应；不得把普通 Langlands 纲领强行改写成空泛的“范畴哲学”。
- 费马大定理作为单独应用章处理：只证明“由半稳定椭圆曲线模性定理、Ribet 降层定理和 Frey 曲线性质推出费马大定理”的严格逻辑链；Wiles-Taylor-Wiles 机器本身必须标注为外部输入，除非另写完整专题卷。

## 外部输入分级和防发散规则

- 外部理论不得自动展开。每次引入外部定理前，先判定它在 Langlands 主线中的角色：核心结构、支撑接口、还是卫星理论。
- 核心结构可以在本书内展开到可计算层：直接定义 Langlands 对象、参数、L 因子、Hecke 作用、Satake 参数、局部-整体相容、函子性或几何 Langlands 基本范畴的材料。
- 支撑接口只给精确定理陈述、假设、归一化、使用位置和资料源：例如完整类域论证明、Tate thesis 完整解析证明、Neron 模型存在性、Tate algorithm 全流程、Taylor-Wiles patching、Arthur trace formula 稳定化、完整 derived algebraic geometry。
- 卫星理论不并入本书主体；只在 `MATH_REVIEW.md` 标成“另卷/专题理论”。例如一般代数几何基础、完整 p-adic Hodge theory、完整 Bruhat-Tits 理论、完整 D-module 六运算构造、完整 Fargues-Scholze 证明。
- 是否展开的判定问题：
  1. 不展开是否会导致某个 Langlands 对象无法定义？
  2. 不展开是否会导致局部因子、参数、导子、Hecke 本征值或归一化无法计算？
  3. 不展开是否会破坏某个应用章的逻辑链？
  4. 展开后是否仍服务于本书主线，而不是转入另一门理论的完整课程？
- 若答案只支持“背景理解”，则保留为外部输入或另卷；若答案支持“定义、计算或逻辑闭环必需”，才写入正文或附录。

## 资料源规则

- 优先使用正式教材、专著、作者讲义和原始论文：Tate, Weil, Langlands, Gelbart, Bump, Goldfeld-Hundley, Jacquet-Langlands, Godement-Jacquet, Arthur, Clozel, Milne, Neukirch, Serre, Silverman, Diamond-Shurman, Cornell-Silverman-Stevens, Harris-Taylor, Bushnell-Henniart, Frenkel, Gaitsgory 等。
- 涉及局部 Langlands、Arthur 参数、端oscopic transfer、trace formula、几何 Langlands 或近期进展时，必须核对一手论文、作者讲义或出版社页面。
- Wikipedia、博客和百科型页面只能用于发现线索，不得作为核心定义或定理的最终依据。
- 不复制资料原文；所有正文用本书自己的中文重写。

## 写作格式

- 文件名使用两位编号或应用编号，例如 `01_global_fields_and_adeles.md`、`90_fermat_last_theorem_application.md`。
- 每章开头列出“本章目标”和“依赖前置知识”。
- 定义、约定、例子、命题、引理、定理、推论、练习使用“**定义 1.2.**”格式。
- 证明结束使用 `$\square$`。
- 每章末尾必须包含“本章小结”和“练习”。
- 公式使用 Markdown/LaTeX；交换图可用矩阵、tikzcd 风格代码块或明确的等式条件描述。
- 术语和符号必须与 `NOTATION.md` 一致；新增全书性符号必须先更新 `NOTATION.md`。

## 严谨性检查

扩写或修改章节后逐项检查：

- 是否已经声明数域、函数域、局部域、赋值归一化和 Haar 测度归一化。
- 每个 restricted product 是否说明“几乎所有位置”的开紧子群或子环。
- 每个特征、表示和 L 函数是否说明局部/整体、复值/$\ell$-adic、连续性、光滑性、可容许性和中心特征等条件。
- 自守形式和自守表示是否区分函数模型、表示模型和 Hecke 本征数据。
- Galois 表示是否说明系数域、拓扑、半单化、ramification 条件和 Frobenius 归一化。
- L 群和 Langlands 参数是否说明 Weil 群、Weil-Deligne 群或 Galois 群版本。
- 函子性是否写成 L 群同态诱导的自守表示转移，并说明预期的局部和全局相容性。
- 外部输入定理是否在 `SOURCES.md` 中可追溯。
- 费马大定理应用章是否把 Frey 曲线、模性定理、Ribet 降层和 `S_2(\Gamma_0(2))=0` 的角色分开陈述。

## 本书口径

- 第一部分建立基础语言：整体域、局部域、adeles、ideles、Haar 测度、Fourier 分析、Tate thesis。
- 第二部分说明 `GL(1)` Langlands 的类域论参数对应，并用 Tate thesis 固定 L 函数解析接口。
- 第三部分进入 `GL(2)`：模形式、椭圆曲线、Galois 表示、局部因子和模性。
- 第四部分进入一般还原群：L 群、局部参数、全局自守表示、函子性、trace formula 和 endoscopy。
- 第五部分进入几何 Langlands：从 Hecke 算子到 Hecke eigensheaves 和谱-自守范畴。
- 应用章把费马大定理作为 `GL(2)/\mathbb Q` 模性思想的实例，而不是把完整 Langlands 纲领误称为费马大定理的直接证明。
