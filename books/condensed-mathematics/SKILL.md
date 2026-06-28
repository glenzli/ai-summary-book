---
name: condensed-mathematics-textbook
description: Use when writing, revising, or checking the Chinese rigorous textbook on condensed mathematics in this repository. Requires primary-source grounding, formal mathematical definitions, theorem-proof structure, source traceability, notation consistency, and avoidance of informal survey prose.
---

# 凝聚数学教材写作技能

本技能约束 `books/condensed-mathematics/` 中教材的写作、扩写、校订和审稿。

## 基本原则

- 以中文叙述，但保留标准英文术语的第一次括注，例如“凝聚集合（condensed set）”。
- 每章必须先给定义，再给命题、证明、例子和练习；不得用类比替代定义。
- 每个非平凡命题必须有证明、证明草图，或明确标注“暂不证明”并说明后续依赖风险。
- 不写“显然”“容易看出”来跳过关键数学步骤；若确实简单，也要写出使用的定义。
- 不把凝聚数学写成科普综述；正文默认读者能接受范畴、拓扑、同调代数的精确定义。
- 不复制讲义原文；所有内容用本书自己的中文重写，并在资料表中记录来源。

## 资料源规则

- 优先使用一手或正式数学资料：Scholze/Clausen 讲义、arXiv 论文、作者主页讲义、正式教材。
- 涉及定义、定理或历史版本时，必须在 `SOURCES.md` 中能追溯到具体来源。
- 涉及近期版本、讲义更新、论文状态时必须联网核查；不要依赖模型记忆。
- 不以 Wikipedia、博客或二手科普作为核心定义来源；可用于发现线索，但不得作为主依据。

## 写作格式

- 文件名使用两位编号，例如 `01_sites_and_sheaves.md`。
- 每章开头列出“本章目标”和“依赖前置知识”。
- 定义使用“**定义 1.2.**”格式；命题、定理、引理、例子、练习同理。
- 公式使用 Markdown/LaTeX，尽量写出对象、态射、函子、自然变换所在的范畴。
- 每章末尾必须有“本章小结”和“练习”。

## 数学严谨性检查

扩写或修改章节后，逐项检查：

- 术语是否已定义。
- 范畴、函子、极限、余极限所在环境是否明确。
- sheaf 条件是否写成等化子、匹配族或相应范畴版本。
- 覆盖族是否说明有限性、联合满射性以及拉回稳定性。
- set-theoretic universe 或小性问题是否被回避得当。
- 例子是否满足前面定义，而不是只满足直觉。
- 引用来源是否已加入 `SOURCES.md`。

## 本书口径

- 第一卷以凝聚集合与凝聚阿贝尔群的基础为主，不急于进入解析凝聚数学或几何应用。
- 采用固定 Grothendieck universe 的方式处理大小问题；正文只在必要时提醒。
- 先使用紧 Hausdorff 空间站点定义凝聚集合，再说明 profinite / extremally disconnected 的等价或简化视角。
- 对高阶结果如 solid modules、analytic rings、six functors，只在后续卷或附录中展开。
