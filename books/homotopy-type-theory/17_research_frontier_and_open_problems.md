# 第十七章：研究边界、开放问题与版本化阅读

## 本章目标

本章总结 HoTT 与单值基础的研究边界，给出后续阅读和扩写的版本化规则。它不是新闻综述，而是教材维护清单：哪些方向成熟，哪些方向需要逐篇核查，哪些方向不能被写入基础层。

## 依赖前置知识

本章依赖全书概念和 `SOURCES.md` 的引用纪律。

## 17.1 已相对稳定的核心

**列表 17.1.** 以下内容可作为 HoTT 教材核心层：

- intensional dependent type theory；
- identity types and path induction；
- equivalences and univalence；
- basic homotopy levels；
- truncations and quotients, when HITs are accepted；
- circle and encode-decode proof of $\pi_1(S^1)$；
- univalent categories and structure identity principle；
- formalization discipline in Coq-HoTT、UniMath、Cubical Agda。

这些内容仍需检查具体定义口径，但作为教材主线已经成熟。

## 17.2 活跃研究方向

**列表 17.2.** 以下方向属于活跃研究或工程发展：

- cubical type theory 的不同模型、规范化和实现；
- 大规模合成同伦论形式化；
- 合成上同调和谱；
- higher category theory in univalent foundations；
- synthetic algebraic geometry and cohesive/modal HoTT；
- 与 proof assistant 的 universe、automation、library engineering 相关问题。

**规则 17.3.** 活跃方向写入正文时必须标注“研究边界”，并记录来源、日期和基础口径。

## 17.3 版本化阅读

**定义 17.4.** 一个资料条目是版本化的，若它至少记录：

1.  标题、作者和链接；
2.  访问日期或版本号；
3.  若是代码库，记录 release、commit 或模块路径；
4.  其被本书使用的具体位置；
5.  是否为外部输入、机器形式化或研究边界。

**原则 17.5.** 任何“最新”断言在六个月后视为需要重新核查。若涉及软件 release 或库结构，重新核查周期应更短。

## 17.4 后续扩写路线

**路线 17.6.** 若继续把本书推进到出版级，应按以下顺序扩写：

1.  完整形式化第 1-5 章路径代数和等价比较；
2.  为第 6-8 章加入单值性、截断和商类型的精细证明；
3.  把第 11 章 $\pi_1(S^1)$ 拆成整数、覆盖、encode、decode、群同构五个章节；
4.  为第 13-14 章加入 UniMath 风格的单值范畴论证明；
5.  为第 15-16 章加入实际库路径和最小可检查代码片段；
6.  把第 12 和第 17 章保持为版本化研究综述，不混入基础证明链。

## 本章小结

完整教材不是一次性罗列“所有结果”，而是维护一条可审查的证明链。本书当前版本给出完整教材第一版；后续工作的核心是把证明说明逐步降级为书内证明或精确外部输入。

## 练习

**练习 17.1.** 选择一个近期 HoTT 论文，按定义 17.4 建立版本化条目。

**练习 17.2.** 解释为什么活跃研究不应作为第一章的基础规则。

**练习 17.3.** 为第十一章圆的基本群证明设计一个五文件扩写计划。

**练习 17.4.** 检查本书一个“证明说明”定理，列出把它改为书内证明所需的前置引理。
