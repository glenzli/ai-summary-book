# 第十五章：形式化库比较

## 本章目标

本章比较 Coq-HoTT、UniMath、Cubical Agda 和 1Lab 的基础口径、适用范围和引用纪律。目标是让本书后续能把“已形式化”说清楚，而不是把不同库的结论混为一谈。

## 依赖前置知识

本章依赖全书前面的基础概念，但不引入新数学公理。涉及软件状态的断言需按 `SOURCES.md` 的核查日期重新确认。

## 15.1 Coq-HoTT

**事实 15.1.** Coq-HoTT 是在 Coq 中发展 HoTT 的库。它使用适配 HoTT 的 Coq 设置，包含路径代数、等价、同伦层级、范畴论和若干 HIT 的公理化接口。

**使用纪律。** 引用 Coq-HoTT 时必须记录：

- Coq 版本和库 release 或 commit；
- 是否使用 `HoTT` 库特定的 tactic 和 universe 设置；
- 结果是否依赖 univalence、funext、HIT 公理或 resizing。

## 15.2 UniMath

**事实 15.2.** UniMath 是基于 Coq 的单值数学库，强调在单值基础中大规模形式化普通数学，尤其是范畴论、代数、同伦层级和结构等同性。

**使用纪律。** 引用 UniMath 时必须记录：

- 定理所在模块；
- 是否采用 UniMath 的基础约定；
- 相关定义是否与本书定义 judgmentally 或 propositionally 一致；
- 命题截断、集合层和 universe 的处理方式。

## 15.3 Cubical Agda

**事实 15.3.** Cubical Agda 在语言层支持 cubical primitives，使 univalence 和许多 HIT 具有计算性表达。`agda/cubical` 是其公开实验库。

**使用纪律。** 引用 Cubical Agda 时必须记录：

- Agda 版本；
- Cubical 选项；
- 库 commit；
- 结果是否使用 cubical primitives、HIT、rewriting 或实验特性；
- 与公理化 HoTT 口径的差异。

## 15.4 1Lab

**事实 15.4.** 1Lab 是基于 Agda 的可浏览形式化数学参考，覆盖单值基础、范畴论、高阶范畴和相关结构。

**使用纪律。** 1Lab 可作为形式化 exposition 和定义网络参考，但正文核心定理仍应优先引用论文、HoTT Book、Rijke 教材或库源码位置。若引用 1Lab，需说明页面版本或访问日期。

## 15.5 不同库之间的迁移风险

**警告 15.5.** 同名术语在不同库中可能不是同一定义：

- $\mathsf{isEquiv}$ 可能以 fiber、半伴随等价或 typeclass 结构定义；
- HIT 可能是公理、module interface 或 cubical primitive；
- universe polymorphism 和 resizing 假设可能不同；
- 证明无关性、命题截断和集合截断实现可能不同。

**原则 15.6.** “某结论在库 $L$ 中形式化”只说明该结论在 $L$ 的基础口径下成立。若要移植到本书，需要给出定义比较和假设列表。

## 15.6 版本化索引

**定义 15.7.** 本书称一个形式化引用是可审计的，当且仅当它给出仓库、commit、模块路径、入口 identifier 和基础假设。仅给出库名不构成可审计引用。

**当前快照。** 附录 S 按 2026-06-29 核查以下形式化入口：

- Coq-HoTT commit `a030184c0bfc9d61f3bcd33c67660b800e106427`；
- UniMath commit `9ed7661d3ad33c74e35824efccf861b4fdc17323`；
- Cubical Agda commit `92166033326aa59800a580b428125f3c654b5e45`。

**使用规则 15.8.** 正文中“机器形式化”标注必须能追到附录 S 或后续同等级索引。若某结果只在库的实验目录中出现，应标注为“实验形式化入口”，不得当作稳定库定理使用。

## 本章小结

形式化库是 HoTT 教材的强支撑，但不是无条件的权威捷径。严格写作必须记录库、版本、模块、基础假设和定义口径；当前版本的具体入口见附录 S。

## 练习

**练习 15.1.** 选择一个 Coq-HoTT 中的路径代数引理，记录它依赖哪些导入。

**练习 15.2.** 比较 UniMath 和本书对“范畴”的定义口径。

**练习 15.3.** 在 Cubical Agda 文档中查找 path type 的说明，并解释它与恒等类型的关系。

**练习 15.4.** 说明为什么“已在某库形式化”仍需要版本号或 commit。
