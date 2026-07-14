# 教材范围与完成条件

《同伦类型论与单值基础》从依赖类型论的判断规则出发，经恒等类型、等价、单值性、HIT、圆的基本群和单值一范畴，进入合成同伦论、模型语义与扩展语言。完整性在这里不意味着把所有高级主题都内部证明，而是让每个结论的逻辑身份、依赖和来源都可追踪。

## C.1 核心内部链

核心链包括第 1-11 章的基础 HoTT 与圆计算，以及第 13-14 章中预范畴、单值范畴、Yoneda、函子范畴、极限和伴随的书内部分。它们的基础证明核位于附录 A-Q、U-W、X 与相关附录中。

函数外延性、univalence 和指定 HIT 规则是明确加入对象语言的原则，不是由低层规则自动推出的 judgmental computation。Rezk completion 的本质像构造与 weak-equivalence 性质在书内处理；限制函子泛性质采用精确外部定理。

## C.2 高级数学边界

第十二章及高级附录讨论 EM 型、Blakers--Massey、Freudenthal、谱、谱序列、Postnikov、Steenrod operations 与 Adams 计算。只有在对象与假设已经完整列出时，正文才作条件化推导；大型存在性、连通性和收敛结论保留外部输入身份。

构造性分析必须说明实数对象、度量表示、完备性强度和截断消去目标。Directed/simplicial type theory、2LTT、cohesive HoTT、HIIT/QIIT 与 cubical 元理论分别属于扩展对象语言或元语言；它们不能无翻译地回流到基础 identity type。

## C.3 教材叙事条件

第 0-17 章必须形成连续教材，而不是项目清单：每章 H1 后先给出能够独立建立动机和语境的自然导言；定义前解释要解决的问题；定理间给出依赖与过渡；章末按该章内容收束。固定“本章目标/依赖前置知识/主线/本章小结”骨架不再允许。

例子应实际展开对象和映射，不能只说“可类似验证”。高级导读可以诚实停止在外部输入或研究边界，但不能用“证明架构”“适当闭包”等措辞代替缺失定义。

## C.4 OET 完成条件

1. 每个定义给出形成对象所需的变量、量词与 universe 条件。
2. 每个书内定理构造目标类型的项，并标出截断、transport、函数外延性、univalence 或 HIT 的使用点。
3. 每个条件化推导列出额外输入；每个外部定理给出可定位的来源和未内部化边界。
4. 对命题截断或 mere existence 消去前，先证明目标是命题；不能从 mere 数据选择代表元进入非命题目标。
5. 模型可靠性只推出解释后的有效性；相对一致性必须列出元理论假设，不等同于对象语言中的绝对一致性。

## C.5 验证入口

完成候选必须通过：

```text
python3 books/homotopy-type-theory/validate.py
python3 books/audit_textbook_narrative.py homotopy-type-theory --strict
python3 books/audit_oet_rigor.py homotopy-type-theory --strict
git diff --check -- books/homotopy-type-theory
```

附录 B 解释证明身份，附录 K 记录不可逆依赖；二者是数学使用规则，不是待办或验收清单。
