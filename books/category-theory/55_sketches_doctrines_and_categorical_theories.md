# 第五十五章：Sketches、Doctrines 与范畴化理论

## 本章目标

本章介绍用范畴论表达“理论”的内部工具。Sketch 指定某些图形应成为极限或余极限；模型是保持这些指定结构的函子。Doctrine 抽象出允许的逻辑结构，例如有限积、有限极限、regular、coherent 或 geometric 结构。它们把代数理论、有限极限理论、几何理论和类型论语义统一为范畴论对象。

## 依赖前置知识

需要极限、余极限、函子范畴、可表现范畴、语法范畴、分类 topos、有限极限理论和模型概念。

## 55.1 Sketch

**定义 55.1.** 一个 sketch 由小范畴 $\mathcal S$、一族指定锥和一族指定余锥组成。若只指定有限极限锥，称为有限极限 sketch；若指定小极限或小余极限，称为相应类型的 sketch。

**定义 55.2.** 设 $\mathcal C$ 有相应极限和余极限。Sketch $\mathcal S$ 在 $\mathcal C$ 中的模型是函子

$$
M:\mathcal S\to\mathcal C
$$

把指定锥送为极限锥，并把指定余锥送为余极限余锥。

**命题 55.3.** 若 sketch 没有指定锥或余锥，则其 $\mathcal C$-模型范畴就是函子范畴 $\operatorname{Fun}(\mathcal S,\mathcal C)$。

**证明.** 没有指定结构时，模型条件为空条件。满足空条件的函子正是所有函子 $\mathcal S\to\mathcal C$，模型间态射为自然变换，故模型范畴为 $\operatorname{Fun}(\mathcal S,\mathcal C)$。$\square$

## 55.2 有限积理论

**定义 55.4.** 有限积理论是带有限积的小范畴 $\mathbb T$。其在有有限积范畴 $\mathcal C$ 中的模型是保持有限积的函子

$$
\mathbb T\to\mathcal C.
$$

**命题 55.5.** 群对象可由有限积理论描述。

**证明.** 群理论有一个 sort $G$，运算 $m:G\times G\to G$、单位 $e:1\to G$、逆 $i:G\to G$，并有结合、单位、逆公理。这些公理都是有限积图上的交换等式。把这些生成对象和态射组织为有限积理论 $\mathbb T_{\mathrm{Grp}}$，保持有限积的函子 $\mathbb T_{\mathrm{Grp}}\to\mathcal C$ 正是 $\mathcal C$ 中满足群公理的群对象。$\square$

## 55.3 Doctrine

**定义 55.6.** 一个 doctrine 指定一类结构和保持该结构的函子，例如：

- finite product doctrine；
- finite limit doctrine；
- regular doctrine；
- coherent doctrine；
- geometric doctrine。

**定义 55.7.** $\mathsf D$-理论是带 $\mathsf D$-结构的小范畴；$\mathsf D$-模型是保持 $\mathsf D$-结构的函子。

**命题 55.8.** Doctrine 越强，模型函子的保持条件越强。

**证明.** 若 doctrine $\mathsf D'$ 包含 $\mathsf D$ 的所有结构并额外指定更多结构，则保持 $\mathsf D'$ 的函子特别保持 $\mathsf D$ 的结构。因此有包含

$$
\operatorname{Mod}_{\mathsf D'}(\mathbb T,\mathcal C)\subseteq
\operatorname{Mod}_{\mathsf D}(\mathbb T,\mathcal C)
$$

在对象层成立，态射层同为自然变换。$\square$

## 55.4 Essentially algebraic theories

**定义 55.9.** Essentially algebraic theory 允许部分运算，但其定义域由有限极限条件给出。等价地，它可由有限极限 sketch 表示。

**例子 55.10.** 小范畴构成 essentially algebraic structure：对象集、态射集、源靶、恒等和复合，其中复合只定义在 pullback

$$
\operatorname{Mor}\times_{\operatorname{Ob}}\operatorname{Mor}
$$

上。

**命题 55.11.** 小范畴的定义可由有限极限 sketch 表示。

**证明.** 取两个 sort $O,M$，态射 $s,t:M\to O$，恒等 $e:O\to M$。可复合态射对象由指定 pullback $M\times_O M$ 给出，复合为 $c:M\times_OM\to M$。结合律和单位律是有限极限图上的交换条件。因此这些数据构成有限极限 sketch，其 Set-模型正是小范畴。$\square$

## 55.5 可表现性

**外部输入定理 55.12.** 小 sketch 在 locally presentable category 中的模型范畴在合适小性条件下 locally presentable；特别，许多代数结构范畴可由 sketch 模型范畴得到。

**命题 55.13.** 若模型条件由极限保存给出，则模型范畴是函子范畴的 full subcategory。

**证明.** 模型是满足额外条件的函子 $\mathcal S\to\mathcal C$，态射是自然变换。任意两个模型之间的态射没有额外结构，因为指定锥的保持是对象条件。因此模型范畴嵌入 $\operatorname{Fun}(\mathcal S,\mathcal C)$ 为 full subcategory。$\square$

**命题 55.14.** 若 $F:\mathcal C\to\mathcal D$ 保持 sketch $\mathcal S$ 所指定的极限与余极限类型，则后合成给出模型范畴上的函子

$$
F_*:\operatorname{Mod}_{\mathcal C}(\mathcal S)\to\operatorname{Mod}_{\mathcal D}(\mathcal S).
$$

**证明.** 设 $M:\mathcal S\to\mathcal C$ 是模型。对任一指定锥，$M$ 将其送为 $\mathcal C$ 中的极限锥；因 $F$ 保持这种极限，$FM$ 将其送为 $\mathcal D$ 中的极限锥。指定余锥同理。自然变换经 $F$ 后仍为自然变换，因此后合成确实给出模型范畴之间的函子。$\square$

## 55.6 本章小结

Sketches 把理论表示为“图形加指定极限/余极限”；doctrines 指定允许的结构和保持结构的函子。有限积理论描述普通代数理论，有限极限 sketch 描述 essentially algebraic theories，geometric doctrine 连接分类 topos。由此，理论与模型本身也成为范畴论对象。

## 练习

**练习 55.1.** 定义 sketch。

**练习 55.2.** 定义 sketch 在 $\mathcal C$ 中的模型。

**练习 55.3.** 证明空 sketch 的模型范畴是函子范畴。

**练习 55.4.** 定义有限积理论。

**练习 55.5.** 说明群对象如何由有限积理论描述。

**练习 55.6.** 定义 doctrine。

**练习 55.7.** 证明 doctrine 越强，模型条件越强。

**练习 55.8.** 定义 essentially algebraic theory。

**练习 55.9.** 说明小范畴为何是 essentially algebraic structure。

**练习 55.10.** 证明小范畴可由有限极限 sketch 表示。

**练习 55.11.** 陈述 sketch 模型范畴的可表现性定理。

**练习 55.12.** 证明模型范畴是函子范畴的 full subcategory。

**练习 55.13.** 证明保持指定极限与余极限的目标函子把 sketch 模型送到 sketch 模型。
