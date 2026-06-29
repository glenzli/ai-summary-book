# 第十六章：模型范畴与同伦范畴

## 本章目标

本章介绍 Quillen 模型范畴：弱等价、纤维化、余纤维化、同伦范畴和导出函子。模型范畴是进入 $\infty$-范畴的一种经典路径。

## 依赖前置知识

需要极限、余极限、函子局部化和基本同伦论直觉。

## 16.1 模型范畴的定义

**定义 16.1.** 模型范畴是完备且余完备的范畴 $\mathcal M$，配有三类态射：

- 弱等价 $W$；
- 纤维化 $Fib$；
- 余纤维化 $Cof$；

满足 Quillen 公理：2-out-of-3、 retract 闭性、提升公理和两个函子分解公理。

**定义 16.2.** 提升公理要求：若 $i$ 是余纤维化且 $p$ 是纤维化，并且二者之一是弱等价，则任意交换方块

$$
\begin{matrix}
A&\to&X\\
\downarrow i&&\downarrow p\\
B&\to&Y
\end{matrix}
$$

存在对角填充 $B\to X$。

**例子 16.3.** 拓扑空间、单纯集、链复形在适当选择弱等价、纤维化、余纤维化后都有模型结构。单纯集上的 Kan-Quillen 模型结构以弱同伦等价为弱等价，以 Kan fibration 为纤维化。

## 16.2 同伦范畴

**定义 16.4.** 模型范畴 $\mathcal M$ 的同伦范畴 $\operatorname{Ho}(\mathcal M)$ 是把弱等价形式反演得到的局部化

$$
\mathcal M[W^{-1}].
$$

**外部输入定理 16.5.** 在模型范畴中，$\operatorname{Ho}(\mathcal M)$ 可通过 cofibrant-fibrant 对象和同伦类态射计算：

$$
\operatorname{Ho}(\mathcal M)(X,Y)
\cong
\pi_0\operatorname{Map}_{\mathcal M}(QX,RY)
$$

在有合适映射空间模型时成立。

## 16.3 Quillen 伴随

**定义 16.6.** 两个模型范畴之间的伴随

$$
F:\mathcal M\rightleftarrows\mathcal N:G
$$

称为 Quillen 伴随，若 $F$ 保持余纤维化和平凡余纤维化；等价地，$G$ 保持纤维化和平凡纤维化。

**定义 16.7.** Quillen 伴随若诱导同伦范畴等价，则称为 Quillen 等价。

**外部输入定理 16.8.** Quillen 伴随诱导总左导出函子和总右导出函子

$$
\mathbb L F:\operatorname{Ho}(\mathcal M)\rightleftarrows
\operatorname{Ho}(\mathcal N):\mathbb R G.
$$

构造上，左导出函子由先取 cofibrant replacement 再施加 $F$ 给出；右导出函子由先取 fibrant replacement 再施加 $G$ 给出。完整证明依赖模型范畴 replacement、Ken Brown 引理和同伦范畴局部化的泛性质，来源见 Quillen、Hovey 和 Hirschhorn。

## 16.4 从模型范畴到 $\infty$-范畴

**外部输入定理 16.9.** 每个相对范畴，特别是模型范畴 $(\mathcal M,W)$，可通过 hammock localization、simplicial localization 或 homotopy coherent nerve 产生一个 $\infty$-范畴。对于良好模型范畴，Quillen 等价诱导等价的 $\infty$-范畴。

来源见 Dwyer-Kan、Lurie、Cisinski 和 Hinich。

## 16.5 相对范畴、离散模型结构与边界

**定义 16.10.** 相对范畴（relative category）是范畴 $\mathcal C$ 连同一个含所有对象和恒等态射的子范畴 $W\subseteq\mathcal C$。$W$ 中态射称为弱等价。模型范畴给出相对范畴 $(\mathcal M,W)$，但相对范畴本身不包含纤维化、余纤维化和分解资料。

**命题 16.11（离散模型结构）.** 若 $\mathcal C$ 完备且余完备，则存在一个模型结构，其弱等价为同构，纤维化和余纤维化为所有态射。

**证明.** 同构对复合、逆和 retract 封闭，因此满足 2-out-of-3 和 retract 闭性。提升公理中，若 $i$ 或 $p$ 是平凡的，则它是同构；含一个同构边的交换方块可用该同构构造唯一对角填充。分解为“平凡余纤维化后接纤维化”时可取

$$
X\xrightarrow{\operatorname{id}_X}X\xrightarrow{f}Y
$$

分解为“余纤维化后接平凡纤维化”时可取

$$
X\xrightarrow{f}Y\xrightarrow{\operatorname{id}_Y}Y.
$$

其中恒等态射是同构，所有态射都是纤维化和余纤维化。故 Quillen 公理成立。$\square$

**例子 16.12.** 在命题 16.11 的模型结构中，所有对象既 cofibrant 又 fibrant，同伦范畴就是 $\mathcal C$ 本身。这个例子说明模型范畴公理允许“没有额外同伦识别”的退化情形。

**命题 16.13.** Quillen 伴随的左伴随保持 cofibrant 对象，右伴随保持 fibrant 对象。

**证明.** 设 $F:\mathcal M\rightleftarrows\mathcal N:G$ 为 Quillen 伴随。若 $X$ cofibrant，则初对象到 $X$ 的态射 $\varnothing\to X$ 是余纤维化。左伴随保持初对象，并保持余纤维化，因此

$$
\varnothing\cong F\varnothing\to FX
$$

是余纤维化，故 $FX$ cofibrant。右伴随保持终对象和纤维化，对偶得到它保持 fibrant 对象。$\square$

**例子 16.14（同伦范畴的边界）.** 两个模型范畴可能有等价的同伦范畴，但底层 $\infty$-范畴不等价；同伦范畴只记录弱等价反演后的 $1$-范畴。判断高阶信息必须使用映射空间、simplicial localization 或后续 $\infty$-范畴模型。

## 16.6 本章小结

模型范畴用三类态射组织同伦论，使得“反演弱等价”可计算。相对范畴记录弱等价，模型结构额外提供可计算的替换、提升和分解机制。它保留了很多一阶范畴工具，但真正的同伦信息不只在同伦范畴中；$\infty$-范畴保留了映射空间和高阶相干信息。

## 练习

**练习 16.1.** 写出 2-out-of-3 公理。

**练习 16.2.** 定义平凡纤维化和平凡余纤维化。

**练习 16.3.** 在链复形范畴中，说明 quasi-isomorphism 为什么是弱等价候选。

**练习 16.4.** 解释同伦范畴为什么可能丢失高阶同伦信息。

**练习 16.5.** 查阅 Kan fibration 的 horn lifting 定义，并与第十七章 quasi-category 的 inner horn 条件比较。

**练习 16.6.** 证明相对范畴的弱等价若取全部同构，则局部化等价于原范畴。

**练习 16.7.** 在离散模型结构中验证提升公理。

**练习 16.8.** 证明 Quillen 伴随的左伴随保持 cofibrant 对象。

**练习 16.9.** 说明相对范畴数据为什么不足以定义 cofibrant replacement。

**练习 16.10.** 解释为什么同伦范畴等价弱于 Quillen 等价。
