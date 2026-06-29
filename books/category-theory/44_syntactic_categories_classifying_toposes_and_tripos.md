# 第四十四章：语法范畴、分类 Topos 与 Tripos

## 本章目标

本章把第四十一章的范畴逻辑推进到“理论由范畴分类”的层次。语法范畴把形式理论中的上下文、项、公式和证明等价组织为范畴；分类 topos 通过几何态射表示模型；tripos-to-topos 构造说明从谓词逻辑数据可生成 topos。核心思想是：逻辑理论的模型范畴可由一个泛对象表示。

## 依赖前置知识

需要有限极限、regular category、Heyting category、Grothendieck topos、几何态射、子对象纤维化、依赖类型论语义、Yoneda 思想和 site/sheaf 技术。

## 44.1 有限极限理论与语法范畴

**定义 44.1.** 一个有限极限理论由 sorts、函数符号、关系符号以及有限极限逻辑中的公理组成。其上下文形如

$$
x_1:A_1,\dots,x_n:A_n.
$$

**定义 44.2.** 理论 $T$ 的有限极限语法范畴 $\mathcal C_T^{\operatorname{syn}}$ 的对象为上下文中的公式

$$
\{\vec x:\vec A\mid \varphi(\vec x)\},
$$

态射由可证唯一存在的函数式关系给出，并按 $T$-可证等价取商。

**外部输入定理 44.3.** 对任意有限极限理论 $T$，语法范畴 $\mathcal C_T^{\operatorname{syn}}$ 有有限极限，并且 $T$ 在任意有限极限范畴 $\mathcal E$ 中的模型自然等价于保持有限极限的函子

$$
\operatorname{Lex}(\mathcal C_T^{\operatorname{syn}},\mathcal E).
$$

**命题 44.4.** 若 $T$ 是空有限积理论，其语法范畴等价于有限集范畴的对偶骨架。

**证明.** 空有限积理论没有非平凡 sorts 外的结构；上下文只记录有限个变量。上下文 $n$ 可视为有限集合 $\{1,\dots,n\}$。从 $n$ 到 $m$ 的项代换给出 $m$ 个由 $n$ 个变量选出的投影，因此等价于函数 $m\to n$。故语法范畴的 Hom 为

$$
\mathcal C_T^{\operatorname{syn}}(n,m)\cong\mathbf{FinSet}(m,n),
$$

这正是有限集范畴的对偶。$\square$

## 44.2 Regular 与 coherent 语法范畴

**定义 44.5.** Regular 逻辑允许有限合取和存在量词。Regular 语法范畴把公式按可证等价取商，并把可证蕴含解释为子对象关系。

**定义 44.6.** Coherent 逻辑允许有限合取、有限析取和存在量词。Coherent category 是 regular category，并且每个子对象格有有限并且 pullback 保持这些有限并。

**命题 44.7.** Coherent category 中，有限析取由子对象有限并解释。

**证明.** 对同一对象 $X$ 上公式 $\varphi,\psi$ 的解释为子对象 $U,V\hookrightarrow X$。Coherent category 中 $\operatorname{Sub}(X)$ 有有限并，定义

$$
\llbracket\varphi\vee\psi\rrbracket=U\vee V.
$$

Pullback 保持有限并保证该解释与替换相容；有限并的泛性质保证它满足析取消去的序关系条件。$\square$

## 44.3 分类 Topos

**定义 44.8.** 几何理论是用有限合取、任意析取和存在量词书写的理论。Grothendieck topos $\mathcal E_T$ 称为几何理论 $T$ 的分类 topos，若对任意 Grothendieck topos $\mathcal F$ 有自然等价

$$
\operatorname{Geom}(\mathcal F,\mathcal E_T)\simeq \operatorname{Mod}_T(\mathcal F),
$$

其中左侧为几何态射范畴，右侧为 $\mathcal F$ 内的 $T$-模型范畴。

**命题 44.9.** 分类 topos 若存在，则在等价意义下唯一。

**证明.** 若 $\mathcal E_T$ 与 $\mathcal E'_T$ 都分类 $T$，则对所有 $\mathcal F$ 有自然等价

$$
\operatorname{Geom}(\mathcal F,\mathcal E_T)\simeq
\operatorname{Mod}_T(\mathcal F)\simeq
\operatorname{Geom}(\mathcal F,\mathcal E'_T).
$$

由 2-Yoneda 原理，两个表示同一 2-函子的对象在 2-范畴 $\mathbf{Topos}$ 中等价。$\square$

**外部输入定理 44.10.** 每个小几何理论都有分类 Grothendieck topos，可由其几何语法站点的 sheaf topos 构造。

## 44.4 模型、泛模型与保守性

**定义 44.11.** 分类 topos $\mathcal E_T$ 中对应恒等几何态射

$$
\operatorname{id}_{\mathcal E_T}:\mathcal E_T\to\mathcal E_T
$$

的模型称为 $T$ 的泛模型，记作 $U_T$。

**命题 44.12.** 任意 $\mathcal F$ 中的 $T$-模型都由泛模型沿唯一对应的几何态射拉回得到。

**证明.** 分类性质给出自然等价

$$
\operatorname{Geom}(\mathcal F,\mathcal E_T)\simeq\operatorname{Mod}_T(\mathcal F).
$$

模型 $M\in\operatorname{Mod}_T(\mathcal F)$ 对应某几何态射 $f:\mathcal F\to\mathcal E_T$。自然性说明 $M$ 正是 $f^*$ 作用于恒等态射对应的泛模型 $U_T$ 得到的模型。$\square$

## 44.5 Tripos 与谓词纤维化

**定义 44.13.** 一个 tripos 是带有足够逻辑结构的反变函子

$$
P:\mathcal C^{op}\to\mathbf{Heyt}
$$

通常要求基范畴 $\mathcal C$ 有有限积，纤维有 Heyting 结构，重索引有左右量词伴随，并满足 Beck-Chevalley 与 generic predicate 条件。

**定义 44.14.** Generic predicate 是对象 $\Omega\in\mathcal C$ 上的谓词 $\top_\Omega\in P(\Omega)$，使任意 $A$ 上任意谓词 $\varphi\in P(A)$ 都由某个 classifying map $\chi_\varphi:A\to\Omega$ 拉回得到。

**命题 44.15.** Elementary topos 的子对象纤维化给出 tripos 的基本例子。

**证明.** 设 $\mathcal E$ 为 elementary topos。取

$$
P(X)=\operatorname{Sub}_{\mathcal E}(X).
$$

Topos 有有限极限，pullback 给出重索引。子对象分类子 $\Omega$ 给出 generic predicate $\top:1\to\Omega$。Topos 的内部 Heyting 结构给出纤维 Heyting 结构，存在和全称量词由相应 pullback 函子的伴随给出，并满足 Beck-Chevalley。故得到 tripos 数据。$\square$

## 44.6 Tripos-to-topos

**外部输入定理 44.16（Tripos-to-topos）.** 每个 tripos $P:\mathcal C^{op}\to\mathbf{Heyt}$ 生成一个 elementary topos $\mathcal E_P$。其对象可表示为 partial equivalence relations，态射为功能性关系的等价类。

**命题 44.17.** 若 tripos 来自 topos 的子对象纤维化，则 tripos-to-topos 构造恢复原 topos 至等价。

**证明.** 来自 topos 的 tripos 已经由对象、子对象、逻辑运算和子对象分类子完整编码其内部逻辑。Tripos-to-topos 把该逻辑中的 partial equivalence relations 作为对象。Topos 中每个对象由其相等关系给出一个 PER；反向地，每个有效 PER 的商在 topos 中存在并给出对象。外部输入中的有效性与商存在性给出两构造互逆，故恢复原 topos。$\square$

## 44.7 本章小结

语法范畴把形式理论变成范畴对象；分类 topos 把模型问题表示为几何态射；泛模型统一所有模型；tripos 把谓词纤维化抽象为逻辑机器，再通过 tripos-to-topos 生成 topos。由此，逻辑与范畴论之间的对应不只是解释关系，而是表示性和分类性关系。

## 练习

**练习 44.1.** 定义有限极限理论的语法范畴。

**练习 44.2.** 说明空有限积理论的语法范畴为何与有限集对偶相关。

**练习 44.3.** 定义 regular 逻辑。

**练习 44.4.** 定义 coherent category。

**练习 44.5.** 说明 coherent category 中析取如何解释。

**练习 44.6.** 定义分类 topos。

**练习 44.7.** 证明分类 topos 的唯一性。

**练习 44.8.** 定义泛模型。

**练习 44.9.** 说明任意模型如何由泛模型拉回。

**练习 44.10.** 定义 tripos。

**练习 44.11.** 定义 generic predicate。

**练习 44.12.** 说明 topos 的子对象纤维化为何给出 tripos。

**练习 44.13.** 陈述 tripos-to-topos 构造。

**练习 44.14.** 解释 tripos-to-topos 如何恢复来自 topos 的 tripos。
