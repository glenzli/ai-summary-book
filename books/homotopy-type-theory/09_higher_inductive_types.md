# 第九章：高阶归纳类型的规则格式

## 本章目标

本章给出高阶归纳类型（higher inductive types, HIT）的规则格式。HIT 不只允许点构造子，还允许路径构造子和更高路径构造子。我们不在本章证明一般 HIT 的元理论存在性，而是逐项声明本书采用的形成、消去和计算规则。本书实际使用的 HIT 输入规则集中列于附录 L；更一般的 HIIT、QIT、QIIT 和计算 HIT 语义见附录 BC。

## 依赖前置知识

本章依赖前八章，尤其是恒等类型、路径代数、截断和商类型。读者应接受：HIT 是对基础类型论的扩展，其模型和计算性需要单独元理论。

## 9.1 归纳类型与高阶归纳类型

普通归纳类型由点构造子生成，例如自然数由 $0$ 和 $\mathsf{succ}$ 生成。HIT 进一步允许声明路径。例如圆类型有一个点和一条非平凡环路。

**定义 9.1（HIT 规则数据，非正式严格格式）.** 一个 HIT 的说明应包含：

1.  形成规则：该类型在哪个宇宙中；
2.  点构造子：给出该类型的项；
3.  路径构造子：给出构造子之间的路径；
4.  更高路径构造子：给出路径之间的路径；
5.  消去原则：定义依赖函数到任意类型族时需要的数据；
6.  计算规则：消去函数在构造子上的行为。

**警告 9.2.** 若只写点和路径构造子而不写消去原则，就没有完整定义 HIT。HIT 的力量来自其依赖消去原则。

## 9.2 圆作为基本例子

**规则 9.3（圆类型的形成与构造）.** 对每个宇宙层级 $i$，本书加入一个圆类型实例

$$
\mathbb S^1_i:\mathcal U_i
$$

及构造子

$$
\mathsf{base}_i:\mathbb S^1_i,
$$
$$
\mathsf{loop}_i:\mathsf{base}_i=\mathsf{base}_i.
$$

固定一次推导中的层级后省略下标 $i$。这一省略表示 universe polymorphism，不表示由 $\mathbb S^1_i:\mathcal U_i$ 自动得到 $\mathbb S^1_i:\mathcal U_j$。

**规则 9.4（圆的非依赖递归及计算）.** 设 $A:\mathcal U_j$。给定
$$
a:A,\qquad \ell:a=a.
$$
则有

$$
\mathsf{rec}_{\mathbb S^1}(A,a,\ell):\mathbb S^1_i\to A.
$$

记该函数为 $f$。本书采用的公理化规则包规定点构造子上的 judgmental 计算

$$
f(\mathsf{base})\equiv a
$$

以及路径构造子上的 propositional computation

$$
\beta_{\mathsf{loop}}:
\mathsf{ap}_f(\mathsf{loop})=\ell.
$$

第二式的两边都是类型 $a=a$ 的项，因此 $\beta_{\mathsf{loop}}$ 是二阶路径；它不是 judgmental equality。

**规则 9.5（圆的依赖消去及计算）.** 设 $P:\mathbb S^1_i\to\mathcal U_j$。给定

$$
b:P(\mathsf{base})
$$

和依赖路径

$$
\ell_P:\mathsf{transport}^{P}(\mathsf{loop},b)=b,
$$

则有消去项

$$
\mathsf{ind}_{\mathbb S^1}(P,b,\ell_P):
\prod_{x:\mathbb S^1_i}P(x).
$$

记该截面为 $s$。点构造子上的计算为 judgmental：

$$
s(\mathsf{base})\equiv b.
$$

因此定义 2.9.1 给出的

$$
\mathsf{apd}_s(\mathsf{loop}):
\mathsf{transport}^{P}(\mathsf{loop},b)=b
$$

与输入 $\ell_P$ 具有相同类型。本书再加入 propositional dependent computation

$$
\beta^{P}_{\mathsf{loop}}:
\mathsf{apd}_s(\mathsf{loop})=\ell_P.
$$

这四项数据，即形成、构造、依赖消去和两种强度不同的计算规则，才是本书所说的“圆 HIT 输入”。非依赖递归规则 9.4 作为同一输入包的接口列出；从常值族消去推回它时，还要使用命题 2.9.2 的常值 transport 路径以及命题 2.18.1 的 $\mathsf{apd}$/$\mathsf{ap}$ 比较，不能把这些相容性当成 judgmental equality。

## 9.3 区间、商和截断

**例 9.6（区间）.** 在固定层级 $i$，区间 HIT $I_i:\mathcal U_i$ 可由两个点 $0_I,1_I:I_i$ 和路径 $\mathsf{seg}:0_I=1_I$ 生成。只有在同时给出依赖消去及其计算规则后，它才构成完整的 HIT 输入；本书不在后文依赖该区间。

**例 9.7（集合商）.** 设 $A:\mathcal U_i$，且 $R(x,y):\mathcal U_j$ 是命题值关系。第八章的集合商 $A/R:\mathcal U_{\max(i,j)}$ 是 HIT：点构造子 $[-]:A\to A/R$，路径构造子把 $R(x,y)$ 送到 $[x]=[y]$，再加入 0-truncation 构造子保证商是集合。该层级是附录 L 的形成规则，不使用 resizing。

**例 9.8（命题截断）.** 对 $A:\mathcal U_i$，命题截断 $\|A\|:\mathcal U_i$ 可看作 HIT：点构造子 $|-|:A\to\|A\|$，并加入路径构造子使任意两点相等。保持在 $\mathcal U_i$ 是输入规则的一部分，而不是命题 resizing。

## 9.4 计算规则的强弱

**定义 9.9（本书的公理化计算口径）.** 对规则 9.3-9.5 的圆，本书固定采用：

1.  点构造子上的 $\beta$-规则是 judgmental computation；
2.  路径构造子上的普通与依赖 $\beta$-规则分别由 $\beta_{\mathsf{loop}}$ 和 $\beta^P_{\mathsf{loop}}$ 这两个路径项给出；
3.  不加入圆的 judgmental $\eta$-规则或唯一性规则；需要的唯一性结论必须在路径类型中证明。

**警告 9.10.** HoTT Book 的公理化 HIT 口径通常把高维构造子上的计算写成 propositional equality。某个 cubical 系统若给出更强的 judgmental computation，那是该系统额外语法与归约关系的结论；不得把它静默并入规则 9.3-9.5。

## 9.5 HIT 的元理论状态

**外部输入定理 9.11（CHM 系统中的特定 HIT 语义）.** Coquand、Huber、Mörtberg 在其 2018 cubical type theory 中为 spheres、torus、suspensions、truncations 和 pushouts 给出语法与构造性 presheaf 语义。对论文实际列出的这些签名，形成、引入、消去和所有构造子上的计算均有 judgmental 规则；这些类型形成子严格保持替换，并位于其参数所在的同一 universe level。

**来源与未重证边界。** 精确来源为 *On Higher Inductive Types in Cubical Type Theory*, LICS 2018, DOI `10.1145/3209108.3209197`, 尤其是第 3.3 节及论文结论。本书不重建 presheaf topos、composition/filling 或初始代数语义。该论文明确把一般 HIT schema 的完整制定与语义留作后续问题，所以本定理不能推出“任意 HIT 都存在”或“任意 HIT 都有 judgmental 高维计算”。HoTT Book 第六章只支撑本书公理化规则的教材形态；更一般的 HIIT/QIIT 接口见附录 BC。

**规则 9.12（QIIT 使用纪律）.** 若某章使用 quotient inductive-inductive type，必须同时列出：

1.  生成的类型和依赖族；
2.  点构造子；
3.  路径或商构造子；
4.  截断构造子；
5.  递归和依赖消去原则；
6.  哪些计算规则为 judgmental，哪些只是 propositional；
7.  universe 层级和严格正性条件。

## 本章小结

HIT 是 HoTT 表达拓扑空间、商、截断和同伦余极限的关键工具。完整 HIT 输入必须包括形成、构造、消去和计算规则，并逐项标注宇宙层级及 judgmental/propositional 强度。外部 cubical 语义只覆盖其明确构造的签名，不提供无条件的一般 HIT schema。

## 练习

**练习 9.1.** 写出命题截断作为 HIT 的点构造子和路径构造子。

**练习 9.2.** 写出区间递归原则，并说明它与路径类型的关系。

**练习 9.3.** 类型检查 $\mathsf{apd}_s(\mathsf{loop})=\ell_P$，并指出点计算 $s(\mathsf{base})\equiv b$ 在端点对齐中的作用。

**练习 9.4.** 比较规则 9.5 与外部输入定理 9.11 的计算强度，并说明为什么后者不能改变本章公理化圆的 judgmental equality。
