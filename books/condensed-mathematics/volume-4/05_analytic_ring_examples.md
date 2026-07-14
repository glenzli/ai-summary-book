# 第五章：analytic ring 的 Dirac--测度计算

普通环只说明哪些元素可以相加相乘，却不说明无限 profinite 参数族应当怎样收敛。analytic
ring 用测度对象 $\mathcal M[S]$ 补上这份信息，并要求解析模无法区分有限 Dirac 组合
$A[\underline S]$ 与允许的测度。真正可计算的量因此不是符号
$(A,\mathcal M)$ 本身，而是二者之差的 cone 以及它对候选模的导出 Hom。

本章先把这个判别写成固定的输入、步骤和输出，再逐项计算有限集合与单点紧化
$\mathbb N\cup\{\infty\}$。后一个例子会实际展示 solid 测度比有限 Dirac 组合多出什么；
随后把同一系数关系搬到 $p$-进 Cantor 系统，并用无统一分母的序列说明普通张量换底
为何失败。Huber pair 的 rational localization 只在最后作为精确外部输入出现，不把
解析几何的深层构造伪装成初等完备化。

## 5.1 Cone 判别给出的计算合同

固定凝聚环 $A$、测试层级 $\mathbf{ED}_\kappa$，以及一套测度理论
$S\mapsto\mathcal M[S]$。对每个 $S$ 有自然的 Dirac 映射

$$
\delta_S:A[\underline S]\longrightarrow\mathcal M[S].
$$

记

$$
K_S^{\mathcal M}
:=
\operatorname{cofib}(\delta_S).
$$

**命题 5.1.1（局部对象判别）。** 对 $C\in D(A)$，以下两件事等价：

1. 对每个 $S\in\mathbf{ED}_\kappa$，自然映射
   $$
   R\underline{\operatorname{Hom}}_A(\mathcal M[S],C)
   \longrightarrow
   R\underline{\operatorname{Hom}}_A(A[\underline S],C)
   $$
   是等价；
2. 对每个 $S$，有
   $$
   R\underline{\operatorname{Hom}}_A(K_S^{\mathcal M},C)\simeq0.
   $$

**证明。** 对余纤维三角

$$
A[\underline S]\longrightarrow\mathcal M[S]
\longrightarrow K_S^{\mathcal M}\longrightarrow A[\underline S][1]
$$

在第一变量施加反变的
$R\underline{\operatorname{Hom}}_A(-,C)$，得到纤维三角。前两项之间的映射是等价，
当且仅当第三项为零。逐个 $S$ 检查即得结论。证毕。

因此一次 analytic 计算必须给出四项：底环 $A$、允许的测试空间 $S$、Dirac 映射
$\delta_S$，以及 cone 在目标对象上的正交性。只写
$A^\square[S]$ 或“完成化”并没有完成判别。

analytic ring 公理在这里对 $S\in\mathbf{ED}_\kappa$ 量化。solid 特例另有一项外部
结构定理：$\mathbb Z^\square[S]$ 与 Dirac 映射自然延拓到全部
$S\in\mathbf{ProFin}_\kappa$，而 solid 对象对这些延拓后的 cone 也正交。下文的
单点紧化与 Cantor 空间使用这个 profinite 延拓，不把非 ED 空间放进 analytic 公理
的原始量词。

## 5.2 有限集合：cone 确实消失

取 solid 解析环

$$
(\mathbb Z,\mathbb Z^\square),
$$

其解析模正是 solid 阿贝尔群。若 $S$ 有限，则

$$
\mathbb Z^\square[S]\cong\mathbb Z[S].
$$

**证明。** 有限离散空间上的连续整值函数是有限秩自由群，整值测度由各点质量唯一
决定。给测度 $\mu$ 取系数 $a_s=\mu(1_{\{s\}})$，便有

$$
\mu=\sum_{s\in S}a_s\delta_s.
$$

所以每个测度都是有限 Dirac 组合，$\delta_S$ 是同构，进而
$K_S^\square\simeq0$。这同时说明有限自由对象已经 solid。证毕。

输入是有限集合及每点的整数质量；步骤是展开到特征函数基；输出为
$\mathbb Z[S]$。这里没有高阶 localization，也没有由无限支撑造成的失败条件。

## 5.3 Worked example：单点紧化上的非 Dirac 测度

令

$$
S=\mathbb N\cup\{\infty\}
$$

带单点紧化拓扑。它是 profinite 空间但不是极不连通空间，因此本节使用 5.1 节说明的
solid profinite 延拓。连续映射 $f:S\to\mathbb Z$ 必须最终等于 $f(\infty)$。若
$e_n$ 是第 $n$ 点的特征函数，则每个 $f$ 唯一写成

$$
f=c\mathbf1+\sum_{n\in F}d_ne_n,
\qquad F\subset\mathbb N\text{ finite}.
$$

于是

$$
C(S,\mathbb Z)
\cong
\mathbb Z\mathbf1\oplus\bigoplus_{n\ge1}\mathbb Ze_n
$$

以及

$$
\mathbb Z^\square[S](*)
=\operatorname{Hom}(C(S,\mathbb Z),\mathbb Z)
\cong
\mathbb Z\times\prod_{n\ge1}\mathbb Z.
$$

在最后一个坐标描述中，$\delta_\infty=(1,0,0,\ldots)$，而
$\delta_n=(1,0,\ldots,1,0,\ldots)$，其中第二分量的 $1$ 位于第 $n$ 个坐标。因此有限
Dirac 组合的像恰为

$$
\mathbb Z\times\bigoplus_{n\ge1}\mathbb Z
\subsetneq
\mathbb Z\times\prod_{n\ge1}\mathbb Z.
$$

严格性可由元素

$$
\mu=(0,1,1,1,\ldots)
$$

看出：它给连续函数 $c\mathbf1+\sum d_ne_n$ 赋值 $\sum d_n$，该和总是有限的，故
$\mu$ 是合法整值测度；但它有无限多个非零坐标，不可能是有限 Dirac 组合。因而
$K_S^\square$ 非零。

这个非零 cone 不是 solid 理论的缺陷。恰恰相反，solidification 强制局部对象 $C$
满足

$$
R\underline{\operatorname{Hom}}(K_S^\square,C)\simeq0,
$$

从而在映入 $C$ 时把 $\mu$ 与 Dirac 数据按 analytic 结构兼容起来。输入是最终常值的
函数群，步骤是取其群对偶并追踪每个 Dirac 质量，输出是乘积中的具体非 Dirac 元素；
若误把 $C(S,\mathbb Z)$ 当成所有函数，第一步就已经改变了对象。

## 5.4 $p$-进 Cantor 系统

令 $S=\{0,1\}^{\mathbb N}$，并以
$S_m=\{0,1\}^m$ 为有限商。对输入的 $p$-进 solid 解析结构，外部构造在全局截面上
给出

$$
\mathbb Z_p^\square[S](*)
\simeq
\varprojlim_m\mathbb Z_p[S_m],
$$

转移映射沿 $S_{m+1}\to S_m$ 推前。把第 $m$ 层写成

$$
\mu_m=\sum_{x\in S_m}a_x^{(m)}[x],
\qquad a_x^{(m)}\in\mathbb Z_p,
$$

则相容性恰为

$$
a_x^{(m)}
=a_{(x,0)}^{(m+1)}+a_{(x,1)}^{(m+1)}.
$$

所以计算步骤可以在每个有限二叉树层完成：输入下一层的 $p$-进系数，把两个子节点
系数相加，检查是否等于父节点；全部层都通过时，输出一个 Cantor cylinder 上有限可加
的 $p$-进测度。若某一层等式失败，该族不属于逆极限。这里的深层输入是这些逆极限
对象确实组成 analytic ring；有限层系数检查及其唯一失败条件已经完全显式。

## 5.5 普通换底漏掉无统一分母的测度

仍取 5.3 节的 $S$。若把整值测度朴素地张量到 $\mathbb Q$，会得到自然映射

$$
(\mathbb Z\times\prod_n\mathbb Z)\otimes_{\mathbb Z}\mathbb Q
\longrightarrow
\mathbb Q\times\prod_n\mathbb Q.
$$

左侧每个元素可表示为 $(a,(b_n))/m$，所有坐标共用一个非零整数分母 $m$；右侧元素

$$
\left(0,1,\frac12,\frac13,\ldots\right)
$$

没有统一分母，因而不在像中。输出不是预期的逐坐标有理测度对象。

这精确定位了失败步骤：普通张量不保持这里出现的无限乘积。有限 $S$ 上同一映射是
有限自由模的换底同构；无限 $S$ 上则必须先指定目标 analytic 结构，再在其局部化张量
范畴中计算。不能用“先忘掉解析结构、普通换底、最后补拓扑”替代这一步。

## 5.6 Huber pair 与 rational localization 的输入边界

离散 Huber pair $(A,A^+)$ 给出解析环 $(A,A^+)^\square$。给定满足 rational-domain
假设的 $g,f_1,\ldots,f_r\in A$，相应区域写成

$$
U=\left\{x\in\operatorname{Spa}(A,A^+):
|f_i(x)|\le |g(x)|\ne0\right\},
$$

其坐标记作

$$
A\left\langle\frac{f_1,\ldots,f_r}{g}\right\rangle.
$$

输入不仅有环 $A$ 和分式 $f_i/g$，还包括整闭子环 $A^+$ 与 rational-domain 条件。
本章的 Huber pair 是离散的：相应坐标环为 $B=A[1/g]$，而 $B^+$ 是
$A^+[f_1/g,\ldots,f_r/g]$ 在 $B$ 中的整闭包；这里没有额外的拓扑完成步骤。一般
拓扑 Huber pair 的 rational localization 才还要赋予相应 Huber 拓扑并完备化，不能
把那一步悄悄写进离散情形。输出是该 rational 区域的 analytic ring 及其解析模拉回。
该构造和 rational descent 是 Huber/凝聚解析几何的外部输入，本书不重证。

若忘记 $A^+$，就无法恢复哪一些元素应视为有界；若 $g,f_i$ 不满足 rational-domain
假设，写出的括号表达式也不自动对应一个覆盖成员。这两种情况都不是计算尚未简化，
而是输入不足，因而没有合法输出。

## 5.7 Cone 是这些例子的共同收束

有限集合上，Dirac 映射已经是同构，cone 为零；单点紧化上，cone 具体记录无限支撑
测度；$p$-进 Cantor 系统把它改写成有限树层之间的相容条件；普通有理换底则因不保持
乘积而漏掉测度。analytic localization 的统一作用，是让候选解析模对所有这些 cone
正交。这个结论来自 5.1 节的纤维三角，完整证明不再依赖例子的几何来源；真正保留为
外部输入的，是 solid、$p$-进和 Huber 测度理论确实满足 analytic ring 公理。

## 练习

**练习 5.1.** 从余纤维三角重新证明命题 5.1.1，并写清映射方向为何反转。

**练习 5.2.** 对 $S=\mathbb N\cup\{\infty\}$ 验证
$\delta_n=(1,e_n)$，并证明 Dirac 像等于
$\mathbb Z\times\bigoplus_n\mathbb Z$。

**练习 5.3.** 在 $S_m=\{0,1\}^m$ 上任选一组 $p$-进系数，构造一层相容提升；说明
提升为何通常不唯一。

**练习 5.4.** 证明
$(0,1,1/2,1/3,\ldots)$ 不在 5.5 节换底映射的像中。

**练习 5.5.** 列出构造 Huber rational localization 时若只给 $A$ 而不给 $A^+$ 所
缺失的两类信息。
